"""
YAML-driven deterministic rule filtering for optimizer candidates.

The rule engine applies a configurable sequence of boolean filters to the
extraction data, producing two outputs:

  1. ``eligible_candidates_df`` -- invoices that pass all rules and are ready
     for the optimizer.
  2. ``excluded_df`` -- invoices removed by at least one rule, annotated with
     the rule name that excluded them.

Rules are defined in ``rules/rules_config.yaml`` and executed top-to-bottom.
Each rule receives the output of the previous rule (pipeline / funnel pattern),
so rule ordering matters: earlier rules reduce the input for later ones.

Supported rule types (see ``_apply_single_rule`` for implementation):
  - ``not_null``           : exclude rows where a column is NA
  - ``equals``             : keep only rows where a column matches a target value
  - ``in``                 : keep only rows where a column is in a set of values
  - ``regex_not_contains`` : exclude rows where a column matches a regex pattern
  - ``numeric_min``        : keep rows where a numeric column is >= (or >) a threshold
  - ``date_after``         : keep rows where a date column is >= another date/column
  - ``deduplicate_by``     : keep the first row per composite key (with fallback columns)

Rules can reference runtime values (e.g. the cohort timestamp) via the
``*_from_context`` pattern in the YAML config. Rules can be conditionally
enabled/disabled via ``enabled`` or ``enabled_from_context``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuleApplicationSummary:
    """Stats for a single rule application: how many rows went in, were excluded, and came out."""
    rule_name: str
    rule_type: str
    input_rows: int
    excluded_rows: int
    output_rows: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "rule_type": self.rule_type,
            "input_rows": self.input_rows,
            "excluded_rows": self.excluded_rows,
            "output_rows": self.output_rows,
        }


@dataclass(frozen=True)
class RuleEngineResult:
    """Complete output of the rule engine: eligible candidates, excluded rows, and per-rule stats."""
    eligible_candidates_df: pd.DataFrame
    excluded_df: pd.DataFrame
    summaries: Tuple[RuleApplicationSummary, ...]


class RulesConfigError(ValueError):
    """Raised when the rule configuration is invalid."""


def load_rules_config(path: str | Path) -> Dict[str, Any]:
    """Load and validate the rules YAML config.

    Validates that the file contains a non-empty ``rules`` list and that each
    rule has at minimum a ``name`` and ``type`` field.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Rules config not found: {source}")
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    rules = payload.get("rules", [])
    if not isinstance(rules, list) or not rules:
        raise RulesConfigError("rules_config must contain a non-empty 'rules' list")

    for idx, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise RulesConfigError(f"Rule at index {idx} must be a mapping")
        if "name" not in rule or "type" not in rule:
            raise RulesConfigError(f"Rule at index {idx} requires 'name' and 'type'")

    return payload


def _resolve_value(rule: Dict[str, Any], context: Dict[str, Any], key: str) -> Any:
    """Resolve a single rule parameter, either from the YAML rule definition or from runtime context.

    If the rule contains ``<key>_from_context``, the value is looked up in the
    runtime context dict. Otherwise, the static value from the rule dict is used.
    """
    ctx_key = rule.get(f"{key}_from_context")
    if ctx_key:
        if ctx_key not in context:
            raise RulesConfigError(
                f"Rule {rule['name']} expects context key '{ctx_key}' for {key}"
            )
        return context[ctx_key]
    return rule.get(key)


def _resolve_values(rule: Dict[str, Any], context: Dict[str, Any], key: str) -> List[Any]:
    """Like _resolve_value but always returns a list (for 'in'-type rules)."""
    ctx_key = rule.get(f"{key}_from_context")
    if ctx_key:
        if ctx_key not in context:
            raise RulesConfigError(
                f"Rule {rule['name']} expects context key '{ctx_key}' for {key}"
            )
        ctx_value = context[ctx_key]
        if isinstance(ctx_value, list):
            return ctx_value
        return [ctx_value]

    value = rule.get(key)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _build_coalesced_key(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    """Build a deduplication key by coalescing multiple columns left-to-right.

    For each row, the key is the first non-NA value across the listed columns.
    This is used by the ``deduplicate_by`` rule type with ``fallback_columns``:
    the primary column (e.g. Invoice Reference) is preferred, falling back to
    the secondary column (e.g. Document Number) when the primary is NA.

    NOTE: This coalesces across different identifier namespaces. A row with
    Invoice Reference = NA and Document Number = "X" will collide with a row
    that has Invoice Reference = "X". In practice, Invoice References and
    Document Numbers have different formats so collisions are unlikely.
    """
    columns = list(columns)
    if not columns:
        return pd.Series([pd.NA] * len(df), index=df.index)

    key = df[columns[0]].astype("string") if columns[0] in df.columns else pd.Series([pd.NA] * len(df), index=df.index, dtype="string")
    for col in columns[1:]:
        if col not in df.columns:
            continue
        candidate = df[col].astype("string")
        key = key.fillna(candidate)
    return key


def _apply_single_rule(
    df: pd.DataFrame,
    rule: Dict[str, Any],
    context: Dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rule_name = rule["name"]
    rule_type = rule["type"]

    if rule_type == "not_null":
        col = rule["column"]
        keep_mask = df[col].notna()

    elif rule_type == "equals":
        col = rule["column"]
        target = _resolve_value(rule, context, "value")
        granularity = _resolve_value(rule, context, "match_granularity")
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            target_ts = pd.to_datetime(target)
            if granularity is None or granularity == "exact":
                keep_mask = df[col] == target_ts
            elif granularity == "date":
                keep_mask = df[col].dt.date == target_ts.date()
            else:
                raise RulesConfigError(
                    f"Rule {rule_name}: unsupported match_granularity '{granularity}'"
                )
        else:
            keep_mask = df[col] == target

    elif rule_type == "in":
        col = rule["column"]
        targets = _resolve_values(rule, context, "values")
        keep_mask = df[col].isin(targets)

    elif rule_type == "regex_not_contains":
        col = rule["column"]
        pattern = rule["pattern"]
        flags = re.IGNORECASE if rule.get("ignore_case", False) else 0
        keep_mask = ~df[col].fillna("").astype(str).str.contains(
            pattern, flags=flags, regex=True
        )

    elif rule_type == "numeric_min":
        col = rule["column"]
        min_value = _resolve_value(rule, context, "min_value")
        if min_value is None:
            # Backward-compatible guard: older API model dropped min_value when saving
            # rules, resulting in null values in persisted rules.yaml.
            logger.warning(
                "Rule %s (%s) missing min_value; defaulting to 0.0 to avoid runtime failure.",
                rule_name,
                rule_type,
            )
            min_value = 0.0
        include_equal = bool(rule.get("include_equal", True))
        numeric_col = pd.to_numeric(df[col], errors="coerce")
        try:
            min_value_float = float(min_value)
        except (TypeError, ValueError) as exc:
            raise RulesConfigError(
                f"Rule {rule_name}: min_value must be numeric, got {min_value!r}"
            ) from exc
        if include_equal:
            keep_mask = numeric_col >= min_value_float
        else:
            keep_mask = numeric_col > min_value_float
        keep_mask = keep_mask.fillna(False)

    elif rule_type == "date_after":
        col = rule["column"]
        other_col = rule.get("other_column")
        if other_col:
            keep_mask = df[col] >= df[other_col]
        else:
            min_value = _resolve_value(rule, context, "min_value")
            keep_mask = df[col] >= pd.to_datetime(min_value)

    elif rule_type == "deduplicate_by":
        cols = rule.get("columns", [])
        fallback_cols = rule.get("fallback_columns", [])

        for col in cols:
            if col not in df.columns:
                raise RulesConfigError(f"Rule {rule_name}: missing dedupe column '{col}'")

        key_cols = list(cols)
        if fallback_cols:
            key_series = _build_coalesced_key(df, key_cols + list(fallback_cols))
        else:
            key_series = df[key_cols].astype("string").fillna("<NA>").agg("|".join, axis=1)

        # Rows without any dedupe key are kept.
        missing_key = key_series.isna() | (key_series.astype(str).str.strip() == "")
        dedupe_key = key_series.where(~missing_key)
        duplicated = dedupe_key.duplicated(keep="first") & ~missing_key
        keep_mask = ~duplicated

    else:
        raise RulesConfigError(f"Unsupported rule type '{rule_type}' in rule {rule_name}")

    kept_df = df[keep_mask].copy()
    excluded_df = df[~keep_mask].copy()
    if not excluded_df.empty:
        excluded_df["excluded_reason"] = rule_name
        excluded_df["excluded_rule_type"] = rule_type

    return kept_df, excluded_df


def apply_rules(
    df: pd.DataFrame,
    rules_config: Dict[str, Any],
    context: Dict[str, Any],
) -> RuleEngineResult:
    """Apply configured rules sequentially (funnel pattern) and return results.

    Each rule receives the survivors from the previous rule. Excluded rows are
    collected separately with the name of the rule that removed them.

    Args:
        df: Full extraction DataFrame (all rows, pre-filtering).
        rules_config: Parsed YAML config dict containing a ``rules`` list.
        context: Runtime context dict with dynamic values (e.g. cohort timestamp,
            match granularity, feature flags) that rules can reference via
            ``*_from_context`` keys.

    Returns:
        RuleEngineResult with eligible candidates, combined exclusions, and
        per-rule application summaries.
    """
    active_df = df.copy()
    excluded_frames: List[pd.DataFrame] = []
    summaries: List[RuleApplicationSummary] = []

    for rule in rules_config["rules"]:
        # Rules can be statically disabled (enabled: false) or conditionally
        # disabled via a runtime context flag (enabled_from_context).
        enabled = rule.get("enabled", True)
        enabled_ctx_key = rule.get("enabled_from_context")
        if enabled_ctx_key:
            enabled = bool(context.get(enabled_ctx_key, False))
        if not enabled:
            continue

        input_rows = len(active_df)
        active_df, excluded = _apply_single_rule(active_df, rule, context)
        output_rows = len(active_df)
        excluded_rows = len(excluded)

        if excluded_rows:
            excluded_frames.append(excluded)

        summaries.append(
            RuleApplicationSummary(
                rule_name=rule["name"],
                rule_type=rule["type"],
                input_rows=input_rows,
                excluded_rows=excluded_rows,
                output_rows=output_rows,
            )
        )

    excluded_df = (
        pd.concat(excluded_frames, ignore_index=True)
        if excluded_frames
        else pd.DataFrame(columns=[*active_df.columns, "excluded_reason", "excluded_rule_type"])
    )

    return RuleEngineResult(
        eligible_candidates_df=active_df.reset_index(drop=True),
        excluded_df=excluded_df.reset_index(drop=True),
        summaries=tuple(summaries),
    )
