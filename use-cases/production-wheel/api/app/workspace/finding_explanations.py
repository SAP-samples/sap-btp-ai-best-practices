"""Evidence for the Workspace Assistant to explain one dataset review finding.

The Datasets page shows extraction findings in short plain words. When a user
asks "why?", the assistant needs more than the finding text: the catalogue
explanation (cause, workbook source, effect, fix) and the actual data of the
affected products. ``explain_finding`` assembles both from a stored snapshot.

Inputs come from HANA through the dataset service: the dataset record (with its
``issues`` and ``metadata``), the ``fini_master`` rows and the
``production_versions`` rows. Nothing is written.

Example::

    explain_finding(dataset, fini_rows, versions, "FINI_DATA_QUALITY", "missing_fixed_pv")
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

# Catalogue shared with the agent: cause, source columns, effect and fix per check.
FINDING_HELP: dict[str, dict[str, Any]] = json.loads(
    Path(__file__).with_name("finding_help.json").read_text()
)

# Affected rows returned to the model; totals always cover every affected row.
MAX_ROWS = 25

# fini_master fields that explain an admission decision.
ROW_FIELDS = (
    "fini_plant_key",
    "material",
    "sefi",
    "production_frequency",
    "forecast_litres_12m",
    "eligible_lines",
    "package_volume",
    "pallet_litres_resolved",
    "fixed_pv",
    "baseline_subgroup",
    "lot_size_considered_litres",
    "model_status",
    "exclusion_reason",
    "optimized_pv_model_status",
    "optimized_pv_exclusion_reason",
)

# Reasons whose explanation needs the SEFI's production versions.
PV_REASONS = {"missing_fixed_pv", "fixed_pv_not_unique_positive_lot"}

# Out-of-scope reason of frequency values without a weekly code.
UNRECOGNIZED_FREQUENCY_REASON = "production_frequency_unrecognized_out_of_scope"
# Out-of-scope reason of a weekly code the settings did not select, e.g. ..._16w_...
NOT_SELECTED_FREQUENCY = re.compile(r"^production_frequency_(\d{2}w)_out_of_scope$")


def _keys(issue: Mapping[str, Any]) -> list[str]:
    """Split a finding's ``entity_key`` ("a|b|c") into its affected keys."""
    return [key.strip() for key in str(issue.get("entity_key") or "").split("|") if key.strip()]


def _row_view(row: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the admission-relevant fields of one fini_master row."""
    return {field: row.get(field) for field in ROW_FIELDS}


def _pv_mode_outcome(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Count affected products by their result in each production version mode."""
    outcome = {"modeled_with_fixed": 0, "modeled_only_with_optimized": 0, "excluded_in_both": 0, "out_of_scope": 0}
    for row in rows:
        fixed = row.get("model_status") == "modeled"
        optimized = row.get("optimized_pv_model_status") == "modeled"
        if row.get("model_status") == "out_of_scope":
            outcome["out_of_scope"] += 1
        elif fixed:
            outcome["modeled_with_fixed"] += 1
        elif optimized:
            outcome["modeled_only_with_optimized"] += 1
        else:
            outcome["excluded_in_both"] += 1
    return outcome


def _help(rule_id: str, reason: str | None) -> dict[str, Any]:
    """Return the catalogue entry of a check, with the reason's details merged in."""
    entry = dict(FINDING_HELP.get(rule_id, {}))
    reasons = entry.pop("reasons", {})
    if reason and reason in reasons:
        entry["reason"] = {"code": reason, **reasons[reason]}
    return entry


def explain_finding(
    dataset: Mapping[str, Any],
    fini_rows: list[Mapping[str, Any]],
    versions: list[Mapping[str, Any]],
    rule_id: str,
    reason: str | None = None,
) -> dict[str, Any]:
    """Collect the catalogue explanation and the data behind one review finding.

    Args:
        dataset: Dataset record with ``dataset_id``, ``issues`` and ``metadata``.
        fini_rows: The snapshot's ``fini_master`` rows.
        versions: The snapshot's ``production_versions`` rows.
        rule_id: Check name, e.g. ``FINI_DATA_QUALITY``, ``FINI_OPTIMIZED_PV_ONLY``,
            ``PRODUCTION_FREQUENCY_UNDETERMINED`` or the scope note
            ``FREQUENCY_NOT_SELECTED``.
        reason: Optional reason code (exclusion reason), or for frequency notes
            the source value or weekly code.

    Returns:
        Dict with ``help`` (cause, source, effect, fix), ``issue_count``,
        ``affected_total``, up to ``MAX_ROWS`` ``affected_rows``,
        ``pv_mode_outcome`` (how many affected products each production version
        mode includes), ``production_versions_by_sefi`` for production-version
        reasons, and ``truncated``.

    Raises:
        ValueError: The check is neither catalogued nor present in this dataset.
    """
    issues = [issue for issue in dataset.get("issues") or [] if issue.get("rule_id") == rule_id]
    if reason and rule_id not in {"PRODUCTION_FREQUENCY_UNDETERMINED", "FREQUENCY_NOT_SELECTED"}:
        issues = [issue for issue in issues if issue.get("reason") == reason]
    if rule_id not in FINDING_HELP and not issues:
        present = sorted({issue.get("rule_id") for issue in dataset.get("issues") or []})
        raise ValueError(f"Unknown finding {rule_id!r}; this dataset has: {present}")

    by_key = {str(row.get("fini_plant_key")): row for row in fini_rows}
    if rule_id == "PRODUCTION_FREQUENCY_UNDETERMINED":
        # Every row without a weekly code; optionally only one source value.
        affected = [
            row for row in fini_rows
            if row.get("exclusion_reason") == UNRECOGNIZED_FREQUENCY_REASON
            and (not reason or str(row.get("production_frequency") or "") == reason)
        ]
    elif rule_id == "FREQUENCY_NOT_SELECTED":
        # Rows whose weekly code was left out by the settings; optionally one code.
        affected = []
        for row in fini_rows:
            match = NOT_SELECTED_FREQUENCY.match(str(row.get("exclusion_reason") or ""))
            if match and (not reason or match.group(1) == reason.lower()):
                affected.append(row)
    else:
        keys = [key for issue in issues for key in _keys(issue)]
        affected = [by_key[key] for key in keys if key in by_key]
        unmatched = [key for key in keys if key not in by_key]

    result: dict[str, Any] = {
        "dataset_id": dataset.get("dataset_id"),
        "rule_id": rule_id,
        "reason": reason,
        "help": _help(rule_id, reason),
        "issue_count": len(issues),
        "severity": issues[0].get("severity") if issues else FINDING_HELP.get(rule_id, {}).get("severity"),
        "affected_total": len(affected),
        "affected_rows": [_row_view(row) for row in affected[:MAX_ROWS]],
        "pv_mode_outcome": _pv_mode_outcome(affected),
        "truncated": len(affected) > MAX_ROWS,
    }
    if rule_id not in {"PRODUCTION_FREQUENCY_UNDETERMINED", "FREQUENCY_NOT_SELECTED"} and unmatched:
        # Keys that are not products (e.g. a lookup table name), shown as listed.
        result["other_affected_keys"] = unmatched[:MAX_ROWS]
    if rule_id == "FREQUENCY_NOT_SELECTED":
        result["modeled_frequencies"] = (dataset.get("metadata") or {}).get("settings", {}).get("modeled_frequencies")
    if reason in PV_REASONS or rule_id == "FINI_OPTIMIZED_PV_ONLY":
        sefis = {str(row.get("sefi")) for row in affected[:MAX_ROWS]}
        catalog: dict[str, list[dict[str, Any]]] = {}
        for version in versions:
            if str(version.get("sefi")) in sefis:
                catalog.setdefault(str(version.get("sefi")), []).append(
                    {"production_version": version.get("production_version"), "lot_size_litres": version.get("lot_size_litres")}
                )
        # SEFIs without any catalog row are listed too, with an empty list.
        result["production_versions_by_sefi"] = {sefi: catalog.get(sefi, []) for sefi in sorted(sefis)}
    return result
