"""
RPT-1 lifetime estimation for multi-week invoice optimization.

Invoices release credit on weekly Tuesday reconciliation files (see
reconciliation_calendar.py). Instead of regressing the lifetime in days, RPT-1 predicts
the reconciliation offset k: how many weekly files after the due date the invoice is
released. The calendar then turns k into a release date and a lifetime measured from the
planned funding Wednesday.

Per candidate:
  1. Features: company code, customer, due weekday, days from funding to due date and
     the customer's historical median k / history size (priors).
  2. Context: the customer's own history (newest first), topped up with same company
     code and then other rows when the customer has little history.
  3. RPT-1 regression on k, rounded to a whole file and clipped to the observed range.
  4. Safety margin: k = max(k_model, ceil(customer k quantile)) so that lifetimes are
     rarely under-estimated for customers that often pay late (configurable).
  5. release = first Tuesday on/after due + 7k; lifetime = release - funding.

The validation behind this design (uncensored chronological holdout, 600 invoices) is
documented in docs/rpt1-lifetime-diagnostics.md.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd

from .reconciliation_calendar import (
    as_of_customer_priors,
    customer_margin_offsets,
    first_reconciliation_on_or_after,
    reconciliation_offset,
    release_timestamp,
    to_naive,
)

logger = logging.getLogger(__name__)

_PRED_TARGET_COLUMN = "TARGET_RECON_K"
_INDEX_COLUMN = "ROW_ID"
_FEATURE_COLUMNS = [
    "COMPANY_CODE",
    "CUSTOMER_ID",
    "DUE_WEEKDAY",
    "DAYS_TO_DUE",
    "CUST_MEDIAN_K",
    "CUST_HISTORY_N",
]
# Customer IDs look numeric; a prefix keeps them categorical for RPT-1.
_CUSTOMER_PREFIX = "CUST_"
# Source column aliases, first match wins (history rows and workspace candidates differ).
_COMPANY_COLUMNS = ["Company Code", "company_code", "seller_id_external"]
_CUSTOMER_COLUMNS = ["Customer", "debtor_id"]
_DUE_COLUMNS = ["Due Date", "DUE DATE", "due_date"]
_HISTORY_FUNDING_COLUMNS = ["credit_start", "Summary File Date (UTC)", "summary_file_date"]
_HISTORY_RELEASE_COLUMNS = ["credit_release", "Reconciliation File Date (UTC)"]
_CANDIDATE_FUNDING_COLUMNS = ["Planned Funding Date", "planned_funding_date", "credit_start",
                              "Summary File Date (UTC)", "Summary File Date"]


@dataclass(frozen=True)
class LifetimeEstimationConfig:
    """Estimator settings.

    Attributes:
        enabled: False returns candidates unchanged (status 'disabled').
        context_min_rows: Minimum context rows per call; sparse customers are topped up.
        context_max_rows: Maximum context rows per call (customer rows first).
        query_batch_size: Maximum candidates per RPT-1 call (one customer per call).
        default_lifetime_weeks: Fallback duration when RPT-1 cannot supply a value.
        prediction_placeholder: Placeholder sent in the target column of query rows.
        timeout_seconds, max_retries, retry_backoff_seconds: RPT-1 HTTP behaviour.
        env_path: Optional dotenv file with AICORE_* and RPT1_MODEL_NAME.
        max_parallel_calls: Concurrent RPT-1 calls (AI Core answers bursts with 429).
        release_margin_quantile: Customer k quantile used as a floor for the predicted
            k (0.75 validated); None disables the margin.
        customer_min_rows: History rows a customer needs before its own median/quantile
            are trusted for the margin.
    """
    enabled: bool = True
    context_min_rows: int = 100
    context_max_rows: int = 800
    query_batch_size: int = 50
    default_lifetime_weeks: int = 4
    prediction_placeholder: str = "[PREDICT]"
    timeout_seconds: int = 90
    max_retries: int = 5
    retry_backoff_seconds: float = 2.0
    env_path: str | None = None
    max_parallel_calls: int = 2
    release_margin_quantile: float | None = 0.75
    customer_min_rows: int = 5


@dataclass(frozen=True)
class _LifetimeBatch:
    batch_id: int
    batch_indices: List[int]
    context_df: pd.DataFrame
    query_df: pd.DataFrame


@dataclass(frozen=True)
class _LifetimeBatchResult:
    batch_id: int
    predicted_rows: List[Dict[str, Any]]
    context_rows: int
    query_rows: int
    input_cells: int
    predictions_returned: int
    api_calls: int
    error: str | None = None


def _coalesce(df: pd.DataFrame, columns: List[str]) -> pd.Series:
    """First non-null value across the given columns (all-NA series when none exist)."""
    result = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    for column in columns:
        if column in df.columns:
            result = result.where(result.notna(), df[column])
    return result


def _identifier(values: pd.Series) -> pd.Series:
    """Normalize IDs to trimmed strings; integral floats (100456220.0) lose the '.0'."""
    def normalize(value: Any) -> Any:
        if value is None or (isinstance(value, float) and not math.isfinite(value)) or pd.isna(value):
            return pd.NA
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        text = str(value).strip()
        return text or pd.NA
    return values.map(normalize)


def _base_features(df: pd.DataFrame, funding_columns: List[str]) -> pd.DataFrame:
    """Company, customer, due date, funding date and the derived calendar features."""
    features = pd.DataFrame(index=df.index)
    features["COMPANY_CODE"] = _identifier(_coalesce(df, _COMPANY_COLUMNS))
    customer = _identifier(_coalesce(df, _CUSTOMER_COLUMNS))
    features["CUSTOMER_KEY"] = customer
    features["CUSTOMER_ID"] = customer.map(lambda value: pd.NA if pd.isna(value) else _CUSTOMER_PREFIX + value)
    features["DUE_AT"] = to_naive(_coalesce(df, _DUE_COLUMNS)).set_axis(df.index)
    features["FUNDING_AT"] = to_naive(_coalesce(df, funding_columns)).set_axis(df.index)
    features["DUE_WEEKDAY"] = features["DUE_AT"].dt.day_name()
    features["DAYS_TO_DUE"] = ((features["DUE_AT"] - features["FUNDING_AT"]).dt.total_seconds() / 86400).round(2)
    return features


def _prepare_history_features(lifecycle_source_df: pd.DataFrame) -> pd.DataFrame:
    """Turn observed lifecycles into context rows with k target and as-of priors.

    Rows without company, customer, due, funding or release dates are dropped. Priors use
    only outcomes known before each row's own funding date.
    """
    columns = [_INDEX_COLUMN, *_FEATURE_COLUMNS, _PRED_TARGET_COLUMN, "CUSTOMER_KEY", "FUNDING_AT", "RELEASE_DAY_FRACTION"]
    if lifecycle_source_df.empty:
        return pd.DataFrame(columns=columns)
    history = _base_features(lifecycle_source_df, _HISTORY_FUNDING_COLUMNS)
    released = to_naive(_coalesce(lifecycle_source_df, _HISTORY_RELEASE_COLUMNS)).set_axis(lifecycle_source_df.index)
    history[_PRED_TARGET_COLUMN] = reconciliation_offset(history["DUE_AT"], released)
    history["RELEASE_DAY_FRACTION"] = (released - released.dt.normalize()).dt.total_seconds() / 86400
    history = history.dropna(subset=["COMPANY_CODE", "CUSTOMER_KEY", "DUE_AT", "FUNDING_AT", _PRED_TARGET_COLUMN]).copy()
    history = history[released.loc[history.index] > history["FUNDING_AT"]]
    history["CUST_MEDIAN_K"], history["CUST_HISTORY_N"] = as_of_customer_priors(
        history["CUSTOMER_KEY"], history["FUNDING_AT"], released.loc[history.index], history[_PRED_TARGET_COLUMN])
    history[_INDEX_COLUMN] = [f"h_{position}" for position in range(len(history))]
    return history[columns].reset_index(drop=True)


def _prepare_candidate_features(candidates_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    """Candidate features; priors come from all admitted history of the same customer."""
    features = _base_features(candidates_df, _CANDIDATE_FUNDING_COLUMNS)
    stats = history_df.groupby("CUSTOMER_KEY")[_PRED_TARGET_COLUMN].agg(["median", "size"])
    features["CUST_MEDIAN_K"] = features["CUSTOMER_KEY"].map(stats["median"]).astype("float64")
    features["CUST_HISTORY_N"] = features["CUSTOMER_KEY"].map(stats["size"]).fillna(0).astype(int)
    features[_INDEX_COLUMN] = [f"q_{index}" for index in range(len(features))]
    return features.reset_index(drop=True)


def select_context(history_df: pd.DataFrame, customer_key: str, company_code: str, *,
                   min_rows: int, max_rows: int) -> pd.DataFrame:
    """Customer-first context for one customer's queries.

    Sends only the customer's own rows (newest first, at most max_rows). When the
    customer has fewer than min_rows, the context is topped up to min_rows with rows of
    the same company code, then any other rows, newest first. Mixing many other
    customers into the context degraded accuracy in validation.

    Returns:
        Context rows with index, feature and target columns.
    """
    tier = (history_df["CUSTOMER_KEY"] != customer_key).astype(int) + (history_df["COMPANY_CODE"] != company_code).astype(int)
    ordered = history_df.assign(_tier=tier).sort_values(["_tier", "FUNDING_AT"], ascending=[True, False], kind="mergesort")
    own = int((ordered["_tier"] == 0).sum())
    limit = min(int(max_rows), max(own, int(min_rows)))
    return ordered.head(limit)[[_INDEX_COLUMN, *_FEATURE_COLUMNS, _PRED_TARGET_COLUMN]].reset_index(drop=True)


def _load_rpt1_client_class() -> Any:
    current_file = Path(__file__).resolve()
    api_dir = current_file.parents[3]
    repo_root = current_file.parents[4]

    candidate_paths = [
        api_dir / "rpt1" / "rpt1_client.py",
        repo_root / "rpt1" / "rpt1_client.py",
        repo_root / "rpt1-example" / "rpt1_client.py",
    ]
    module_path = next((path for path in candidate_paths if path.exists()), None)
    if module_path is None:
        checked = ", ".join(str(path) for path in candidate_paths)
        raise FileNotFoundError(f"RPT-1 client not found. Checked: {checked}")

    spec = importlib.util.spec_from_file_location("rpt1_client_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "RPT1Client"):
        raise ImportError("rpt1_client.py does not export RPT1Client")
    return module.RPT1Client


def _existing_lifetime_mask(df: pd.DataFrame) -> pd.Series:
    """Rows that already carry a lifetime are not re-estimated."""
    mask = pd.Series(False, index=df.index)
    for column in ("expected_lifetime_weeks", "expected_lifetime_days"):
        if column in df.columns:
            mask |= df[column].notna()
    return mask


def _build_candidate_batches(
    *,
    feature_df: pd.DataFrame,
    history_df: pd.DataFrame,
    candidate_indices: List[int],
    config: LifetimeEstimationConfig,
) -> List[_LifetimeBatch]:
    """One context per (customer, company code); candidates split into query batches."""
    batches: List[_LifetimeBatch] = []
    usable = [idx for idx in candidate_indices
              if pd.notna(feature_df.at[idx, "CUSTOMER_KEY"]) and pd.notna(feature_df.at[idx, "DUE_AT"])]
    groups = feature_df.loc[usable].groupby(["CUSTOMER_KEY", "COMPANY_CODE"], dropna=False, sort=False)
    for (customer_key, company_code), group in groups:
        context_df = select_context(history_df, customer_key, company_code,
                                    min_rows=config.context_min_rows, max_rows=config.context_max_rows)
        if context_df.empty:
            continue
        indices = group.index.tolist()
        size = max(1, int(config.query_batch_size))
        for start in range(0, len(indices), size):
            batch_indices = indices[start:start + size]
            query_df = feature_df.loc[batch_indices, [_INDEX_COLUMN, *_FEATURE_COLUMNS]].reset_index(drop=True)
            batches.append(_LifetimeBatch(len(batches), batch_indices, context_df, query_df))
    return batches


def _predict_batch(*, client: Any, batch: _LifetimeBatch, prediction_placeholder: str) -> _LifetimeBatchResult:
    """Call RPT-1 once; return whole-file k per candidate (rounded, clipped to context range)."""
    try:
        client.fit(
            context_df=batch.context_df,
            target_columns=[_PRED_TARGET_COLUMN],
            index_column=_INDEX_COLUMN,
            task_types={_PRED_TARGET_COLUMN: "regression"},
            prediction_placeholder=prediction_placeholder,
        )
        result = client.predict(batch.query_df)
        usage = getattr(result, "metadata", {}).get("rpt1_usage", {})
        lowest, highest = batch.context_df[_PRED_TARGET_COLUMN].min(), batch.context_df[_PRED_TARGET_COLUMN].max()
        id_to_candidate = dict(zip(batch.query_df[_INDEX_COLUMN], batch.batch_indices))
        predicted_rows: List[Dict[str, Any]] = []
        for _, row in result.predictions_df.iterrows():
            idx = id_to_candidate.get(str(row.get(_INDEX_COLUMN, "")))
            raw = pd.to_numeric(row.get(_PRED_TARGET_COLUMN), errors="coerce")
            if idx is None or pd.isna(raw):
                continue
            confidence = pd.to_numeric(row.get(f"{_PRED_TARGET_COLUMN}__confidence"), errors="coerce")
            predicted_rows.append({
                "candidate_idx": idx,
                "recon_k": int(min(max(round(float(raw)), lowest), highest)),
                "confidence": float(confidence) if pd.notna(confidence) else pd.NA,
            })
        return _LifetimeBatchResult(
            batch_id=batch.batch_id,
            predicted_rows=predicted_rows,
            context_rows=len(batch.context_df),
            query_rows=len(batch.query_df),
            input_cells=int(usage.get("input_cells", (len(batch.context_df) + len(batch.query_df)) * len(batch.context_df.columns))),
            predictions_returned=int(usage.get("prediction_count", len(result.predictions_df))),
            api_calls=1,
        )
    except Exception as exc:  # pragma: no cover - runtime/network specific
        return _LifetimeBatchResult(
            batch_id=batch.batch_id,
            predicted_rows=[],
            context_rows=len(batch.context_df),
            query_rows=len(batch.query_df),
            input_cells=(len(batch.context_df) + len(batch.query_df)) * len(batch.context_df.columns),
            predictions_returned=0,
            api_calls=0,
            error=str(exc),
        )


def estimate_candidate_lifetime_with_rpt1(
    candidates_df: pd.DataFrame,
    lifecycle_source_df: pd.DataFrame,
    *,
    config: LifetimeEstimationConfig,
    progress_callback: Callable[[Dict[str, Any]], None] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Estimate expected invoice lifetimes with RPT-1 and the reconciliation calendar.

    Args:
        candidates_df: Invoices to estimate. Needs company code, customer and due date;
            the funding date comes from 'Planned Funding Date' (workspace runs) or the
            history funding columns (evaluation). Rows that already carry a lifetime are
            left untouched.
        lifecycle_source_df: Observed lifecycles (normalized HANA dataset rows) with
            funding ('credit_start') and release ('credit_release') timestamps.
        config: Estimator settings.
        progress_callback: Optional callable receiving counters after every batch.

    Returns:
        (candidates, report). Candidates gain expected_lifetime_days (whole days,
        >= 1), expected_lifetime_weeks, expected_lifetime_confidence,
        expected_lifetime_source ('RPT-1' or 'fallback_default_weeks'),
        expected_recon_k (after margin), expected_recon_k_model (RPT-1 output) and
        expected_release_date (ISO date). The report carries call counters and errors.
    """
    report: Dict[str, Any] = {
        "enabled": bool(config.enabled),
        "status": "skipped",
        "target_formulation": "rpt1_reconciliation_offset_k",
        "release_margin_quantile": config.release_margin_quantile,
        "requested_candidates": int(len(candidates_df)),
        "predicted_candidates": 0,
        "context_min_rows": int(config.context_min_rows),
        "context_max_rows": int(config.context_max_rows),
        "query_batch_size": int(config.query_batch_size),
        "historical_rows_available": 0,
        "api_calls": 0,
        "rpt1_input_cells_sent": 0,
        "rpt1_predictions_returned": 0,
        "max_context_rows_sent": 0,
        "max_query_rows_sent": 0,
        "avg_context_rows_sent": 0.0,
        "avg_query_rows_sent": 0.0,
        "batches_total": 0,
        "batches_completed": 0,
        "max_parallel_calls": int(config.max_parallel_calls),
        "retryable_error_count": 0,
        "margin_applied_candidates": 0,
        "fallback_candidates": 0,
        "errors": [],
    }

    if candidates_df.empty:
        report["status"] = "no_candidates"
        return candidates_df.copy(), report
    if not config.enabled:
        report["status"] = "disabled"
        return candidates_df.copy(), report

    history_df = _prepare_history_features(lifecycle_source_df)
    report["historical_rows_available"] = int(len(history_df))
    if history_df.empty:
        report["status"] = "missing_history"
        return candidates_df.copy(), report

    candidates_out = candidates_df.copy().reset_index(drop=True)
    for column in ("expected_lifetime_days", "expected_lifetime_weeks", "expected_lifetime_confidence",
                   "expected_lifetime_source", "expected_recon_k", "expected_recon_k_model", "expected_release_date"):
        if column not in candidates_out.columns:
            candidates_out[column] = pd.NA

    def _apply_default_lifetime_fallback(indices: List[int]) -> int:
        """Four-week fallback for rows without a model value; acknowledged later by the user."""
        weeks = max(1, int(config.default_lifetime_weeks))
        assigned = 0
        for idx in indices:
            if pd.isna(candidates_out.at[idx, "expected_lifetime_weeks"]):
                candidates_out.at[idx, "expected_lifetime_weeks"] = weeks
                assigned += 1
            if pd.isna(candidates_out.at[idx, "expected_lifetime_days"]):
                candidates_out.at[idx, "expected_lifetime_days"] = weeks * 7
            if pd.isna(candidates_out.at[idx, "expected_lifetime_source"]):
                candidates_out.at[idx, "expected_lifetime_source"] = "fallback_default_weeks"
        return assigned

    feature_df = _prepare_candidate_features(candidates_out, history_df)
    candidate_indices = feature_df[~_existing_lifetime_mask(candidates_out)].index.tolist()
    if not candidate_indices:
        report["status"] = "already_populated"
        return candidates_out, report

    try:
        RPT1Client = _load_rpt1_client_class()
        probe = RPT1Client.from_env(
            env_path=config.env_path,
            timeout_seconds=int(config.timeout_seconds),
            max_retries=int(config.max_retries),
            retry_backoff_seconds=float(config.retry_backoff_seconds),
        )
        # Resolve the AI Core deployment once up front: a missing or ambiguous model
        # fails here as init_failed, and worker clients reuse the cached result.
        resolve = getattr(probe, "_resolve_deployment_url", None)
        if callable(resolve):
            resolve()
            report["rpt1_model"] = {
                "model_name": getattr(probe, "model_name", None),
                "model_version": getattr(probe, "model_version", None),
                "deployment_id": getattr(probe, "deployment_id", None),
            }
    except Exception as exc:  # pragma: no cover - environment-specific
        logger.warning("RPT-1 initialization failed, using default lifetime fallback: %s", exc)
        report["status"] = "init_failed"
        report["errors"].append(str(exc))
        report["fallback_candidates"] = int(_apply_default_lifetime_fallback(candidate_indices))
        return candidates_out, report

    batches = _build_candidate_batches(
        feature_df=feature_df, history_df=history_df, candidate_indices=candidate_indices, config=config)
    report["batches_total"] = int(len(batches))
    if not batches:
        report["status"] = "no_batches"
        report["fallback_candidates"] = int(_apply_default_lifetime_fallback(candidate_indices))
        return candidates_out, report

    def _emit_progress_update() -> None:
        if progress_callback is None:
            return
        progress_callback({key: int(report[key]) for key in (
            "batches_total", "batches_completed", "api_calls", "rpt1_input_cells_sent",
            "rpt1_predictions_returned", "predicted_candidates", "fallback_candidates",
            "retryable_error_count", "max_parallel_calls")})

    thread_local = threading.local()

    def _process_batch(batch: _LifetimeBatch) -> _LifetimeBatchResult:
        client = getattr(thread_local, "client", None)
        if client is None:
            client = RPT1Client.from_env(
                env_path=config.env_path,
                timeout_seconds=int(config.timeout_seconds),
                max_retries=int(config.max_retries),
                retry_backoff_seconds=float(config.retry_backoff_seconds),
            )
            thread_local.client = client
        return _predict_batch(client=client, batch=batch, prediction_placeholder=config.prediction_placeholder)

    # Calendar inputs shared by every prediction.
    day_fraction = float(history_df["RELEASE_DAY_FRACTION"].median())
    margins = customer_margin_offsets(history_df["CUSTOMER_KEY"], history_df[_PRED_TARGET_COLUMN],
                                      config.release_margin_quantile, config.customer_min_rows)

    predicted_indices: set[int] = set()
    total_context_rows_sent = 0
    total_query_rows_sent = 0
    max_workers = max(1, int(config.max_parallel_calls))
    report["max_parallel_calls"] = max_workers
    _emit_progress_update()

    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = [executor.submit(_process_batch, batch) for batch in batches]
        for future in as_completed(futures):
            batch_result = future.result()
            report["batches_completed"] += 1
            report["api_calls"] += int(batch_result.api_calls)
            report["rpt1_input_cells_sent"] += int(batch_result.input_cells)
            report["rpt1_predictions_returned"] += int(batch_result.predictions_returned)
            total_context_rows_sent += int(batch_result.context_rows)
            total_query_rows_sent += int(batch_result.query_rows)
            report["max_context_rows_sent"] = max(int(report["max_context_rows_sent"]), int(batch_result.context_rows))
            report["max_query_rows_sent"] = max(int(report["max_query_rows_sent"]), int(batch_result.query_rows))

            if batch_result.error:
                report["errors"].append(f"batch_id={batch_result.batch_id}: {batch_result.error}")
                if any(code in str(batch_result.error) for code in ("429", "502", "503", "504")):
                    report["retryable_error_count"] += 1
                _emit_progress_update()
                continue

            for pred in batch_result.predicted_rows:
                idx = int(pred["candidate_idx"])
                features = feature_df.loc[idx]
                model_k = int(pred["recon_k"])
                final_k = max(model_k, margins.get(features["CUSTOMER_KEY"], model_k))
                release = release_timestamp(pd.Series([features["DUE_AT"]]), pd.Series([final_k]), day_fraction).iloc[0]
                funding = features["FUNDING_AT"]
                if pd.isna(release) or pd.isna(funding):
                    continue  # no funding date: leave the row for the fallback
                lifetime_days = max(1, int(math.ceil((release - funding).total_seconds() / 86400)))
                candidates_out.at[idx, "expected_lifetime_days"] = lifetime_days
                candidates_out.at[idx, "expected_lifetime_weeks"] = max(1, int(math.ceil(lifetime_days / 7.0)))
                candidates_out.at[idx, "expected_lifetime_confidence"] = pred["confidence"]
                candidates_out.at[idx, "expected_lifetime_source"] = "RPT-1"
                candidates_out.at[idx, "expected_recon_k"] = final_k
                candidates_out.at[idx, "expected_recon_k_model"] = model_k
                candidates_out.at[idx, "expected_release_date"] = release.date().isoformat()
                report["margin_applied_candidates"] += int(final_k > model_k)
                predicted_indices.add(idx)

            report["predicted_candidates"] = len(predicted_indices)
            _emit_progress_update()
    finally:
        executor.shutdown(wait=True)

    remaining_indices = [idx for idx in candidate_indices if idx not in predicted_indices]
    report["fallback_candidates"] = int(_apply_default_lifetime_fallback(remaining_indices))
    report["predicted_candidates"] = len(predicted_indices)
    _emit_progress_update()
    if report["api_calls"] > 0:
        calls = float(report["api_calls"])
        report["avg_context_rows_sent"] = total_context_rows_sent / calls
        report["avg_query_rows_sent"] = total_query_rows_sent / calls
    report["status"] = "completed" if predicted_indices else "no_predictions"
    return candidates_out, report
