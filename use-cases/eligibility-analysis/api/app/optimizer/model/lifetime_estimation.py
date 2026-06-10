"""
RPT-1 lifetime estimation for multi-week invoice optimization.

This module predicts expected invoice lifetime (in days) using SAP RPT-1
regression with in-context learning. For each candidate invoice, it builds a
relevant context window from historical lifecycle rows and calls RPT-1.
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
from typing import Any, Callable, Dict, Iterable, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

_PRED_TARGET_COLUMN = "TARGET_LIFETIME_DAYS"
_INDEX_COLUMN = "INVOICE_REF"
_FEATURE_COLUMNS = [
    "COMPANY_CODE",
    "CUSTOMER_ID",
    "PROGRAM_ID",
    "FUNDING_CURRENCY",
    "ORIGINAL_CURRENCY",
    "PURCHASE_PRICE",
    "INVOICE_AMOUNT",
    "ISSUANCE_DATE",
    "DUE_DATE",
    "TENOR_DAYS",
]


@dataclass(frozen=True)
class LifetimeEstimationConfig:
    enabled: bool = True
    context_min_rows: int = 500
    context_max_rows: int = 800
    query_batch_size: int = 25
    grouping_columns: Tuple[str, ...] = (
        "COMPANY_CODE",
        "CUSTOMER_ID",
        "PROGRAM_ID",
        "FUNDING_CURRENCY",
        "ORIGINAL_CURRENCY",
    )
    default_lifetime_weeks: int = 4
    prediction_placeholder: str = "[PREDICT]"
    timeout_seconds: int = 90
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0
    env_path: str | None = None
    max_parallel_calls: int = 2


@dataclass(frozen=True)
class _LifetimeBatch:
    batch_id: int
    batch_indices: List[int]
    context_df: pd.DataFrame
    query_df: pd.DataFrame
    id_to_candidate_idx: Dict[str, int]


@dataclass(frozen=True)
class _LifetimeBatchResult:
    batch_id: int
    predicted_rows: List[Dict[str, Any]]
    context_rows: int
    query_rows: int
    api_calls: int
    error: str | None = None


def _coalesce_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    if df.empty:
        return pd.Series([], dtype="object")
    base = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    for col in columns:
        if col in df.columns:
            base = base.where(base.notna(), df[col])
    return base


def _normalize_text(series: pd.Series) -> pd.Series:
    normalized = series.astype("object")
    normalized = normalized.where(pd.notna(normalized), pd.NA)
    normalized = normalized.where(
        normalized.isna(),
        normalized.astype(str).str.strip(),
    )
    normalized = normalized.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return normalized


def _prepare_history_features(lifecycle_source_df: pd.DataFrame) -> pd.DataFrame:
    if lifecycle_source_df.empty:
        return pd.DataFrame(columns=[_INDEX_COLUMN, *_FEATURE_COLUMNS, _PRED_TARGET_COLUMN, "CREDIT_START_DATE"])

    history = pd.DataFrame(index=lifecycle_source_df.index)
    history[_INDEX_COLUMN] = _normalize_text(
        _coalesce_columns(lifecycle_source_df, ["Invoice Reference", "invoice_reference", "optimizer_row_id"])
    ).fillna("history-row")
    history["COMPANY_CODE"] = _normalize_text(
        _coalesce_columns(lifecycle_source_df, ["Company Code", "company_code", "seller_id_external"])
    )
    history["CUSTOMER_ID"] = _normalize_text(
        _coalesce_columns(lifecycle_source_df, ["Customer", "debtor_id"])
    )
    history["PROGRAM_ID"] = _normalize_text(
        _coalesce_columns(lifecycle_source_df, ["PROGRAMA", "program_id"])
    )
    history["FUNDING_CURRENCY"] = _normalize_text(
        _coalesce_columns(lifecycle_source_df, ["Funding Currency", "funding_currency"])
    )
    history["ORIGINAL_CURRENCY"] = _normalize_text(
        _coalesce_columns(lifecycle_source_df, ["Currency", "ORIGINAL CURRENCY", "original_currency"])
    )

    history["PURCHASE_PRICE"] = pd.to_numeric(
        _coalesce_columns(lifecycle_source_df, ["Purchase Price", "candidate_amount"]),
        errors="coerce",
    )
    history["INVOICE_AMOUNT"] = pd.to_numeric(
        _coalesce_columns(lifecycle_source_df, ["Amount", "invoice_amount"]),
        errors="coerce",
    )
    history["ISSUANCE_DATE"] = pd.to_datetime(
        _coalesce_columns(lifecycle_source_df, ["Issuance date", "ISSUANCE DATE", "issuance_date"]),
        errors="coerce",
    )
    history["DUE_DATE"] = pd.to_datetime(
        _coalesce_columns(lifecycle_source_df, ["Due Date", "DUE DATE", "due_date"]),
        errors="coerce",
    )
    history["TENOR_DAYS"] = (history["DUE_DATE"] - history["ISSUANCE_DATE"]).dt.days
    history["CREDIT_START_DATE"] = pd.to_datetime(
        _coalesce_columns(lifecycle_source_df, ["credit_start", "Summary File Date (UTC)", "summary_file_date"]),
        errors="coerce",
    )
    history[_PRED_TARGET_COLUMN] = pd.to_numeric(
        _coalesce_columns(lifecycle_source_df, ["credit_duration_days"]),
        errors="coerce",
    )

    history = history[history[_PRED_TARGET_COLUMN].notna()].copy()
    history = history[history[_PRED_TARGET_COLUMN] > 0].copy()
    history[_PRED_TARGET_COLUMN] = history[_PRED_TARGET_COLUMN].clip(lower=1, upper=3650)

    return history.reset_index(drop=True)


def _prepare_candidate_features(candidates_df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=candidates_df.index)
    features[_INDEX_COLUMN] = _normalize_text(
        _coalesce_columns(candidates_df, ["Invoice Reference", "invoice_reference", "optimizer_row_id"])
    ).fillna("candidate-row")
    features["COMPANY_CODE"] = _normalize_text(
        _coalesce_columns(candidates_df, ["Company Code", "company_code", "seller_id_external"])
    )
    features["CUSTOMER_ID"] = _normalize_text(
        _coalesce_columns(candidates_df, ["Customer", "debtor_id"])
    )
    features["PROGRAM_ID"] = _normalize_text(
        _coalesce_columns(candidates_df, ["PROGRAMA", "program_id"])
    )
    features["FUNDING_CURRENCY"] = _normalize_text(
        _coalesce_columns(candidates_df, ["Funding Currency", "FUNDING CURRENCY", "funding_currency"])
    )
    features["ORIGINAL_CURRENCY"] = _normalize_text(
        _coalesce_columns(candidates_df, ["Currency", "ORIGINAL CURRENCY", "original_currency"])
    )
    features["PURCHASE_PRICE"] = pd.to_numeric(
        _coalesce_columns(candidates_df, ["Purchase Price", "candidate_amount"]),
        errors="coerce",
    )
    features["INVOICE_AMOUNT"] = pd.to_numeric(
        _coalesce_columns(candidates_df, ["Amount", "invoice_amount"]),
        errors="coerce",
    )
    features["ISSUANCE_DATE"] = pd.to_datetime(
        _coalesce_columns(candidates_df, ["Issuance date", "ISSUANCE DATE", "issuance_date"]),
        errors="coerce",
    )
    features["DUE_DATE"] = pd.to_datetime(
        _coalesce_columns(candidates_df, ["Due Date", "DUE DATE", "due_date"]),
        errors="coerce",
    )
    features["TENOR_DAYS"] = (features["DUE_DATE"] - features["ISSUANCE_DATE"]).dt.days

    return features.reset_index(drop=True)


def build_lifetime_payload(
    history_features_df: pd.DataFrame,
    candidate_feature_row: pd.Series,
    *,
    context_min_rows: int = 500,
    context_max_rows: int = 800,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build context/query payload frames for one invoice prediction."""
    if history_features_df.empty:
        return (
            pd.DataFrame(columns=[_INDEX_COLUMN, *_FEATURE_COLUMNS, _PRED_TARGET_COLUMN]),
            pd.DataFrame(columns=[_INDEX_COLUMN, *_FEATURE_COLUMNS]),
        )

    min_rows = max(1, int(context_min_rows))
    max_rows = max(min_rows, int(context_max_rows))

    ranked = history_features_df.copy()
    score = pd.Series(0.0, index=ranked.index, dtype="float64")

    for col, weight in (
        ("CUSTOMER_ID", 5.0),
        ("COMPANY_CODE", 4.0),
        ("PROGRAM_ID", 2.0),
        ("FUNDING_CURRENCY", 1.0),
        ("ORIGINAL_CURRENCY", 1.0),
    ):
        candidate_value = candidate_feature_row.get(col)
        if pd.notna(candidate_value):
            score += (ranked[col] == candidate_value).astype(float) * weight

    candidate_price = pd.to_numeric(candidate_feature_row.get("PURCHASE_PRICE"), errors="coerce")
    if pd.notna(candidate_price) and candidate_price > 0:
        rel_diff = (ranked["PURCHASE_PRICE"] - float(candidate_price)).abs() / float(candidate_price)
        score += (1.0 - rel_diff.fillna(1.0).clip(lower=0.0, upper=1.0))

    candidate_tenor = pd.to_numeric(candidate_feature_row.get("TENOR_DAYS"), errors="coerce")
    if pd.notna(candidate_tenor):
        tenor_diff = (ranked["TENOR_DAYS"] - float(candidate_tenor)).abs()
        score += (1.0 - (tenor_diff.fillna(365.0) / 365.0).clip(lower=0.0, upper=1.0))

    if "CREDIT_START_DATE" in ranked.columns:
        recency_rank = ranked["CREDIT_START_DATE"].rank(method="average", pct=True)
        score += recency_rank.fillna(0.0) * 0.25

    ranked["_rank_score"] = score
    ranked = ranked.sort_values(["_rank_score", "CREDIT_START_DATE"], ascending=[False, False], kind="mergesort")

    top = ranked.head(max_rows).copy()
    if len(top) < min_rows and len(ranked) > len(top):
        remainder = ranked.iloc[len(top):]
        top = pd.concat([top, remainder.head(min_rows - len(top))], ignore_index=True)

    context_df = top[[_INDEX_COLUMN, *_FEATURE_COLUMNS, _PRED_TARGET_COLUMN]].reset_index(drop=True)

    query_row = {
        _INDEX_COLUMN: candidate_feature_row.get(_INDEX_COLUMN),
    }
    for col in _FEATURE_COLUMNS:
        query_row[col] = candidate_feature_row.get(col)
    query_df = pd.DataFrame([query_row], columns=[_INDEX_COLUMN, *_FEATURE_COLUMNS])

    return context_df, query_df


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
    if df.empty:
        return pd.Series([], dtype="bool")
    has_weeks = "expected_lifetime_weeks" in df.columns and df["expected_lifetime_weeks"].notna()
    has_days = "expected_lifetime_days" in df.columns and df["expected_lifetime_days"].notna()
    if isinstance(has_weeks, bool):
        has_weeks = pd.Series([has_weeks] * len(df), index=df.index)
    if isinstance(has_days, bool):
        has_days = pd.Series([has_days] * len(df), index=df.index)
    return has_weeks | has_days


def _safe_key_value(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"
    return str(value)


def _chunked(values: List[int], chunk_size: int) -> Iterable[List[int]]:
    size = max(1, int(chunk_size))
    for pos in range(0, len(values), size):
        yield values[pos : pos + size]


def _grouped_candidate_indices(
    feature_df: pd.DataFrame,
    candidate_indices: List[int],
    grouping_columns: Tuple[str, ...],
) -> List[List[int]]:
    if not candidate_indices:
        return []

    if not grouping_columns:
        return [candidate_indices]

    groups: Dict[Tuple[str, ...], List[int]] = {}
    for idx in candidate_indices:
        row = feature_df.loc[idx]
        key = tuple(_safe_key_value(row.get(col)) for col in grouping_columns)
        groups.setdefault(key, []).append(idx)

    # Largest groups first helps reduce calls for dominant cohorts.
    ordered = sorted(groups.values(), key=lambda g: len(g), reverse=True)
    return ordered


def _build_candidate_batches(
    *,
    feature_df: pd.DataFrame,
    history_df: pd.DataFrame,
    candidate_indices: List[int],
    config: LifetimeEstimationConfig,
) -> List[_LifetimeBatch]:
    batches: List[_LifetimeBatch] = []
    grouped_indices = _grouped_candidate_indices(
        feature_df,
        candidate_indices,
        tuple(config.grouping_columns),
    )
    batch_id = 0
    for group_indices in grouped_indices:
        for batch_indices in _chunked(group_indices, int(config.query_batch_size)):
            if not batch_indices:
                continue
            representative = feature_df.loc[batch_indices[0]]
            context_df, _ = build_lifetime_payload(
                history_df,
                representative,
                context_min_rows=config.context_min_rows,
                context_max_rows=config.context_max_rows,
            )
            if context_df.empty:
                continue
            batch_features = feature_df.loc[batch_indices, [_INDEX_COLUMN, *_FEATURE_COLUMNS]].copy()
            query_ids = [f"q_{idx}" for idx in batch_indices]
            id_to_candidate_idx = dict(zip(query_ids, batch_indices))
            batch_features[_INDEX_COLUMN] = query_ids
            query_df = batch_features.reset_index(drop=True)
            batches.append(
                _LifetimeBatch(
                    batch_id=batch_id,
                    batch_indices=list(batch_indices),
                    context_df=context_df,
                    query_df=query_df,
                    id_to_candidate_idx=id_to_candidate_idx,
                )
            )
            batch_id += 1
    return batches


def _predict_batch(
    *,
    client: Any,
    batch: _LifetimeBatch,
    prediction_placeholder: str,
) -> _LifetimeBatchResult:
    try:
        client.fit(
            context_df=batch.context_df,
            target_columns=[_PRED_TARGET_COLUMN],
            index_column=_INDEX_COLUMN,
            task_types={_PRED_TARGET_COLUMN: "regression"},
            prediction_placeholder=prediction_placeholder,
        )
        result = client.predict(batch.query_df)
        predicted_rows: List[Dict[str, Any]] = []
        if not result.predictions_df.empty:
            for _, pred_row in result.predictions_df.iterrows():
                query_id = str(pred_row.get(_INDEX_COLUMN, ""))
                idx = batch.id_to_candidate_idx.get(query_id)
                if idx is None:
                    continue
                raw_value = pd.to_numeric(pred_row.get(_PRED_TARGET_COLUMN), errors="coerce")
                if pd.isna(raw_value):
                    continue
                confidence = pd.to_numeric(
                    pred_row.get(f"{_PRED_TARGET_COLUMN}__confidence"),
                    errors="coerce",
                )
                predicted_rows.append(
                    {
                        "candidate_idx": idx,
                        "lifetime_days": max(1, int(round(float(raw_value)))),
                        "confidence": float(confidence) if pd.notna(confidence) else pd.NA,
                    }
                )
        return _LifetimeBatchResult(
            batch_id=batch.batch_id,
            predicted_rows=predicted_rows,
            context_rows=int(len(batch.context_df)),
            query_rows=int(len(batch.query_df)),
            api_calls=1,
        )
    except Exception as exc:  # pragma: no cover - runtime/network specific
        return _LifetimeBatchResult(
            batch_id=batch.batch_id,
            predicted_rows=[],
            context_rows=int(len(batch.context_df)),
            query_rows=int(len(batch.query_df)),
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
    """Estimate expected lifetime for invoices using RPT-1 regression."""
    report: Dict[str, Any] = {
        "enabled": bool(config.enabled),
        "status": "skipped",
        "requested_candidates": int(len(candidates_df)),
        "predicted_candidates": 0,
        "context_min_rows": int(config.context_min_rows),
        "context_max_rows": int(config.context_max_rows),
        "query_batch_size": int(config.query_batch_size),
        "grouping_columns": list(config.grouping_columns),
        "historical_rows_available": 0,
        "api_calls": 0,
        "max_context_rows_sent": 0,
        "max_query_rows_sent": 0,
        "avg_context_rows_sent": 0.0,
        "avg_query_rows_sent": 0.0,
        "batches_total": 0,
        "batches_completed": 0,
        "max_parallel_calls": int(config.max_parallel_calls),
        "retryable_error_count": 0,
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
    if "expected_lifetime_days" not in candidates_out.columns:
        candidates_out["expected_lifetime_days"] = pd.NA
    if "expected_lifetime_weeks" not in candidates_out.columns:
        candidates_out["expected_lifetime_weeks"] = pd.NA
    if "expected_lifetime_confidence" not in candidates_out.columns:
        candidates_out["expected_lifetime_confidence"] = pd.NA
    if "expected_lifetime_source" not in candidates_out.columns:
        candidates_out["expected_lifetime_source"] = pd.NA

    def _apply_default_lifetime_fallback(indices: List[int]) -> int:
        if not indices:
            return 0
        assigned = 0
        fallback_weeks = max(1, int(config.default_lifetime_weeks))
        fallback_days = int(fallback_weeks * 7)
        for idx in indices:
            if pd.isna(candidates_out.at[idx, "expected_lifetime_weeks"]):
                candidates_out.at[idx, "expected_lifetime_weeks"] = fallback_weeks
                assigned += 1
            if pd.isna(candidates_out.at[idx, "expected_lifetime_days"]):
                candidates_out.at[idx, "expected_lifetime_days"] = fallback_days
            if pd.isna(candidates_out.at[idx, "expected_lifetime_source"]):
                candidates_out.at[idx, "expected_lifetime_source"] = "fallback_default_weeks"
        return assigned

    feature_df = _prepare_candidate_features(candidates_out)
    missing_mask = ~_existing_lifetime_mask(candidates_out)
    candidate_indices = feature_df[missing_mask].index.tolist()
    if not candidate_indices:
        report["status"] = "already_populated"
        return candidates_out, report

    try:
        RPT1Client = _load_rpt1_client_class()
    except Exception as exc:  # pragma: no cover - environment-specific
        logger.warning("RPT-1 initialization failed, using default lifetime fallback: %s", exc)
        report["status"] = "init_failed"
        report["errors"].append(str(exc))
        report["fallback_candidates"] = int(_apply_default_lifetime_fallback(candidate_indices))
        return candidates_out, report

    try:
        _ = RPT1Client.from_env(
            env_path=config.env_path,
            timeout_seconds=int(config.timeout_seconds),
            max_retries=int(config.max_retries),
            retry_backoff_seconds=float(config.retry_backoff_seconds),
        )
    except Exception as exc:  # pragma: no cover - environment-specific
        logger.warning("RPT-1 initialization failed, using default lifetime fallback: %s", exc)
        report["status"] = "init_failed"
        report["errors"].append(str(exc))
        report["fallback_candidates"] = int(_apply_default_lifetime_fallback(candidate_indices))
        return candidates_out, report

    batches = _build_candidate_batches(
        feature_df=feature_df,
        history_df=history_df,
        candidate_indices=candidate_indices,
        config=config,
    )
    report["batches_total"] = int(len(batches))
    if not batches:
        report["status"] = "no_batches"
        report["fallback_candidates"] = int(_apply_default_lifetime_fallback(candidate_indices))
        return candidates_out, report

    def _emit_progress_update() -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "batches_total": int(report["batches_total"]),
                "batches_completed": int(report["batches_completed"]),
                "api_calls": int(report["api_calls"]),
                "predicted_candidates": int(report["predicted_candidates"]),
                "fallback_candidates": int(report["fallback_candidates"]),
                "retryable_errors": int(report["retryable_error_count"]),
                "max_parallel_calls": int(report["max_parallel_calls"]),
            }
        )

    predicted = 0
    predicted_indices: set[int] = set()
    total_context_rows_sent = 0
    total_query_rows_sent = 0
    thread_local = threading.local()

    def _get_client() -> Any:
        client = getattr(thread_local, "client", None)
        if client is None:
            client = RPT1Client.from_env(
                env_path=config.env_path,
                timeout_seconds=int(config.timeout_seconds),
                max_retries=int(config.max_retries),
                retry_backoff_seconds=float(config.retry_backoff_seconds),
            )
            thread_local.client = client
        return client

    def _process_batch(batch: _LifetimeBatch) -> _LifetimeBatchResult:
        client = _get_client()
        return _predict_batch(
            client=client,
            batch=batch,
            prediction_placeholder=config.prediction_placeholder,
        )

    max_workers = max(1, int(config.max_parallel_calls))
    report["max_parallel_calls"] = max_workers
    _emit_progress_update()

    if max_workers == 1:
        results_iterable = (_process_batch(batch) for batch in batches)
    else:
        executor = ThreadPoolExecutor(max_workers=max_workers)
        future_map = {executor.submit(_process_batch, batch): batch.batch_id for batch in batches}
        results_iterable = (future.result() for future in as_completed(future_map))

    try:
        for batch_result in results_iterable:
            report["batches_completed"] = int(report["batches_completed"]) + 1
            report["api_calls"] = int(report["api_calls"]) + int(batch_result.api_calls)
            total_context_rows_sent += int(batch_result.context_rows)
            total_query_rows_sent += int(batch_result.query_rows)
            report["max_context_rows_sent"] = max(int(report["max_context_rows_sent"]), int(batch_result.context_rows))
            report["max_query_rows_sent"] = max(int(report["max_query_rows_sent"]), int(batch_result.query_rows))

            if batch_result.error:
                report["errors"].append(f"batch_id={batch_result.batch_id}: {batch_result.error}")
                upper_error = str(batch_result.error).upper()
                if "429" in upper_error or "503" in upper_error or "502" in upper_error or "504" in upper_error:
                    report["retryable_error_count"] = int(report["retryable_error_count"]) + 1
                _emit_progress_update()
                continue

            for pred in batch_result.predicted_rows:
                idx = int(pred["candidate_idx"])
                lifetime_days = int(pred["lifetime_days"])
                lifetime_weeks = max(1, int(math.ceil(lifetime_days / 7.0)))
                confidence = pred["confidence"]
                candidates_out.at[idx, "expected_lifetime_days"] = lifetime_days
                candidates_out.at[idx, "expected_lifetime_weeks"] = lifetime_weeks
                candidates_out.at[idx, "expected_lifetime_confidence"] = confidence
                candidates_out.at[idx, "expected_lifetime_source"] = "RPT-1"
                predicted += 1
                predicted_indices.add(idx)

            report["predicted_candidates"] = int(predicted)
            _emit_progress_update()
    finally:
        if max_workers > 1:
            executor.shutdown(wait=True)

    remaining_indices = [idx for idx in candidate_indices if idx not in predicted_indices]
    report["fallback_candidates"] = int(_apply_default_lifetime_fallback(remaining_indices))
    report["predicted_candidates"] = predicted
    _emit_progress_update()
    if int(report["api_calls"]) > 0:
        calls = float(report["api_calls"])
        report["avg_context_rows_sent"] = total_context_rows_sent / calls
        report["avg_query_rows_sent"] = total_query_rows_sent / calls
    report["status"] = "completed" if predicted > 0 else "no_predictions"
    return candidates_out, report
