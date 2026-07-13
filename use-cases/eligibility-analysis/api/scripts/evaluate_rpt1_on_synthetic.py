#!/usr/bin/env python3
"""
Evaluate RPT-1 lifetime prediction on synthetic extraction candidates.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import pandas as pd


def _resolve_paths() -> Tuple[Path, Path]:
    script_path = Path(__file__).resolve()
    api_dir = script_path.parents[1]
    repo_root = script_path.parents[2]
    if str(api_dir) not in sys.path:
        sys.path.insert(0, str(api_dir))
    return repo_root, api_dir


REPO_ROOT, API_DIR = _resolve_paths()
DEFAULT_CONTEXT_INPUT_PATH = REPO_ROOT / "data" / "2026" / "EXTRACTION BTP.xlsx"

from app.optimizer.io.load_extraction import load_extraction  # noqa: E402
from app.optimizer.model.lifecycle import derive_lifecycle  # noqa: E402
from app.optimizer.model.lifetime_estimation import (  # noqa: E402
    LifetimeEstimationConfig,
    estimate_candidate_lifetime_with_rpt1,
)


def _mape(actual: pd.Series, predicted: pd.Series) -> float | None:
    valid = actual > 0
    if not valid.any():
        return None
    return float((((predicted[valid] - actual[valid]).abs() / actual[valid]).mean()) * 100.0)


def _compute_metrics(
    merged_df: pd.DataFrame,
    default_lifetime_weeks: int,
) -> Dict[str, Any]:
    valid = merged_df[
        merged_df["actual_lifetime_days"].notna()
        & merged_df["predicted_lifetime_days"].notna()
    ].copy()
    baseline_days = float(max(1, int(default_lifetime_weeks)) * 7)

    if valid.empty:
        return {
            "evaluated_rows": 0,
            "mae_days": None,
            "rmse_days": None,
            "mape_pct": None,
            "baseline_default_days": baseline_days,
            "baseline_mae_days": None,
            "baseline_rmse_days": None,
        }

    actual = pd.to_numeric(valid["actual_lifetime_days"], errors="coerce")
    pred = pd.to_numeric(valid["predicted_lifetime_days"], errors="coerce")
    err = pred - actual
    baseline_err = baseline_days - actual
    return {
        "evaluated_rows": int(len(valid)),
        "mae_days": float(err.abs().mean()),
        "rmse_days": float(math.sqrt((err.pow(2)).mean())),
        "mape_pct": _mape(actual, pred),
        "baseline_default_days": baseline_days,
        "baseline_mae_days": float(baseline_err.abs().mean()),
        "baseline_rmse_days": float(math.sqrt((baseline_err.pow(2)).mean())),
    }


def evaluate_rpt1_on_synthetic(
    *,
    candidate_input_path: str | Path,
    context_input_path: str | Path = DEFAULT_CONTEXT_INPUT_PATH,
    output_dir: str | Path | None = None,
    candidate_sheet_name: str = "SAPUI5 Export",
    context_sheet_name: str = "SAPUI5 Export",
    release_event: str = "reconciliation_file_date",
    default_lifetime_weeks: int = 4,
    context_min_rows: int = 500,
    context_max_rows: int = 800,
    query_batch_size: int = 25,
    timeout_seconds: int = 90,
    max_retries: int = 3,
    retry_backoff_seconds: float = 1.0,
    rpt1_env_path: str | None = None,
    estimator: Callable[..., Tuple[pd.DataFrame, Dict[str, Any]]] = estimate_candidate_lifetime_with_rpt1,
) -> Dict[str, Any]:
    candidate_path = Path(candidate_input_path)
    context_path = Path(context_input_path)
    if not candidate_path.exists():
        raise FileNotFoundError(f"Candidate synthetic extraction file not found: {candidate_path}")
    if not context_path.exists():
        raise FileNotFoundError(f"Context extraction file not found: {context_path}")

    out_dir = Path(output_dir) if output_dir else candidate_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_df, _ = load_extraction(candidate_path, sheet_name=candidate_sheet_name)
    if candidate_df.empty:
        raise ValueError("Candidate synthetic extraction is empty.")

    if "Synthetic Row Type" in candidate_df.columns:
        candidate_eval = candidate_df[candidate_df["Synthetic Row Type"] == "candidate"].copy()
    else:
        candidate_eval = candidate_df.copy()

    if candidate_eval.empty:
        raise ValueError("No candidate rows found for evaluation (Synthetic Row Type = candidate).")

    if "Reconciliation File Date (UTC)" not in candidate_eval.columns:
        raise ValueError("Candidate data is missing 'Reconciliation File Date (UTC)'.")

    recon_present = candidate_eval["Reconciliation File Date (UTC)"].notna().sum()
    if recon_present == 0:
        raise ValueError(
            "Synthetic candidates do not include reconciliation end-dates. "
            "Regenerate with --enable-reconciliation-date to run reliability evaluation."
        )

    candidate_eval = derive_lifecycle(candidate_eval, release_event=release_event)
    candidate_eval["actual_lifetime_days"] = pd.to_numeric(
        candidate_eval["credit_duration_days"], errors="coerce"
    )

    context_df, _ = load_extraction(context_path, sheet_name=context_sheet_name)
    context_lifecycle_df = derive_lifecycle(context_df, release_event=release_event)

    lifetime_cfg = LifetimeEstimationConfig(
        enabled=True,
        context_min_rows=int(context_min_rows),
        context_max_rows=int(context_max_rows),
        query_batch_size=int(query_batch_size),
        default_lifetime_weeks=int(default_lifetime_weeks),
        timeout_seconds=int(timeout_seconds),
        max_retries=int(max_retries),
        retry_backoff_seconds=float(retry_backoff_seconds),
        env_path=rpt1_env_path,
    )
    predicted_df, estimation_report = estimator(
        candidate_eval.copy(),
        context_lifecycle_df,
        config=lifetime_cfg,
    )

    result = candidate_eval.copy().reset_index(drop=True)
    predicted_df = predicted_df.reset_index(drop=True)
    result["predicted_lifetime_days"] = pd.to_numeric(
        predicted_df.get("expected_lifetime_days"), errors="coerce"
    )
    result["predicted_lifetime_weeks"] = pd.to_numeric(
        predicted_df.get("expected_lifetime_weeks"), errors="coerce"
    )
    result["prediction_confidence"] = pd.to_numeric(
        predicted_df.get("expected_lifetime_confidence"), errors="coerce"
    )
    result["prediction_source"] = predicted_df.get("expected_lifetime_source")
    result["abs_error_days"] = (
        result["predicted_lifetime_days"] - result["actual_lifetime_days"]
    ).abs()

    metrics = _compute_metrics(result, default_lifetime_weeks=default_lifetime_weeks)

    predictions_path = out_dir / "rpt1_eval_predictions.csv"
    metrics_path = out_dir / "rpt1_eval_metrics.json"

    export_columns = [
        "Invoice Reference",
        "Company Code",
        "Customer",
        "Offer File Date (UTC)",
        "Summary File Date (UTC)",
        "Reconciliation File Date (UTC)",
        "actual_lifetime_days",
        "predicted_lifetime_days",
        "predicted_lifetime_weeks",
        "prediction_confidence",
        "prediction_source",
        "abs_error_days",
    ]
    present_columns = [col for col in export_columns if col in result.columns]
    result[present_columns].to_csv(predictions_path, index=False)

    payload = {
        "candidate_input_path": str(candidate_path),
        "context_input_path": str(context_path),
        "release_event": release_event,
        "default_lifetime_weeks": int(default_lifetime_weeks),
        "rows_total_candidates": int(len(candidate_eval)),
        "rows_with_reconciliation_date": int(recon_present),
        "metrics": metrics,
        "lifetime_estimation_report": estimation_report,
        "predictions_output": str(predictions_path),
    }
    metrics_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    return {
        "metrics_output": str(metrics_path),
        "predictions_output": str(predictions_path),
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RPT-1 lifetime prediction on synthetic extraction data.")
    parser.add_argument(
        "--candidate-input-path",
        required=True,
        help="Path to synthetic extraction workbook (SAPUI5 Export).",
    )
    parser.add_argument(
        "--context-input-path",
        default=str(DEFAULT_CONTEXT_INPUT_PATH),
        help="Historical lifecycle context extraction path.",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory for evaluation artifacts.")
    parser.add_argument("--candidate-sheet-name", default="SAPUI5 Export", help="Candidate sheet name.")
    parser.add_argument("--context-sheet-name", default="SAPUI5 Export", help="Context sheet name.")
    parser.add_argument(
        "--release-event",
        default="reconciliation_file_date",
        choices=[
            "reconciliation_file_date",
            "paid_on",
            "reconciliation",
            "min_paid_or_repurchase",
        ],
        help="Lifecycle release event mode.",
    )
    parser.add_argument("--default-lifetime-weeks", type=int, default=4, help="Fallback lifetime weeks.")
    parser.add_argument("--context-min-rows", type=int, default=500, help="RPT-1 context minimum rows.")
    parser.add_argument("--context-max-rows", type=int, default=800, help="RPT-1 context maximum rows.")
    parser.add_argument("--query-batch-size", type=int, default=25, help="RPT-1 query batch size.")
    parser.add_argument("--timeout-seconds", type=int, default=90, help="RPT-1 timeout per call.")
    parser.add_argument("--max-retries", type=int, default=3, help="RPT-1 max retries.")
    parser.add_argument("--retry-backoff-seconds", type=float, default=1.0, help="RPT-1 retry backoff.")
    parser.add_argument("--rpt1-env-path", default=None, help="Optional .env path for RPT-1 credentials.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = evaluate_rpt1_on_synthetic(
        candidate_input_path=args.candidate_input_path,
        context_input_path=args.context_input_path,
        output_dir=args.output_dir,
        candidate_sheet_name=args.candidate_sheet_name,
        context_sheet_name=args.context_sheet_name,
        release_event=args.release_event,
        default_lifetime_weeks=args.default_lifetime_weeks,
        context_min_rows=args.context_min_rows,
        context_max_rows=args.context_max_rows,
        query_batch_size=args.query_batch_size,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        rpt1_env_path=args.rpt1_env_path,
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
