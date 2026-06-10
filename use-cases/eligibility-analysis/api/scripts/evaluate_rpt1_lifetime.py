#!/usr/bin/env python3
"""
Backtest RPT-1 lifetime estimation on historical extraction data.

This script uses rows with known credit lifecycle duration as ground truth and
evaluates how accurately the RPT-1 in-context regression step predicts lifetime
for held-out invoices.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd


def _resolve_paths() -> Tuple[Path, Path]:
    script_path = Path(__file__).resolve()
    api_dir = script_path.parents[1]
    repo_root = script_path.parents[2]
    if str(api_dir) not in sys.path:
        sys.path.insert(0, str(api_dir))
    return repo_root, api_dir


REPO_ROOT, API_DIR = _resolve_paths()

from app.optimizer.io.load_extraction import load_extraction  # noqa: E402
from app.optimizer.model.lifecycle import derive_lifecycle  # noqa: E402
from app.optimizer.model.lifetime_estimation import (  # noqa: E402
    LifetimeEstimationConfig,
    estimate_candidate_lifetime_with_rpt1,
)


def _default_env_path() -> str | None:
    rpt_env = REPO_ROOT / "rpt1" / ".env"
    api_env = API_DIR / ".env"
    if rpt_env.exists():
        return str(rpt_env)
    if api_env.exists():
        return str(api_env)
    return None


def _default_input_path() -> str:
    return str(REPO_ROOT / "data" / "2026" / "EXTRACTION BTP.xlsx")


def _prepare_ground_truth(
    input_path: str,
    sheet_name: str,
    release_event: str,
) -> pd.DataFrame:
    extraction_df, _ = load_extraction(input_path=input_path, sheet_name=sheet_name)
    lifecycle_df = derive_lifecycle(extraction_df, release_event=release_event)

    lifecycle_df["credit_duration_days"] = pd.to_numeric(
        lifecycle_df.get("credit_duration_days"),
        errors="coerce",
    )
    gt = lifecycle_df[
        lifecycle_df["credit_duration_days"].notna()
        & (lifecycle_df["credit_duration_days"] > 0)
    ].copy()
    gt["credit_duration_days"] = gt["credit_duration_days"].astype(float)
    return gt.reset_index(drop=True)


def _train_test_split(
    df: pd.DataFrame,
    *,
    test_ratio: float,
    max_test_rows: int | None,
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < 2:
        raise ValueError("Need at least 2 rows with known lifetime for evaluation.")

    shuffled = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    test_size = max(1, int(round(len(shuffled) * test_ratio)))
    if max_test_rows is not None and max_test_rows > 0:
        test_size = min(test_size, int(max_test_rows))
    test_size = min(test_size, len(shuffled) - 1)

    test_df = shuffled.iloc[:test_size].copy().reset_index(drop=True)
    train_df = shuffled.iloc[test_size:].copy().reset_index(drop=True)
    return train_df, test_df


def _bucketize(days: pd.Series) -> pd.Series:
    bins = [0, 30, 60, 90, 180, float("inf")]
    labels = ["0-30", "31-60", "61-90", "91-180", "181+"]
    return pd.cut(days, bins=bins, labels=labels, include_lowest=True)


def _compute_metrics(result_df: pd.DataFrame, default_lifetime_weeks: int) -> Dict[str, Any]:
    baseline_days = float(default_lifetime_weeks * 7)
    actual_all = pd.to_numeric(result_df["actual_lifetime_days"], errors="coerce")
    baseline_valid = actual_all[actual_all.notna()]
    baseline_mae = (
        float((baseline_days - baseline_valid).abs().mean())
        if not baseline_valid.empty
        else None
    )
    baseline_rmse = (
        float(math.sqrt(((baseline_days - baseline_valid).pow(2)).mean()))
        if not baseline_valid.empty
        else None
    )

    valid = result_df[result_df["predicted_lifetime_days"].notna()].copy()
    if valid.empty:
        return {
            "predictions_made": 0,
            "mae_days": None,
            "rmse_days": None,
            "mape_pct": None,
            "r2": None,
            "bucket_accuracy_pct": None,
            "baseline_default_days": baseline_days,
            "baseline_mae_days": baseline_mae,
            "baseline_rmse_days": baseline_rmse,
        }

    actual = pd.to_numeric(valid["actual_lifetime_days"], errors="coerce")
    pred = pd.to_numeric(valid["predicted_lifetime_days"], errors="coerce")
    err = pred - actual
    abs_err = err.abs()

    mae = float(abs_err.mean())
    rmse = float(math.sqrt((err.pow(2)).mean()))
    positive_actual = actual[actual > 0]
    mape = (
        float((abs_err[positive_actual.index] / positive_actual).mean() * 100.0)
        if not positive_actual.empty
        else None
    )

    sst = float(((actual - actual.mean()) ** 2).sum())
    sse = float((err ** 2).sum())
    r2 = (1.0 - sse / sst) if sst > 0 else None

    actual_bucket = _bucketize(actual)
    pred_bucket = _bucketize(pred)
    bucket_acc = float((actual_bucket == pred_bucket).mean() * 100.0)

    baseline_err = baseline_days - actual
    baseline_mae_pred_only = float(baseline_err.abs().mean())
    baseline_rmse_pred_only = float(math.sqrt((baseline_err.pow(2)).mean()))

    return {
        "predictions_made": int(len(valid)),
        "mae_days": mae,
        "rmse_days": rmse,
        "mape_pct": mape,
        "r2": r2,
        "bucket_accuracy_pct": bucket_acc,
        "baseline_default_days": baseline_days,
        "baseline_mae_days": baseline_mae,
        "baseline_rmse_days": baseline_rmse,
        "baseline_mae_days_on_predicted_subset": baseline_mae_pred_only,
        "baseline_rmse_days_on_predicted_subset": baseline_rmse_pred_only,
    }


def _build_result_frame(test_df: pd.DataFrame, predicted_df: pd.DataFrame) -> pd.DataFrame:
    out = test_df.copy().reset_index(drop=True)
    out["actual_lifetime_days"] = pd.to_numeric(out["credit_duration_days"], errors="coerce")
    out["predicted_lifetime_days"] = pd.to_numeric(
        predicted_df.get("expected_lifetime_days"),
        errors="coerce",
    )
    out["predicted_lifetime_weeks"] = pd.to_numeric(
        predicted_df.get("expected_lifetime_weeks"),
        errors="coerce",
    )
    out["prediction_confidence"] = pd.to_numeric(
        predicted_df.get("expected_lifetime_confidence"),
        errors="coerce",
    )
    out["prediction_source"] = predicted_df.get("expected_lifetime_source")
    out["abs_error_days"] = (out["predicted_lifetime_days"] - out["actual_lifetime_days"]).abs()
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RPT-1 invoice lifetime prediction accuracy.")
    parser.add_argument("--input-path", default=_default_input_path(), help="Path to extraction Excel.")
    parser.add_argument("--sheet-name", default="SAPUI5 Export", help="Sheet name in extraction workbook.")
    parser.add_argument(
        "--release-event",
        default="reconciliation_file_date",
        choices=[
            "reconciliation_file_date",
            "paid_on",
            "reconciliation",
            "min_paid_or_repurchase",
        ],
        help="Lifecycle release event used for ground-truth duration.",
    )
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Fraction of rows used as test set.")
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=100,
        help="Upper bound on test rows to score (controls API call volume).",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for split.")
    parser.add_argument("--context-min-rows", type=int, default=500, help="Min context rows target.")
    parser.add_argument("--context-max-rows", type=int, default=800, help="Max context rows.")
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=25,
        help="How many query invoices to score per RPT-1 call.",
    )
    parser.add_argument(
        "--grouping-columns",
        default="COMPANY_CODE,CUSTOMER_ID,PROGRAM_ID,FUNDING_CURRENCY,ORIGINAL_CURRENCY",
        help="Comma-separated grouping columns for batching similar invoices.",
    )
    parser.add_argument("--default-lifetime-weeks", type=int, default=4, help="Default fallback lifetime in weeks.")
    parser.add_argument("--env-path", default=_default_env_path(), help="Path to .env with RPT-1 credentials.")
    parser.add_argument(
        "--disable-rpt1",
        action="store_true",
        help="Disable RPT-1 calls (useful for dry-run sanity checks).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(API_DIR / "data" / "lifetime_eval"),
        help="Directory where CSV/JSON outputs are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading extraction file: {args.input_path}")
    gt_df = _prepare_ground_truth(
        input_path=args.input_path,
        sheet_name=args.sheet_name,
        release_event=args.release_event,
    )
    print(f"Rows with known lifetime: {len(gt_df)}")

    train_df, test_df = _train_test_split(
        gt_df,
        test_ratio=args.test_ratio,
        max_test_rows=args.max_test_rows,
        random_seed=args.random_seed,
    )
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")

    grouping_columns = tuple(
        [part.strip() for part in str(args.grouping_columns).split(",") if part.strip()]
    )

    cfg = LifetimeEstimationConfig(
        enabled=not bool(args.disable_rpt1),
        context_min_rows=int(args.context_min_rows),
        context_max_rows=int(args.context_max_rows),
        query_batch_size=int(args.query_batch_size),
        grouping_columns=grouping_columns,
        default_lifetime_weeks=int(args.default_lifetime_weeks),
        env_path=args.env_path,
    )

    predicted_candidates_df, rpt_report = estimate_candidate_lifetime_with_rpt1(
        candidates_df=test_df.copy(),
        lifecycle_source_df=train_df.copy(),
        config=cfg,
    )

    result_df = _build_result_frame(test_df, predicted_candidates_df)
    metrics = _compute_metrics(result_df, default_lifetime_weeks=args.default_lifetime_weeks)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(Path(args.input_path).resolve()),
        "sheet_name": args.sheet_name,
        "release_event": args.release_event,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "grouping_columns": list(grouping_columns),
        "rpt1_report": rpt_report,
        "metrics": metrics,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    summary_path = output_dir / f"rpt1_lifetime_eval_summary_{stamp}.json"
    rows_path = output_dir / f"rpt1_lifetime_eval_rows_{stamp}.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    keep_cols = [
        "Invoice Reference",
        "Company Code",
        "Customer",
        "Issuance date",
        "Due Date",
        "actual_lifetime_days",
        "predicted_lifetime_days",
        "predicted_lifetime_weeks",
        "prediction_confidence",
        "prediction_source",
        "abs_error_days",
    ]
    available_cols = [c for c in keep_cols if c in result_df.columns]
    result_df.sort_values("abs_error_days", ascending=False, na_position="last")[available_cols].to_csv(
        rows_path, index=False
    )

    print("")
    print("=== RPT-1 Lifetime Evaluation ===")
    print(json.dumps(summary["metrics"], indent=2))
    print(f"RPT-1 status: {rpt_report.get('status')}")
    print(
        "Call stats: "
        f"api_calls={rpt_report.get('api_calls')}, "
        f"max_context_rows_sent={rpt_report.get('max_context_rows_sent')}, "
        f"avg_context_rows_sent={rpt_report.get('avg_context_rows_sent')}, "
        f"max_query_rows_sent={rpt_report.get('max_query_rows_sent')}, "
        f"avg_query_rows_sent={rpt_report.get('avg_query_rows_sent')}"
    )
    errors = rpt_report.get("errors") or []
    if errors:
        print(f"RPT-1 errors (showing up to 3/{len(errors)}):")
        for err in errors[:3]:
            print(f"- {err}")
    print(f"Summary JSON: {summary_path}")
    print(f"Rows CSV    : {rows_path}")


if __name__ == "__main__":
    main()
