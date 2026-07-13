"""
Run-level metrics and markdown report generation.

This module computes KPIs comparing the "baseline" (all eligible candidates
submitted without optimization) vs the "optimized" submission (the optimizer's
selected subset). Key metrics include:

  - Selected amount vs candidate amount (how much of the eligible pool is funded)
  - Top-3 customer concentration (risk concentration check)
  - FAILED-reason proxy count reduction (estimates how many limit failures the
    optimizer would prevent by not submitting invoices that would exceed limits)

The markdown summary is written to ``run_summary.md`` for quick human review.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from ..model.lifecycle import LifecycleProfile
from ..opt.optimizer_multi_week import MultiWeekOptimizationResult
from ..opt.optimizer_single_week import OptimizationResult


def _reason_failed_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask: True where the Reason column contains 'FAILED' (case-insensitive).

    This is a proxy for invoices that historically failed due to limit violations.
    The extraction's Reason field contains strings like "FAILED - Limit exceeded".
    """
    if "Reason" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    return df["Reason"].fillna("").astype(str).str.contains("FAILED", case=False, regex=True)


def compute_run_metrics(
    candidates_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    rule_excluded_df: pd.DataFrame,
    optimization_result: OptimizationResult,
    lifecycle_profile: LifecycleProfile,
) -> Dict[str, object]:
    """Compute all run-level KPIs for the summary report."""
    baseline_submitted = len(candidates_df)
    optimized_submitted = len(selected_df)

    baseline_failed_proxy = int(_reason_failed_mask(candidates_df).sum())
    optimized_failed_proxy = int(_reason_failed_mask(selected_df).sum())

    selected_total = float(selected_df["Purchase Price"].sum()) if not selected_df.empty else 0.0
    candidate_total = float(candidates_df["Purchase Price"].sum()) if not candidates_df.empty else 0.0

    top_customer_share = 0.0
    if not selected_df.empty:
        by_customer = (
            selected_df.groupby("Customer", dropna=False)["Purchase Price"].sum().sort_values(ascending=False)
        )
        top_total = float(by_customer.head(3).sum())
        top_customer_share = (top_total / selected_total * 100.0) if selected_total > 0 else 0.0

    return {
        "baseline_submitted_count": baseline_submitted,
        "optimized_submitted_count": optimized_submitted,
        "rule_excluded_count": int(len(rule_excluded_df)),
        "not_selected_count": int(len(optimization_result.not_selected_df)),
        "baseline_failed_proxy_count": baseline_failed_proxy,
        "optimized_failed_proxy_count": optimized_failed_proxy,
        "proxy_failed_reduction_count": baseline_failed_proxy - optimized_failed_proxy,
        "candidate_total_amount": candidate_total,
        "selected_total_amount": selected_total,
        "selected_amount_ratio_pct": (
            selected_total / candidate_total * 100.0 if candidate_total > 0 else 0.0
        ),
        "top3_customer_concentration_pct": top_customer_share,
        "optimizer_status": optimization_result.status,
        "facility_usage": optimization_result.facility_usage,
        "customer_usage": optimization_result.customer_usage,
        "group_usage": optimization_result.group_usage,
        "lifecycle_profile": lifecycle_profile.to_dict(),
    }


def _amount_column(df: pd.DataFrame) -> str:
    if "Purchase Price" in df.columns:
        return "Purchase Price"
    if "candidate_amount" in df.columns:
        return "candidate_amount"
    return "Purchase Price"


def _top3_customer_share(df: pd.DataFrame, amount_col: str) -> float:
    if df.empty:
        return 0.0
    customer_col = "Customer" if "Customer" in df.columns else "debtor_id"
    if customer_col not in df.columns:
        return 0.0
    by_customer = df.groupby(customer_col, dropna=False)[amount_col].sum().sort_values(ascending=False)
    total = float(df[amount_col].sum())
    if total <= 0:
        return 0.0
    return float(by_customer.head(3).sum()) / total * 100.0


def compute_multi_week_run_metrics(
    candidates_df: pd.DataFrame,
    optimization_result: MultiWeekOptimizationResult,
    rule_excluded_df: pd.DataFrame,
    explained_not_selected_df: pd.DataFrame,
    lifecycle_profile: LifecycleProfile,
) -> Dict[str, object]:
    """Compute run metrics for multi-week planning."""
    amount_col = _amount_column(candidates_df)
    candidate_total = float(candidates_df[amount_col].sum()) if not candidates_df.empty else 0.0

    selected_count = int(len(optimization_result.selected_df))
    not_selected_count = int(len(optimization_result.not_selected_df))

    selected_df = optimization_result.selected_df.copy()
    selected_amount_col = _amount_column(selected_df)
    if selected_amount_col not in selected_df.columns:
        selected_df[selected_amount_col] = 0.0
    else:
        selected_df[selected_amount_col] = pd.to_numeric(
            selected_df[selected_amount_col], errors="coerce"
        ).fillna(0.0)

    # Use the unique selected invoice amounts, not the solver objective which
    # double-counts invoices planned across multiple weeks (attempt_cap > 1).
    selected_total = float(selected_df[selected_amount_col].sum()) if not selected_df.empty else 0.0

    deferred_reasons = (
        explained_not_selected_df["excluded_reason"].value_counts().to_dict()
        if not explained_not_selected_df.empty and "excluded_reason" in explained_not_selected_df.columns
        else {}
    )

    return {
        "planning_mode": "multi_week",
        "baseline_submitted_count": int(len(candidates_df)),
        "optimized_submitted_count": selected_count,
        "rule_excluded_count": int(len(rule_excluded_df)),
        "not_selected_count": not_selected_count,
        "candidate_total_amount": candidate_total,
        "selected_total_amount": selected_total,
        "selected_amount_ratio_pct": (
            selected_total / candidate_total * 100.0 if candidate_total > 0 else 0.0
        ),
        "top3_customer_concentration_pct": _top3_customer_share(
            selected_df, selected_amount_col
        ) if not selected_df.empty else 0.0,
        "optimizer_status": optimization_result.status,
        "weekly_plan_count": int(len(optimization_result.weekly_plan_df)),
        "horizon_weeks": int(len(optimization_result.week_starts)),
        "facility_weekly_usage": optimization_result.facility_weekly_usage,
        "customer_weekly_usage": optimization_result.customer_weekly_usage,
        "group_weekly_usage": optimization_result.group_weekly_usage,
        "deferred_reasons": {str(k): int(v) for k, v in deferred_reasons.items()},
        "lifecycle_profile": lifecycle_profile.to_dict(),
    }


def render_run_summary_markdown(
    cohort_timestamp: str,
    metrics: Dict[str, object],
) -> str:
    lines = [
        f"# Optimizer v0.01 Run Summary ({cohort_timestamp})",
        "",
        "## Selection metrics",
        f"- Baseline submitted candidates (post-rule): {metrics['baseline_submitted_count']}",
        f"- Optimized submitted invoices: {metrics['optimized_submitted_count']}",
        f"- Rule-excluded invoices (filtered before optimizer): {metrics['rule_excluded_count']}",
        f"- Optimizer non-selected invoices (from candidates): {metrics['not_selected_count']}",
        f"- Selected amount: {metrics['selected_total_amount']:.2f}",
        f"- Candidate amount: {metrics['candidate_total_amount']:.2f}",
        f"- Selected amount ratio: {metrics['selected_amount_ratio_pct']:.2f}%",
        f"- Top-3 customer concentration (selected amount): {metrics['top3_customer_concentration_pct']:.2f}%",
        f"- Optimizer status: {metrics['optimizer_status']}",
        "",
        "## Failed-reason proxy",
        f"- Baseline FAILED proxy count: {metrics['baseline_failed_proxy_count']}",
        f"- Optimized FAILED proxy count: {metrics['optimized_failed_proxy_count']}",
        f"- Proxy reduction: {metrics['proxy_failed_reduction_count']}",
        "",
        "## Lifecycle profiling",
        f"- Total profiled rows: {metrics['lifecycle_profile']['total_rows']}",
        f"- Missing credit end %: {metrics['lifecycle_profile']['missing_credit_end_pct']:.2f}%",
        f"- Repurchase %: {metrics['lifecycle_profile']['repurchase_pct']:.2f}%",
    ]
    return "\n".join(lines)


def render_multi_week_run_summary_markdown(
    cohort_timestamp: str,
    metrics: Dict[str, object],
) -> str:
    lines = [
        f"# Optimizer Multi-week Run Summary ({cohort_timestamp})",
        "",
        "## Planning metrics",
        f"- Baseline candidates (post-rule): {metrics['baseline_submitted_count']}",
        f"- Planned invoices: {metrics['optimized_submitted_count']}",
        f"- Weekly plan rows: {metrics['weekly_plan_count']}",
        f"- Rule-excluded invoices (filtered before optimizer): {metrics['rule_excluded_count']}",
        f"- Not selected invoices (from candidates): {metrics['not_selected_count']}",
        f"- Candidate amount: {metrics['candidate_total_amount']:.2f}",
        f"- Planned amount: {metrics['selected_total_amount']:.2f}",
        f"- Planned amount ratio: {metrics['selected_amount_ratio_pct']:.2f}%",
        f"- Top-3 customer concentration: {metrics['top3_customer_concentration_pct']:.2f}%",
        f"- Optimizer status: {metrics['optimizer_status']}",
        f"- Horizon weeks: {metrics['horizon_weeks']}",
        "",
        "## Deferred reasons",
    ]

    deferred = metrics.get("deferred_reasons") or {}
    if deferred:
        for reason, count in deferred.items():
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")

    lines += [
        "",
        "## Lifecycle profiling",
        f"- Total profiled rows: {metrics['lifecycle_profile']['total_rows']}",
        f"- Missing credit end %: {metrics['lifecycle_profile']['missing_credit_end_pct']:.2f}%",
        f"- Repurchase %: {metrics['lifecycle_profile']['repurchase_pct']:.2f}%",
    ]
    return "\n".join(lines)
