"""
Deterministic explanations for invoices not scheduled in multi-week planning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from ..model.limits import ResolvedLimits
from .optimizer_multi_week import MultiWeekOptimizationResult


@dataclass(frozen=True)
class MultiWeekExplanationResult:
    explained_not_selected_df: pd.DataFrame


def _default_lifetime(row: pd.Series) -> int:
    if "expected_lifetime_weeks" in row and pd.notna(row.get("expected_lifetime_weeks")):
        return max(1, int(row["expected_lifetime_weeks"]))
    if "expected_lifetime_days" in row and pd.notna(row.get("expected_lifetime_days")):
        return max(1, int((float(row["expected_lifetime_days"]) + 6) // 7))
    return 4


def explain_multi_week_non_selection(
    candidates_df: pd.DataFrame,
    optimization_result: MultiWeekOptimizationResult,
    limits: ResolvedLimits,
) -> MultiWeekExplanationResult:
    """Assign reason codes to invoices not scheduled in any planning week."""
    not_selected = optimization_result.not_selected_df.copy()
    if not not_selected.empty:
        data = not_selected
    else:
        data = pd.DataFrame(columns=[*candidates_df.columns, "excluded_reason"])

    if data.empty:
        return MultiWeekExplanationResult(explained_not_selected_df=data)

    week_keys = [w.date().isoformat() for w in optimization_result.week_starts]

    reasons: List[str] = []
    details: List[str] = []

    for _, row in data.iterrows():
        due = pd.to_datetime(row.get("Due Date"), errors="coerce")
        offer = pd.to_datetime(
            row.get("Offer File Date (UTC)", row.get("offer_file_date")),
            errors="coerce",
        )
        offer_week = offer.to_period("W-MON").start_time if pd.notna(offer) else pd.NaT
        company = str(row.get("Company Code", row.get("company_code", "")))
        customer = str(row.get("Customer", row.get("debtor_id", "")))
        group = limits.customer_to_group.get(customer, "")
        amount = float(row.get("Purchase Price", row.get("candidate_amount", 0.0)) or 0.0)
        lifetime = _default_lifetime(row)

        allowed_weeks = []
        for idx, week in enumerate(optimization_result.week_starts):
            if pd.notna(offer_week) and week < offer_week:
                continue
            if pd.isna(due) or week <= due:
                allowed_weeks.append(idx)

        if not allowed_weeks:
            reasons.append("EXPIRED_WINDOW")
            details.append("Invoice due date is before all planning weeks.")
            continue

        any_feasible_week = False
        violations = {"FACILITY_CAP_BINDING": 0, "CUSTOMER_CAP_BINDING": 0, "GROUP_CAP_BINDING": 0}

        for start_idx in allowed_weeks:
            feasible = True
            for t in range(start_idx, min(len(week_keys), start_idx + lifetime)):
                wk = week_keys[t]

                facility_usage = optimization_result.facility_weekly_usage.get(wk, {}).get(company)
                if facility_usage:
                    if facility_usage["used_total"] + amount > facility_usage["limit"] + 1e-9:
                        feasible = False
                        violations["FACILITY_CAP_BINDING"] += 1
                        break

                customer_usage = optimization_result.customer_weekly_usage.get(wk, {}).get(customer)
                if customer_usage:
                    if customer_usage["used_total"] + amount > customer_usage["limit"] + 1e-9:
                        feasible = False
                        violations["CUSTOMER_CAP_BINDING"] += 1
                        break

                if group:
                    group_usage = optimization_result.group_weekly_usage.get(wk, {}).get(group)
                    if group_usage and group_usage["used_total"] + amount > group_usage["limit"] + 1e-9:
                        feasible = False
                        violations["GROUP_CAP_BINDING"] += 1
                        break

            if feasible:
                any_feasible_week = True
                break

        if any_feasible_week:
            reasons.append("DEFERRED_FOR_CAPACITY")
            details.append("At least one feasible week exists, but optimizer selected alternatives with better objective value.")
            continue

        # Select dominant binding reason.
        dominant = max(violations, key=violations.get)
        reasons.append(dominant)
        details.append(
            f"Violations by type: facility={violations['FACILITY_CAP_BINDING']}, "
            f"customer={violations['CUSTOMER_CAP_BINDING']}, "
            f"group={violations['GROUP_CAP_BINDING']}"
        )

    data["excluded_reason"] = reasons
    data["excluded_reason_detail"] = details
    data["excluded_stage"] = "optimizer"

    return MultiWeekExplanationResult(explained_not_selected_df=data)
