"""
Deterministic explanations for invoices not selected by the optimizer.

This module assigns a human-readable reason to each non-selected invoice by
performing a **greedy single-invoice headroom probe**: for each non-selected
invoice, it checks whether adding that single invoice to the already-selected
set would violate any facility, customer, or group limit.

IMPORTANT LIMITATION: This is NOT a combinatorial explanation. Two invoices
that are individually feasible but jointly infeasible will both receive
"not chosen (budget used elsewhere)" because neither one alone violates a
limit. The explanation identifies the tightest binding constraint when one
exists, but falls back to a generic reason when the exclusion is driven by
the optimizer's combinatorial trade-off across multiple invoices.

The check order (facility -> customer -> group -> fallback) determines which
reason is reported when multiple limits are simultaneously binding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from ..model.limits import ResolvedLimits


@dataclass(frozen=True)
class ExplanationResult:
    """Container for the explained non-selected invoices DataFrame."""
    explained_not_selected_df: pd.DataFrame


def _amount_to_cents(value: float) -> int:
    """Convert a monetary float to integer cents (same conversion as the solver)."""
    return int(round(float(value) * 100))


def explain_non_selection(
    selected_df: pd.DataFrame,
    not_selected_df: pd.DataFrame,
    limits: ResolvedLimits,
) -> ExplanationResult:
    """Assign a deterministic exclusion reason to each non-selected invoice.

    The method reconstructs the headroom consumed by the selected set, then
    for each non-selected invoice checks (in priority order):
      1. Would adding it exceed the facility limit for its Company Code?
      2. Would adding it exceed the customer limit for its Customer?
      3. Would adding it exceed the group limit for its customer group?
      4. If none of the above, it was excluded by the optimizer's combinatorial
         choice ("budget used elsewhere").

    Each invoice receives exactly one reason (the first applicable).

    Args:
        selected_df: Invoices selected by the optimizer (x_i = 1).
        not_selected_df: Invoices not selected by the optimizer (x_i = 0).
        limits: Resolved limits in integer cents.

    Returns:
        ExplanationResult with ``excluded_reason`` and ``excluded_reason_detail``
        columns appended to the not-selected DataFrame.
    """
    if not not_selected_df.empty:
        data = not_selected_df.copy()
    else:
        data = pd.DataFrame(columns=[*not_selected_df.columns, "excluded_reason"])

    # Reconstruct the headroom consumed by selected invoices.
    facility_used: Dict[str, int] = {}
    customer_used: Dict[str, int] = {}
    group_used: Dict[str, int] = {}

    for _, row in selected_df.iterrows():
        amount = _amount_to_cents(row["Purchase Price"])
        company = str(row["Company Code"])
        customer = str(row["Customer"])
        group = limits.customer_to_group.get(customer, "")

        facility_used[company] = facility_used.get(company, 0) + amount
        customer_used[customer] = customer_used.get(customer, 0) + amount
        if group:
            group_used[group] = group_used.get(group, 0) + amount

    # Probe each non-selected invoice against remaining headroom.
    reasons = []
    details = []
    for _, row in data.iterrows():
        amount = _amount_to_cents(row["Purchase Price"])
        company = str(row["Company Code"])
        customer = str(row["Customer"])
        group = limits.customer_to_group.get(customer, "")

        facility_limit = limits.facility_limits_by_company_code.get(company)
        customer_limit = limits.customer_limits.get(customer)
        group_limit = limits.group_limits.get(group) if group else None

        # Check facility headroom first (broadest constraint).
        if facility_limit is not None and facility_used.get(company, 0) + amount > facility_limit:
            reasons.append("exceeded facility headroom")
            details.append(
                f"company={company}, needed={amount/100:.2f}, "
                f"used={facility_used.get(company, 0)/100:.2f}, limit={facility_limit/100:.2f}"
            )
            continue

        # Check customer headroom (concentration constraint).
        if customer_limit is not None and customer_used.get(customer, 0) + amount > customer_limit:
            reasons.append("exceeded customer headroom")
            details.append(
                f"customer={customer}, needed={amount/100:.2f}, "
                f"used={customer_used.get(customer, 0)/100:.2f}, limit={customer_limit/100:.2f}"
            )
            continue

        # Check group headroom (optional group-level constraint).
        if (
            group
            and group_limit is not None
            and group_used.get(group, 0) + amount > group_limit
        ):
            reasons.append("exceeded group headroom")
            details.append(
                f"group={group}, needed={amount/100:.2f}, "
                f"used={group_used.get(group, 0)/100:.2f}, limit={group_limit/100:.2f}"
            )
            continue

        # Fallback: no single limit is violated; the optimizer made a
        # combinatorial choice to allocate capacity elsewhere.
        reasons.append("not chosen (budget used elsewhere)")
        details.append("no single limit violation; selected set used capacity more efficiently")

    if not data.empty:
        data["excluded_reason"] = reasons
        data["excluded_reason_detail"] = details

    return ExplanationResult(explained_not_selected_df=data)
