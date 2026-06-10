"""
Single-week deterministic optimizer using OR-Tools CP-SAT.

This module solves a multi-constraint binary knapsack problem: given a set of
candidate invoices from one cohort week, select the subset that maximizes total
funded amount (Purchase Price) while respecting three tiers of credit limits:

  1. Facility limit per Company Code (aggregate cap per legal entity)
  2. Customer limit per Customer ID (concentration cap per buyer)
  3. Group limit per customer group (optional, aggregate cap across related buyers)

All monetary arithmetic inside the solver uses integer cents to avoid
floating-point imprecision in the CP-SAT integer programming model.

A deterministic tie-break second pass ensures that when multiple optimal
solutions exist, the solver consistently picks the same one (preferring
earlier-sorted rows by a stable sort key).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from ..model.limits import ResolvedLimits

try:
    from ortools.sat.python import cp_model
except ImportError:  # pragma: no cover - tested via skip behavior
    cp_model = None


@dataclass(frozen=True)
class OptimizerSettings:
    """Solver tuning parameters passed to CP-SAT."""
    max_time_seconds: int = 60
    random_seed: int = 0
    # Keep at 1 for determinism; increase for faster solve at the cost of reproducibility.
    num_search_workers: int = 1


@dataclass(frozen=True)
class OptimizationResult:
    """Immutable container for a single-week optimization run output."""
    selected_df: pd.DataFrame          # Invoices chosen for the offer (x_i = 1)
    not_selected_df: pd.DataFrame      # Invoices not chosen (x_i = 0)
    objective_amount: float             # Total Purchase Price of selected invoices
    status: str                         # CP-SAT solver status name (OPTIMAL / FEASIBLE)
    facility_usage: Dict[str, Dict[str, float]]   # Per-company: used / limit / utilization %
    customer_usage: Dict[str, Dict[str, float]]    # Per-customer: used / limit / utilization %
    group_usage: Dict[str, Dict[str, float]]       # Per-group: used / limit / utilization %


def _to_cents(series: pd.Series) -> List[int]:
    """Convert a pandas Series of monetary floats to integer cents.

    This avoids floating-point representation issues inside the CP-SAT model,
    which only supports integer coefficients and constraints.
    """
    return [int(round(float(v) * 100)) for v in series.tolist()]


def _sort_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Sort candidates by a stable composite key for deterministic solver input.

    The sort order matters for the tie-break pass: when multiple optimal solutions
    exist, the solver will prefer earlier rows in this ordering.
    """
    sort_columns = [
        "Company Code",
        "Customer",
        "Invoice Reference",
        "Document Number",
        "optimizer_row_id",
    ]
    safe = df.copy()
    for col in sort_columns:
        if col not in safe.columns:
            safe[col] = ""
    return safe.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def _build_usage_dict(
    used_new: Dict[str, int],
    limits: Dict[str, int],
    used_base: Dict[str, int] | None = None,
) -> Dict[str, Dict[str, float]]:
    """Build a human-readable usage summary (in money units, not cents).

    For each entity (company / customer / group), report the amount used by
    selected invoices, the configured limit, and the utilization percentage.
    """
    output: Dict[str, Dict[str, float]] = {}
    base = used_base or {}
    keys = set(used_new.keys()) | set(limits.keys()) | set(base.keys())
    for key in sorted(keys):
        limit = limits.get(key, 0)
        new_value = used_new.get(key, 0)
        base_value = base.get(key, 0)
        total_value = new_value + base_value
        utilization = (total_value / limit * 100.0) if limit > 0 else 0.0
        output[key] = {
            # Backward-compatible key.
            "used": total_value / 100.0,
            "used_new": new_value / 100.0,
            "used_base": base_value / 100.0,
            "used_total": total_value / 100.0,
            "limit": limit / 100.0,
            "utilization_pct": utilization,
        }
    return output


def optimize_single_week(
    candidates_df: pd.DataFrame,
    limits: ResolvedLimits,
    settings: OptimizerSettings | None = None,
) -> OptimizationResult:
    """Solve a single-week invoice selection problem.

    The formulation is a multi-constraint binary knapsack:

        max  sum(x_i * amount_i)            -- maximize total funded amount
        s.t. sum(x_i * amount_i | company_code_i == cc) <= facility_limit[cc]   for each cc
             sum(x_i * amount_i | customer_i == cust)   <= customer_limit[cust]  for each cust
             sum(x_i * amount_i | group_i == grp)       <= group_limit[grp]      for each grp
             x_i in {0, 1}

    After the first solve, a tie-break pass fixes the optimal objective value
    and re-solves to prefer earlier-sorted rows (deterministic selection).

    NOTE: The tie-break pass is only meaningful when the first solve returns
    OPTIMAL. When FEASIBLE (time limit hit), the tie-break pass pins to the
    best-found (possibly suboptimal) amount. This is acceptable for v0.01 but
    should be revisited if FEASIBLE status is observed on production data.

    Args:
        candidates_df: Eligible invoices for this cohort (post-rule-filtering).
        limits: Resolved facility / customer / group limits in integer cents.
        settings: Solver parameters (time limit, seed, workers).

    Returns:
        OptimizationResult with selected/not-selected DataFrames and usage stats.
    """
    if cp_model is None:
        raise ImportError(
            "OR-Tools is not installed. Install 'ortools' to run optimizer_single_week."
        )

    settings = settings or OptimizerSettings()

    if candidates_df.empty:
        return OptimizationResult(
            selected_df=candidates_df.copy(),
            not_selected_df=candidates_df.copy(),
            objective_amount=0.0,
            status="EMPTY",
            facility_usage={},
            customer_usage={},
            group_usage={},
        )

    # Sort for determinism: the row order affects the tie-break pass.
    data = _sort_candidates(candidates_df)
    amount_cents = _to_cents(data["Purchase Price"])

    # Pre-compute entity membership lists for constraint building.
    companies = data["Company Code"].astype(str).tolist()
    customers = data["Customer"].astype(str).tolist()
    groups = [limits.customer_to_group.get(customer, "") for customer in customers]
    base_facility = dict(limits.base_exposure_facility)
    base_customer = dict(limits.base_exposure_customer)
    base_group = dict(limits.base_exposure_group)

    base_violations: list[str] = []
    for company_code, limit in limits.facility_limits_by_company_code.items():
        base = base_facility.get(company_code, 0)
        if base > limit:
            base_violations.append(
                f"facility {company_code}: base={base / 100.0:.2f} > limit={limit / 100.0:.2f}"
            )
    for customer_id, limit in limits.customer_limits.items():
        base = base_customer.get(customer_id, 0)
        if base > limit:
            base_violations.append(
                f"customer {customer_id}: base={base / 100.0:.2f} > limit={limit / 100.0:.2f}"
            )
    for group_id, limit in limits.group_limits.items():
        base = base_group.get(group_id, 0)
        if base > limit:
            base_violations.append(
                f"group {group_id}: base={base / 100.0:.2f} > limit={limit / 100.0:.2f}"
            )
    if base_violations:
        details = "; ".join(base_violations[:5])
        raise RuntimeError(
            "Single-week optimizer failed before solve: base exposure exceeds limits. "
            f"Details: {details}"
        )

    # --- Build the CP-SAT model ---
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x_{i}") for i in range(len(data))]

    # Constraint tier 1: facility limit per Company Code.
    # NOTE: only company codes present in the limits dict are constrained.
    # Invoices belonging to an unlisted company code face no facility cap.
    for company_code, limit in limits.facility_limits_by_company_code.items():
        idx = [i for i, cc in enumerate(companies) if cc == company_code]
        if idx:
            model.Add(base_facility.get(company_code, 0) + sum(x[i] * amount_cents[i] for i in idx) <= limit)

    # Constraint tier 2: customer concentration limit.
    for customer_id, limit in limits.customer_limits.items():
        idx = [i for i, cust in enumerate(customers) if cust == customer_id]
        if idx:
            model.Add(base_customer.get(customer_id, 0) + sum(x[i] * amount_cents[i] for i in idx) <= limit)

    # Constraint tier 3: group limit (optional; only active when customer_to_group is populated).
    for group_id, limit in limits.group_limits.items():
        idx = [i for i, grp in enumerate(groups) if grp == group_id]
        if idx:
            model.Add(base_group.get(group_id, 0) + sum(x[i] * amount_cents[i] for i in idx) <= limit)

    # Primary objective: maximize total selected Purchase Price (in cents).
    total_amount_expr = sum(x[i] * amount_cents[i] for i in range(len(data)))
    model.Maximize(total_amount_expr)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = settings.max_time_seconds
    solver.parameters.random_seed = settings.random_seed
    solver.parameters.num_search_workers = settings.num_search_workers

    # --- First solve: find maximum total amount ---
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"Optimizer failed with status {solver.StatusName(status)}")

    # --- Deterministic tie-break pass ---
    # Fix the total amount to the best found value, then maximize a secondary
    # objective that prefers earlier-sorted rows (higher weight for lower index).
    # In CP-SAT, the last Maximize() call replaces the previous objective.
    best_amount = int(solver.Value(total_amount_expr))
    first_pass_selection = [int(solver.Value(var)) for var in x]

    model.Add(total_amount_expr == best_amount)
    tie_expr = sum(x[i] * (len(data) - i) for i in range(len(data)))
    model.Maximize(tie_expr)

    status_tie = solver.Solve(model)
    if status_tie in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        selected_mask = [bool(solver.Value(var)) for var in x]
        solve_status = solver.StatusName(status_tie)
    else:
        # Tie-break failed (unlikely); fall back to first-pass selection.
        selected_mask = [bool(v) for v in first_pass_selection]
        solve_status = solver.StatusName(status)

    # --- Partition results and compute usage statistics ---
    data["selected"] = selected_mask
    selected_df = data[data["selected"]].copy().reset_index(drop=True)
    not_selected_df = data[~data["selected"]].copy().reset_index(drop=True)

    # Recompute usage from selected rows (in cents) for the output report.
    facility_used: Dict[str, int] = {}
    customer_used: Dict[str, int] = {}
    group_used: Dict[str, int] = {}

    for _, row in selected_df.iterrows():
        amount = int(round(float(row["Purchase Price"]) * 100))
        company = str(row["Company Code"])
        customer = str(row["Customer"])
        group = limits.customer_to_group.get(customer, "")

        facility_used[company] = facility_used.get(company, 0) + amount
        customer_used[customer] = customer_used.get(customer, 0) + amount
        if group:
            group_used[group] = group_used.get(group, 0) + amount

    objective_amount = sum(amount_cents[i] for i, selected in enumerate(selected_mask) if selected)

    return OptimizationResult(
        selected_df=selected_df.drop(columns=["selected"]),
        not_selected_df=not_selected_df.drop(columns=["selected"]),
        objective_amount=objective_amount / 100.0,
        status=solve_status,
        facility_usage=_build_usage_dict(facility_used, limits.facility_limits_by_company_code, base_facility),
        customer_usage=_build_usage_dict(customer_used, limits.customer_limits, base_customer),
        group_usage=_build_usage_dict(group_used, limits.group_limits, base_group),
    )
