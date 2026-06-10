"""
Multi-week deterministic optimizer using OR-Tools CP-SAT.

Decision variable:
  x[i,t] = 1 if invoice i is submitted on planning week t.

Each invoice is selected in at most one week.  The attempt_cap parameter
controls the *scheduling window*: only the first ``attempt_cap`` eligible
weeks (ordered chronologically) are available per invoice.  A low value
pins invoices close to their first eligible week; a high value gives the
solver more flexibility to defer submissions to less congested weeks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from ..model.limits import ResolvedLimits

try:
    from ortools.sat.python import cp_model
except ImportError:  # pragma: no cover
    cp_model = None


@dataclass(frozen=True)
class MultiWeekOptimizerSettings:
    max_time_seconds: int = 60
    random_seed: int = 0
    num_search_workers: int = 1
    horizon_weeks: int = 8
    attempt_cap: int = 1
    default_lifetime_weeks: int = 4


@dataclass(frozen=True)
class MultiWeekOptimizationResult:
    weekly_plan_df: pd.DataFrame
    selected_df: pd.DataFrame
    not_selected_df: pd.DataFrame
    status: str
    objective_amount: float
    week_starts: tuple[pd.Timestamp, ...]
    facility_weekly_usage: Dict[str, Dict[str, Dict[str, float]]]
    customer_weekly_usage: Dict[str, Dict[str, Dict[str, float]]]
    group_weekly_usage: Dict[str, Dict[str, Dict[str, float]]]


def _to_cents(amount: float) -> int:
    return int(round(float(amount) * 100))


def _safe_float(value: object) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _coerce_week_starts(week_starts: List[pd.Timestamp] | tuple[pd.Timestamp, ...]) -> List[pd.Timestamp]:
    return [pd.to_datetime(w).to_period("W-MON").start_time for w in list(week_starts)]


def _amount_series(candidates_df: pd.DataFrame) -> pd.Series:
    if "Purchase Price" in candidates_df.columns:
        return pd.to_numeric(candidates_df["Purchase Price"], errors="coerce").fillna(0.0)
    if "candidate_amount" in candidates_df.columns:
        return pd.to_numeric(candidates_df["candidate_amount"], errors="coerce").fillna(0.0)
    raise ValueError("Candidates must include 'Purchase Price' or 'candidate_amount'")


def _lifetime_weeks_for_row(
    row: pd.Series,
    default_lifetime_weeks: int,
) -> int:
    if "expected_lifetime_weeks" in row and pd.notna(row.get("expected_lifetime_weeks")):
        return max(1, int(row["expected_lifetime_weeks"]))
    if "expected_lifetime_days" in row and pd.notna(row.get("expected_lifetime_days")):
        return max(1, int((float(row["expected_lifetime_days"]) + 6) // 7))
    return max(1, int(default_lifetime_weeks))


def _base_lookup(
    base_weekly_exposure: Dict[pd.Timestamp, Dict[str, Dict[str, float]]] | None,
    week_start: pd.Timestamp,
    entity_type: str,
) -> Dict[str, int]:
    if not base_weekly_exposure:
        return {}
    week_data = base_weekly_exposure.get(week_start, {})
    entities = week_data.get(entity_type, {})
    return {str(k): _to_cents(v) for k, v in entities.items()}


def _collect_base_limit_violations(
    weeks: List[pd.Timestamp],
    limits: ResolvedLimits,
    base_weekly_exposure: Dict[pd.Timestamp, Dict[str, Dict[str, float]]] | None,
) -> List[Dict[str, object]]:
    checks = (
        ("facility", limits.facility_limits_by_company_code),
        ("customer", limits.customer_limits),
        ("group", limits.group_limits),
    )
    violations: List[Dict[str, object]] = []
    for week in weeks:
        week_iso = pd.to_datetime(week).date().isoformat()
        for entity_type, limit_map in checks:
            base_map = _base_lookup(base_weekly_exposure, week, entity_type)
            for entity_id, limit_cents in limit_map.items():
                used_cents = int(base_map.get(entity_id, 0))
                if used_cents <= int(limit_cents):
                    continue
                excess_cents = used_cents - int(limit_cents)
                violations.append(
                    {
                        "week_start": week_iso,
                        "entity_type": entity_type,
                        "entity_id": str(entity_id),
                        "used_cents": used_cents,
                        "limit_cents": int(limit_cents),
                        "excess_cents": excess_cents,
                    }
                )
    violations.sort(key=lambda row: int(row["excess_cents"]), reverse=True)
    return violations


def optimize_multi_week(
    candidates_df: pd.DataFrame,
    limits: ResolvedLimits,
    week_starts: List[pd.Timestamp] | tuple[pd.Timestamp, ...],
    base_weekly_exposure: Dict[pd.Timestamp, Dict[str, Dict[str, float]]] | None = None,
    settings: MultiWeekOptimizerSettings | None = None,
) -> MultiWeekOptimizationResult:
    if cp_model is None:
        raise ImportError("OR-Tools is not installed. Install 'ortools' to run multi-week optimizer.")

    settings = settings or MultiWeekOptimizerSettings()
    weeks = _coerce_week_starts(week_starts)
    if not weeks:
        raise ValueError("week_starts cannot be empty")

    if candidates_df.empty:
        return MultiWeekOptimizationResult(
            weekly_plan_df=pd.DataFrame(),
            selected_df=candidates_df.copy(),
            not_selected_df=candidates_df.copy(),
            status="EMPTY",
            objective_amount=0.0,
            week_starts=tuple(weeks),
            facility_weekly_usage={},
            customer_weekly_usage={},
            group_weekly_usage={},
        )

    data = candidates_df.copy().reset_index(drop=True)
    amounts = _amount_series(data)
    amount_cents = [_to_cents(v) for v in amounts.tolist()]

    if "Company Code" not in data.columns and "company_code" in data.columns:
        data["Company Code"] = data["company_code"]
    if "Customer" not in data.columns and "debtor_id" in data.columns:
        data["Customer"] = data["debtor_id"]

    companies = data["Company Code"].astype(str).tolist()
    customers = data["Customer"].astype(str).tolist()
    groups = [limits.customer_to_group.get(customer, "") for customer in customers]
    due_dates = pd.to_datetime(data.get("Due Date"), errors="coerce")
    if "Offer File Date (UTC)" in data.columns:
        offer_dates = pd.to_datetime(data["Offer File Date (UTC)"], errors="coerce")
    elif "offer_file_date" in data.columns:
        offer_dates = pd.to_datetime(data["offer_file_date"], errors="coerce")
    else:
        offer_dates = pd.Series([pd.NaT] * len(data), index=data.index)
    offer_weeks = offer_dates.dt.to_period("W-MON").dt.start_time
    lifetime_weeks = [
        _lifetime_weeks_for_row(row, settings.default_lifetime_weeks)
        for _, row in data.iterrows()
    ]

    base_violations = _collect_base_limit_violations(weeks, limits, base_weekly_exposure)
    if base_violations:
        top_rows = base_violations[:5]
        details = "; ".join(
            (
                f"{row['week_start']} {row['entity_type']}={row['entity_id']} "
                f"used={float(int(row['used_cents']) / 100.0):.2f} "
                f"limit={float(int(row['limit_cents']) / 100.0):.2f}"
            )
            for row in top_rows
        )
        raise RuntimeError(
            "Multi-week optimizer failed with status INFEASIBLE: "
            f"base exposure exceeds limits for {len(base_violations)} week-entity constraints. "
            f"Top violations: {details}"
        )

    model = cp_model.CpModel()
    x: Dict[Tuple[int, int], "cp_model.IntVar"] = {}

    def allowed_week(i: int, t: int) -> bool:
        due = due_dates.iloc[i] if i < len(due_dates) else pd.NaT
        offer_week = offer_weeks.iloc[i] if i < len(offer_weeks) else pd.NaT
        week = pd.to_datetime(weeks[t])
        if pd.notna(offer_week) and week < offer_week:
            return False
        if pd.notna(due) and week > due:
            return False
        return True

    # Create decision variables for the first attempt_cap eligible weeks per
    # invoice.  This limits the scheduling window: attempt_cap=1 pins each
    # invoice to its first eligible week, higher values allow deferral.
    cap = int(settings.attempt_cap)
    for i in range(len(data)):
        eligible_count = 0
        for t in range(len(weeks)):
            if eligible_count >= cap:
                break
            if allowed_week(i, t):
                x[(i, t)] = model.NewBoolVar(f"x_{i}_{t}")
                eligible_count += 1

    # Each invoice can be selected in at most one week.
    for i in range(len(data)):
        vars_i = [x[(i, t)] for t in range(len(weeks)) if (i, t) in x]
        if vars_i:
            model.Add(sum(vars_i) <= 1)

    # Weekly capacity constraints with carry-over lifetime effect.
    for t, week in enumerate(weeks):
        base_facility = _base_lookup(base_weekly_exposure, week, "facility")
        base_customer = _base_lookup(base_weekly_exposure, week, "customer")
        base_group = _base_lookup(base_weekly_exposure, week, "group")

        for company_code, limit_cents in limits.facility_limits_by_company_code.items():
            expr_terms = []
            for i, company in enumerate(companies):
                if company != company_code:
                    continue
                life = lifetime_weeks[i]
                for tau in range(t + 1):
                    if (i, tau) not in x:
                        continue
                    if t < (tau + life):
                        expr_terms.append(x[(i, tau)] * amount_cents[i])
            base = base_facility.get(company_code, 0)
            model.Add(base + sum(expr_terms) <= limit_cents)

        for customer_id, limit_cents in limits.customer_limits.items():
            expr_terms = []
            for i, customer in enumerate(customers):
                if customer != customer_id:
                    continue
                life = lifetime_weeks[i]
                for tau in range(t + 1):
                    if (i, tau) not in x:
                        continue
                    if t < (tau + life):
                        expr_terms.append(x[(i, tau)] * amount_cents[i])
            base = base_customer.get(customer_id, 0)
            model.Add(base + sum(expr_terms) <= limit_cents)

        for group_id, limit_cents in limits.group_limits.items():
            expr_terms = []
            for i, group in enumerate(groups):
                if group != group_id:
                    continue
                life = lifetime_weeks[i]
                for tau in range(t + 1):
                    if (i, tau) not in x:
                        continue
                    if t < (tau + life):
                        expr_terms.append(x[(i, tau)] * amount_cents[i])
            base = base_group.get(group_id, 0)
            model.Add(base + sum(expr_terms) <= limit_cents)

    # Primary objective: maximize funded amount.
    obj_terms = [x[(i, t)] * amount_cents[i] for (i, t) in x]
    total_amount_expr = sum(obj_terms) if obj_terms else 0
    model.Maximize(total_amount_expr)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = settings.max_time_seconds
    solver.parameters.random_seed = settings.random_seed
    solver.parameters.num_search_workers = settings.num_search_workers

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"Multi-week optimizer failed with status {solver.StatusName(status)}")

    best_amount = int(solver.Value(total_amount_expr)) if obj_terms else 0
    first_solution = {
        key: int(solver.Value(var))
        for key, var in x.items()
    }

    # Secondary deterministic objective: prefer earlier weeks.
    if obj_terms:
        model.Add(total_amount_expr == best_amount)
        accel_terms = []
        for (i, t), var in x.items():
            accel_weight = len(weeks) - t
            accel_terms.append(var * amount_cents[i] * accel_weight)
        model.Maximize(sum(accel_terms))
        status_tie = solver.Solve(model)
        if status_tie in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            solution = {key: int(solver.Value(var)) for key, var in x.items()}
            solve_status = solver.StatusName(status_tie)
        else:
            solution = first_solution
            solve_status = solver.StatusName(status)
    else:
        solution = first_solution
        solve_status = solver.StatusName(status)

    plan_rows: List[Dict[str, object]] = []
    selected_mask = [False] * len(data)
    for (i, t), selected in solution.items():
        if not selected:
            continue
        selected_mask[i] = True
        row = data.iloc[i].to_dict()
        row["planned_week_index"] = int(t + 1)
        row["planned_week_start"] = pd.to_datetime(weeks[t])
        row["planned_week_start_iso"] = pd.to_datetime(weeks[t]).date().isoformat()
        row["expected_lifetime_weeks"] = int(lifetime_weeks[i])
        plan_rows.append(row)

    weekly_plan_df = pd.DataFrame(plan_rows)
    selected_df = data[pd.Series(selected_mask)].copy().reset_index(drop=True)
    not_selected_df = data[~pd.Series(selected_mask)].copy().reset_index(drop=True)

    # Usage breakdown by week and entity.
    def usage_for(entity_values: List[str], limits_dict: Dict[str, int], entity_type: str) -> Dict[str, Dict[str, Dict[str, float]]]:
        output: Dict[str, Dict[str, Dict[str, float]]] = {}
        for t, week in enumerate(weeks):
            week_key = pd.to_datetime(week).date().isoformat()
            base_lookup = _base_lookup(base_weekly_exposure, week, entity_type)
            entity_totals: Dict[str, int] = {}

            for i, entity_id in enumerate(entity_values):
                life = lifetime_weeks[i]
                for tau in range(t + 1):
                    if (i, tau) not in solution or not solution[(i, tau)]:
                        continue
                    if t < (tau + life):
                        entity_totals[entity_id] = entity_totals.get(entity_id, 0) + amount_cents[i]

            week_usage: Dict[str, Dict[str, float]] = {}
            for entity_id in sorted(set(entity_totals.keys()) | set(limits_dict.keys()) | set(base_lookup.keys())):
                used_new = entity_totals.get(entity_id, 0)
                used_base = base_lookup.get(entity_id, 0)
                used_total = used_new + used_base
                limit = limits_dict.get(entity_id, 0)
                utilization = (used_total / limit * 100.0) if limit > 0 else 0.0
                week_usage[entity_id] = {
                    "used_new": used_new / 100.0,
                    "used_base": used_base / 100.0,
                    "used_total": used_total / 100.0,
                    "limit": limit / 100.0,
                    "utilization_pct": utilization,
                }
            output[week_key] = week_usage
        return output

    objective_amount = best_amount / 100.0

    return MultiWeekOptimizationResult(
        weekly_plan_df=weekly_plan_df,
        selected_df=selected_df,
        not_selected_df=not_selected_df,
        status=solve_status,
        objective_amount=objective_amount,
        week_starts=tuple(weeks),
        facility_weekly_usage=usage_for(companies, limits.facility_limits_by_company_code, "facility"),
        customer_weekly_usage=usage_for(customers, limits.customer_limits, "customer"),
        group_weekly_usage=usage_for(groups, limits.group_limits, "group"),
    )
