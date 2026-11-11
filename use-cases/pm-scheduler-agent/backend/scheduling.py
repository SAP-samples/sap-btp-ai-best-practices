"""Scheduling engine — pure Python, no UI dependencies."""
from datetime import date, timedelta
import pandas as pd
from .data import row_to_dict

SCHEDULABLE_PRIORITIES = {"Medium", "Low"}   # P3, P4
CRITICALITY_RANK = {"A": 1, "B": 2, "C": 3}


def score_order(row) -> float:
    """Lower = higher scheduling priority.
    Weights: order-type urgency 40% | criticality 30% | deadline 30%.
    """
    order_type_rank = 1 if str(row.get("ORDER_TYPE_CODE", "")) == "MN03" else 2
    c = CRITICALITY_RANK.get(str(row.get("EQUIPMENT_CRITICALITY", "B")), 2)
    deadline = row.get("LATEST_EXECTN_FINISH_DATE")
    days_to_deadline = 9999
    if pd.notna(deadline):
        days_to_deadline = max((pd.Timestamp(deadline) - pd.Timestamp.today()).days, 0)
    urgency = 1 if days_to_deadline <= 7 else (2 if days_to_deadline <= 30 else 3)
    return order_type_rank * 0.4 + c * 0.3 + urgency * 0.3


def _is_candidate(row, ws: pd.Timestamp, we: pd.Timestamp) -> bool:
    order_type = str(row.get("ORDER_TYPE_CODE", ""))
    start = row.get("BASIC_START_DATE")
    finish = row.get("BASIC_FINISH_DATE")
    if order_type == "MN03":
        return pd.notna(start) and (ws <= pd.Timestamp(start) <= we)
    return (pd.notna(start) and pd.notna(finish) and
            pd.Timestamp(start) <= we and pd.Timestamp(finish) >= ws)


def build_weekly_schedule(
    orders: pd.DataFrame,
    cap: pd.DataFrame,
    week_start: date,
    work_centers: list,
    plant: str,
    priority_filter: list | None = None,
    released_only: bool = True,
) -> dict:
    """Greedy weekly scheduler. Returns dict keyed by work center."""
    if priority_filter is None:
        priority_filter = list(SCHEDULABLE_PRIORITIES)

    week_end = week_start + timedelta(days=6)
    ws, we = pd.Timestamp(week_start), pd.Timestamp(week_end)

    # Per-day capacity: {wc: {date: hours}}
    day_cap_df = (
        cap[
            (cap["DATE"] >= ws) & (cap["DATE"] <= we) &
            (cap["OPER_WORK_CENTER"].isin(work_centers))
        ]
        .groupby(["OPER_WORK_CENTER", "DATE"])["CAPACITY"]
        .sum()
    )
    daily_cap: dict = {wc: {} for wc in work_centers}
    for (wc_key, dt), cap_val in day_cap_df.items():
        daily_cap[wc_key][dt.date()] = float(cap_val)

    weekly_cap = {wc: sum(daily_cap[wc].values()) for wc in work_centers}

    filtered = orders[orders["PLANT_NAME"] == plant].copy()
    if released_only:
        filtered = filtered[filtered["WO_PHASE"] == "Released"]
    filtered = filtered[
        filtered["PRIORITY"].isin(priority_filter) &
        filtered["OPER_WORK_CENTER"].isin(work_centers)
    ]

    mask = filtered.apply(lambda r: _is_candidate(r, ws, we), axis=1)
    candidates = filtered[mask].copy()
    candidates["_score"] = candidates.apply(score_order, axis=1)
    candidates = candidates.sort_values("_score")

    schedule: dict = {
        wc: {"scheduled": [], "unscheduled": [], "capacity_used": 0.0,
             "capacity_available": weekly_cap.get(wc, 0.0)}
        for wc in work_centers
    }

    for _, row in candidates.iterrows():
        wc = row["OPER_WORK_CENTER"]
        if wc not in schedule:
            continue
        work_h = (row["ACTIVITY_WORK_INVOLVE"]
                  if row["ACTIVITY_WORK_INVOLVE"] > 0
                  else row["ACTIVITY_NORMAL_DURATION"])
        avail_total = schedule[wc]["capacity_available"] - schedule[wc]["capacity_used"]
        if work_h > avail_total:
            schedule[wc]["unscheduled"].append(row)
            continue

        # Find the first working day with enough remaining capacity
        days = sorted(daily_cap[wc].keys())
        assigned_day = None
        for d in days:
            if work_h <= daily_cap[wc][d]:
                assigned_day = d
                break
        # Multi-day job: assign to first day that still has any capacity
        if assigned_day is None:
            for d in days:
                if daily_cap[wc][d] > 0:
                    assigned_day = d
                    break
        if assigned_day is None:
            assigned_day = days[0] if days else week_start

        row_d = row_to_dict(row)
        row_d["SCHED_DATE"] = assigned_day.isoformat()
        schedule[wc]["scheduled"].append(row_d)
        schedule[wc]["capacity_used"] += work_h
        daily_cap[wc][assigned_day] = max(0.0, daily_cap[wc][assigned_day] - work_h)

    return schedule


def find_opportunity_orders(
    orders: pd.DataFrame,
    equipment_no: str,
    week_start: date,
    plant: str,
) -> list[dict]:
    """Future-week RTS P3/P4 orders on the same equipment."""
    week_end = pd.Timestamp(week_start + timedelta(days=6))

    mask = (
        (orders["EQUIPMENT_NO"].astype(str).str.split(".").str[0] ==
         str(equipment_no).split(".")[0]) &
        (orders["PLANT_NAME"] == plant) &
        (orders["PRIORITY"].isin(SCHEDULABLE_PRIORITIES)) &
        (orders["ORDER_SUBPHASE"].str.contains("Ready to Schedule", na=False)) &
        (orders["BASIC_START_DATE"] > week_end)
    )
    result = orders[mask].copy().sort_values("BASIC_START_DATE")
    if not result.empty:
        result["WEEK_BUCKET"] = result["BASIC_START_DATE"].dt.to_period("W").astype(str)
    return result


def schedule_to_json(schedule: dict) -> list[dict]:
    """Convert schedule dict to JSON-serialisable structure."""
    out = []
    for wc, data in schedule.items():
        out.append({
            "work_center": wc,
            "capacity_available": round(data["capacity_available"], 2),
            "capacity_used": round(data["capacity_used"], 2),
            "load_pct": round(
                data["capacity_used"] / data["capacity_available"] * 100
                if data["capacity_available"] > 0 else 0, 1
            ),
            "scheduled": [row_to_dict(r) for r in data["scheduled"]],
            "unscheduled": [row_to_dict(r) for r in data["unscheduled"]],
        })
    return out
