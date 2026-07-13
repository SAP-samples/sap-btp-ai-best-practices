"""
Weekly base exposure reconstruction for multi-week optimization.

Base exposure captures already-live invoices before new optimization decisions.
It is reconstructed from lifecycle events without joining row content across
different source files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import pandas as pd


@dataclass(frozen=True)
class ExposureReconstructionResult:
    weekly_exposure_df: pd.DataFrame
    week_starts: tuple[pd.Timestamp, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "week_starts": [w.isoformat() for w in self.week_starts],
            "rows": int(len(self.weekly_exposure_df)),
        }


def normalize_week_start(value: pd.Timestamp | str) -> pd.Timestamp:
    ts = pd.to_datetime(value)
    return ts.to_period("W-MON").start_time


def build_week_starts(
    planning_start_date: str | pd.Timestamp,
    horizon_weeks: int,
) -> List[pd.Timestamp]:
    if horizon_weeks <= 0:
        return []
    first = normalize_week_start(pd.to_datetime(planning_start_date))
    return [first + pd.Timedelta(weeks=i) for i in range(horizon_weeks)]


def _safe_float(value: object) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def reconstruct_base_weekly_exposure(
    lifecycle_df: pd.DataFrame,
    week_starts: Iterable[pd.Timestamp],
    customer_to_group: Dict[str, str] | None = None,
    amount_column: str = "purchase_price",
) -> ExposureReconstructionResult:
    """Reconstruct base weekly exposures by facility/customer/group.

    Conservative weekly policy: an invoice counts in week W when:
      credit_start <= W and (credit_release is null or credit_release >= W)
    """
    customer_to_group = customer_to_group or {}
    week_list = [normalize_week_start(w) for w in week_starts]

    if not week_list or lifecycle_df.empty:
        empty = pd.DataFrame(
            columns=["week_start", "entity_type", "entity_id", "exposure_amount"]
        )
        return ExposureReconstructionResult(empty, tuple(week_list))

    data = lifecycle_df.copy()
    if amount_column not in data.columns:
        if "Purchase Price" in data.columns:
            amount_column = "Purchase Price"
        elif "candidate_amount" in data.columns:
            amount_column = "candidate_amount"

    if "company_code" not in data.columns and "Company Code" in data.columns:
        data["company_code"] = data["Company Code"]
    if "debtor_id" not in data.columns and "Customer" in data.columns:
        data["debtor_id"] = data["Customer"]

    data["credit_start"] = pd.to_datetime(data.get("credit_start"), errors="coerce")
    data["credit_release"] = pd.to_datetime(data.get("credit_release"), errors="coerce")

    rows: List[Dict[str, object]] = []
    for week_start in week_list:
        active = data[
            (data["credit_start"].notna())
            & (data["credit_start"] <= week_start)
            & (
                data["credit_release"].isna()
                | (data["credit_release"] >= week_start)
            )
        ]

        facility = active.groupby("company_code", dropna=True)[amount_column].sum()
        for entity_id, amount in facility.items():
            rows.append(
                {
                    "week_start": week_start,
                    "entity_type": "facility",
                    "entity_id": str(entity_id),
                    "exposure_amount": _safe_float(amount),
                }
            )

        customer = active.groupby("debtor_id", dropna=True)[amount_column].sum()
        for entity_id, amount in customer.items():
            rows.append(
                {
                    "week_start": week_start,
                    "entity_type": "customer",
                    "entity_id": str(entity_id),
                    "exposure_amount": _safe_float(amount),
                }
            )

        if customer_to_group:
            with_groups = active.copy()
            with_groups["group_id"] = with_groups["debtor_id"].astype(str).map(customer_to_group)
            group = with_groups.groupby("group_id", dropna=True)[amount_column].sum()
            for entity_id, amount in group.items():
                rows.append(
                    {
                        "week_start": week_start,
                        "entity_type": "group",
                        "entity_id": str(entity_id),
                        "exposure_amount": _safe_float(amount),
                    }
                )

    exposure_df = pd.DataFrame(rows)
    if exposure_df.empty:
        exposure_df = pd.DataFrame(
            columns=["week_start", "entity_type", "entity_id", "exposure_amount"]
        )

    return ExposureReconstructionResult(
        weekly_exposure_df=exposure_df,
        week_starts=tuple(week_list),
    )


def weekly_exposure_lookup(
    weekly_exposure_df: pd.DataFrame,
    entity_type: str,
) -> Dict[pd.Timestamp, Dict[str, float]]:
    """Convert exposure DataFrame into week->entity->amount lookup."""
    if weekly_exposure_df.empty:
        return {}

    data = weekly_exposure_df[weekly_exposure_df["entity_type"] == entity_type]
    output: Dict[pd.Timestamp, Dict[str, float]] = {}
    for week, week_df in data.groupby("week_start"):
        output[pd.to_datetime(week)] = {
            str(row["entity_id"]): _safe_float(row["exposure_amount"])
            for _, row in week_df.iterrows()
        }
    return output
