"""
Lifecycle derivation and profiling for invoice credit windows.

In the BTP factoring flow, each funded invoice consumes credit from the moment
it is funded (credit start) until the credit is released (credit end). The
duration of this "alive window" determines how long an invoice occupies capacity.

This module derives:
  - ``credit_start``: when the invoice starts consuming credit (Summary File Date).
  - ``credit_end``: when the credit is released, based on a configurable release event.
  - ``credit_duration_days``: the number of days the invoice was "alive".

The release event is configurable via feature flag:
  - ``reconciliation_file_date``: credit releases when reconciliation file is processed.
  - ``paid_on``: legacy mode, credit releases when the buyer pays (Paid On date).
  - ``reconciliation``: legacy mode, credit releases at reconciliation date.
  - ``min_paid_or_repurchase``: legacy sensitivity mode.

In v0.01 this is used for profiling only (understanding the distribution of alive
windows). In the future multi-week optimizer, the alive window will be a core
constraint: invoices that are still alive consume capacity in subsequent weeks.

NOTE: For recent/future cohorts, most invoices will not yet have a credit end
date, resulting in high ``missing_credit_end_pct``. The profiling is most
informative when run on historical data across multiple cohorts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

# Column name constants matching the extraction schema.
CREDIT_START_COLUMN = "Summary File Date (UTC)"
PAID_ON_COLUMN = "Paid On (Europe, Madrid)"
RECONCILIATION_COLUMN = "Reconciliation Date"
RECONCILIATION_FILE_COLUMN = "Reconciliation File Date (UTC)"
REPURCHASE_DATE_COLUMN = "Repurchase Date"


@dataclass(frozen=True)
class LifecycleProfile:
    """Aggregate statistics about credit lifecycle durations in a dataset."""
    total_rows: int
    rows_with_credit_start: int
    rows_with_credit_end: int
    missing_credit_end_pct: float        # % of rows without a resolved credit end date
    repurchase_pct: float                 # % of rows with a non-zero Repurchase value
    duration_days_stats: Dict[str, float] # count, mean, median, p10, p90 of alive duration

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_rows": self.total_rows,
            "rows_with_credit_start": self.rows_with_credit_start,
            "rows_with_credit_end": self.rows_with_credit_end,
            "missing_credit_end_pct": self.missing_credit_end_pct,
            "repurchase_pct": self.repurchase_pct,
            "duration_days_stats": self.duration_days_stats,
        }


def derive_lifecycle(
    df: pd.DataFrame,
    release_event: str = "reconciliation_file_date",
) -> pd.DataFrame:
    """Attach credit_start, credit_end and credit_duration_days columns.

    Args:
        df: DataFrame with extraction date columns already parsed.
        release_event: One of:
            - 'reconciliation_file_date' (default)
            - 'paid_on'
            - 'reconciliation'
            - 'min_paid_or_repurchase'

    Returns:
        Copy of df with three new columns: credit_start, credit_end, credit_duration_days.
    """
    out = df.copy()
    if "credit_start" in out.columns:
        out["credit_start"] = pd.to_datetime(out["credit_start"], errors="coerce")
    else:
        out["credit_start"] = pd.to_datetime(out.get(CREDIT_START_COLUMN), errors="coerce")

    paid_on = pd.to_datetime(out.get(PAID_ON_COLUMN), errors="coerce")
    reconciliation = pd.to_datetime(out.get(RECONCILIATION_COLUMN), errors="coerce")
    reconciliation_file = pd.to_datetime(out.get(RECONCILIATION_FILE_COLUMN), errors="coerce")
    repurchase = pd.to_datetime(out.get(REPURCHASE_DATE_COLUMN), errors="coerce")

    if release_event == "reconciliation_file_date":
        credit_end = reconciliation_file
    elif release_event == "paid_on":
        credit_end = paid_on
    elif release_event == "reconciliation":
        credit_end = reconciliation
    elif release_event == "min_paid_or_repurchase":
        both = pd.concat([paid_on, repurchase], axis=1)
        credit_end = both.min(axis=1)
    else:
        raise ValueError(
            "release_event must be one of: "
            "reconciliation_file_date, paid_on, reconciliation, min_paid_or_repurchase"
        )

    out["credit_end"] = credit_end
    out["credit_release"] = out["credit_end"]
    out["credit_duration_days"] = (out["credit_end"] - out["credit_start"]).dt.days
    return out


def profile_lifecycle(df_with_lifecycle: pd.DataFrame) -> LifecycleProfile:
    """Compute aggregate lifecycle statistics from a DataFrame with lifecycle columns."""
    total = len(df_with_lifecycle)
    rows_with_credit_start = int(df_with_lifecycle["credit_start"].notna().sum())
    rows_with_credit_end = int(df_with_lifecycle["credit_end"].notna().sum())

    repurchase = pd.to_numeric(df_with_lifecycle.get("Repurchase"), errors="coerce")
    repurchase_pct = float((repurchase.fillna(0) > 0).mean() * 100) if total else 0.0

    duration = pd.to_numeric(df_with_lifecycle["credit_duration_days"], errors="coerce").dropna()
    if duration.empty:
        stats = {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p90": 0.0,
        }
    else:
        stats = {
            "count": int(duration.size),
            "mean": float(duration.mean()),
            "median": float(duration.median()),
            "p10": float(duration.quantile(0.10)),
            "p90": float(duration.quantile(0.90)),
        }

    missing_credit_end_pct = float(((total - rows_with_credit_end) / total) * 100) if total else 0.0

    return LifecycleProfile(
        total_rows=total,
        rows_with_credit_start=rows_with_credit_start,
        rows_with_credit_end=rows_with_credit_end,
        missing_credit_end_pct=missing_credit_end_pct,
        repurchase_pct=repurchase_pct,
        duration_days_stats=stats,
    )
