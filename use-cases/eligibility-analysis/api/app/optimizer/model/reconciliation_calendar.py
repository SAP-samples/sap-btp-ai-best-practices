"""Weekly reconciliation calendar used to turn a predicted file offset into a lifetime.

Operating cycle (receivables programme):
  Tuesday   offer file sent
  Wednesday summary file: invoices purchased (funding date)
  Tuesday   reconciliation file: reconciled invoices release credit

Every observed release falls on a Tuesday reconciliation file, so an invoice's release
is described by one small integer:

  recon_k    = number of weekly files between the first Tuesday on or after the due
               date and the file that released the invoice (0 = that first Tuesday)
  release_at = first Tuesday on or after the due date + 7 * recon_k days

RPT-1 predicts recon_k; this module holds the deterministic calendar arithmetic and the
customer statistics derived from history. All functions accept pandas Series and return
tz-naive timestamps.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

RECONCILIATION_WEEKDAY = 1  # Tuesday (Monday = 0)
FUNDING_WEEKDAY_OFFSET = 2  # Wednesday = Monday planning week start + 2 days


def to_naive(values) -> pd.Series:
    """Parse dates/timestamps, convert aware values to UTC and drop the timezone.

    Args:
        values: Series (or list) of dates, ISO strings or timestamps.

    Returns:
        Series of tz-naive pandas Timestamps (NaT where unparsable).
    """
    parsed = pd.to_datetime(pd.Series(values), errors="coerce", utc=True)
    return parsed.dt.tz_convert(None)


def first_reconciliation_on_or_after(dates: pd.Series) -> pd.Series:
    """Midnight of the first Tuesday on or after each date.

    Args:
        dates: tz-naive timestamps (for example due dates).

    Returns:
        Series of Tuesday midnights; NaT where the input is NaT.
    """
    days = dates.dt.normalize()
    return days + pd.to_timedelta(((RECONCILIATION_WEEKDAY - days.dt.weekday) % 7).fillna(0), unit="D")


def reconciliation_offset(due: pd.Series, released: pd.Series) -> pd.Series:
    """Observed recon_k: weekly files between the first file after due and the release.

    Args:
        due: tz-naive due dates.
        released: tz-naive release (reconciliation file) timestamps.

    Returns:
        Float series of whole weeks (NaN where either date is missing). Negative values
        mean the invoice was released before its due date (early clearing).
    """
    weeks = (released.dt.normalize() - first_reconciliation_on_or_after(due)).dt.days / 7
    return weeks.round()


def release_timestamp(due: pd.Series, recon_k: pd.Series, day_fraction: float) -> pd.Series:
    """Release timestamp for a given file offset.

    Args:
        due: tz-naive due dates.
        recon_k: integer offsets (weekly files after the first file on/after due).
        day_fraction: typical time of day of the reconciliation file, in days (0-1).

    Returns:
        tz-naive release timestamps.
    """
    days = pd.to_numeric(pd.Series(recon_k, index=due.index), errors="coerce") * 7 + float(day_fraction)
    return first_reconciliation_on_or_after(due) + pd.to_timedelta(days.fillna(0), unit="D").where(days.notna())


def planned_funding_dates(offer_dates: pd.Series, planning_start) -> pd.Series:
    """Wednesday of the first planning week an invoice may be funded in.

    Mirrors the optimizer's week eligibility (a Monday week start must be on or after
    the offer timestamp) and never precedes the run's first planning week. Summary files
    (purchases) are processed on Wednesdays, hence Monday + 2 days.

    Args:
        offer_dates: offer file timestamps per candidate (NaT allowed).
        planning_start: Monday that starts the planning horizon (date, str or Timestamp).

    Returns:
        tz-naive Wednesday midnights, one per candidate.
    """
    start = pd.Timestamp(planning_start).normalize()
    offers = to_naive(offer_dates).reset_index(drop=True)
    # First Monday that is >= the offer timestamp (a Monday midnight offer counts as that week).
    offer_days = offers.dt.normalize()
    monday = offer_days + pd.to_timedelta(((7 - offer_days.dt.weekday) % 7).fillna(0), unit="D")
    monday = monday.where(monday >= offers, monday + pd.Timedelta(days=7))
    week = monday.where(monday.notna() & (monday > start), start)
    return (week + pd.Timedelta(days=FUNDING_WEEKDAY_OFFSET)).set_axis(pd.Series(offer_dates).index)


def as_of_customer_priors(customers: pd.Series, funded_at: pd.Series, known_at: pd.Series,
                          recon_k: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Customer median recon_k and history size, using only outcomes known before funding.

    For every history row, only rows of the same customer whose outcome was known
    strictly before that row's funding date contribute, which is what the business knew
    when the invoice was funded. This keeps the prior free of leakage when history rows
    are used as RPT-1 context examples.

    Args:
        customers: customer identifier per row.
        funded_at: tz-naive funding timestamps.
        known_at: tz-naive outcome-availability timestamps (release file).
        recon_k: observed offsets.

    Returns:
        (median_k, history_count) Series aligned with the inputs; median is NaN when no
        earlier outcome exists.
    """
    median = pd.Series(np.nan, index=customers.index, dtype="float64")
    count = pd.Series(0, index=customers.index, dtype="int64")
    for _, index in customers.groupby(customers).groups.items():
        known = known_at.loc[index].values.astype("datetime64[ns]")
        order = np.argsort(known, kind="stable")
        known_sorted, k_sorted = known[order], recon_k.loc[index].values[order]
        counts = np.searchsorted(known_sorted, funded_at.loc[index].values.astype("datetime64[ns]"), side="left")
        count.loc[index] = counts
        median.loc[index] = [float(np.median(k_sorted[:n])) if n else np.nan for n in counts]
    return median, count


def customer_margin_offsets(customers: pd.Series, recon_k: pd.Series, quantile: float | None,
                            min_rows: int) -> dict[str, int]:
    """Minimum file offset per customer used as a safety margin.

    Args:
        customers: customer identifier per history row.
        recon_k: observed offsets per history row.
        quantile: e.g. 0.75; None disables the margin.
        min_rows: customers with fewer rows get no margin.

    Returns:
        {customer: ceil(quantile of recon_k)} for customers with enough history.
    """
    if quantile is None:
        return {}
    frame = pd.DataFrame({"customer": customers.values, "k": recon_k.values}).dropna()
    stats = frame.groupby("customer")["k"].agg(["size", lambda values: values.quantile(quantile)])
    stats.columns = ["size", "quantile"]
    eligible = stats[stats["size"] >= int(min_rows)]
    return {customer: int(math.ceil(value)) for customer, value in eligible["quantile"].items()}


if __name__ == "__main__":
    # Self-check: Tuesday arithmetic, offsets and Wednesday funding weeks.
    due = to_naive(["2025-12-02", "2025-12-03", "2025-12-08"])  # Tue, Wed, Mon
    assert list(first_reconciliation_on_or_after(due).dt.strftime("%a %d")) == ["Tue 02", "Tue 09", "Tue 09"]
    released = to_naive(["2025-12-09 13:00", "2025-12-09 09:00", "2025-12-23 10:00"])
    assert list(reconciliation_offset(due, released)) == [1, 0, 2]
    assert release_timestamp(due, pd.Series([1, 0, 2]), 0.5).dt.strftime("%m-%d %H").tolist() == ["12-09 12", "12-09 12", "12-23 12"]
    funding = planned_funding_dates(pd.Series(["2025-11-26 10:00", "2025-12-01 00:00", None]), "2025-11-24")
    assert funding.dt.strftime("%a %m-%d").tolist() == ["Wed 12-03", "Wed 12-03", "Wed 11-26"]
    print("reconciliation calendar self-check passed")
