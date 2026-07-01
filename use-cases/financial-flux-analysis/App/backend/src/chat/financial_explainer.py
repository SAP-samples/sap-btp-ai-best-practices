"""
Financial explainer helpers for the chat module.
Computes totals, variance summaries, and monthly breakdowns
from a filtered DataFrame slice.
"""

from typing import Dict, List, Optional

import pandas as pd

AMOUNT_NUMERIC = "amount_numeric"
FISCAL_YEAR    = "fiscal_year"
FISCAL_MONTH   = "fiscal_month"

MONTH_LABELS = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


def compute_totals(
    df: pd.DataFrame,
    current_year: Optional[int],
    previous_year: Optional[int],
) -> Dict:
    """
    Compute total amounts for current and previous year from a filtered DataFrame.
    Returns dict with total_current, total_previous, delta, delta_pct.
    """
    if df.empty or AMOUNT_NUMERIC not in df.columns or FISCAL_YEAR not in df.columns:
        return {
            "total_current": 0.0,
            "total_previous": 0.0,
            "delta": 0.0,
            "delta_pct": 0.0,
        }

    cur_total  = float(df[df[FISCAL_YEAR] == current_year][AMOUNT_NUMERIC].sum()) if current_year else 0.0
    prev_total = float(df[df[FISCAL_YEAR] == previous_year][AMOUNT_NUMERIC].sum()) if previous_year else 0.0
    delta      = cur_total - prev_total
    delta_pct  = round((delta / abs(prev_total)) * 100, 1) if prev_total != 0 else 0.0

    return {
        "total_current":  round(cur_total, 2),
        "total_previous": round(prev_total, 2),
        "delta":          round(delta, 2),
        "delta_pct":      delta_pct,
    }


def compute_monthly_breakdown(
    df: pd.DataFrame,
    current_year: Optional[int],
    previous_year: Optional[int],
) -> List[Dict]:
    """Monthly breakdown for current vs previous year."""
    if df.empty or FISCAL_YEAR not in df.columns or FISCAL_MONTH not in df.columns:
        return []

    cur_df  = df[df[FISCAL_YEAR] == current_year]  if current_year  else pd.DataFrame()
    prev_df = df[df[FISCAL_YEAR] == previous_year] if previous_year else pd.DataFrame()

    cur_monthly  = cur_df.groupby(FISCAL_MONTH)[AMOUNT_NUMERIC].sum()  if not cur_df.empty  else pd.Series(dtype=float)
    prev_monthly = prev_df.groupby(FISCAL_MONTH)[AMOUNT_NUMERIC].sum() if not prev_df.empty else pd.Series(dtype=float)

    all_months = set(cur_monthly.index) | set(prev_monthly.index)
    results = []
    for month in sorted(all_months):
        m  = int(month)
        ca = float(cur_monthly.get(month, 0.0))
        pa = float(prev_monthly.get(month, 0.0))
        results.append({
            "month":    m,
            "label":    MONTH_LABELS.get(m, str(m)),
            "current":  round(ca, 2),
            "previous": round(pa, 2),
            "delta":    round(ca - pa, 2),
        })
    return results


def build_fallback_response(
    user_message: str,
    entities: Dict,
    totals: Dict,
    current_year: Optional[int],
    previous_year: Optional[int],
    matched_rows: int,
) -> str:
    """
    Rule-based fallback response when the LLM is unavailable.
    """
    lines = ["**Financial Analysis Summary** (AI Hub unavailable — rule-based response)", ""]

    if entities.get("gl_account"):
        lines.append(f"**G/L Account:** {entities['gl_account']}")
    if entities.get("cost_center"):
        lines.append(f"**Cost Center:** {entities['cost_center']}")
    if entities.get("profit_center"):
        lines.append(f"**Profit Center:** {entities['profit_center']}")

    if current_year and previous_year:
        lines.append(f"**Period:** FY {previous_year} vs FY {current_year}")

    lines.append(f"**Matched records:** {matched_rows:,}")
    lines.append("")

    cur  = totals.get("total_current", 0.0)
    prev = totals.get("total_previous", 0.0)
    delta = totals.get("delta", 0.0)
    pct   = totals.get("delta_pct", 0.0)

    if cur != 0 or prev != 0:
        direction = "increased" if delta >= 0 else "decreased"
        lines.append(
            f"The selected dimension {direction} by **${abs(delta):,.2f}** "
            f"({pct:+.1f}%) year-over-year."
        )
        if previous_year:
            lines.append(f"- FY {previous_year}: ${prev:,.2f}")
        if current_year:
            lines.append(f"- FY {current_year}: ${cur:,.2f}")
    else:
        lines.append("No financial data found for the specified filters.")
        lines.append("Please verify the account number, year, or search terms.")

    lines.append("")
    lines.append("*Note: AI Hub is currently unavailable. Connect to SAP Gen AI Hub for full analysis.*")

    return "\n".join(lines)