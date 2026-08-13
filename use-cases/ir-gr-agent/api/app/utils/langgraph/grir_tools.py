"""GR/IR procurement tools for the LangGraph chat agent."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Data loading — once at import time
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parents[3] / "data"

_NUMERIC_COLS = [
    "Goods Receipt Amount",
    "Invoice Receipt Amount",
    "Balance Amount",
    "Goods Receipt Quantity",
    "Invoice Receipt Quantity",
    "Balance Quantity",
    "Is Invoice Goods Amount Surplus",
    "Is Goods Receipt Goods Amount Surplus",
    "Is Invoice Goods Quantity Surplus",
    "Is Goods Receipt Goods Quantity Surplus",
    "No Invoice Receipt Posted",
]


def _load_source() -> pd.DataFrame:
    path = _DATA_DIR / "Source.csv"
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip()
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Latest Posting Date"] = pd.to_datetime(df["Latest Posting Date"], errors="coerce")
    return df


_SOURCE_DF: pd.DataFrame = _load_source()

# ---------------------------------------------------------------------------
# MR11 eligibility — tolerance limits: 5% of GR amount OR abs $25
# ---------------------------------------------------------------------------

_MR11_PCT_THRESHOLD = 0.05
_MR11_ABS_THRESHOLD = 25.0


def _mr11_eligibility(row: pd.Series) -> Dict[str, Any]:
    """Return MR11 write-off eligibility based on tolerance limits."""
    balance_amt = float(row.get("Balance Amount") or 0)
    gr_amt = float(row.get("Goods Receipt Amount") or 0)
    abs_balance = abs(balance_amt)

    within_abs = abs_balance <= _MR11_ABS_THRESHOLD
    pct_variance = (abs_balance / abs(gr_amt)) if gr_amt != 0 else None
    within_pct = (pct_variance is not None) and (pct_variance <= _MR11_PCT_THRESHOLD)

    eligible = abs_balance > 0 and (within_abs or within_pct)
    return {
        "mr11_eligible": eligible,
        "balance_amount": balance_amt,
        "abs_threshold_met": within_abs,
        "pct_variance": round(pct_variance * 100, 2) if pct_variance is not None else None,
        "pct_threshold_met": within_pct,
    }


# ---------------------------------------------------------------------------
# Issue classification — rule-based, matches Reasoning.csv patterns
# ---------------------------------------------------------------------------

def _classify_issue(row: pd.Series) -> str:
    """Return the issue type label for a single GR/IR row."""
    no_ir = row.get("No Invoice Receipt Posted", 0)
    gr_surplus_amt = row.get("Is Goods Receipt Goods Amount Surplus", 0)
    ir_surplus_qty = row.get("Is Invoice Goods Quantity Surplus", 0)
    gr_surplus_qty = row.get("Is Goods Receipt Goods Quantity Surplus", 0)
    ir_surplus_amt = row.get("Is Invoice Goods Amount Surplus", 0)
    balance_qty = row.get("Balance Quantity", 0)
    balance_amt = row.get("Balance Amount", 0)
    gr_amt = row.get("Goods Receipt Amount", 0)
    ir_amt = row.get("Invoice Receipt Amount", 0)

    if gr_amt == 0 and ir_amt != 0 and no_ir == 0:
        return "Missing GR"
    if no_ir == 1 and gr_amt != 0:
        return "Missing IR"
    if ir_surplus_amt == 1 and gr_surplus_amt == 0 and balance_amt > 0:
        return "GR Reversals"
    if gr_surplus_qty == 1 and balance_qty != 0:
        return "Quantity Mismatch"
    if balance_qty == 0 and balance_amt != 0:
        return "Qty Match, Price Mismatch"
    if balance_amt != 0 and abs(balance_amt) < 10:
        return "Residual Difference"
    if abs(balance_amt) >= 100_000:
        return "High Dollar- Urgency"
    if balance_amt != 0:
        return "Balance Discrepancy"
    return "No Issue"


def _po_type(row: pd.Series) -> str:
    """Return 'Corporate' (8-series) or 'Retail' (5-series) based on PO number prefix."""
    po = str(row.get("Purchasing Document") or "").strip()
    if po.startswith("8"):
        return "Corporate"
    if po.startswith("5"):
        return "Retail"
    return "Other"


def _enrich_row(row: pd.Series) -> Dict[str, Any]:
    """Return a row dict enriched with issue type, MR11 eligibility, and PO type."""
    d = row.to_dict()
    if pd.notna(d.get("Latest Posting Date")):
        d["Latest Posting Date"] = str(d["Latest Posting Date"])[:10]
    d["_issue_type"] = _classify_issue(row)
    d["_mr11"] = _mr11_eligibility(row)
    d["_po_type"] = _po_type(row)
    return d


# ---------------------------------------------------------------------------
# LangChain tools
# ---------------------------------------------------------------------------

@tool("lookup_po")
def lookup_po(po_number: str) -> Dict[str, Any]:
    """Look up all line items for a specific Purchase Order number.

    Returns each line's data including issue type classification and MR11
    write-off eligibility (based on 5% variance or $25 absolute tolerance).

    Args:
        po_number: The purchasing document number (e.g. '8000003848' or '530588').
    """
    po = str(po_number).strip()
    df = _SOURCE_DF[_SOURCE_DF["Purchasing Document"] == po]

    if df.empty:
        return {"found": False, "po_number": po, "rows": []}

    rows = [_enrich_row(row) for _, row in df.iterrows()]
    return {"found": True, "po_number": po, "rows": rows}


@tool("search_pos")
def search_pos(
    supplier_name: str = "",
    issue_type: str = "",
    po_type: str = "",
    mr11_eligible: bool = False,
    min_balance_abs: float = 0.0,
    limit: int = 20,
) -> Dict[str, Any]:
    """Search Purchase Orders by supplier name, issue type, PO type, or minimum absolute balance.

    Each result includes the issue type classification, MR11 eligibility, and PO type.

    Args:
        supplier_name: Partial supplier name (case-insensitive). Leave empty to skip.
        issue_type: One of: 'Quantity Mismatch', 'Missing GR', 'Missing IR',
            'Residual Difference', 'High Dollar- Urgency', 'GR Reversals',
            'Qty Match, Price Mismatch'. Leave empty to skip.
        po_type: 'Corporate' (8-series POs), 'Retail' (5-series POs), or leave empty for all.
        mr11_eligible: If True, only return rows eligible for MR11 write-off (abs ≤ $25 OR ≤ 5% variance).
        min_balance_abs: Only return rows where abs(Balance Amount) >= this value.
        limit: Maximum number of results to return (default 20).
    """
    df = _SOURCE_DF.copy()

    if supplier_name:
        df = df[df["Name of Supplier"].str.contains(supplier_name, case=False, na=False)]
    if min_balance_abs > 0:
        df = df[df["Balance Amount"].abs() >= min_balance_abs]

    enriched = [_enrich_row(row) for _, row in df.iterrows()]
    if issue_type:
        enriched = [r for r in enriched if r["_issue_type"] == issue_type]
    if po_type:
        enriched = [r for r in enriched if r["_po_type"].lower() == po_type.lower()]
    if mr11_eligible:
        enriched = [r for r in enriched if r["_mr11"]["mr11_eligible"]]

    enriched = enriched[:limit]
    return {"count": len(enriched), "rows": enriched}


@tool("list_issue_summary")
def list_issue_summary() -> Dict[str, Any]:
    """Return a summary of all GR/IR issues across the full dataset.

    Includes issue type counts and the total number of MR11-eligible items.
    """
    df = _SOURCE_DF.copy()
    enriched = [_enrich_row(row) for _, row in df.iterrows()]

    issue_counts: Dict[str, int] = {}
    mr11_count = 0
    for r in enriched:
        issue_counts[r["_issue_type"]] = issue_counts.get(r["_issue_type"], 0) + 1
        if r["_mr11"]["mr11_eligible"]:
            mr11_count += 1

    return {
        "total_pos": len(enriched),
        "issue_counts": issue_counts,
        "mr11_eligible_count": mr11_count,
    }


@tool("vendor_discrepancy_analysis")
def vendor_discrepancy_analysis(top_n: int = 10, sort_by: str = "volume") -> Dict[str, Any]:
    """Rank vendors by volume and severity of GR/IR discrepancies.

    Returns each vendor's total item count, issue count, discrepancy rate (%),
    total open balance, issue type breakdown, and MR11-eligible item count.

    Args:
        top_n: Number of top vendors to return (default 10).
        sort_by: 'volume' (most issues by count, default) or 'rate' (highest discrepancy percentage).
    """
    df = _SOURCE_DF.copy()
    enriched = [_enrich_row(row) for _, row in df.iterrows()]

    vendor_stats: Dict[str, Any] = {}
    for r in enriched:
        name = r.get("Name of Supplier") or "Unknown"
        if name not in vendor_stats:
            vendor_stats[name] = {
                "supplier": name,
                "total_items": 0,
                "items_with_issues": 0,
                "discrepancy_rate_pct": 0.0,
                "total_open_balance": 0.0,
                "issue_breakdown": {},
                "mr11_eligible_items": 0,
            }
        s = vendor_stats[name]
        s["total_items"] += 1
        issue = r["_issue_type"]
        if issue != "No Issue":
            s["items_with_issues"] += 1
            s["issue_breakdown"][issue] = s["issue_breakdown"].get(issue, 0) + 1
        s["total_open_balance"] += float(r.get("Balance Amount") or 0)
        if r["_mr11"]["mr11_eligible"]:
            s["mr11_eligible_items"] += 1

    # Compute discrepancy rate for each vendor
    for s in vendor_stats.values():
        s["discrepancy_rate_pct"] = round(
            (s["items_with_issues"] / s["total_items"] * 100) if s["total_items"] > 0 else 0.0, 1
        )
        s["total_open_balance"] = round(s["total_open_balance"], 2)

    if sort_by == "rate":
        ranked = sorted(
            vendor_stats.values(),
            key=lambda v: (v["discrepancy_rate_pct"], v["items_with_issues"]),
            reverse=True,
        )[:top_n]
    else:
        ranked = sorted(
            vendor_stats.values(),
            key=lambda v: (v["items_with_issues"], abs(v["total_open_balance"])),
            reverse=True,
        )[:top_n]

    return {"top_vendors": ranked, "total_vendors_analyzed": len(vendor_stats)}


@tool("find_aged_items")
def find_aged_items(days_threshold: int = 90, limit: int = 20) -> Dict[str, Any]:
    """Find GR/IR items that have been open longer than a given number of days.

    Sorted oldest-first, then by largest absolute balance. Useful for
    prioritising follow-up on stale open items.

    Args:
        days_threshold: Minimum number of days since Latest Posting Date (default 90).
        limit: Maximum number of results to return (default 20).
    """
    today = pd.Timestamp(date.today())
    df = _SOURCE_DF.copy()
    df = df[df["Latest Posting Date"].notna()]
    df["_days_open"] = (today - df["Latest Posting Date"]).dt.days
    df = df[df["_days_open"] >= days_threshold]
    df = df[df["Balance Amount"] != 0]  # only open items
    df = df.sort_values(["_days_open", "Balance Amount"], ascending=[False, True])
    df = df.head(limit)

    rows = []
    for _, row in df.iterrows():
        r = _enrich_row(row)
        r["_days_open"] = int(row["_days_open"])
        rows.append(r)

    return {
        "count": len(rows),
        "days_threshold": days_threshold,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# PO closure — in-memory simulation (demo only)
# ---------------------------------------------------------------------------

_closed_pos: set[str] = set()


@tool("close_po")
def close_po(po_number: str) -> Dict[str, Any]:
    """Mark a Purchase Order as closed in this session (demo simulation).

    Call this only after the user has explicitly confirmed they want to close
    the PO. In production this would trigger a PO close transaction in SAP.

    Args:
        po_number: The purchasing document number to close.

    Returns:
        A dict with 'closed' (bool) and a confirmation message.
    """
    po = str(po_number).strip()
    _closed_pos.add(po)
    return {
        "closed": True,
        "po_number": po,
        "message": f"PO {po} has been closed.",
    }


@tool("check_po_closed")
def check_po_closed(po_number: str) -> Dict[str, Any]:
    """Check whether a PO has been closed in this session.

    Args:
        po_number: The purchasing document number to check.
    """
    po = str(po_number).strip()
    return {"po_number": po, "is_closed": po in _closed_pos}


# ---------------------------------------------------------------------------
# Notification — mock simulation (demo only)
# ---------------------------------------------------------------------------

@tool("send_notification")
def send_notification(
    recipient_type: str,
    po_number: str,
    supplier_name: str = "",
    notification_message: str = "",
) -> Dict[str, Any]:
    """Send a mock notification to the relevant party for a GR/IR issue (demo simulation).

    Use this when the user asks to notify, alert, or send a message to a vendor,
    PTP team, or Logistics team about a PO issue.

    Args:
        recipient_type: Who to notify — one of: 'Vendor', 'PTP Team', 'Logistics/PTP'
        po_number: The purchasing document number this notification relates to.
        supplier_name: The supplier/vendor name (if known).
        notification_message: A short plain-English summary of the issue and requested action.

    Returns:
        A dict confirming the notification was sent, with all details for display.
    """
    return {
        "notification_sent": True,
        "recipient_type": recipient_type,
        "po_number": str(po_number).strip(),
        "supplier_name": supplier_name,
        "notification_message": notification_message,
        "status": "Delivered",
    }


@tool("vendor_trend_analysis")
def vendor_trend_analysis(
    supplier_name: str,
    metric: str = "volume",
    issue_type: str = "",
) -> Dict[str, Any]:
    """Analyse month-over-month GR/IR discrepancy trend for a specific supplier.

    Returns monthly data points suitable for rendering a trend line chart.

    Args:
        supplier_name: Partial or full supplier name (case-insensitive).
        metric: 'volume' (count of discrepant items per month, default) or
                'rate' (discrepant items as % of that month's total items).
        issue_type: Filter to a specific issue type (e.g. 'Missing GR').
                    Leave empty to include all issue types (default).

    Returns:
        A dict with 'supplier', 'metric', 'issue_type', 'months' (list of YYYY-MM strings),
        'values' (numeric data points), 'total_items_per_month', and 'chart_type'='line'.
    """
    df = _SOURCE_DF.copy()
    df = df[df["Name of Supplier"].str.contains(supplier_name, case=False, na=False)]

    if df.empty:
        return {
            "found": False,
            "supplier_name": supplier_name,
            "months": [],
            "values": [],
            "chart_type": "line",
        }

    matched_supplier = df["Name of Supplier"].iloc[0]

    enriched = [_enrich_row(row) for _, row in df.iterrows()]

    # Group by month using Latest Posting Date
    from collections import defaultdict
    monthly_total: Dict[str, int] = defaultdict(int)
    monthly_issues: Dict[str, int] = defaultdict(int)

    for r in enriched:
        date_str = r.get("Latest Posting Date") or ""
        if not date_str or len(date_str) < 7:
            continue
        month = date_str[:7]  # YYYY-MM
        monthly_total[month] += 1
        issue = r["_issue_type"]
        if issue == "No Issue":
            continue
        if issue_type and issue != issue_type:
            continue
        monthly_issues[month] += 1

    # Build sorted month list covering all months with activity
    all_months = sorted(set(list(monthly_total.keys()) + list(monthly_issues.keys())))

    if metric == "rate":
        values = [
            round(monthly_issues.get(m, 0) / monthly_total[m] * 100, 1)
            if monthly_total.get(m, 0) > 0 else 0.0
            for m in all_months
        ]
    else:
        values = [monthly_issues.get(m, 0) for m in all_months]

    total_items_per_month = [monthly_total.get(m, 0) for m in all_months]

    return {
        "found": True,
        "supplier": matched_supplier,
        "metric": metric,
        "issue_type": issue_type or "All Issues",
        "months": all_months,
        "values": values,
        "total_items_per_month": total_items_per_month,
        "chart_type": "line",
        "y_label": "Discrepancy Rate (%)" if metric == "rate" else "Issue Count",
    }


__all__ = [
    "lookup_po",
    "search_pos",
    "list_issue_summary",
    "vendor_discrepancy_analysis",
    "vendor_trend_analysis",
    "find_aged_items",
    "close_po",
    "check_po_closed",
    "send_notification",
]
