"""Assemble optimizer run data into a structured ``ReportContext``.

Reads the metadata dict produced by ``pipeline_runner`` together with the Excel
output files (selected.xlsx, excluded.xlsx) and computes derived aggregations
needed by the report builder.  The returned ``ReportContext`` is a plain
dataclass -- no I/O, no side-effects -- making it easy to test and serialize.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ReportContext:
    """All data required to render the full optimizer report."""

    # Run identification
    cohort: str = ""
    planning_mode: str = "single_week"
    horizon_weeks: int = 1
    solver_status: str = "N/A"
    planning_start_date: str = ""

    # Selection KPIs
    candidate_count: int = 0
    selected_count: int = 0
    excluded_count: int = 0
    not_selected_count: int = 0
    candidate_amount: float = 0.0
    selected_amount: float = 0.0
    selection_ratio_pct: float = 0.0
    top3_concentration_pct: float = 0.0

    # Invoice detail tables (capped for the report, full data in Excel)
    selected_invoices: List[Dict[str, Any]] = field(default_factory=list)
    excluded_invoices: List[Dict[str, Any]] = field(default_factory=list)
    total_selected_invoices: int = 0
    total_excluded_invoices: int = 0

    # Exclusion summary (aggregated by reason)
    exclusion_summary: List[Dict[str, Any]] = field(default_factory=list)

    # Weekly schedule (multi-week only)
    weekly_schedule: List[Dict[str, Any]] = field(default_factory=list)

    # Utilization tables
    facility_utilization: List[Dict[str, Any]] = field(default_factory=list)
    customer_utilization: List[Dict[str, Any]] = field(default_factory=list)

    # Top customers by selected amount
    top_customers: List[Dict[str, Any]] = field(default_factory=list)

    # Entities where utilization >= 95% in any week
    binding_constraints: List[Dict[str, Any]] = field(default_factory=list)

    # Rule funnel
    rule_summaries: List[Dict[str, Any]] = field(default_factory=list)

    # Lifecycle & RPT-1
    lifecycle_profile: Dict[str, Any] = field(default_factory=dict)
    lifetime_estimation: Dict[str, Any] = field(default_factory=dict)

    # Misc
    limits: Dict[str, Any] = field(default_factory=dict)
    load_report: Dict[str, Any] = field(default_factory=dict)
    deferred_reasons: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Excel column subsets for the report tables
# ---------------------------------------------------------------------------

_SELECTED_COLUMNS = [
    "Invoice Reference",
    "Customer",
    "Company Code",
    "Purchase Price",
    "Currency",
    "Due Date",
    "planned_week_start_iso",
    "expected_lifetime_weeks",
]

_EXCLUDED_COLUMNS = [
    "Invoice Reference",
    "Customer",
    "Purchase Price",
    "excluded_stage",
    "excluded_reason",
    "excluded_reason_detail",
]


def _safe_read_excel(path: Path, columns: List[str]) -> pd.DataFrame:
    """Read an Excel file, selecting only the requested columns that exist."""
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_excel(path, engine="openpyxl")
        available = [c for c in columns if c in df.columns]
        return df[available] if available else df
    except Exception as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return pd.DataFrame()


def _build_exclusion_summary(excluded_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Aggregate excluded invoices by reason with count and total amount."""
    if excluded_df.empty:
        return []

    reason_col = "excluded_reason" if "excluded_reason" in excluded_df.columns else None
    stage_col = "excluded_stage" if "excluded_stage" in excluded_df.columns else None
    amount_col = "Purchase Price" if "Purchase Price" in excluded_df.columns else None

    if reason_col is None:
        return []

    groups = excluded_df.groupby(reason_col, dropna=False)
    rows: List[Dict[str, Any]] = []
    for reason, grp in groups:
        row: Dict[str, Any] = {
            "reason": str(reason) if reason else "Unknown",
            "count": len(grp),
        }
        if stage_col:
            row["stage"] = str(grp[stage_col].iloc[0]) if not grp[stage_col].isna().all() else ""
        if amount_col and amount_col in grp.columns:
            row["total_amount"] = float(grp[amount_col].sum())
        rows.append(row)

    return sorted(rows, key=lambda r: r["count"], reverse=True)


def _build_weekly_schedule(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build a per-week summary from the weekly_plan list in metadata."""
    weekly_plan = metadata.get("weekly_plan") or []
    if not weekly_plan:
        return []

    by_week: Dict[str, Dict[str, Any]] = {}
    for row in weekly_plan:
        week_iso = str(row.get("planned_week_start_iso", ""))
        week_idx = row.get("planned_week_index", 0)
        amount = float(row.get("purchase_price", 0) or row.get("Purchase Price", 0) or 0)

        if week_iso not in by_week:
            by_week[week_iso] = {
                "week_index": int(week_idx),
                "week_start": week_iso,
                "invoice_count": 0,
                "total_amount": 0.0,
            }
        by_week[week_iso]["invoice_count"] += 1
        by_week[week_iso]["total_amount"] += amount

    return sorted(by_week.values(), key=lambda w: w["week_start"])


def _build_facility_utilization(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten facility_weekly_usage into constrained per-facility-per-week rows."""
    weekly = metrics.get("facility_weekly_usage") or {}
    if not weekly:
        single = metrics.get("facility_usage") or {}
        if single:
            weekly = {"current": single}

    rows: List[Dict[str, Any]] = []
    for week, entities in sorted(weekly.items()):
        for entity_id, usage in sorted(entities.items()):
            if not isinstance(usage, dict):
                continue
            limit = float(usage.get("limit", 0) or 0)
            if limit <= 0:
                # Unconstrained entities are noise in this section.
                continue
            rows.append({
                "week": week,
                "entity_id": entity_id,
                "used_new": float(usage.get("used_new", 0) or 0),
                "used_base": float(usage.get("used_base", 0) or 0),
                "used_total": float(usage.get("used_total", usage.get("used", 0)) or 0),
                "limit": limit,
                "utilization_pct": float(usage.get("utilization_pct", 0) or 0),
            })
    return sorted(rows, key=lambda r: (str(r["entity_id"]), str(r["week"])))


def _build_customer_utilization(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten customer_weekly_usage, keeping only the highest-utilization snapshot per customer."""
    weekly = metrics.get("customer_weekly_usage") or {}
    if not weekly:
        single = metrics.get("customer_usage") or {}
        if single:
            weekly = {"current": single}

    peak: Dict[str, Dict[str, Any]] = {}
    for week, entities in weekly.items():
        for entity_id, usage in entities.items():
            if not isinstance(usage, dict):
                continue
            pct = usage.get("utilization_pct", 0)
            if entity_id not in peak or pct > peak[entity_id].get("utilization_pct", 0):
                peak[entity_id] = {
                    "entity_id": entity_id,
                    "week": week,
                    "used_new": usage.get("used_new", 0),
                    "used_base": usage.get("used_base", 0),
                    "used_total": usage.get("used_total", usage.get("used", 0)),
                    "limit": usage.get("limit", 0),
                    "utilization_pct": pct,
                }

    return sorted(peak.values(), key=lambda r: r["utilization_pct"], reverse=True)


def _build_top_customers(selected_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Top 10 customers by total selected amount with share percentage."""
    if selected_df.empty:
        return []

    cust_col = "Customer" if "Customer" in selected_df.columns else None
    amt_col = "Purchase Price" if "Purchase Price" in selected_df.columns else None
    if not cust_col or not amt_col:
        return []

    by_cust = (
        selected_df.groupby(cust_col, dropna=False)[amt_col]
        .sum()
        .sort_values(ascending=False)
    )
    total = float(by_cust.sum())
    rows: List[Dict[str, Any]] = []
    for cust, amount in by_cust.head(10).items():
        rows.append({
            "customer": str(cust),
            "selected_amount": float(amount),
            "share_pct": float(amount) / total * 100.0 if total > 0 else 0.0,
        })
    return rows


def _find_binding_constraints(metrics: Dict[str, Any], threshold: float = 95.0) -> List[Dict[str, Any]]:
    """Entities where utilization_pct >= *threshold* in any week."""
    results: List[Dict[str, Any]] = []
    seen = set()

    for source_key, entity_type in [
        ("facility_weekly_usage", "facility"),
        ("customer_weekly_usage", "customer"),
        ("group_weekly_usage", "group"),
    ]:
        weekly = metrics.get(source_key) or {}
        for week, entities in weekly.items():
            for entity_id, usage in entities.items():
                if not isinstance(usage, dict):
                    continue
                pct = usage.get("utilization_pct", 0)
                if pct >= threshold:
                    key = (entity_type, entity_id)
                    if key not in seen:
                        seen.add(key)
                        results.append({
                            "entity_type": entity_type,
                            "entity_id": entity_id,
                            "peak_utilization_pct": pct,
                            "peak_week": week,
                            "limit": usage.get("limit", 0),
                        })

    return sorted(results, key=lambda r: r["peak_utilization_pct"], reverse=True)


def assemble_report_context(
    metadata: Dict[str, Any],
    output_dir: Path,
    *,
    selected_df: Optional[pd.DataFrame] = None,
    excluded_df: Optional[pd.DataFrame] = None,
) -> ReportContext:
    """Build a ``ReportContext`` from pipeline metadata and output files.

    Args:
        metadata: The dict written to ``run_metadata.json`` by the pipeline.
        output_dir: Directory containing selected.xlsx, excluded.xlsx, etc.
        selected_df: Optional preloaded selected invoices dataframe.
        excluded_df: Optional preloaded excluded invoices dataframe.

    Returns:
        Fully-populated ``ReportContext``.
    """
    metrics = metadata.get("metrics") or {}
    planning_mode = metadata.get("planning_mode", metrics.get("planning_mode", "single_week"))

    # Read Excel files for invoice details
    if selected_df is None:
        selected_df = _safe_read_excel(output_dir / "selected.xlsx", _SELECTED_COLUMNS)
    else:
        selected_df = selected_df.copy()
        available = [c for c in _SELECTED_COLUMNS if c in selected_df.columns]
        selected_df = selected_df[available] if available else selected_df

    if excluded_df is None:
        excluded_df = _safe_read_excel(output_dir / "excluded.xlsx", _EXCLUDED_COLUMNS)
    else:
        excluded_df = excluded_df.copy()
        available = [c for c in _EXCLUDED_COLUMNS if c in excluded_df.columns]
        excluded_df = excluded_df[available] if available else excluded_df

    # Build derived structures
    exclusion_summary = _build_exclusion_summary(excluded_df)
    weekly_schedule = _build_weekly_schedule(metadata) if planning_mode == "multi_week" else []
    facility_util = _build_facility_utilization(metrics)
    customer_util = _build_customer_utilization(metrics)
    top_customers = _build_top_customers(selected_df)
    binding = _find_binding_constraints(metrics)

    # Cap invoice lists for the report
    max_selected = 25
    max_excluded = 15
    selected_records = selected_df.head(max_selected).fillna("").to_dict("records")
    excluded_records = excluded_df.head(max_excluded).fillna("").to_dict("records")

    return ReportContext(
        cohort=str(metadata.get("cohort", "")),
        planning_mode=planning_mode,
        horizon_weeks=int(metadata.get("horizon_weeks", metrics.get("horizon_weeks", 1))),
        solver_status=str(metrics.get("optimizer_status", "N/A")),
        planning_start_date=str(metadata.get("planning_start_date", "")),
        candidate_count=int(metrics.get("baseline_submitted_count", 0)),
        selected_count=int(metrics.get("optimized_submitted_count", 0)),
        excluded_count=int(metrics.get("rule_excluded_count", 0)),
        not_selected_count=int(metrics.get("not_selected_count", 0)),
        candidate_amount=float(metrics.get("candidate_total_amount", 0)),
        selected_amount=float(metrics.get("selected_total_amount", 0)),
        selection_ratio_pct=float(metrics.get("selected_amount_ratio_pct", 0)),
        top3_concentration_pct=float(metrics.get("top3_customer_concentration_pct", 0)),
        selected_invoices=selected_records,
        excluded_invoices=excluded_records,
        total_selected_invoices=len(selected_df),
        total_excluded_invoices=len(excluded_df),
        exclusion_summary=exclusion_summary,
        weekly_schedule=weekly_schedule,
        facility_utilization=facility_util,
        customer_utilization=customer_util,
        top_customers=top_customers,
        binding_constraints=binding,
        rule_summaries=metadata.get("rule_summaries") or [],
        lifecycle_profile=metrics.get("lifecycle_profile") or {},
        lifetime_estimation=metadata.get("lifetime_estimation") or {},
        limits=metadata.get("limits") or {},
        load_report=metadata.get("load_report") or {},
        deferred_reasons=metrics.get("deferred_reasons") or metadata.get("deferred_reasons") or {},
    )
