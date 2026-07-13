"""LangGraph tools for the credit optimizer (read-only, summary-first)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from ...optimizer.report.labels import RULE_DESCRIPTIONS
from ...services.optimizer.process_manager import ProcessManager

DEFAULT_ROW_LIMIT = 50
MAX_ROW_LIMIT = 100
DEFAULT_TOP_N = 20
MAX_TOP_N = 100


_OPTIMIZER_REASON_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "planning_window_offer_file_date": {
        "stage": "rule",
        "meaning": (
            "Offer File Date week is outside the configured multi-week planning window "
            "(from planning start week to horizon end week, inclusive)."
        ),
        "source": "pipeline_prefilter",
    },
    "EXPIRED_WINDOW": {
        "stage": "optimizer",
        "meaning": (
            "No feasible planning week remains for this invoice because all candidate weeks "
            "are outside the allowed offer/due-date window."
        ),
        "source": "multi_week_optimizer_explainer",
    },
    "DEFERRED_FOR_CAPACITY": {
        "stage": "optimizer",
        "meaning": (
            "At least one feasible week existed, but the optimizer selected other invoices "
            "with a better objective under shared limits."
        ),
        "source": "multi_week_optimizer_explainer",
    },
    "FACILITY_CAP_BINDING": {
        "stage": "optimizer",
        "meaning": "Adding the invoice would breach at least one facility limit in the active weeks.",
        "source": "multi_week_optimizer_explainer",
    },
    "CUSTOMER_CAP_BINDING": {
        "stage": "optimizer",
        "meaning": "Adding the invoice would breach at least one customer limit in the active weeks.",
        "source": "multi_week_optimizer_explainer",
    },
    "GROUP_CAP_BINDING": {
        "stage": "optimizer",
        "meaning": "Adding the invoice would breach at least one group limit in the active weeks.",
        "source": "multi_week_optimizer_explainer",
    },
    "exceeded facility headroom": {
        "stage": "optimizer",
        "meaning": "Invoice was not selected because it would exceed remaining facility headroom.",
        "source": "single_week_optimizer_explainer",
    },
    "exceeded customer headroom": {
        "stage": "optimizer",
        "meaning": "Invoice was not selected because it would exceed remaining customer headroom.",
        "source": "single_week_optimizer_explainer",
    },
    "exceeded group headroom": {
        "stage": "optimizer",
        "meaning": "Invoice was not selected because it would exceed remaining group headroom.",
        "source": "single_week_optimizer_explainer",
    },
    "not chosen (budget used elsewhere)": {
        "stage": "optimizer",
        "meaning": "Invoice was feasible but not selected after the optimizer allocated budget to other invoices.",
        "source": "single_week_optimizer_explainer",
    },
}
_RULE_REASON_DEFINITIONS: Dict[str, Dict[str, str]] = {
    rule_name: {
        "stage": "rule",
        "meaning": description,
        "source": "rule_engine",
    }
    for rule_name, description in RULE_DESCRIPTIONS.items()
}
_REASON_DEFINITIONS: Dict[str, Dict[str, str]] = {
    **_RULE_REASON_DEFINITIONS,
    **_OPTIMIZER_REASON_DEFINITIONS,
}
_REASON_LOOKUP_LOWER: Dict[str, str] = {key.lower(): key for key in _REASON_DEFINITIONS.keys()}


def _get_manager() -> ProcessManager:
    return ProcessManager()


def _clamp_row_limit(limit: int) -> int:
    return max(1, min(int(limit), MAX_ROW_LIMIT))


def _clamp_top_n(top_n: int) -> int:
    return max(1, min(int(top_n), MAX_TOP_N))


def _pagination(total: int, limit: int, offset: int, returned: int) -> Dict[str, Any]:
    has_more = (offset + returned) < total
    return {
        "total": int(total),
        "offset": int(offset),
        "limit": int(limit),
        "returned": int(returned),
        "has_more": bool(has_more),
    }


def _envelope(
    process_id: Optional[str],
    data: Any,
    summary: Dict[str, Any],
    guidance: str,
    *,
    total: Optional[int] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> Dict[str, Any]:
    if isinstance(data, list):
        returned = len(data)
    elif data is None:
        returned = 0
    else:
        returned = 1

    computed_total = int(total if total is not None else returned)
    computed_limit = int(limit if limit is not None else max(returned, 1))
    paging = _pagination(computed_total, computed_limit, offset, returned)
    truncated = bool(paging["has_more"])
    next_offset = (offset + returned) if truncated else None

    return {
        "process_id": process_id or "",
        "data": data,
        "summary": summary,
        "pagination": paging,
        "truncated": truncated,
        "next_offset": next_offset,
        "guidance": guidance,
    }


def _error(
    message: str,
    process_id: Optional[str] = None,
    guidance: str = "Verify the process ID and try again.",
) -> Dict[str, Any]:
    return {
        "process_id": process_id or "",
        "data": [],
        "summary": {"error": message},
        "pagination": _pagination(0, 1, 0, 0),
        "truncated": False,
        "next_offset": None,
        "guidance": guidance,
        "error": message,
    }


def _normalize_reason_text(reason: Any) -> str:
    return " ".join(str(reason or "").strip().split())


def _resolve_reason_definition(reason: Any) -> Dict[str, Any]:
    raw = _normalize_reason_text(reason)
    if not raw:
        return {
            "reason": "",
            "known": False,
            "stage": "",
            "meaning": "No exclusion reason was provided.",
            "source": "",
        }

    canonical = _REASON_LOOKUP_LOWER.get(raw.lower())
    if canonical:
        definition = _REASON_DEFINITIONS[canonical]
        return {
            "reason": canonical,
            "known": True,
            "stage": definition.get("stage", ""),
            "meaning": definition.get("meaning", ""),
            "source": definition.get("source", ""),
        }

    return {
        "reason": raw,
        "known": False,
        "stage": "",
        "meaning": (
            "No explicit definition is registered for this reason code. "
            "Use get_optimizer_invoice_rows on excluded invoices for row-level details."
        ),
        "source": "",
    }


def _reason_definitions_for(reasons: List[Any]) -> List[Dict[str, Any]]:
    seen = set()
    entries: List[Dict[str, Any]] = []
    for reason in reasons:
        normalized = _normalize_reason_text(reason)
        if not normalized:
            continue
        key = _REASON_LOOKUP_LOWER.get(normalized.lower(), normalized.lower())
        if key in seen:
            continue
        seen.add(key)
        entries.append(_resolve_reason_definition(normalized))
    entries.sort(key=lambda item: (0 if item.get("known") else 1, str(item.get("reason", "")).lower()))
    return entries


@tool("get_optimizer_reason_legend")
def get_optimizer_reason_legend(reason: Optional[str] = None) -> Dict[str, Any]:
    """Get explicit meanings for optimizer exclusion reason codes/messages."""
    if reason and reason.strip():
        queries = [part.strip() for part in reason.split(",") if part.strip()]
        matched: List[Dict[str, Any]] = []
        seen = set()
        for query in queries:
            added_for_query = False
            exact = _resolve_reason_definition(query)
            if exact.get("known"):
                key = str(exact.get("reason", "")).lower()
                if key not in seen:
                    seen.add(key)
                    matched.append(exact)
                    added_for_query = True
                continue

            needle = query.lower()
            for code in sorted(_REASON_DEFINITIONS.keys()):
                if needle in code.lower():
                    key = code.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    matched.append(_resolve_reason_definition(code))
                    added_for_query = True

            if exact.get("reason") and not added_for_query:
                key = str(exact.get("reason", "")).lower()
                if key not in seen:
                    seen.add(key)
                    matched.append(exact)

        return _envelope(
            process_id="",
            data=matched,
            summary={
                "query": reason,
                "returned": len(matched),
                "known_reason_count": len(_REASON_DEFINITIONS),
            },
            guidance=(
                "For process-specific counts, use get_optimizer_exclusion_summary. "
                "For invoice examples, use get_optimizer_invoice_rows (bucket='excluded')."
            ),
            total=len(matched),
            limit=max(len(matched), 1),
            offset=0,
        )

    all_entries = [_resolve_reason_definition(code) for code in sorted(_REASON_DEFINITIONS.keys())]
    return _envelope(
        process_id="",
        data=all_entries,
        summary={"returned": len(all_entries), "known_reason_count": len(_REASON_DEFINITIONS)},
        guidance="Use reason='EXPIRED_WINDOW' (or comma-separated values) to look up specific codes quickly.",
        total=len(all_entries),
        limit=len(all_entries),
        offset=0,
    )


@tool("list_optimizer_processes")
def list_optimizer_processes(
    status: Optional[str] = None,
    limit: int = DEFAULT_ROW_LIMIT,
    offset: int = 0,
) -> Dict[str, Any]:
    """List optimization processes with compact status and KPI fields."""
    mgr = _get_manager()
    limit = _clamp_row_limit(limit)
    offset = max(0, int(offset))
    records = mgr.list_processes(status=status, limit=limit, offset=offset)
    total = mgr.store.count_processes(status=status)

    processes = []
    for r in records:
        processes.append(
            {
                "process_id": r["id"],
                "status": r["status"],
                "cohort": r.get("cohort"),
                "planning_mode": r.get("planning_mode"),
                "source_profile": r.get("source_profile"),
                "extraction_filename": r.get("extraction_filename"),
                "created_at": r.get("created_at"),
                "candidate_count": r.get("candidate_count"),
                "selected_count": r.get("selected_count"),
                "excluded_count": r.get("excluded_count"),
                "selected_amount": r.get("selected_amount"),
                "candidate_amount": r.get("candidate_amount"),
                "optimizer_status": r.get("optimizer_status"),
                "error_message": r.get("error_message"),
            }
        )

    return _envelope(
        process_id="",
        data=processes,
        summary={
            "status_filter": status,
            "count": len(processes),
        },
        guidance="Use resolve_optimizer_process_id for short/truncated IDs before deeper queries.",
        total=total,
        limit=limit,
        offset=offset,
    )


@tool("resolve_optimizer_process_id")
def resolve_optimizer_process_id(
    process_ref: str,
    status: Optional[str] = None,
    limit: int = DEFAULT_ROW_LIMIT,
) -> Dict[str, Any]:
    """Resolve a full or truncated process reference to a unique process_id."""
    if not process_ref or not process_ref.strip():
        return _error("process_ref is required.")

    mgr = _get_manager()
    limit = _clamp_row_limit(limit)
    resolved = mgr.resolve_process_id(process_ref=process_ref, status=status, limit=limit)
    matches = resolved.get("matches", [])
    match_type = resolved.get("match_type")
    process_id = resolved.get("process_id")

    if match_type == "not_found":
        guidance = "No process matched this reference. Use list_optimizer_processes and provide a longer prefix."
    elif match_type == "prefix_ambiguous":
        guidance = "Multiple processes match this prefix. Provide a longer process ID."
    else:
        guidance = "Use the resolved process_id for overview, summary, or row drill-down tools."

    return _envelope(
        process_id=process_id,
        data=matches,
        summary={
            "input_process_ref": process_ref,
            "match_type": match_type,
            "resolved_process_id": process_id,
            "match_count": len(matches),
            "scanned": resolved.get("scanned", 0),
        },
        guidance=guidance,
        total=len(matches),
        limit=limit,
        offset=0,
    )


@tool("get_optimizer_limits")
def get_optimizer_limits(process_id: str) -> Dict[str, Any]:
    """Get read-only limits configuration for an optimization process."""
    mgr = _get_manager()
    try:
        limits = mgr.get_limits(process_id)
    except ValueError as exc:
        return _error(str(exc), process_id=process_id)

    return _envelope(
        process_id=process_id,
        data=limits,
        summary={"keys": sorted(list(limits.keys())), "key_count": len(limits)},
        guidance="Limits are read-only in chat. Update configuration through the UI/API workflow.",
    )


@tool("get_optimizer_overview")
def get_optimizer_overview(process_id: str) -> Dict[str, Any]:
    """Get compact, report-like KPI overview for a process."""
    mgr = _get_manager()
    try:
        summary = mgr.get_overview_summary(process_id)
    except ValueError as exc:
        return _error(str(exc), process_id=process_id)

    deferred_sorted = sorted(
        (summary.get("deferred_reasons") or {}).items(),
        key=lambda item: item[1],
        reverse=True,
    )
    deferred_reasons = [reason for reason, _ in deferred_sorted]
    data = {
        "cohort": summary.get("cohort"),
        "planning_mode": summary.get("planning_mode"),
        "source_profile": summary.get("source_profile"),
        "horizon_weeks": summary.get("horizon_weeks"),
        "kpis": summary.get("kpis", {}),
        "deferred_reasons": [{"reason": reason, "count": count} for reason, count in deferred_sorted],
        "deferred_reason_definitions": _reason_definitions_for(deferred_reasons),
        "binding_constraints_top": (summary.get("binding_constraints") or [])[:10],
        "top_customers": (summary.get("top_customers") or [])[:10],
        "weekly_schedule_top": (summary.get("weekly_schedule_summary") or [])[:10],
        "row_counts": summary.get("row_counts", {}),
    }

    return _envelope(
        process_id=process_id,
        data=data,
        summary={
            "optimizer_status": (summary.get("kpis") or {}).get("optimizer_status"),
            "selection_ratio_pct": (summary.get("kpis") or {}).get("selected_amount_ratio_pct"),
            "binding_constraints_count": len(summary.get("binding_constraints") or []),
        },
        guidance="For details, use exclusion/utilization summary tools. For samples, use paged row tools.",
    )


@tool("get_optimizer_exclusion_summary")
def get_optimizer_exclusion_summary(
    process_id: str,
    stage: Optional[str] = None,
    top_n: int = DEFAULT_TOP_N,
) -> Dict[str, Any]:
    """Get aggregated exclusion reasons (count and amount), optionally filtered by stage."""
    mgr = _get_manager()
    try:
        summary = mgr.get_exclusions_summary(process_id)
    except ValueError as exc:
        return _error(str(exc), process_id=process_id)

    rows = list(summary.get("rows") or [])
    if stage:
        wanted = stage.strip().lower()
        rows = [row for row in rows if wanted in str(row.get("stage", "")).lower()]

    rows = sorted(rows, key=lambda row: (-int(row.get("count", 0)), str(row.get("reason", ""))))
    top_n = _clamp_top_n(top_n)
    data = rows[:top_n]
    reason_definitions = _reason_definitions_for([row.get("reason") for row in rows])

    return _envelope(
        process_id=process_id,
        data=data,
        summary={
            "stage_filter": stage,
            "available_reasons": len(rows),
            "returned_reasons": len(data),
            "reason_definitions": reason_definitions,
        },
        guidance=f"If you need invoice-level samples, use get_optimizer_invoice_rows (bucket='excluded'). Full file: /api/optimizer/processes/{process_id}/download/excluded",
        total=len(rows),
        limit=top_n,
        offset=0,
    )


@tool("get_optimizer_utilization_summary")
def get_optimizer_utilization_summary(
    process_id: str,
    entity_type: str = "facility",
    view: str = "peak",
    week_start: Optional[str] = None,
    top_n: int = DEFAULT_TOP_N,
) -> Dict[str, Any]:
    """Get utilization summaries without returning full exposure tables."""
    mgr = _get_manager()
    try:
        summary = mgr.get_utilization_summary(process_id)
    except ValueError as exc:
        return _error(str(exc), process_id=process_id)

    normalized_entity = (entity_type or "facility").strip().lower()
    if normalized_entity not in {"facility", "customer", "group"}:
        return _error(
            f"Unknown entity_type '{entity_type}'. Use facility, customer, or group.",
            process_id=process_id,
            guidance="Valid values: facility, customer, group.",
        )

    rows = list((summary.get("rows") or {}).get(normalized_entity, []))
    if not rows:
        return _envelope(
            process_id=process_id,
            data=[],
            summary={"entity_type": normalized_entity, "view": view, "returned_rows": 0},
            guidance=f"No utilization rows available for entity_type={normalized_entity}.",
            total=0,
            limit=_clamp_top_n(top_n),
            offset=0,
        )

    normalized_view = (view or "peak").strip().lower()
    result_rows: List[Dict[str, Any]] = []
    if normalized_view == "peak":
        by_entity: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            entity_id = str(row.get("entity_id", ""))
            if entity_id not in by_entity or float(row.get("utilization_pct", 0) or 0) > float(
                by_entity[entity_id].get("utilization_pct", 0) or 0
            ):
                by_entity[entity_id] = row
        result_rows = list(by_entity.values())
    elif normalized_view == "latest":
        by_entity = {}
        for row in rows:
            entity_id = str(row.get("entity_id", ""))
            current_week = str(row.get("week_start", ""))
            previous_week = str(by_entity.get(entity_id, {}).get("week_start", ""))
            if entity_id not in by_entity or current_week > previous_week:
                by_entity[entity_id] = row
        result_rows = list(by_entity.values())
    elif normalized_view == "week":
        if not week_start:
            return _error(
                "week_start is required when view='week'.",
                process_id=process_id,
                guidance="Provide an explicit week_start (for example 2025-02-04).",
            )
        result_rows = [row for row in rows if str(row.get("week_start", "")) == str(week_start)]
    else:
        return _error(
            f"Unknown view '{view}'. Use peak, latest, or week.",
            process_id=process_id,
            guidance="Valid values: peak, latest, week.",
        )

    result_rows = sorted(
        result_rows,
        key=lambda row: (
            -float(row.get("utilization_pct", 0) or 0),
            str(row.get("entity_id", "")),
            str(row.get("week_start", "")),
        ),
    )

    top_n = _clamp_top_n(top_n)
    data = result_rows[:top_n]
    available_types = summary.get("available_entity_types") or ["facility", "customer", "group"]

    return _envelope(
        process_id=process_id,
        data=data,
        summary={
            "entity_type": normalized_entity,
            "view": normalized_view,
            "week_start": week_start,
            "available_entity_types": available_types,
            "returned_rows": len(data),
        },
        guidance=f"For row-level drill-down, use get_optimizer_weekly_exposure_rows. Full file: /api/optimizer/processes/{process_id}/download/weekly-exposure",
        total=len(result_rows),
        limit=top_n,
        offset=0,
    )


@tool("get_optimizer_weekly_schedule_summary")
def get_optimizer_weekly_schedule_summary(
    process_id: str,
    top_n: int = DEFAULT_TOP_N,
) -> Dict[str, Any]:
    """Get per-week invoice count and amount summary for multi-week planning."""
    mgr = _get_manager()
    try:
        summary = mgr.get_schedule_summary(process_id)
    except ValueError as exc:
        return _error(str(exc), process_id=process_id)

    rows = sorted(
        list(summary.get("rows") or []),
        key=lambda row: (str(row.get("week_start", "")), int(row.get("week_index", 0) or 0)),
    )
    top_n = _clamp_top_n(top_n)
    data = rows[:top_n]
    total_amount = float(sum(float(row.get("total_amount", 0) or 0) for row in rows))
    total_invoices = int(sum(int(row.get("invoice_count", 0) or 0) for row in rows))

    return _envelope(
        process_id=process_id,
        data=data,
        summary={
            "planning_mode": summary.get("planning_mode"),
            "weeks_available": len(rows),
            "invoice_count_total": total_invoices,
            "amount_total": total_amount,
        },
        guidance=f"For invoice-level weekly rows, use get_optimizer_invoice_rows (bucket='weekly_plan'). Full file: /api/optimizer/processes/{process_id}/download/weekly-plan",
        total=len(rows),
        limit=top_n,
        offset=0,
    )


@tool("get_optimizer_invoice_decision")
def get_optimizer_invoice_decision(
    process_id: str,
    invoice_reference: str,
    match_mode: str = "contains",
    max_matches: int = DEFAULT_TOP_N,
) -> Dict[str, Any]:
    """Explain invoice decision outcome (selected or excluded) by reference."""
    if not invoice_reference or not invoice_reference.strip():
        return _error("invoice_reference is required.", process_id=process_id)

    mgr = _get_manager()
    decisions = mgr.find_invoice_decisions(
        process_id=process_id,
        invoice_reference=invoice_reference,
        match_mode=match_mode,
        max_matches=max_matches,
    )

    data = decisions.get("matches", [])
    total = int(decisions.get("total", 0))
    max_matches = _clamp_top_n(max_matches)
    if total == 0:
        guidance = "No matching invoice found in selected/excluded outputs. Try a longer reference fragment."
    elif total > len(data):
        guidance = "Multiple matches found. Narrow the reference or switch match_mode to exact."
    else:
        guidance = "Use get_optimizer_invoice_rows for broader filtered samples."

    reason_definitions = _reason_definitions_for([row.get("excluded_reason") for row in data])
    return _envelope(
        process_id=process_id,
        data=data,
        summary={
            "invoice_reference": invoice_reference,
            "match_mode": match_mode,
            "matches_found": len(data),
            "reason_definitions": reason_definitions,
        },
        guidance=guidance,
        total=total,
        limit=max_matches,
        offset=0,
    )


@tool("get_optimizer_invoice_rows")
def get_optimizer_invoice_rows(
    process_id: str,
    bucket: str = "excluded",
    invoice_ref: Optional[str] = None,
    customer: Optional[str] = None,
    company_code: Optional[str] = None,
    excluded_stage: Optional[str] = None,
    excluded_reason: Optional[str] = None,
    week_start: Optional[str] = None,
    limit: int = DEFAULT_ROW_LIMIT,
    offset: int = 0,
) -> Dict[str, Any]:
    """Get paginated invoice rows (selected, excluded, or weekly_plan) with filters."""
    normalized_bucket = (bucket or "excluded").strip().lower()
    if normalized_bucket not in {"selected", "excluded", "weekly_plan"}:
        return _error(
            f"Unknown bucket '{bucket}'. Use selected, excluded, or weekly_plan.",
            process_id=process_id,
            guidance="Valid bucket values: selected, excluded, weekly_plan.",
        )

    mgr = _get_manager()
    limit = _clamp_row_limit(limit)
    offset = max(0, int(offset))
    paged = mgr.get_invoice_rows(
        process_id=process_id,
        bucket=normalized_bucket,
        invoice_ref=invoice_ref,
        customer=customer,
        company_code=company_code,
        excluded_stage=excluded_stage,
        excluded_reason=excluded_reason,
        week_start=week_start,
        limit=limit,
        offset=offset,
    )
    rows = paged["rows"]

    return _envelope(
        process_id=process_id,
        data=rows,
        summary={
            "bucket": normalized_bucket,
            "filters": {
                "invoice_ref": invoice_ref,
                "customer": customer,
                "company_code": company_code,
                "excluded_stage": excluded_stage,
                "excluded_reason": excluded_reason,
                "week_start": week_start,
            },
            "returned_rows": len(rows),
        },
        guidance=f"Use offset pagination for more rows. For full data, use /api/optimizer/processes/{process_id}/download/{'excluded' if normalized_bucket == 'excluded' else ('selected' if normalized_bucket == 'selected' else 'weekly-plan')}",
        total=paged["total"],
        limit=paged["limit"],
        offset=paged["offset"],
    )


@tool("get_optimizer_weekly_exposure_rows")
def get_optimizer_weekly_exposure_rows(
    process_id: str,
    entity_type: Optional[str] = None,
    entity_id: Optional[str] = None,
    week_start: Optional[str] = None,
    limit: int = DEFAULT_ROW_LIMIT,
    offset: int = 0,
) -> Dict[str, Any]:
    """Get paginated weekly exposure rows with optional filters."""
    mgr = _get_manager()
    limit = _clamp_row_limit(limit)
    offset = max(0, int(offset))
    paged = mgr.get_weekly_exposure_rows(
        process_id=process_id,
        limit=limit,
        offset=offset,
        entity_type=entity_type,
        entity_id=entity_id,
        week_start=week_start,
    )
    rows = paged["rows"]
    return _envelope(
        process_id=process_id,
        data=rows,
        summary={
            "filters": {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "week_start": week_start,
            },
            "returned_rows": len(rows),
        },
        guidance=f"Use offset pagination for additional rows. Full file: /api/optimizer/processes/{process_id}/download/weekly-exposure",
        total=paged["total"],
        limit=paged["limit"],
        offset=paged["offset"],
    )


OPTIMIZER_TOOLS = [
    list_optimizer_processes,
    resolve_optimizer_process_id,
    get_optimizer_limits,
    get_optimizer_reason_legend,
    get_optimizer_overview,
    get_optimizer_exclusion_summary,
    get_optimizer_utilization_summary,
    get_optimizer_weekly_schedule_summary,
    get_optimizer_invoice_decision,
    get_optimizer_invoice_rows,
    get_optimizer_weekly_exposure_rows,
]
