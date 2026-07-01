"""
FastAPI application — Financial Flux Analysis.
Backend performs ALL calculations. HANA only returns raw data.
"""

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from services.financial_processor import FinancialProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Financial Flux Analysis API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton processor — caches enriched data in memory
_processor = FinancialProcessor()

# ---------------------------------------------------------------------------
# Chat IA module — register router (additive, no existing routes modified)
# ---------------------------------------------------------------------------
try:
    from src.chat.chat_routes import router as chat_router, init_chat_service
    init_chat_service(_processor)
    app.include_router(chat_router)
    logger.info("Chat IA module loaded successfully.")
except Exception as _chat_init_err:
    logger.warning("Chat IA module could not be loaded: %s", _chat_init_err)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    try:
        from loaders.raw_loader import check_connectivity
        hana = check_connectivity()
    except Exception as exc:
        hana = {"connected": False, "message": str(exc)}
    return {
        "status": "ok",
        "hana_connected": hana["connected"],
        "hana_message":   hana["message"],
        "cached_rows":    _processor.get_row_count(),
    }


# ---------------------------------------------------------------------------
# Initialization — single call to get years + company codes
# ---------------------------------------------------------------------------

@app.get("/api/init-data")
async def get_init_data(company_code: Optional[str] = None):
    """
    Returns available fiscal years, company codes, segments, and functional areas.
    Called ONCE on frontend startup.
    """
    try:
        years        = _processor.get_available_years(company_code=company_code or None)
        filter_vals  = _processor.get_filter_values()
        return {
            "status":          "success",
            "years":           [str(y) for y in years],
            "companyCodes":    filter_vals["companyCodes"],
            "segments":        filter_vals["segments"],
            "functionalAreas": filter_vals["functionalAreas"],
        }
    except Exception as exc:
        logger.error("init-data error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Grouped FY vs FY analysis
# ---------------------------------------------------------------------------

@app.get("/api/grouped-analysis")
async def get_grouped_analysis(
    group_by:        str            = "G/L Account",
    current_year:    Optional[int]  = None,
    previous_year:   Optional[int]  = None,
    company_code:    Optional[str]  = None,
    segment:         Optional[str]  = None,
    functional_area: Optional[str]  = None,
):
    """
    FY vs FY comparison aggregated by group_by dimension.
    Returns: key, name, currentAmount, previousAmount, variance, variancePercent, records
    """
    try:
        if current_year is None or previous_year is None:
            return {
                "status":  "error",
                "message": "current_year and previous_year are required.",
            }
        results = _processor.get_grouped_analysis(
            group_by=group_by,
            current_year=int(current_year),
            previous_year=int(previous_year),
            company_code=company_code or None,
            segment=segment or None,
            functional_area=functional_area or None,
        )
        return {
            "status":       "success",
            "groupBy":      group_by,
            "currentYear":  current_year,
            "previousYear": previous_year,
            "results":      results,
        }
    except Exception as exc:
        logger.error("grouped-analysis error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Monthly drilldown
# ---------------------------------------------------------------------------

@app.get("/api/group-detail")
async def get_group_detail(
    group_by:        str            = "G/L Account",
    key:             Optional[str]  = None,
    current_year:    Optional[int]  = None,
    previous_year:   Optional[int]  = None,
    company_code:    Optional[str]  = None,
    segment:         Optional[str]  = None,
    functional_area: Optional[str]  = None,
):
    """
    Monthly breakdown for a specific key.
    Returns: month, monthLabel, currentAmount, previousAmount, delta
    """
    try:
        if not key or current_year is None or previous_year is None:
            return {
                "status":  "error",
                "message": "key, current_year, and previous_year are required.",
            }
        detail = _processor.get_group_detail(
            group_by=group_by,
            key=key,
            current_year=int(current_year),
            previous_year=int(previous_year),
            company_code=company_code or None,
            segment=segment or None,
            functional_area=functional_area or None,
        )
        return {
            "status":  "success",
            "groupBy": group_by,
            "key":     key,
            "detail":  detail,
        }
    except Exception as exc:
        logger.error("group-detail error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Refresh cache
# ---------------------------------------------------------------------------

@app.post("/api/refresh")
async def refresh_data():
    """Force reload of all data from HANA."""
    try:
        rows = _processor.refresh()
        return {"status": "success", "message": "Data refreshed.", "rows": rows}
    except Exception as exc:
        logger.error("refresh error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# AI Explain — per-account LLM analysis
# ---------------------------------------------------------------------------

class AIExplainRequest(BaseModel):
    account_number:  str
    group_by:        str = "G/L Account"
    current_year:    int
    previous_year:   int
    company_code:    Optional[str] = None
    segment:         Optional[str] = None
    functional_area: Optional[str] = None


# ---------------------------------------------------------------------------
# Netting analysis helpers
# ---------------------------------------------------------------------------

# SAP document types that represent internal reallocations / reclassifications
_REALLOCATION_DOC_TYPES = {"RC", "CO", "AB", "SA", "JE", "KP"}
# SAP document types that represent external vendor/procurement activity
_EXTERNAL_DOC_TYPES = {"KR", "RE", "WE", "KG", "KZ"}
# CONCUR-related journal entry text patterns (employee expense, not procurement)
_CONCUR_PATTERNS = {"concur", "employee", "expense report", "travel reimburs"}


def _detect_netting(doc_type_breakdown: list) -> dict:
    """
    Detect balancing RC/CO entries and compute net external activity.
    Returns netting metadata for prompt injection.
    """
    realloc_total = 0.0
    external_total = 0.0
    realloc_types = []
    external_types = []

    for dt in doc_type_breakdown:
        je_type = str(dt.get("jeType", "")).strip().upper()
        amount  = float(dt.get("amount", 0.0))
        if je_type in _REALLOCATION_DOC_TYPES:
            realloc_total += amount
            realloc_types.append(je_type)
        elif je_type in _EXTERNAL_DOC_TYPES:
            external_total += amount
            external_types.append(je_type)

    has_balancing = abs(realloc_total) > 0 and abs(realloc_total) > abs(external_total)
    return {
        "has_balancing_entries": has_balancing,
        "reallocation_total":    round(realloc_total, 2),
        "external_total":        round(external_total, 2),
        "reallocation_types":    list(set(realloc_types)),
        "external_types":        list(set(external_types)),
    }


def _is_concur_posting(supplier_breakdown: list, je_texts: list) -> bool:
    """Return True if the activity appears to be CONCUR employee expense, not procurement."""
    for je in je_texts:
        text_lower = str(je.get("text", "")).lower()
        if any(p in text_lower for p in _CONCUR_PATTERNS):
            return True
    return False


def _build_account_prompt(data: dict) -> str:
    key        = data.get("key", "")
    name       = data.get("accountName", "")
    group_by   = data.get("groupBy", "G/L Account")
    cur_year   = data.get("currentYear")
    prev_year  = data.get("previousYear")
    cur_total  = data.get("currentTotal",  0.0)
    prev_total = data.get("previousTotal", 0.0)
    delta      = data.get("delta",         0.0)
    pct        = data.get("deltaPercent",  0.0)

    doc_type_breakdown  = data.get("docTypeBreakdown",  [])
    supplier_breakdown  = data.get("supplierBreakdown", [])
    je_texts            = data.get("journalEntryTexts", [])
    gl_account_breakdown = data.get("glAccountBreakdown", [])
    top_gl_key          = data.get("topGlAccount",     "")
    top_gl_name         = data.get("topGlAccountName", "")

    # Netting analysis
    netting   = _detect_netting(doc_type_breakdown)
    is_concur = _is_concur_posting(supplier_breakdown, je_texts)

    account_nature     = data.get("accountNature",     "UNKNOWN")
    variance_direction = data.get("varianceDirection", "NEUTRAL")
    narrative_context  = data.get("narrativeContext",  "")
    is_favorable       = data.get("isFavorable")

    # Build accounting-aware direction statement for the prompt header
    if is_favorable is True:
        direction_stmt = f"VARIANCE DIRECTION: FAVORABLE — {narrative_context}"
    elif is_favorable is False:
        direction_stmt = f"VARIANCE DIRECTION: UNFAVORABLE — {narrative_context}"
    else:
        direction_stmt = (
            f"VARIANCE DIRECTION: NEUTRAL — account nature is {account_nature}. "
            f"Do not assign a favorable/unfavorable judgment without SAP classification data."
        )

    # Dimension label for the prompt header
    dim_label = group_by  # "G/L Account", "Profit Center", "Cost Center", "Financial Statement Line Item", etc.

    prompt = (
        f"You are a senior finance controller analyzing year-over-year financial variances.\n\n"
        f"Analyze the following {dim_label} data and provide a concise executive summary (250-350 words).\n\n"
        f"{dim_label.upper()}: {key} — {name}\n"
    )

    # For Financial Statement Line Item: clarify it is a P&L/BS grouping, not a GL account number
    if group_by == "Financial Statement Line Item":
        prompt += (
            f"NOTE: '{key}' is a Financial Statement Line Item grouping (from SAP v__glaccountgrouping). "
            f"It is NOT a G/L account number. It represents a P&L or Balance Sheet line item category "
            f"that aggregates multiple G/L accounts. Reference it by its full name in the narrative.\n"
        )

    # For PC/CC/FS analysis: show the top GL account driving the variance
    if group_by != "G/L Account" and top_gl_key:
        prompt += (
            f"PRIMARY VARIANCE DRIVER (G/L Account): {top_gl_key} — {top_gl_name}\n"
            f"ACCOUNT NATURE (from SAP master data, based on G/L Account {top_gl_key}): {account_nature}\n"
        )
    else:
        prompt += f"ACCOUNT NATURE (from SAP master data): {account_nature}\n"

    prompt += (
        f"PERIOD: FY {prev_year} vs FY {cur_year}\n\n"
        f"YEAR-OVER-YEAR SUMMARY:\n"
        f"  FY {prev_year} Total : ${prev_total:,.2f}\n"
        f"  FY {cur_year} Total  : ${cur_total:,.2f}\n"
        f"  Annual Delta         : ${delta:,.2f} ({pct:+.1f}%)\n"
        f"  Records (current)    : {data.get('currentRecords', 0)}\n"
        f"  Segment              : {data.get('segment', 'N/A')}\n\n"
        f"MONTHLY BREAKDOWN (FY {cur_year} vs FY {prev_year}):\n"
    )
    for m in data.get("monthly", []):
        prompt += f"  {m['label']}: ${m['current']:,.2f} vs ${m['previous']:,.2f}  (Δ ${m['delta']:,.2f})\n"

    # For PC/CC/FS: inject GL account breakdown BEFORE document types
    # The LLM sees real GL accounts, not the dimension key
    if group_by != "G/L Account" and gl_account_breakdown:
        prompt += f"\nTOP G/L ACCOUNTS WITHIN {dim_label.upper()} {key} — by absolute variance:\n"
        for gl in gl_account_breakdown[:8]:
            flash_info = f" | Flash: {gl['flashLineItem']}" if gl.get("flashLineItem") else ""
            prompt += (
                f"  - GL {gl['glAccount']} ({gl['glAccountName']}){flash_info}: "
                f"Δ ${gl['delta']:,.2f} "
                f"(FY{cur_year}: ${gl['current']:,.2f} vs FY{prev_year}: ${gl['previous']:,.2f})\n"
            )

    if doc_type_breakdown:
        prompt += f"\nDOCUMENT TYPE BREAKDOWN (FY {cur_year}):\n"
        for dt in doc_type_breakdown:
            prompt += (
                f"  - {dt['jeTypeName']} ({dt['jeType']}): "
                f"${dt['amount']:,.2f}  [{dt['count']} documents]\n"
            )

    # Netting context — inject before supplier section
    if netting["has_balancing_entries"]:
        prompt += (
            f"\nNETTING ANALYSIS:\n"
            f"  Internal reallocation/reclassification entries detected "
            f"(types: {', '.join(netting['reallocation_types'])}).\n"
            f"  Reallocation total: ${netting['reallocation_total']:,.2f}\n"
            f"  External operational total: ${netting['external_total']:,.2f}\n"
            f"  These offsetting entries neutralize each other operationally.\n"
            f"  Net external activity = ${netting['external_total']:,.2f}\n"
        )

    # Supplier attribution — strict SAP rules
    if supplier_breakdown and not is_concur:
        prompt += f"\nTOP SUPPLIERS BY AMOUNT (FY {cur_year}) — from SAP supplier fields only:\n"
        for s in supplier_breakdown:
            prompt += f"  - {s['supplier']}: ${s['amount']:,.2f}\n"
    elif is_concur:
        prompt += (
            f"\nSUPPLIER ATTRIBUTION: CONCUR employee expense posting detected. "
            f"This is NOT a procurement/vendor transaction. "
            f"Do NOT attribute to any supplier.\n"
        )
    else:
        prompt += (
            f"\nSUPPLIER ATTRIBUTION: No explicit supplier data available in SAP fields. "
            f"Do NOT infer or hallucinate a supplier name.\n"
        )

    if data.get("flashVariance"):
        prompt += (
            f"\nFLASH EXPENSE COMPARATIVE ANALYSIS — SAP v_glaccount "
            f"(FY {prev_year} vs FY {cur_year}):\n"
        )
        for item in data["flashVariance"]:
            cat      = item.get("category", "")
            prev_amt = item.get("previous", 0.0)
            cur_amt  = item.get("current",  0.0)
            delta_v  = item.get("delta",    0.0)
            d_pct    = item.get("deltaPercent")
            is_fav   = item.get("isFavorable")
            impact   = "Favorable" if is_fav is True else ("Unfavorable" if is_fav is False else "Neutral")
            pct_str  = f" ({d_pct:+.1f}%)" if d_pct is not None else ""
            prompt += (
                f"  - {cat}: "
                f"FY{prev_year} ${prev_amt:,.2f} → FY{cur_year} ${cur_amt:,.2f} "
                f"| Δ ${delta_v:,.2f}{pct_str} [{impact}]\n"
            )
            # Period concentration — show active months only
            periods = item.get("periods", [])
            active  = [p for p in periods if p.get("current", 0) != 0 or p.get("previous", 0) != 0]
            if active:
                period_parts = [
                    f"{p['monthLabel']}: "
                    f"FY{prev_year} ${p.get('previous', 0):,.2f} / "
                    f"FY{cur_year} ${p.get('current', 0):,.2f} "
                    f"(Δ ${p.get('delta', 0):,.2f})"
                    for p in active[:6]
                ]
                prompt += f"    Periods: {'; '.join(period_parts)}\n"
            # Net-zero internal reallocations
            if item.get("hasNetZeroActivity"):
                je_types = [n["jeType"] for n in item.get("netZeroActivity", [])]
                prompt += (
                    f"    ⚠ Net-zero internal reallocations detected "
                    f"(types: {', '.join(je_types)}). "
                    f"These offset each other and do NOT represent operational spend.\n"
                )
        prompt += (
            f"\nFLASH NARRATIVE RULES:\n"
            f"  - Reference dominant Flash categories by name (from SAP v_glaccount).\n"
            f"  - Describe period concentration if activity is concentrated in specific months.\n"
            f"  - If net-zero internal reallocations are present, state explicitly that they "
            f"offset each other and are NOT operational expense.\n"
            f"  - Use ONLY SAP-provided Flash Line Item values. "
            f"NEVER use 'Other' unless SAP itself contains 'Other'.\n"
        )
    elif data.get("flashSummary"):
        prompt += f"\nFLASH LINE ITEM — SAP v_glaccount (FY {cur_year}):\n"
        for cat, amt in data["flashSummary"].items():
            prompt += f"  - {cat}: ${amt:,.2f}\n"
    else:
        prompt += (
            f"\nFLASH LINE ITEM: No SAP Flash Line Item data available for this {dim_label}. "
            f"Do NOT infer or fabricate expense categories.\n"
        )

    if data.get("profitCenterBreakdown") and group_by != "Profit Center":
        prompt += f"\nTOP PROFIT CENTERS (FY {cur_year}):\n"
        for pc in data["profitCenterBreakdown"]:
            prompt += f"  - {pc['profitCenter']}: ${pc['amount']:,.2f}\n"

    if data.get("costCenterBreakdown") and group_by != "Cost Center":
        prompt += f"\nTOP COST CENTERS (FY {cur_year}):\n"
        for cc in data["costCenterBreakdown"]:
            prompt += f"  - {cc['costCenter']}: ${cc['amount']:,.2f}\n"

    if je_texts:
        prompt += "\nMOST FREQUENT JOURNAL ENTRY NARRATIVES:\n"
        for je in je_texts[:8]:
            prompt += f"  - '{je['text']}' ({je['count']} times)\n"

    prompt += f"\nACCOUNTING CONTEXT:\n  {direction_stmt}\n"

    # For PC/CC/FS: add explicit instruction to reference GL accounts in narrative
    if group_by != "G/L Account" and gl_account_breakdown:
        if group_by == "Financial Statement Line Item":
            prompt += (
                f"\nNARRATIVE REQUIREMENT FOR FINANCIAL STATEMENT LINE ITEM ANALYSIS:\n"
                f"  The analysis is for Financial Statement Line Item '{key}' ({name}). "
                f"This is a P&L/Balance Sheet grouping category — NOT a G/L account number. "
                f"The variance is driven by the G/L accounts listed above. "
                f"Your narrative MUST reference the Financial Statement Line Item by its full name ('{key}') "
                f"and the actual G/L account numbers and names as the primary drivers. "
                f"NEVER refer to '{key}' as if it were a G/L account number.\n"
            )
        else:
            prompt += (
                f"\nNARRATIVE REQUIREMENT FOR {dim_label.upper()} ANALYSIS:\n"
                f"  The analysis is for {dim_label} {key} ({name}). "
                f"The variance is driven by the G/L accounts listed above — NOT by the {dim_label} ID itself. "
                f"Your narrative MUST reference the actual G/L account numbers and names "
                f"(e.g., 'G/L Account {top_gl_key} ({top_gl_name})') as the primary drivers. "
                f"NEVER say 'account {key}' as if {key} were a G/L account.\n"
            )

    prompt += (
        "\nCRITICAL ANALYSIS RULES:\n"
        f"1. VARIANCE DIRECTION: The dominant account nature is {account_nature} in SAP "
        f"(derived from G/L Account {top_gl_key or key}). "
        f"The variance is {variance_direction}. "
        f"{narrative_context} "
        "Use this accounting context in ALL narrative wording. "
        "NEVER describe an expense decrease as 'unfavorable' or 'negative'. "
        "NEVER describe a revenue increase as 'unfavorable'. "
        "Align every sentence with the correct P&L impact direction.\n"
        "2. NETTING: If balancing reallocation entries (RC/CO/AB) are present, "
        "analyze the NET EFFECT only — not gross movement. "
        "Offsetting entries that cancel each other are internal accounting activity, "
        "NOT operational spend. State this explicitly.\n"
        "3. FLASH LINE ITEM: Use ONLY the SAP-provided Flash Line Item value. "
        "NEVER use 'Other' unless SAP itself contains 'Other'. "
        "If Flash Line Item is absent, state SAP classification is unavailable.\n"
        "4. SUPPLIER: Use ONLY explicitly populated SAP supplier fields. "
        "NEVER infer a supplier from offsetting accounts, journal text, or historical data. "
        "If CONCUR is detected, classify as employee expense — not procurement.\n"
        "5. AMOUNTS: Reference specific dollar amounts and document types.\n"
        "6. ACCURACY: Every statement must be grounded in the SAP data above. "
        "Do not hallucinate accounting interpretations.\n\n"
        "Provide your analysis:"
    )
    return prompt


def _build_fallback_explanation(data: dict) -> str:
    """Rule-based fallback when LLM is unavailable."""
    delta = data.get("delta", 0.0)
    pct   = data.get("deltaPercent", 0.0)
    direction = "increased" if delta >= 0 else "decreased"
    lines = [
        f"Account {data.get('key', '')} ({data.get('accountName', '')}) {direction} by "
        f"${abs(delta):,.2f} ({pct:+.1f}%) year-over-year.",
        f"FY {data.get('previousYear')} total: ${data.get('previousTotal', 0):,.2f}",
        f"FY {data.get('currentYear')} total: ${data.get('currentTotal', 0):,.2f}",
    ]
    if data.get("segment"):
        lines.append(f"Segment: {data['segment']}")
    if data.get("profitCenterBreakdown"):
        top_pc = data["profitCenterBreakdown"][0]
        lines.append(f"Top profit center: {top_pc['profitCenter']} (${top_pc['amount']:,.2f})")
    if data.get("journalEntryTexts"):
        top_je = data["journalEntryTexts"][0]
        lines.append(f"Most common journal entry: '{top_je['text']}' ({top_je['count']} times)")
    lines.append("(AI explanation unavailable — showing rule-based summary)")
    return "\n".join(lines)


@app.post("/api/ai-explain")
async def ai_explain(request: AIExplainRequest):
    """Generate AI explanation for a specific account using LLM."""
    try:
        data = _processor.get_ai_explanation_data(
            group_by=request.group_by,
            key=request.account_number,
            current_year=request.current_year,
            previous_year=request.previous_year,
            company_code=request.company_code or None,
            segment=request.segment or None,
            functional_area=request.functional_area or None,
        )
        if not data:
            return {"status": "error", "message": "No data found for the specified account."}

        # Attempt LLM explanation; fall back to rule-based summary on any error
        explanation = ""
        try:
            from src.explain_llm import ask_llm_simple
            prompt = _build_account_prompt(data)
            explanation = ask_llm_simple(prompt)
        except Exception as llm_err:
            logger.warning("LLM call failed (%s) — using fallback.", llm_err)
            explanation = _build_fallback_explanation(data)

        return {"status": "success", "explanation": explanation, **data}

    except Exception as exc:
        logger.error("ai-explain error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Account line items
# ---------------------------------------------------------------------------

@app.get("/api/account-line-items")
async def get_account_line_items(
    group_by:        str           = "G/L Account",
    key:             Optional[str] = None,
    current_year:    Optional[int] = None,
    previous_year:   Optional[int] = None,
    company_code:    Optional[str] = None,
    segment:         Optional[str] = None,
    functional_area: Optional[str] = None,
    limit:           int           = 200,
):
    """Return raw line items for a specific account key (both years)."""
    try:
        if not key or current_year is None or previous_year is None:
            return {
                "status":  "error",
                "message": "key, current_year, and previous_year are required.",
            }
        items = _processor.get_account_line_items(
            group_by=group_by,
            key=key,
            current_year=int(current_year),
            previous_year=int(previous_year),
            company_code=company_code or None,
            segment=segment or None,
            functional_area=functional_area or None,
            limit=limit,
        )
        return {"status": "success", "items": items}
    except Exception as exc:
        logger.error("account-line-items error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
