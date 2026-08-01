"""
Prompt builder for the AI Chat module.
Constructs structured prompts for the SAP Gen AI Hub LLM.
"""

from typing import Dict, List, Optional


SYSTEM_PERSONA = """You are a senior financial analyst and SAP Controlling specialist at the company.
You have deep expertise in:
- Year-over-year variance analysis
- SAP journal entry interpretation (document types WE, RE, SA, etc.)
- Cost center and profit center accounting
- G/L account analysis and financial statement line items
- Supplier spend analysis
- Flash Line Item expense grouping from SAP v_glaccount:
    * Travel and Entertainment
    * Employee Expense
    * Outside Services
    * Repair and Maintenance
    * Depreciation
    * Utilities
    * Insurance
    * Rent
- FS Item and Financial Statement Line Item hierarchy

FLASH LINE ITEM PRIORITY:
When analyzing expenses, always use the Flash Line Item field (from SAP v_glaccount) as the
authoritative expense category. This is the official SAP semantic grouping. Prioritize it over
journal entry text or account names when categorizing expenses.

FS HIERARCHY:
Use the following priority order for financial categorization:
  1. Flash Line Item  (most specific — operational grouping)
  2. FS Item Name     (financial statement grouping)
  3. Financial Statement Line Item  (balance sheet / P&L line)

Your responses must be:
- Concise and financially meaningful (150-300 words unless a detailed drill-down is requested)
- Grounded in the data provided — never invent numbers
- Professional, suitable for executive reporting
- Specific: cite account numbers, amounts, percentages, and dimensions
- Actionable: identify drivers, anomalies, and trends
- Conversational: maintain context across follow-up questions

When data is insufficient, say so clearly and suggest what additional filters would help."""


def build_chat_prompt(
    user_message: str,
    conversation_history: str,
    financial_context: str,
    entities: Dict,
    current_year: Optional[int] = None,
    previous_year: Optional[int] = None,
) -> str:
    """
    Build the full structured prompt for the LLM.
    """
    parts = [SYSTEM_PERSONA, ""]

    # Conversation history
    if conversation_history.strip():
        parts.append("=== CONVERSATION HISTORY ===")
        parts.append(conversation_history)
        parts.append("")

    # Active context / detected entities
    entity_lines = _format_entities(entities, current_year, previous_year)
    if entity_lines:
        parts.append("=== ACTIVE CONTEXT ===")
        parts.append(entity_lines)
        parts.append("")

    # Financial data
    if financial_context.strip():
        parts.append("=== FINANCIAL DATA ===")
        parts.append(financial_context)
        parts.append("")

    # Current question
    parts.append("=== USER QUESTION ===")
    parts.append(user_message)
    parts.append("")
    parts.append("=== YOUR RESPONSE ===")

    return "\n".join(parts)


def _format_entities(
    entities: Dict,
    current_year: Optional[int],
    previous_year: Optional[int],
) -> str:
    lines = []
    if current_year:
        lines.append(f"Current FY: {current_year}")
    if previous_year:
        lines.append(f"Previous FY: {previous_year}")
    if entities.get("gl_account"):
        lines.append(f"G/L Account: {entities['gl_account']}")
    if entities.get("cost_center"):
        lines.append(f"Cost Center: {entities['cost_center']}")
    if entities.get("profit_center"):
        lines.append(f"Profit Center: {entities['profit_center']}")
    if entities.get("wbs_element"):
        lines.append(f"WBS Element: {entities['wbs_element']}")
    if entities.get("purchasing_doc"):
        lines.append(f"Purchasing Document: {entities['purchasing_doc']}")
    if entities.get("supplier_name"):
        lines.append(f"Supplier: {entities['supplier_name']}")
    if entities.get("quarter"):
        lines.append(f"Quarter: Q{entities['quarter']}")
    if entities.get("flash_category"):
        lines.append(f"Flash Category (SAP): {entities['flash_category']}")
    elif entities.get("semantic_category"):
        lines.append(f"Category: {entities['semantic_category']}")
    return "\n".join(lines)


def build_financial_context(
    matched_rows: int,
    filters_applied: List[str],
    gl_summary: List[Dict],
    cost_center_summary: List[Dict],
    supplier_summary: List[Dict],
    je_text_summary: List[Dict],
    variance_summary: List[Dict],
    current_year: Optional[int] = None,
    previous_year: Optional[int] = None,
    total_current: Optional[float] = None,
    total_previous: Optional[float] = None,
    flash_summary: Optional[List[Dict]] = None,
) -> str:
    """
    Build the financial context block to inject into the prompt.
    flash_summary: aggregated Flash Line Item breakdown (from v_glaccount).
    """
    lines = []

    if filters_applied:
        lines.append(f"Filters: {', '.join(filters_applied)}")
    lines.append(f"Matched records: {matched_rows:,}")

    if total_current is not None and total_previous is not None:
        delta = total_current - total_previous
        pct = ((delta / abs(total_previous)) * 100) if total_previous != 0 else 0.0
        lines.append("")
        lines.append("YEAR-OVER-YEAR TOTALS:")
        if previous_year:
            lines.append(f"  FY {previous_year}: ${total_previous:,.2f}")
        if current_year:
            lines.append(f"  FY {current_year}: ${total_current:,.2f}")
        lines.append(f"  Delta: ${delta:,.2f} ({pct:+.1f}%)")

    # Flash Line Item breakdown — authoritative SAP expense grouping (v_glaccount)
    if flash_summary:
        source = flash_summary[0].get("source", "Flash Line Item") if flash_summary else "Flash Line Item"
        lines.append("")
        lines.append(f"FLASH LINE ITEM BREAKDOWN [{source}] (authoritative SAP grouping):")
        for f in flash_summary[:12]:
            lines.append(
                f"  {f['flash_category']}: ${f['amount']:,.2f}  [{f['records']:,} records]"
            )

    if variance_summary:
        lines.append("")
        lines.append("TOP VARIANCES (by absolute delta):")
        for v in variance_summary[:8]:
            pct_str = f" ({v['delta_pct']:+.1f}%)" if v.get("delta_pct") is not None else ""
            lines.append(
                f"  {v['key']}: ${v['current']:,.2f} vs ${v['previous']:,.2f}"
                f"  Δ ${v['delta']:,.2f}{pct_str}"
            )

    if gl_summary:
        lines.append("")
        lines.append("TOP G/L ACCOUNTS (by amount):")
        for g in gl_summary[:8]:
            name = f" — {g['name']}" if g.get("name") else ""
            lines.append(f"  {g['gl_account']}{name}: ${g['amount']:,.2f}")

    if cost_center_summary:
        lines.append("")
        lines.append("TOP COST CENTERS:")
        for c in cost_center_summary[:6]:
            lines.append(f"  {c['cost_center']}: ${c['amount']:,.2f}")

    if supplier_summary:
        lines.append("")
        lines.append("TOP SUPPLIERS:")
        for s in supplier_summary[:6]:
            lines.append(f"  {s['supplier']}: ${s['amount']:,.2f}")

    if je_text_summary:
        lines.append("")
        lines.append("MOST FREQUENT JOURNAL ENTRY TEXTS:")
        for j in je_text_summary[:6]:
            lines.append(f"  '{j['text']}' ({j['count']} times)")

    return "\n".join(lines)


def build_no_data_prompt(
    user_message: str,
    conversation_history: str,
    entities: Dict,
    current_year: Optional[int] = None,
    previous_year: Optional[int] = None,
) -> str:
    """
    Prompt when no matching records were found in the dataset.
    """
    parts = [SYSTEM_PERSONA, ""]

    if conversation_history.strip():
        parts.append("=== CONVERSATION HISTORY ===")
        parts.append(conversation_history)
        parts.append("")

    entity_lines = _format_entities(entities, current_year, previous_year)
    if entity_lines:
        parts.append("=== ACTIVE CONTEXT ===")
        parts.append(entity_lines)
        parts.append("")

    parts.append("=== NOTE ===")
    parts.append(
        "No matching records were found in the dataset for the current filters. "
        "Please acknowledge this clearly and suggest what the user could try "
        "(e.g., different account number, different year, broader search terms)."
    )
    parts.append("")
    parts.append("=== USER QUESTION ===")
    parts.append(user_message)
    parts.append("")
    parts.append("=== YOUR RESPONSE ===")

    return "\n".join(parts)