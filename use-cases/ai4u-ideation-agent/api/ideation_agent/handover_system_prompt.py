"""System prompt for the Handover Agent (Requirements Specialist).

Unlike the ideation agent (creative Process Architect), the handover agent
is structured, efficient, and evidence-based. It helps fill form gaps
by searching linked context and asking targeted questions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# Section definitions matching the frontend form.
# Keys match the panel ID suffixes in handover.html (e.g. "customer" → id="section-customer").
# Field names match the HTML input IDs (e.g. "advisor_name" → id="f-advisor_name").
SECTIONS = {
    "customer": {
        "title": "Customer & Workshop Context",
        "fields": [
            "customer", "industry", "workshop_date", "workshop_format",
            "advisor_name", "primary_contact", "tech_contact",
        ],
    },
    "usecase": {
        "title": "Use Case Definition",
        "fields": ["usecase_title", "problem_statement", "priority"],
    },
    "process": {
        "title": "Current Process (As-Is)",
        "fields": [
            "process_name", "process_freq", "process_volume",
            "process_team", "process_people",
        ],
    },
    "data": {
        "title": "Data Sources & Semantics",
        "fields": ["data_input", "data_output", "data_usage"],
        "tables": ["data_sources"],
    },
    "impact": {
        "title": "Business Impact",
        "special": "business_impact",  # impact table with current/target rows
    },
    "ai": {
        "title": "AI Capabilities",
        "special": "ai_capabilities",  # clickable capability grid
    },
    "tobe": {
        "title": "To-Be Process & Scope",
        "fields": ["tobe_narrative", "in_scope", "out_scope", "done_definition"],
    },
    "rules": {
        "title": "Business Rules, Thresholds & KPIs",
        "tables": ["business_rules", "thresholds", "approvals", "edge_cases", "kpis"],
    },
    "tech": {
        "title": "Technical Integration",
        "tables": ["tech_components", "integrations", "contacts"],
    },
    "checklist": {
        "title": "Handover Checklist",
        "fields": ["open_notes"],
        "special": "checklist",
    },
}

REQUIRED_FIELDS = [
    "customer", "industry", "usecase_title", "problem_statement",
    "process_name", "process_freq", "data_input", "tobe_narrative",
]


def get_handover_system_prompt(
    form_state: Dict[str, Any],
    context_sources: List[Dict[str, Any]],
    target_section: Optional[str] = None,
) -> str:
    """Build the system prompt for the handover agent.

    Args:
        form_state: Current form payload (all field values).
        context_sources: List of linked context sources with extracted content.
        target_section: Optional section ID the user wants help with.

    Returns:
        Complete system prompt string.
    """
    # Build form status summary
    form_summary = _build_form_summary(form_state)

    # Build context inventory
    context_inventory = _build_context_inventory(context_sources)

    # Build section focus hint
    section_hint = ""
    if target_section and target_section in SECTIONS:
        sec = SECTIONS[target_section]
        fields_list = sec.get('fields', []) + sec.get('tables', [])
        if sec.get('special'):
            fields_list.append(sec['special'])
        section_hint = f"""
## CURRENT FOCUS
The user wants help with: **{sec['title']}**
Focus your search_context calls on information relevant to this section.
Fields in this section: {', '.join(fields_list)}
"""

    return f"""You are a **Requirements Specialist** helping fill a handover workspace form.

## YOUR ROLE
You are structured, efficient, and evidence-based. You help the user complete their handover form by:
1. Searching linked context (ideation outputs, transcripts, notes) for relevant information
2. Proposing field values based on evidence found
3. Asking targeted questions about gaps you cannot fill from context
4. Tracking progress and highlighting what's still missing

## RULES
- **Always search before suggesting.** Use search_context to find evidence before proposing values.
- **Cite your sources.** When proposing values, mention where you found the information.
- **Ask 1-3 questions per turn** about gaps you couldn't fill from context. No more.
- **Accept "skip" or "I don't know" gracefully.** Not every field needs filling.
- **Never hallucinate values.** If you can't find information, say so explicitly.
- **Keep responses concise** — 2-4 sentences for explanations, not paragraphs.
- **Use fill_fields to propose values** — never ask the user to copy-paste into the form.

## RESPONSE FORMAT
Your final message should always:
1. Summarize what you did (what fields you proposed, what you found/didn't find)
2. Ask 1-3 specific follow-up questions about remaining gaps
3. If no gaps remain, congratulate the user on completion

## TOOLS
- `search_context(query)` — Search all linked sources. Use specific keywords.
- `fill_fields(fields)` — Propose values for form fields. These appear as suggestions the user can accept/dismiss.
- `get_section_status(section_id)` — Check which fields in a section are filled vs empty. Section IDs: customer, usecase, process, data, impact, ai, tobe, rules, tech, checklist.

## CURRENT FORM STATUS
{form_summary}

## AVAILABLE CONTEXT
{context_inventory}
{section_hint}"""


def _build_form_summary(form_state: Dict[str, Any]) -> str:
    """Summarize which fields are filled and which are empty."""
    if not form_state:
        return "The form is completely empty. No fields have been filled yet."

    filled = []
    empty_required = []
    empty_optional = []

    for section_id, section in SECTIONS.items():
        for field in section.get("fields", []):
            value = form_state.get(field)
            if value and str(value).strip():
                filled.append(field)
            elif field in REQUIRED_FIELDS:
                empty_required.append(field)
            else:
                empty_optional.append(field)

        for table in section.get("tables", []):
            value = form_state.get(table)
            if value and isinstance(value, list) and len(value) > 0:
                filled.append(f"{table} (table, {len(value)} rows)")
            else:
                empty_optional.append(f"{table} (table)")

        # Handle special sections (impact table, AI capabilities grid)
        special = section.get("special")
        if special == "business_impact":
            impact = form_state.get("business_impact")
            if isinstance(impact, dict):
                filled_rows = sum(
                    1 for v in impact.values()
                    if isinstance(v, dict) and any(str(c).strip() for c in v.values())
                )
                if filled_rows:
                    filled.append(f"business_impact ({filled_rows} rows)")
                else:
                    empty_optional.append("business_impact (table)")
            else:
                empty_optional.append("business_impact (table)")
        elif special == "ai_capabilities":
            caps = form_state.get("ai_capabilities")
            if isinstance(caps, list) and len(caps) > 0:
                filled.append(f"ai_capabilities ({len(caps)} selected)")
            else:
                empty_optional.append("ai_capabilities")
        elif special == "checklist":
            checklist = form_state.get("checklist")
            if isinstance(checklist, list) and len(checklist) > 0:
                checked = sum(1 for item in checklist if item.get("checked"))
                filled.append(f"checklist ({checked}/{len(checklist)} checked)")
            else:
                empty_optional.append("checklist (not started)")

    lines = []
    if filled:
        lines.append(f"**Filled ({len(filled)}):** {', '.join(filled[:15])}")
        if len(filled) > 15:
            lines.append(f"  ... and {len(filled) - 15} more")
    if empty_required:
        lines.append(f"**Empty REQUIRED ({len(empty_required)}):** {', '.join(empty_required)}")
    if empty_optional:
        lines.append(f"**Empty optional ({len(empty_optional)}):** {', '.join(empty_optional[:10])}")
        if len(empty_optional) > 10:
            lines.append(f"  ... and {len(empty_optional) - 10} more")

    total = len(filled) + len(empty_required) + len(empty_optional)
    pct = round(len(filled) / total * 100) if total > 0 else 0
    lines.insert(0, f"**Progress: {pct}%** ({len(filled)}/{total} fields)")

    return "\n".join(lines)


def _build_context_inventory(context_sources: List[Dict[str, Any]]) -> str:
    """List available context sources."""
    if not context_sources:
        return "No context sources linked. The user should add ideation conversations or notes first."

    lines = []
    for source in context_sources:
        stype = source.get("source_type", "unknown")
        sref = source.get("source_ref", "")
        has_extracted = bool(source.get("extracted_fields"))
        status = "✓ extracted" if has_extracted else "○ not yet extracted"
        label = f"- **{stype}**: {sref[:50] if sref else '(no reference)'} [{status}]"
        lines.append(label)

    return "\n".join(lines)
