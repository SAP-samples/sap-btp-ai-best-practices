"""Handover export renderers — PDF, Markdown, and ZIP bundle.

Converts the handover export data dict (as built by the export endpoint)
into professional document formats for human consumption.
"""

from __future__ import annotations

import io
import json
import logging
import re
import zipfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Shared label maps (mirror frontend constants) ────────────────────────

AI_CAPABILITIES_MAP: Dict[str, str] = {
    "doc_understanding": "Document Understanding",
    "classification": "Classification",
    "anomaly_detection": "Anomaly Detection",
    "nlu": "Natural Language Understanding",
    "predictive": "Predictive / Forecasting",
    "recommendation": "Recommendation Engine",
    "conversational": "Conversational AI / Joule",
    "agentic": "Agentic / Multi-Step Automation",
    "rag": "RAG (Retrieval-Augmented Generation)",
    "computer_vision": "Computer Vision",
    "generative_text": "Generative Text",
    "other": "Other",
}

CHECKLIST_ITEMS: Dict[str, str] = {
    # Solution Advisor - Before Submitting
    "problem_clear": "Problem statement quantifies the pain (§2)",
    "impact_estimated": "Business impact estimated (§5)",
    "asis_documented": "As-is process documented (§3)",
    "asis_uploaded": "As-is process diagram uploaded (§3)",
    "data_sources": "Data sources identified with access info (§4)",
    "sample_data": "Sample data uploaded (§4)",
    "data_io_described": "Input/output/data usage described (§4)",
    "ai_caps_selected": "AI capabilities mapped (§6)",
    "tobe_written": "To-be narrative completed (§7)",
    "tobe_uploaded": "To-be diagram uploaded (§7)",
    "scope_defined": "In/out scope clearly defined (§7)",
    "rules_captured": "Business rules documented (§8)",
    "logic_uploaded": "Policy documentation uploaded (§8)",
    "kpis_defined": "KPIs with baselines defined (§8)",
    "tech_identified": "Technical landscape mapped (§9)",
    "contacts_listed": "IT contacts listed (§9)",
    "transcript_uploaded": "Workshop transcript uploaded (§10)",
    "docs_attached": "Additional documentation attached (§10)",
    # Build Team - On Receipt
    "scope_clear": "Scope and boundaries are clear",
    "data_accessible": "Sample data is accessible",
    "rules_unambiguous": "Rules are unambiguous",
    "success_measurable": "Success criteria are measurable",
    "contacts_responsive": "Technical contacts responsive",
    "ready_to_start": "Ready to start development",
}

IMPACT_DIMENSIONS: Dict[str, str] = {
    "volume": "Volume",
    "time_spent": "Time Spent Manually",
    "error_rate": "Error Rate",
    "cost_per_error": "Cost per Error",
    "annual_cost": "Annual Cost",
    "downstream": "Downstream Impact",
}

# Table schemas: key → (title, column headers)
TABLE_SCHEMAS: Dict[str, tuple] = {
    "data_sources": ("Data Sources", ["Source Name", "System", "Data Type", "Owner", "Access", "Export", "Location"]),
    "business_rules": ("Business Rules", ["Rule", "Condition (If)", "Action (Then)", "Source"]),
    "thresholds": ("Thresholds", ["Field / Variable", "Threshold / Range", "Action Below", "Action Above"]),
    "approvals": ("Approval Matrix", ["Scenario", "Approver", "Escalation Path", "SLA"]),
    "edge_cases": ("Edge Cases", ["Edge Case", "Description", "How to Handle"]),
    "kpis": ("KPIs & Metrics", ["KPI / Metric", "Current Baseline", "Target with AI"]),
    "tech_components": ("Technical Components", ["Component", "Version / Release", "Deployment"]),
    "integrations": ("Integrations", ["System A", "System B", "Method", "Frequency"]),
    "contacts": ("Contacts", ["Name", "Role", "Email / Phone", "Responsibility"]),
}

# Column keys for each table (order matches headers above)
TABLE_KEYS: Dict[str, List[str]] = {
    "data_sources": ["name", "system", "data_type", "owner", "accessibility", "export_method", "location"],
    "business_rules": ["description", "condition", "action", "source"],
    "thresholds": ["field", "range", "action_below", "action_above"],
    "approvals": ["scenario", "approver", "escalation", "sla"],
    "edge_cases": ["case", "description", "handling"],
    "kpis": ["metric", "current", "target"],
    "tech_components": ["component", "version", "deployment"],
    "integrations": ["system_a", "system_b", "method", "frequency"],
    "contacts": ["name", "role", "contact", "responsibility"],
}


# ── Helpers ───────────────────────────────────────────────────────────────

def _safe(value: Any, default: str = "—") -> str:
    """Safely convert value to display string."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    return str(value).strip()


def _sanitize_filename(title: str, max_len: int = 100) -> str:
    """Sanitize a string for use in a filename."""
    safe = re.sub(r'[/\\:*?"<>|]', "_", title)
    return safe[:max_len].strip()


def _has_table_data(rows: Optional[List[Dict]], keys: List[str]) -> bool:
    """Check if a table has any non-empty rows."""
    if not rows:
        return False
    for row in rows:
        if any(_safe(row.get(k), "") for k in keys):
            return True
    return False


def _format_timestamp(ts: Any) -> str:
    """Format a timestamp for display."""
    if not ts:
        return "—"
    if isinstance(ts, str):
        return ts[:19].replace("T", " ")
    return str(ts)


def _latin1(text: str) -> str:
    """Replace Unicode chars with latin-1 safe equivalents for PDF (Helvetica)."""
    return (
        text
        .replace("—", "--")   # em-dash
        .replace("–", "-")    # en-dash
        .replace("•", "*")    # bullet
        .replace("✓", "[x]")  # checkmark
        .replace("✗", "[ ]")  # cross
        .replace("ℹ", "i")    # info symbol
        .replace("‘", "'")    # left single quote
        .replace("’", "'")    # right single quote
        .replace("“", '"')    # left double quote
        .replace("”", '"')    # right double quote
        .replace("…", "...")  # ellipsis
        .encode("latin-1", errors="replace")
        .decode("latin-1")
    )


def _safe_pdf(value: Any, default: str = "-") -> str:
    """_safe() + latin-1 sanitization for PDF output."""
    return _latin1(_safe(value, default))

# ── ZIP Bundle Renderer ─────────────────────────────────────────────────


ZONE_FOLDER_NAMES: Dict[str, str] = {
    "asis": "3 - Current Process (As-Is)",
    "data": "4 - Data Sources",
    "tobe": "7 - To-Be Process",
    "logic": "8 - Business Rules & KPIs",
    "transcript": "10 - Transcripts",
    "docs": "10 - Supporting Documents",
    "context": "Pre-fill Sources",
}


def render_zip(
    export_data: dict,
    file_contents: list,
    ideation_messages: Optional[List[Dict]] = None,
    ideation_conversation: Optional[Dict] = None,
) -> bytes:
    """Bundle PDF + JSON + uploaded files + ideation transcript into a ZIP archive.

    Parameters
    ----------
    export_data : dict
        The handover export payload (same as JSON export).
    file_contents : list
        List of dicts with ``filename``, ``zone``, and ``data`` (bytes).
    ideation_messages : list, optional
        Messages from the linked ideation conversation (from ``store.get_messages()``).
    ideation_conversation : dict, optional
        Conversation metadata (title, created_at, etc.).
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Rendered PDF document
        try:
            pdf_bytes = render_pdf(export_data)
            zf.writestr("handover.pdf", pdf_bytes)
        except Exception:
            logger.warning("Failed to render PDF for ZIP bundle", exc_info=True)

        # Structured JSON data
        json_str = json.dumps(export_data, indent=2, default=str)
        zf.writestr("handover.json", json_str)

        # Uploaded files organized by section
        for f in file_contents:
            zone = f.get("zone", "other")
            folder = ZONE_FOLDER_NAMES.get(zone, zone)
            filename = f["filename"]
            data = f.get("data") or b""
            zf.writestr(f"files/{folder}/{filename}", data)

        # Ideation conversation transcript + final result
        if ideation_messages:
            transcript = _render_ideation_transcript(ideation_messages, ideation_conversation)
            zf.writestr("ideation/conversation.md", transcript.encode("utf-8"))

            final_result = _extract_ideation_final(ideation_messages)
            if final_result:
                zf.writestr(
                    "ideation/use-case-definition.json",
                    json.dumps(final_result, indent=2, default=str).encode("utf-8"),
                )

    return buf.getvalue()


def _render_ideation_transcript(
    messages: List[Dict], conversation: Optional[Dict] = None
) -> str:
    """Render ideation conversation messages as a readable Markdown transcript."""
    lines: List[str] = []

    title = "Ideation Conversation"
    if conversation:
        title = conversation.get("title") or title

    lines.append(f"# {title}\n")

    if conversation:
        created = _format_timestamp(conversation.get("created_at"))
        lines.append(f"**Date:** {created}")
        company = conversation.get("company_name")
        if company:
            lines.append(f"**Company:** {company}")
        lines.append("")

    lines.append("---\n")

    for msg in messages:
        role = msg.get("role", "unknown")
        content = (msg.get("content") or "").strip()
        if not content:
            continue

        metadata = msg.get("metadata") or {}
        ui = metadata.get("ui") or {}
        kind = ui.get("kind", "")

        if role == "user":
            lines.append("### User\n")
            lines.append(content)
            lines.append("")
        elif role == "agent":
            if kind == "final":
                lines.append("### AI Agent — Final Delivery\n")
            else:
                lines.append("### AI Agent\n")
            lines.append(content)
            lines.append("")

    lines.append("---\n")
    lines.append("*Exported from AI4U Ideation Agent*\n")
    return "\n".join(lines)


def _extract_ideation_final(messages: List[Dict]) -> Optional[Dict]:
    """Extract the structured use-case definition from the final delivery message."""
    for msg in reversed(messages):
        metadata = msg.get("metadata") or {}
        ui = metadata.get("ui") or {}
        if ui.get("kind") == "final":
            # The ui dict contains the structured fields
            result = {k: v for k, v in ui.items() if k != "kind"}
            if result:
                return result
    return None


# ── Markdown Renderer ────────────────────────────────────────────────────

def render_markdown(export_data: dict) -> str:
    """Render handover export data as a Markdown document."""
    p = export_data.get("payload") or {}
    lines: List[str] = []

    def heading(level: int, text: str):
        lines.append(f"\n{'#' * level} {text}\n")

    def kv(label: str, value: Any):
        v = _safe(value)
        if v != "—":
            lines.append(f"**{label}:** {v}")

    def kv_block(label: str, value: Any):
        """Key-value where value may be multi-line."""
        v = _safe(value)
        if v != "—":
            lines.append(f"\n**{label}:**\n{v}")

    def table(headers: List[str], rows: List[List[str]]):
        if not rows:
            return
        lines.append("")
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            cells = [c.replace("|", "\\|").replace("\n", " ") for c in row]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # ── Title ──
    title = _safe(p.get("usecase_title"), "Untitled")
    lines.append(f"# Handover Package: {title}\n")
    lines.append(f"**Status:** {_safe(export_data.get('status'), 'draft').title()}")
    lines.append(f"**Export Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    lines.append("---")

    # ── Section 1: Customer & Workshop ──
    heading(2, "1. Customer & Workshop Context")
    kv("Customer", p.get("customer"))
    kv("Industry", p.get("industry"))
    kv("Workshop Date", p.get("workshop_date"))
    kv("Workshop Format", p.get("workshop_format"))
    kv("Solution Advisor", p.get("advisor_name"))
    kv("Primary Contact", p.get("primary_contact"))
    kv("Technical Contact", p.get("tech_contact"))

    # ── Section 2: Use Case ──
    heading(2, "2. Use Case")
    kv("Title", p.get("usecase_title"))
    kv("Priority", p.get("priority"))
    kv_block("Problem Statement", p.get("problem_statement"))

    # ── Section 3: As-Is Process ──
    heading(2, "3. As-Is Process")
    kv("Process Name", p.get("process_name"))
    kv("Trigger / Frequency", p.get("process_freq"))
    kv("Volume", p.get("process_volume"))
    kv("Team / Roles", p.get("process_team"))
    kv("People Involved", p.get("process_people"))

    # ── Section 4: Data ──
    heading(2, "4. Data Landscape")
    kv_block("Data Input", p.get("data_input"))
    kv_block("Data Output", p.get("data_output"))
    kv_block("Data Usage", p.get("data_usage"))
    _render_md_table(lines, p, "data_sources", heading, table)

    # ── Section 5: Business Impact ──
    heading(2, "5. Business Impact")
    impact = p.get("business_impact") or {}
    if impact:
        headers = ["Dimension", "Current", "Target"]
        rows = []
        for key, label in IMPACT_DIMENSIONS.items():
            dim = impact.get(key, {})
            current = _safe(dim.get("current"), "")
            target = _safe(dim.get("target"), "")
            if current or target:
                rows.append([label, current, target])
        if rows:
            table(headers, rows)

    # ── Section 6: AI Capabilities ──
    heading(2, "6. AI Capabilities")
    caps = p.get("ai_capabilities") or []
    if caps:
        for cap_id in caps:
            label = AI_CAPABILITIES_MAP.get(cap_id, cap_id)
            lines.append(f"- {label}")
        lines.append("")
    else:
        lines.append("_No capabilities selected._\n")

    # ── Section 7: To-Be / Scope ──
    heading(2, "7. To-Be Process & Scope")
    kv_block("To-Be Narrative", p.get("tobe_narrative"))
    kv_block("In Scope", p.get("in_scope"))
    kv_block("Out of Scope", p.get("out_scope"))
    kv_block("Definition of Done", p.get("done_definition"))

    # ── Section 8: Business Rules ──
    heading(2, "8. Business Rules & KPIs")
    for tbl_key in ("business_rules", "thresholds", "approvals", "edge_cases", "kpis"):
        _render_md_table(lines, p, tbl_key, heading, table)

    # ── Section 9: Technical Landscape ──
    heading(2, "9. Technical Landscape")
    for tbl_key in ("tech_components", "integrations", "contacts"):
        _render_md_table(lines, p, tbl_key, heading, table)

    # ── Section 10: Checklist ──
    heading(2, "10. Handover Checklist")
    checklist = p.get("checklist") or []
    if checklist:
        checked_keys = {item["key"] for item in checklist if item.get("checked")}

        lines.append("\n**Solution Advisor — Before Submitting:**\n")
        advisor_keys = list(CHECKLIST_ITEMS.keys())[:18]
        for key in advisor_keys:
            label = CHECKLIST_ITEMS.get(key, key)
            mark = "✓" if key in checked_keys else "✗"
            lines.append(f"- [{mark}] {label}")

        lines.append("\n**Build Team — On Receipt:**\n")
        team_keys = list(CHECKLIST_ITEMS.keys())[18:]
        for key in team_keys:
            label = CHECKLIST_ITEMS.get(key, key)
            mark = "✓" if key in checked_keys else "✗"
            lines.append(f"- [{mark}] {label}")
        lines.append("")

    kv_block("Open Notes", p.get("open_notes"))

    # ── Metadata ──
    lines.append("\n---\n")
    heading(2, "Export Metadata")
    kv("Handover ID", export_data.get("id"))
    kv("Status", export_data.get("status"))
    kv("Created", _format_timestamp(export_data.get("created_at")))
    kv("Last Updated", _format_timestamp(export_data.get("updated_at")))
    kv("Submitted", _format_timestamp(export_data.get("submitted_at")))
    kv("Linked Conversation", export_data.get("linked_conversation_id"))

    # Context sources
    sources = export_data.get("context_sources") or []
    if sources:
        lines.append(f"\n**Context Sources:** {len(sources)} linked")
        for s in sources:
            stype = _safe(s.get("source_type"), "unknown")
            sref = _safe(s.get("source_ref"), "")
            lines.append(f"- {stype}: {sref}")

    # Files
    files = export_data.get("files") or []
    if files:
        lines.append(f"\n**Attached Files:** {len(files)}")
        for f in files:
            fname = _safe(f.get("filename"), "?")
            zone = _safe(f.get("zone"), "")
            size = f.get("size", 0)
            size_str = f"{size / 1024:.0f} KB" if size else "—"
            lines.append(f"- {zone}/{fname} ({size_str})")

    return "\n".join(lines) + "\n"


def _render_md_table(lines, payload, table_key, heading_fn, table_fn):
    """Render a dynamic table in markdown if it has data."""
    schema = TABLE_SCHEMAS.get(table_key)
    keys = TABLE_KEYS.get(table_key)
    if not schema or not keys:
        return
    title, headers = schema
    rows_data = payload.get(table_key) or []
    if not _has_table_data(rows_data, keys):
        return
    lines.append(f"\n### {title}\n")
    rows = []
    for row in rows_data:
        vals = [_safe(row.get(k), "") for k in keys]
        if any(vals):
            rows.append(vals)
    table_fn(headers, rows)


# ── PDF Renderer ─────────────────────────────────────────────────────────

# SAP-inspired colors
_BRAND_BLUE = (0, 112, 242)      # #0070F2
_DARK_TEXT = (51, 51, 51)         # #333333
_GRAY_TEXT = (102, 102, 102)      # #666666
_HEADER_BG = (0, 82, 178)        # Darker blue for section headers
_TABLE_HEADER_BG = (235, 237, 240)  # #EBEDF0
_TABLE_BORDER = (210, 210, 210)   # #D2D2D2
_TABLE_ALT_ROW = (248, 249, 250)  # #F8F9FA
_WHITE = (255, 255, 255)


def render_pdf(export_data: dict) -> bytes:
    """Render handover export data as a PDF document. Returns raw bytes."""
    from fpdf import FPDF

    p = export_data.get("payload") or {}

    class HandoverPDF(FPDF):
        """Custom PDF with header/footer."""

        def __init__(self):
            super().__init__(orientation="P", unit="mm", format="A4")
            self.set_auto_page_break(auto=True, margin=20)
            self._title_text = _safe_pdf(p.get("usecase_title"), "Untitled")

        def header(self):
            if self.page_no() == 1:
                return  # Cover page has its own header
            self.set_font("Helvetica", "B", 8)
            self.set_text_color(*_GRAY_TEXT)
            self.cell(0, 6, self._title_text[:80], align="L")
            self.ln(8)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "", 7)
            self.set_text_color(*_GRAY_TEXT)
            self.cell(0, 8, f"Page {self.page_no()}/{{nb}}", align="C")

    pdf = HandoverPDF()
    pdf.alias_nb_pages()

    # ── Cover Page ──
    pdf.add_page()
    # Blue band
    pdf.set_fill_color(*_BRAND_BLUE)
    pdf.rect(0, 0, 210, 22, "F")
    pdf.set_y(28)

    # Title
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(*_DARK_TEXT)
    title = _safe_pdf(p.get("usecase_title"), "Untitled")
    pdf.multi_cell(0, 12, "Handover Package", align="L")
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(*_BRAND_BLUE)
    pdf.multi_cell(0, 9, title, align="L")
    pdf.ln(8)

    # Cover details
    pdf.set_text_color(*_DARK_TEXT)
    _pdf_kv_line(pdf, "Customer", p.get("customer"))
    _pdf_kv_line(pdf, "Industry", p.get("industry"))
    _pdf_kv_line(pdf, "Priority", p.get("priority"))
    _pdf_kv_line(pdf, "Status", _safe_pdf(export_data.get("status"), "draft").title())
    pdf.ln(4)
    _pdf_kv_line(pdf, "Solution Advisor", p.get("advisor_name"))
    _pdf_kv_line(pdf, "Workshop Date", p.get("workshop_date"))
    _pdf_kv_line(pdf, "Workshop Format", p.get("workshop_format"))

    # Footer on cover
    pdf.set_y(-30)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*_GRAY_TEXT)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pdf.cell(0, 6, f"Generated from AI4U Ideation Agent -- {now}", align="L")

    # ── Section 1: Customer ──
    pdf.add_page()
    _pdf_section_header(pdf, "1", "Customer & Workshop Context")
    _pdf_kv_grid(pdf, [
        ("Customer", p.get("customer")),
        ("Industry", p.get("industry")),
        ("Workshop Date", p.get("workshop_date")),
        ("Workshop Format", p.get("workshop_format")),
        ("Solution Advisor", p.get("advisor_name")),
        ("Primary Contact", p.get("primary_contact")),
        ("Technical Contact", p.get("tech_contact")),
    ])

    # ── Section 2: Use Case ──
    _pdf_section_header(pdf, "2", "Use Case")
    _pdf_kv_grid(pdf, [
        ("Title", p.get("usecase_title")),
        ("Priority", p.get("priority")),
    ])
    _pdf_text_block(pdf, "Problem Statement", p.get("problem_statement"))

    # ── Section 3: As-Is Process ──
    _pdf_section_header(pdf, "3", "As-Is Process")
    _pdf_kv_grid(pdf, [
        ("Process Name", p.get("process_name")),
        ("Trigger / Frequency", p.get("process_freq")),
        ("Volume", p.get("process_volume")),
        ("Team / Roles", p.get("process_team")),
        ("People Involved", p.get("process_people")),
    ])

    # ── Section 4: Data ──
    _pdf_section_header(pdf, "4", "Data Landscape")
    _pdf_text_block(pdf, "Data Input", p.get("data_input"))
    _pdf_text_block(pdf, "Data Output", p.get("data_output"))
    _pdf_text_block(pdf, "Data Usage", p.get("data_usage"))
    _pdf_dynamic_table(pdf, p, "data_sources")

    # ── Section 5: Business Impact ──
    _pdf_section_header(pdf, "5", "Business Impact")
    _pdf_impact_table(pdf, p.get("business_impact"))

    # ── Section 6: AI Capabilities ──
    _pdf_section_header(pdf, "6", "AI Capabilities")
    caps = p.get("ai_capabilities") or []
    if caps:
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*_DARK_TEXT)
        for cap_id in caps:
            label = _latin1(AI_CAPABILITIES_MAP.get(cap_id, cap_id))
            pdf.cell(5)
            pdf.cell(0, 7, f"-  {label}", ln=True)
        pdf.ln(4)
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(*_GRAY_TEXT)
        pdf.cell(0, 7, "No capabilities selected.", ln=True)
        pdf.ln(4)

    # ── Section 7: To-Be / Scope ──
    _pdf_section_header(pdf, "7", "To-Be Process & Scope")
    _pdf_text_block(pdf, "To-Be Narrative", p.get("tobe_narrative"))
    _pdf_text_block(pdf, "In Scope", p.get("in_scope"))
    _pdf_text_block(pdf, "Out of Scope", p.get("out_scope"))
    _pdf_text_block(pdf, "Definition of Done", p.get("done_definition"))

    # ── Section 8: Business Rules ──
    _pdf_section_header(pdf, "8", "Business Rules & KPIs")
    for tbl_key in ("business_rules", "thresholds", "approvals", "edge_cases", "kpis"):
        _pdf_dynamic_table(pdf, p, tbl_key)

    # ── Section 9: Technical Landscape ──
    _pdf_section_header(pdf, "9", "Technical Landscape")
    for tbl_key in ("tech_components", "integrations", "contacts"):
        _pdf_dynamic_table(pdf, p, tbl_key)

    # ── Section 10: Checklist ──
    _pdf_section_header(pdf, "10", "Handover Checklist")
    _pdf_checklist(pdf, p.get("checklist"))
    _pdf_text_block(pdf, "Open Notes", p.get("open_notes"))

    # ── Export Metadata ──
    _pdf_section_header(pdf, "i", "Export Metadata")
    _pdf_kv_grid(pdf, [
        ("Handover ID", export_data.get("id")),
        ("Status", export_data.get("status")),
        ("Created", _format_timestamp(export_data.get("created_at"))),
        ("Last Updated", _format_timestamp(export_data.get("updated_at"))),
        ("Submitted", _format_timestamp(export_data.get("submitted_at"))),
        ("Linked Conversation", export_data.get("linked_conversation_id")),
    ])

    # Context sources
    sources = export_data.get("context_sources") or []
    if sources:
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*_DARK_TEXT)
        pdf.cell(0, 7, f"Context Sources ({len(sources)})", ln=True)
        pdf.set_font("Helvetica", "", 9)
        for s in sources:
            stype = _safe_pdf(s.get("source_type"), "unknown")
            sref = _safe_pdf(s.get("source_ref"), "")
            pdf.cell(5)
            pdf.cell(0, 6, f"-  {stype}: {sref}", ln=True)
        pdf.ln(2)

    # Files
    files = export_data.get("files") or []
    if files:
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 7, f"Attached Files ({len(files)})", ln=True)
        pdf.set_font("Helvetica", "", 9)
        for f in files:
            fname = _safe_pdf(f.get("filename"), "?")
            zone = _safe_pdf(f.get("zone"), "")
            size = f.get("size", 0)
            size_str = f"{size / 1024:.0f} KB" if size else ""
            pdf.cell(5)
            pdf.cell(0, 6, f"-  {zone}/{fname} ({size_str})", ln=True)

    return bytes(pdf.output())


# ── PDF Helpers ───────────────────────────────────────────────────────────

def _pdf_kv_line(pdf, label: str, value: Any):
    """Render a single label: value line on the cover page."""
    v = _safe_pdf(value)
    if v == "-":
        return
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*_GRAY_TEXT)
    pdf.cell(45, 8, f"{label}:", align="L")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*_DARK_TEXT)
    pdf.cell(0, 8, v, ln=True)


def _pdf_section_header(pdf, number: str, title: str):
    """Render a section header bar."""
    # Check if we need a new page (at least 30mm space needed)
    if pdf.get_y() > 250:
        pdf.add_page()

    pdf.ln(6)
    pdf.set_fill_color(*_HEADER_BG)
    pdf.set_text_color(*_WHITE)
    pdf.set_font("Helvetica", "B", 11)
    y = pdf.get_y()
    pdf.rect(pdf.l_margin, y, 210 - pdf.l_margin - pdf.r_margin, 8, "F")
    pdf.set_xy(pdf.l_margin + 3, y)
    pdf.cell(0, 8, f"  {number}. {title}" if number.isdigit() else f"  {title}")
    pdf.set_y(y + 10)
    pdf.set_text_color(*_DARK_TEXT)


def _pdf_kv_grid(pdf, pairs: List[tuple]):
    """Render key-value pairs in a compact grid."""
    filtered = [(k, v) for k, v in pairs if _safe(v) != "\u2014"]
    if not filtered:
        return
    col_w = (210 - pdf.l_margin - pdf.r_margin) / 2
    for i, (label, value) in enumerate(filtered):
        v = _safe_pdf(value)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*_GRAY_TEXT)
        pdf.cell(35, 6, f"{label}:")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*_DARK_TEXT)
        remaining = col_w - 35
        pdf.cell(remaining, 6, v[:60])
        # Two columns: second item on same line
        if i % 2 == 0 and i + 1 < len(filtered):
            pass  # Stay on same line
        else:
            pdf.ln()
    pdf.ln(4)


def _pdf_text_block(pdf, label: str, value: Any):
    """Render a label + multi-line text block."""
    v = _safe_pdf(value)
    if v == "-":
        return
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*_GRAY_TEXT)
    pdf.cell(0, 6, f"{label}:", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*_DARK_TEXT)
    pdf.multi_cell(0, 5, v)
    pdf.ln(3)


def _pdf_dynamic_table(pdf, payload: dict, table_key: str):
    """Render a dynamic table if it has data."""
    schema = TABLE_SCHEMAS.get(table_key)
    keys = TABLE_KEYS.get(table_key)
    if not schema or not keys:
        return
    title, headers = schema
    rows_data = payload.get(table_key) or []
    if not _has_table_data(rows_data, keys):
        return

    # Check space — if less than 40mm, new page
    if pdf.get_y() > 230:
        pdf.add_page()

    # Sub-heading
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*_DARK_TEXT)
    pdf.cell(0, 7, title, ln=True)
    pdf.ln(1)

    # Calculate column widths (distribute evenly within page width)
    page_w = 210 - pdf.l_margin - pdf.r_margin
    col_count = len(headers)
    col_w = page_w / col_count

    # Header row
    pdf.set_fill_color(*_TABLE_HEADER_BG)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*_DARK_TEXT)
    for h in headers:
        pdf.cell(col_w, 7, h[:20], border=1, fill=True)
    pdf.ln()

    # Data rows
    pdf.set_font("Helvetica", "", 8)
    for idx, row in enumerate(rows_data):
        vals = [_safe_pdf(row.get(k), "") for k in keys]
        if not any(vals):
            continue
        # Alternating row color
        if idx % 2 == 1:
            pdf.set_fill_color(*_TABLE_ALT_ROW)
            fill = True
        else:
            fill = False

        # Check for page break
        if pdf.get_y() > 270:
            pdf.add_page()

        for val in vals:
            pdf.cell(col_w, 6, val[:30], border=1, fill=fill)
        pdf.ln()

    pdf.ln(4)


def _pdf_impact_table(pdf, impact: Optional[dict]):
    """Render the business impact table."""
    if not impact:
        return

    headers = ["Dimension", "Current", "Target"]
    rows = []
    for key, label in IMPACT_DIMENSIONS.items():
        dim = impact.get(key, {})
        if not isinstance(dim, dict):
            continue
        current = _safe_pdf(dim.get("current"), "")
        target = _safe_pdf(dim.get("target"), "")
        if current or target:
            rows.append((label, current, target))

    if not rows:
        return

    page_w = 210 - pdf.l_margin - pdf.r_margin
    col_widths = [page_w * 0.30, page_w * 0.35, page_w * 0.35]

    # Header
    pdf.set_fill_color(*_TABLE_HEADER_BG)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*_DARK_TEXT)
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 7, h, border=1, fill=True)
    pdf.ln()

    # Rows
    pdf.set_font("Helvetica", "", 8)
    for idx, (label, current, target) in enumerate(rows):
        if idx % 2 == 1:
            pdf.set_fill_color(*_TABLE_ALT_ROW)
            fill = True
        else:
            fill = False
        pdf.cell(col_widths[0], 6, label, border=1, fill=fill)
        pdf.cell(col_widths[1], 6, current[:50], border=1, fill=fill)
        pdf.cell(col_widths[2], 6, target[:50], border=1, fill=fill)
        pdf.ln()

    pdf.ln(4)


def _pdf_checklist(pdf, checklist: Optional[List[dict]]):
    """Render the checklist with check/cross symbols."""
    if not checklist:
        return

    checked_keys = {item["key"] for item in checklist if item.get("checked")}
    all_keys = list(CHECKLIST_ITEMS.keys())
    advisor_keys = all_keys[:18]
    team_keys = all_keys[18:]

    def render_group(group_title: str, keys: List[str]):
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*_DARK_TEXT)
        pdf.cell(0, 7, group_title, ln=True)
        pdf.set_font("Helvetica", "", 9)
        for key in keys:
            label = _latin1(CHECKLIST_ITEMS.get(key, key))
            if key in checked_keys:
                mark = "[x]"
                pdf.set_text_color(16, 124, 16)  # Green
            else:
                mark = "[ ]"
                pdf.set_text_color(187, 0, 0)  # Red
            pdf.cell(5)
            pdf.cell(8, 6, mark)
            pdf.set_text_color(*_DARK_TEXT)
            pdf.cell(0, 6, label, ln=True)
        pdf.ln(2)

    render_group("Solution Advisor -- Before Submitting", advisor_keys)
    render_group("Build Team -- On Receipt", team_keys)
