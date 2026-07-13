"""Orchestrator for the full optimizer report.

Builds deterministic markdown sections from a ``ReportContext``, optionally
calls an LLM for narrative analysis (executive summary + constraint analysis),
then triggers PDF and DOCX rendering.

Entry point
-----------
``generate_full_report(metadata, output_dir, generate_narrative=True)``
    Returns ``(md_path, pdf_path, docx_path)``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .labels import RULE_DESCRIPTIONS, humanize_identifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(value: float) -> str:
    return f"{value:,.2f}"


def _pct(value: float) -> str:
    return f"{value:.1f}%"


# ---------------------------------------------------------------------------
# Deterministic section builders
# ---------------------------------------------------------------------------

def _build_header_section(ctx) -> str:
    mode_label = "Multi-week" if ctx.planning_mode == "multi_week" else "Single-week"
    lines = [
        f"# Optimizer Run Report -- Cohort {ctx.cohort}",
        "",
        f"**Planning mode:** {mode_label}",
    ]
    if ctx.planning_mode == "multi_week":
        lines.append(f"**Horizon:** {ctx.horizon_weeks} weeks (start {ctx.planning_start_date})")
    lines.append(f"**Solver status:** {ctx.solver_status}")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    return "\n".join(lines)


def _build_selection_metrics_table(ctx) -> str:
    lines = [
        "## Selection Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Candidates (post-rule, optimizer input) | {ctx.candidate_count} |",
        f"| Selected | {ctx.selected_count} |",
        f"| Rule-excluded (filtered before optimizer) | {ctx.excluded_count} |",
        f"| Optimizer not-selected (from candidates) | {ctx.not_selected_count} |",
        f"| Candidate amount | {_fmt(ctx.candidate_amount)} |",
        f"| Selected amount | {_fmt(ctx.selected_amount)} |",
        f"| Selection ratio | {_pct(ctx.selection_ratio_pct)} |",
        f"| Top-3 customer concentration | {_pct(ctx.top3_concentration_pct)} |",
        "",
        "*Candidates are post-rule eligible invoices. Rule-excluded invoices are removed from the original input before optimization.*",
        "",
    ]
    return "\n".join(lines)


def _build_selected_invoices_table(ctx) -> str:
    if not ctx.selected_invoices:
        return ""

    lines = [
        "## Selected Invoices",
        "",
        "| Invoice Reference | Customer | Company Code | Purchase Price | Currency | Due Date | Planned Week | Lifetime (wk) |",
        "|-------------------|----------|--------------|----------------|----------|----------|--------------|---------------|",
    ]

    for inv in ctx.selected_invoices:
        ref = inv.get("Invoice Reference", "")
        cust = inv.get("Customer", "")
        cc = inv.get("Company Code", "")
        price = inv.get("Purchase Price", "")
        price_str = _fmt(float(price)) if price != "" else ""
        cur = inv.get("Currency", "")
        due = str(inv.get("Due Date", ""))[:10]
        week = inv.get("planned_week_start_iso", "")
        lt = inv.get("expected_lifetime_weeks", "")
        lines.append(f"| {ref} | {cust} | {cc} | {price_str} | {cur} | {due} | {week} | {lt} |")

    if ctx.total_selected_invoices > len(ctx.selected_invoices):
        lines.append("")
        remaining = ctx.total_selected_invoices - len(ctx.selected_invoices)
        lines.append(f"*{remaining} additional selected invoices omitted -- see selected.xlsx for the full list.*")

    lines.append("")
    return "\n".join(lines)


def _build_excluded_invoices_section(ctx) -> str:
    parts: list[str] = []

    if ctx.exclusion_summary:
        header = [
            "## Excluded Invoices",
            "",
            "### Exclusion Summary",
            "",
            "| Reason | Stage | Count | Total Amount |",
            "|--------|-------|-------|--------------|",
        ]
        for row in ctx.exclusion_summary:
            amt = _fmt(row["total_amount"]) if "total_amount" in row else ""
            stage = row.get("stage", "")
            reason = humanize_identifier(str(row.get("reason", "")))
            header.append(f"| {reason} | {stage} | {row['count']} | {amt} |")
        header.append("")
        parts.append("\n".join(header))

    if ctx.deferred_reasons:
        dr_lines = ["### Deferred Reasons (Optimizer)", ""]
        for reason, count in ctx.deferred_reasons.items():
            dr_lines.append(f"- **{humanize_identifier(reason)}**: {count} invoice(s)")
        dr_lines.append("")
        parts.append("\n".join(dr_lines))

    if ctx.excluded_invoices:
        detail = [
            "### Excluded Invoice Details",
            "",
            "| Invoice Reference | Customer | Purchase Price | Stage | Reason | Detail |",
            "|-------------------|----------|----------------|-------|--------|--------|",
        ]
        for inv in ctx.excluded_invoices:
            ref = inv.get("Invoice Reference", "")
            cust = inv.get("Customer", "")
            price = inv.get("Purchase Price", "")
            price_str = _fmt(float(price)) if price != "" else ""
            stage = inv.get("excluded_stage", "")
            reason = humanize_identifier(str(inv.get("excluded_reason", "")))
            detail_text = inv.get("excluded_reason_detail", "")
            detail.append(f"| {ref} | {cust} | {price_str} | {stage} | {reason} | {detail_text} |")

        if ctx.total_excluded_invoices > len(ctx.excluded_invoices):
            detail.append("")
            remaining = ctx.total_excluded_invoices - len(ctx.excluded_invoices)
            detail.append(f"*{remaining} additional excluded invoices omitted -- see excluded.xlsx.*")
        detail.append("")
        parts.append("\n".join(detail))

    if not parts and not ctx.exclusion_summary and not ctx.excluded_invoices and not ctx.deferred_reasons:
        return ""

    return "\n".join(parts)


def _build_weekly_schedule(ctx) -> str:
    if ctx.planning_mode != "multi_week" or not ctx.weekly_schedule:
        return ""

    lines = [
        "## Weekly Schedule",
        "",
        "| Week | Start Date | Invoices | Total Amount |",
        "|------|------------|----------|--------------|",
    ]

    total_inv = 0
    total_amt = 0.0
    for ws in ctx.weekly_schedule:
        lines.append(
            f"| {ws['week_index']} | {ws['week_start']} | {ws['invoice_count']} | {_fmt(ws['total_amount'])} |"
        )
        total_inv += ws["invoice_count"]
        total_amt += ws["total_amount"]

    lines.append(f"| **Total** | | **{total_inv}** | **{_fmt(total_amt)}** |")
    lines.append("")
    return "\n".join(lines)


def _build_facility_utilization(ctx) -> str:
    if not ctx.facility_utilization:
        return ""

    lines = [
        "## Facility Utilization",
        "",
        "| Facility | Week | New | Base | Total | Limit | Util % |",
        "|----------|------|-----|------|-------|-------|--------|",
    ]

    rows = sorted(
        ctx.facility_utilization,
        key=lambda row: (str(row.get("entity_id", "")), str(row.get("week", ""))),
    )
    for row in rows:
        lines.append(
            f"| {row['entity_id']} | {row['week']} | {_fmt(row['used_new'])} | "
            f"{_fmt(row['used_base'])} | {_fmt(row['used_total'])} | "
            f"{_fmt(row['limit'])} | {_pct(row['utilization_pct'])} |"
        )
    lines.append("")
    lines.append("*Only constrained facilities (limit > 0) are shown.*")
    lines.append("")
    return "\n".join(lines)


def _build_customer_concentration(ctx) -> str:
    if not ctx.top_customers:
        return ""

    lines = [
        "## Customer Concentration",
        "",
        "| Customer | Selected Amount | Share % |",
        "|----------|-----------------|---------|",
    ]

    for row in ctx.top_customers:
        lines.append(
            f"| {row['customer']} | {_fmt(row['selected_amount'])} | {_pct(row['share_pct'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def _build_rule_funnel(ctx) -> str:
    if not ctx.rule_summaries:
        return ""

    lines = [
        "## Rule Funnel",
        "",
        "| Rule Name | Type | Input | Output | Excluded | Description |",
        "|-----------|------|-------|--------|----------|-------------|",
    ]

    for rs in ctx.rule_summaries:
        raw_name = rs.get("rule_name", "")
        display_name = humanize_identifier(raw_name)
        rule_type = humanize_identifier(rs.get("rule_type", ""))
        desc = RULE_DESCRIPTIONS.get(raw_name, "")
        lines.append(
            f"| {display_name} | {rule_type} | "
            f"{rs.get('input_rows', 0)} | {rs.get('output_rows', 0)} | "
            f"{rs.get('excluded_rows', 0)} | {desc} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM narrative generation
# ---------------------------------------------------------------------------

def _generate_narrative(ctx) -> Optional[Tuple[str, str]]:
    """Call the LLM for executive summary and constraint analysis.

    Returns ``(executive_summary_md, constraint_analysis_md)`` or ``None``
    if the call fails.
    """
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from app.a2a.common import make_llm
        from .report_prompt import OPTIMIZER_REPORT_SYSTEM_PROMPT, build_llm_user_prompt

        llm = make_llm(model_name="gpt-4.1", temperature=0.2)
        user_prompt = build_llm_user_prompt(ctx)

        response = llm.invoke([
            SystemMessage(content=OPTIMIZER_REPORT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])

        text = response.content if hasattr(response, "content") else str(response)

        # Split response into Section A and Section B.  The LLM is instructed
        # to label them but we tolerate missing labels gracefully.
        section_a = text
        section_b = ""

        for marker in ["Section B", "## Section B", "**Section B"]:
            idx = text.find(marker)
            if idx > 0:
                section_a = text[:idx].strip()
                section_b = text[idx:].strip()
                break

        # Strip any "Section A" header the LLM might have included
        for prefix in ["Section A:", "## Section A:", "**Section A:**", "## Section A", "Section A"]:
            if section_a.startswith(prefix):
                section_a = section_a[len(prefix):].strip()
                break

        for prefix in ["Section B:", "## Section B:", "**Section B:**", "## Section B", "Section B"]:
            if section_b.startswith(prefix):
                section_b = section_b[len(prefix):].strip()
                break

        return (section_a, section_b)

    except Exception as exc:
        logger.warning("LLM narrative generation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_full_report(
    metadata: Dict[str, Any],
    output_dir: Path,
    generate_narrative: bool = True,
    *,
    selected_df=None,
    excluded_df=None,
) -> Tuple[Path, Path, Path]:
    """Build the full optimizer report as Markdown, PDF, and DOCX.

    Args:
        metadata: The run_metadata dict from the pipeline.
        output_dir: Directory containing output Excel files and where the
            report files will be written.
        generate_narrative: If *True* (default), call the LLM for executive
            summary and constraint analysis sections.
        selected_df: Optional preloaded selected rows dataframe.
        excluded_df: Optional preloaded excluded rows dataframe.

    Returns:
        Tuple of ``(md_path, pdf_path, docx_path)``.
    """
    from .report_context import assemble_report_context
    from .markdown_to_pdf import markdown_to_pdf
    from .markdown_to_docx import markdown_to_docx

    output_dir = Path(output_dir)

    # 1. Assemble context
    ctx = assemble_report_context(
        metadata,
        output_dir,
        selected_df=selected_df,
        excluded_df=excluded_df,
    )

    # 2. Build deterministic sections
    sections: list[str] = []
    sections.append(_build_header_section(ctx))

    # 3. LLM narrative (executive summary placed early)
    narrative: Optional[Tuple[str, str]] = None
    if generate_narrative:
        narrative = _generate_narrative(ctx)

    if narrative:
        sections.append("## Executive Summary\n")
        sections.append(narrative[0])
        sections.append("")
    else:
        sections.append("## Executive Summary\n")
        sections.append(
            f"The optimizer processed **{ctx.candidate_count}** candidate invoices "
            f"and selected **{ctx.selected_count}** for a total of **{_fmt(ctx.selected_amount)}** "
            f"({_pct(ctx.selection_ratio_pct)} of the candidate pool). "
            f"Solver status: **{ctx.solver_status}**."
        )
        sections.append("")

    if narrative and narrative[1]:
        sections.append("## Constraint Analysis & Recommendations\n")
        sections.append(narrative[1])
        sections.append("")

    sections.append("---\n")
    sections.append(_build_selection_metrics_table(ctx))
    sections.append(_build_selected_invoices_table(ctx))

    excluded_section = _build_excluded_invoices_section(ctx)
    if excluded_section:
        sections.append(excluded_section)

    schedule = _build_weekly_schedule(ctx)
    if schedule:
        sections.append(schedule)

    sections.append(_build_facility_utilization(ctx))
    sections.append(_build_customer_concentration(ctx))
    sections.append(_build_rule_funnel(ctx))

    # 4. Assemble final markdown
    markdown = "\n".join(s for s in sections if s)

    # 5. Write outputs
    md_path = output_dir / "run_summary.md"
    pdf_path = output_dir / "run_summary.pdf"
    docx_path = output_dir / "run_summary.docx"

    md_path.write_text(markdown, encoding="utf-8")
    logger.info("Markdown report written to %s", md_path)

    markdown_to_pdf(markdown, pdf_path)
    markdown_to_docx(markdown, docx_path)

    return md_path, pdf_path, docx_path
