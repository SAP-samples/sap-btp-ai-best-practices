"""Orchestrator for the pattern analysis PDF report.

Builds deterministic markdown sections from analysis results, renders
matplotlib charts, optionally calls an LLM for narrative (executive summary
+ recommendations), then renders everything to PDF.

Entry point
-----------
``generate_pattern_report(filters)``
    Returns raw PDF bytes.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from ...models.patterns import PatternFilters

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

def _build_header(filters: "PatternFilters", summary) -> str:
    lines = [
        "# Eligibility Pattern Analysis Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    ]
    filter_parts = []
    if filters.seller_id:
        filter_parts.append(f"Seller: {filters.seller_id}")
    if filters.debtor_id:
        filter_parts.append(f"Debtor: {filters.debtor_id}")
    if filters.programa:
        filter_parts.append(f"Program: {filters.programa}")
    if filters.insurer_id:
        filter_parts.append(f"Insurer: {filters.insurer_id}")
    scope = ", ".join(filter_parts) if filter_parts else "All data"
    lines.append(f"**Scope:** {scope}")
    lines.append(f"**Lookback:** {filters.lookback_days} days")
    if summary.analysis_window_start and summary.analysis_window_end:
        lines.append(
            f"**Period:** {str(summary.analysis_window_start)[:10]} to "
            f"{str(summary.analysis_window_end)[:10]}"
        )
    lines.append("")
    return "\n".join(lines)


def _build_summary_metrics_table(summary) -> str:
    rate = summary.overall_eligibility_rate
    lines = [
        "## Summary Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Invoices Analyzed | {summary.total_invoices_analyzed} |",
        f"| Overall Eligibility Rate | {_pct(rate * 100)} |",
        f"| Total Patterns Detected | {summary.total_patterns} |",
        f"| High Severity | {summary.high_severity} |",
        f"| Medium Severity | {summary.medium_severity} |",
        f"| Low Severity | {summary.low_severity} |",
        "",
    ]
    return "\n".join(lines)


def _build_pattern_alerts_table(alerts: list) -> str:
    if not alerts:
        return ""
    lines = [
        "## Pattern Alerts",
        "",
        "| Severity | Type | Title | Description |",
        "|----------|------|-------|-------------|",
    ]
    for a in alerts:
        sev = a.severity.value.upper() if hasattr(a.severity, "value") else str(a.severity)
        ptype = a.pattern_type.value if hasattr(a.pattern_type, "value") else str(a.pattern_type)
        lines.append(f"| **{sev}** | {ptype} | {a.title} | {a.description[:120]}{'...' if len(a.description) > 120 else ''} |")
    lines.append("")
    return "\n".join(lines)


def _build_debtor_profiles_table(profiles: list) -> str:
    if not profiles:
        return ""
    lines = [
        "## Top Debtor Profiles",
        "",
        "| Debtor | Seller | Total | Non-Eligible | Rate | Dominant Rule | Amount at Risk |",
        "|--------|--------|-------|--------------|------|---------------|----------------|",
    ]
    for p in profiles[:15]:
        debtor = getattr(p, "debtor_name", None) or getattr(p, "debtor_id", "-")
        seller = getattr(p, "seller_name", None) or getattr(p, "seller_id", "-")
        rate = getattr(p, "rejection_rate", 0)
        dom = getattr(p, "dominant_rule", "-") or "-"
        amount = getattr(p, "total_amount_rejected", None)
        amt_str = _fmt(float(amount)) if amount else "-"
        total = getattr(p, "total_invoices", 0)
        rejected = getattr(p, "rejected_invoices", 0)
        lines.append(
            f"| {debtor} | {seller} | {total} | {rejected} | "
            f"{_pct(rate * 100)} | {dom} | {amt_str} |"
        )
    if len(profiles) > 15:
        lines.append(f"\n*{len(profiles) - 15} additional profiles omitted.*")
    lines.append("")
    return "\n".join(lines)


def _build_trend_data_table(trend_data: list) -> str:
    if not trend_data:
        return ""
    lines = [
        "## Trend Data",
        "",
        "| Period | Total | Non-Eligible | Rate |",
        "|--------|-------|--------------|------|",
    ]
    for t in trend_data:
        if isinstance(t, dict):
            period = str(t.get("period_start", ""))[:10]
            total = t.get("total_invoices", 0)
            rejected = t.get("rejected_invoices", 0)
            rate = t.get("rejection_rate", 0)
        else:
            period = str(getattr(t, "period_start", ""))[:10]
            total = getattr(t, "total_invoices", 0)
            rejected = getattr(t, "rejected_invoices", 0)
            rate = getattr(t, "rejection_rate", 0)
        lines.append(f"| {period} | {total} | {rejected} | {_pct(rate * 100)} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM narrative generation
# ---------------------------------------------------------------------------

def _generate_narrative(
    filters: "PatternFilters",
    summary,
    profiles: list,
    trend_data: list,
) -> Optional[Tuple[str, str]]:
    """Call the LLM for executive summary and recommendations.

    Returns ``(executive_summary_md, recommendations_md)`` or ``None``
    if the call fails.
    """
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from ...a2a.common import make_llm
        from .pattern_report_prompt import (
            PATTERN_REPORT_SYSTEM_PROMPT,
            build_pattern_report_user_prompt,
        )

        llm = make_llm(model_name="gpt-4.1", temperature=0.2)
        user_prompt = build_pattern_report_user_prompt(
            filters, summary, profiles, trend_data
        )

        response = llm.invoke([
            SystemMessage(content=PATTERN_REPORT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])

        text = response.content if hasattr(response, "content") else str(response)

        # Split response into Section A and Section B
        section_a = text
        section_b = ""

        for marker in ["Section B", "## Section B", "**Section B"]:
            idx = text.find(marker)
            if idx > 0:
                section_a = text[:idx].strip()
                section_b = text[idx:].strip()
                break

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

def generate_pattern_report(filters: "PatternFilters") -> bytes:
    """Generate a pattern analysis PDF report and return raw bytes.

    Args:
        filters: PatternFilters with all active filter dimensions.

    Returns:
        Raw PDF bytes.
    """
    from ...optimizer.report.markdown_to_pdf import markdown_to_pdf
    from .pattern_analyzer import PatternAnalyzer
    from .pattern_chart_renderer import render_rule_distribution_chart, render_trend_chart

    analyzer = PatternAnalyzer()
    tmp_dir = Path(tempfile.mkdtemp(prefix="pattern_report_"))

    try:
        # 1. Run analysis
        summary = analyzer.analyze_all(filters=filters)
        profiles = analyzer.get_debtor_profiles(filters=filters)

        auto_granularity = "quarter" if filters.lookback_days >= 180 else "week"
        trend_data = analyzer.get_rejection_trend(
            granularity=auto_granularity, filters=filters
        )

        # Serialize trend_data to dicts for chart renderer
        trend_dicts = [t.model_dump() for t in trend_data]

        # 2. Build deterministic sections
        sections: list[str] = []
        sections.append(_build_header(filters, summary))

        # 3. LLM narrative (executive summary placed early)
        narrative = _generate_narrative(filters, summary, profiles, trend_dicts)

        if narrative:
            sections.append("## Executive Summary\n")
            sections.append(narrative[0])
            sections.append("")
        else:
            sections.append("## Executive Summary\n")
            sections.append(
                f"The pattern analyzer processed **{summary.total_invoices_analyzed}** invoices "
                f"and detected **{summary.total_patterns}** patterns "
                f"({summary.high_severity} high, {summary.medium_severity} medium). "
                f"Overall eligibility rate: **{_pct(summary.overall_eligibility_rate * 100)}**."
            )
            sections.append("")

        if narrative and narrative[1]:
            sections.append("## Recommendations\n")
            sections.append(narrative[1])
            sections.append("")

        sections.append("---\n")
        sections.append(_build_summary_metrics_table(summary))
        sections.append(_build_pattern_alerts_table(summary.patterns))

        # 4. Render charts
        trend_chart_path = render_trend_chart(trend_dicts, tmp_dir / "trend.png")
        if trend_chart_path:
            sections.append(f"![Non-Eligibility Trend]({trend_chart_path})\n")

        rule_chart_path = render_rule_distribution_chart(trend_dicts, tmp_dir / "rule_dist.png")
        if rule_chart_path:
            sections.append(f"![Rule Distribution]({rule_chart_path})\n")

        sections.append(_build_debtor_profiles_table(profiles))
        sections.append(_build_trend_data_table(trend_dicts))

        # 5. Assemble markdown and render to PDF
        markdown = "\n".join(s for s in sections if s)
        pdf_path = tmp_dir / "pattern_report.pdf"
        markdown_to_pdf(markdown, pdf_path)

        # 6. Read and return bytes
        return pdf_path.read_bytes()

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
