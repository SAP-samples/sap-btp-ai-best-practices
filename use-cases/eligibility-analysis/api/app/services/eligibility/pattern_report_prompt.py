"""LLM prompt construction for pattern analysis report narrative.

The system prompt instructs the model to produce two clearly-delimited
markdown sections:

  * **Section A -- Executive Summary**: 3-5 bullet points summarising
    the key findings.
  * **Section B -- Trend Interpretation & Recommendations**: 2-4 paragraphs
    analysing trends and providing prioritised recommendations.

The ``build_pattern_report_user_prompt`` helper serializes analysis
results into compact text that the LLM can reference without hallucination.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ...models.patterns import PatternFilters, PatternSummary


PATTERN_REPORT_SYSTEM_PROMPT = """\
You are a factoring risk analyst writing sections of an eligibility pattern analysis report.

You will receive structured data from an invoice eligibility pattern analyzer. Your job
is to produce TWO clearly-labelled markdown sections.

## Section A: Executive Summary

Write 3-5 bullet points that concisely summarise the key findings:
- Overall eligibility rate and total invoices analyzed
- Most significant patterns detected (by severity)
- Entities (sellers/debtors) with the highest risk exposure
- Quarter-over-quarter trends if available

Use **bold** for key figures.

## Section B: Trend Interpretation & Recommendations

Write 2-4 paragraphs analysing:
- What the non-eligibility trends reveal (seasonal patterns, worsening/improving)
- Which rule(s) are driving the most non-eligibility and why
- Prioritised, actionable recommendations ranked by potential impact
- Specific entities that should receive attention

## Rules

1. Use ONLY the data provided. Never invent numbers.
2. Format percentages to one decimal place (e.g. 42.3%).
3. Format currency amounts with commas and two decimals.
4. Use **bold** markdown for key figures.
5. Do NOT add markdown headers -- the caller wraps your output.
6. Keep language professional and concise.
"""


def build_pattern_report_user_prompt(
    filters: "PatternFilters",
    summary: "PatternSummary",
    profiles: list,
    trend_data: list,
) -> str:
    """Serialize pattern analysis results into structured text for the LLM."""
    lines: list[str] = []

    # Active filters
    lines.append("=== ACTIVE FILTERS ===")
    filter_parts = []
    if filters.seller_id:
        filter_parts.append(f"Seller: {filters.seller_id}")
    if filters.debtor_id:
        filter_parts.append(f"Debtor: {filters.debtor_id}")
    if filters.programa:
        filter_parts.append(f"Program: {filters.programa}")
    if filters.insurer_id:
        filter_parts.append(f"Insurer: {filters.insurer_id}")
    lines.append(", ".join(filter_parts) if filter_parts else "No filters (all data)")
    lines.append(f"Lookback: {filters.lookback_days} days")
    lines.append("")

    # Summary metrics
    lines.append("=== SUMMARY METRICS ===")
    lines.append(f"Total invoices analyzed: {summary.total_invoices_analyzed}")
    lines.append(f"Overall eligibility rate: {summary.overall_eligibility_rate:.1%}")
    lines.append(f"Total patterns detected: {summary.total_patterns}")
    lines.append(f"  High severity: {summary.high_severity}")
    lines.append(f"  Medium severity: {summary.medium_severity}")
    lines.append(f"  Low severity: {summary.low_severity}")
    if summary.analysis_window_start:
        lines.append(f"Analysis window: {summary.analysis_window_start} to {summary.analysis_window_end}")
    lines.append("")

    # Pattern alerts
    if summary.patterns:
        lines.append("=== PATTERN ALERTS ===")
        for p in summary.patterns:
            lines.append(
                f"- [{p.severity.value.upper()}] {p.pattern_type.value}: {p.title}"
            )
            if p.metrics:
                metrics_str = ", ".join(f"{k}={v}" for k, v in p.metrics.items())
                lines.append(f"  Metrics: {metrics_str}")
        lines.append("")

    # Debtor profiles (top 10)
    if profiles:
        lines.append("=== TOP DEBTOR PROFILES (by non-eligible count) ===")
        for p in profiles[:10]:
            debtor_label = getattr(p, "debtor_name", None) or getattr(p, "debtor_id", "?")
            seller_label = getattr(p, "seller_name", None) or getattr(p, "seller_id", "?")
            rate = getattr(p, "rejection_rate", 0)
            dom_rule = getattr(p, "dominant_rule", "-")
            amount = getattr(p, "total_amount_rejected", None)
            amt_str = f", amount={amount:,.2f}" if amount else ""
            lines.append(
                f"- {debtor_label} (seller: {seller_label}): "
                f"rate={rate:.1%}, dominant_rule={dom_rule}{amt_str}"
            )
        lines.append("")

    # Trend data
    if trend_data:
        lines.append("=== TREND DATA ===")
        for t in trend_data:
            period = t.get("period_start", "") if isinstance(t, dict) else getattr(t, "period_start", "")
            total = t.get("total_invoices", 0) if isinstance(t, dict) else getattr(t, "total_invoices", 0)
            rejected = t.get("rejected_invoices", 0) if isinstance(t, dict) else getattr(t, "rejected_invoices", 0)
            rate = t.get("rejection_rate", 0) if isinstance(t, dict) else getattr(t, "rejection_rate", 0)
            by_rule = t.get("rejection_by_rule", {}) if isinstance(t, dict) else getattr(t, "rejection_by_rule", {})
            top_rules = sorted(by_rule.items(), key=lambda x: x[1], reverse=True)[:3]
            rules_str = ", ".join(f"{r}={c}" for r, c in top_rules) if top_rules else "-"
            lines.append(
                f"- {period}: total={total}, rejected={rejected}, "
                f"rate={rate:.1%}, top_rules=[{rules_str}]"
            )
        lines.append("")

    lines.append("Please produce Section A and Section B now.")
    return "\n".join(lines)
