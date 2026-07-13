"""LLM prompt construction for optimizer report narrative sections.

The system prompt instructs the model to produce two clearly-delimited
markdown sections:

  * **Section A -- Executive Summary**: 3--5 bullet points summarising the
    optimisation outcome.
  * **Section B -- Constraint Analysis & Recommendations**: 2--4 paragraphs
    analysing which constraints are binding and what operational adjustments
    might improve future runs.

The ``build_llm_user_prompt`` helper serializes a ``ReportContext`` into a
compact textual summary that the LLM can reference without hallucination.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .labels import RULE_DESCRIPTIONS, humanize_identifier

if TYPE_CHECKING:
    from .report_context import ReportContext


OPTIMIZER_REPORT_SYSTEM_PROMPT = """\
You are a financial operations analyst writing sections of an optimizer run report.

You will receive structured data from an invoice-selection optimizer.  Your job
is to produce TWO clearly-labelled markdown sections.

## Section A: Executive Summary

Write 3-5 bullet points that concisely summarise the run outcome:
- How many invoices were considered vs. selected
- Total amounts and the selection ratio
- Key constraints that limited selection (if any)
- Solver status and planning mode

Use **bold** for key figures (amounts, percentages, counts).

## Section B: Constraint Analysis & Recommendations

Write 2-4 paragraphs analysing:
- Which facility, customer, or group constraints are closest to (or at) their
  limits
- Why certain invoices were excluded or deferred.  When referencing exclusion
  rules, briefly explain what the rule does (a RULE DESCRIPTIONS section is
  provided in the data).
- What operational changes could improve the selection ratio in future runs
  (e.g. raising a limit, splitting customer exposure, adjusting the planning
  horizon)

## Rules

1. Use ONLY the data provided.  Never invent numbers.
2. Refer to entities by their IDs exactly as given.
3. Format currency amounts with commas and two decimals (e.g. 72,500.00).
4. Use **bold** markdown for key figures.
5. Do NOT add markdown headers -- the caller will wrap your output in the
   appropriate heading structure.
6. Keep language professional and concise.
7. Never use raw machine codes for rule/reason labels (e.g. CUSTOMER_CAP_BINDING);
   always use humanized labels (e.g. Customer Cap Binding).
"""


def build_llm_user_prompt(ctx: "ReportContext") -> str:
    """Serialize a ``ReportContext`` into structured text for the LLM."""
    lines: list[str] = []

    lines.append("=== RUN OVERVIEW ===")
    lines.append(f"Cohort: {ctx.cohort}")
    lines.append(f"Planning mode: {ctx.planning_mode}")
    lines.append(f"Horizon weeks: {ctx.horizon_weeks}")
    lines.append(f"Solver status: {ctx.solver_status}")
    lines.append("")

    lines.append("=== SELECTION METRICS ===")
    lines.append(f"Candidates (post-rule, optimizer input): {ctx.candidate_count}")
    lines.append(f"Selected: {ctx.selected_count}")
    lines.append(f"Rule-excluded (filtered before optimizer): {ctx.excluded_count}")
    lines.append(f"Optimizer not-selected (from candidates): {ctx.not_selected_count}")
    lines.append(f"Candidate amount: {ctx.candidate_amount:,.2f}")
    lines.append(f"Selected amount: {ctx.selected_amount:,.2f}")
    lines.append(f"Selection ratio: {ctx.selection_ratio_pct:.1f}%")
    lines.append(f"Top-3 customer concentration: {ctx.top3_concentration_pct:.1f}%")
    lines.append(
        "Note: candidates are post-rule eligible invoices, while rule-excluded "
        "counts invoices removed from the original input before optimization."
    )
    lines.append("")

    if ctx.rule_summaries:
        lines.append("=== RULE DESCRIPTIONS ===")
        for rs in ctx.rule_summaries:
            raw_name = rs.get("rule_name", "")
            desc = RULE_DESCRIPTIONS.get(raw_name, "")
            display = humanize_identifier(raw_name)
            excluded = rs.get("excluded_rows", 0)
            if desc:
                lines.append(f"- {display}: {desc} [excluded {excluded}]")
            else:
                lines.append(f"- {display}: [excluded {excluded}]")
        lines.append("")

    if ctx.exclusion_summary:
        lines.append("=== EXCLUSION SUMMARY ===")
        for row in ctx.exclusion_summary:
            amt = f" (total {row['total_amount']:,.2f})" if "total_amount" in row else ""
            reason = humanize_identifier(str(row.get("reason", "")))
            lines.append(f"- {reason}: {row['count']} invoice(s){amt}")
        lines.append("")

    if ctx.deferred_reasons:
        lines.append("=== DEFERRED REASONS ===")
        for reason, count in ctx.deferred_reasons.items():
            lines.append(f"- {humanize_identifier(reason)}: {count}")
        lines.append("")

    if ctx.binding_constraints:
        lines.append("=== BINDING CONSTRAINTS (>= 95% utilization) ===")
        for bc in ctx.binding_constraints:
            lines.append(
                f"- {bc['entity_type']} {bc['entity_id']}: "
                f"{bc['peak_utilization_pct']:.1f}% in week {bc['peak_week']} "
                f"(limit {bc['limit']:,.2f})"
            )
        lines.append("")

    if ctx.facility_utilization:
        lines.append("=== FACILITY UTILIZATION (sample) ===")
        for row in ctx.facility_utilization[:16]:
            lines.append(
                f"- {row['entity_id']} week {row['week']}: "
                f"new={row['used_new']:,.2f} base={row['used_base']:,.2f} "
                f"total={row['used_total']:,.2f} limit={row['limit']:,.2f} "
                f"util={row['utilization_pct']:.1f}%"
            )
        lines.append("")

    if ctx.customer_utilization:
        lines.append("=== CUSTOMER UTILIZATION (peak per customer) ===")
        for row in ctx.customer_utilization[:10]:
            lines.append(
                f"- {row['entity_id']}: {row['utilization_pct']:.1f}% "
                f"(total={row['used_total']:,.2f} / limit={row['limit']:,.2f})"
            )
        lines.append("")

    if ctx.weekly_schedule:
        lines.append("=== WEEKLY SCHEDULE ===")
        for ws in ctx.weekly_schedule:
            lines.append(
                f"- Week {ws['week_index']} ({ws['week_start']}): "
                f"{ws['invoice_count']} invoices, {ws['total_amount']:,.2f}"
            )
        lines.append("")

    if ctx.lifecycle_profile:
        lines.append("=== LIFECYCLE PROFILE ===")
        lp = ctx.lifecycle_profile
        lines.append(f"Total profiled rows: {lp.get('total_rows', 0)}")
        lines.append(f"Missing credit end %: {lp.get('missing_credit_end_pct', 0):.1f}%")
        lines.append(f"Repurchase %: {lp.get('repurchase_pct', 0):.1f}%")
        dur = lp.get("duration_days_stats") or {}
        if dur:
            lines.append(
                f"Duration days: mean={dur.get('mean', 0):.0f} "
                f"median={dur.get('median', 0):.0f} "
                f"p10={dur.get('p10', 0):.0f} p90={dur.get('p90', 0):.0f}"
            )
        lines.append("")

    lines.append("Please produce Section A and Section B now.")
    return "\n".join(lines)
