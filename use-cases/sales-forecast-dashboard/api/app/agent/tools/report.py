"""
Report generation tool for the forecasting agent.

This tool compiles forecasting session insights into a PDF report using an
LLM-based approach. Results are automatically stored by analysis tools and
read from session state - no need for the agent to pass data explicitly.

Per Agent_plan.md: Export Tools
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from ..common import make_llm
from ..session import get_session


# ============================================================================
# Report System Prompt
# ============================================================================

REPORT_SYSTEM_PROMPT = """You are a business analyst generating a PDF report that summarizes a forecasting conversation.

## Your Task

Analyze the CONVERSATION HISTORY to understand:
1. What the user originally asked for
2. What analyses were performed
3. What data is available

Then generate a report that ONLY includes sections relevant to what was actually discussed.

## Available Sections (include ONLY if relevant to the conversation)

| Section | Include When... |
|---------|-----------------|
| Executive Summary | Always include |
| Business Health Overview | Diagnostic analysis was performed (health metrics, YoY trends) |
| Regional Performance | DMA-level diagnostics were discussed |
| Store Performance | Store-level diagnostics or rankings were discussed |
| Declining Stores Analysis | Declining stores were identified and analyzed |
| Top/Bottom Performers | Performance rankings were discussed |
| Forecast Overview | A forecast was run |
| Week-by-Week Breakdown | A forecast was run (always include with forecast) |
| Forecast vs Previous Year | User asked to compare with last year AND yoy_actuals data exists |
| What-If Scenario Comparison | User created modified scenarios (scenario_comparison data exists with delta != 0) |
| SHAP Driver Analysis | User asked for explanation of changes (explanation data exists) |
| Sensitivity Analysis | User ran sensitivity analysis (sensitivity data exists) |
| Charts | Plot files were generated (ALWAYS include if plots exist) |
| Recommendations | Based on analyses performed |

## CONVERSATION-ONLY MODE

When no structured data (forecast_results, scenario_comparison, etc.) is provided but conversation history exists:
1. Extract ALL relevant findings from the ASSISTANT responses in the conversation
2. Include specific numbers, percentages, and store/DMA names mentioned
3. Organize findings into appropriate sections based on what was discussed
4. The conversation IS the data source - extract everything the assistant reported

## CRITICAL RULES

1. **No fake What-If comparisons**: Do NOT show "What-If vs Baseline" if scenario_comparison is missing or shows delta = $0. This means no scenarios were created.

2. **YoY = Forecast vs Previous Year Actuals**: If yoy_actuals data exists, show a comparison of forecast predictions against actual sales from the same period last year. Use columns like "2025 Forecast" vs "2024 Actual".

3. **Charts are mandatory**: If plot_files are listed, you MUST reference them in the report and they will be embedded at the end.

4. **Use ONLY provided data**: Never invent numbers. If data is missing, omit that section entirely.

5. **Conversation context matters**: Read the conversation to understand the user's intent and tailor the report accordingly.

## FEATURE NAME FORMATTING

When displaying feature names, use these human-readable names:

| Technical Name | Display Name |
|----------------|--------------|
| pct_primary_financing_roll_mean_4 | Primary Financing % |
| pct_secondary_financing_roll_mean_4 | Secondary Financing % |
| pct_tertiary_financing_roll_mean_4 | Tertiary Financing % |
| staffing_unique_associates_roll_mean_4 | Unique Associates |
| staffing_hours_roll_mean_4 | Staffing Hours |
| pct_omni_channel_roll_mean_4 | Omni-Channel % |
| pct_value_product_roll_mean_4 | Value Product % |
| pct_premium_product_roll_mean_4 | Premium Product % |
| pct_white_glove_roll_mean_4 | White Glove % |
| brand_awareness_dma_roll_mean_4 | Brand Awareness |
| brand_consideration_dma_roll_mean_4 | Brand Consideration |
| cannibalization_pressure | Cannibalization Pressure |
| min_dist_new_store_km | Distance to Nearest New Store |
| num_new_stores_within_10mi_last_52wk | New Stores (10mi) |
| num_new_stores_within_20mi_last_52wk | New Stores (20mi) |
| weeks_since_open | Store Age (Weeks) |
| merchandising_sf | Store Size (sq ft) |
| is_outlet | Outlet Store |
| is_comp_store | Comp Store |
| is_new_store | New Store |

Never show raw technical names. Convert unknown features to Title Case.

## SHAP VALUE INTERPRETATION (CRITICAL)

SHAP values measure how much each feature CONTRIBUTED to the sales prediction change.
Understanding this correctly is essential for accurate reporting.

### What SHAP Delta Values Mean

- **SHAP delta** = (scenario SHAP contribution) - (baseline SHAP contribution)
- This measures the CHANGE in a feature's contribution to the prediction

### How to Interpret the Sign

| SHAP Delta | Meaning |
|------------|---------|
| **Positive (+)** | This feature change INCREASED predicted sales |
| **Negative (-)** | This feature change DECREASED predicted sales |

**CRITICAL**: The sign of the SHAP delta tells you the EFFECT on sales, NOT whether the feature value went up or down.

### Examples of Correct Interpretation

1. **Staffing Hours increased, SHAP = +0.05**
   - CORRECT: "Increasing staffing hours contributed positively to sales (+0.05 impact)"
   - WRONG: "Staffing hours offset other factors"

2. **Unique Associates decreased, SHAP = +0.04**
   - CORRECT: "Reducing unique associates contributed positively to sales (+0.04 impact)"
   - This happens because the feature has negative elasticity (more associates = lower sales)
   - WRONG: "Reducing associates partially offset the gains" (the SHAP is POSITIVE, so it ADDED to gains)

3. **Unique Associates decreased, SHAP = -0.04**
   - CORRECT: "Reducing unique associates negatively impacted sales (-0.04 impact)"
   - WRONG: "The reduction helped offset losses" (negative SHAP means it HURT sales)

### Cross-Validation Rule

The sum of all SHAP deltas should approximately match the total sales percentage change.
If the scenario shows +6% sales vs baseline, the SHAP impacts should sum to roughly +0.06.

### Common Mistakes to AVOID

1. **Never say a positive SHAP "offset" gains** - positive SHAP = contributed TO gains
2. **Never say a negative SHAP "helped" sales** - negative SHAP = hurt sales
3. **Don't confuse feature direction with impact direction** - a DECREASE in a feature can have POSITIVE impact if the feature has negative elasticity

## Output Format

Generate markdown with ONLY relevant sections:

# {title}

**Generated:** {date}
**Store(s):** {store_ids} (or "All Stores" if portfolio-wide)
**Channel:** {channel}
**Analysis Period:** {date_range or "Current"}

---

## Executive Summary

3-5 bullet points with SPECIFIC NUMBERS:
- Key metrics (total sales, YoY change, store count, etc.)
- Most important findings
- Critical issues requiring attention

---

## Business Health Overview

(Include if diagnostic/health analysis was discussed)

| Metric | Value |
|--------|-------|
| Total Stores | X |
| Total Predicted Sales | $X |
| Weighted Avg YoY Change | X% |
| Stores Growing | X |
| Stores Declining | X |

---

## Regional Performance

(Include if DMA-level analysis was discussed)

| DMA | Stores | Avg YoY Change | Status |
|-----|--------|----------------|--------|
| ... | ... | ... | ... |

---

## Declining Stores Analysis

(Include if declining stores were identified)

| DMA | Store Count | Avg YoY Change |
|-----|-------------|----------------|
| ... | ... | ... |

List specific stores with largest declines and their metrics.

---

## Top/Bottom Performers

(Include if performance rankings were discussed)

### Top Performers
| Rank | Store | YoY Change |
|------|-------|------------|
| ... | ... | ... |

### Bottom Performers
| Rank | Store | YoY Change |
|------|-------|------------|
| ... | ... | ... |

---

## Forecast Overview

(Include if a forecast was run)

| Metric | Value |
|--------|-------|
| Total Forecast Sales | ${amount} |
| (other relevant metrics from data) |

---

## Week-by-Week Breakdown

(Always include if forecast was run)

If YoY data exists, use this format:
| Week | Date | 2025 Forecast ($) | 2024 Actual ($) | YoY Change |
|------|------|-------------------|-----------------|------------|

If no YoY, use simpler format:
| Week | Date | Forecast ($) |
|------|------|--------------|

---

## Forecast vs Previous Year

(ONLY if yoy_actuals data exists)

Summary of YoY comparison with totals and percentage change.

---

## What-If Scenario Analysis

(ONLY if scenario_comparison shows actual differences, delta != 0)

| Scenario | Total Sales | vs Baseline |
|----------|-------------|-------------|

---

## Key Drivers (SHAP Analysis)

(ONLY if explanation data exists)

Explain what drove the forecast change. Remember:
- Positive SHAP = contributed to sales INCREASE
- Negative SHAP = contributed to sales DECREASE

### Positive Drivers (Helped Sales)
For each driver, state: "[Feature] contributed +X.XX to sales by [explanation of what changed]"

### Negative Drivers (Hurt Sales)
For each driver, state: "[Feature] reduced sales by X.XX due to [explanation]"

**IMPORTANT**: Do NOT say positive drivers "offset" anything - they ADDED to sales.
Do NOT say negative drivers "helped" - they REDUCED sales.

---

## Sensitivity Analysis

(ONLY if sensitivity data exists)

| Lever | Elasticity | Interpretation |
|-------|------------|----------------|

---

## Charts

(ONLY if plot_files exist)

Reference the generated charts here. They will be embedded at the end of the PDF.

---

## Recommendations

Based on the analysis, provide 2-3 actionable recommendations with specific numbers.
"""


# ============================================================================
# Helper Functions
# ============================================================================

def _format_conversation_history(messages: list) -> str:
    """
    Format conversation messages for the report prompt.

    Extracts human and AI messages, summarizing tool calls.
    In conversation-only mode, this is the PRIMARY data source,
    so we preserve more content than in structured mode.

    Args:
        messages: List of LangChain message objects

    Returns:
        Formatted conversation string
    """
    if not messages:
        return "(No conversation history available)"

    formatted = []
    for msg in messages:
        msg_type = type(msg).__name__

        if msg_type == "HumanMessage":
            content = msg.content if hasattr(msg, 'content') else str(msg)
            # Preserve more content for conversation-only reports
            if len(content) > 1000:
                content = content[:1000] + "..."
            formatted.append(f"USER: {content}")

        elif msg_type == "AIMessage":
            content = msg.content if hasattr(msg, 'content') else ""
            # Check for tool calls
            tool_calls = getattr(msg, 'tool_calls', [])
            if tool_calls:
                tool_names = [tc.get('name', 'unknown') for tc in tool_calls if isinstance(tc, dict)]
                if tool_names:
                    formatted.append(f"ASSISTANT: [Called tools: {', '.join(tool_names)}]")
            if content:
                # Preserve more content - this is critical for conversation-only reports
                # where all data must be extracted from the conversation
                if len(content) > 2000:
                    content = content[:2000] + "..."
                formatted.append(f"ASSISTANT: {content}")

    # Return more exchanges to ensure we capture diagnostic findings
    # Gemini Flash has 1M context, so we can afford more history
    return "\n\n".join(formatted[-40:])


def _build_report_prompt(
    forecast_results: Optional[str],
    scenario_comparison: Optional[str],
    yoy_explanation: Optional[str],
    yoy_actuals: Optional[str],
    sensitivity_analysis: Optional[str],
    conversation_history: list,
    user_instructions: Optional[str],
    context: Dict[str, Any],
    plot_files: List[str],
) -> str:
    """
    Build the prompt with conversation context and all available data.

    Args:
        forecast_results: Raw JSON from run_forecast_model
        scenario_comparison: Raw JSON from compare_scenarios
        yoy_explanation: Raw JSON from explain_forecast_change
        yoy_actuals: Raw JSON from fetch_previous_year_actuals
        sensitivity_analysis: Raw JSON from analyze_sensitivity
        conversation_history: List of conversation messages
        user_instructions: User's focus/emphasis for the report
        context: Session context (title, dates, stores, etc.)
        plot_files: List of generated plot file paths

    Returns:
        Complete prompt string for the LLM
    """
    sections = []

    # Determine if this is conversation-only mode
    is_conversation_only = context.get("mode") == "conversation"
    has_structured_data = any([forecast_results, scenario_comparison, yoy_explanation, yoy_actuals, sensitivity_analysis])

    # 1. Conversation history - the primary data source in conversation mode
    sections.append(f"""## Conversation History
{_format_conversation_history(conversation_history)}
""")

    # 2. Report context section
    store_filter = context.get('store_filter', [])
    store_display = "All Stores" if not store_filter else str(store_filter)

    sections.append(f"""## Report Context
- Title: {context.get('title', 'Analysis Report')}
- Generated Date: {context.get('generated_date', 'N/A')}
- Analysis Scope: {store_display}
- Channel: {context.get('channel', 'B&M')}
- Mode: {"Conversation-based extraction" if is_conversation_only else "Structured data + conversation"}
""")

    # 3. Available data summary - helps LLM know what sections to include
    data_flags = []
    if is_conversation_only:
        data_flags.append("- **MODE: CONVERSATION-ONLY** - Extract ALL findings from conversation history")
        data_flags.append("- Structured forecast data: No")
        data_flags.append("- Structured scenario data: No")
        data_flags.append("- Data source: Conversation history (extract all numbers, metrics, findings)")
    else:
        data_flags.append(f"- Forecast results: {'Yes' if forecast_results else 'No'}")
        data_flags.append(f"- YoY actuals (previous year sales): {'Yes' if yoy_actuals else 'No'}")
        data_flags.append(f"- Scenario comparison: {'Yes' if scenario_comparison else 'No'}")
        data_flags.append(f"- SHAP explanation: {'Yes' if yoy_explanation else 'No'}")
        data_flags.append(f"- Sensitivity analysis: {'Yes' if sensitivity_analysis else 'No'}")
    data_flags.append(f"- Generated plots: {len(plot_files)} file(s)")

    sections.append(f"""## Available Data
{chr(10).join(data_flags)}
""")

    # 4. User instructions (priority)
    if user_instructions:
        sections.append(f"""## USER INSTRUCTIONS (PRIORITY)
{user_instructions}

Tailor the report to address these specific instructions.
""")

    # 5. Data sections - pass raw JSON for maximum fidelity
    if forecast_results:
        sections.append(f"""## Forecast Results Data
```json
{forecast_results}
```
""")

    if yoy_actuals:
        sections.append(f"""## Previous Year Actuals Data
Use this to create "Forecast vs Previous Year" comparison.
```json
{yoy_actuals}
```
""")

    if scenario_comparison:
        sections.append(f"""## Scenario Comparison Data
Only use this if delta values are non-zero (actual scenario differences exist).
```json
{scenario_comparison}
```
""")

    if yoy_explanation:
        sections.append(f"""## SHAP Explanation Data
```json
{yoy_explanation}
```
""")

    if sensitivity_analysis:
        sections.append(f"""## Sensitivity Analysis Data
```json
{sensitivity_analysis}
```
""")

    # 6. Plot files reference
    if plot_files:
        plot_names = [Path(p).name for p in plot_files]
        sections.append(f"""## Generated Charts
The following charts were generated and WILL BE EMBEDDED at the end of the PDF.
Reference them in your report:
{chr(10).join(f'- {name}' for name in plot_names)}
""")

    # 7. Final instructions - different for conversation vs structured mode
    if is_conversation_only:
        sections.append("""## Instructions (CONVERSATION-ONLY MODE)

IMPORTANT: No structured JSON data is available. You MUST extract ALL findings from the conversation history above.

1. **Read the conversation carefully** - Extract every metric, percentage, store name, DMA name, and finding mentioned
2. **Organize into appropriate sections** - Use Business Health Overview, Regional Performance, Declining Stores, etc.
3. **Include ALL specific numbers** - If the assistant mentioned "3.55% YoY change", include it
4. **List all stores/DMAs mentioned** - Include names and their associated metrics
5. **Create actionable recommendations** - Based on the findings discussed
6. **Reference charts if they were generated**

Do NOT invent data. Only use information explicitly mentioned in the conversation.
""")
    else:
        sections.append("""## Instructions
Generate the report using ONLY the data provided above.
- Include sections ONLY for data that exists
- Do NOT include What-If/Baseline comparison if scenario_comparison is missing or shows $0 delta
- If yoy_actuals exists, include a Forecast vs Previous Year comparison
- Reference the charts if plot files are listed
- Use specific numbers from the JSON data
""")

    return "\n\n".join(sections)


def _validate_plot_files(plot_paths: List[str]) -> List[str]:
    """
    Validate and filter plot file paths.

    Args:
        plot_paths: List of file paths to PNG files

    Returns:
        List of validated file paths that exist
    """
    valid_paths = []
    for path in plot_paths:
        if path and Path(path).exists() and path.endswith('.png'):
            valid_paths.append(path)
    return list(dict.fromkeys(valid_paths))  # Dedupe preserving order


def _render_line_with_inline_bold(pdf, line: str, font_size: int = 10, line_height: int = 6):
    """
    Render a line that may contain **bold** segments inline.

    Args:
        pdf: FPDF instance
        line: Text line that may contain **bold** markers
        font_size: Font size in points
        line_height: Line height in mm
    """
    import re

    pdf.set_font("OpenSans", "", font_size)
    pdf.set_text_color(50, 50, 50)

    # Split by bold markers, keeping the markers in the result
    parts = re.split(r'(\*\*[^*]+\*\*)', line)

    for part in parts:
        if part.startswith('**') and part.endswith('**'):
            # Bold segment - strip the ** markers
            pdf.set_font("OpenSans", "B", font_size)
            pdf.write(line_height, part[2:-2])
            pdf.set_font("OpenSans", "", font_size)
        elif part:
            # Normal segment
            pdf.write(line_height, part)

    pdf.ln(line_height)


def _render_markdown_table(pdf, table_lines: List[str]):
    """
    Render a markdown table using fpdf2's modern table API.

    This approach handles:
    - Automatic column width calculation based on content
    - Text wrapping within cells (no overflow)
    - Proper multi-line cell handling

    Args:
        pdf: FPDF instance
        table_lines: List of markdown table lines (header, separator, data rows)
    """
    from fpdf import FontFace

    if len(table_lines) < 2:
        return

    # Parse header row
    header = [cell.strip() for cell in table_lines[0].split('|') if cell.strip()]
    if not header:
        return

    # Parse data rows (skip separator line at index 1)
    data_rows = []
    for line in table_lines[2:]:
        if line.strip().startswith('|'):
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            if cells:
                # Pad row to match header length if needed
                while len(cells) < len(header):
                    cells.append('')
                data_rows.append(cells)

    # Build table data: header + data rows
    table_data = [header] + data_rows

    # Define header style with gray background
    headings_style = FontFace(emphasis="BOLD", fill_color=(240, 240, 240))

    # Set font for table content
    pdf.set_font("OpenSans", "", 9)
    pdf.set_text_color(50, 50, 50)

    # Render using fpdf2's table() context manager
    # This handles auto-column widths and text wrapping
    with pdf.table(
        table_data,
        headings_style=headings_style,
        cell_fill_color=(255, 255, 255),
        cell_fill_mode="ROWS",
        line_height=pdf.font_size * 1.8,
        text_align="LEFT",
        first_row_as_headings=True,
        borders_layout="SINGLE_TOP_LINE",
    ):
        pass

    pdf.ln(3)


def markdown_to_pdf(
    markdown_content: str,
    plot_files: List[str],
    _context: Dict[str, Any],
) -> Path:
    """
    Convert LLM-generated markdown to PDF with embedded plots.

    Performs simple markdown parsing and renders to PDF using fpdf2.

    Args:
        markdown_content: Markdown text from the LLM
        plot_files: List of PNG file paths to embed
        _context: Session context (unused, kept for API compatibility)

    Returns:
        Path to the generated PDF file
    """
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Load Unicode-supporting OpenSans fonts
    font_dir = Path(__file__).parent / "fonts" / "Open_Sans" / "static"
    pdf.add_font("OpenSans", fname=str(font_dir / "OpenSans-Regular.ttf"))
    pdf.add_font("OpenSans", style="B", fname=str(font_dir / "OpenSans-Bold.ttf"))

    pdf.add_page()

    # Markdown parsing with proper table and inline bold handling
    lines = markdown_content.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]
        pdf.set_x(pdf.l_margin)  # Reset cursor to left margin

        # Check for table block - collect all consecutive table lines
        if line.strip().startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            _render_markdown_table(pdf, table_lines)
            continue

        # H1 headers
        if line.startswith("# "):
            pdf.set_font("OpenSans", "B", 18)
            pdf.set_text_color(46, 134, 171)
            pdf.multi_cell(0, 10, line[2:])
            pdf.ln(5)
        # H2 headers
        elif line.startswith("## "):
            pdf.set_font("OpenSans", "B", 14)
            pdf.set_text_color(46, 134, 171)
            pdf.multi_cell(0, 8, line[3:])
            pdf.ln(3)
        # H3 headers
        elif line.startswith("### "):
            pdf.set_font("OpenSans", "B", 12)
            pdf.set_text_color(70, 70, 70)
            pdf.multi_cell(0, 7, line[4:])
            pdf.ln(2)
        # Bullet points - check for inline bold
        elif line.startswith("- ") or line.startswith("* "):
            content = line[2:]
            if "**" in content:
                pdf.set_font("OpenSans", "", 10)
                pdf.set_text_color(50, 50, 50)
                pdf.write(6, "  - ")
                _render_line_with_inline_bold(pdf, content, 10, 6)
            else:
                pdf.set_font("OpenSans", "", 10)
                pdf.set_text_color(50, 50, 50)
                pdf.multi_cell(0, 6, "  - " + content)
        # Numbered lists - check for inline bold
        elif line.startswith(("1. ", "2. ", "3. ", "4. ", "5. ", "6. ", "7. ", "8. ", "9. ")):
            if "**" in line:
                _render_line_with_inline_bold(pdf, line, 10, 6)
            else:
                pdf.set_font("OpenSans", "", 10)
                pdf.set_text_color(50, 50, 50)
                pdf.multi_cell(0, 6, line)
        # Horizontal rule
        elif line.strip() == "---":
            pdf.ln(5)
        # Lines with inline bold (like **Generated:** 2024-01-01)
        elif "**" in line:
            _render_line_with_inline_bold(pdf, line, 10, 6)
        # Regular text
        elif line.strip():
            pdf.set_font("OpenSans", "", 10)
            pdf.set_text_color(50, 50, 50)
            pdf.multi_cell(0, 6, line)

        i += 1

    # Add charts at the end
    if plot_files:
        pdf.add_page()
        pdf.set_font("OpenSans", "B", 14)
        pdf.set_text_color(46, 134, 171)
        pdf.cell(0, 10, "Charts", 0, 1)
        pdf.ln(5)

        for plot_file in plot_files:
            if Path(plot_file).exists():
                try:
                    # Check if we need a new page (image might not fit)
                    if pdf.get_y() > 200:
                        pdf.add_page()
                    pdf.image(plot_file, x=15, w=180)
                    pdf.ln(10)
                except Exception:
                    pass  # Skip problematic images

    # Save to output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"forecast_report_{timestamp}.pdf"
    pdf.output(str(pdf_path))

    return pdf_path


# ============================================================================
# Main Tool Function
# ============================================================================

@tool
def generate_report(
    user_instructions: Optional[str] = None,
    title: Optional[str] = None,
    include_plots: bool = True,
    mode: str = "auto",
) -> Dict[str, Any]:
    """
    Generate a PDF report from the current session's conversation and analysis.

    This tool can generate reports in two modes:
    - "auto" (default): Uses stored analysis results if available, otherwise
      extracts findings from conversation history
    - "conversation": Explicitly generates from conversation history only,
      useful for exporting diagnostic findings or any discussion

    The tool reads stored results from forecasting tools when available:
    - run_forecast_model
    - compare_scenarios
    - explain_forecast_change
    - analyze_sensitivity
    - fetch_previous_year_actuals

    When no stored results exist, the tool extracts findings from the
    conversation history, making it useful for exporting diagnostic analyses,
    business health overviews, or any other discussion.

    Args:
        user_instructions: Focus/emphasis for the report content. Examples:
            - "Focus on declining stores analysis"
            - "Summarize the business health findings"
            - "Create an executive summary for leadership"
        title: Report title. Default: auto-generated from context.
        include_plots: Whether to embed PNG charts. Default: True.
        mode: Report generation mode. Options:
            - "auto": Use stored results if available, else use conversation
            - "conversation": Generate from conversation history only

    Returns:
        Dictionary containing:
        - status: "generated" on success
        - file_path: Absolute path to PDF file
        - summary: Brief description of report contents
        - mode_used: The actual mode used for generation

    Examples:
        >>> generate_report()  # Auto mode - uses best available data
        >>> generate_report(user_instructions="Focus on declining stores")
        >>> generate_report(mode="conversation")  # Force conversation-only
        >>> generate_report(title="Business Health Analysis Q1 2025")
    """
    session = get_session()
    state = session.get_state()

    # 1. Get stored results from session
    stored = session.get_stored_results()

    # 2. Check what data is available
    has_structured_data = any([
        stored["forecast_results"],
        stored["scenario_comparison"],
        stored["explanation"],
        stored["sensitivity"],
        stored["yoy_actuals"],
    ])

    # 3. Get conversation history
    conversation_history = session.get_messages()
    has_conversation = len(conversation_history) > 0

    # 4. Determine actual mode to use
    if mode == "conversation":
        # Force conversation-only mode
        actual_mode = "conversation"
    elif mode == "auto":
        # Use structured data if available, otherwise conversation
        actual_mode = "structured" if has_structured_data else "conversation"
    else:
        return {
            "error": f"Invalid mode '{mode}'. Must be 'auto' or 'conversation'.",
            "hint": "Use mode='auto' for automatic detection or mode='conversation' for conversation-only.",
        }

    # 5. Validate we have something to report on
    if actual_mode == "conversation" and not has_conversation:
        return {
            "error": "No conversation history found to generate report from.",
            "hint": "Have a conversation with the assistant first, then request a report.",
        }

    if actual_mode == "structured" and not has_structured_data:
        # This shouldn't happen with auto mode, but handle it gracefully
        actual_mode = "conversation"
        if not has_conversation:
            return {
                "error": "No data available for report generation.",
                "hint": "Run analysis tools or have a conversation first.",
            }

    # 3. Build context from session state
    store_filter = state.get("store_filter", [])
    origin_date = state.get("origin_date", "N/A")
    horizon_weeks = state.get("horizon_weeks", 13)

    # Calculate end date
    try:
        if origin_date != "N/A":
            origin_dt = datetime.strptime(origin_date, "%Y-%m-%d")
            end_date = (origin_dt + timedelta(weeks=horizon_weeks)).strftime("%Y-%m-%d")
        else:
            end_date = "N/A"
    except Exception:
        end_date = "N/A"

    # Generate default title based on mode
    if title:
        report_title = title
    elif actual_mode == "conversation":
        report_title = "Business Analysis Report"
    elif store_filter:
        report_title = f"Forecast Analysis - Store {store_filter}"
    else:
        report_title = "Sales Forecast Analysis"

    context = {
        "title": report_title,
        "generated_date": datetime.now().strftime("%Y-%m-%d"),
        "origin_date": origin_date,
        "end_date": end_date,
        "horizon_weeks": horizon_weeks,
        "channel": state.get("channel", "B&M"),
        "store_filter": store_filter,
        "mode": actual_mode,
    }

    # 6. Validate and filter plot files
    validated_plot_files = []
    if include_plots and stored["plot_files"]:
        validated_plot_files = _validate_plot_files(stored["plot_files"])

    # 7. Initialize LLM
    try:
        llm = make_llm(provider="vertexai", model_name="gemini-2.5-flash", temperature=0, max_tokens=16384)
    except Exception as e:
        return {
            "error": f"Failed to initialize LLM: {str(e)}",
            "hint": "Check your API credentials and connectivity.",
        }

    # 8. Build prompt with conversation context and all stored data
    # In conversation mode, we pass None for structured data to force extraction from conversation
    if actual_mode == "conversation":
        report_prompt = _build_report_prompt(
            forecast_results=None,
            scenario_comparison=None,
            yoy_explanation=None,
            yoy_actuals=None,
            sensitivity_analysis=None,
            conversation_history=conversation_history,
            user_instructions=user_instructions,
            context=context,
            plot_files=validated_plot_files,
        )
    else:
        report_prompt = _build_report_prompt(
            forecast_results=stored["forecast_results"],
            scenario_comparison=stored["scenario_comparison"],
            yoy_explanation=stored["explanation"],
            yoy_actuals=stored["yoy_actuals"],
            sensitivity_analysis=stored["sensitivity"],
            conversation_history=conversation_history,
            user_instructions=user_instructions,
            context=context,
            plot_files=validated_plot_files,
        )

    # 8. Call LLM to generate report content
    try:
        from app.agent.common import normalize_llm_response
        response = llm.invoke([
            SystemMessage(content=REPORT_SYSTEM_PROMPT),
            HumanMessage(content=report_prompt),
        ])
        markdown_content = normalize_llm_response(response.content)
    except Exception as e:
        return {
            "error": f"LLM call failed: {str(e)}",
            "hint": "Check API connectivity and try again.",
        }

    # 9. Convert markdown to PDF with embedded plots
    try:
        pdf_path = markdown_to_pdf(markdown_content, validated_plot_files, context)
        # Track generated PDF for chat attachments
        try:
            session.add_export_file(str(pdf_path.absolute()))
        except Exception:
            pass  # Don't fail if tracking fails
    except ImportError:
        return {
            "error": "fpdf2 library not installed. Run: pip install fpdf2",
            "hint": "Install the PDF generation library.",
        }
    except Exception as e:
        return {
            "error": f"PDF generation failed: {str(e)}",
            "hint": "Check file permissions and disk space.",
        }

    # Build summary based on mode
    data_sources = []
    if actual_mode == "conversation":
        data_sources.append("conversation history")
    else:
        if stored["forecast_results"]:
            data_sources.append("forecast")
        if stored["yoy_actuals"]:
            data_sources.append("YoY actuals")
        if stored["scenario_comparison"]:
            data_sources.append("scenario comparison")
        if stored["explanation"]:
            data_sources.append("SHAP explanation")
        if stored["sensitivity"]:
            data_sources.append("sensitivity")

    focus_summary = ""
    if user_instructions:
        truncated = user_instructions[:50] + "..." if len(user_instructions) > 50 else user_instructions
        focus_summary = f" focused on: {truncated}"

    return {
        "status": "generated",
        "file_path": str(pdf_path.absolute()),
        "summary": f"Generated report{focus_summary} with {len(validated_plot_files)} chart(s). Data sources: {', '.join(data_sources)}.",
        "plots_included": validated_plot_files,
        "data_sources": data_sources,
        "mode_used": actual_mode,
        "hint": "Open the PDF to view the complete analysis report.",
    }


# Export all tools
__all__ = [
    "generate_report",
]
