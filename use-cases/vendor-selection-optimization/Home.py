"""
Home page for the multipage Streamlit application.

This entry point applies the shared theme and provides a concise overview of
the available analysis pages. Navigation is handled by Streamlit's built-in
"pages" mechanism (files in resources/pages/).
"""

import streamlit as st
from ui import components
from ui.theme import apply_template_theme
from ui.content import descriptions

def render_overview() -> None:
    """Render a short recap of each page's functionality."""
    st.markdown(
        "Use the left navigation to open a page. First select a material in the sidebar (Select Material Description)."
    )

    st.markdown("---")
    st.subheader("Pages")
    st.markdown(
        "- **Vendor Selection Assistant**: AI-powered vendor recommendations with interactive comparison tables, cost breakdowns, and performance visualizations. Click the button to generate supplier recommendations for your selected material.\n"
        "- **Optimized Vendor Comparison**: Compare historical vs optimized vendor allocations with before/after scatter plots, cost savings analysis, and allocation tables showing suggested vendor distribution.\n"
        "- **Scatter Analysis**: Interactive scatter plot analyzing lead time versus OTIF (On-Time In-Full) performance with quadrant analysis to identify ideal vendors and improvement opportunities.\n"
        "- **Performance Heatmap**: Normalized performance heatmap comparing vendors by lead time, OTIF rate, base price, tariff impact, and logistics costs with top performers highlighted.\n"
        "- **Geographic View**: Country-level choropleth maps showing geographic distribution of lead time, OTIF rates, tariff impact, and logistics costs with country performance summary tables.\n"
        "- **Data Table**: Interactive data table with vendor performance metrics, search functionality, customizable column selection, sorting, and CSV export capabilities.\n"
        "- **Settings**: Configure tariff values and economic impact parameters with optimization profile management. Switch between profiles to load different optimization configurations."
    )


def main() -> None:
    # Page configuration and theme
    components.setup_page_config()
    apply_template_theme()
    components.apply_custom_styles()

    # Header
    st.title("AI Supplier Sourcing Optimizer")
    st.subheader(
        "Analyze supplier performance metrics with interactive visualizations and optimization insights."
    )

    render_overview()


if __name__ == "__main__":
    main()


