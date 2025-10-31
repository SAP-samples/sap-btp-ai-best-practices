"""
Shared textual content for the multipage UI.

Exposes the `descriptions` dictionary used by `Home.py` and all
individual page scripts to render a brief page summary above the
top metrics bubbles.
"""

descriptions = {
    "Vendor Selection Assistant": (
        "AI-powered vendor recommendations with interactive comparison tables, "
        "cost breakdowns, and performance visualizations. Click the button to "
        "generate supplier recommendations for your selected material."
    ),
    "Optimized Vendor Comparison": (
        "Compare historical vs optimized vendor allocations with before/after "
        "scatter plots, cost savings analysis, and allocation tables showing "
        "suggested vendor distribution."
    ),
    "Scatter Analysis": (
        "Interactive scatter plot analyzing lead time versus OTIF (On-Time "
        "In-Full) performance with quadrant analysis to identify ideal vendors "
        "and improvement opportunities."
    ),
    "Performance Heatmap": (
        "Normalized performance heatmap comparing vendors by lead time, OTIF "
        "rate, base price, tariff impact, and logistics costs with top "
        "performers highlighted."
    ),
    "Geographic View": (
        "Country-level choropleth maps showing geographic distribution of lead "
        "time, OTIF rates, tariff impact, and logistics costs with country "
        "performance summary tables."
    ),
    "Data Table": (
        "Interactive data table with vendor performance metrics, search "
        "functionality, customizable column selection, sorting, and CSV export "
        "capabilities."
    ),
    "Settings": (
        "Configure tariff values and economic impact parameters with "
        "optimization profile management. Switch between profiles to load "
        "different optimization configurations."
    ),
}


