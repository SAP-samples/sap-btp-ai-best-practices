from __future__ import annotations

import sys
from pathlib import Path

UI_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = UI_ROOT.parent

for path in (UI_ROOT, PROJECT_ROOT):
    str_path = str(path)
    if str_path not in sys.path:
        sys.path.insert(0, str_path)

import streamlit as st

import app_setup
from utils import state

APP_TITLE = app_setup.APP_TITLE
PAGE_INFO = {
    "**Select a Sales Order**": {
        "switch_target": "pages/1_Select_a_Sales_Order",
        "page_link": "pages/1_Select_a_Sales_Order.py",
        "icon": "",
        "description": "Investigate individual orders, run AI classification, and explore SHAP insights.",
    },
    "**Monthly Dashboard**": {
        "switch_target": "pages/2_Monthly_Dashboard",
        "page_link": "pages/2_Monthly_Dashboard.py",
        "icon": "",
        "description": "Review aggregated anomaly trends and drill into daily performance metrics.",
    },
    "**Fine Tuning**": {
        "switch_target": "pages/3_Fine_Tuning",
        "page_link": "pages/3_Fine_Tuning.py",
        "icon": "",
        "description": "Manage training data, monitor model performance, and retrain anomaly detectors.",
    },
}
PAGE_TARGETS = {tab: info["switch_target"] for tab, info in PAGE_INFO.items()}


def _format_metric(value: int | float | None) -> str:
    """Format numeric metrics with thousands separators."""
    if value is None:
        return "N/A"
    return f"{value:,}"


def _redirect_from_query_param() -> bool:
    """Deep-link to the requested page when a tab query parameter is provided."""
    tab_param = st.query_params.get("tab")
    last_param = st.session_state.get("_last_tab_param")

    if not tab_param:
        st.session_state["_last_tab_param"] = None
        return False

    display_tab = state.TAB_MAPPING.get(tab_param)
    target_page = PAGE_TARGETS.get(display_tab)

    if target_page and tab_param != last_param:
        st.session_state["_last_tab_param"] = tab_param
        st.switch_page(target_page)
        return True

    st.session_state["_last_tab_param"] = tab_param
    return False


def _render_page_link(page_link: str, label: str, icon: str, description: str) -> None:
    """Render a navigation link using the provided label and description."""
    if hasattr(st, "page_link"):
        st.page_link(page_link, label=f"{icon} {label}")
    else:
        if st.button(f"{icon} {label}", use_container_width=True):
            st.switch_page(page_link.replace(".py", ""))
    st.caption(description)


def main() -> None:
    """Render the Streamlit landing page with template styling."""
    app_setup.configure_page(APP_TITLE)
    app_setup.apply_base_theme()
    app_setup.initialize_state()

    if _redirect_from_query_param():
        return

    app_setup.show_url_warnings()

    st.title(APP_TITLE)
    st.markdown(
        """
        <div class="success-message">
            This workspace unifies anomaly detection analytics, fine-tuning workflows, and explainability tools.
            Use the links below or the sidebar navigation to jump into the experience you need.
        </div>
        """,
        unsafe_allow_html=True,
    )

    data_loaded = app_setup.ensure_data_loaded()
    loaded_data = app_setup.get_loaded_data() if data_loaded else None

    if loaded_data:
        results_df = loaded_data.results
        total_orders = int(results_df.shape[0])
        predicted_col = "predicted_anomaly" if "predicted_anomaly" in results_df.columns else None
        anomalies = int(results_df[predicted_col].sum()) if predicted_col else None
        blocked_col = "blocked_by_business_rules" if "blocked_by_business_rules" in results_df.columns else None
        blocked = int(results_df[blocked_col].sum()) if blocked_col else None
        customer_col = "Sold To number" if "Sold To number" in results_df.columns else None
        unique_customers = results_df[customer_col].nunique() if customer_col else None

        metric_columns = st.columns(3)
        with metric_columns[0]:
            st.metric("Total Sales Orders", _format_metric(total_orders))
        with metric_columns[1]:
            st.metric("AI-Flagged Anomalies", _format_metric(anomalies))
        with metric_columns[2]:
            st.metric("Business-Rule Blocks", _format_metric(blocked))

        st.caption(f"Dataset: `{loaded_data.data_path}` — Unique customers: {_format_metric(unique_customers)}")

    link_columns = st.columns(len(PAGE_INFO))
    for column, (label, info) in zip(link_columns, PAGE_INFO.items()):
        with column:
            _render_page_link(
                info["page_link"],
                label,
                info["icon"],
                info["description"],
            )

    st.markdown("---")
    st.subheader("About This Application")
    st.markdown(
        """
        - Aligns sales order monitoring with AI-driven anomaly detection
        - Surfaces feature-level explanations and SHAP insights
        - Supports monthly cohort analysis to track operational performance
        - Provides fine-tuning workflows for continuous model improvement
        """
    )


if __name__ == "__main__":
    main()
