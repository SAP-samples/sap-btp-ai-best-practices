from __future__ import annotations

"""Monthly dashboard view."""

from datetime import datetime
from typing import Optional

import pandas as pd
import streamlit as st

from utils import state, anomaly, formatting
from views.helpers import calendar as calendar_helper
from views.helpers import charts
from views import state_sync, components
from services import ai_classification, model_runtime, shap_explanations, order_selection
from visualization.feature_analysis import create_feature_plots


def render_monthly_dashboard(results_df: pd.DataFrame, features_df: pd.DataFrame) -> None:
    st.header("Monthly Anomaly Overview")

    if "Sales Document Created Date" not in results_df.columns or results_df["Sales Document Created Date"].isnull().all():
        st.error("Sales Document Created Date column is missing or empty.")
        return

    results_df = results_df.dropna(subset=["Sales Document Created Date"])
    if results_df.empty:
        st.warning("No valid date data available after cleaning.")
        return

    results_df["year"] = results_df["Sales Document Created Date"].dt.year
    results_df["month"] = results_df["Sales Document Created Date"].dt.month

    available_years = sorted(results_df["year"].unique(), reverse=True)
    available_months = sorted(results_df["month"].unique())

    _render_date_selector(available_years, available_months)

    selected_year = st.session_state.get("selected_year")
    selected_month = st.session_state.get("selected_month")
    selected_day_input = st.session_state.get("selected_day_input")

    if selected_year is None or selected_month is None:
        st.info("Please select a year and month to view the dashboard.")
        return

    monthly_data = results_df[
        (results_df["year"] == selected_year)
        & (results_df["month"] == selected_month)
    ].copy()

    if monthly_data.empty:
        st.warning(f"No data found for {selected_month}/{selected_year}.")
        return

    daily_summary = monthly_data.groupby(monthly_data["Sales Document Created Date"].dt.date).agg(
        total_orders=("Sales Document Number", "count"),
        anomaly_count=("predicted_anomaly", "sum"),
    ).reset_index()
    daily_summary["anomaly_rate"] = daily_summary["anomaly_count"] / daily_summary["total_orders"]
    daily_summary.rename(columns={"Sales Document Created Date": "date"}, inplace=True)
    daily_summary["date"] = pd.to_datetime(daily_summary["date"])

    calendar_helper.render_calendar_heatmap(selected_year, selected_month, daily_summary)
    st.markdown("---")

    selected_day = _resolve_selected_day(selected_day_input, monthly_data, daily_summary)
    if selected_day is None:
        return

    selected_date = datetime(selected_year, selected_month, selected_day).date()
    daily_data = monthly_data[monthly_data["Sales Document Created Date"].dt.date == selected_date].copy()

    if daily_data.empty:
        st.warning(f"No orders found for {selected_date.strftime('%Y-%m-%d')}.")
        return

    _render_daily_metrics(daily_data)
    _render_daily_charts(daily_data)
    _render_orders_table(daily_data, features_df)


def _render_date_selector(available_years, available_months) -> None:
    selector_col1, selector_col2, selector_col3 = st.columns([1, 1, 1])

    default_year = st.session_state.get("selected_year")
    if default_year not in available_years:
        default_year = available_years[0]
    default_month = st.session_state.get("selected_month")
    if default_month not in available_months:
        default_month = available_months[-1]

    month_names = {i: datetime(2000, i, 1).strftime("%B") for i in available_months}

    with selector_col1:
        st.selectbox(
            "Select Year:",
            available_years,
            index=available_years.index(default_year),
            key="selected_year",
            on_change=state.sync_url_params,
        )
    with selector_col2:
        st.selectbox(
            "Select Month:",
            available_months,
            format_func=lambda x: month_names[x],
            index=available_months.index(default_month),
            key="selected_month",
            on_change=state.sync_url_params,
        )
    with selector_col3:
        st.text_input(
            "Select Day:",
            placeholder="e.g., 15",
            key="selected_day_input",
            value=str(st.session_state.get("selected_day_input", "")),
            on_change=state.sync_url_params,
        )


def _resolve_selected_day(selected_day_input: str, monthly_data: pd.DataFrame, daily_summary: pd.DataFrame) -> Optional[int]:
    available_days = sorted(monthly_data["Sales Document Created Date"].dt.day.unique())

    if not available_days:
        st.warning("No specific days with data found for this month.")
        return None

    if selected_day_input:
        try:
            day_input = int(selected_day_input)
            if 1 <= day_input <= 31 and day_input in available_days:
                return day_input
            st.error(
                "Day {day_input} is not available in the selected month. Available days: {', '.join(map(str, available_days))}"
            )
        except ValueError:
            st.error("Please enter a valid day number (1-31)")

    default_day = available_days[0]
    if not daily_summary.empty:
        day_with_most_anomalies = daily_summary.loc[daily_summary["anomaly_count"].idxmax()]["date"].day
        if day_with_most_anomalies in available_days:
            default_day = day_with_most_anomalies
    st.info(f"Showing data for day {default_day} (default). Enter a day number above to view a specific day.")
    return default_day


def _render_daily_metrics(daily_data: pd.DataFrame) -> None:
    total_orders = len(daily_data)
    anomalies = daily_data[daily_data["predicted_anomaly"] == 1]
    anomaly_count = len(anomalies)
    anomaly_rate = anomaly_count / total_orders if total_orders > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Orders", total_orders)
    with col2:
        st.metric("Anomalies Detected", anomaly_count)
    with col3:
        st.metric("Anomaly Rate", f"{anomaly_rate:.1%}")
    with col4:
        total_value = daily_data["Order item value"].sum()
        st.metric("Total Order Value", f"${total_value:,.0f}")
    with col5:
        anomaly_total_value = anomalies["Order item value"].sum()
        st.metric("Total Value Detected Anomalies", f"${anomaly_total_value:,.0f}")


def _render_daily_charts(daily_data: pd.DataFrame) -> None:
    anomalies = daily_data[daily_data["predicted_anomaly"] == 1]

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(charts.build_top_materials_chart(anomalies), use_container_width=True)

    with col2:
        st.plotly_chart(charts.build_anomaly_score_distribution(daily_data), use_container_width=True)


def _render_orders_table(daily_data: pd.DataFrame, features_df: pd.DataFrame) -> None:
    st.subheader("Orders by Anomaly Score")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        daily_sorted = daily_data.sort_values("anomaly_score", ascending=True)

        display_columns = [
            "Sales Document Number",
            "Sales Document Item",
            "Customer PO number",
            "Material Description",
            "Sales Order item qty",
            "Unit Price",
            "anomaly_score",
            "predicted_anomaly",
        ]

        display_df = daily_sorted[display_columns].copy()
        display_df.columns = ["Doc Number", "Item", "PO Number", "Material", "Qty", "Unit Price", "Score", "Anomaly"]
        display_df["Unit Price"] = display_df["Unit Price"].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
        display_df["Anomaly"] = display_df["Anomaly"].apply(lambda x: "Yes" if x == 1 else "No")
        display_df["Material"] = display_df["Material"].str[:50] + "..."

        st.caption("Click on any row in the table below to automatically load its analysis in the panel →")

        event = st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=600,
            on_select="rerun",
            selection_mode="single-row",
            key="orders_table",
        )

        if event.selection.rows:
            selected_row_idx = event.selection.rows[0]
            selected_order = daily_sorted.iloc[selected_row_idx]
            doc_number = order_selection.normalize_identifier(selected_order["Sales Document Number"])
            doc_item = order_selection.normalize_identifier(selected_order["Sales Document Item"])

            if doc_number and doc_item:
                state_sync.set_current_order(doc_number, doc_item)
                st.success(
                    f"Selected: Document {doc_number} - Item {doc_item} | View details in the panel →"
                )

    with col_right:
        st.subheader("Selected Order Analysis")

        doc_num = st.session_state.get("search_doc_number")
        doc_item = st.session_state.get("search_doc_item")

        if not doc_num or not doc_item:
            st.info("Click on any row in the orders table to view detailed analysis here.")
            return

        # Locate the selected order within today's data
        normalized_docs = daily_data["Sales Document Number"].map(order_selection.normalize_identifier)
        normalized_items = daily_data["Sales Document Item"].map(order_selection.normalize_identifier)
        target_doc = order_selection.normalize_identifier(doc_num)
        target_item = order_selection.normalize_identifier(doc_item)

        selected_mask = (normalized_docs == target_doc) & (normalized_items == target_item)
        if not selected_mask.any():
            st.warning("Selected order not found for the chosen day.")
            return

        row = daily_data[selected_mask].iloc[0]

        # Ensure SHAP is available for this order
        doc_number = order_selection.normalize_identifier(row.get("Sales Document Number"))
        doc_item = order_selection.normalize_identifier(row.get("Sales Document Item"))
        daily_shap_key = f"on_demand_shap_{doc_number}_{doc_item}"
        if daily_shap_key not in st.session_state:
            computed_shap = model_runtime.compute_shap(row, features_df)
            shap_df = shap_explanations.build_shap_dataframe(computed_shap or row.get("shap_explanation"), features_df, row)
            st.session_state[daily_shap_key] = shap_df
        elif st.session_state[daily_shap_key] is None:
            computed_shap = model_runtime.compute_shap(row, features_df)
            shap_df = shap_explanations.build_shap_dataframe(computed_shap or row.get("shap_explanation"), features_df, row)
            st.session_state[daily_shap_key] = shap_df

        # Anomaly status
        score = row["anomaly_score"]
        model_type = row.get("model_used", "sklearn")
        category, css_class, _, _ = anomaly.get_anomaly_category(score, model_type)
        is_anomaly = row.get("predicted_anomaly", 0) == 1
        is_blocked = row.get("blocked_by_business_rules", False)

        status_text = "ANOMALY DETECTED" if is_anomaly else "NORMAL ORDER"
        if is_blocked:
            status_text += " - BLOCKED BY BUSINESS RULES"
        bg_color = "#ffcccc" if is_anomaly else ("#fff4cc" if is_blocked else "#ccffcc")
        components.anomaly_status_card(status_text, bg_color)

        # Business rules details
        if is_blocked and row.get("business_rules_explanation"):
            violations = [v.strip() for v in row["business_rules_explanation"].split(";") if v.strip()]
            if violations:
                st.markdown("**Business Rules Violation Details:**")
                for violation in violations:
                    st.markdown(f"- {violation}")

        # AI Actions
        daily_ai_key = f"daily_ai_explanation_{doc_number}_{doc_item}"
        daily_binary_key = f"daily_ai_binary_{doc_number}_{doc_item}"

        act_col1, act_col2 = st.columns([2, 1])
        with act_col1:
            button_label = "Generate AI Explanation" if daily_ai_key not in st.session_state else "Regenerate"
            if st.button(button_label, key=f"daily_btn_{daily_ai_key}"):
                with st.spinner("Generating AI explanation..."):
                    explanation = ai_classification.generate_full_explanation(row, features_df)
                    st.session_state[daily_ai_key] = explanation
        with act_col2:
            if st.button("Quick AI Classification", key=f"daily_binary_btn_{daily_binary_key}"):
                with st.spinner("Generating quick AI classification..."):
                    result, _ = ai_classification.run_binary_classification(row, features_df)
                    st.session_state[daily_binary_key] = result

        if daily_binary_key in st.session_state:
            result_text = st.session_state[daily_binary_key]
            bg = "#ffcccc" if "Anomalous" in result_text else ("#ccffcc" if "Normal" in result_text else "#fff4cc")
            st.markdown(
                f"""
                <div style="background-color: {bg}; padding: 12px; margin: 8px 0; border-radius: 8px; text-align: center; font-weight: bold;">
                    {result_text}
                </div>
                """,
                unsafe_allow_html=True,
            )

        if daily_ai_key in st.session_state:
            st.markdown("### AI Analysis Details")
            st.markdown(st.session_state[daily_ai_key])

        # Order Information
        st.markdown("#### Order Information")
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.write(f"**Sales Document:** {doc_number} - {doc_item}")
            st.write(f"**Customer PO:** {row['Customer PO number']}")
            st.write(f"**Customer:** {row['Sold To number']}")
            st.write(f"**Ship-To:** {row['Ship-To Party']}")
        with info_col2:
            st.write(f"**Material:** {row['Material Number']}")
            st.write(f"**Description:** {row['Material Description'][:40]}...")
            st.write(f"**Quantity:** {row['Sales Order item qty']}")
            st.write(f"**Value:** ${row['Order item value']:,.2f}")

        # SHAP Table
        shap_df = st.session_state.get(daily_shap_key)
        if shap_df is not None and not shap_df.empty:
            st.markdown("#### Key Contributing Features")
            components.shap_table(shap_df, caption=f"Showing all {len(shap_df)} features")

        # Visual Analysis Plot
        st.markdown("#### Visual Analysis")
        fig = create_feature_plots(row, features_df, compute_shap_on_demand=False)
        st.pyplot(fig)
