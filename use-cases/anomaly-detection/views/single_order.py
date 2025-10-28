from __future__ import annotations

"""Single sales order analysis view."""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import pandas as pd
import streamlit as st

from services import ai_classification, model_runtime, order_selection, shap_explanations
from utils import anomaly, formatting, state
from views import components, forms, state_sync
from utils.explanation_parser import parse_rule_based_explanation_with_model_filtering
from visualization.feature_analysis import create_feature_plots


@dataclass
class OrderContext:
    row: pd.Series
    shap_key: str
    binary_key: str
    ai_key: str
    results_directory: Optional[Path]


def render_single_order_view(results_df: pd.DataFrame, features_df: pd.DataFrame, results_directory: Optional[Path] = None) -> None:
    st.header("Single Sales Order Analysis")

    selected_key = forms.order_search_form()

    st.markdown("""
    <style>
        .metric-container {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .rule-item {
            background-color: #fff3cd;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            border-left: 4px solid #ffc107;
        }
        .shap-positive {
            color: #28a745;
            font-weight: bold;
        }
        .shap-negative {
            color: #dc3545;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

    if st.button("Select Random Anomalous Order", use_container_width=True, key="random_anomaly_btn", type="secondary"):
        random_key = order_selection.select_random_anomalous_order(results_df)
        if random_key:
            state_sync.set_current_order(random_key.document_number, random_key.document_item)
            state.sync_url_params()
        else:
            st.info("No anomalous orders found in the dataset.")

    order_context = _resolve_order_context(results_df, features_df, results_directory)

    if not order_context:
        st.info("Please enter a Sales Document Number and Item, or click the random button to view an order.")
        return

    _render_order_details(order_context.row)
    _trigger_ai_classification(order_context, features_df)
    _render_status(order_context.row, order_context.binary_key)
    _render_ai_actions(order_context, features_df)
    _render_rule_based_explanations(order_context.row)
    _render_shap_explanations(order_context.row, features_df, order_context.shap_key)
    _render_feature_table_and_plots(order_context.row, features_df, order_context.shap_key)


def _resolve_order_context(results_df: pd.DataFrame, features_df: pd.DataFrame, results_directory: Optional[Path]) -> Optional[OrderContext]:
    doc_num = st.session_state.get("search_doc_number")
    doc_item = st.session_state.get("search_doc_item")

    if not doc_num or not doc_item:
        return None

    row = order_selection.find_order(results_df, doc_num, doc_item)
    if row is None:
        st.warning("Sales order not found. Please check the document number and item.")
        return None

    normalized_doc = order_selection.normalize_identifier(doc_num)
    normalized_item = order_selection.normalize_identifier(doc_item)

    shap_key = f"on_demand_shap_{normalized_doc}_{normalized_item}"
    binary_key = f"ai_binary_{normalized_doc}_{normalized_item}"
    ai_key = f"ai_explanation_{normalized_doc}_{normalized_item}"

    if shap_key not in st.session_state:
        computed_shap = model_runtime.compute_shap(row, features_df, results_directory)
        shap_df = shap_explanations.build_shap_dataframe(computed_shap or row.get("shap_explanation"), features_df, row)
        st.session_state[shap_key] = shap_df

    if shap_key in st.session_state and st.session_state[shap_key] is None:
        computed_shap = model_runtime.compute_shap(row, features_df, results_directory)
        shap_df = shap_explanations.build_shap_dataframe(computed_shap or row.get("shap_explanation"), features_df, row)
        st.session_state[shap_key] = shap_df

    return OrderContext(row=row, shap_key=shap_key, binary_key=binary_key, ai_key=ai_key, results_directory=results_directory)


def _render_order_details(row: pd.Series) -> None:
    st.subheader("Order Details")
    details = {
        "Sales Document": f"{order_selection.normalize_identifier(row['Sales Document Number'])} - {order_selection.normalize_identifier(row['Sales Document Item'])}",
        "Customer": row['Sold To number'],
        "Ship-To": row['Ship-To Party'],
        "Material Number": row['Material Number'],
        "Material Description": row['Material Description'],
        "Order Date": row['Sales Document Created Date'].strftime('%Y-%m-%d') if pd.notna(row['Sales Document Created Date']) else 'N/A',
        "Customer PO": row.get('Customer PO number', 'N/A'),
    }

    col1, col2 = st.columns(2)
    items = list(details.items())
    midpoint = len(items) // 2
    with col1:
        for key, value in items[:midpoint]:
            st.write(f"**{key}:** {value}")
    with col2:
        for key, value in items[midpoint:]:
            st.write(f"**{key}:** {value}")


def _trigger_ai_classification(context: OrderContext, features_df: pd.DataFrame) -> None:
    if context.binary_key in st.session_state:
        return

    with st.spinner("Running AI Anomaly Detection..."):
        result, _ = ai_classification.run_binary_classification(context.row, features_df)
        st.session_state[context.binary_key] = result


def _render_status(row: pd.Series, binary_key: str) -> None:
    ai_result = st.session_state.get(binary_key, "Loading...")
    is_ai_anomaly = "Anomalous" in ai_result or "Anomaly" in ai_result
    is_blocked = row.get('blocked_by_business_rules', False)

    status_text = 'ANOMALY DETECTED' if is_ai_anomaly else 'NORMAL ORDER'
    if is_blocked:
        status_text += ' - BLOCKED BY BUSINESS RULES'

    bg_color = '#ffcccc' if is_ai_anomaly else '#ccffcc'
    if is_blocked and not is_ai_anomaly:
        bg_color = '#fff4cc'

    components.anomaly_status_card(status_text, bg_color)

    if is_blocked and row.get('business_rules_explanation'):
        violations = [v.strip() for v in row['business_rules_explanation'].split(';') if v.strip()]
        if violations:
            st.markdown("**Business Rules Violation Details:**")
            for violation in violations:
                st.markdown(f"- {violation}")


def _render_ai_actions(context: OrderContext, features_df: pd.DataFrame) -> None:
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("Learn More", key=f"learn_more_{context.ai_key}"):
            st.session_state.pop(context.ai_key, None)
            with st.spinner("Generating AI explanation..."):
                explanation = ai_classification.generate_full_explanation(context.row, features_df)
                st.session_state[context.ai_key] = explanation

    with col2:
        if st.button("Send for Approval", key=f"approval_{context.ai_key}"):
            st.success("Sales Order sent for approval")

    with col3:
        if st.button("Edit Order", key=f"edit_{context.ai_key}"):
            st.info("Order ready to be edited in S/4HANA")

    if context.ai_key in st.session_state:
        st.markdown("### AI Analysis Details")
        st.markdown(st.session_state[context.ai_key])


def _render_rule_based_explanations(row: pd.Series) -> None:
    explanation = row.get('anomaly_explanation')
    if not explanation or explanation == 'N/A':
        return

    st.markdown("### Rule-Based Anomaly Flags")
    rules = parse_rule_based_explanation_with_model_filtering(explanation)
    if rules:
        for rule in rules:
            st.markdown(
                f"- **{rule['feature_display_name']}:** {formatting.format_feature_value(rule['actual_value'], rule['feature_display_name'])}"
                f" ({rule['status']}). Expected: {rule['expected_value']}"
            )


def _render_shap_explanations(row: pd.Series, features_df: pd.DataFrame, shap_key: str) -> None:
    shap_df = st.session_state.get(shap_key)
    if shap_df is None:
        shap_df = shap_explanations.build_shap_dataframe(row.get("shap_explanation"), features_df, row)
        st.session_state[shap_key] = shap_df

    if shap_df is not None and not shap_df.empty:
        st.markdown("### Key Contributing Features")
        components.shap_table(shap_df, caption=f"Showing all {len(shap_df)} features")


def _render_feature_comparison(row: pd.Series, features_df: pd.DataFrame) -> None:
    st.markdown("### Feature Comparison Table")
    st.dataframe(_create_feature_comparison_table(row, features_df), use_container_width=True, hide_index=True)


def _render_feature_table_and_plots(row: pd.Series, features_df: pd.DataFrame, shap_key: str) -> None:
    """Render the left-right layout with comparison table and visual analysis plot.

    Left: Table comparing actual values vs typical/historical ranges.
    Right: Matplotlib figure created by create_feature_plots for the same order.
    """
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Feature Comparison Table")
        st.dataframe(_create_feature_comparison_table(row, features_df), use_container_width=True, hide_index=True)

    with col_right:
        st.subheader("Visual Analysis")
        # Use pre-computed SHAP text when available; otherwise rely on row shap_explanation
        pre_computed_shap = None
        fig = create_feature_plots(
            row,
            features_df,
            save_for_ai_analysis=False,
            compute_shap_on_demand=False,
            pre_computed_shap=pre_computed_shap,
        )
        st.pyplot(fig)


def _create_feature_comparison_table(row: pd.Series, features_df: pd.DataFrame) -> pd.DataFrame:
    key_features = [
        ('Sales Order item qty', 'p05', 'p95', 'Quantity'),
        ('Unit Price', 'price_p05', 'price_p95', 'Unit Price'),
        ('Order item value', 'expected_order_item_value', None, 'Order Value'),
        ('current_month_total_qty', 'monthly_qty_p05', 'monthly_qty_p95', 'Monthly Volume'),
        ('fulfillment_duration_days', 'fulfillment_p05', 'fulfillment_p95', 'Fulfillment Days'),
        ('qty_z_score', None, None, 'Quantity Z-Score'),
        ('qty_deviation_from_mean', None, None, 'Qty Deviation from Mean'),
    ]

    comparison_data = []

    for feature, range_low, range_high, display_name in key_features:
        if feature in row:
            actual = formatting.format_value(row[feature], feature)
            expected_range = "N/A"
            if feature == 'qty_deviation_from_mean':
                expected_range = "0 (Expected Deviation)"
            elif feature == 'qty_z_score':
                expected_range = "0 (Typical: -2 to +2)"
            elif range_low and range_high and range_low in row and range_high in row:
                if pd.notna(row[range_low]) and pd.notna(row[range_high]):
                    typical_low = formatting.format_value(row[range_low], range_low)
                    typical_high = formatting.format_value(row[range_high], range_high)
                    expected_range = f"{typical_low} - {typical_high}"
            elif range_low and range_low in row and pd.notna(row[range_low]):
                expected_range = formatting.format_value(row[range_low], range_low)
            comparison_data.append({
                'Feature': display_name,
                'Actual Value': actual,
                'Expected Range': expected_range,
            })

    return pd.DataFrame(comparison_data)
