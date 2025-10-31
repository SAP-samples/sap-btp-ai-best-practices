from __future__ import annotations

"""Fine tuning and model configuration view."""

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from services.data_loader import find_best_results_directory


def render_fine_tuning_view(features_df: pd.DataFrame, results_directory: Optional[Path] = None) -> None:
    """Render the fine tuning tab."""

    st.header("Fine Tuning & Model Configuration")

    if features_df is None or features_df.empty:
        st.warning("Feature dataset unavailable. Cannot display statistics.")
        return

    _render_dataset_statistics(features_df)
    st.markdown("---")
    _render_feature_selection()
    st.markdown("---")
    _render_model_configuration()
    st.markdown("---")
    _render_training_section()
    st.markdown("---")
    _render_model_information(results_directory)


def _render_dataset_statistics(features_df: pd.DataFrame) -> None:
    st.subheader("Current Test Dataset Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", f"{len(features_df):,}")

    with col2:
        anomaly_rate = _get_current_anomaly_rate()
        st.metric("Current Anomaly Rate", f"{anomaly_rate:.2%}" if anomaly_rate is not None else "N/A")

    with col3:
        unique_customers = features_df["Sold To number"].nunique()
        st.metric("Unique Customers", f"{unique_customers:,}")

    with col4:
        unique_materials = features_df["Material Number"].nunique()
        st.metric("Unique Materials", f"{unique_materials:,}")


def _get_current_anomaly_rate() -> float | None:
    latest_results_dir = find_best_results_directory()
    if not latest_results_dir:
        return None

    metadata_file = Path(latest_results_dir) / "models" / "stratified_models_metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, "r") as handle:
                metadata = json.load(handle)
                return metadata.get("anomaly_rate")
        except Exception:
            pass

    results_path = Path(latest_results_dir) / "anomaly_detection_results.csv"
    if results_path.exists():
        try:
            results_df = pd.read_csv(results_path)
            return results_df["predicted_anomaly"].sum() / len(results_df)
        except Exception:
            return None
    return None


def _render_feature_selection() -> None:
    st.subheader("Feature Selection")
    st.markdown("Select which features to include in the anomaly detection model training:")

    feature_categories = {
        "Quantity-Based Features": [
            ("qty_z_score", "Z-score deviation from historical mean - captures statistical outliers"),
            ("qty_deviation_from_mean", "Raw deviation from mean - absolute measure of unusual quantities"),
            ("Sales Order item qty", "Raw quantity - base measure for volume anomalies"),
            ("current_month_total_qty", "Monthly accumulation - detects volume breaches"),
        ],
        "Pricing Features": [
            ("Unit Price", "Raw unit price - base measure for pricing anomalies"),
            ("expected_order_item_value", "Expected value calculation - helps detect value mismatches"),
        ],
        "Temporal Features": [
            ("fulfillment_duration_days", "Delivery time - unusual processing times"),
        ],
        "Boolean Anomaly Flags": [
            ("is_first_time_cust_material_order", "First-time orders - new customer-product combinations"),
            ("is_rare_material", "Rare drug indicators - unusual product requests"),
            ("is_qty_outside_typical_range", "Quantity outliers - statistical anomalies"),
            ("is_suspected_duplicate_order", "Potential duplicates - operational anomalies"),
            ("is_monthly_qty_outside_typical_range", "Monthly volume breaches - accumulation anomalies"),
            ("is_unusual_unit_price", "Price outliers - pricing anomalies"),
            ("is_value_mismatch_price_qty", "Value calculation errors - data quality issues"),
            ("is_unusual_fulfillment_time", "Delivery time outliers - operational anomalies"),
        ],
    }

    if "selected_features" not in st.session_state:
        st.session_state.selected_features = [feature for group in feature_categories.values() for feature, _ in group]

    for category_name, features in feature_categories.items():
        st.markdown(f"**{category_name}**")
        for feature_name, description in features:
            current_value = feature_name in st.session_state.selected_features
            checked = st.checkbox(
                f"**{feature_name.replace('_', ' ').title()}**",
                value=current_value,
                help=description,
                key=f"feature_{feature_name}",
            )
            if checked and feature_name not in st.session_state.selected_features:
                st.session_state.selected_features.append(feature_name)
            elif not checked and feature_name in st.session_state.selected_features:
                st.session_state.selected_features.remove(feature_name)
        st.markdown("")

    st.markdown(
        f"**Selected Features: {len(st.session_state.selected_features)} / {sum(len(v) for v in feature_categories.values())}**"
    )


def _render_model_configuration() -> None:
    st.subheader("Model Configuration")

    with st.expander("Isolation Forest Parameters", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.number_input(
                "Number of Estimators",
                min_value=50,
                max_value=500,
                value=150,
                step=10,
                help="Number of trees in the forest. More trees = better performance but slower training.",
                key="n_estimators",
            )

        with col2:
            st.selectbox(
                "Max Samples per Tree",
                options=[256, 512, 1024, "auto"],
                index=1,
                key="max_samples",
                help="Maximum number of samples used to train each tree.",
            )

        st.checkbox(
            "Use Auto Contamination Rate",
            value=True,
            help="Let the algorithm automatically determine the optimal contamination rate",
            key="use_auto_contamination",
        )

        if not st.session_state.use_auto_contamination:
            st.slider(
                "Manual Contamination Rate",
                min_value=0.01,
                max_value=0.20,
                value=0.05,
                step=0.01,
                format="%.2f",
                key="contamination_rate",
                help="Expected proportion of anomalies in the dataset",
            )
        else:
            st.slider(
                "Manual Contamination Rate",
                min_value=0.01,
                max_value=0.20,
                value=0.05,
                step=0.01,
                format="%.2f",
                disabled=True,
                key="contamination_rate_disabled",
                help="Expected proportion of anomalies (disabled when auto mode is enabled)",
            )

        st.checkbox(
            "Enable Customer Stratification",
            value=True,
            key="enable_stratified",
            help="Train separate models for different customer tiers (recommended)",
        )

        st.info(
            "ℹ**Note**: Direct anomaly score threshold adjustment is not available. The model uses statistical thresholds based on the contamination rate."
        )


def _render_training_section() -> None:
    st.subheader("Model Training")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("**Ready to retrain the model with your selected configuration?**")
        st.markdown(f"• **Features selected**: {len(st.session_state.selected_features)}")
        st.markdown(f"• **Customer stratification**: {'Enabled' if st.session_state.enable_stratified else 'Disabled'}")
        contamination = "auto" if st.session_state.use_auto_contamination else st.session_state.contamination_rate
        st.markdown(f"• **Contamination rate**: {contamination}")
        st.markdown(f"• **Estimators**: {st.session_state.n_estimators}")

    with col2:
        st.button("Retrain Model", type="primary", use_container_width=True, key="retrain", disabled=True)
        st.caption("For this deployment, retraining is disabled")


def _render_model_information(results_directory: Optional[Path]) -> None:
    st.subheader("Current Model Information")

    try:
        from model_service import model_service

        if results_directory:
            model_loaded = model_service.load_models(str(results_directory))
        else:
            model_loaded = model_service.load_models("")
        model_info = model_service.get_model_info()
    except Exception:
        model_loaded = False
        model_info = {}

    if model_loaded:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Loaded Model Details:**")
            st.write(f"• **Type**: {model_info.get('model_type', 'Unknown')}")
            st.write(f"• **Stratified**: {'Yes' if model_info.get('is_stratified', False) else 'No'}")
            st.write(f"• **Feature Count**: {model_info.get('feature_count', 0)}")
        with col2:
            st.markdown("**Model Features:**")
            features = model_info.get("features", [])
            if features:
                for i, feature in enumerate(features[:10], 1):
                    st.write(f"{i}. {feature}")
                if len(features) > 10:
                    st.write(f"... and {len(features) - 10} more")
            else:
                st.write("No feature information available")
    else:
        if results_directory:
            st.warning(f"No compatible models found in {results_directory}. Train a new model to enable on-demand SHAP computation.")
        else:
            st.warning("No models currently loaded. Train a new model to enable on-demand SHAP computation.")
