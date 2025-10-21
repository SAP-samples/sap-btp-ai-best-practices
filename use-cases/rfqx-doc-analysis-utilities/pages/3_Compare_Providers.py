"""
Supplier comparison page.

Provides the side-by-side comparison functionality originally found in the
"Compare Documents" tab.
"""

from __future__ import annotations

import streamlit as st

from app_context import apply_template_theme, ensure_session_state, get_comparator
from ui_components import StatusDisplay

apply_template_theme()
ensure_session_state()

comparator = get_comparator()

st.title("Compare Suppliers")
st.caption(
    "Generate structured comparisons across processed suppliers to highlight key differences."
)

completed_providers = [
    provider for provider in st.session_state.providers if provider.get("status") == "completed"
]

if len(completed_providers) < 2:
    st.warning(
        "Please process at least two suppliers before running a comparison. Visit the Process Supplier Documents page."
    )
    if st.session_state.providers:
        st.subheader("Current Supplier Status")
        for provider in st.session_state.providers:
            with st.expander(
                f"{provider['name']} – {provider['status'].title()}",
                expanded=False,
            ):
                StatusDisplay.render_provider_status(provider)
    st.stop()

provider_names = [provider["name"] for provider in completed_providers]

st.info(f"Available suppliers: {', '.join(provider_names)}")

st.subheader("Select Suppliers to Compare")
selected_provider_names = st.multiselect(
    "Choose suppliers to compare (select two or more):",
    options=provider_names,
    default=provider_names[: min(len(provider_names), 3)],
    help="Select at least two suppliers for comparison.",
)

if st.button("Compare Suppliers", type="primary", use_container_width=True):
    if len(selected_provider_names) < 2:
        st.warning("Select at least two suppliers to run the comparison.")
    else:
        selected_data = []
        for name in selected_provider_names:
            provider_data = next(
                provider["extracted_data"]
                for provider in completed_providers
                if provider["name"] == name
            )
            selected_data.append(provider_data)

        st.subheader(f"Comparing: {', '.join(selected_provider_names)}")
        comparison_result = comparator.compare_providers(selected_data)

        if "error" in comparison_result:
            st.error(f"Comparison failed: {comparison_result['error']}")
        else:
            st.session_state.comparison_result = comparison_result
            StatusDisplay.render_comparison_summary(comparison_result)
            st.divider()

            field_comparison = comparison_result.get("field_by_field_comparison", {})
            category_order = [
                "project_information",
                "key_dates_deadlines",
                "scope_technical_requirements",
                "supplier_requirements",
                "evaluation_criteria",
                "pricing_payment",
                "legal_contractual",
                "compliance_exclusion_grounds",
                "sustainability_social_value",
                "contract_management_reporting",
                "manually_requested_features",
                "dynamically_fetched_features",
            ]

            for category in category_order:
                if category in field_comparison:
                    features = field_comparison[category]
                    with st.expander(category.replace("_", " ").title(), expanded=False):
                        for feature_name, values in features.items():
                            st.write(f"**{feature_name.replace('_', ' ').title()}:**")
                            columns = st.columns(len(selected_provider_names))
                            for idx, provider_name in enumerate(selected_provider_names):
                                provider_key = f"provider_{idx + 1}"
                                provider_value = values.get(provider_key, "Not Found")
                                with columns[idx]:
                                    if provider_value == "Not Found":
                                        st.markdown(
                                            f"**{provider_name}:** "
                                            "<span style='color: red;'><i>Not Found</i></span>",
                                            unsafe_allow_html=True,
                                        )
                                    else:
                                        st.write(f"**{provider_name}:** {provider_value}")

                        st.markdown("---")
