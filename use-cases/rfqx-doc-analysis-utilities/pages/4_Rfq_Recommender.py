"""
RFQ recommender and insights page.

This page generates knowledge-graph visualisations and narrative reports to
support supplier recommendation decisions.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app_context import (
    apply_template_theme,
    create_combined_graph_visualization,
    ensure_session_state,
    get_graph_processor,
)

apply_template_theme()
ensure_session_state()

st.title("RFQ Insights & Recommendations")
st.caption(
    "Build knowledge-graph visualisations and AI-generated analyses for processed suppliers."
)

completed_providers = [
    provider for provider in st.session_state.providers if provider.get("status") == "completed"
]

if len(completed_providers) < 2:
    st.warning(
        "Process at least two suppliers before generating recommendations. Visit the Process Supplier Documents page."
    )
    st.stop()

provider_names = [provider["name"] for provider in completed_providers]
st.info(f"Available suppliers: {', '.join(provider_names)} ({len(completed_providers)} total)")

st.subheader("Analysis Options")
selected_for_analysis = st.multiselect(
    "Select suppliers for detailed analysis:",
    options=provider_names,
    default=provider_names,
    help="Choose which suppliers to include in the recommendation analysis.",
)

existing_analysis_providers = st.session_state.get("analysis_metadata", {}).get("providers_analyzed", [])
has_existing_graph = st.session_state.get("graph_image") is not None
has_existing_report = bool(st.session_state.get("comparison_report"))

providers_match = set(selected_for_analysis) == set(existing_analysis_providers)
has_valid_cache = providers_match and has_existing_graph and has_existing_report

if has_valid_cache:
    col_load, col_generate = st.columns(2)
    with col_load:
        load_button = st.button(
            "Load Existing Analysis",
            type="secondary",
            use_container_width=True,
            help="Load a previously generated analysis for the selected suppliers.",
        )
    with col_generate:
        generate_button = st.button(
            "Regenerate Analysis",
            type="primary",
            use_container_width=True,
            help="Generate a fresh analysis for the selected suppliers.",
        )
else:
    load_button = False
    generate_button = st.button(
        "Generate Recommendation Analysis",
        type="primary",
        use_container_width=True,
        help="Generate new analysis for the selected suppliers.",
    )


def _load_existing_analysis() -> None:
    try:
        with st.status("Loading existing analysis...", expanded=False) as status:
            status.update(label="Existing analysis loaded successfully!", state="complete")

        if st.session_state.get("interactive_graph") is not None:
            with st.expander("Knowledge Graph Visualisation", expanded=False):
                st.plotly_chart(st.session_state.interactive_graph, use_container_width=True)
                st.caption("Interactive knowledge graph for selected suppliers.")
        elif st.session_state.get("graph_image") is not None:
            with st.expander("Knowledge Graph Visualisation", expanded=False):
                st.image(st.session_state.graph_image, caption="Knowledge graph preview")

        if st.session_state.get("comparison_report"):
            st.subheader("Comprehensive Analysis Report")
            st.markdown(st.session_state.comparison_report)
            st.success("Existing supplier analysis loaded successfully!")
    except Exception as err:
        st.error(f"Error loading existing analysis: {err}")


def _generate_analysis(provider_names_selection: list[str]) -> None:
    if not provider_names_selection:
        st.warning("Select at least one supplier for analysis.")
        return

    providers_data = {}
    for provider_name in provider_names_selection:
        provider = next(provider for provider in completed_providers if provider["name"] == provider_name)
        providers_data[provider_name] = provider["extracted_data"]

    graph_processor = get_graph_processor()

    with st.status("Generating knowledge graph visualisation...", expanded=False) as status:
        graph_image, interactive_graph = create_combined_graph_visualization(providers_data)
        st.session_state.graph_image = graph_image
        st.session_state.interactive_graph = interactive_graph
        status.update(label="Knowledge graph visualisation completed!", state="complete")

    if st.session_state.interactive_graph is not None:
        with st.expander("Knowledge Graph Visualisation", expanded=False):
            st.plotly_chart(st.session_state.interactive_graph, use_container_width=True)
            st.caption("Interactive knowledge graph for selected suppliers.")

    st.subheader("Comprehensive Analysis Report")
    report_container = st.empty()
    accumulated_report = ""

    with st.status("Generating analysis report...", expanded=False) as status:
        try:
            for chunk in graph_processor.generate_comparison_report_from_graphs_streaming(providers_data):
                accumulated_report += chunk
                report_container.markdown(accumulated_report)

            st.session_state.comparison_report = accumulated_report
            status.update(label="Analysis report completed!", state="complete")
        except Exception as stream_error:
            st.error(f"Streaming error: {stream_error}")
            st.warning("Falling back to standard report generation...")
            with st.spinner("Generating analysis report..."):
                st.session_state.comparison_report = graph_processor.generate_comparison_report_from_graphs(providers_data)
                report_container.markdown(st.session_state.comparison_report)
            status.update(label="Analysis completed (fallback mode)", state="complete")

    st.session_state.analysis_metadata = {
        "providers_analyzed": provider_names_selection,
        "analysis_date": pd.Timestamp.now().isoformat(),
        "total_providers": len(provider_names_selection),
    }

    st.success("Supplier analysis completed successfully!")


if load_button and has_valid_cache:
    _load_existing_analysis()
elif generate_button:
    _generate_analysis(selected_for_analysis)

if st.session_state.get("comparison_report"):
    try:
        from pdf_generator import create_pdf_from_markdown

        pdf_buffer = create_pdf_from_markdown(
            markdown_content=st.session_state.comparison_report,
            project_name=st.session_state.get("current_project") or "RFQ Analysis",
            filename=f"provider_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
        )

        st.download_button(
            label="Download PDF Report",
            data=pdf_buffer.getvalue(),
            file_name=f"provider_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            help="Download the generated analysis report as a PDF file.",
        )
    except ImportError as err:
        st.error(f"PDF generation not available: {err}. Please install required dependencies.")
        st.download_button(
            label="Download Report (Markdown)",
            data=st.session_state.comparison_report,
            file_name=f"provider_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            help="Download the generated analysis report as a Markdown file.",
        )
    except Exception as err:
        st.error(f"Error generating PDF: {err}")
        st.download_button(
            label="Download Report (Markdown)",
            data=st.session_state.comparison_report,
            file_name=f"provider_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            help="Download the generated analysis report as a Markdown file.",
        )

if completed_providers and st.session_state.get("analysis_metadata"):
    st.divider()
    st.subheader("Supplier-Specific Insights")

    for provider in completed_providers:
        if provider["name"] in (selected_for_analysis or []):
            with st.expander(f"Insights: {provider['name']}", expanded=False):
                metadata = provider["extracted_data"].get("_metadata", {})
                col_left, col_right = st.columns(2)
                with col_left:
                    st.metric(
                        "Files Processed",
                        f"{metadata.get('valid_files', 0)}/{metadata.get('total_files', 0)}",
                    )
                    st.metric(
                        "Attributes Extracted",
                        f"{metadata.get('found_fields', 0)}/{metadata.get('total_fields', 0)}",
                    )
                with col_right:
                    st.metric(
                        "Content Size",
                        f"{metadata.get('aggregated_tokens', 0):,} tokens",
                    )
                    if metadata.get("file_errors"):
                        st.error(f"{len(metadata['file_errors'])} file errors")

                extracted_data = provider["extracted_data"]
                if "dynamically_fetched_features" in extracted_data:
                    st.write("**Dynamic Features Found:**")
                    for key, value in extracted_data["dynamically_fetched_features"].items():
                        if value and value != "Not Found":
                            st.write(f"• **{key.replace('_', ' ').title()}:** {value[:200]}...")

                if "manually_requested_features" in extracted_data:
                    st.write("**Custom Attributes:**")
                    for key, value in extracted_data["manually_requested_features"].items():
                        if value and value != "Not Found":
                            st.write(f"• **{key.replace('_', ' ').title()}:** {value[:200]}...")
