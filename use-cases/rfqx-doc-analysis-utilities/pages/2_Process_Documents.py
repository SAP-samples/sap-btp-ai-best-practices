"""
Supplier document processing page.

Migrated from the original Streamlit tab to align with the SAP template layout.
"""

from __future__ import annotations

import concurrent.futures

import streamlit as st

from app_context import apply_template_theme, ensure_session_state, get_comparator
from rfq_schema import RFQ_EXTRACTION_SCHEMA
from ui_components import (
    FeatureConfiguration,
    FileUploadHelper,
    ProviderManager,
    StatusDisplay,
)

apply_template_theme()
ensure_session_state()

comparator = get_comparator()

st.title("Process Supplier Documents")
st.caption(
    "Upload supplier RFQ documents, select extraction attributes, and launch processing jobs."
)

if not st.session_state.current_project:
    st.warning("Please create or load a project from the Project Setup page before processing documents.")
    st.stop()

st.info(f"**Current Project:** {st.session_state.current_project}")

# Step 1: Attribute Extraction Configuration
filtered_schema, dynamic_extraction, has_selected_features = FeatureConfiguration.render_toggleable_feature_configuration(
    RFQ_EXTRACTION_SCHEMA
)

st.divider()

# Step 2: Supplier count selection
num_providers = ProviderManager.render_provider_count_selector()

# Step 3: Supplier sections
st.subheader("Supplier Document Upload")
providers_ready = True
for index in range(num_providers):
    provider = ProviderManager.render_provider_section(index)
    if not provider.get("files"):
        providers_ready = False

FileUploadHelper.render_file_upload_help()

st.divider()

# Show processing status if suppliers have been processed
if any(provider["status"] != "pending" for provider in st.session_state.providers):
    st.subheader("Processing Status")
    for index, provider in enumerate(st.session_state.providers):
        with st.expander(
            f"Status: {provider['name']}",
            expanded=provider["status"] == "processing",
        ):
            StatusDisplay.render_provider_status(provider)

# Processing controls
col_action, col_hint = st.columns([1, 3])
with col_action:
    ready_to_process = providers_ready and has_selected_features
    process_button = st.button(
        "Process All Suppliers",
        type="primary",
        disabled=not ready_to_process,
        use_container_width=True,
    )

with col_hint:
    if not has_selected_features:
        st.caption("Please select at least one extraction feature before processing.")
    elif not providers_ready:
        st.caption("Upload files for each supplier to enable processing.")
    elif any(provider["status"] == "processing" for provider in st.session_state.providers):
        st.caption("Processing in progress...")
    else:
        ready_count = sum(1 for provider in st.session_state.providers if provider.get("files"))
        selected_features = sum(
            len(fields)
            for fields in filtered_schema.values()
            if isinstance(fields, dict)
        ) + (1 if dynamic_extraction else 0)
        st.caption(f"Ready to process {ready_count} suppliers with {selected_features} selected features.")


def _handle_processing() -> None:
    """Process all configured suppliers using the comparator."""
    if not filtered_schema and not dynamic_extraction:
        st.error("Cannot process documents: No features are selected for extraction.")
        st.stop()

    providers_to_process = [
        (index, provider)
        for index, provider in enumerate(st.session_state.providers)
        if provider.get("files")
    ]

    if not providers_to_process:
        st.error("No suppliers have files to process.")
        st.stop()

    for index, _ in providers_to_process:
        st.session_state.providers[index]["status"] = "processing"

    progress_placeholder = st.empty()
    status_containers = {
        index: st.expander(
            f"Processing {provider['name']}",
            expanded=True,
        )
        for index, provider in providers_to_process
    }

    def process_single_provider(provider_index, provider_data):
        try:
            result = comparator.process_provider_documents(
                uploaded_files=provider_data["files"],
                provider_name=provider_data["name"],
                custom_features=st.session_state.get("custom_features", []),
                enable_dynamic_extraction=dynamic_extraction,
                filtered_schema=filtered_schema,
            )
            return provider_index, result, None
        except Exception as err:  # pragma: no cover - runtime safety
            return provider_index, None, str(err)

    completed_count = 0
    error_count = 0
    max_workers = min(len(providers_to_process), 5)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_provider, idx, provider): (idx, provider)
            for idx, provider in providers_to_process
        }

        progress_placeholder.info(
            f"Processing {len(providers_to_process)} suppliers in parallel (up to {max_workers} simultaneously)..."
        )

        for future in concurrent.futures.as_completed(futures):
            provider_index, provider = futures[future]
            result_index, result, error = future.result()

            with status_containers[provider_index]:
                if error:
                    st.error(f"Processing failed: {error}")
                    st.session_state.providers[provider_index]["status"] = "error"
                    st.session_state.providers[provider_index]["error_message"] = error
                    error_count += 1
                else:
                    st.session_state.providers[provider_index]["status"] = "completed"
                    st.session_state.providers[provider_index]["processed_content"] = result.get(
                        "_raw_content", ""
                    )
                    st.session_state.providers[provider_index]["extracted_data"] = result
                    st.session_state.providers[provider_index]["token_count"] = result.get(
                        "_metadata", {}
                    ).get("aggregated_tokens", 0)
                    StatusDisplay.render_provider_status(st.session_state.providers[provider_index])
                    st.success("Supplier processed successfully.")
                    completed_count += 1

            progress_placeholder.info(
                f"Completed {completed_count}/{len(providers_to_process)} suppliers "
                f"(errors: {error_count})."
            )

    progress_placeholder.success(
        f"Processing finished: {completed_count} succeeded, {error_count} failed."
    )

    st.session_state.project_loaded = True
    st.success("All supplier processing tasks completed.")
    st.rerun()


if process_button and providers_ready and has_selected_features:
    _handle_processing()
