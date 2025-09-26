"""
SAP RFQx Document Analysis Application

A Streamlit application for analyzing and comparing RFQ documents using
the simplified RAG approach with direct PDF-to-LLM processing.
"""

import os
import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import io
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import networkx as nx
import concurrent.futures
import time
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="SAP RFQx Document Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Import the main components
from main import SimplifiedRFQComparator
from rfq_schema import RFQ_EXTRACTION_SCHEMA, create_dynamic_schema, get_filtered_extraction_instructions
from graph_processor import GraphProcessor
from ui_components import (ProviderManager, FeatureConfiguration, StatusDisplay, 
                         FileUploadHelper, initialize_session_state)
from file_processor import FileProcessor
from project_manager import ProjectManager
from country_risk_manager import CountryRiskManager

# Application title
st.title("SAP RFQx Document Analysis Application")

# Initialize the comparator
@st.cache_resource
def get_comparator():
    return SimplifiedRFQComparator()

comparator = get_comparator()

# Initialize the project manager
@st.cache_resource
def get_project_manager():
    return ProjectManager()

project_manager = get_project_manager()

# Initialize the country risk manager
@st.cache_resource
def get_country_risk_manager():
    return CountryRiskManager()

country_risk_manager = get_country_risk_manager()

# Initialize the graph processor
@st.cache_resource
def get_graph_processor():
    return GraphProcessor(llm_client=get_comparator().client)

graph_processor = get_graph_processor()

def create_combined_graph_visualization(documents_data: Dict[str, Dict[str, Any]]) -> Tuple[io.BytesIO, go.Figure]:
    """
    Creates a combined visualization of knowledge graphs from multiple documents.
    
    Args:
        documents_data: Dictionary with document names as keys and extracted JSON data as values.
        
    Returns:
        Tuple of (BytesIO buffer with WebP image, Plotly Figure for interactive display).
    """
    # Use cached GraphProcessor
    graph_processor = get_graph_processor()
    
    # Create graphs for all documents
    graphs = {name: graph_processor.create_graph_from_json(data) for name, data in documents_data.items()}
    
    # Create a combined graph for visualization
    combined_graph = nx.DiGraph()
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'plum', 'orange', 'cyan']
    color_map = {}
    
    for i, (doc_name, graph) in enumerate(graphs.items()):
        doc_color = colors[i % len(colors)]
        
        # Add all nodes from this document's graph
        for node, attrs in graph.nodes(data=True):
            combined_graph.add_node(node, **attrs, document=doc_name, color=doc_color)
            color_map[node] = doc_color
            
        # Add all edges from this document's graph
        for u, v, attrs in graph.edges(data=True):
            combined_graph.add_edge(u, v, **attrs, document=doc_name)
    
    # Create optimized high-quality visualization
    plt.figure(figsize=(24, 18))  # Balanced size for quality vs file size
    pos = nx.spring_layout(combined_graph, seed=42, k=2.0, iterations=100)  # Better spacing
    
    # Get node labels (truncated)
    node_labels = {n: combined_graph.nodes[n].get('label', n) for n in combined_graph.nodes()}
    
    # Get edge labels
    edge_labels = {(u, v): attrs.get('label', '') for u, v, attrs in combined_graph.edges(data=True)}
    
    # Draw nodes with colors based on document - optimized sizes
    node_colors = [color_map.get(node, 'gray') for node in combined_graph.nodes()]
    nx.draw(combined_graph, pos, labels=node_labels, with_labels=True, 
            node_color=node_colors, node_size=4000, font_size=9,  # Balanced sizing
            edge_color='gray', arrowsize=22, font_weight='bold',
            linewidths=1.5, width=1.5)  # Moderate edge thickness
    
    # Draw edge labels with readable font
    nx.draw_networkx_edge_labels(combined_graph, pos, edge_labels=edge_labels, 
                                 font_color='red', font_size=7)
    
    # Create legend with appropriate markers
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=colors[i % len(colors)], 
                                 markersize=12, label=doc_name)
                      for i, doc_name in enumerate(documents_data.keys())]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1), 
               fontsize=10)
    
    plt.title("Combined Knowledge Graph - RFQ Documents Comparison", 
              fontsize=18, fontweight='bold')
    
    # Save to BytesIO buffer with optimized DPI
    buf = io.BytesIO()
    plt.savefig(buf, format='webp', dpi=200, bbox_inches='tight', facecolor='white',
                pad_inches=0.3)
    buf.seek(0)
    plt.close()  # Important: close the figure to free memory
    
    # Create interactive Plotly graph
    interactive_fig = graph_processor.create_interactive_graph(graphs)
    
    return buf, interactive_fig

# Create tabs
tab_setup, tab1, tab2, tab3, tab4 = st.tabs(["Setup", "Process Documents", "Compare Documents", "RFQ Recommender", "Document Chat"])

# Initialize session state
initialize_session_state()

# Add project-related session state
if 'current_project' not in st.session_state:
    st.session_state.current_project = None
if 'project_loaded' not in st.session_state:
    st.session_state.project_loaded = False

# Setup Tab: Project Management
with tab_setup:
    st.header("Project Management")
    
    # Create two columns for project actions
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Create New Project")
        
        # Project name input
        new_project_name = st.text_input(
            "Project Name",
            placeholder="Enter a descriptive name for your project",
            help="Use alphanumeric characters, spaces, hyphens, and underscores only"
        )
        
        # Create project button
        if st.button("Create Project", type="primary", disabled=not new_project_name):
            try:
                # Create the project
                project_manager.create_project(new_project_name)
                
                # Reset session state for new project
                initialize_session_state()
                st.session_state.current_project = new_project_name
                st.session_state.project_loaded = True
                
                st.success(f"Project '{new_project_name}' created successfully!")
                st.info("You can now proceed to the 'Process Documents' tab to upload and analyze documents.")
                
                # Save initial state
                project_manager.save_project(new_project_name, st.session_state)
                
                # Rerun to update UI
                st.rerun()
                
            except ValueError as e:
                st.error(f"{str(e)}")
            except Exception as e:
                st.error(f"Failed to create project: {str(e)}")
    
    with col2:
        st.subheader("Load Existing Project")
        
        # Get list of projects
        projects = project_manager.list_projects()
        
        if projects:
            # Create project selection dropdown
            project_names = [p['project_name'] for p in projects]
            selected_project = st.selectbox(
                "Select Project",
                options=[""] + project_names,
                format_func=lambda x: "-- Select a project --" if x == "" else x
            )
            
            if selected_project:
                # Show project info
                try:
                    project_info = project_manager.get_project_info(selected_project)
                    
                    with st.expander("Project Details", expanded=True):
                        st.write(f"**Created:** {project_info['created_at'][:10]}")
                        st.write(f"**Last Modified:** {project_info['last_modified'][:10]}")
                        
                        if project_info['providers']:
                            st.write("**Suppliers:**")
                            for provider in project_info['providers']:
                                st.write(f"- {provider['name']} ({provider['file_count']} files)")
                                if provider['files']:
                                    # Show files in a simple indented list
                                    for file in provider['files']:
                                        st.write(f"    {file}")
                        else:
                            st.write("*No suppliers uploaded yet*")
                        
                        if project_info['has_analysis']:
                            st.write("**Analysis Report Available**")
                        if project_info['has_graph']:
                            st.write("**Knowledge Graph Available**")
                    
                    # Load and Delete buttons
                    col_load, col_delete = st.columns(2)
                    
                    with col_load:
                        if st.button("Load Project", type="primary"):
                            try:
                                # Load project state
                                project_state = project_manager.load_project(selected_project)
                                
                                # Update session state
                                for key, value in project_state.items():
                                    if key not in ['comparison_report', 'graph_image']:  # Handle these specially
                                        st.session_state[key] = value
                                
                                # Handle special cases
                                if 'comparison_report' in project_state:
                                    st.session_state.comparison_report = project_state['comparison_report']
                                if 'graph_image' in project_state:
                                    st.session_state.graph_image = project_state['graph_image']
                                    # Regenerate interactive graph from providers data
                                    if 'providers' in st.session_state and st.session_state.providers:
                                        completed_providers = [p for p in st.session_state.providers if p.get('extracted_data')]
                                        if completed_providers:
                                            providers_data = {p['name']: p['extracted_data'] for p in completed_providers}
                                            _, interactive_graph = create_combined_graph_visualization(providers_data)
                                            st.session_state.interactive_graph = interactive_graph
                                
                                st.session_state.current_project = selected_project
                                st.session_state.project_loaded = True
                                
                                st.success(f"Project '{selected_project}' loaded successfully!")
                                st.info("Navigate to other tabs to continue your analysis.")
                                
                                # Rerun to update UI
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Failed to load project: {str(e)}")
                    
                    with col_delete:
                        if st.button("Delete Project", type="secondary"):
                            st.session_state[f'confirm_delete_{selected_project}'] = True
                    
                    # Confirmation dialog for deletion
                    if st.session_state.get(f'confirm_delete_{selected_project}', False):
                        st.warning(f"Are you sure you want to delete project '{selected_project}'? This action cannot be undone.")
                        col_confirm, col_cancel = st.columns(2)
                        
                        with col_confirm:
                            if st.button("Yes, Delete", type="primary"):
                                try:
                                    project_manager.delete_project(selected_project)
                                    st.success(f"Project '{selected_project}' deleted successfully!")
                                    
                                    # Clear session state if this was the current project
                                    if st.session_state.current_project == selected_project:
                                        initialize_session_state()
                                        st.session_state.current_project = None
                                        st.session_state.project_loaded = False
                                    
                                    # Clear confirmation flag
                                    del st.session_state[f'confirm_delete_{selected_project}']
                                    
                                    # Rerun to update UI
                                    st.rerun()
                                    
                                except Exception as e:
                                    st.error(f"Failed to delete project: {str(e)}")
                        
                        with col_cancel:
                            if st.button("Cancel"):
                                del st.session_state[f'confirm_delete_{selected_project}']
                                st.rerun()
                
                except Exception as e:
                    st.error(f"Error loading project details: {str(e)}")
        else:
            st.info("No existing projects found. Create a new project to get started!")
    
    # Show current project status
    st.markdown("---")
    if st.session_state.current_project:
        st.success(f"**Current Project:** {st.session_state.current_project}")
        
        # Add save button if project is loaded and has been modified
        if st.session_state.project_loaded and st.session_state.providers:
            if st.button("Save Current Progress", type="secondary"):
                try:
                    project_manager.save_project(st.session_state.current_project, st.session_state)
                    st.success("Project saved successfully!")
                except Exception as e:
                    st.error(f"Failed to save project: {str(e)}")
    else:
        st.info("Create a new project or load an existing one to begin your RFQ analysis.")

# Tab 1: Process Documents
with tab1:
    st.header("Multi-Supplier Document Processing")
    
    # Check if a project is loaded
    if not st.session_state.current_project:
        st.warning("Please create or load a project in the 'Setup' tab before processing documents.")
        st.stop()
    
    # Show current project
    st.info(f"**Current Project:** {st.session_state.current_project}")
    
    # Step 1: Attribute Extraction Configuration (moved to top)
    filtered_schema, dynamic_extraction, has_selected_features = FeatureConfiguration.render_toggleable_feature_configuration(RFQ_EXTRACTION_SCHEMA)
    
    st.markdown("---")
    
    # Step 2: Supplier count selection
    num_providers = ProviderManager.render_provider_count_selector()
    
    # Step 3: Supplier sections
    st.subheader("Supplier Document Upload")
    
    # Render provider sections
    providers_ready = True
    for i in range(num_providers):
        provider = ProviderManager.render_provider_section(i)
        if not provider.get('files'):
            providers_ready = False
    
    # Display file upload help
    FileUploadHelper.render_file_upload_help()
    
    # Step 4: Process Documents button
    st.markdown("---")
    
    # Show processing status if any provider has been processed
    if any(provider['status'] != 'pending' for provider in st.session_state.providers):
        st.subheader("Processing Status")
        for i, provider in enumerate(st.session_state.providers):
            with st.expander(f"Status: {provider['name']}", expanded=provider['status'] == 'processing'):
                StatusDisplay.render_provider_status(provider)
    
    # Processing button and logic
    col1, col2 = st.columns([1, 3])
    with col1:
        # Check if both providers and features are ready
        ready_to_process = providers_ready and has_selected_features
        process_button = st.button("Process All Supplier", type="primary", disabled=not ready_to_process, key="process_all_providers")
    
    with col2:
        if not has_selected_features:
            st.caption("Please select at least one feature before processing")
        elif not providers_ready:
            st.caption("Please upload files for all suppliers before processing")
        elif any(provider['status'] == 'processing' for provider in st.session_state.providers):
            st.caption("Processing in progress...")
        else:
            ready_count = sum(1 for provider in st.session_state.providers if provider.get('files'))
            selected_features = sum(
                len(fields) for fields in filtered_schema.values() 
                if isinstance(fields, dict)
            ) + (1 if dynamic_extraction else 0)
            st.caption(f"Ready to process {ready_count} suppliers with {selected_features} selected features")
    
    if process_button and providers_ready and has_selected_features:
        # Validate that features are selected before processing
        if not filtered_schema and not dynamic_extraction:
            st.error("Cannot process documents: No features are selected for extraction!")
            st.stop()
        
        # Process all providers in parallel
        total_providers = len(st.session_state.providers)
        providers_to_process = [(i, provider) for i, provider in enumerate(st.session_state.providers) 
                               if provider.get('files')]
        
        if not providers_to_process:
            st.error("No providers have files to process.")
            st.stop()
        
        # Update all statuses to processing
        for i, provider in providers_to_process:
            st.session_state.providers[i]['status'] = 'processing'
        
        # Create progress placeholder
        progress_placeholder = st.empty()
        status_containers = {}
        
        # Create containers for each provider's status
        for i, provider in providers_to_process:
            status_containers[i] = st.expander(f"Processing {provider['name']}", expanded=True)
        
        # Function to process a single provider
        def process_single_provider(provider_index, provider_data):
            """Process a single provider and return the result with index."""
            try:
                result = comparator.process_provider_documents(
                    uploaded_files=provider_data['files'],
                    provider_name=provider_data['name'],
                    custom_features=st.session_state.get('custom_features', []),
                    enable_dynamic_extraction=dynamic_extraction,
                    filtered_schema=filtered_schema
                )
                return provider_index, result, None
            except Exception as e:
                return provider_index, None, str(e)
        
        # Process all providers in parallel
        completed_count = 0
        error_count = 0
        
        # Determine optimal number of workers (limit to avoid overwhelming the system)
        max_workers = min(len(providers_to_process), 5)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_provider = {
                executor.submit(process_single_provider, i, provider): (i, provider)
                for i, provider in providers_to_process
            }
            
            # Update progress message
            progress_placeholder.info(f"Processing {len(providers_to_process)} suppliers in parallel (up to {max_workers} simultaneously)...")
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_provider):
                provider_index, provider = future_to_provider[future]
                
                try:
                    idx, result, error = future.result()
                    
                    if error:
                        # Handle processing exception
                        st.session_state.providers[idx]['status'] = 'error'
                        st.session_state.providers[idx]['error_message'] = error
                        error_count += 1
                        
                        with status_containers[idx]:
                            st.error(f"Failed to process {provider['name']}: {error}")
                    
                    elif result and ("processing_error" in result or "extraction_error" in result):
                        # Handle processing errors
                        error_msg = result.get('processing_error') or result.get('extraction_error')
                        st.session_state.providers[idx]['status'] = 'error'
                        st.session_state.providers[idx]['error_message'] = error_msg
                        error_count += 1
                        
                        with status_containers[idx]:
                            st.error(f"Failed to process {provider['name']}: {error_msg}")
                    
                    else:
                        # Success
                        st.session_state.providers[idx]['status'] = 'completed'
                        st.session_state.providers[idx]['extracted_data'] = result
                        st.session_state.providers[idx]['token_count'] = result.get('_metadata', {}).get('aggregated_tokens', 0)
                        
                        # Update document_features for backward compatibility
                        st.session_state.document_features[provider['name']] = result
                        completed_count += 1
                        
                        # Show success metrics
                        with status_containers[idx]:
                            metadata = result.get('_metadata', {})
                            total_files = metadata.get('total_files', 0)
                            valid_files = metadata.get('valid_files', 0)
                            found_fields = metadata.get('found_fields', 0)
                            total_fields = metadata.get('total_fields', 0)
                            
                            st.success(f"Successfully processed {provider['name']}")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Files Processed", f"{valid_files}/{total_files}")
                            with col2:
                                st.metric("Attributes Extracted", f"{found_fields}/{total_fields}")
                            with col3:
                                st.metric("Tokens", f"{metadata.get('aggregated_tokens', 0):,}")
                
                except Exception as e:
                    # Handle unexpected errors
                    st.session_state.providers[provider_index]['status'] = 'error'
                    st.session_state.providers[provider_index]['error_message'] = str(e)
                    error_count += 1
                    
                    with status_containers[provider_index]:
                        st.error(f"Unexpected error processing {provider['name']}: {str(e)}")
                
                # Update progress
                processed = completed_count + error_count
                remaining = len(providers_to_process) - processed
                if remaining > 0:
                    progress_placeholder.info(f"Processed {processed}/{len(providers_to_process)} suppliers. {remaining} still processing...")
                else:
                    progress_placeholder.empty()
        
        # Final summary
        if completed_count > 0:
            st.success(f"Processing complete! {completed_count} suppliers processed successfully.")
            if error_count > 0:
                st.warning(f"{error_count} suppliers had errors.")
            
            # Auto-save project if one is loaded
            if st.session_state.current_project:
                try:
                    project_manager.save_project(st.session_state.current_project, st.session_state)
                except Exception as e:
                    st.warning(f"Could not auto-save project: {str(e)}")
        else:
            st.error("No suppliers were processed successfully.")
        
        # Trigger rerun to update the UI
        st.rerun()

# Tab 2: Compare Suppliers
with tab2:
    st.header("Supplier Comparison Analysis")
    
    # Check if providers are processed
    completed_providers = [p for p in st.session_state.providers if p['status'] == 'completed']
    
    if len(completed_providers) < 2:
        st.warning("Please process at least 2 suppliers in the 'Process Documents' tab before comparing.")
        
        # Show current provider status
        if st.session_state.providers:
            st.subheader("Current Supplier Status")
            for provider in st.session_state.providers:
                with st.expander(f"{provider['name']} - {provider['status'].title()}", expanded=False):
                    StatusDisplay.render_provider_status(provider)
    else:
        # Show available providers
        provider_names = [p['name'] for p in completed_providers]
        st.info(f"Available suppliers: {', '.join(provider_names)}")
        
        # Provider selection interface
        st.subheader("Select Suppliers to Compare")
        selected_provider_names = st.multiselect(
            "Choose suppliers to compare (select 2 or more):",
            options=provider_names,
            default=provider_names[:min(len(provider_names), 3)],  # Default to first 3 or all if less than 3
            help="Select at least 2 suppliers for comparison"
        )
        
        # Compare Providers button
        if st.button("Compare Suppliers", type="primary", key="compare_providers"):
            if len(selected_provider_names) >= 2:
                # Get selected provider data
                selected_providers_data = []
                for provider_name in selected_provider_names:
                    provider_data = next(p['extracted_data'] for p in completed_providers if p['name'] == provider_name)
                    selected_providers_data.append(provider_data)
                
                # Display which providers are being compared
                st.subheader(f"Comparing: {', '.join(selected_provider_names)}")
                
                # Generate comparison using the new provider comparison method
                comparison_result = comparator.compare_providers(selected_providers_data)
                
                if "error" in comparison_result:
                    st.error(f"Comparison failed: {comparison_result['error']}")
                else:
                    # Store comparison result for other tabs
                    st.session_state.comparison_result = comparison_result
                    
                    # Display comparison summary
                    StatusDisplay.render_comparison_summary(comparison_result)
                    
                    st.markdown("---")
                    
                    # Display detailed comparison for each category in consistent order
                    field_comparison = comparison_result["field_by_field_comparison"]
                    
                    # Define the consistent ordering based on RFQ schema
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
                        "dynamically_fetched_features"
                    ]
                    
                    # Process categories in the defined order
                    for category_name in category_order:
                        if category_name in field_comparison:
                            features = field_comparison[category_name]
                            with st.expander(f"{category_name.replace('_', ' ').title()}", expanded=False):
                                # Display each feature with provider information in text format
                                for feature_name, values in features.items():
                                    st.write(f"**{feature_name.replace('_', ' ').title()}:**")
                                    
                                    # Create columns for each provider
                                    provider_cols = st.columns(len(selected_provider_names))
                                    for i, provider_name in enumerate(selected_provider_names):
                                        with provider_cols[i]:
                                            provider_key = f"provider_{i+1}"
                                            provider_value = values.get(provider_key, "Not Found")
                                            
                                            if provider_value == "Not Found":
                                                st.markdown(f"**{provider_name}:** <span style='color: red;'><i>Not Found</i></span>", unsafe_allow_html=True)
                                            else:
                                                st.write(f"**{provider_name}:** {provider_value}")
                                    
                                    st.markdown("---")
                    
                    # Add Integrations section (always last)
                    with st.expander("Integrations", expanded=False):
                        # Internal Costs section
                        st.write("**Internal Costs**")
                        provider_cols = st.columns(len(selected_provider_names))
                        for i, provider_name in enumerate(selected_provider_names):
                            with provider_cols[i]:
                                st.write(f"**{provider_name}:**")
                                st.markdown('<a href="#" style="text-decoration: none; color: #0066cc;">Internal Costs Calculation</a>', unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # External Costs section
                        st.write("**External Costs**")
                        provider_cols = st.columns(len(selected_provider_names))
                        for i, provider_name in enumerate(selected_provider_names):
                            with provider_cols[i]:
                                st.write(f"**{provider_name}:**")
                                st.markdown('<a href="#" style="text-decoration: none; color: #0066cc;">External Costs Calculation</a>', unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # Risk Management section
                        st.write("**Risk Management**")
                        provider_cols = st.columns(len(selected_provider_names))
                        for i, provider_name in enumerate(selected_provider_names):
                            with provider_cols[i]:
                                st.write(f"**{provider_name}:**")
                                st.markdown('<a href="#" style="text-decoration: none; color: #0066cc;">Risk Management</a>', unsafe_allow_html=True)
                    
            else:
                st.warning("Please select at least 2 providers to compare.")

# Tab 3: RFQ Recommender
with tab3:
    st.header("Provider Recommendation Analysis")
    
    # Check if providers are processed
    completed_providers = [p for p in st.session_state.providers if p['status'] == 'completed']
    
    if len(completed_providers) < 2:
        st.warning("Please process at least 2 suppliers in the 'Process Documents' tab before generating recommendations.")
        
        # Show processing guidance
        if st.session_state.providers:
            pending_providers = [p for p in st.session_state.providers if p['status'] == 'pending']
            if pending_providers:
                st.info(f"You have {len(pending_providers)} suppliers ready to process. Go to the 'Process Documents' tab to analyze them.")
    else:
        # Show available providers
        provider_names = [p['name'] for p in completed_providers]
        st.info(f"Available suppliers: {', '.join(provider_names)} ({len(completed_providers)} total)")
        
        # Analysis options
        st.subheader("Analysis Options")
        
        # Provider selection for analysis
        selected_for_analysis = st.multiselect(
            "Select suppliers for detailed analysis:",
            options=provider_names,
            default=provider_names,  # Default to all
            help="Choose which suppliers to include in the recommendation analysis"
        )
        
        # Check if analysis already exists for the selected providers
        existing_analysis_providers = st.session_state.get('analysis_metadata', {}).get('providers_analyzed', [])
        has_existing_graph = st.session_state.get('graph_image') is not None
        has_existing_report = st.session_state.get('comparison_report', '') != ''
        
        # Determine if we have valid cached analysis for current selection
        providers_match = set(selected_for_analysis) == set(existing_analysis_providers)
        has_valid_cache = providers_match and has_existing_graph and has_existing_report
        
        # Show appropriate buttons based on cache status
        if has_valid_cache:
            # Show both buttons when we have existing analysis
            col1, col2 = st.columns(2)
            with col1:
                load_button = st.button("Load Existing Analysis", type="secondary", key="load_analysis", 
                                      help="Load previously generated analysis for the selected suppliers", use_container_width=True)
            with col2:
                generate_button = st.button("Regenerate Analysis", type="primary", key="regenerate_analysis", 
                                          help="Generate new analysis for the selected suppliers", use_container_width=True)
        else:
            # Show only generate button when no existing analysis
            generate_button = st.button("Generate Recommendation Analysis", type="primary", key="generate_analysis", 
                                      help="Generate new analysis for the selected suppliers")
            load_button = False
        
        # Handle button actions
        if (has_valid_cache and load_button) or (not has_valid_cache and generate_button) or (has_valid_cache and generate_button):
            if not selected_for_analysis:
                st.warning("Please select at least one suppliers for analysis.")
            else:
                if has_valid_cache and load_button:
                    # Load existing analysis
                    try:
                        with st.status("Loading existing analysis...", expanded=False) as status:
                            status.update(label="Existing analysis loaded successfully!", state="complete")
                        
                    
                        if st.session_state.interactive_graph is not None:
                            with st.expander("Knowledge Graph Visualization", expanded=False):
                                st.plotly_chart(st.session_state.interactive_graph, use_container_width=True)
                        elif st.session_state.graph_image is not None:
                            with st.expander("Knowledge Graph Visualization", expanded=False):
                                st.image(st.session_state.graph_image, ...)
                                st.caption("This visualization shows...")
                        
                        # Display existing report
                        # st.subheader("Comprehensive Analysis Report")
                        st.markdown(st.session_state.comparison_report)
                        
                        st.success("Existing supplier analysis loaded successfully!")
                        
                    except Exception as e:
                        st.error(f"Error loading existing analysis: {str(e)}")
                        
                else:
                    # Generate new analysis (either first time or regenerate)
                    try:
                        # Prepare provider data for analysis
                        providers_data = {}
                        for provider_name in selected_for_analysis:
                            provider = next(p for p in completed_providers if p['name'] == provider_name)
                            providers_data[provider_name] = provider['extracted_data']
                        
                        # Use cached GraphProcessor
                        graph_processor = get_graph_processor()
                        
                        # Show progress for graph visualization generation
                        with st.status("Generating knowledge graph visualization...", expanded=False) as status:
                            graph_image, interactive_graph = create_combined_graph_visualization(providers_data)
                            st.session_state.graph_image = graph_image
                            st.session_state.interactive_graph = interactive_graph
                            status.update(label="Knowledge graph visualization completed!", state="complete")
                        
                        if st.session_state.interactive_graph is not None:
                            with st.expander("Knowledge Graph Visualization", expanded=False):
                                st.plotly_chart(st.session_state.interactive_graph, use_container_width=True)
                                st.caption("Interactive Knowledge Graph")
                        
                        # Create container for streaming report
                        st.subheader("Comprehensive Analysis Report")
                        report_container = st.empty()
                        accumulated_report = ""
                        
                        # Stream the report generation
                        with st.status("Generating analysis report...", expanded=False) as status:
                            try:
                                for chunk in graph_processor.generate_comparison_report_from_graphs_streaming(providers_data):
                                    accumulated_report += chunk
                                    # Update the display with accumulated content
                                    report_container.markdown(accumulated_report)
                                
                                # Store the completed report
                                st.session_state.comparison_report = accumulated_report
                                status.update(label="Analysis report completed!", state="complete")
                                
                            except Exception as stream_error:
                                st.error(f"Error during streaming generation: {str(stream_error)}")
                                # Fallback to non-streaming method
                                st.warning("Falling back to standard generation...")
                                with st.spinner("Generating analysis report..."):
                                    st.session_state.comparison_report = graph_processor.generate_comparison_report_from_graphs(providers_data)
                                    report_container.markdown(st.session_state.comparison_report)
                                status.update(label="Analysis completed (fallback mode)", state="complete")
                        
                        # Store analysis metadata
                        st.session_state.analysis_metadata = {
                            "providers_analyzed": selected_for_analysis,
                            "analysis_date": pd.Timestamp.now().isoformat(),
                            "total_providers": len(selected_for_analysis)
                        }
                        
                        st.success("Supplier analysis completed successfully!")
                        
                        # Auto-save project if one is loaded
                        if st.session_state.current_project:
                            try:
                                project_manager.save_project(st.session_state.current_project, st.session_state)
                            except Exception as e:
                                st.warning(f"Could not auto-save project: {str(e)}")
                        
                    except Exception as e:
                        st.error(f"Error generating analysis: {str(e)}")
                        st.session_state.comparison_report = ""
                        st.session_state.graph_image = None
                        st.session_state.interactive_graph = None
        
        # Add download option for the report if it exists
        if st.session_state.comparison_report:
            try:
                from pdf_generator import create_pdf_from_markdown
                
                # Generate PDF from markdown content
                pdf_buffer = create_pdf_from_markdown(
                    markdown_content=st.session_state.comparison_report,
                    project_name=st.session_state.current_project or "RFQ Analysis",
                    filename=f"provider_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
                )
                
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_buffer.getvalue(),
                    file_name=f"provider_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    help="Download the generated analysis report as a PDF file"
                )
            except ImportError as e:
                st.error(f"PDF generation not available: {str(e)}. Please install required dependencies.")
                # Fallback to markdown download
                st.download_button(
                    label="Download Report (Markdown)",
                    data=st.session_state.comparison_report,
                    file_name=f"provider_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    help="Download the generated analysis report as a Markdown file"
                )
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")
                # Fallback to markdown download
                st.download_button(
                    label="Download Report (Markdown)",
                    data=st.session_state.comparison_report,
                    file_name=f"provider_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    help="Download the generated analysis report as a Markdown file"
                )
        
        # Provider-specific insights
        if completed_providers and hasattr(st.session_state, 'analysis_metadata'):
            st.markdown("---")
            st.subheader("Supplier-Specific Insights")
            
            for provider in completed_providers:
                if provider['name'] in selected_for_analysis:
                    with st.expander(f"Insights: {provider['name']}", expanded=False):
                        metadata = provider['extracted_data'].get('_metadata', {})
                        
                        # Show key metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Files Processed", f"{metadata.get('valid_files', 0)}/{metadata.get('total_files', 0)}")
                            st.metric("Attributes Extracted", f"{metadata.get('found_fields', 0)}/{metadata.get('total_fields', 0)}")
                        with col2:
                            st.metric("Content Size", f"{metadata.get('aggregated_tokens', 0):,} tokens")
                            if metadata.get('file_errors'):
                                st.error(f"{len(metadata['file_errors'])} file errors")
                        
                        # Show unique features if dynamic extraction was enabled
                        extracted_data = provider['extracted_data']
                        if 'dynamically_fetched_features' in extracted_data:
                            st.write("**Dynamic Features Found:**")
                            dynamic_features = extracted_data['dynamically_fetched_features']
                            for key, value in dynamic_features.items():
                                if value and value != "Not Found":
                                    st.write(f"• **{key.replace('_', ' ').title()}:** {value[:200]}...")
                        
                        # Show custom attribute if any were added
                        if 'manually_requested_features' in extracted_data:
                            st.write("**Custom Attributes:**")
                            custom_features = extracted_data['manually_requested_features']
                            for key, value in custom_features.items():
                                if value and value != "Not Found":
                                    st.write(f"• **{key.replace('_', ' ').title()}:** {value[:200]}...")

# Tab 4: Provider Chat
with tab4:
    st.header("Supplier Document Chat")
    
    # Check if providers are processed
    completed_providers = [p for p in st.session_state.providers if p['status'] == 'completed']
    
    if not completed_providers:
        st.warning("Please process suppliers in the 'Process Documents' tab before using the chat feature.")
        
        # Show guidance
        if st.session_state.providers:
            pending_count = sum(1 for p in st.session_state.providers if p['status'] == 'pending')
            if pending_count > 0:
                st.info(f"You have {pending_count} suppliers ready to process.")
    else:
        # Show available providers
        provider_names = [p['name'] for p in completed_providers]
        st.info(f"Available suppliers: {', '.join(provider_names)} ({len(completed_providers)} total)")
        
        # Chat interface configuration
        st.subheader("Chat Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            # Provider selection
            selected_providers = st.multiselect(
                "Select suppliers to query:",
                options=provider_names,
                default=provider_names,  # Default to all providers
                help="Choose which suppliers to ask the question"
            )
        
        with col2:
            # Query mode selection
            query_mode = st.selectbox(
                "Query Mode:",
                options=[
                    "Individual Responses",
                    "Comparative Analysis", 
                    "Summary Across All"
                ],
                help="Choose how to present the responses"
            )
        
        # Main chat interface
        st.subheader("Ask Questions About Your Suppliers")
        
        # Pre-defined common questions
        st.write("**Common Questions:**")
        common_questions = [
            "What is the submission deadline?",
            "What are the key technical requirements?",
            "What is the estimated contract value?",
            "What are the evaluation criteria?",
            "What insurance requirements must be met?",
            "What are the payment terms?",
            "What are the mandatory supplier requirements?"
        ]
        
        question_cols = st.columns(3)
        for i, question in enumerate(common_questions):
            with question_cols[i % 3]:
                if st.button(f"{question}", key=f"common_q_{i}", help=question):
                    st.session_state.current_query = question
        
        # Custom query input
        query = st.text_area(
            "Or enter your custom question:", 
            value=st.session_state.get('current_query', ''),
            placeholder="e.g., Compare the environmental compliance requirements across suppliers...",
            height=100
        )
        
        # Clear query button
        if st.button("Clear Query", key="clear_query"):
            st.session_state.current_query = ""
            st.rerun()
        
        # Ask question button
        if st.button("Ask Question", type="primary", key="ask_question_main"):
            if not query:
                st.warning("Please enter a question.")
            elif not selected_providers:
                st.warning("Please select at least one suppliers.")
            else:
                try:
                    st.subheader("Responses")
                    
                    if query_mode == "Individual Responses":
                        # Show individual responses from each suppliers with streaming
                        for provider_name in selected_providers:
                            provider = next(p for p in completed_providers if p['name'] == provider_name)
                            
                            with st.expander(f"💼 Response from {provider_name}", expanded=True):
                                try:
                                    # Get the raw aggregated content for this suppliers
                                    extracted_data = provider['extracted_data']
                                    raw_content = extracted_data.get('_raw_content', '')
                                    
                                    if not raw_content:
                                        st.warning(f"No raw content available for {provider_name}. Using extracted features as fallback.")
                                        # Fallback to extracted features
                                        content_summary = f"Supplier: {provider_name}\n\n"
                                        for category, fields in extracted_data.items():
                                            if not category.startswith('_') and isinstance(fields, dict):
                                                content_summary += f"{category.replace('_', ' ').title()}:\n"
                                                for field, value in fields.items():
                                                    if value and value != "Not Found":
                                                        content_summary += f"- {field.replace('_', ' ').title()}: {value}\n"
                                                content_summary += "\n"
                                        raw_content = content_summary
                                    
                                    # Optimize content if too large
                                    from document_processor import optimize_text_for_context, check_context_limits
                                    temp_content = {
                                        'full_text': raw_content,
                                        'token_count': len(raw_content.split()),
                                        'filename': f"{provider_name}_aggregated_documents"
                                    }
                                    
                                    # Check and optimize if needed
                                    limits_check = check_context_limits(temp_content, max_tokens=1000000)
                                    if not limits_check["within_limits"]:
                                        st.info(f"Optimizing large content for {provider_name} ({temp_content['token_count']:,} tokens)")
                                        raw_content = optimize_text_for_context(raw_content, target_tokens=800000)
                                    
                                    # Create properly formatted data for LLM client
                                    llm_data = {
                                        'full_text': raw_content,
                                        'token_count': len(raw_content.split()),
                                        'filename': f"{provider_name}_aggregated_documents"
                                    }
                                    
                                    # Create container for streaming response
                                    response_container = st.empty()
                                    accumulated_response = ""
                                    
                                    # Show progress message
                                    progress_msg = st.info(f"Analyzing {provider_name}...")
                                    
                                    try:
                                        # Stream the response
                                        for chunk in comparator.client.answer_specific_query_streaming(llm_data, query):
                                            accumulated_response += chunk
                                            response_container.markdown(accumulated_response)
                                        
                                        progress_msg.success(f"Analysis of {provider_name} completed!")
                                        
                                    except Exception as stream_error:
                                        progress_msg.error(f"Streaming error: {str(stream_error)}")
                                        # Fallback to non-streaming
                                        st.warning("Falling back to standard response...")
                                        response = comparator.client.answer_specific_query(llm_data, query)
                                        response_container.markdown(response)
                                        st.success(f"Analysis of {provider_name} completed (fallback mode)")
                                    
                                except Exception as e:
                                    st.error(f"Error querying {provider_name}: {str(e)}")
                        
                    elif query_mode == "Comparative Analysis":
                        # Show comparative analysis across selected providers with streaming
                        st.write("**Comparative Analysis:**")
                        
                        # Collect raw content from all selected providers
                        all_raw_content = []
                        for provider_name in selected_providers:
                            provider = next(p for p in completed_providers if p['name'] == provider_name)
                            extracted_data = provider['extracted_data']
                            raw_content = extracted_data.get('_raw_content', '')
                            
                            if raw_content:
                                all_raw_content.append(f"=== {provider_name.upper()} ===\n{raw_content}")
                            else:
                                st.warning(f"No raw content available for {provider_name}. Using extracted features as fallback.")
                                # Fallback to extracted features
                                content_summary = f"Supplier: {provider_name}\n\n"
                                for category, fields in extracted_data.items():
                                    if not category.startswith('_') and isinstance(fields, dict):
                                        content_summary += f"{category.replace('_', ' ').title()}:\n"
                                        for field, value in fields.items():
                                            if value and value != "Not Found":
                                                content_summary += f"- {field.replace('_', ' ').title()}: {value}\n"
                                        content_summary += "\n"
                                all_raw_content.append(f"=== {provider_name.upper()} ===\n{content_summary}")
                        
                        # Combine all supplier content
                        combined_content = "\n\n".join(all_raw_content)
                        
                        # Generate comparative response
                        comparative_prompt = f"""
                        Based on the following complete document content from multiple suppliers, please answer this question with a comparative analysis: "{query}"
                        
                        Complete Document Content:
                        {combined_content}

                        Please provide a structured comparison highlighting differences and similarities between the suppliers.
                        """
                        
                        # Optimize content if too large
                        from document_processor import optimize_text_for_context, check_context_limits
                        temp_content = {
                            'full_text': comparative_prompt,
                            'token_count': len(comparative_prompt.split()),
                            'filename': 'comparative_analysis'
                        }
                        
                        # Check and optimize if needed
                        limits_check = check_context_limits(temp_content, max_tokens=1000000)
                        if not limits_check["within_limits"]:
                            st.info(f"Optimizing large comparative content ({temp_content['token_count']:,} tokens)")
                            comparative_prompt = optimize_text_for_context(comparative_prompt, target_tokens=800000)
                        
                        # Create properly formatted data for LLM client
                        llm_data = {
                            'full_text': comparative_prompt,
                            'token_count': len(comparative_prompt.split()),
                            'filename': 'comparative_analysis'
                        }
                        
                        # Create container for streaming response
                        comparison_container = st.empty()
                        accumulated_comparison = ""
                        
                        # Show progress message
                        progress_msg = st.info("Generating comparative analysis...")
                        
                        try:
                            for chunk in comparator.client.answer_specific_query_streaming(llm_data, query):
                                accumulated_comparison += chunk
                                comparison_container.markdown(accumulated_comparison)
                            
                            progress_msg.success("Comparative analysis completed!")
                            
                        except Exception as stream_error:
                            progress_msg.error(f"Streaming error: {str(stream_error)}")
                            # Fallback to non-streaming
                            st.warning("Falling back to standard response...")
                            comparative_response = comparator.client.answer_specific_query(llm_data, query)
                            comparison_container.markdown(comparative_response)
                            st.success("Comparative analysis completed (fallback mode)")
                    
                    elif query_mode == "Summary Across All":
                        # Show summary across all selected providers with streaming
                        st.write("**Summary Across All Suppliers:**")
                        
                        # Collect raw content from all selected suppliers
                        all_raw_content = []
                        for provider_name in selected_providers:
                            provider = next(p for p in completed_providers if p['name'] == provider_name)
                            extracted_data = provider['extracted_data']
                            raw_content = extracted_data.get('_raw_content', '')
                            
                            if raw_content:
                                all_raw_content.append(f"=== {provider_name.upper()} ===\n{raw_content}")
                            else:
                                st.warning(f"No raw content available for {provider_name}. Using extracted features as fallback.")
                                # Fallback to extracted features
                                content_summary = f"Supplier: {provider_name}\n\n"
                                for category, fields in extracted_data.items():
                                    if not category.startswith('_') and isinstance(fields, dict):
                                        content_summary += f"{category.replace('_', ' ').title()}:\n"
                                        for field, value in fields.items():
                                            if value and value != "Not Found":
                                                content_summary += f"- {field.replace('_', ' ').title()}: {value}\n"
                                        content_summary += "\n"
                                all_raw_content.append(f"=== {provider_name.upper()} ===\n{content_summary}")
                        
                        # Combine all supplier content
                        combined_content = "\n\n".join(all_raw_content)
                        
                        # Generate summary response with comprehensive data
                        comprehensive_content = f"""
                        Based on complete document content from {len(selected_providers)} suppliers, please provide a comprehensive summary answering: "{query}"
                        
                        Suppliers: {', '.join(selected_providers)}
                        
                        Complete Document Content:
                        {combined_content}

                        Please synthesize the information and provide key insights, trends, and recommendations.
                        """
                        
                        # Optimize content if too large
                        from document_processor import optimize_text_for_context, check_context_limits
                        temp_content = {
                            'full_text': comprehensive_content,
                            'token_count': len(comprehensive_content.split()),
                            'filename': 'summary_analysis'
                        }
                        
                        # Check and optimize if needed
                        limits_check = check_context_limits(temp_content, max_tokens=1000000)
                        if not limits_check["within_limits"]:
                            st.info(f"Optimizing large summary content ({temp_content['token_count']:,} tokens)")
                            comprehensive_content = optimize_text_for_context(comprehensive_content, target_tokens=800000)
                        
                        # Create properly formatted data for LLM client
                        llm_data = {
                            'full_text': comprehensive_content,
                            'token_count': len(comprehensive_content.split()),
                            'filename': 'summary_analysis'
                        }
                        
                        # Create container for streaming response
                        summary_container = st.empty()
                        accumulated_summary = ""
                        
                        # Show progress message
                        progress_msg = st.info("Generating summary analysis...")
                        
                        try:
                            for chunk in comparator.client.answer_specific_query_streaming(llm_data, query):
                                accumulated_summary += chunk
                                summary_container.markdown(accumulated_summary)
                            
                            progress_msg.success("Summary analysis completed!")
                            
                        except Exception as stream_error:
                            progress_msg.error(f"Streaming error: {str(stream_error)}")
                            # Fallback to non-streaming
                            st.warning("Falling back to standard response...")
                            summary_response = comparator.client.answer_specific_query(llm_data, query)
                            summary_container.markdown(summary_response)
                            st.success("Summary analysis completed (fallback mode)")
                        
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "query": query,
                        "providers": selected_providers,
                        "mode": query_mode,
                        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("---")
            st.subheader("Recent Chat History")
            
            # Show last 5 queries
            recent_chats = list(reversed(st.session_state.chat_history[-5:]))
            
            for i, chat in enumerate(recent_chats):
                with st.expander(f"[{chat['timestamp']}] {chat['query'][:50]}...", expanded=False):
                    st.write(f"**Question:** {chat['query']}")
                    st.write(f"**Suppliers:** {', '.join(chat.get('providers', []))}")
                    st.write(f"**Mode:** {chat.get('mode', 'Individual Responses')}")
            
            # Clear history button
            if st.button("Clear Chat History", key="clear_chat_history"):
                st.session_state.chat_history = []
                st.rerun()


# Footer
st.markdown("---")
st.markdown("**SAP RFQx Document Analysis Application** - Analyze and compare RFQ documents with AI")
