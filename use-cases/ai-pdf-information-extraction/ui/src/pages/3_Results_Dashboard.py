"""
Results Dashboard Page - View and export all extraction results.
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from pathlib import Path

# Import utilities
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils import (
    load_css_files,
    initialize_session_state,
    export_results_to_json,
    export_results_to_csv,
    combine_results_for_export,
    get_document_type_display_name
)

# Page Configuration
st.set_page_config(
    page_title="Results Dashboard",
    page_icon="static/images/SAP_logo_square.png",
    layout="wide"
)
st.logo("static/images/SAP_logo.svg")

# Load CSS
css_files = [
    os.path.join(Path(__file__).parent.parent.parent, "static", "styles", "variables.css"),
    os.path.join(Path(__file__).parent.parent.parent, "static", "styles", "style.css"),
]
load_css_files(css_files)

# Initialize session state
initialize_session_state()

# Title
st.title("Results Dashboard")
st.markdown("View and export all processed extraction results")

# Initialize custom document types if not present
if "custom_document_types" not in st.session_state:
    st.session_state.custom_document_types = {}

# Check if there are any results
has_results = bool(st.session_state.extraction_results)

if has_results:
    # Summary section
    st.subheader("Processed Documents Summary")
    
    # Calculate statistics
    total_documents = 0
    total_fields = 0
    doc_counts = {}
    
    for doc_id, df in st.session_state.extraction_results.items():
        # Get document name from custom types or use doc_id
        if doc_id in st.session_state.custom_document_types:
            doc_name = st.session_state.custom_document_types[doc_id]['name']
        else:
            doc_name = doc_id
        # Count unique files if "File" column exists
        if "File" in df.columns:
            unique_files = df["File"].nunique()
        else:
            unique_files = 1
        doc_counts[doc_name] = unique_files
        total_documents += unique_files
        total_fields += len(df)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", total_documents)
    
    with col2:
        st.metric("Total Fields Extracted", total_fields)
    
    with col3:
        st.metric("Document Types", len(doc_counts))
    
    with col4:
        # Most processed document type
        if doc_counts:
            most_processed = max(doc_counts, key=doc_counts.get)
            st.metric("Most Processed", most_processed)
    
    # Document breakdown
    if doc_counts:
        st.markdown("### Breakdown by Document Type")
        breakdown_df = pd.DataFrame([
            {"Document Type": doc_type, "Count": count}
            for doc_type, count in doc_counts.items()
        ])
        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Detailed results section
    st.subheader("Detailed Results")
    
    # Create tabs for each document type
    tab_names = []
    tab_data = []
    
    for doc_id, df in st.session_state.extraction_results.items():
        if doc_id in st.session_state.custom_document_types:
            display_name = st.session_state.custom_document_types[doc_id]['name']
        else:
            display_name = doc_id
        tab_names.append(f"{display_name} ({len(df)} fields)")
        tab_data.append((doc_id, df))
    
    if tab_names:
        tabs = st.tabs(tab_names)
        
        for idx, (tab, (doc_id, df)) in enumerate(zip(tabs, tab_data)):
            with tab:
                if doc_id in st.session_state.custom_document_types:
                    doc_name = st.session_state.custom_document_types[doc_id]['name']
                else:
                    doc_name = doc_id
                st.markdown(f"### {doc_name}")
                
                # Display options
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Search/filter
                    search_term = st.text_input(
                        "Search in results",
                        key=f"search_{doc_id}",
                        placeholder="Type to filter..."
                    )
                
                with col2:
                    # Display mode
                    display_mode = st.selectbox(
                        "View",
                        ["Table", "JSON"],
                        key=f"display_{doc_id}"
                    )
                
                # Apply search filter if provided
                display_df = df.copy()
                if search_term:
                    mask = display_df.apply(
                        lambda row: search_term.lower() in str(row).lower(), 
                        axis=1
                    )
                    display_df = display_df[mask]
                
                # Display data
                if display_mode == "Table":
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                else:
                    # Convert to JSON for display
                    json_data = display_df.to_dict('records')
                    st.json(json_data)
                
                # Statistics for this document type
                with st.expander("Statistics"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Fields", len(df))
                    
                    with col2:
                        if "File" in df.columns:
                            st.metric("Unique Files", df["File"].nunique())
                    
                    with col3:
                        # Count non-empty values
                        if "Value" in df.columns:
                            non_empty = df["Value"].notna().sum()
                            completion_rate = (non_empty / len(df) * 100) if len(df) > 0 else 0
                            st.metric("Completion Rate", f"{completion_rate:.1f}%")
    
    st.markdown("---")
    
    # Export section
    st.markdown("### Export All as CSV")
    st.markdown("Download all results in a unified CSV file")
    
    # Combine all data
    combined_df = combine_results_for_export(st.session_state.extraction_results)
    
    if not combined_df.empty:
        csv_str = export_results_to_csv(combined_df)
        
        st.download_button(
            label="Download CSV",
            data=csv_str,
            file_name=f"document_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="download_all_csv",
            use_container_width=True
        )
    else:
        st.info("No data to export to CSV")
    
    st.markdown("---")
    
    # # Data management section
    # st.subheader("Gestión de Datos")
    # 
    # col1, col2, col3 = st.columns(3)
    # 
    # with col1:
    #     if st.button("🗑️ Limpiar Todos los Resultados", type="secondary", use_container_width=True):
    #         st.session_state.extraction_results = {}
    #         st.session_state.batch_extraction_results = []
    #         st.success("Todos los resultados han sido eliminados")
    #         st.rerun()
    # 
    # with col2:
    #     # Clear specific type
    #     doc_type_to_clear = st.selectbox(
    #         "Limpiar tipo específico",
    #         ["Seleccionar..."] + list(st.session_state.extraction_results.keys()),
    #         key="clear_specific_type"
    #     )
    #     
    #     if doc_type_to_clear != "Seleccionar...":
    #         if st.button(f"Limpiar {get_document_type_display_name(doc_type_to_clear)}", key="clear_specific_btn"):
    #             del st.session_state.extraction_results[doc_type_to_clear]
    #             st.success(f"Resultados de {get_document_type_display_name(doc_type_to_clear)} eliminados")
    #             st.rerun()
    # 
    # with col3:
    #     st.metric(
    #         "Espacio en Memoria",
    #         f"{sum(df.memory_usage(deep=True).sum() for df in st.session_state.extraction_results.values()) / 1024:.1f} KB"
    #     )

else:
    # No results available
    st.info("No results to display")
    
    st.markdown("""
    ### How to get results:
    
    1. **Define Document Types**: Go to "Extraction Config" to create custom document types
    
    2. **Process Documents**: Go to "Document Processing" to process single or multiple documents
    
    3. Results will automatically appear here after processing
    
    ### Dashboard Features:
    
    - **Consolidated View**: All results in one place
    - **Flexible Export**: CSV format with selection options
    - **Search and Filter**: Find information quickly
    - **Statistics**: Processing and completion metrics
    - **Data Management**: Clear results selectively
    """)
    
    # Quick actions
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Go to Extraction Config", use_container_width=True):
            st.switch_page("pages/1_Extraction_Config.py")
    
    with col2:
        if st.button("Go to Document Processing", use_container_width=True):
            st.switch_page("pages/2_Document_Processing.py")

# Sidebar with information
with st.sidebar:
    st.header("Dashboard Information")
    
    st.markdown("""
    ### Features:
    
    - **Summary**: General statistics
    - **Detailed Results**: By document type
    - **Search**: Real-time filtering
    - **Export**: CSV format
    - **Management**: Selective clearing
    
    ### Export Format:
         
    - **CSV**: Flat table for analysis
    
    ### Tips:
    
    - Use search to find specific fields
    - Export regularly for backup
    - Clear old results to free memory
    """)
    
    if has_results:
        st.markdown("---")
        st.markdown("### Session Statistics")
        st.metric("Document Types", len(st.session_state.extraction_results))
        
        total_records = sum(len(df) for df in st.session_state.extraction_results.values())
        st.metric("Total Records", total_records)