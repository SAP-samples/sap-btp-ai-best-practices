"""
Document Processing Page - Process multiple PDF documents using custom document types.
"""

import streamlit as st
import pandas as pd
import time
import os
from typing import List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path

# Import utilities and API client
import sys
sys.path.append(str(Path(__file__).parent.parent))

from api_client import (
    extract_single_document
)
from utils import (
    load_css_files,
    initialize_session_state,
    format_file_size,
    parse_api_error,
    create_results_dataframe
)

# Page Configuration
st.set_page_config(
    page_title="Document Processing",
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

# Initialize custom document types if not present
if "custom_document_types" not in st.session_state:
    st.session_state.custom_document_types = {}

# Initialize batch files in session state
if "batch_files" not in st.session_state:
    st.session_state.batch_files = {}

# Define helper functions
def process_batch_files(max_concurrent: int, total_files: int):
    """
    Process all batch files.
    
    Args:
        max_concurrent: Maximum concurrent processing
        total_files: Total number of files to process
    """
    with st.spinner(f"Preparing {total_files} documents for processing..."):
        # Prepare files for API
        all_files = []
        document_types = []
        file_to_doc_mapping = {}  # Map file names to document types
        
        for doc_id, files in st.session_state.batch_files.items():
            if files and doc_id in st.session_state.custom_document_types:
                doc_data = st.session_state.custom_document_types[doc_id]
                
                for file in files:
                    # Read file content
                    file.seek(0)  # Reset file pointer
                    content = file.read()
                    all_files.append((file.name, content, "application/pdf"))
                    document_types.append("custom")  # Always use custom type
                    file_to_doc_mapping[file.name] = {
                        "doc_id": doc_id,
                        "doc_name": doc_data['name'],
                        "questions": doc_data['questions'],
                        "fields": doc_data['fields']
                    }
        
        if not all_files:
            st.error("No files to process")
            return
        
        # Process files one by one (since we need custom questions for each)
        progress_bar = st.progress(0, text="Starting processing...")
        status_container = st.container()
        results = []
        
        for idx, (file_info, doc_type) in enumerate(zip(all_files, document_types)):
            file_name, file_content, content_type = file_info
            doc_config = file_to_doc_mapping[file_name]
            
            # Update progress
            progress = (idx + 1) / len(all_files)
            progress_bar.progress(progress, text=f"Processing {file_name}...")
            
            # Process individual file
            response = extract_single_document(
                file_content=file_content,
                filename=file_name,
                document_type="custom",
                questions=doc_config['questions'],
                fields=doc_config['fields'],
                temperature=0.1,
                language="en"
            )
            
            # Store result
            result = {
                "filename": file_name,
                "doc_id": doc_config['doc_id'],
                "doc_name": doc_config['doc_name'],
                "success": response.get("success", False),
                "response": response
            }
            results.append(result)
            
            # Update status
            with status_container:
                if result["success"]:
                    st.success(f"✅ {file_name} - {doc_config['doc_name']}")
                else:
                    st.error(f"❌ {file_name} - {doc_config['doc_name']}: {parse_api_error(response)}")
        
        # Complete processing
        progress_bar.progress(1.0, text="Processing complete!")
        
        # Display results summary
        display_batch_results(results)

def display_batch_results(results: List[Dict[str, Any]]):
    """
    Display batch processing results.
    
    Args:
        results: List of processing results
    """
    st.subheader("Batch Processing Results")
    
    # Statistics
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Successful", successful)
    with col2:
        st.metric("Failed", failed)
    with col3:
        success_rate = (successful / len(results) * 100) if results else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Results table
    results_display = []
    for result in results:
        results_display.append({
            "File": result["filename"],
            "Document Type": result["doc_name"],
            "Status": "✅ Completed" if result["success"] else "❌ Error",
            "Error": "" if result["success"] else parse_api_error(result["response"])
        })
    
    df_results = pd.DataFrame(results_display)
    st.dataframe(df_results, use_container_width=True, hide_index=True)
    
    # Save successful results to session state
    saved_count = 0
    for result in results:
        if result["success"] and result["response"].get("results"):
            doc_id = result["doc_id"]
            df = create_results_dataframe(result["response"])
            
            if not df.empty:
                # Add metadata
                df["File"] = result["filename"]
                df["Document Type"] = result["doc_name"]
                df["Processed At"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                if "extraction_results" not in st.session_state:
                    st.session_state.extraction_results = {}
                
                if doc_id in st.session_state.extraction_results:
                    st.session_state.extraction_results[doc_id] = pd.concat([
                        st.session_state.extraction_results[doc_id],
                        df
                    ], ignore_index=True)
                else:
                    st.session_state.extraction_results[doc_id] = df
                
                saved_count += 1
    
    if saved_count > 0:
        st.success(f"Results from {saved_count} successful extractions saved to session. View them in Results Dashboard.")

# Title
st.title("Document Processing")
st.markdown("Process single or multiple documents using your custom document types")

# Check if there are any document types defined
if not st.session_state.custom_document_types:
    st.warning("No document types defined yet!")
    st.info("Please go to 'Extraction Config' page to create custom document types first.")
    
    if st.button("Go to Extraction Config", type="primary"):
        st.switch_page("pages/1_Extraction_Config.py")
else:
    # Tab interface for single vs batch processing
    tab1, tab2 = st.tabs(["Single Document Processing", "Batch Processing"])
    
    # Tab 1: Single Document Processing
    with tab1:
        st.header("Single Document Processing")
        st.markdown("Process one document at a time with immediate results")
        
        # Select document type
        doc_type_options = {
            doc_id: doc_data['name'] 
            for doc_id, doc_data in st.session_state.custom_document_types.items()
        }
        
        selected_doc_id = st.selectbox(
            "Select Document Type",
            options=list(doc_type_options.keys()),
            format_func=lambda x: doc_type_options[x],
            help="Choose which document type to use for extraction",
            key="single_doc_type"
        )
        
        if selected_doc_id:
            doc_data = st.session_state.custom_document_types[selected_doc_id]
            
            # Display document type info
            with st.expander("Document Type Information", expanded=False):
                st.markdown(f"**Type:** {doc_data['name']}")
                st.markdown(f"**Questions:** {len(doc_data['questions'])}")
                if doc_data['questions']:
                    for i, (q, f) in enumerate(zip(doc_data['questions'], doc_data['fields']), 1):
                        st.markdown(f"{i}. {q} → *{f}*")
            
            # File upload
            uploaded_file = st.file_uploader(
                f"Upload {doc_data['name']} Document",
                type=['pdf'],
                help="Select a PDF file to process",
                key="single_upload"
            )
            
            if uploaded_file:
                st.info(f"File loaded: {uploaded_file.name} ({format_file_size(uploaded_file.size)})")
                
                if st.button("Process Document", type="primary", use_container_width=True, key="process_single"):
                    if not doc_data['questions']:
                        st.error("This document type has no questions defined. Please configure it first.")
                    else:
                        # Process the document
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            # Read file content
                            file_content = uploaded_file.read()
                            
                            # Call API
                            response = extract_single_document(
                                file_content=file_content,
                                filename=uploaded_file.name,
                                document_type="custom",
                                questions=doc_data['questions'],
                                fields=doc_data['fields'],
                                temperature=0.1,
                                language="en"
                            )
                            
                            # Handle response
                            if response.get("success"):
                                st.success("Document processed successfully!")
                                
                                # Create and display results
                                df = create_results_dataframe(response)
                                
                                if not df.empty:
                                    st.subheader("Extraction Results")
                                    st.dataframe(df, use_container_width=True, hide_index=True)
                                    
                                    # Save to session state
                                    if "extraction_results" not in st.session_state:
                                        st.session_state.extraction_results = {}
                                    
                                    # Add filename to dataframe
                                    df["File"] = uploaded_file.name
                                    df["Document Type"] = doc_data['name']
                                    
                                    if selected_doc_id in st.session_state.extraction_results:
                                        st.session_state.extraction_results[selected_doc_id] = pd.concat([
                                            st.session_state.extraction_results[selected_doc_id],
                                            df
                                        ], ignore_index=True)
                                    else:
                                        st.session_state.extraction_results[selected_doc_id] = df
                                    
                                    st.success("Results saved to session!")
                                else:
                                    st.warning("No results could be extracted from the document")
                            else:
                                error_msg = parse_api_error(response)
                                st.error(f"Processing failed: {error_msg}")
    
    # Tab 2: Batch Processing
    with tab2:
        st.header("Batch Processing")
        st.markdown("Process multiple documents simultaneously")
        
        # Initialize batch files for each document type
        for doc_id in st.session_state.custom_document_types:
            if doc_id not in st.session_state.batch_files:
                st.session_state.batch_files[doc_id] = []
        
        # File upload sections for each document type
        st.subheader("Upload Files by Document Type")
        
        # Create columns based on number of document types
        num_types = len(st.session_state.custom_document_types)
        if num_types > 0:
            # Create two columns for better layout
            col1, col2 = st.columns(2)
            
            for idx, (doc_id, doc_data) in enumerate(st.session_state.custom_document_types.items()):
                # Alternate between columns
                with col1 if idx % 2 == 0 else col2:
                    st.markdown(f"**{doc_data['name']}**")
                    
                    files = st.file_uploader(
                        f"Select PDF files",
                        type=['pdf'],
                        accept_multiple_files=True,
                        key=f"batch_{doc_id}",
                        help=f"Upload multiple {doc_data['name']} documents"
                    )
                    
                    if files:
                        st.caption(f"{len(files)} file(s) selected")
                        st.session_state.batch_files[doc_id] = files
                    else:
                        st.session_state.batch_files[doc_id] = []
        
        # Calculate total files
        total_files = sum(len(files) for files in st.session_state.batch_files.values())
        
        if total_files > 0:
            st.markdown("---")
            st.subheader("Files Summary")
            
            # Create summary table
            files_summary = []
            for doc_id, files in st.session_state.batch_files.items():
                if files:
                    doc_name = st.session_state.custom_document_types[doc_id]['name']
                    for file in files:
                        files_summary.append({
                            "File Name": file.name,
                            "Document Type": doc_name,
                            "Size": format_file_size(file.size)
                        })
            
            if files_summary:
                df_files = pd.DataFrame(files_summary)
                st.dataframe(df_files, use_container_width=True, hide_index=True)
            
            # Processing configuration
            st.markdown("---")
            st.subheader("Processing Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                max_concurrent = st.slider(
                    "Maximum Concurrent Processing",
                    min_value=1,
                    max_value=20,
                    value=10,
                    help="Number of documents to process simultaneously"
                )
            
            with col2:
                st.metric("Total Files", total_files)
            
            # Process button
            st.markdown("---")
            if st.button("Process All Documents", type="primary", use_container_width=True):
                process_batch_files(max_concurrent, total_files)
        else:
            st.info("Upload files to the document type sections above to begin batch processing")

# Sidebar with information
with st.sidebar:
    st.header("Information")
    
    st.markdown("""
    ### Processing Options:
    
    **Single Document:**
    - Process one document at a time
    - Immediate results display
    - Good for testing and validation
    
    **Batch Processing:**
    - Process multiple documents
    - Organized by document type
    - Concurrent processing for speed
    
    ### Tips:
    
    - Define document types first in Extraction Config
    - Test with single documents before batch processing
    - Adjust concurrent processing based on your system
    - Check Results Dashboard for all extracted data
    """)
    
    st.markdown("---")
    
    # Statistics
    if st.session_state.custom_document_types:
        st.markdown("### Document Types")
        for doc_id, doc_data in st.session_state.custom_document_types.items():
            st.markdown(f"- **{doc_data['name']}**: {len(doc_data['questions'])} questions")
    
    if "extraction_results" in st.session_state and st.session_state.extraction_results:
        st.markdown("---")
        st.markdown("### Session Results")
        total_extractions = sum(
            len(df) for df in st.session_state.extraction_results.values()
        )
        st.metric("Total Extractions", total_extractions)
        
        if st.button("Clear Session Results", type="secondary"):
            st.session_state.extraction_results = {}
            st.success("Session results cleared")
            st.rerun()