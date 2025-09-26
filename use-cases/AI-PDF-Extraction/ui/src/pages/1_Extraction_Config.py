"""
Extraction Config Page - Define and manage custom document types.
"""

import streamlit as st
import pandas as pd
import json
import os
import uuid
import time
from typing import List, Dict, Any
from pathlib import Path

# Import utilities and API client
import sys
sys.path.append(str(Path(__file__).parent.parent))

from api_client import extract_single_document
from utils import (
    load_css_files,
    initialize_session_state,
    validate_pdf_file,
    create_results_dataframe,
    parse_api_error
)

# Page Configuration
st.set_page_config(
    page_title="Extraction Config",
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

# Initialize custom document types in session state if not present
if "custom_document_types" not in st.session_state:
    st.session_state.custom_document_types = {}


# Title
st.title("Extraction Configuration")
st.markdown("Define custom document types and configure extraction questions")

# Helper functions
def generate_doc_type_id():
    """Generate a unique ID for a document type."""
    return f"doc_type_{uuid.uuid4().hex[:8]}"

def add_document_type(name: str):
    """Add a new document type to session state."""
    doc_id = generate_doc_type_id()
    st.session_state.custom_document_types[doc_id] = {
        "name": name,
        "questions": [],
        "fields": []
    }
    return doc_id

def delete_document_type(doc_id: str):
    """Delete a document type from session state."""
    if doc_id in st.session_state.custom_document_types:
        del st.session_state.custom_document_types[doc_id]
        # Also remove any results for this type
        if "extraction_results" in st.session_state and doc_id in st.session_state.extraction_results:
            del st.session_state.extraction_results[doc_id]

def update_document_type(doc_id: str, name: str, questions: List[str], fields: List[str]):
    """Update a document type in session state."""
    if doc_id in st.session_state.custom_document_types:
        st.session_state.custom_document_types[doc_id] = {
            "name": name,
            "questions": questions,
            "fields": fields
        }

# Main interface
tab1, tab2, tab3 = st.tabs(["Create New Type", "Manage Existing Types", "Test Extraction"])

# Initialize session state for form
if "create_num_questions" not in st.session_state:
    st.session_state.create_num_questions = 3
if "create_doc_name" not in st.session_state:
    st.session_state.create_doc_name = ""
if "create_questions" not in st.session_state:
    st.session_state.create_questions = {}
if "create_fields" not in st.session_state:
    st.session_state.create_fields = {}

# Tab 1: Create New Document Type
with tab1:
    st.header("Create New Document Type")
    
    st.markdown("### Document Type Details")
    
    # Document name input
    doc_name = st.text_input(
        "Document Type Name",
        value=st.session_state.create_doc_name,
        placeholder="e.g., Invoice, Contract, Purchase Order",
        help="Give your document type a descriptive name",
        key="doc_name_input"
    )
    st.session_state.create_doc_name = doc_name
    
    st.markdown("### Extraction Questions")
    st.info("Add questions that will be used to extract information from this document type.")
    
    # Number of questions selector (dynamic)
    num_questions = st.number_input(
        "Number of questions",
        min_value=1,
        max_value=20,
        value=st.session_state.create_num_questions,
        help="How many pieces of information do you want to extract?",
        key="num_questions_input"
    )
    
    # Update session state if number changed
    if num_questions != st.session_state.create_num_questions:
        st.session_state.create_num_questions = num_questions
        st.rerun()
    
    # Collect questions and fields
    questions = []
    fields = []
    all_filled = True
    missing_fields = []
    
    for i in range(num_questions):
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Initialize session state for this question if not exists
            if f"q_{i}" not in st.session_state.create_questions:
                st.session_state.create_questions[f"q_{i}"] = ""
            
            question = st.text_input(
                f"Question {i+1}",
                value=st.session_state.create_questions[f"q_{i}"],
                placeholder="e.g., What is the invoice number?",
                key=f"create_new_q_{i}"
            )
            st.session_state.create_questions[f"q_{i}"] = question
            questions.append(question)
            
            if not question:
                all_filled = False
                missing_fields.append(f"Question {i+1}")
        
        with col2:
            # Initialize session state for this field if not exists
            if f"f_{i}" not in st.session_state.create_fields:
                st.session_state.create_fields[f"f_{i}"] = ""
            
            field = st.text_input(
                f"Field Name {i+1}",
                value=st.session_state.create_fields[f"f_{i}"],
                placeholder="e.g., Invoice Number",
                key=f"create_new_f_{i}"
            )
            st.session_state.create_fields[f"f_{i}"] = field
            fields.append(field)
            
            if not field:
                all_filled = False
                missing_fields.append(f"Field Name {i+1}")
    
    # Check if document name is filled
    if not doc_name:
        all_filled = False
        missing_fields.insert(0, "Document Type Name")
    
    # Create button
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        if st.button("Create Document Type", type="primary", use_container_width=True):
            if all_filled:
                # Create the document type
                doc_id = add_document_type(doc_name)
                update_document_type(doc_id, doc_name, questions, fields)
                
                # Clear session state for next creation
                st.session_state.create_doc_name = ""
                st.session_state.create_questions = {}
                st.session_state.create_fields = {}
                st.session_state.create_num_questions = 3
                
                st.success(f"✅ Document type '{doc_name}' created successfully!")
                # st.balloons()
                time.sleep(1)
                st.rerun()
            else:
                st.error(f"Please fill in all fields. Missing: {', '.join(missing_fields)}")
    
    with col2:
        if st.button("Clear Form", type="secondary", use_container_width=True):
            # Clear all form data
            st.session_state.create_doc_name = ""
            st.session_state.create_questions = {}
            st.session_state.create_fields = {}
            st.session_state.create_num_questions = 3
            st.rerun()
    
    with col3:
        # Show a preview button
        if st.button("Preview", use_container_width=True):
            if doc_name and any(questions):
                with st.expander("Document Type Preview", expanded=True):
                    st.markdown(f"**Name:** {doc_name}")
                    st.markdown("**Questions and Fields:**")
                    for i, (q, f) in enumerate(zip(questions, fields), 1):
                        if q and f:
                            st.markdown(f"{i}. {q} → *{f}*")
            else:
                st.warning("Add a name and at least one question to preview")

# Tab 2: Manage Existing Document Types
with tab2:
    st.header("Manage Existing Document Types")
    
    if st.session_state.custom_document_types:
        # Display each document type
        for doc_id, doc_data in st.session_state.custom_document_types.items():
            with st.expander(f"📄 {doc_data['name']}", expanded=False):
                # Edit mode toggle
                edit_key = f"edit_{doc_id}"
                if edit_key not in st.session_state:
                    st.session_state[edit_key] = False
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**Document Type ID:** `{doc_id}`")
                
                with col2:
                    if st.button("Edit", key=f"edit_btn_{doc_id}"):
                        st.session_state[edit_key] = not st.session_state[edit_key]
                
                with col3:
                    if st.button("Delete", key=f"del_btn_{doc_id}", type="secondary"):
                        delete_document_type(doc_id)
                        st.success(f"Document type '{doc_data['name']}' deleted")
                        st.rerun()
                
                if st.session_state[edit_key]:
                    # Edit mode
                    st.markdown("### Edit Document Type")
                    
                    # Edit name
                    new_name = st.text_input(
                        "Document Type Name",
                        value=doc_data['name'],
                        key=f"edit_name_{doc_id}"
                    )
                    
                    # Edit questions and fields
                    st.markdown("### Questions and Fields")
                    
                    temp_questions = []
                    temp_fields = []
                    
                    for idx, (question, field) in enumerate(zip(doc_data['questions'], doc_data['fields'])):
                        col1, col2, col3 = st.columns([3, 2, 1])
                        
                        with col1:
                            q = st.text_input(
                                f"Question {idx+1}",
                                value=question,
                                key=f"edit_q_{doc_id}_{idx}"
                            )
                            temp_questions.append(q)
                        
                        with col2:
                            f = st.text_input(
                                f"Field {idx+1}",
                                value=field,
                                key=f"edit_f_{doc_id}_{idx}"
                            )
                            temp_fields.append(f)
                    
                    # Add new question
                    st.markdown("#### Add New Question")
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        new_question = st.text_input(
                            "New Question",
                            placeholder="Enter a new question",
                            key=f"add_q_{doc_id}"
                        )
                    
                    with col2:
                        new_field = st.text_input(
                            "Field Name",
                            placeholder="Enter field name",
                            key=f"add_f_{doc_id}"
                        )
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("Add Question", key=f"add_btn_{doc_id}"):
                            if new_question and new_field:
                                temp_questions.append(new_question)
                                temp_fields.append(new_field)
                                update_document_type(doc_id, new_name, temp_questions, temp_fields)
                                st.success("Question added")
                                st.rerun()
                    
                    with col2:
                        if st.button("Save Changes", key=f"save_{doc_id}", type="primary"):
                            update_document_type(doc_id, new_name, temp_questions, temp_fields)
                            st.session_state[edit_key] = False
                            st.success("Changes saved")
                            st.rerun()
                    
                    with col3:
                        if st.button("Cancel", key=f"cancel_{doc_id}"):
                            st.session_state[edit_key] = False
                            st.rerun()
                
                else:
                    # View mode - display questions and fields
                    st.markdown("### Questions and Fields")
                    
                    if doc_data['questions']:
                        df = pd.DataFrame({
                            "Question": doc_data['questions'],
                            "Field Name": doc_data['fields']
                        })
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.warning("No questions defined for this document type")
    else:
        st.info("No document types created yet. Go to 'Create New Type' tab to get started.")

# Tab 3: Test Extraction
with tab3:
    st.header("Test Document Extraction")
    st.markdown("Test your custom document types by extracting information from a sample PDF")
    
    if st.session_state.custom_document_types:
        # Select document type
        doc_type_options = {
            doc_id: doc_data['name'] 
            for doc_id, doc_data in st.session_state.custom_document_types.items()
        }
        
        selected_doc_id = st.selectbox(
            "Select Document Type",
            options=list(doc_type_options.keys()),
            format_func=lambda x: doc_type_options[x],
            help="Choose which document type to use for extraction"
        )
        
        if selected_doc_id:
            doc_data = st.session_state.custom_document_types[selected_doc_id]
            
            # Display selected type info
            with st.expander("Document Type Details", expanded=False):
                st.markdown(f"**Type:** {doc_data['name']}")
                st.markdown(f"**Number of Questions:** {len(doc_data['questions'])}")
                
                if doc_data['questions']:
                    st.markdown("**Questions:**")
                    for i, (q, f) in enumerate(zip(doc_data['questions'], doc_data['fields']), 1):
                        st.markdown(f"{i}. {q} → *{f}*")
            
            # File upload
            uploaded_file = st.file_uploader(
                "Upload PDF Document",
                type=['pdf'],
                help="Select a PDF file to test extraction"
            )
            
            if uploaded_file:
                st.info(f"File loaded: {uploaded_file.name}")
                
                if st.button("Extract Information", type="primary", use_container_width=True):
                    if not doc_data['questions']:
                        st.error("This document type has no questions defined. Please add questions first.")
                    else:
                        # Process the document
                        with st.spinner(f"Extracting information from {uploaded_file.name}..."):
                            # Read file content
                            file_content = uploaded_file.read()
                            
                            # Call API with custom document type
                            response = extract_single_document(
                                file_content=file_content,
                                filename=uploaded_file.name,
                                document_type="custom",  # Use custom type
                                questions=doc_data['questions'],
                                fields=doc_data['fields'],
                                temperature=0.1,
                                language="en"  # Change to English
                            )
                            
                            # Handle response
                            if response.get("success"):
                                st.success("Extraction completed successfully!")
                                
                                # Create and display results DataFrame
                                df = create_results_dataframe(response)
                                
                                if not df.empty:
                                    st.subheader("Extraction Results")
                                    st.dataframe(df, use_container_width=True, hide_index=True)
                                    
                                    # Option to save results
                                    if st.button("Save to Results", key="save_test_results"):
                                        if "extraction_results" not in st.session_state:
                                            st.session_state.extraction_results = {}
                                        st.session_state.extraction_results[selected_doc_id] = df
                                        st.success("Results saved to session!")
                                else:
                                    st.warning("No results could be extracted from the document")
                            else:
                                error_msg = parse_api_error(response)
                                st.error(f"Extraction failed: {error_msg}")
    else:
        st.info("No document types created yet. Please create a document type first.")

# Sidebar with information
with st.sidebar:
    st.header("Information")
    st.markdown("""
    ### How to Use:
    
    1. **Create New Type**: Define custom document types with specific extraction questions
    
    2. **Manage Types**: Edit or delete existing document types
    
    3. **Test Extraction**: Upload a sample PDF to test your configuration
    
    ### Tips:
    
    - Be specific with your questions
    - Use clear field names for easy identification
    - Test your configuration before batch processing
    - Questions should be answerable from the document
    
    ### Examples:
    
    **Invoice Document Type:**
    - What is the invoice number?
    - What is the total amount?
    - What is the due date?
    
    **Contract Document Type:**
    - Who are the contracting parties?
    - What is the contract value?
    - What is the contract duration?
    """)
    
    st.markdown("---")
    
    # Statistics
    if st.session_state.custom_document_types:
        st.markdown("### Current Statistics")
        st.metric("Document Types", len(st.session_state.custom_document_types))
        
        total_questions = sum(
            len(doc['questions']) 
            for doc in st.session_state.custom_document_types.values()
        )
        st.metric("Total Questions", total_questions)
        
        if st.button("Clear All Document Types", type="secondary"):
            st.session_state.custom_document_types = {}
            st.session_state.extraction_results = {}
            st.success("All document types cleared")
            st.rerun()