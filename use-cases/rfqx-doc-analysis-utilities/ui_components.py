"""
Reusable UI components for the RFQ application.

This module contains Streamlit UI components that can be reused across different
parts of the application to maintain consistency and reduce code duplication.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import uuid


class ProviderManager:
    """Manages provider-related UI components and state."""
    
    @staticmethod
    def render_provider_count_selector() -> int:
        """
        Render a selector for the number of providers.
        
        Returns:
            Selected number of providers
        """
        st.subheader("Provider Configuration")
        
        num_providers = st.number_input(
            "How many providers do you want to compare?",
            min_value=1,
            max_value=10,
            value=st.session_state.get('num_providers', 2),
            help="Select the number of different providers/suppliers you want to analyze"
        )
        
        # Initialize providers list if it doesn't exist or if count changed
        if ('providers' not in st.session_state or 
            len(st.session_state.providers) != num_providers):
            ProviderManager._initialize_providers(num_providers)
        
        st.session_state.num_providers = num_providers
        return num_providers
    
    @staticmethod
    def _initialize_providers(num_providers: int):
        """Initialize the providers list in session state."""
        providers = []
        for i in range(num_providers):
            providers.append({
                'id': str(uuid.uuid4()),
                'name': f'Provider {i + 1}',
                'files': [],
                'processed_content': '',
                'extracted_data': {},
                'status': 'pending',  # pending, processing, completed, error
                'error_message': '',
                'token_count': 0
            })
        st.session_state.providers = providers
    
    @staticmethod
    def render_provider_section(provider_idx: int) -> Dict[str, Any]:
        """
        Render a complete provider section with file upload and configuration.
        
        Args:
            provider_idx: Index of the provider in the session state
            
        Returns:
            Updated provider data
        """
        provider = st.session_state.providers[provider_idx]
        
        with st.expander(f"{provider['name']}", expanded=True):
            # Provider name input
            new_name = st.text_input(
                "Provider Name",
                value=provider['name'],
                key=f"provider_name_{provider['id']}",
                help="Enter a descriptive name for this provider"
            )
            
            if new_name != provider['name']:
                st.session_state.providers[provider_idx]['name'] = new_name
                provider['name'] = new_name
            
            # File uploader
            uploaded_files = st.file_uploader(
                "Upload RFQ Documents",
                type=['pdf', 'xlsx', 'xls', 'csv'],
                accept_multiple_files=True,
                key=f"uploader_{provider['id']}",
                help="Upload PDF, Excel (.xlsx, .xls), or CSV files related to this provider"
            )
            
            if uploaded_files:
                st.session_state.providers[provider_idx]['files'] = uploaded_files
                
                # Show file preview
                st.write("**Uploaded Files:**")
                for file in uploaded_files:
                    file_size = file.size / 1024  # KB
                    if file_size < 1024:
                        size_str = f"{file_size:.1f} KB"
                    else:
                        size_str = f"{file_size/1024:.1f} MB"
                    
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"{file.name}")
                    with col2:
                        st.write(f"{size_str}")
                    with col3:
                        st.write(f"{file.type.split('/')[-1].upper()}")
            
            # Show status if provider has been processed
            if provider['status'] != 'pending':
                StatusDisplay.render_provider_status(provider)
        
        return st.session_state.providers[provider_idx]


class FeatureConfiguration:
    """Manages feature configuration UI components."""
    
    @staticmethod
    def render_feature_schema_viewer(base_schema: Dict[str, Dict[str, str]]) -> Tuple[Dict[str, Dict[str, str]], bool]:
        """
        Render an expandable view of the feature extraction schema with custom feature input.
        
        Args:
            base_schema: The base RFQ extraction schema
            
        Returns:
            Tuple of (updated_schema, dynamic_extraction_enabled)
        """
        st.subheader("Attribute Extraction Configuration")
        
        with st.expander("View Extractable Features", expanded=False):
            st.write("The following features will be extracted from each provider's documents:")
            
            # Display base schema categories
            for category, fields in base_schema.items():
                st.write(f"**{category.replace('_', ' ').title()}**")
                
                for field_name, field_type in fields.items():
                    st.write(f"  • {field_name.replace('_', ' ').title()}")
                
                st.write("")  # Add spacing
            
            # Custom features section
            st.write("**Manually Requested Attributes**")
            
            # Initialize custom features in session state
            if 'custom_features' not in st.session_state:
                st.session_state.custom_features = []
            
            # Input for new custom feature
            col1, col2 = st.columns([3, 1])
            with col1:
                new_feature = st.text_input(
                    "Add custom attribute",
                    placeholder="e.g., Environmental compliance requirements",
                    key="new_custom_feature"
                )
            with col2:
                if st.button("Add Attribute", type="secondary"):
                    if new_feature and new_feature not in st.session_state.custom_features:
                        st.session_state.custom_features.append(new_feature)
                        st.rerun()
            
            # Display existing custom features
            if st.session_state.custom_features:
                for i, feature in enumerate(st.session_state.custom_features):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"  • {feature}")
                    with col2:
                        if st.button("Remove", key=f"remove_feature_{i}", type="secondary"):
                            st.session_state.custom_features.pop(i)
                            st.rerun()
            else:
                st.write("  *No custom attributes added*")
        
        # Dynamic extraction toggle
        dynamic_extraction = st.checkbox(
            "Enable Dynamic Attribute Extraction",
            value=st.session_state.get('dynamic_extraction_enabled', False),
            help="When enabled, the system will attempt to extract additional attributes beyond the predefined schema"
        )
        st.session_state.dynamic_extraction_enabled = dynamic_extraction
        
        # Create updated schema with custom features
        updated_schema = base_schema.copy()
        if st.session_state.custom_features:
            updated_schema['manually_requested_features'] = {
                feature.lower().replace(' ', '_'): 'string' 
                for feature in st.session_state.custom_features
            }
        
        return updated_schema, dynamic_extraction
    
    @staticmethod
    def render_toggleable_feature_configuration(base_schema: Dict[str, Dict[str, str]]) -> Tuple[Dict[str, Dict[str, str]], bool, bool]:
        """
        Render toggleable feature configuration with individual and group-level controls.
        
        Args:
            base_schema: The base RFQ extraction schema
            
        Returns:
            Tuple of (filtered_schema, dynamic_extraction_enabled, has_selected_features)
        """
        st.subheader("Attribute Extraction Configuration")
        
        # Initialize feature activation state if not exists
        if 'feature_activation' not in st.session_state:
            # Initialize all features as deactivated (False)
            st.session_state.feature_activation = {}
            for category, fields in base_schema.items():
                st.session_state.feature_activation[category] = {}
                for field_name in fields.keys():
                    st.session_state.feature_activation[category][field_name] = False
        
        # Initialize custom features in session state
        if 'custom_features' not in st.session_state:
            st.session_state.custom_features = []
        
        with st.expander("Configure Attribute Extraction", expanded=True):
            st.write("Select which attributes to extract from documents. **All attributes are deactivated by default.**")
            st.warning("You must select at least one attribute to process documents.")
            
            # Track total selected features
            total_selected = 0
            
            # Render each category with group controls
            for category, fields in base_schema.items():
                category_title = category.replace('_', ' ').title()
                
                # Count selected fields in this category
                selected_in_category = sum(1 for field in fields.keys() 
                                         if st.session_state.feature_activation.get(category, {}).get(field, False))
                total_fields_in_category = len(fields)
                
                # Determine group checkbox state
                all_selected = selected_in_category == total_fields_in_category and total_fields_in_category > 0
                some_selected = selected_in_category > 0 and not all_selected
                
                # Category header with group checkbox
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{category_title}**")
                with col2:
                    # Group toggle checkbox
                    group_key = f"group_{category}"
                    group_checked = st.checkbox(
                        f"Select All ({selected_in_category}/{total_fields_in_category})",
                        value=all_selected,
                        key=group_key,
                        help=f"Toggle all attributes in {category_title}"
                    )
                    
                    # Handle group toggle
                    if group_checked != all_selected:
                        for field_name in fields.keys():
                            st.session_state.feature_activation[category][field_name] = group_checked
                        st.rerun()
                
                # Individual feature checkboxes
                for field_name, field_type in fields.items():
                    field_title = field_name.replace('_', ' ').title()
                    field_key = f"field_{category}_{field_name}"
                    
                    field_checked = st.checkbox(
                        f"   {field_title}",
                        value=st.session_state.feature_activation.get(category, {}).get(field_name, False),
                        key=field_key,
                        help=f"Extract: {field_title}"
                    )
                    
                    # Update field state
                    if category not in st.session_state.feature_activation:
                        st.session_state.feature_activation[category] = {}
                    st.session_state.feature_activation[category][field_name] = field_checked
                    
                    if field_checked:
                        total_selected += 1
                
                st.write("")  # Add spacing between categories
            
            # Custom features section
            st.markdown("---")
            st.write("**Custom Attributes**")
            
            # Input for new custom feature
            col1, col2 = st.columns([3, 1])
            with col1:
                new_feature = st.text_input(
                    "Add custom attribute",
                    placeholder="e.g., Environmental compliance requirements",
                    key="new_custom_feature"
                )
            with col2:
                if st.button("Add Attribute", type="secondary"):
                    if new_feature and new_feature not in st.session_state.custom_features:
                        st.session_state.custom_features.append(new_feature)
                        # Initialize custom feature activation state
                        if 'manually_requested_features' not in st.session_state.feature_activation:
                            st.session_state.feature_activation['manually_requested_features'] = {}
                        clean_name = new_feature.lower().replace(' ', '_')
                        st.session_state.feature_activation['manually_requested_features'][clean_name] = False
                        st.rerun()
            
            # Display existing custom features with checkboxes
            if st.session_state.custom_features:
                if 'manually_requested_features' not in st.session_state.feature_activation:
                    st.session_state.feature_activation['manually_requested_features'] = {}
                
                for i, feature in enumerate(st.session_state.custom_features):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    clean_name = feature.lower().replace(' ', '_')
                    
                    with col1:
                        feature_checked = st.checkbox(
                            f"   {feature}",
                            value=st.session_state.feature_activation['manually_requested_features'].get(clean_name, False),
                            key=f"custom_feature_{i}"
                        )
                        st.session_state.feature_activation['manually_requested_features'][clean_name] = feature_checked
                        if feature_checked:
                            total_selected += 1
                    
                    with col3:
                        if st.button("Remove", key=f"remove_feature_{i}", type="secondary"):
                            st.session_state.custom_features.pop(i)
                            # Remove from activation state
                            if clean_name in st.session_state.feature_activation.get('manually_requested_features', {}):
                                del st.session_state.feature_activation['manually_requested_features'][clean_name]
                            st.rerun()
            else:
                st.write("  *No custom attributes added*")
            
            # Dynamic extraction toggle
            dynamic_extraction = st.checkbox(
                "Enable Dynamic Attribute Extraction",
                value=st.session_state.get('dynamic_extraction_enabled', False),
                help="When enabled, the system will attempt to extract additional features beyond the predefined schema"
            )
            st.session_state.dynamic_extraction_enabled = dynamic_extraction
            if dynamic_extraction:
                total_selected += 1  # Count dynamic extraction as selected features
            
            # Show selection summary
            st.markdown("---")
            if total_selected == 0:
                st.error(f"No attributes selected! Please select at least one feature to proceed.")
            else:
                st.success(f"{total_selected} attributes selected for extraction")
        
        # Create filtered schema with only active features
        filtered_schema = {}
        has_selected_features = total_selected > 0
        
        for category, fields in base_schema.items():
            active_fields = {}
            for field_name, field_type in fields.items():
                if st.session_state.feature_activation.get(category, {}).get(field_name, False):
                    active_fields[field_name] = field_type
            
            if active_fields:  # Only include categories with active fields
                filtered_schema[category] = active_fields
        
        # Add custom features if any are active
        if st.session_state.custom_features:
            active_custom_features = {}
            for feature in st.session_state.custom_features:
                clean_name = feature.lower().replace(' ', '_')
                if st.session_state.feature_activation.get('manually_requested_features', {}).get(clean_name, False):
                    active_custom_features[clean_name] = 'string'
            
            if active_custom_features:
                filtered_schema['manually_requested_features'] = active_custom_features
        
        return filtered_schema, dynamic_extraction, has_selected_features


class StatusDisplay:
    """Handles status and progress display components."""
    
    @staticmethod
    def render_provider_status(provider: Dict[str, Any]):
        """Render the status of a provider."""
        status = provider['status']
        
        if status == 'pending':
            st.info("Pending processing")
        elif status == 'processing':
            st.info("Processing documents...")
        elif status == 'completed':
            st.success("Processing completed")
            if provider.get('token_count', 0) > 0:
                st.caption(f"Processed content: ~{provider['token_count']:,} tokens")
        elif status == 'error':
            st.error(f"Error: {provider.get('error_message', 'Unknown error')}")
    
    @staticmethod
    def render_processing_progress(current: int, total: int, message: str = "Processing..."):
        """Render a progress bar with message."""
        progress = current / total if total > 0 else 0
        st.progress(progress)
        st.caption(f"{message} ({current}/{total})")
    
    @staticmethod
    def render_comparison_summary(comparison_data: Dict[str, Any]):
        """Render a summary of comparison results."""
        if not comparison_data:
            return
        
        metadata = comparison_data.get('metadata', {})
        stats = comparison_data.get('summary_statistics', {})
        
        st.subheader("Comparison Summary")
        
        # Determine if this is document comparison or provider comparison
        comparison_type = metadata.get('comparison_type', 'document')
        is_provider_comparison = comparison_type == 'multi_provider'
        
        # Get the count and appropriate labels
        if is_provider_comparison:
            entity_count = metadata.get('number_of_providers', 0)
            entity_label = "Providers"
            found_in_all_key = 'found_in_all_providers'
            completeness_prefix = 'provider'
            entity_key_prefix = 'provider_'
        else:
            entity_count = metadata.get('number_of_documents', 0)
            entity_label = "Documents"
            found_in_all_key = 'found_in_all_documents'
            completeness_prefix = 'doc'
            entity_key_prefix = 'document_'
        
        # Basic metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                entity_label, 
                entity_count
            )
        
        with col2:
            st.metric(
                "Total Fields", 
                stats.get('total_fields', 0)
            )
        
        with col3:
            st.metric(
                f"Found in All", 
                stats.get(found_in_all_key, 0)
            )
        
        with col4:
            st.metric(
                "Not Found", 
                stats.get('not_found_in_any', 0)
            )
        
        # Entity completeness (documents or providers)
        completeness_title = f"**{entity_label} Completeness:**"
        st.write(completeness_title)
        
        for i in range(1, entity_count + 1):
            entity_name = metadata.get(f'{entity_key_prefix}{i}', f'{entity_label[:-1]} {i}')
            completeness = stats.get(f'{completeness_prefix}{i}_completeness_percentage', 0)
            
            # Create a simple progress bar for completeness
            st.write(f"**{entity_name}:** {completeness:.1f}%")
            st.progress(completeness / 100)


class FileUploadHelper:
    """Helper components for file upload functionality."""
    
    @staticmethod
    def validate_uploaded_files(uploaded_files: List) -> Tuple[List, List[str]]:
        """
        Validate uploaded files and return valid files and error messages.
        
        Args:
            uploaded_files: List of uploaded file objects
            
        Returns:
            Tuple of (valid_files, error_messages)
        """
        valid_files = []
        errors = []
        
        max_file_size = 50 * 1024 * 1024  # 50MB limit
        
        for file in uploaded_files:
            # Check file size
            if file.size > max_file_size:
                errors.append(f"{file.name}: File too large (max 50MB)")
                continue
            
            # Check file type
            allowed_types = [
                'application/pdf',
                'application/vnd.ms-excel',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                'text/csv',
                'application/csv'
            ]
            
            if file.type not in allowed_types:
                errors.append(f"{file.name}: Unsupported file type ({file.type})")
                continue
            
            valid_files.append(file)
        
        return valid_files, errors
    
    @staticmethod
    def render_file_upload_help():
        """Render help information for file uploads."""
        with st.expander("File Upload Guidelines", expanded=False):
            st.write("""
            **Supported File Types:**
            - PDF documents (.pdf)
            - Excel spreadsheets (.xlsx, .xls)
            - CSV files (.csv)
            
            **File Size Limits:**
            - Maximum 50MB per file
            - For best performance, keep total upload under 100MB per provider
            
            **Tips for Best Results:**
            - Group documents by provider/supplier
            - Include all relevant documents for each provider in their section
            - Ensure Excel files have clear headers and data structure
            - CSV files should have column headers
            
            **Privacy & Security:**
            - Documents are processed locally when possible
            - Only statistical summaries and samples are sent to AI models
            - Your data is not permanently stored
            """)


def initialize_session_state():
    """Initialize all required session state variables."""
    defaults = {
        'providers': [],
        'num_providers': 2,
        'custom_features': [],
        'dynamic_extraction_enabled': False,
        'feature_activation': {},  # New: tracks which features are activated
        'processed_documents': {},
        'document_features': {},
        'chat_history': [],
        'graph_data': {},
        'comparison_report': "",
        'graph_image': None,
        'interactive_graph': None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value