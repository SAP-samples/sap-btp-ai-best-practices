"""
UI components for the AI Supplier Sourcing Optimizer.
Contains functions for creating UI elements, filters, and dashboard layout.
"""

import streamlit as st
from datetime import datetime
from core import utils
from ui import visualization
from config import settings


def apply_custom_styles():
    """Apply custom CSS styles for the dashboard"""
    st.markdown("""
    <style>
    /* Style the header area */
    .stApp h1 {
        /* Original gradient styling - kept for future use
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        */
        color: #0059bf;
        font-weight: 700;
    }
    
    /* Pill-like tab styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: transparent;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 25px;
        padding: 0.5rem 1.5rem;
        border: 2px solid #e5e7eb;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #f3f4f6;
        border-color: #3b82f6;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: #667eea;
    }
    
    /* Pill-like metric boxes */
    div[data-testid="metric-container"] {
        background: white;
        border-radius: 20px;
        padding: 1.2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #e5e7eb;
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
        border-color: #3b82f6;
    }
    
    div[data-testid="metric-container"] > div {
        font-weight: 600;
    }
    
    /* Adjust main container spacing */
    .block-container {
        padding-top: 1rem;
    }
    
    /* Page intro banner */
    .page-intro {
        background: #f5f6f7; /* template secondary background */
        border: 1px solid #e6e9f0;
        color: #32363a;
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        font-size: 1.05rem;
        line-height: 1.6;
        margin-bottom: 0.75rem;
    }

    /* Metrics grid layout - force four columns on wide screens */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 1rem;
        align-items: stretch;
        margin: 0.25rem 0 0.75rem 0;
    }
    @media (max-width: 1100px) {
        .metrics-grid { grid-template-columns: repeat(2, 1fr); }
    }

    /* Custom metric pills for vendor analysis */
    .metric-pill {
        background: white;
        border-radius: 24px;
        padding: 1.1rem 1.6rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        border: 1px solid #e6e9f0;
        transition: all 0.3s ease;
        text-align: center;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-pill:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 28px rgba(0,0,0,0.12);
        border-color: #0a6ed1;
    }
    
    .metric-pill-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #111827;
        margin-bottom: 0.25rem;
    }
    
    .metric-pill-label {
        font-size: 1.1rem;
        color: #4b5563;
        font-weight: 600; /* semibold */
        margin-bottom: 0.5rem;
    }
    
    .metric-pill-vendor {
        font-size: 0.85rem;
        color: #4b5563;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)


def setup_page_config():
    """Set up Streamlit page configuration"""
    st.set_page_config(
        page_title=settings.PAGE_TITLE,
        page_icon=settings.PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # Hide the default header
    # st.markdown("""
    # <style>
    #     header[data-testid="stHeader"] {
    #         visibility: hidden;
    #     }
    # </style>
    # """, unsafe_allow_html=True)



def create_sidebar_filters(df, all_materials=None):
    """Create and render sidebar filters with memory optimization
    
    Args:
        df: Current filtered dataframe
        all_materials: Dict with all available materials (MATNR and MAKTX)
    """
    st.sidebar.header("Configuration")
    st.sidebar.subheader("Filters")
    
    # Load material relationships for dependent filtering with profile awareness
    from core import data_loader
    from optimization.profile_manager import ProfileManager
    
    # Get active profile
    profile_manager = ProfileManager(".")
    active_profile = profile_manager.get_active_profile()
    
    # Use profile-aware data loading
    material_relationships = data_loader.get_material_relationships_profile_aware(active_profile)
    
    # Material filters with dependent selection
    # First, check if user has previously selected values (for maintaining state)
    prev_matnr = st.session_state.get('prev_matnr', 'All')
    prev_maktx = st.session_state.get('prev_maktx', 'All')
    
    # Material Description (MAKTX) filter - Show first as it's more user-friendly
    if all_materials and 'MAKTX' in all_materials:
        maktx_options = ['All'] + all_materials['MAKTX']
    else:
        # Fallback to current df
        if len(df) > 10000:
            material_counts = df['MAKTX'].value_counts()
            maktx_options = ['All'] + material_counts.index.tolist()
        else:
            maktx_options = ['All'] + sorted(df['MAKTX'].unique())
    
    # Check if we need to auto-select MAKTX based on selected MATNR
    auto_selected_maktx = None
    if 'matnr_filter' in st.session_state and st.session_state.matnr_filter != 'All':
        # Check if MAKTX should be auto-selected
        corresponding_maktx = material_relationships['matnr_to_maktx'].get(st.session_state.matnr_filter)
        if corresponding_maktx and corresponding_maktx in maktx_options:
            # Only auto-select if current MAKTX is 'All'
            if 'maktx_filter' not in st.session_state or st.session_state.maktx_filter == 'All':
                auto_selected_maktx = corresponding_maktx
    
    # Use index for selection if auto-selecting
    if auto_selected_maktx:
        default_index = maktx_options.index(auto_selected_maktx)
    else:
        # Use existing selection or default to 'All'
        current_maktx = st.session_state.get('maktx_filter', 'All')
        default_index = maktx_options.index(current_maktx) if current_maktx in maktx_options else 0
    
    selected_material = st.sidebar.selectbox(
        "Select Material Description", 
        maktx_options,
        index=default_index,
        key="maktx_filter",
        help="Select a specific material description"
    )
    
    # Per new design: remove MATNR filter from the sidebar and always return 'All'
    selected_matnr = 'All'
    
    # Store current selections for next render
    st.session_state.prev_matnr = selected_matnr
    st.session_state.prev_maktx = selected_material
    
    # Check data size for optimization
    data_size = len(df) if len(df) > 0 else 0
    use_optimized_filters = data_size > 10000
    
    # Vendor filter - optimized for large datasets (persist across pages)
    if data_size > 0 and 'Supplier_Name' in df.columns:
        if use_optimized_filters:
            vendor_counts = df['Supplier_Name'].value_counts()
            vendors = ['All'] + vendor_counts.index.tolist()
        else:
            vendors = ['All'] + sorted(df['Supplier_Name'].unique())
    else:
        vendors = ['All']
    # Ensure session value is valid for current options
    if 'vendor_filter' in st.session_state and st.session_state.vendor_filter not in vendors:
        st.session_state.vendor_filter = 'All'
    selected_vendor = st.sidebar.selectbox("Select Vendor", vendors, key="vendor_filter")
    
    # Country filter - optimized for large datasets (persist across pages)
    if data_size > 0 and 'Country' in df.columns:
        if use_optimized_filters:
            country_counts = df['Country'].value_counts()
            countries = ['All'] + country_counts.index.tolist()
        else:
            countries = ['All'] + sorted(df['Country'].unique())
    else:
        countries = ['All']
    if 'country_filter' in st.session_state and st.session_state.country_filter not in countries:
        st.session_state.country_filter = 'All'
    selected_country = st.sidebar.selectbox("Select Country", countries, key="country_filter")
    
    # Lead time filter - use efficient min/max extraction
    if data_size > 0 and 'MedianLeadTimeDays' in df.columns:
        lt_series = df['MedianLeadTimeDays']
        min_lt, max_lt = float(lt_series.min()), float(lt_series.max())
        # Handle case where all values are the same
        if min_lt == max_lt:
            # Add small buffer for slider
            if min_lt > 0:
                min_lt = min_lt * 0.9
                max_lt = max_lt * 1.1
            else:
                # If value is 0, create a 0-10 range
                min_lt = 0.0
                max_lt = 10.0
    else:
        min_lt, max_lt = 0.0, 100.0  # Default range
    
    # Persist slider across pages; clamp previous values to current bounds
    prev_lt_range = st.session_state.get('lead_time_range', (min_lt, max_lt))
    try:
        prev_start = float(prev_lt_range[0])
        prev_end = float(prev_lt_range[1])
    except Exception:
        prev_start, prev_end = min_lt, max_lt
    start = max(min_lt, min(prev_start, max_lt))
    end = max(min_lt, min(prev_end, max_lt))
    if end < start:
        start, end = min_lt, max_lt
    lt_range = st.sidebar.slider(
        "Lead Time Range (Days)", 
        min_value=min_lt, 
        max_value=max_lt, 
        value=(start, end),
        step=1.0,
        key="lead_time_range"
    )
    
    # Minimum PO count filter - use efficient max extraction
    if data_size > 0 and 'POLineItemCount' in df.columns:
        po_max = int(df['POLineItemCount'].max())
    else:
        po_max = 100  # Default max
    prev_min_po = int(st.session_state.get('min_po_line_items', 0))
    if prev_min_po < 0 or prev_min_po > po_max:
        prev_min_po = 0
    min_po_count = st.sidebar.number_input(
        "Minimum PO Line Items", 
        min_value=0, 
        max_value=po_max, 
        value=prev_min_po,
        key="min_po_line_items",
        help="Filter vendors with minimum number of PO line items"
    )
    
    # Return filter values as a dictionary
    filters = {
        'text_filters': {
            'MATNR': selected_matnr,
            'MAKTX': selected_material,
            'Supplier_Name': selected_vendor,
            'Country': selected_country
        },
        'range_filters': {
            'MedianLeadTimeDays': lt_range
        },
        'min_filters': {
            'POLineItemCount': min_po_count
        }
    }
    
    return filters


def display_sidebar_metrics(df_filtered):
    """Display summary metrics in the sidebar with memory info"""
    st.sidebar.markdown("---")
    st.sidebar.metric("Vendors in Analysis", len(df_filtered))
    
    # Handle empty dataframe or missing columns gracefully
    if len(df_filtered) == 0:
        st.sidebar.metric("Materials", 0)
        st.sidebar.metric("Unique Vendors", 0)
        st.sidebar.metric("Countries", 0)
    else:
        # Check for required columns and provide defaults if missing
        materials_count = df_filtered['MAKTX'].nunique() if 'MAKTX' in df_filtered.columns else 0
        vendors_count = df_filtered['Supplier_Name'].nunique() if 'Supplier_Name' in df_filtered.columns else 0
        countries_count = df_filtered['Country'].nunique() if 'Country' in df_filtered.columns else 0
        
        st.sidebar.metric("Materials", materials_count)
        st.sidebar.metric("Unique Vendors", vendors_count)
        st.sidebar.metric("Countries", countries_count)
    
    # Memory usage information for large datasets
    if len(df_filtered) > 1000:
        try:
            from core.utils import get_memory_usage_info
            memory_info = get_memory_usage_info(df_filtered)
            
            with st.sidebar.expander("Performance Info"):
                st.write(f"**Data Size:** {len(df_filtered):,} rows × {len(df_filtered.columns)} columns")
                st.write(f"**Memory Usage:** {memory_info['total_memory_mb']:.1f} MB")
                st.write(f"**Memory per Row:** {memory_info['memory_per_row']/1024:.1f} KB")
                
                # Performance recommendations
                if memory_info['total_memory_mb'] > 100:
                    st.warning("⚠️ Large dataset detected. Some visualizations may be sampled for performance.")
                elif memory_info['total_memory_mb'] > 50:
                    st.info("Moderate dataset size. Performance optimizations are active.")
        except ImportError:
            pass  # Skip if utils not available


def display_cost_config_status(costs_config):
    """Display cost configuration status in the sidebar"""
    if costs_config:
        active_costs = [cost for cost, active in costs_config.items() if active == "True"]
        st.sidebar.success(f"Cost configuration loaded: {len(active_costs)} active components")


def display_key_metrics(df_filtered, df=None):
    """Display key metrics in a four-column grid."""
    avg_lead_time = df_filtered['MedianLeadTimeDays'].mean()
    avg_ontime = df_filtered['OnTimeRate'].mean()
    avg_infull = df_filtered['InFullRate'].mean()
    avg_otif = df_filtered['Avg_OTIF_Rate'].mean()

    html = f"""
    <div class="metrics-grid">
        <div class="metric-pill">
            <div class="metric-pill-label">Avg Lead Time</div>
            <div class="metric-pill-value">{avg_lead_time:.1f} days</div>
        </div>
        <div class="metric-pill">
            <div class="metric-pill-label">Avg On-Time Rate</div>
            <div class="metric-pill-value">{avg_ontime:.1%}</div>
        </div>
        <div class="metric-pill">
            <div class="metric-pill-label">Avg In-Full Rate</div>
            <div class="metric-pill-value">{avg_infull:.1%}</div>
        </div>
        <div class="metric-pill">
            <div class="metric-pill-label">Avg OTIF Rate</div>
            <div class="metric-pill-value">{avg_otif:.1%}</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)




def display_page_intro(page_name: str, df_filtered):
    """Render page description and the top metrics bubbles.

    The description content is sourced from `ui.content.descriptions` to keep
    copy consistent across all pages. Metrics are shown only when a filter has
    produced a non-empty dataframe.
    """
    try:
        from ui.content import descriptions
        if page_name in descriptions:
            st.markdown(
                f"""
                <div class="page-intro">{descriptions[page_name]}</div>
                """,
                unsafe_allow_html=True,
            )
    except Exception:
        # Fail gracefully if descriptions are unavailable
        pass

    if df_filtered is not None and len(df_filtered) > 0:
        display_key_metrics(df_filtered)

def render_scatter_analysis_tab(tab, df_filtered):
    """Render the scatter analysis tab with memory protection"""
    with tab:
        st.subheader("Lead Time vs OTIF Performance")
        
        # CRITICAL: Prevent memory explosion on large datasets
        if len(df_filtered) > 25000:
            st.error(f"Dataset too large ({len(df_filtered):,} rows). Scatter analysis disabled to prevent memory issues.")
            st.info("Please apply filters to reduce dataset size below 25,000 rows.")
            return
        
        # Sample data for performance if still large
        df_plot = df_filtered
        if len(df_filtered) > 2000:
            df_plot = df_filtered.sample(n=2000, random_state=42)
            st.warning(f"Displaying 2,000 sampled data points out of {len(df_filtered):,} for performance")
        
        # Create scatter plot
        fig_scatter = visualization.create_scatter_plot(df_plot)
        
        # Display with full interactivity
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Analysis insights
        with st.expander("Analysis Insights"):
            st.write("**Quadrant Analysis:**")
            st.write("- **Top Right**: High OTIF, High Lead Time - Reliable but slow")
            st.write("- **Top Left**: High OTIF, Low Lead Time - **Ideal vendors**")
            st.write("- **Bottom Right**: Low OTIF, High Lead Time - **Needs improvement**")
            st.write("- **Bottom Left**: Low OTIF, Low Lead Time - Fast but unreliable")


def render_heatmap_tab(tab, df_filtered):
    """Render the performance heatmap tab with memory optimization"""
    with tab:
        st.subheader("Performance Heatmap (By Vendor Plant)")
        
        # Memory optimization: Limit heatmap size for large datasets
        max_heatmap_rows = 100  # Configurable limit for performance
        data_size = len(df_filtered)
        
        if data_size > 1:
            # For large datasets, sample or show top performers only
            if data_size > max_heatmap_rows:
                st.info(f"Showing top {max_heatmap_rows} performers out of {data_size} vendors for optimal performance.")
                # Sample top performers based on OTIF rate
                df_for_heatmap = df_filtered.nlargest(max_heatmap_rows, 'Avg_OTIF_Rate')
            else:
                df_for_heatmap = df_filtered
            
            # Create heatmap and get normalized data
            heatmap_result = visualization.create_performance_heatmap(df_for_heatmap)
            
            if heatmap_result:
                fig_heatmap, heatmap_normalized = heatmap_result
                
                # Add explanation of the metrics
                with st.expander("Understanding the heatmap"):
                    st.markdown("""
                    This heatmap shows the actual performance values for each vendor plant, with color coding based on normalized scores (0=worst to 1=best):
                    
                    - **Lead Time**: Shows actual number of days. Shorter lead times are better (greener).
                    - **On-Time Rate**: Shows actual percentage. Higher on-time rates are better (greener).
                    - **In-Full Rate**: Shows actual percentage. Higher in-full rates are better (greener).
                    - **OTIF Rate**: Shows actual combined on-time in-full percentage. Higher rates are better (greener).
                    - **Base Price**: Shows actual unit price in USD. Lower prices are better (greener).
                    - **Tariff Impact**: Shows actual tariff impact percentage. Lower impacts are better (greener).
                    
                    **Vendor-Plant Format**: Each row represents a unique vendor plant shown as Vendor-Country code format (e.g., "VendorName-US").
                    If there are multiple plants with the same vendor and country, the vendor ID is included in parentheses for uniqueness.
                    
                    **Overall Score**: The table below shows the normalized scores including an Overall Score, which is the average of all normalized metrics.
                    """)
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Top performers (based on normalized scores)
                st.subheader("Top Performers (Normalized Scores)")
                top_performers = heatmap_normalized.head(5)
                
                # Display top performers table
                # Format values for display while keeping them in a DataFrame
                formatted_df = top_performers.copy()
                for col in formatted_df.columns:
                    formatted_df[col] = formatted_df[col].round(3)
                
                # Display as a table with overall score
                st.dataframe(formatted_df, use_container_width=True)
                
                # Additional details for each top performer
                st.subheader("Details for Top 5 Performers")
                for idx, (vendor, scores) in enumerate(top_performers.iterrows(), 1):
                    with st.expander(f"#{idx}: {vendor} - Overall: {scores['Overall']:.3f}"):
                        # Extract original data for this vendor
                        try:
                            # Extract vendor data with exact match on VendorFullID - use df_for_heatmap for consistency
                            vendor_data = df_for_heatmap[df_for_heatmap['VendorFullID'] == vendor].iloc[0]
                            
                            # Create two columns for the details
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Performance Metrics:**")
                                st.markdown(f"- Lead Time: {vendor_data['MedianLeadTimeDays']:.1f} days")
                                st.markdown(f"- On-Time Rate: {vendor_data['OnTimeRate']:.1%}")
                                st.markdown(f"- In-Full Rate: {vendor_data['InFullRate']:.1%}")
                                st.markdown(f"- OTIF Rate: {vendor_data['Avg_OTIF_Rate']:.1%}")
                            
                            with col2:
                                st.markdown("**Cost Metrics:**")
                                st.markdown(f"- Base Price: ${vendor_data['EffectiveCostPerUnit_USD']:.2f} per unit")
                                st.markdown(f"- Tariff Impact: {vendor_data['TariffImpact_raw_percent']:.2f}%")
                                if 'POLineItemCount' in vendor_data:
                                    st.markdown(f"- PO Count: {vendor_data['POLineItemCount']} line items")
                                if 'Country' in vendor_data:
                                    st.markdown(f"- Country: {vendor_data['Country']}")
                        except (KeyError, IndexError) as e:
                            st.warning(f"Unable to fetch detailed data for this vendor: {e}")
        else:
            st.info("Need more than one vendor for heatmap analysis")


def render_geographic_tab(tab, df_filtered):
    """Render the geographic view tab with memory optimization"""
    with tab:
        st.subheader("Geographic Performance Distribution")
        
        # Check if we have country data
        if df_filtered['Country'].nunique() > 1 and 'Unknown' not in df_filtered['Country'].unique():
            
            # Memory-efficient country-level aggregation - only select needed columns
            agg_columns = {
                'MedianLeadTimeDays': 'mean',
                'Avg_OTIF_Rate': 'mean',
                'TariffImpact_raw_percent': 'mean',
                'Supplier_ID': 'count',
                'POLineItemCount': 'sum'
            }
            
            # Add logistics cost if available
            if 'cost_Logistics' in df_filtered.columns:
                agg_columns['cost_Logistics'] = 'mean'
            
            # Perform aggregation with explicit column selection to reduce memory usage
            country_agg = df_filtered.groupby('Country', as_index=False, observed=False).agg(agg_columns).round(3)
            
            # Rename columns efficiently
            rename_map = {
                'MedianLeadTimeDays': 'Avg_Lead_Time',
                'Avg_OTIF_Rate': 'Avg_OTIF_Rate',
                'TariffImpact_raw_percent': 'Avg_Tariff_Impact',
                'Supplier_ID': 'Vendor_Count',
                'POLineItemCount': 'Total_PO_Lines'
            }
            
            if 'cost_Logistics' in agg_columns:
                rename_map['cost_Logistics'] = 'Avg_Logistics_Cost'
            
            country_agg = country_agg.rename(columns=rename_map)
            
            # Create choropleth maps
            
            # Lead Time Map - Lower is better (green)
            st.subheader("Average Lead Time by Country")
            fig_map_lt = visualization.create_country_choropleth(
                country_agg,
                metric='Avg_Lead_Time',
                title="Average Lead Time by Country (Lower is Better)",
                color_scale='RdYlGn_r',  # Red to Green (reversed), low values are green
                labels={'Avg_Lead_Time': 'Avg Lead Time (Days)'}
            )
            st.plotly_chart(fig_map_lt, use_container_width=True)
            
            # OTIF Rate Map - Higher is better (green)
            st.subheader("Average OTIF Rate by Country")
            fig_map_otif = visualization.create_country_choropleth(
                country_agg,
                metric='Avg_OTIF_Rate',
                title="Average On-Time In-Full Rate by Country (Higher is Better)",
                color_scale='RdYlGn',  # Red to Green, higher values are green
                labels={'Avg_OTIF_Rate': 'Avg OTIF Rate'}
            )
            st.plotly_chart(fig_map_otif, use_container_width=True)
            
            # Tariff Impact Map - Lower is better (green)
            st.subheader("Average Tariff Impact by Country")
            fig_map_tariff = visualization.create_country_choropleth(
                country_agg,
                metric='Avg_Tariff_Impact',
                title="Average Tariff Impact by Country (Lower is Better)",
                color_scale='RdYlGn_r',  # Red to Green (reversed), lower values are green
                labels={'Avg_Tariff_Impact': 'Avg Tariff Impact (%)'}
            )
            st.plotly_chart(fig_map_tariff, use_container_width=True)
            
            # Logistics Cost Map - Lower is better (green) - only if column exists
            if 'Avg_Logistics_Cost' in country_agg.columns:
                st.subheader("Average Logistics Cost by Country")
                fig_map_logistics = visualization.create_country_choropleth(
                    country_agg,
                    metric='Avg_Logistics_Cost',
                    title="Average Logistics Cost by Country (Lower is Better)",
                    color_scale='RdYlGn_r',  # Red to Green (reversed), lower values are green
                    labels={'Avg_Logistics_Cost': 'Avg Logistics Cost (USD)'}
                )
                st.plotly_chart(fig_map_logistics, use_container_width=True)
            
            # Country comparison table
            st.subheader("Country Performance Summary")
            st.dataframe(country_agg.sort_values('Avg_Lead_Time'), use_container_width=True)
                
        else:
            st.info("Geographic maps are not available - insufficient country data or mapping incomplete.")
            
        # Material-level aggregation (always show this)
        render_material_performance(df_filtered)


def render_material_performance(df_filtered):
    """Render material performance charts and tables with memory optimization"""
    st.subheader("Performance by Material")
    
    # Memory optimization: Efficient aggregation with only required columns
    agg_dict = {
        'MedianLeadTimeDays': 'mean',
        'Avg_OTIF_Rate': 'mean',
        'Supplier_ID': 'count',
        'POLineItemCount': 'sum',
        'EffectiveCostPerUnit_USD': 'mean',
        'TariffImpact_raw_percent': 'mean'
    }
    
    # Only include columns that exist in the dataframe
    available_agg = {k: v for k, v in agg_dict.items() if k in df_filtered.columns}
    
    if not available_agg:
        st.warning("Insufficient data columns for material performance analysis.")
        return
    
    material_agg = df_filtered.groupby('MAKTX', as_index=False, observed=False).agg(available_agg).round(3)
    
    # Dynamic column renaming based on available columns
    rename_map = {
        'MedianLeadTimeDays': 'Avg_Lead_Time',
        'Avg_OTIF_Rate': 'Avg_OTIF_Rate',
        'Supplier_ID': 'Vendor_Count',
        'POLineItemCount': 'Total_PO_Lines',
        'EffectiveCostPerUnit_USD': 'Avg_Cost_Per_Unit',
        'TariffImpact_raw_percent': 'Avg_Tariff_Impact'
    }
    
    # Apply only available renames
    actual_renames = {k: v for k, v in rename_map.items() if k in material_agg.columns}
    material_agg = material_agg.rename(columns=actual_renames)
    
    # Sort by available cost column or first numeric column
    sort_column = 'Avg_Cost_Per_Unit' if 'Avg_Cost_Per_Unit' in material_agg.columns else material_agg.select_dtypes(include=['number']).columns[0]
    st.dataframe(material_agg.sort_values(sort_column), use_container_width=True)
    
    # Material comparison charts - only if we have sufficient data and columns
    if len(material_agg) > 1:
        col1, col2 = st.columns(2)
        
        # Lead time chart - only if lead time column exists
        if 'Avg_Lead_Time' in material_agg.columns:
            with col1:
                # Limit to top 10 for performance
                chart_data = material_agg.sort_values('Avg_Lead_Time').head(10)
                fig_lead_time = visualization.create_material_bar_chart(
                    chart_data,
                    x='Avg_Lead_Time',
                    y='MAKTX',  # Use original column name
                    orientation='h',
                    title="Average Lead Time by Material (Top 10)",
                    labels={'Avg_Lead_Time': 'Days'},
                    color='RdYlGn_r'
                )
                st.plotly_chart(fig_lead_time, use_container_width=True)
        
        # Cost chart - only if cost column exists
        if 'Avg_Cost_Per_Unit' in material_agg.columns:
            with col2:
                # Limit to top 10 for performance
                chart_data = material_agg.sort_values('Avg_Cost_Per_Unit').head(10)
                fig_cost = visualization.create_material_bar_chart(
                    chart_data,
                    x='Avg_Cost_Per_Unit',
                    y='MAKTX',  # Use original column name
                    orientation='h',
                    title="Average Cost Per Unit by Material (Top 10)",
                    labels={'Avg_Cost_Per_Unit': 'Cost (USD)'},
                    color='RdYlGn_r'
                )
                st.plotly_chart(fig_cost, use_container_width=True)


def render_data_table_tab(tab, df_filtered):
    """Render the data table tab with aggressive memory protection"""
    with tab:
        st.subheader("Vendor Performance Data")
        
        # CRITICAL: Prevent memory explosion on large datasets
        if len(df_filtered) > 100000:
            st.error(f"Dataset too large ({len(df_filtered):,} rows). Data table disabled to prevent memory issues.")
            st.info("Please apply filters to reduce dataset size below 100,000 rows.")
            return
        
        # Memory optimization: Limit displayed rows for large datasets
        max_display_rows = 2000  # Reduced from 5000
        data_size = len(df_filtered)
        
        if data_size > 10000:
            st.warning(f"Large dataset ({data_size:,} rows). Performance may be impacted.")
        
        # Add search functionality
        search_term = st.text_input("Search vendors:", placeholder="Enter vendor name or ID...")
        
        # Memory-efficient filtering: Use boolean indexing without copy until necessary
        if search_term:
            search_mask = (
                df_filtered['Supplier_Name'].str.contains(search_term, case=False, na=False) |
                df_filtered['Supplier_ID'].str.contains(search_term, case=False, na=False)
            )
            display_df = df_filtered[search_mask]
        else:
            display_df = df_filtered
        
        # Memory optimization: Limit rows if dataset is very large
        if len(display_df) > max_display_rows:
            st.warning(f"⚠️ Dataset is large ({len(display_df):,} rows). Showing top {max_display_rows:,} rows sorted by OTIF rate for optimal performance.")
            # Sort and take top rows before formatting to save memory
            sort_col = 'Avg_OTIF_Rate' if 'Avg_OTIF_Rate' in display_df.columns else display_df.columns[0]
            display_df = display_df.nlargest(max_display_rows, sort_col)
        
        # Column selection with cost components automatically included
        # Get cost columns before formatting to avoid processing entire dataframe
        cost_columns = [col for col in display_df.columns if col.startswith('cost_')]
        default_columns = ['Supplier_Name', 'MAKTX', 'Country', 'MedianLeadTimeDays', 'OnTimeRate', 'InFullRate', 
                          'Avg_OTIF_Rate', 'EffectiveCostPerUnit_USD', 'POLineItemCount'] + cost_columns
        
        # Filter default columns to only those that exist
        available_defaults = [col for col in default_columns if col in display_df.columns]
        
        cols_to_show = st.multiselect(
            "Select columns to display:",
            options=list(display_df.columns),
            default=available_defaults
        )
        
        if cols_to_show:
            # Memory optimization: Only format the columns that will be displayed
            display_subset = display_df[cols_to_show].copy()
            display_df_formatted = utils.format_display_dataframe(display_subset)
            
            # Sort efficiently
            sort_col = 'Avg_OTIF_Rate' if 'Avg_OTIF_Rate' in cols_to_show else cols_to_show[0]
            if sort_col in display_df_formatted.columns:
                # For formatted columns, use original data for sorting
                if sort_col in display_subset.columns:
                    sort_indices = display_subset[sort_col].sort_values(ascending=False).index
                    display_df_formatted = display_df_formatted.loc[sort_indices]
            
            st.dataframe(
                display_df_formatted,
                use_container_width=True,
                height=400
            )
            
            # Download button - only generate CSV when needed
            @st.cache_data
            def generate_csv(data, columns):
                return data[columns].to_csv(index=False)
            
            if st.download_button(
                label="Download CSV",
                data=generate_csv(display_subset, cols_to_show),
                file_name=f"vendor_performance_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            ):
                st.success("CSV download initiated!")
        
        # Display data size information
        if data_size > 1000:
            st.info(f"Dataset info: {data_size:,} total vendors, {len(display_df):,} after filtering")


# STANDALONE VERSIONS FOR LAZY LOADING
def render_scatter_analysis_tab_standalone(df_filtered):
    """Standalone version for lazy loading"""
    class DummyTab:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    render_scatter_analysis_tab(DummyTab(), df_filtered)

def render_heatmap_tab_standalone(df_filtered):
    """Standalone version for lazy loading"""
    class DummyTab:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    render_heatmap_tab(DummyTab(), df_filtered)

def render_geographic_tab_standalone(df_filtered):
    """Standalone version for lazy loading"""
    class DummyTab:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    render_geographic_tab(DummyTab(), df_filtered)

def render_data_table_tab_standalone(df_filtered):
    """Standalone version for lazy loading"""
    class DummyTab:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    render_data_table_tab(DummyTab(), df_filtered)