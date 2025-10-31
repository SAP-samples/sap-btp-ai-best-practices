"""
Vendor analysis functionality for the Vendor Performance Dashboard.
Contains functions for analyzing vendor performance and generating recommendations.
"""

import streamlit as st
from datetime import datetime
from core import utils
from ui import visualization
from config import settings

# Try to import the AI query parser (optional)
try:
    from ai.genai_query_parser import parse_procurement_query, generate_vendor_suggestion
    AI_ENABLED = True
except ImportError:
    AI_ENABLED = False
    print("Warning: AI query parser not available. Vendor Selection Assistant will run without AI features.")


def show_vendor_analysis(material_df, material, quantity, costs_config, matnr=None):
    """Display vendor analysis for a specific material with cost filtering
    
    Args:
        material_df: Dataframe with vendor data for the material
        material: Material description (MAKTX)
        quantity: Requested quantity (optional)
        costs_config: Cost configuration
        matnr: Material number (optional) for more specific identification
    """
    
    # Determine active cost components
    active_cost_columns = utils.get_active_cost_columns(material_df, costs_config)
    
    # Display vendor comparison with MATNR if available
    if matnr:
        st.markdown(f"### Vendor Comparison for Material: **{material.upper()}**")
        st.markdown(f"**Material Number:** {matnr}")
    else:
        st.markdown(f"### Vendor Comparison for Material: **{material.upper()}**")
    
    # Key metrics overview - best price, fastest delivery, etc.
    show_key_metrics(material_df)
    
    # Vendor comparison table (shown first by design)
    show_enhanced_vendor_comparison(material_df)
    
    # AI Vendor Suggestion (shown after the comparison by design)
    if AI_ENABLED and len(material_df) > 0:
        show_ai_recommendation(material_df, material, quantity, active_cost_columns)

    # Add a separator for better visual organization
    st.markdown("---")
    
    # Visualizations
    st.markdown("### Visual Analysis")
    
    # Cost comparison chart
    fig_cost = visualization.create_cost_comparison_chart(material_df, material)
    if fig_cost:
        st.plotly_chart(fig_cost, use_container_width=True)
    
    # Cost breakdown analysis (using only active cost components)
    show_cost_breakdown(material_df, active_cost_columns, costs_config)
    
    # Performance scatter plot
    fig_scatter = visualization.create_performance_scatter(material_df, material)
    if fig_scatter:
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Download data option
    show_download_option(material_df, material)
    
    if quantity:
        st.info(f"Note: The requested quantity of {quantity:,} units has been identified and can be considered for your procurement decision.")


def show_ai_recommendation(material_df, material, quantity, active_cost_columns):
    """Show AI-powered vendor recommendation"""
    with st.spinner("Generating AI vendor recommendation..."):
        top_n_vendors = 5
        economic_metrics = ['EffectiveCostPerUnit_USD', 'AvgLeadTimeDays_raw', 'OnTimeRate_raw', 'InFullRate_raw']
        
        context_columns = ['VendorFullID'] + economic_metrics + active_cost_columns
        actual_context_columns = [col for col in context_columns if col in material_df.columns]
        
        top_vendors_df = material_df.nsmallest(top_n_vendors, 'EffectiveCostPerUnit_USD')[actual_context_columns].copy()
        vendor_data_json = top_vendors_df.to_json(orient='records', indent=2)
        
        ai_suggestion = generate_vendor_suggestion(vendor_data_json, material, quantity)
    
    if ai_suggestion:
        st.markdown("---")
        st.markdown("### AI Vendor Recommendation")
        st.markdown(ai_suggestion)
        st.markdown("---")


def show_key_metrics(material_df):
    """Show key performance metrics for vendors in a single four-column row."""
    min_cost = material_df['EffectiveCostPerUnit_USD'].min()
    best_price_vendor = material_df.loc[material_df['EffectiveCostPerUnit_USD'] == min_cost, 'VendorFullID'].values[0]

    min_lead_time = material_df['AvgLeadTimeDays_raw'].min()
    fastest_vendor = material_df.loc[material_df['AvgLeadTimeDays_raw'] == min_lead_time, 'VendorFullID'].values[0]

    max_ontime = material_df['OnTimeRate_raw'].max()
    most_ontime_vendor = material_df.loc[material_df['OnTimeRate_raw'] == max_ontime, 'VendorFullID'].values[0]

    max_infull = material_df['InFullRate_raw'].max()
    best_infull_vendor = material_df.loc[material_df['InFullRate_raw'] == max_infull, 'VendorFullID'].values[0]

    html = f"""
    <div class=\"metrics-grid\">
        <div class=\"metric-pill\">
            <div class=\"metric-pill-label\">Best Price</div>
            <div class=\"metric-pill-value\">${min_cost:.2f}</div>
            <div class=\"metric-pill-vendor\">{best_price_vendor}</div>
        </div>
        <div class=\"metric-pill\">
            <div class=\"metric-pill-label\">Fastest Delivery</div>
            <div class=\"metric-pill-value\">{min_lead_time:.0f} days</div>
            <div class=\"metric-pill-vendor\">{fastest_vendor}</div>
        </div>
        <div class=\"metric-pill\">
            <div class=\"metric-pill-label\">Best On-Time Rate</div>
            <div class=\"metric-pill-value\">{max_ontime:.1%}</div>
            <div class=\"metric-pill-vendor\">{most_ontime_vendor}</div>
        </div>
        <div class=\"metric-pill\">
            <div class=\"metric-pill-label\">Best In-Full Rate</div>
            <div class=\"metric-pill-value\">{max_infull:.1%}</div>
            <div class=\"metric-pill-vendor\">{best_infull_vendor}</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def show_vendor_comparison_table(material_df):
    """Show detailed vendor comparison table"""
    st.markdown("### Detailed Vendor Comparison")
    
    # Get columns to display
    actual_display_columns = [col for col in settings.DEFAULT_DISPLAY_COLUMNS if col in material_df.columns]
    
    # Format the dataframe for display
    display_df = utils.format_display_dataframe(material_df[actual_display_columns])
    
    # Display the dataframe
    sort_column = 'Effective Cost/Unit' if 'Effective Cost/Unit' in display_df.columns else display_df.columns[0]
    st.dataframe(
        display_df.sort_values(sort_column),
        use_container_width=True,
        height=400
    )
    
    # Show the enhanced vendor comparison view
    # show_enhanced_vendor_comparison(material_df)


def show_enhanced_vendor_comparison(material_df):
    """Show enhanced vendor comparison with transposed view and color coding"""
    st.markdown("### Vendor Comparison View")
    # st.markdown("### 🎯 Enhanced Vendor Comparison View")
    
    # Get top 5 vendors by effective cost
    top_vendors = material_df.nsmallest(5, 'EffectiveCostPerUnit_USD').copy()
    
    if len(top_vendors) == 0:
        st.warning("No vendors available for comparison")
        return
    
    # Identify the best vendor (lowest effective cost)
    best_vendor_idx = top_vendors['EffectiveCostPerUnit_USD'].idxmin()
    best_vendor_id = top_vendors.loc[best_vendor_idx, 'VendorFullID']
    
    # Prepare metrics for comparison
    metrics = {
        'Material Number': ('MATNR', 'text', None),  # Informational - no better/worse
        'Effective Cost/Unit': ('EffectiveCostPerUnit_USD', 'currency', True),  # lower is better
        'Base Price/Unit': ('AvgUnitPriceUSD_raw', 'currency', True),  # lower is better - This is the base price without tariffs and other costs
        'Tariff Impact': ('TariffImpact_raw_percent', 'tariff_percentage', True),  # lower is better
        'Logistics Cost': ('cost_Logistics', 'currency', True),  # lower is better - shipping and logistics costs
        'Lead Time (Days)': ('AvgLeadTimeDays_raw', 'number', True),  # lower is better
        'On-Time Rate': ('OnTimeRate_raw', 'percentage', False),  # higher is better
        'In-Full Rate': ('InFullRate_raw', 'percentage', False),  # higher is better
        'PO Count': ('POLineItemCount', 'number', False)  # higher is better (more experience)
    }
    
    # Create HTML for the enhanced table
    html = """
    <style>
        .vendor-comparison-table {
            border-collapse: collapse;
            width: 100%;
            margin-top: 20px;
            font-family: Arial, sans-serif;
        }
        .vendor-comparison-table th, .vendor-comparison-table td {
            border: 1px solid #e0e0e0;
            padding: 12px;
            text-align: center;
        }
        .vendor-comparison-table th {
            background-color: #f5f5f5;
            font-weight: bold;
        }
        .vendor-comparison-table .metric-name {
            text-align: left;
            font-weight: 500;
            background-color: #fafafa;
        }
        .best-vendor {
            background-color: #e3f2fd !important;
            border: 2px solid #2196f3 !important;
            font-weight: bold;
        }
        .better-value {
            color: #4caf50;
            font-weight: 500;
        }
        .worse-value {
            color: #f44336;
            font-weight: 500;
        }
        .vendor-header {
            font-size: 14px;
            font-weight: bold;
        }
        .highest-score-label {
            background-color: #2196f3;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            margin-bottom: 8px;
        }
        .create-po-btn {
            background-color: #0a6ed1;
            color: #ffffff;
            border: none;
            border-radius: 6px;
            padding: 6px 10px;
            cursor: pointer;
        }
    </style>
    
    <table class="vendor-comparison-table">
        <thead>
            <tr>
                <th class="metric-name">Metric</th>
    """
    
    # Add vendor headers
    for _, vendor in top_vendors.iterrows():
        vendor_id = vendor['VendorFullID']
        is_best = vendor_id == best_vendor_id
        header_class = 'best-vendor' if is_best else ''
        
        html += f'<th class="{header_class}">'
        if is_best:
            html += '<div class="highest-score-label">Recommended</div>'
        html += f'<div class="vendor-header">{vendor_id}</div></th>'
    
    html += "</tr></thead><tbody>"
    
    # Add metric rows
    for metric_name, (col_name, format_type, lower_is_better) in metrics.items():
        if col_name not in top_vendors.columns:
            continue
            
        html += f'<tr><td class="metric-name">{metric_name}</td>'
        
        # Get the recommended vendor's value for comparison
        recommended_value = top_vendors.loc[best_vendor_idx, col_name]
        
        # Add values for each vendor
        for _, vendor in top_vendors.iterrows():
            value = vendor[col_name]
            vendor_id = vendor['VendorFullID']
            is_best_vendor = vendor_id == best_vendor_id
            
            # Format value
            if format_type == 'currency':
                formatted_value = f"${value:.2f}"
            elif format_type == 'percentage':
                formatted_value = f"{value:.1%}"
            elif format_type == 'tariff_percentage':
                formatted_value = f"{value:.1f}%"
            elif format_type == 'text':
                formatted_value = str(value)
            else:
                formatted_value = f"{value:.1f}" if isinstance(value, float) else str(value)
            
            # Determine color coding
            cell_class = ''
            if is_best_vendor:
                cell_class = 'best-vendor'
            elif lower_is_better is not None:  # Only apply comparison if lower_is_better is defined
                # Compare with recommended vendor's value to determine if better or worse
                if lower_is_better:
                    if value < recommended_value:
                        cell_class = 'better-value'
                    elif value > recommended_value:
                        cell_class = 'worse-value'
                else:
                    if value > recommended_value:
                        cell_class = 'better-value'
                    elif value < recommended_value:
                        cell_class = 'worse-value'
            
            # Add indicator for non-best vendors
            indicator = ''
            if not is_best_vendor and value != recommended_value and lower_is_better is not None:
                if lower_is_better:
                    indicator = ' ▲' if value > recommended_value else ' ▼'
                else:
                    indicator = ' ▲' if value > recommended_value else ' ▼'
            
            html += f'<td class="{cell_class}">{formatted_value}{indicator}</td>'
        
        html += "</tr>"
    
    # Action row with mock Create PO buttons
    html += '<tr><td class="metric-name">Action</td>'
    for _, vendor in top_vendors.iterrows():
        html += '<td><button class="create-po-btn">Create PO</button></td>'
    html += "</tr>"

    html += "</tbody></table>"
    
    # Display the enhanced table
    st.markdown(html, unsafe_allow_html=True)
    
    # Add legend
    st.markdown("""
    <div style="margin-top: 20px; padding: 10px; background-color: #f5f5f5; border-radius: 4px;">
        <strong>Legend:</strong>
        <span style="color: #4caf50; margin-left: 10px;">● Green = Better than recommended</span>
        <span style="color: #f44336; margin-left: 10px;">● Red = Worse than recommended</span>
        <span style="background-color: #e3f2fd; padding: 2px 6px; margin-left: 10px; border: 1px solid #2196f3;">Blue = Recommended vendor</span>
    </div>
    """, unsafe_allow_html=True)


def show_cost_breakdown(material_df, active_cost_columns, costs_config):
    """Show cost component breakdown analysis"""
    if active_cost_columns and any(material_df[col].sum() > 0 for col in active_cost_columns):
        st.markdown("### Cost Component Breakdown (Active Components Only)")
        
        # Show which cost components are active
        with st.expander("Active Cost Components"):
            active_names = [col.replace('cost_', '').replace('_', ' ').title() for col in active_cost_columns]
            st.write(f"**Active:** {', '.join(active_names)}")
            if costs_config:
                inactive_cost_components = [col for col, active in costs_config.items() if active == "False"]
                if inactive_cost_components:
                    inactive_names = [col.replace('cost_', '').replace('_', ' ').title() for col in inactive_cost_components]
                    st.write(f"**Inactive:** {', '.join(inactive_names)}")

            st.write("Detailed explanation of each cost component and their calculation can be found in the tab 'Settings'.")
        
        # Create a consolidated cost breakdown chart
        fig_breakdown = visualization.create_cost_breakdown_chart(material_df, active_cost_columns)
        if fig_breakdown:
            st.plotly_chart(fig_breakdown, use_container_width=True)
        else:
            st.info("No cost breakdown data available for the selected vendors.")


def show_download_option(material_df, material):
    """Show option to download vendor data as CSV"""
    st.markdown("### Export Data")
    csv_data = material_df.to_csv(index=False)
    st.download_button(
        label="Download Vendor Data as CSV",
        data=csv_data,
        file_name=f"vendor_analysis_{material}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )


def render_vendor_selection_assistant(tab, df_filtered, costs_config):
    """Render the vendor selection assistant tab"""
    with tab:
        st.subheader("AI-Powered Vendor Selection Assistant")
        
        # Check if dataframe is empty
        if df_filtered.empty or 'MAKTX' not in df_filtered.columns:
            st.warning("No data available. Please select a Material Number (MATNR) or Material Description (MAKTX) from the sidebar to begin analysis.")
            return
        
        # Load material relationships
        from core import data_loader
        material_relationships = data_loader.get_material_relationships()
        
        if not AI_ENABLED:
            st.warning("⚠️ AI features are not available. Missing genai_query_parser module.")
            st.info("You can still use the manual search functionality below.")
            manual_search = True
            analyze_button = False
            user_query = ""
        # else:
        #     st.markdown("### Query Input")
            
        #     # AI-powered query input
        #     user_query = st.text_area(
        #         "Enter your procurement query:", 
        #         value="I want to buy 1000 units of EMN-MOTOR",
        #         help="Describe what you want to procure. Example: 'I need 500 units of EMN-HANDLE' or 'Looking for ECR-SENSOR suppliers'"
        #     )
            
        #     col1, col2 = st.columns([1, 1])
        #     with col1:
        #         analyze_button = st.button("Analyze Query with AI", type="primary")
            
        #     with col2:
        #         manual_search = st.checkbox("Use Manual Search Instead")
        
        manual_search = True
        # Manual search options
        if manual_search or not AI_ENABLED:
            # st.markdown("#### Manual Search")
            available_materials = sorted(df_filtered['MAKTX'].unique())
            selected_material_assistant = st.selectbox(
                "Select Material (MAKTX):",
                options=[''] + available_materials,
                help="Choose from available materials in the dataset"
            )
            
            if selected_material_assistant:
                # Check if this MAKTX has multiple MATNRs
                variant_count = material_relationships['maktx_counts'].get(selected_material_assistant, 0)
                
                if variant_count > 1:
                    # Show variant alert
                    st.warning(f"⚠️ **Multiple Material Numbers Found**")
                    st.info(f"The material description '{selected_material_assistant}' has **{variant_count} different material numbers (variants)**. Please select a specific material number for accurate vendor analysis.")
                    
                    # Get available MATNRs for this MAKTX
                    available_matnrs = material_relationships['maktx_to_matnrs'].get(selected_material_assistant, [])
                    
                    # Filter df to only show MATNRs for this MAKTX
                    if 'MATNR' in df_filtered.columns:
                        maktx_filtered_df = df_filtered[df_filtered['MAKTX'] == selected_material_assistant]
                        available_matnrs_in_data = sorted(maktx_filtered_df['MATNR'].unique())
                    else:
                        available_matnrs_in_data = available_matnrs
                    
                    # Show MATNR selector
                    selected_matnr_assistant = st.selectbox(
                        "Select Material Number (MATNR):",
                        options=[''] + available_matnrs_in_data,
                        help=f"Select a specific material number variant for {selected_material_assistant}"
                    )
                    
                    if selected_matnr_assistant:
                        # Filter by both MAKTX and MATNR
                        material_df = df_filtered[
                            (df_filtered['MAKTX'] == selected_material_assistant) & 
                            (df_filtered['MATNR'] == selected_matnr_assistant)
                        ].copy()
                        
                        if not material_df.empty:
                            st.success(f"Found {len(material_df)} vendors for {selected_material_assistant} (Material #: {selected_matnr_assistant})")
                            show_vendor_analysis(material_df, selected_material_assistant, None, costs_config, matnr=selected_matnr_assistant)
                        else:
                            st.error("No vendors found for this material number.")
                    else:
                        st.info("Please select a material number to continue.")
                else:
                    # Single MATNR or no MATNR info - proceed as before
                    material_df = df_filtered[df_filtered['MAKTX'] == selected_material_assistant].copy()
                    
                    # Get the MATNR if it exists and is unique
                    matnr_value = None
                    if 'MATNR' in material_df.columns:
                        unique_matnrs = material_df['MATNR'].unique()
                        if len(unique_matnrs) == 1:
                            matnr_value = unique_matnrs[0]
                    
                    if not material_df.empty:
                        if matnr_value:
                            st.success(f"Found {len(material_df)} vendors for {selected_material_assistant} (Material #: {matnr_value})")
                        else:
                            st.success(f"Found {len(material_df)} vendors for {selected_material_assistant}")
                        show_vendor_analysis(material_df, selected_material_assistant, None, costs_config, matnr=matnr_value)
            else:
                st.info("Please select a material to see vendor analysis.")
        
        # # AI-powered analysis
        # elif analyze_button and user_query.strip():
        #     process_ai_query(user_query, df_filtered, costs_config)


# def process_ai_query(user_query, df_filtered, costs_config):
#     """Process an AI query and display results"""
#     # Load material relationships
#     from core import data_loader
#     material_relationships = data_loader.get_material_relationships()
    
#     with st.spinner("Analyzing your query with AI..."):
#         parsed_info = parse_procurement_query(user_query)
    
#     if not parsed_info:
#         st.error("Failed to parse the query. The AI service might be unavailable.")
#     elif parsed_info.get("error"):
#         st.error(f"Error during query parsing: {parsed_info.get('error')}")
#     else:
#         material = parsed_info.get("material")
#         quantity = parsed_info.get("quantity")
        
#         st.markdown("### Query Interpretation")
#         col1, col2 = st.columns(2)
#         with col1:
#             st.metric("Identified Material", material if material else "Not specified")
#         with col2:
#             st.metric("Identified Quantity", f"{quantity:,}" if quantity else "Not specified")
        
#         if not material:
#             st.warning("Material could not be identified. Please try rephrasing your query.")
#             available_materials = sorted(df_filtered['MAKTX'].unique())
#             st.info(f"Available materials: {', '.join(available_materials[:10])}{'...' if len(available_materials) > 10 else ''}")
#         else:
#             # Filter data for the identified material
#             material_df = df_filtered[df_filtered['MAKTX'].str.contains(material, case=False, na=False)].copy()
            
#             if material_df.empty:
#                 st.warning(f"No vendors found for material '{material}'. Trying broader search...")
#                 # Try a broader search
#                 material_df = df_filtered[df_filtered['MAKTX'].str.contains(material.split('-')[0] if '-' in material else material, case=False, na=False)].copy()
                
#                 if material_df.empty:
#                     available_materials = sorted(df_filtered['MAKTX'].unique())
#                     st.error(f"No vendors found. Available materials: {', '.join(available_materials[:20])}")
#                 else:
#                     # Check for multiple material numbers
#                     check_and_process_material_variants(material_df, material, quantity, costs_config, material_relationships)
#             else:
#                 # Check for multiple material numbers
#                 check_and_process_material_variants(material_df, material, quantity, costs_config, material_relationships)


def check_and_process_material_variants(material_df, material, quantity, costs_config, material_relationships):
    """Check if material has multiple variants and handle accordingly"""
    # Get unique MAKTX values in the filtered data
    unique_maktx = material_df['MAKTX'].unique()
    
    # If multiple MAKTX found, let user select
    if len(unique_maktx) > 1:
        st.info(f"Found multiple material descriptions matching '{material}'")
        selected_maktx = st.selectbox(
            "Select specific material description:",
            options=unique_maktx,
            help="Multiple materials matched your search. Please select the specific one."
        )
        material_df = material_df[material_df['MAKTX'] == selected_maktx].copy()
        material = selected_maktx
    else:
        material = unique_maktx[0]
    
    # Now check for multiple MATNRs
    if 'MATNR' in material_df.columns:
        unique_matnrs = material_df['MATNR'].unique()
        variant_count = len(unique_matnrs)
        
        if variant_count > 1:
            # Show variant alert
            st.warning(f"⚠️ **Multiple Material Numbers Found**")
            st.info(f"The material '{material}' has **{variant_count} different material numbers (variants)**. Please select a specific material number for accurate vendor analysis.")
            
            # Show MATNR selector
            selected_matnr = st.selectbox(
                "Select Material Number (MATNR):",
                options=[''] + list(unique_matnrs),
                help=f"Select a specific material number variant for {material}"
            )
            
            if selected_matnr:
                # Filter by MATNR
                material_df = material_df[material_df['MATNR'] == selected_matnr].copy()
                show_vendor_analysis(material_df, material, quantity, costs_config, matnr=selected_matnr)
            else:
                st.info("Please select a material number to continue.")
        else:
            # Single MATNR
            matnr = unique_matnrs[0] if variant_count == 1 else None
            show_vendor_analysis(material_df, material, quantity, costs_config, matnr=matnr)
    else:
        # No MATNR column
        show_vendor_analysis(material_df, material, quantity, costs_config)


def render_vendor_selection_assistant_standalone(df_filtered, costs_config):
    """Standalone version for lazy loading"""
    class DummyTab:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    
    render_vendor_selection_assistant(DummyTab(), df_filtered, costs_config)