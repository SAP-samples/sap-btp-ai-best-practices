"""
Optimized vendor selection functionality for the Vendor Performance Dashboard.
Contains functions for analyzing and visualizing optimization results.
"""

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Updated imports for new structure
from core import utils
from config import settings as config
from optimization.profile_manager import ProfileManager

# Import the optimization insights generator (try/except to handle missing dependencies)
try:
    from ai.insights_generator import generate_optimization_insights, prepare_optimization_data_for_insights, generate_actionable_todo_list
    INSIGHTS_ENABLED = True
except ImportError:
    INSIGHTS_ENABLED = False
    print("Warning: Optimization insights generator not available. Using static insights instead.")


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_optimization_data_cached(data_hash, profile_id=None):
    """
    Cached version of optimization data loading with profile awareness
    
    Args:
        data_hash (str): Hash representing the current data state
        profile_id (str): Profile ID to use. If None, uses the active profile.
    
    Returns:
        tuple: (comparison_df, optimized_df) or None if data loading fails
    """
    try:
        # Initialize ProfileManager and get profile-specific paths
        profile_manager = ProfileManager(".")
        if profile_id is None:
            profile_id = profile_manager.get_active_profile()
        
        profile_tables_dir = profile_manager.get_profile_tables_dir(profile_id)
        
        # Construct profile-specific file paths
        comparison_file = f'{profile_tables_dir}/comparison.csv'
        optimized_file = f'{profile_tables_dir}/optimized_allocation_matnr_vendor_matnr_ranking_tariff_values.csv'
        
        # Check if files exist before loading
        import os
        if not os.path.exists(comparison_file):
            st.error(f"Comparison file not found for {profile_id}: {comparison_file}")
            return None
        if not os.path.exists(optimized_file):
            st.error(f"Optimization results file not found for {profile_id}: {optimized_file}")
            return None
        
        # Load optimization results
        comparison_df = pd.read_csv(comparison_file)
        optimized_df = pd.read_csv(optimized_file)
        
        # Basic validation
        if comparison_df.empty or optimized_df.empty:
            st.warning(f"Optimization data files are empty for profile {profile_id}")
            return None
            
        # Fix data types to ensure consistent types for joining
        comparison_df['LIFNR'] = comparison_df['LIFNR'].astype(str).str.strip()
        optimized_df['LIFNR'] = optimized_df['LIFNR'].astype(str).str.strip()
        
        st.success(f"Optimization data loaded successfully from profile {profile_id}!")
        return comparison_df, optimized_df
        
    except FileNotFoundError as e:
        st.error(f"Optimization data files not found for profile {profile_id}. Please run optimization for this profile.")
        return None
    except Exception as e:
        st.error(f"Error loading optimization data for profile {profile_id}: {e}")
        return None

def load_optimization_data(df_base, profile_id=None):
    """
    Load optimization data files and merge with base vendor data
    
    Args:
        df_base (pd.DataFrame): Base vendor performance dataframe
        profile_id (str): Profile ID to use. If None, uses the active profile.
    
    Returns:
        tuple: (comparison_df, optimized_df, merged_df) or None if data loading fails
    """
    # Get profile ID if not provided
    if profile_id is None:
        profile_manager = ProfileManager(".")
        profile_id = profile_manager.get_active_profile()
    
    # Create a data hash for caching (include profile_id)
    data_hash = f"{len(df_base)}_{df_base['LIFNR'].iloc[0] if len(df_base) > 0 else 'empty'}_{profile_id}"
    
    # Try to load from cache first
    cached_data = load_optimization_data_cached(data_hash, profile_id)
    if cached_data is None:
        st.error(f"Optimization data not available for profile {profile_id}. Please ensure the required files exist in the profile's tables directory.")
        st.info(f"Required files: comparison.csv and optimized_allocation_matnr_vendor_matnr_ranking_tariff_values.csv")
        return None
    
    comparison_df, optimized_df = cached_data
    
    # Create a merged dataframe for comprehensive analysis
    # Use current vendor data as base and add optimization data
    
    # Ensure df_base also has string type for LIFNR
    df_base_copy = df_base.copy()
    df_base_copy['LIFNR'] = df_base_copy['LIFNR'].astype(str).str.strip()
    
    # Check if MAKTX exists in df_base_copy, if not, we'll get it from comparison_df
    merge_columns = ['LIFNR', 'MATNR']
    if 'MAKTX' not in df_base_copy.columns:
        # If MAKTX is not in df_base, include it from comparison_df
        comparison_columns = ['LIFNR', 'MATNR', 'MAKTX', 'Historical_Allocated_Quantity', 
                            'Optimized_Allocated_Quantity', 'Delta_Allocated_Quantity',
                            'Historical_Total_Effective_Cost_for_Combo', 
                            'Optimized_Total_Effective_Cost_for_Combo',
                            'Delta_Total_Effective_Cost_for_Combo']
    else:
        # If MAKTX exists in df_base, don't include it from comparison_df to avoid conflicts
        comparison_columns = ['LIFNR', 'MATNR', 'Historical_Allocated_Quantity', 
                            'Optimized_Allocated_Quantity', 'Delta_Allocated_Quantity',
                            'Historical_Total_Effective_Cost_for_Combo', 
                            'Optimized_Total_Effective_Cost_for_Combo',
                            'Delta_Total_Effective_Cost_for_Combo']
    
    merged_df = pd.merge(
        df_base_copy,
        comparison_df[comparison_columns],
        on=merge_columns,
        how='left'
    )
    
    # Fill missing values for vendors without optimization data
    merged_df['Historical_Allocated_Quantity'] = merged_df['Historical_Allocated_Quantity'].fillna(0)
    merged_df['Optimized_Allocated_Quantity'] = merged_df['Optimized_Allocated_Quantity'].fillna(0)
    merged_df['Delta_Allocated_Quantity'] = merged_df['Delta_Allocated_Quantity'].fillna(0)
    merged_df['Historical_Total_Effective_Cost_for_Combo'] = merged_df['Historical_Total_Effective_Cost_for_Combo'].fillna(0)
    merged_df['Optimized_Total_Effective_Cost_for_Combo'] = merged_df['Optimized_Total_Effective_Cost_for_Combo'].fillna(0)
    merged_df['Delta_Total_Effective_Cost_for_Combo'] = merged_df['Delta_Total_Effective_Cost_for_Combo'].fillna(0)
    
    return comparison_df, optimized_df, merged_df


def calculate_optimization_metrics(comparison_df, filters=None):
    """
    Calculate overall optimization impact metrics
    
    Args:
        comparison_df (pd.DataFrame): Comparison data with historical and optimized allocation
        filters (dict, optional): Filter settings from the UI

    Returns:
        dict: Calculated metrics
    """
    metrics = {}
    
    # Total cost savings
    metrics['total_cost_savings'] = comparison_df['Delta_Total_Effective_Cost_for_Combo'].sum()
    
    # Count of vendors with quantity adjustments (non-zero delta)
    metrics['allocation_changes'] = comparison_df[comparison_df['Delta_Allocated_Quantity'] != 0].shape[0]
    
    # Average cost reduction per unit (weighted by allocation)
    total_historical_cost = comparison_df['Historical_Total_Effective_Cost_for_Combo'].sum()
    total_optimized_cost = comparison_df['Optimized_Total_Effective_Cost_for_Combo'].sum()
    total_historical_quantity = comparison_df['Historical_Allocated_Quantity'].sum()
    total_optimized_quantity = comparison_df['Optimized_Allocated_Quantity'].sum()
    
    if total_historical_quantity > 0 and total_optimized_quantity > 0:
        historical_cost_per_unit = total_historical_cost / total_historical_quantity
        optimized_cost_per_unit = total_optimized_cost / total_optimized_quantity
        metrics['avg_cost_reduction'] = historical_cost_per_unit - optimized_cost_per_unit
    else:
        metrics['avg_cost_reduction'] = 0
    
    # Most improved category (material with highest relative savings)
    # If there's a single material filter applied, use that as the most improved category
    if filters and filters.get('text_filters', {}).get('MAKTX', 'All') != 'All':
        # If we're filtering to a specific material, that's our most improved category
        metrics['most_improved_category'] = filters['text_filters']['MAKTX']
        # Calculate savings percentage for this specific material
        material_data = comparison_df[comparison_df['MAKTX'] == filters['text_filters']['MAKTX']]
        if not material_data.empty:
            historical_cost = material_data['Historical_Total_Effective_Cost_for_Combo'].sum()
            optimized_cost = material_data['Optimized_Total_Effective_Cost_for_Combo'].sum()
            savings = historical_cost - optimized_cost
            if historical_cost > 0:
                metrics['most_improved_savings_pct'] = (savings / historical_cost) * 100
            else:
                metrics['most_improved_savings_pct'] = 0
        else:
            metrics['most_improved_savings_pct'] = 0
    else:
        # Calculate most improved category across all materials in the comparison data
        # First, aggregate by MATNR, then get the description
        material_savings = comparison_df.groupby('MATNR', observed=False).agg({
            'Historical_Total_Effective_Cost_for_Combo': 'sum',
            'Optimized_Total_Effective_Cost_for_Combo': 'sum',
            'MAKTX': 'first'  # Get the description for display
        })
        
        material_savings['savings'] = material_savings['Historical_Total_Effective_Cost_for_Combo'] - material_savings['Optimized_Total_Effective_Cost_for_Combo']
        material_savings['savings_percent'] = material_savings['savings'] / material_savings['Historical_Total_Effective_Cost_for_Combo'].replace(0, 1) * 100
        
        if not material_savings.empty:
            most_improved_idx = material_savings['savings_percent'].idxmax()
            # Display MAKTX but the aggregation was done by MATNR
            metrics['most_improved_category'] = material_savings.loc[most_improved_idx, 'MAKTX']
            metrics['most_improved_savings_pct'] = material_savings.loc[most_improved_idx, 'savings_percent']
        else:
            metrics['most_improved_category'] = "N/A"
            metrics['most_improved_savings_pct'] = 0
    
    return metrics


def create_dual_scatter_plots(df_filtered, historical_medians=None):
    """
    Create side-by-side scatter plots comparing historical vs optimized vendor performance
    
    Args:
        df_filtered (pd.DataFrame): Filtered dataframe with both historical and optimized data
        historical_medians (dict): Optional dictionary with historical median values for reference lines

    Returns:
        plotly.graph_objects.Figure: Figure with side-by-side scatter plots
    """
    # Filter data to only include vendors with optimization data
    df_with_opt = df_filtered[
        (df_filtered['Historical_Allocated_Quantity'] > 0) | 
        (df_filtered['Optimized_Allocated_Quantity'] > 0)
    ].copy()
    
    if df_with_opt.empty:
        return None
    
    # Calculate historical medians for reference lines if not provided
    if historical_medians is None:
        historical_medians = {
            'MedianLeadTimeDays': df_with_opt['MedianLeadTimeDays'].median(),
            'Avg_OTIF_Rate': df_with_opt['Avg_OTIF_Rate'].median()
        }
    
    # Create a subplot with 1 row and 2 columns
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Current State (Historical Allocation)", "Optimized State"),
        shared_yaxes=True,
        shared_xaxes=True,
        horizontal_spacing=0.02
    )
    
    # Common axis ranges
    x_range = [
        df_with_opt['MedianLeadTimeDays'].min() * 0.9,
        df_with_opt['MedianLeadTimeDays'].max() * 1.1
    ]
    y_range = [
        df_with_opt['Avg_OTIF_Rate'].min() * 0.9,
        min(1.0, df_with_opt['Avg_OTIF_Rate'].max() * 1.1)
    ]
    
    # Common color scale range for consistent coloring across both plots
    color_min = df_with_opt['EffectiveCostPerUnit_USD'].min()
    color_max = df_with_opt['EffectiveCostPerUnit_USD'].max()
    
    # LEFT PLOT - Historical state
    # Create a scatter trace for historical data
    historical_data = df_with_opt[df_with_opt['Historical_Allocated_Quantity'] > 0]
    if not historical_data.empty:
        # Generate hover text
        hover_text = []
        for _, row in historical_data.iterrows():
            hover_parts = [
                f"<b>{row['Supplier_Name']}</b><br>",
                f"LIFNR: {row['LIFNR']}<br>",
                f"Country: {row['Country']}<br>"
            ]
            # Add Material if MAKTX exists
            if 'MAKTX' in row and pd.notna(row['MAKTX']):
                hover_parts.append(f"Material: {row['MAKTX']}<br>")
            elif 'MATNR' in row:
                hover_parts.append(f"Material #: {row['MATNR']}<br>")
                
            hover_parts.extend([
                f"Lead Time: {row['MedianLeadTimeDays']:.1f} days<br>",
                f"OTIF Rate: {row['Avg_OTIF_Rate']:.1%}<br>",
                f"Historical Allocation: {row['Historical_Allocated_Quantity']:.0f} units<br>",
                f"Cost/Unit: ${row['EffectiveCostPerUnit_USD']:.2f}<br>",
                f"Total Cost: ${row['Historical_Total_Effective_Cost_for_Combo']:.2f}"
            ])
            hover_text.append(''.join(hover_parts))
        
        fig.add_trace(
            go.Scatter(
                x=historical_data['MedianLeadTimeDays'],
                y=historical_data['Avg_OTIF_Rate'],
                mode='markers',
                marker=dict(
                    size=historical_data['Historical_Allocated_Quantity'].apply(lambda x: max(10, min(50, x/20))),
                    color=historical_data['EffectiveCostPerUnit_USD'],
                    colorscale='Viridis',
                    cmin=color_min,
                    cmax=color_max,
                    colorbar=dict(title="Cost/Unit", x=0.45, xanchor="center"),
                    showscale=True
                ),
                text=hover_text,
                hoverinfo='text',
                name='Historical'
            ),
            row=1, col=1
        )
    
    # RIGHT PLOT - Optimized state
    # Create a scatter trace for optimized data
    optimized_data = df_with_opt[df_with_opt['Optimized_Allocated_Quantity'] > 0]
    if not optimized_data.empty:
        # Generate hover text
        hover_text = []
        for _, row in optimized_data.iterrows():
            hover_parts = [
                f"<b>{row['Supplier_Name']}</b><br>",
                f"LIFNR: {row['LIFNR']}<br>",
                f"Country: {row['Country']}<br>"
            ]
            # Add Material if MAKTX exists
            if 'MAKTX' in row and pd.notna(row['MAKTX']):
                hover_parts.append(f"Material: {row['MAKTX']}<br>")
            elif 'MATNR' in row:
                hover_parts.append(f"Material #: {row['MATNR']}<br>")
                
            hover_parts.extend([
                f"Lead Time: {row['MedianLeadTimeDays']:.1f} days<br>",
                f"OTIF Rate: {row['Avg_OTIF_Rate']:.1%}<br>",
                f"Optimized Allocation: {row['Optimized_Allocated_Quantity']:.0f} units<br>",
                f"Cost/Unit: ${row['EffectiveCostPerUnit_USD']:.2f}<br>",
                f"Total Cost: ${row['Optimized_Total_Effective_Cost_for_Combo']:.2f}"
            ])
            hover_text.append(''.join(hover_parts))
        
        fig.add_trace(
            go.Scatter(
                x=optimized_data['MedianLeadTimeDays'],
                y=optimized_data['Avg_OTIF_Rate'],
                mode='markers',
                marker=dict(
                    size=optimized_data['Optimized_Allocated_Quantity'].apply(lambda x: max(10, min(50, x/20))),
                    color=optimized_data['EffectiveCostPerUnit_USD'],
                    colorscale='Viridis',
                    cmin=color_min,
                    cmax=color_max,
                    showscale=False,
                    line=dict(width=2, color='black')
                ),
                text=hover_text,
                hoverinfo='text',
                name='Optimized'
            ),
            row=1, col=2
        )
    
    # Add reference lines (using historical medians for both plots)
    # Vertical line at median lead time
    fig.add_vline(
        x=historical_medians['MedianLeadTimeDays'], 
        line_dash="dot", 
        line_color="red",
        row=1, col=1
    )
    fig.add_vline(
        x=historical_medians['MedianLeadTimeDays'], 
        line_dash="dot", 
        line_color="red",
        row=1, col=2
    )
    
    # Horizontal line at median OTIF rate
    fig.add_hline(
        y=historical_medians['Avg_OTIF_Rate'], 
        line_dash="dot", 
        line_color="red",
        row=1, col=1
    )
    fig.add_hline(
        y=historical_medians['Avg_OTIF_Rate'], 
        line_dash="dot", 
        line_color="red",
        row=1, col=2
    )
    
    # Annotations to label reference lines
    fig.add_annotation(
        x=historical_medians['MedianLeadTimeDays'],
        y=y_range[0],
        text=f"Median Lead Time: {historical_medians['MedianLeadTimeDays']:.1f}d",
        showarrow=False,
        yshift=-40,
        row=1, col=1
    )
    
    fig.add_annotation(
        x=x_range[0],
        y=historical_medians['Avg_OTIF_Rate'],
        text=f"Median OTIF: {historical_medians['Avg_OTIF_Rate']:.1%}",
        showarrow=False,
        xshift=-40,
        textangle=-90,
        row=1, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        title="Lead Time vs OTIF Performance: Historical vs Optimized Allocation",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update x and y axis properties for both subplots
    fig.update_xaxes(
        title_text="Lead Time (Days)",
        range=x_range,
        row=1, col=1
    )
    fig.update_xaxes(
        title_text="Lead Time (Days)",
        range=x_range,
        row=1, col=2
    )
    
    fig.update_yaxes(
        title_text="OTIF Rate",
        range=y_range,
        tickformat=".0%",
        row=1, col=1
    )
    fig.update_yaxes(
        range=y_range,
        tickformat=".0%",
        row=1, col=2
    )
    
    # Add quadrant labels
    quadrant_labels = [
        {"x": x_range[0] + (x_range[1] - x_range[0]) * 0.25, 
         "y": y_range[0] + (y_range[1] - y_range[0]) * 0.75, 
         "text": "Fast & Reliable<br>(Ideal)", 
         "color": "green"},
        {"x": x_range[0] + (x_range[1] - x_range[0]) * 0.75, 
         "y": y_range[0] + (y_range[1] - y_range[0]) * 0.75, 
         "text": "Slow but Reliable", 
         "color": "orange"},
        {"x": x_range[0] + (x_range[1] - x_range[0]) * 0.25, 
         "y": y_range[0] + (y_range[1] - y_range[0]) * 0.25, 
         "text": "Fast but Unreliable", 
         "color": "orange"},
        {"x": x_range[0] + (x_range[1] - x_range[0]) * 0.75, 
         "y": y_range[0] + (y_range[1] - y_range[0]) * 0.25, 
         "text": "Slow & Unreliable<br>(Needs Improvement)", 
         "color": "red"}
    ]
    
    # Add annotations for both subplots
    for label in quadrant_labels:
        for col in [1, 2]:
            fig.add_annotation(
                x=label["x"],
                y=label["y"],
                text=label["text"],
                font=dict(
                    family="Arial",
                    size=10,
                    color=label["color"]
                ),
                align="center",
                showarrow=False,
                row=1, col=col
            )
    
    return fig


def create_comparison_tables(df_filtered):
    """
    Create side-by-side comparison tables for historical vs optimized vendor rankings
    
    Args:
        df_filtered (pd.DataFrame): Filtered dataframe with both historical and optimized data

    Returns:
        tuple: (historical_table, optimized_table) DataFrames formatted for display
    """
    # Filter data to only include vendors with optimization data
    df_with_opt = df_filtered[
        (df_filtered['Historical_Allocated_Quantity'] > 0) | 
        (df_filtered['Optimized_Allocated_Quantity'] > 0)
    ].copy()
    
    if df_with_opt.empty:
        return None, None
    
    # Create historical rankings table
    historical_table = df_with_opt[df_with_opt['Historical_Allocated_Quantity'] > 0].copy()
    
    # Select columns based on what's available
    columns_to_select = ['Supplier_Name', 'MATNR']
    column_names = ['Vendor', 'Material #']
    
    if 'MAKTX' in historical_table.columns:
        columns_to_select.append('MAKTX')
        column_names.append('Description')
    
    columns_to_select.extend(['MedianLeadTimeDays', 'Avg_OTIF_Rate', 
                             'Historical_Allocated_Quantity', 'EffectiveCostPerUnit_USD', 
                             'Historical_Total_Effective_Cost_for_Combo'])
    column_names.extend(['Lead Time', 'OTIF Rate', 'Allocation', 'Cost/Unit', 'Total Cost'])
    
    historical_table = historical_table[columns_to_select]
    historical_table.columns = column_names
    historical_table = historical_table.sort_values('Total Cost', ascending=False)
    
    # Create optimized rankings table
    optimized_table = df_with_opt[df_with_opt['Optimized_Allocated_Quantity'] > 0].copy()
    
    # Select columns based on what's available
    columns_to_select = ['Supplier_Name', 'MATNR']
    column_names = ['Vendor', 'Material #']
    
    if 'MAKTX' in optimized_table.columns:
        columns_to_select.append('MAKTX')
        column_names.append('Description')
    
    columns_to_select.extend(['MedianLeadTimeDays', 'Avg_OTIF_Rate', 
                             'Optimized_Allocated_Quantity', 'EffectiveCostPerUnit_USD', 
                             'Optimized_Total_Effective_Cost_for_Combo'])
    column_names.extend(['Lead Time', 'OTIF Rate', 'Allocation', 'Cost/Unit', 'Total Cost'])
    
    optimized_table = optimized_table[columns_to_select]
    optimized_table.columns = column_names
    optimized_table = optimized_table.sort_values('Total Cost', ascending=False)
    
    # Format the tables
    for table in [historical_table, optimized_table]:
        table['Lead Time'] = table['Lead Time'].apply(lambda x: f"{x:.1f} days")
        table['OTIF Rate'] = table['OTIF Rate'].apply(lambda x: f"{x:.1%}")
        table['Cost/Unit'] = table['Cost/Unit'].apply(lambda x: f"${x:.2f}")
        table['Total Cost'] = table['Total Cost'].apply(lambda x: f"${x:.2f}")
    
    return historical_table, optimized_table


def create_optimization_flow_chart(df_filtered):
    """
    Create a Sankey diagram showing allocation flow from historical to optimized state
    
    Args:
        df_filtered (pd.DataFrame): Filtered dataframe with both historical and optimized data

    Returns:
        plotly.graph_objects.Figure: Sankey diagram figure
    """
    # Filter data to include only vendors with optimization data
    df_with_opt = df_filtered[
        (df_filtered['Historical_Allocated_Quantity'] > 0) | 
        (df_filtered['Optimized_Allocated_Quantity'] > 0)
    ].copy()
    
    if df_with_opt.empty:
        return None
    
    # Prepare data for Sankey diagram
    # We need to create source-target pairs for flows
    
    # Create node data: historical vendors → material → optimized vendors
    # Group by MATNR but display MAKTX
    material_info = df_with_opt.groupby('MATNR')['MAKTX'].first().to_dict()
    material_numbers = list(material_info.keys())
    material_descriptions = [material_info[matnr] for matnr in material_numbers]
    
    historical_vendors = df_with_opt[df_with_opt['Historical_Allocated_Quantity'] > 0]['VendorFullID'].unique()
    optimized_vendors = df_with_opt[df_with_opt['Optimized_Allocated_Quantity'] > 0]['VendorFullID'].unique()
    
    # Create node labels
    node_labels = []
    node_labels.extend([f"H: {vendor}" for vendor in historical_vendors])
    node_labels.extend(material_descriptions)  # Display descriptions
    node_labels.extend([f"O: {vendor}" for vendor in optimized_vendors])
    
    # Map node labels to indices - use MATNR for mapping
    hist_vendor_idx = {vendor: i for i, vendor in enumerate(historical_vendors)}
    material_idx = {matnr: i + len(historical_vendors) for i, matnr in enumerate(material_numbers)}
    opt_vendor_idx = {vendor: i + len(historical_vendors) + len(material_numbers) for i, vendor in enumerate(optimized_vendors)}
    
    # Create source, target, and value arrays for Sankey diagram
    source = []
    target = []
    value = []
    
    # Historical vendor → Material flows
    for _, row in df_with_opt[df_with_opt['Historical_Allocated_Quantity'] > 0].iterrows():
        source.append(hist_vendor_idx[row['VendorFullID']])
        target.append(material_idx[row['MATNR']])  # Use MATNR for mapping
        value.append(row['Historical_Allocated_Quantity'])
    
    # Material → Optimized vendor flows
    for _, row in df_with_opt[df_with_opt['Optimized_Allocated_Quantity'] > 0].iterrows():
        source.append(material_idx[row['MATNR']])  # Use MATNR for mapping
        target.append(opt_vendor_idx[row['VendorFullID']])
        value.append(row['Optimized_Allocated_Quantity'])
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        ),
        # Text styling for better visibility
        textfont=dict(
            color="white",
            size=12,
            family="Arial, sans-serif"
        )
    )])
    
    fig.update_layout(
        title_text="Allocation Flow: Historical → Optimized",
        height=500
    )
    
    return fig


def create_top_materials_cost_breakdown(comparison_df, costs_config=None):
    """
    Create a horizontal stacked bar chart showing top 5 materials with highest savings opportunity,
    broken down by cost component
    
    Args:
        comparison_df (pd.DataFrame): Comparison data with cost component breakdown
        costs_config (dict): Optional configuration indicating which cost components to include
        
    Returns:
        plotly.graph_objects.Figure: Stacked horizontal bar chart or None if insufficient data
    """
    # Get unique materials
    unique_materials = comparison_df['MAKTX'].unique()
    
    if len(unique_materials) < 2:
        return None
        
    # Get delta cost component columns
    delta_cost_columns = [col for col in comparison_df.columns if col.startswith('Delta_Total_cost_')]
    
    if not delta_cost_columns:
        return None
        
    # Filter based on active cost components if config is provided
    if costs_config:
        active_columns = []
        for col in delta_cost_columns:
            # Extract component name from column
            component = col.replace('Delta_Total_', '')
            if component in costs_config and costs_config[component] == "True":
                active_columns.append(col)
        
        # Use active columns if any were found
        if active_columns:
            delta_cost_columns = active_columns
    
    # Group by material number and sum the delta costs, keep MAKTX for display
    material_aggregation = comparison_df.groupby('MATNR', observed=False).agg(
        {**{col: 'sum' for col in delta_cost_columns + ['Delta_Total_Effective_Cost_for_Combo']},
         'MAKTX': 'first'}  # Keep description for display
    ).reset_index()
    
    # Calculate absolute savings (positive value means cost reduction)
    material_aggregation['total_absolute_savings'] = -material_aggregation['Delta_Total_Effective_Cost_for_Combo']
    
    # Sort by total absolute savings and get top 5
    top_5_materials = material_aggregation.nlargest(5, 'total_absolute_savings')
    
    if top_5_materials.empty:
        return None
        
    # Create figure
    fig = go.Figure()
    
    # Extract component names and prepare data for stacked bars
    component_names = []
    component_data = []
    
    for col in delta_cost_columns:
        component = col.replace('Delta_Total_cost_', '').replace('_', ' ').title()
        component_names.append(component)
        # Convert to positive values for savings (negative delta means savings)
        component_data.append(-top_5_materials[col].values)
    
    # Add traces for each cost component
    for i, (name, data) in enumerate(zip(component_names, component_data)):
        fig.add_trace(go.Bar(
            name=name,
            y=top_5_materials['MAKTX'],
            x=data,
            orientation='h',
            hovertemplate='<b>%{y}</b><br>' +
                          f'{name}: $%{{x:,.2f}}<br>' +
                          '<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title="Top 5 Materials with Highest Optimization Opportunity - Cost Breakdown",
        xaxis_title="Savings by Component (USD)",
        yaxis_title="Material",
        barmode='stack',
        height=350,
        margin=dict(l=200, r=150, t=60, b=50),
        xaxis=dict(
            tickprefix="$",
            tickformat=",.0f"
        ),
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=10)
        ),
        yaxis=dict(
            autorange="reversed"  # Show highest savings at top
        )
    )
    
    # Add total savings annotation for each bar
    for idx, row in top_5_materials.iterrows():
        total_savings = row['total_absolute_savings']
        fig.add_annotation(
            x=total_savings,
            y=row['MAKTX'],
            text=f"${total_savings:,.0f}",
            xanchor="left",
            showarrow=False,
            font=dict(size=10, color="black"),
            xshift=5
        )
    
    return fig


def create_cost_breakdown_waterfall(comparison_df, costs_config=None):
    """
    Create a waterfall chart showing cost component changes from historical to optimized
    
    Args:
        comparison_df (pd.DataFrame): Comparison data with cost component breakdown
        costs_config (dict): Optional configuration indicating which cost components to include

    Returns:
        plotly.graph_objects.Figure: Waterfall chart figure
    """
    # Get cost component columns
    cost_columns = [col for col in comparison_df.columns if col.startswith('Delta_Total_cost_')]
    
    if not cost_columns:
        return None
    
    # Filter based on active cost components if config is provided
    if costs_config:
        active_components = []
        for component, is_active in costs_config.items():
            if is_active == "True":
                # Convert from config name to delta column name
                delta_col = f"Delta_Total_{component}"
                if delta_col in cost_columns:
                    active_components.append(delta_col)
        
        # If we have active components, use only those
        if active_components:
            cost_columns = active_components
    
    # Calculate the sum of each cost component delta
    cost_deltas = {}
    for col in cost_columns:
        # Extract component name from column name
        component = col.replace('Delta_Total_', '')
        cost_deltas[component] = comparison_df[col].sum()
    
    # Sort components by absolute impact (largest first)
    sorted_components = sorted(cost_deltas.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Create waterfall chart
    measures = ['relative'] * len(sorted_components) + ['total']
    x_labels = [component.replace('_', ' ').title() for component, _ in sorted_components] + ['Total Savings']
    y_values = [delta for _, delta in sorted_components] + [sum(cost_deltas.values())]
    
    # Set colors - green for cost reductions, red for cost increases
    colors = ['green' if val < 0 else 'red' for val in y_values[:-1]] + ['blue']
    
    fig = go.Figure(go.Waterfall(
        name="Cost Component Changes",
        orientation="v",
        measure=measures,
        x=x_labels,
        y=y_values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "green"}},
        increasing={"marker": {"color": "red"}},
        totals={"marker": {"color": "blue"}}
    ))
    
    fig.update_layout(
        title="Cost Impact Breakdown by Component (Active Components Only)",
        height=400,
        showlegend=False,
        yaxis=dict(
            title="Delta Cost (USD)",
            tickprefix="$"
        )
    )
    
    return fig


def show_static_insights(performance_changes):
    """
    Show static insights and recommendations based on performance changes
    
    Args:
        performance_changes (pd.DataFrame): DataFrame with performance metrics before and after optimization
    """
    # Key insights text
    st.markdown("#### Key Insights")
    
    # Generate insights based on the data
    insights = []
    
    # Lead time insight
    lead_time_change = performance_changes.loc[0, 'Percent Change']
    if lead_time_change < 0:
        insights.append(f"Average lead time improved by {abs(lead_time_change):.1f}%, reducing inventory holding costs and improving responsiveness.")
    else:
        insights.append(f"Average lead time increased by {lead_time_change:.1f}%, but this trade-off enabled cost savings in other areas.")
    
    # OTIF rate insight
    otif_change = performance_changes.loc[1, 'Percent Change']
    if otif_change > 0:
        insights.append(f"OTIF performance improved by {otif_change:.1f}%, reducing supply disruptions and improving customer satisfaction.")
    else:
        insights.append(f"OTIF performance decreased slightly by {abs(otif_change):.1f}%, but this trade-off enabled significant cost savings.")
    
    # Cost insight
    cost_change = performance_changes.loc[2, 'Percent Change']
    insights.append(f"Total procurement cost reduced by {abs(cost_change):.1f}%, representing significant savings while balancing performance metrics.")
    
    # Display insights
    for insight in insights:
        st.markdown(f"- {insight}")
    
    # Recommendations
    st.markdown("#### Recommendations")
    st.markdown("- Monitor vendor performance after reallocation to ensure continued compliance with contracted metrics")
    st.markdown("- Consider gradual implementation of allocation changes to minimize supply chain disruption")
    st.markdown("- Review vendor contracts to align incentives with optimized allocation strategy")
    st.markdown("- Implement regular optimization cycles to continuously improve supply chain performance")


def render_optimized_selection_tab(tab, df_filtered, filters=None):
    """
    Render the optimized vendor comparison tab
    
    Args:
        tab: Streamlit tab object
        df_filtered (pd.DataFrame): Filtered vendor data
        filters (dict, optional): Filter settings from the UI
    """
    with tab:
        st.subheader("Optimized Vendor Comparison")
        
        # Load costs configuration
        try:
            with open(config.COSTS_CONFIG_FILE, 'r') as f:
                config_data = json.load(f)
                # Handle both old and new format
                if 'cost_components' in config_data:
                    costs_config = config_data['cost_components']
                else:
                    costs_config = config_data
        except Exception:
            costs_config = None
            
        # Get active profile and load optimization data
        profile_manager = ProfileManager(".")
        active_profile = profile_manager.get_active_profile()
        optimization_data = load_optimization_data(df_filtered, active_profile)
        
        if optimization_data:
            comparison_df, optimized_df, merged_df = optimization_data
            
            # Filter comparison_df to only include entries that match our filtered data
            # Check if MAKTX exists in df_filtered before using it
            if 'MAKTX' in df_filtered.columns:
                # IMPORTANT: Get unique LIFNR+MAKTX combinations to avoid duplicates
                unique_vendor_material = df_filtered[['LIFNR', 'MAKTX']].drop_duplicates()
                filtered_comparison_df = pd.merge(
                    comparison_df,
                    unique_vendor_material,
                    on=['LIFNR', 'MAKTX'],
                    how='inner'
                )
            else:
                # If MAKTX is not in df_filtered, use MATNR instead
                unique_vendor_material = df_filtered[['LIFNR', 'MATNR']].drop_duplicates()
                filtered_comparison_df = pd.merge(
                    comparison_df,
                    unique_vendor_material,
                    on=['LIFNR', 'MATNR'],
                    how='inner'
                )
            
            # Calculate optimization metrics using filtered data
            metrics = calculate_optimization_metrics(filtered_comparison_df, filters=filters)
            
            # Header metrics section (4-column layout)
            st.markdown("### Optimization Impact Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            # Calculate the percentage based on filtered data
            historical_total = filtered_comparison_df['Historical_Total_Effective_Cost_for_Combo'].sum()
            
            with col1:
                # Determine if we're saving or spending more
                # Delta_Total_Effective_Cost_for_Combo is negative when we save money
                is_saving = metrics['total_cost_savings'] < 0
                cost_change_abs = abs(metrics['total_cost_savings'])
                cost_change_pct = (metrics['total_cost_savings'] / historical_total * 100) if historical_total > 0 else 0
                
                if is_saving:
                    label = "Total Cost Savings"
                    value_str = f"${cost_change_abs:,.2f}"
                    delta_str = f"-{abs(cost_change_pct):.1f}%" if historical_total > 0 else "N/A"
                else:
                    label = "Total Cost Increase"
                    value_str = f"${cost_change_abs:,.2f}"
                    delta_str = f"+{abs(cost_change_pct):.1f}%" if historical_total > 0 else "N/A"
                
                st.metric(
                    label,
                    value_str,
                    delta=delta_str,
                    delta_color="normal" if is_saving else "inverse"
                )
                
            with col2:
                st.metric(
                    "Allocation Changes",
                    f"{metrics['allocation_changes']}",
                    delta=f"{metrics['allocation_changes'] / len(filtered_comparison_df) * 100:.1f}%" if len(filtered_comparison_df) > 0 else "N/A"
                )
                
            with col3:
                # Calculate the historical cost per unit based on filtered data
                historical_quantity = filtered_comparison_df['Historical_Allocated_Quantity'].sum()
                historical_cost_per_unit = (historical_total / historical_quantity) if historical_quantity > 0 else 0
                
                # Determine if unit cost is reduced or increased
                # avg_cost_reduction is positive when we save money per unit
                is_unit_saving = metrics['avg_cost_reduction'] > 0
                unit_cost_change_abs = abs(metrics['avg_cost_reduction'])
                unit_cost_change_pct = (metrics['avg_cost_reduction'] / historical_cost_per_unit * 100) if historical_cost_per_unit > 0 else 0
                
                if is_unit_saving:
                    label = "Avg Cost Reduction/Unit"
                    delta_str = f"-{abs(unit_cost_change_pct):.1f}%" if historical_cost_per_unit > 0 else "N/A"
                else:
                    label = "Avg Cost Increase/Unit"
                    delta_str = f"+{abs(unit_cost_change_pct):.1f}%" if historical_cost_per_unit > 0 else "N/A"
                
                st.metric(
                    label,
                    f"${unit_cost_change_abs:.2f}",
                    delta=delta_str,
                    delta_color="inverse"
                )
                
            with col4:
                # Handle the most improved category
                # most_improved_savings_pct is positive when we save money
                is_category_saving = metrics['most_improved_savings_pct'] > 0
                category_change_abs = abs(metrics['most_improved_savings_pct'])
                
                if is_category_saving:
                    label = "Most Improved Category"
                    delta_str = f"-{category_change_abs:.1f}%"
                else:
                    label = "Most Affected Category"
                    delta_str = f"+{category_change_abs:.1f}%"
                
                st.metric(
                    label,
                    f"{metrics['most_improved_category']}",
                    delta=delta_str,
                    delta_color="normal" if is_category_saving else "inverse"
                )
            
            # Dual scatter plot analysis
            st.markdown("### Performance Quadrant Analysis")
            
            # Calculate historical medians for reference lines
            historical_medians = {
                'MedianLeadTimeDays': df_filtered['MedianLeadTimeDays'].median(),
                'Avg_OTIF_Rate': df_filtered['Avg_OTIF_Rate'].median()
            }
            
            # Create and display dual scatter plots
            scatter_fig = create_dual_scatter_plots(merged_df, historical_medians)
            if scatter_fig:
                st.plotly_chart(scatter_fig, use_container_width=True)
                
                # Analysis insights
                with st.expander("Analysis Insights"):
                    st.write("**Quadrant Analysis:**")
                    st.write("- **Top Left**: Fast & Reliable - Ideal vendors with low lead times and high OTIF rates")
                    st.write("- **Top Right**: Reliable but Slow - Good OTIF performance but longer lead times")
                    st.write("- **Bottom Left**: Fast but Unreliable - Quick delivery but inconsistent performance")
                    st.write("- **Bottom Right**: Slow & Unreliable - Vendors with highest improvement opportunity")
                    st.write("\n**Optimization Strategy:**")
                    st.write("- Point size represents allocation quantity - larger points indicate higher allocation")
                    st.write("- Color represents cost - compare the color distribution between historical and optimized views")
            else:
                st.info("Insufficient data for scatter plot visualization")
            
            # Interactive comparison tables
            st.markdown("### Vendor Allocation Comparison")
            
            # Create comparison tables
            historical_table, optimized_table = create_comparison_tables(merged_df)
            
            if historical_table is not None and optimized_table is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Historical Allocation")
                    st.dataframe(historical_table, use_container_width=True, height=300)
                
                with col2:
                    st.subheader("Optimized Allocation")
                    st.dataframe(optimized_table, use_container_width=True, height=300)
            else:
                st.info("Insufficient data for comparison tables")
            
            # Optimization insights panel
            st.markdown("### Optimization Insights")
            
            # Create allocation flow diagram
            flow_fig = create_optimization_flow_chart(merged_df)
            if flow_fig:
                st.plotly_chart(flow_fig, use_container_width=True)
            
            # Create top 5 materials cost breakdown chart
            top_materials_fig = create_top_materials_cost_breakdown(filtered_comparison_df, costs_config)
            if top_materials_fig:
                st.plotly_chart(top_materials_fig, use_container_width=True)
            
            # Create cost breakdown waterfall chart using filtered data
            waterfall_fig = create_cost_breakdown_waterfall(filtered_comparison_df, costs_config)
            if waterfall_fig:
                st.plotly_chart(waterfall_fig, use_container_width=True)
                
                # Display which cost components are active
                if costs_config:
                    active_components = [comp.replace("cost_", "").replace("_", " ").title() for comp, status in costs_config.items() if status == "True"]
                    inactive_components = [comp.replace("cost_", "").replace("_", " ").title() for comp, status in costs_config.items() if status == "False"]
                    
                    st.info(f"**Active cost components:** {', '.join(active_components)}")
                    # if inactive_components:
                    #     st.caption(f"Inactive components (excluded from analysis): {', '.join(inactive_components)}")
            
            # Generate and display prioritized actionable to-do list
            st.markdown("### Actionable To-Do List")
            st.markdown("Prioritized tasks based on economic impact")
            
            if INSIGHTS_ENABLED:
                try:
                    # Generate to-do list
                    todo_list = generate_actionable_todo_list(filtered_comparison_df, df_filtered)
                    
                    # DEBUG: Print the entire todo_list structure
                    print("\n=== DEBUG: COMPLETE TODO LIST STRUCTURE ===")
                    print(f"Number of material groups: {len(todo_list) if todo_list else 0}")
                    
                    if todo_list:
                        # Create expanders for each material group
                        for idx, task_group in enumerate(todo_list):
                            material = task_group["material"]
                            impact_formatted = task_group["economic_impact_formatted"]
                            display_impact = task_group["display_impact"]
                            is_savings = task_group["savings"]
                            effort_level = task_group["effort_level"]
                            
                            # DEBUG: Print details for each material group
                            print(f"\n--- Material Group {idx + 1}: {material} ---")
                            print(f"Economic Impact: {display_impact} ({impact_formatted})")
                            print(f"Is Savings: {is_savings}")
                            print(f"Effort Level: {effort_level}")
                            print(f"Number of tasks: {len(task_group['tasks'])}")
                            
                            impact_color = "green" if is_savings else "red"
                            impact_icon = "📉" if is_savings else "📈"
                            impact_sign = "+" if display_impact > 0 else "-"
                            
                            # Create expander title with economic impact
                            expander_title = f"{material} - Impact: {impact_icon} {impact_sign}{impact_formatted} - Effort: {effort_level}"
                            with st.expander(expander_title):
                                # Display tasks as a list
                                for i, task in enumerate(task_group["tasks"], 1):
                                    task_impact = task["economic_impact_formatted"]
                                    display_impact = task["display_impact"]
                                    
                                    # DEBUG: Print individual task details
                                    print(f"  Task {i}: {task['description']}")
                                    print(f"    Impact: {display_impact} ({task_impact})")
                                    
                                    # Keep the same symbols but use the sign from display_impact
                                    task_marker = "✅" if display_impact > 0 else "⚠️"
                                    impact_sign = "+" if display_impact > 0 else "-"
                                    # Handle the sign correctly
                                    impact_display = f"{impact_sign}{task_impact}" if display_impact != 0 else task_impact
                                    st.markdown(f"{i}. {task_marker} **{task['description']}** (Impact: {impact_display})")
                    
                        print("\n=== END DEBUG OUTPUT ===\n")
                    
                    else:
                        st.info("No changes are recommended in the current allocation.")
                except Exception as e:
                    st.error(f"Error generating actionable to-do list: {e}")
                    st.info("Try reloading the page or contact support if the issue persists.")
            else:
                st.warning("To-do list generation requires the Optimization Insights module. Please ensure it's properly installed.")
            
            # # Performance impact summary
            # with st.expander("Performance Impact Summary"):
            #     # Calculate performance improvements
            #     # We need to make sure we're joining on the same indices for the dot product
                
            #     # Now merge with performance metrics
            #     # Get unique vendor-material combinations with their metrics
            #     unique_metrics = df_filtered[['LIFNR', 'MATNR', 'MedianLeadTimeDays', 'Avg_OTIF_Rate']].drop_duplicates(subset=['LIFNR', 'MATNR'])
            #     comp_with_metrics = pd.merge(
            #         filtered_comparison_df,
            #         unique_metrics,
            #         on=['LIFNR', 'MATNR'],
            #         how='inner'
            #     )
                
            #     # Now we can safely calculate weighted averages
            #     if not comp_with_metrics.empty and comp_with_metrics['Historical_Allocated_Quantity'].sum() > 0:
            #         historical_lead_time = (comp_with_metrics['Historical_Allocated_Quantity'] * comp_with_metrics['MedianLeadTimeDays']).sum() / comp_with_metrics['Historical_Allocated_Quantity'].sum()
            #         historical_otif = (comp_with_metrics['Historical_Allocated_Quantity'] * comp_with_metrics['Avg_OTIF_Rate']).sum() / comp_with_metrics['Historical_Allocated_Quantity'].sum()
                    
            #         if comp_with_metrics['Optimized_Allocated_Quantity'].sum() > 0:
            #             optimized_lead_time = (comp_with_metrics['Optimized_Allocated_Quantity'] * comp_with_metrics['MedianLeadTimeDays']).sum() / comp_with_metrics['Optimized_Allocated_Quantity'].sum()
            #             optimized_otif = (comp_with_metrics['Optimized_Allocated_Quantity'] * comp_with_metrics['Avg_OTIF_Rate']).sum() / comp_with_metrics['Optimized_Allocated_Quantity'].sum()
            #         else:
            #             optimized_lead_time = historical_lead_time
            #             optimized_otif = historical_otif
            #     else:
            #         historical_lead_time = df_filtered['MedianLeadTimeDays'].mean()
            #         historical_otif = df_filtered['Avg_OTIF_Rate'].mean()
            #         optimized_lead_time = historical_lead_time
            #         optimized_otif = historical_otif
                
            #     # Use the filtered comparison data for cost calculation
            #     historical_total_cost = filtered_comparison_df['Historical_Total_Effective_Cost_for_Combo'].sum()
            #     optimized_total_cost = filtered_comparison_df['Optimized_Total_Effective_Cost_for_Combo'].sum()
                
            #     # These variables will be accessible for insights generation
            #     global_historical_total = historical_total_cost
            #     global_optimized_total = optimized_total_cost
                
            #     performance_changes = pd.DataFrame({
            #         'Metric': ['Lead Time', 'OTIF Rate', 'Total Cost'],
            #         'Historical': [
            #             historical_lead_time,
            #             historical_otif,
            #             historical_total_cost
            #         ],
            #         'Optimized': [
            #             optimized_lead_time,
            #             optimized_otif,
            #             optimized_total_cost
            #         ]
            #     })
                
            #     performance_changes['Change'] = performance_changes['Optimized'] - performance_changes['Historical']
            #     performance_changes['Percent Change'] = (performance_changes['Change'] / performance_changes['Historical']) * 100
                
            #     # Format for display
            #     performance_changes_display = performance_changes.copy()
            #     performance_changes_display['Historical'] = performance_changes_display.apply(
            #         lambda row: f"{row['Historical']:.1f} days" if row['Metric'] == 'Lead Time' 
            #         else f"{row['Historical']:.1%}" if row['Metric'] == 'OTIF Rate'
            #         else f"${row['Historical']:,.2f}", axis=1
            #     )
                
            #     performance_changes_display['Optimized'] = performance_changes_display.apply(
            #         lambda row: f"{row['Optimized']:.1f} days" if row['Metric'] == 'Lead Time' 
            #         else f"{row['Optimized']:.1%}" if row['Metric'] == 'OTIF Rate'
            #         else f"${row['Optimized']:,.2f}", axis=1
            #     )
                
            #     performance_changes_display['Change'] = performance_changes_display.apply(
            #         lambda row: f"{row['Change']:.1f} days" if row['Metric'] == 'Lead Time' 
            #         else f"{row['Change']:.1%}" if row['Metric'] == 'OTIF Rate'
            #         else f"${row['Change']:,.2f}", axis=1
            #     )
                
            #     performance_changes_display['Percent Change'] = performance_changes_display['Percent Change'].apply(
            #         lambda x: f"{x:.1f}%"
            #     )
                
            #     st.table(performance_changes_display)
                
            #     # Add a button to generate insights on demand
            #     col1, col2 = st.columns([2, 1])
            #     with col1:
            #         st.markdown("### AI-Powered Optimization Insights")
            #         if INSIGHTS_ENABLED:
            #             st.info("Click the button to generate AI-powered insights (this may take a few moments)")
            #         else:
            #             st.warning("AI insights module not available. Basic insights will be shown instead.")
                
            #     with col2:
            #         if INSIGHTS_ENABLED:
            #             generate_button = st.button("Generate AI Insights", type="primary")
            #         else:
            #             generate_button = st.button("Show Basic Insights", type="primary")
                
            #     # Only generate insights when the button is clicked
            #     if generate_button:
            #         if INSIGHTS_ENABLED:
            #             # Prepare data for insights generation
            #             with st.spinner("Generating AI-powered optimization insights..."):
            #                 try:
            #                     # Extract cost component data from the filtered comparison data
            #                     if waterfall_fig:
            #                         # Get cost component changes from filtered comparison data
            #                         cost_component_data = {}
            #                         for col in [c for c in filtered_comparison_df.columns if c.startswith('Delta_Total_cost_')]:
            #                             component = col.replace('Delta_Total_', '')
            #                             # Only include cost components that are active in costs_config
            #                             if costs_config and component in costs_config:
            #                                 if costs_config[component] == "True":
            #                                     cost_component_data[component] = filtered_comparison_df[col].sum()
            #                             else:
            #                                 # If no costs_config, include all components
            #                                 cost_component_data[component] = filtered_comparison_df[col].sum()
            #                     else:
            #                         cost_component_data = {}
                                    
            #                     # Convert performance changes to the format needed for insights
            #                     perf_data = {}
            #                     for i, row in performance_changes.iterrows():
            #                         metric = row['Metric']
            #                         perf_data[metric.lower().replace(" ", "_")] = {
            #                             "historical": float(row['Historical']),
            #                             "optimized": float(row['Optimized']),
            #                             "change": float(row['Change']),
            #                             "percent_change": float(row['Percent Change'])
            #                         }
                                    
            #                     # Generate insights
            #                     optimization_data = prepare_optimization_data_for_insights(
            #                         metrics, 
            #                         performance_changes, 
            #                         cost_component_data
            #                     )
                                
            #                     # Pass the costs_config to filter active cost components
            #                     insights_data = generate_optimization_insights(
            #                         *optimization_data,
            #                         costs_config=costs_config
            #                     )
                                
            #                     # Display insights
            #                     st.markdown("#### AI-Generated Key Insights")
            #                     for insight in insights_data["insights"]:
            #                         st.markdown(f"- {insight}")
                                
            #                     # Display recommendations
            #                     st.markdown("#### AI-Generated Recommendations")
            #                     for recommendation in insights_data["recommendations"]:
            #                         st.markdown(f"- {recommendation}")
                                    
            #                 except Exception as e:
            #                     st.error(f"Error generating dynamic insights: {e}")
            #                     # Fall back to static insights
            #                     show_static_insights(performance_changes)
            #         else:
            #             # Use static insights if dynamic insights are not available
            #             st.info("Showing basic analysis insights (AI module not available)")
            #             show_static_insights(performance_changes)
        
        else:
            st.warning("Optimization data not available. Please ensure the required files exist in the tables directory.")
            st.info("Required files: comparison.csv and optimized_allocation_maktx_vendor_maktx_ranking_tariff_values.csv")


def render_optimized_selection_tab_standalone(df_filtered, filters=None):
    """
    Render the optimized vendor comparison tab without tab context (for lazy loading)
    
    Args:
        df_filtered (pd.DataFrame): Filtered vendor data
        filters (dict, optional): Filter settings from the UI
    """
    # Simply call the existing function with a dummy tab context
    class DummyTab:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    
    render_optimized_selection_tab(DummyTab(), df_filtered, filters)