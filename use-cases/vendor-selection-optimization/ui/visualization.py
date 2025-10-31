"""
Visualization components for the Vendor Performance Dashboard.
Contains functions for creating charts and visual elements.
"""

import pandas as pd
import plotly.express as px
import streamlit as st
from config import settings
from core import utils
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_scatter_plot(df_filtered):
    """Create a scatter plot for lead time vs OTIF performance with memory optimization"""
    # Memory optimization: Limit data points for large datasets
    max_scatter_points = 2000  # Configurable limit for performance
    data_size = len(df_filtered)
    
    if data_size > max_scatter_points:
        # Sample data strategically: keep outliers and representative points
        # Take top and bottom performers plus random sample
        top_performers = df_filtered.nlargest(max_scatter_points // 4, 'Avg_OTIF_Rate')
        bottom_performers = df_filtered.nsmallest(max_scatter_points // 4, 'Avg_OTIF_Rate')
        remaining_data = df_filtered.drop(top_performers.index.union(bottom_performers.index))
        
        if len(remaining_data) > 0:
            random_sample = remaining_data.sample(n=min(max_scatter_points // 2, len(remaining_data)), random_state=42)
            plot_data = pd.concat([top_performers, bottom_performers, random_sample])
        else:
            plot_data = pd.concat([top_performers, bottom_performers])
    else:
        plot_data = df_filtered
    
    # Optimize hover data - only include essential columns that exist
    hover_data_dict = {}
    hover_columns = {
        'MedianLeadTimeDays': ':.1f',
        'Avg_OTIF_Rate': ':.2%',
        'OnTimeRate': ':.2%',
        'InFullRate': ':.2%',
        'POLineItemCount': True,
        'MAKTX': True,
        'EffectiveCostPerUnit_USD': ':.2f',
        'Country': True
    }
    
    # Only include hover data for columns that exist
    for col, format_spec in hover_columns.items():
        if col in plot_data.columns:
            hover_data_dict[col] = format_spec
    
    # Scatter plot with optimized data
    fig_scatter = px.scatter(
        plot_data, 
        x='MedianLeadTimeDays', 
        y='Avg_OTIF_Rate',
        color='MAKTX',
        size='POLineItemCount' if 'POLineItemCount' in plot_data.columns else None,
        hover_name='VendorFullID' if 'VendorFullID' in plot_data.columns else 'Supplier_Name',
        hover_data=hover_data_dict,
        title=f"Vendor Performance: Lead Time vs OTIF Rate ({len(plot_data):,} vendors)",
        labels={
            'MedianLeadTimeDays': 'Median Lead Time (Days)',
            'Avg_OTIF_Rate': 'OTIF Rate',
            'POLineItemCount': 'PO Line Items',
            'MAKTX': 'Material'
        },
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    # Add reference lines if we have enough data
    if len(df_filtered) > 1:
        avg_lt = df_filtered['MedianLeadTimeDays'].median()
        avg_otif = df_filtered['Avg_OTIF_Rate'].median()
        
        fig_scatter.add_hline(
            y=avg_otif, 
            line_dash="dot", 
            line_color="red",
            annotation_text=f"Median OTIF ({avg_otif:.1%})",
            annotation_position="bottom right"
        )
        fig_scatter.add_vline(
            x=avg_lt, 
            line_dash="dot", 
            line_color="red",
            annotation_text=f"Median Lead Time ({avg_lt:.1f}d)",
            annotation_position="top left"
        )
    
    fig_scatter.update_layout(
        yaxis_tickformat=".0%",
        height=settings.CHART_HEIGHT_LARGE,
        showlegend=True
    )
    
    return fig_scatter


def create_performance_heatmap(df_filtered):
    """Create a performance heatmap for multiple metrics"""
    if len(df_filtered) <= 1:
        return None
    
    # Make sure we have the required columns
    required_columns = ['LIFNR', 'NAME1', 'LAND1', 'Country', 'MedianLeadTimeDays', 'OnTimeRate', 
                        'InFullRate', 'Avg_OTIF_Rate', 'EffectiveCostPerUnit_USD', 'TariffImpact_raw_percent']
    
    if not all(col in df_filtered.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df_filtered.columns]
        print(f"Missing columns for heatmap: {missing_cols}")
        
        # If we're missing the country code column but have Country, try to use that
        if 'LAND1' not in df_filtered.columns and 'Country' in df_filtered.columns:
            df_filtered['LAND1'] = df_filtered['Country'].apply(lambda x: str(x)[:2].upper())
    
    # Use existing VendorFullID which includes LIFNR codes (format: NAME1-LIFNR (LAND1))
    # This provides consistency with other tabs and better vendor identification
    if 'VendorFullID' not in df_filtered.columns:
        print("Warning: VendorFullID column not found. Creating fallback identifier.")
        # Fallback: create VendorFullID if not available
        df_filtered['VendorFullID'] = df_filtered['NAME1'] + '-' + df_filtered['LIFNR'].astype(str) + ' (' + df_filtered.get('LAND1', 'XX').astype(str) + ')'
    
    # Get data for the heatmap using VendorFullID as the index
    heatmap_data = df_filtered.set_index('VendorFullID')[
        ['MedianLeadTimeDays', 'OnTimeRate', 'InFullRate', 'Avg_OTIF_Rate', 
         'EffectiveCostPerUnit_USD', 'cost_Logistics', 'TariffImpact_raw_percent']
    ].copy()
    
    # Ensure no duplicate indices by keeping the first occurrence
    if heatmap_data.index.duplicated().any():
        print(f"Warning: Found {heatmap_data.index.duplicated().sum()} duplicate vendor IDs in heatmap data. Keeping first occurrence.")
        heatmap_data = heatmap_data[~heatmap_data.index.duplicated(keep='first')]
    
    # Check if we have all the required columns after filtering
    if 'EffectiveCostPerUnit_USD' not in heatmap_data.columns:
        print("Missing base price column 'EffectiveCostPerUnit_USD'")
        # Add a dummy column with 0s
        heatmap_data['EffectiveCostPerUnit_USD'] = 0
    
    if 'cost_Logistics' not in heatmap_data.columns:
        print("Missing logistics cost column 'cost_Logistics'")
        # Add a dummy column with 0s
        heatmap_data['cost_Logistics'] = 0
    
    if 'TariffImpact_raw_percent' not in heatmap_data.columns:
        print("Missing tariff impact column 'TariffImpact_raw_percent'")
        # Add a dummy column with 0s
        heatmap_data['TariffImpact_raw_percent'] = 0
    
    # Normalize metrics
    heatmap_normalized = pd.DataFrame({
        'Lead Time': utils.normalize_min_max(heatmap_data['MedianLeadTimeDays'], lower_is_better=True),
        'On-Time Rate': utils.normalize_min_max(heatmap_data['OnTimeRate']),
        'In-Full Rate': utils.normalize_min_max(heatmap_data['InFullRate']),
        'OTIF Rate': utils.normalize_min_max(heatmap_data['Avg_OTIF_Rate']),
        'Base Price': utils.normalize_min_max(heatmap_data['EffectiveCostPerUnit_USD'], lower_is_better=True),
        'Logistics Cost': utils.normalize_min_max(heatmap_data['cost_Logistics'], lower_is_better=True),
        'Tariff Impact': utils.normalize_min_max(heatmap_data['TariffImpact_raw_percent'], lower_is_better=True)
    })
    
    # Sort by overall performance
    heatmap_normalized['Overall'] = heatmap_normalized.mean(axis=1)
    heatmap_normalized = heatmap_normalized.sort_values('Overall', ascending=False)
    
    # Create a mapping of display formats based on metric type
    display_formats = {
        'Lead Time': lambda x: f"{x:.1f} days",
        'On-Time Rate': lambda x: f"{x:.1%}",
        'In-Full Rate': lambda x: f"{x:.1%}",
        'OTIF Rate': lambda x: f"{x:.1%}",
        'Base Price': lambda x: f"${x:.2f}",
        'Logistics Cost': lambda x: f"${x:.2f}",
        'Tariff Impact': lambda x: f"{x:.2f}%"
    }
    
    # Create a dataframe with raw values but in the same order as normalized scores
    raw_data_for_display = pd.DataFrame(index=heatmap_normalized.index)
    
    # Map column names from normalized to raw data
    column_mapping = {
        'Lead Time': 'MedianLeadTimeDays',
        'On-Time Rate': 'OnTimeRate',
        'In-Full Rate': 'InFullRate',
        'OTIF Rate': 'Avg_OTIF_Rate',
        'Base Price': 'EffectiveCostPerUnit_USD',
        'Logistics Cost': 'cost_Logistics',
        'Tariff Impact': 'TariffImpact_raw_percent'
    }
    
    # Fill the raw data dataframe with values from the original data
    # Handle potential duplicate indices by using .loc with the first match
    for norm_col, raw_col in column_mapping.items():
        if raw_col in heatmap_data.columns:
            # Safely extract values handling potential duplicates
            raw_values = []
            for idx in raw_data_for_display.index:
                try:
                    # Get the value(s) for this index
                    value = heatmap_data.loc[idx, raw_col]
                    # If multiple matches, take the first one
                    if hasattr(value, '__iter__') and not isinstance(value, str):
                        value = value.iloc[0] if hasattr(value, 'iloc') else value[0]
                    raw_values.append(value)
                except (KeyError, IndexError):
                    # Handle missing indices gracefully
                    raw_values.append(0.0)
            raw_data_for_display[norm_col] = raw_values
        else:
            # If column doesn't exist, fill with default values
            raw_data_for_display[norm_col] = 0.0
    
    # Get the normalized scores for coloring
    z_values = heatmap_normalized.drop('Overall', axis=1).values
    
    # Get raw values for text display
    text_values = [[display_formats[col](val) for col, val in row.items()] 
                    for idx, row in raw_data_for_display.iterrows()]
    
    # Create the heatmap
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=z_values,
        x=list(raw_data_for_display.columns),
        y=list(raw_data_for_display.index),
        text=text_values,
        texttemplate="%{text}",
        colorscale=settings.COLOR_SCALE,
        showscale=True,
        colorbar=dict(title="Normalized Score<br>(0=worst, 1=best)")
    ))
    
    fig_heatmap.update_layout(
        title="Vendor Plant Performance Heatmap",
        xaxis_title="Performance Metric",
        yaxis_title="Vendor-Plant",
        xaxis_side="top"
    )
    
    fig_heatmap.update_layout(
        height=max(400, len(raw_data_for_display) * 40),
        xaxis_side="top"
    )
    
    return fig_heatmap, heatmap_normalized


def create_country_choropleth(country_agg, metric, title, color_scale, labels):
    """Create a choropleth map for country performance"""
    fig_map = px.choropleth(
        country_agg,
        locations='Country',
        locationmode='country names',
        color=metric,
        hover_name='Country',
        hover_data={
            'Avg_Lead_Time': ':.1f',
            'Avg_OTIF_Rate': ':.2%',
            'Avg_Tariff_Impact': ':.2f',
            'Avg_Logistics_Cost': ':.2f',
            'Vendor_Count': True
        },
        color_continuous_scale=color_scale,
        title=title,
        labels=labels
    )
    fig_map.update_layout(height=settings.CHART_HEIGHT, margin={"r":0,"t":50,"l":0,"b":0})
    return fig_map


def create_material_bar_chart(material_agg, x, y, title, labels, color, orientation='h'):
    """Create a bar chart for material analysis"""
    if orientation == 'h':
        # Horizontal bar chart (good for long material names)
        fig = px.bar(
            material_agg.sort_values(x).head(10),
            x=x,
            y=y,
            orientation='h',
            title=title,
            labels=labels,
            color=x,
            color_continuous_scale=color
        )
    else:
        # Vertical bar chart
        fig = px.bar(
            material_agg.sort_values(x).head(10),
            x=y,
            y=x,
            title=title,
            labels=labels,
            color=x,
            color_continuous_scale=color
        )
    
    fig.update_layout(height=400, showlegend=False)
    return fig


def create_cost_breakdown_chart(material_df, active_cost_columns):
    """Create a cost component breakdown chart with memory optimization"""
    if not active_cost_columns or not any(material_df[col].abs().sum() > 0 for col in active_cost_columns):
        return None
    
    # Memory optimization: Only work with required columns
    required_cols = ['VendorFullID'] + [col for col in active_cost_columns if col in material_df.columns]
    if len(required_cols) <= 1:  # Only VendorFullID
        return None
    
    # Create a consolidated cost breakdown chart with minimal memory footprint
    plot_df = material_df[required_cols].copy()
    plot_df = plot_df.set_index('VendorFullID')
    
    # Remove vendors with all zero costs (check absolute values to include negative components)
    cost_cols = [col for col in active_cost_columns if col in plot_df.columns]
    if not cost_cols:
        return None
    
    non_zero_mask = plot_df[cost_cols].abs().sum(axis=1) > 0
    plot_df = plot_df[non_zero_mask]
    
    if plot_df.empty:
        return None
    
    # Memory optimization: Limit to top 10 vendors for performance
    max_vendors = min(10, len(plot_df))
    
    # Calculate total cost for ranking, but limit processing
    total_costs = plot_df[cost_cols].sum(axis=1)
    top_vendor_indices = total_costs.abs().nsmallest(max_vendors).index
    top_vendors = plot_df.loc[top_vendor_indices]
    
    # Define color mapping for each cost component
    component_colors = {
        'cost_BasePrice': '#1f77b4',  # Blue
        'cost_Tariff': '#ff7f0e',  # Orange
        'cost_Logistics': '#17becf',  # Cyan
        'cost_Holding_LeadTime': '#2ca02c',  # Green
        'cost_Holding_LTVariability': '#d62728',  # Red
        'cost_Holding_Lateness': '#9467bd',  # Purple
        'cost_Inefficiency_InFull': '#8c564b',  # Brown
        'cost_Risk_PriceVolatility': '#e377c2',  # Pink
        'cost_Impact_PriceTrend': '#7f7f7f',  # Gray
    }
    
    # Clean names for display
    component_display_names = {
        'cost_BasePrice': 'Base Price',
        'cost_Tariff': 'Tariff',
        'cost_Logistics': 'Logistics',
        'cost_Holding_LeadTime': 'Holding (Lead Time)',
        'cost_Holding_LTVariability': 'Holding (LT Variability)',
        'cost_Holding_Lateness': 'Holding (Lateness)',
        'cost_Inefficiency_InFull': 'Inefficiency (In-Full)',
        'cost_Risk_PriceVolatility': 'Risk (Price Volatility)',
        'cost_Impact_PriceTrend': 'Impact (Price Trend)',
    }
    
    # Check if we have any negative values
    has_negative_values = any((top_vendors[col] < 0).any() for col in active_cost_columns)
    
    if has_negative_values:
        # Use grouped bar chart with positive and negative bars
        fig_breakdown = go.Figure()
        
        # Process each cost component
        for col in active_cost_columns:
            if col not in top_vendors.columns:
                continue
                
            display_name = component_display_names.get(col, col.replace('cost_', '').replace('_', ' ').title())
            base_color = component_colors.get(col, '#17becf')
            
            # Get values for this component across all vendors
            values = top_vendors[col].values
            vendors = list(top_vendors.index)
            
            # Separate positive and negative values
            positive_values = [v if v > 0 else 0 for v in values]
            negative_values = [v if v < 0 else 0 for v in values]
            
            # Add positive bars if any
            if any(positive_values):
                fig_breakdown.add_trace(
                    go.Bar(
                        name=f"{display_name} (+)",
                        x=vendors,
                        y=positive_values,
                        marker_color=base_color,
                        marker_pattern_shape="" if not any(negative_values) else "",
                        legendgroup=display_name,
                        hovertemplate='<b>%{x}</b><br>' + display_name + ' (Cost): $%{y:.2f}<extra></extra>'
                    )
                )
            
            # Add negative bars if any
            if any(negative_values):
                # Use lighter/different shade for negative values
                fig_breakdown.add_trace(
                    go.Bar(
                        name=f"{display_name} (-)",
                        x=vendors,
                        y=negative_values,
                        marker_color=base_color,
                        marker_opacity=0.5,
                        marker_pattern_shape="/",  # Add pattern to distinguish
                        legendgroup=display_name,
                        hovertemplate='<b>%{x}</b><br>' + display_name + ' (Savings): $%{y:.2f}<extra></extra>'
                    )
                )
        
        # Add a horizontal line at y=0
        fig_breakdown.add_hline(y=0, line_color='black', line_width=1, line_dash='solid')
        
        # Add total effective cost as scatter points
        total_costs = [top_vendors.loc[vendor, active_cost_columns].sum() 
                      for vendor in top_vendors.index]
        fig_breakdown.add_trace(
            go.Scatter(
                name='Total Effective Cost',
                x=list(top_vendors.index),
                y=total_costs,
                mode='markers+text',
                marker=dict(size=10, color='black', symbol='diamond'),
                text=[f"${cost:.2f}" for cost in total_costs],
                textposition='top center',
                showlegend=True,
                hovertemplate="<b>%{x}</b><br>Total: $%{y:.2f}<extra></extra>"
            )
        )
        
        fig_breakdown.update_layout(
            title="Cost Component Breakdown by Vendor",
            xaxis_title="Vendor",
            yaxis_title="Cost (USD)",
            xaxis_tickangle=-45,
            height=settings.CHART_HEIGHT + 100,
            barmode='stack',
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                font=dict(size=10)
            ),
            hovermode='x unified'
        )
        
        # # Add annotation explaining negative values
        # fig_breakdown.add_annotation(
        #     text="Hatched bars indicate cost savings (negative values)",
        #     xref="paper", yref="paper",
        #     x=1.02, y=1,
        #     showarrow=False,
        #     font=dict(size=10, color="gray"),
        #     xanchor="left",
        #     yanchor="top"
        # )
    else:
        # Use standard stacked bar chart if no negative values
        # Prepare data for stacked bar chart with custom colors
        fig_breakdown = go.Figure()
        
        for col in active_cost_columns:
            if col not in top_vendors.columns:
                continue
                
            display_name = component_display_names.get(col, col.replace('cost_', '').replace('_', ' ').title())
            color = component_colors.get(col, '#17becf')
            
            fig_breakdown.add_trace(
                go.Bar(
                    name=display_name,
                    x=list(top_vendors.index),
                    y=top_vendors[col],
                    marker_color=color,
                    hovertemplate='<b>%{x}</b><br>' + display_name + ': $%{y:.2f}<extra></extra>'
                )
            )
        
        fig_breakdown.update_layout(
            title="Cost Component Breakdown by Vendor (Active Components)",
            xaxis_title="Vendor",
            yaxis_title="Cost (USD)",
            xaxis_tickangle=-45,
            height=settings.CHART_HEIGHT,
            barmode='stack',
            showlegend=True,
            hovermode='x unified'
        )
    
    return fig_breakdown


def create_cost_comparison_chart(material_df, material):
    """Create a cost comparison chart by vendor"""
    if 'EffectiveCostPerUnit_USD' not in material_df.columns:
        return None
    
    fig_cost = px.bar(
        material_df.sort_values('EffectiveCostPerUnit_USD').head(10),
        x='VendorFullID',
        y='EffectiveCostPerUnit_USD',
        title=f"Effective Cost Per Unit Comparison - {material}",
        labels={'EffectiveCostPerUnit_USD': 'Cost (USD)', 'VendorFullID': 'Vendor'},
        color='EffectiveCostPerUnit_USD',
        color_continuous_scale=settings.COLOR_SCALE_REVERSED
    )
    fig_cost.update_layout(xaxis_tickangle=-45, height=settings.CHART_HEIGHT)
    
    return fig_cost


def create_performance_scatter(material_df, material):
    """Create a performance scatter plot for vendor analysis"""
    if not all(col in material_df.columns for col in ['AvgLeadTimeDays_raw', 'OnTimeRate_raw', 'EffectiveCostPerUnit_USD']):
        return None
    
    fig_scatter = px.scatter(
        material_df,
        x='AvgLeadTimeDays_raw',
        y='OnTimeRate_raw',
        size='EffectiveCostPerUnit_USD',
        hover_name='VendorFullID',
        title=f"Lead Time vs On-Time Rate - {material}",
        labels={
            'AvgLeadTimeDays_raw': 'Lead Time (Days)',
            'OnTimeRate_raw': 'On-Time Rate',
            'EffectiveCostPerUnit_USD': 'Effective Cost/Unit'
        }
    )
    fig_scatter.update_layout(height=settings.CHART_HEIGHT)
    
    return fig_scatter