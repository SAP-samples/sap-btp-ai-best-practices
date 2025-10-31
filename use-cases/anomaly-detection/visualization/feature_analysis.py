"""
Feature analysis visualization module.

This module contains plotting functions for analyzing order features
without Streamlit dependencies to avoid import conflicts.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, percentileofscore
import tempfile
import os
from typing import List
import config.settings as settings
from utils.formatters import format_feature_name


def create_feature_plots(row, features_df, save_for_ai_analysis=False, compute_shap_on_demand=False, pre_computed_shap=None):
    """
    Create visualizations for feature analysis
    
    Args:
        row: Order data row
        features_df: Historical features data
        save_for_ai_analysis: If True, save plots as temporary images for AI analysis
        compute_shap_on_demand: If True, compute SHAP values on demand for plot 4
        pre_computed_shap: Pre-computed SHAP text to use instead of computing on-demand
        
    Returns:
        fig: Matplotlib figure object
        image_paths: List of saved image paths (if save_for_ai_analysis=True), else just fig
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Feature Analysis for SO: {row.get("Customer PO number", "N/A")}', fontsize=16)
    
    # Plot 1: Quantity Analysis
    ax1 = axes[0, 0]
    current_qty = row.get('Sales Order item qty', np.nan)
    current_material_number = row.get('Material Number')
    current_sold_to_number = row.get('Sold To number')
    
    # Debug information to help diagnose missing data
    historical_df = features_df if features_df is not None else pd.DataFrame()
    num_rows = len(historical_df) if not historical_df.empty else 0
    unique_materials = historical_df['Material Number'].nunique() if 'Material Number' in historical_df else 0
    print(f"[DEBUG] Plot 1 - Current qty: {current_qty}, Material: {current_material_number}, Sold-To: {current_sold_to_number}, Historical rows: {num_rows}, Unique materials: {unique_materials}")

    if (
        not pd.isna(current_qty)
        and current_material_number is not None
        and current_sold_to_number is not None
        and not historical_df.empty
        and {'Material Number', 'Sold To number', 'Sales Order item qty'} <= set(historical_df.columns)
    ):
        # Filter history to the specific customer-material pair so we mirror row-level statistics.
        customer_material_data = historical_df[
            (historical_df['Material Number'] == current_material_number)
            & (historical_df['Sold To number'] == current_sold_to_number)
        ]
        historical_quantities = customer_material_data['Sales Order item qty'].dropna().values
        
        if historical_quantities.size == 0:
            ax1.text(0.5, 0.5, 'No historical quantity data\nfor this customer-material.', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Order Quantity Distribution')
        elif historical_quantities.size < 5 or len(np.unique(historical_quantities)) == 1:
            # Fallback plots for sparse data
            if len(np.unique(historical_quantities)) == 1:
                fixed_value = historical_quantities[0]
                ax1.bar([fixed_value], [1], width=max(1, fixed_value * 0.1), color='skyblue', alpha=0.7, label=f'Historical: {fixed_value:.0f} (Fixed)')
                ax1.set_xlim(max(0, fixed_value - max(1, fixed_value*0.2)), fixed_value + max(1, fixed_value*0.2))
                ax1.set_ylim(0, 1.2)
                title_suffix = f"(Fixed Historical Value: {fixed_value:.0f})"
            else:
                y_strip = np.zeros(historical_quantities.size)
                ax1.scatter(historical_quantities, y_strip, color='skyblue', alpha=0.7, label='Historical Qtys')
                ax1.set_yticks([])
                title_suffix = f"(Based on {historical_quantities.size} historical orders)"

            ax1.axvline(current_qty, color='r', linestyle='--', linewidth=2, label=f'Actual Qty: {current_qty:.0f}')
            ax1.set_title(f'Order Quantity Distribution\n{title_suffix}')
            ax1.set_xlabel('Order Quantity')
            ax1.legend()
        else:
            try:
                kde = gaussian_kde(historical_quantities)
                x_min_plot = 0
                x_max_plot = max(historical_quantities.max(), current_qty) * 1.2
                if x_max_plot == 0:
                    x_max_plot = current_qty * 1.2 if current_qty > 0 else 10
                if x_min_plot == x_max_plot:
                    x_max_plot = x_min_plot + 1

                x_kde_axis = np.linspace(x_min_plot, x_max_plot, 200)
                y_kde_values = kde(x_kde_axis)

                ax1.plot(x_kde_axis, y_kde_values, 'b-', label='Historical Distribution')
                ax1.fill_between(x_kde_axis, y_kde_values, alpha=0.3, color='blue')
                ax1.axvline(current_qty, color='r', linestyle='--', linewidth=2, label=f'Actual Qty: {current_qty:.0f}')
                
                median_historical_qty = np.median(historical_quantities)
                ax1.axvline(median_historical_qty, color='g', linestyle='-', linewidth=1.5, label=f'Median Hist. Qty: {median_historical_qty:.0f}')
                
                mean_historical_qty = np.mean(historical_quantities)
                ax1.axvline(mean_historical_qty, color='orange', linestyle=':', linewidth=1.5, label=f'Mean Hist. Qty: {mean_historical_qty:.0f}')
                
                percentile_val = percentileofscore(historical_quantities, current_qty, kind='strict')
                ax1.set_title(f'Order Quantity Distribution\nActual Qty: {current_qty:.0f} ({percentile_val:.0f}th percentile)')
                ax1.set_xlabel('Order Quantity')
                ax1.set_ylabel('Estimated Density')
                ax1.legend(fontsize=10)
                ax1.set_ylim(bottom=0)
                ax1.set_xlim(left=max(0, x_kde_axis.min()))
            except Exception as e:
                ax1.text(0.5, 0.5, f'Could not generate plot:\n{str(e)[:50]}...', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Order Quantity Distribution')
    else:
        ax1.text(0.5, 0.5, 'Quantity data not available.', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Order Quantity Distribution')

    # Plot 2: Price Analysis (Violin Plot)
    ax2 = axes[0, 1]
    current_price = row.get('Unit Price', np.nan)
    current_material_number = row.get('Material Number')

    historical_prices_df = historical_df if historical_df is not None else pd.DataFrame()
    print(f"[DEBUG] Plot 2 - Current price: {current_price}, Material: {current_material_number}, Sold-To: {current_sold_to_number}, Historical rows: {len(historical_prices_df)}")

    if (
        not pd.isna(current_price)
        and current_material_number is not None
        and current_sold_to_number is not None
        and not historical_prices_df.empty
        and {'Material Number', 'Sold To number', 'Unit Price'} <= set(historical_prices_df.columns)
    ):
        # Match pricing history to the same customer-material combination for consistent statistics.
        customer_material_price_data = historical_prices_df[
            (historical_prices_df['Material Number'] == current_material_number)
            & (historical_prices_df['Sold To number'] == current_sold_to_number)
        ]
        historical_prices = customer_material_price_data['Unit Price'].dropna().values

        if historical_prices.size == 0:
            ax2.text(0.5, 0.5, 'No historical price data\nfor this customer-material.', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Unit Price Distribution')
        elif historical_prices.size < 3 or len(np.unique(historical_prices)) == 1:
            if len(np.unique(historical_prices)) == 1:
                fixed_value = historical_prices[0]
                ax2.bar([0], [fixed_value], width=0.4, color='skyblue', alpha=0.7, label=f'Historical: ${fixed_value:.2f} (Fixed)')
                ax2.set_xlim(-0.5, 0.5)
                ax2.set_ylim(max(0, fixed_value - max(1, fixed_value*0.2)), fixed_value + max(1, fixed_value*0.2))
                title_suffix = f"(Fixed Historical Price: ${fixed_value:.2f})"
            else:
                x_strip = np.zeros(historical_prices.size)
                ax2.scatter(x_strip, historical_prices, color='skyblue', alpha=0.7, label='Historical Prices')
                ax2.set_xlim(-0.5, 0.5)
                ax2.set_xticks([])
                title_suffix = f"(Based on {historical_prices.size} historical orders)"

            ax2.axhline(current_price, color='r', linestyle='--', linewidth=2, label=f'Current: ${current_price:.2f}')
            ax2.set_title(f'Unit Price Distribution\n{title_suffix}')
            ax2.set_ylabel('Unit Price ($)')
            ax2.legend(fontsize=10)
        else:
            try:
                violin_parts = ax2.violinplot([historical_prices], positions=[0], widths=0.6, showmeans=True, showmedians=True)
                for pc in violin_parts['bodies']:
                    pc.set_facecolor('lightblue')
                    pc.set_alpha(0.7)
                    pc.set_edgecolor('blue')
                violin_parts['cmeans'].set_color('green')
                violin_parts['cmeans'].set_linewidth(2)
                violin_parts['cmeans'].set_label('Mean')
                violin_parts['cmedians'].set_color('orange')
                violin_parts['cmedians'].set_linewidth(2)
                violin_parts['cmedians'].set_label('Median')

                ax2.axhline(current_price, color='red', linestyle='--', linewidth=3, label=f'Current: ${current_price:.2f}')
                percentile_val = percentileofscore(historical_prices, current_price, kind='strict')
                ax2.set_title(f'Unit Price Distribution (Violin Plot)\nCurrent: ${current_price:.2f} ({percentile_val:.0f}th percentile)')
                ax2.set_ylabel('Unit Price ($)')
                ax2.set_xlim(-0.8, 0.8)
                ax2.set_xticks([])
                ax2.legend()

                price_min = min(historical_prices.min(), current_price)
                price_max = max(historical_prices.max(), current_price)
                price_range = price_max - price_min
                ax2.set_ylim(max(0, price_min - price_range*0.1), price_max + price_range*0.1)

            except Exception as e:
                ax2.text(0.5, 0.5, f'Could not generate plot:\n{str(e)[:50]}...', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Unit Price Distribution')
    else:
        ax2.text(0.5, 0.5, 'Price data not available.', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Unit Price Distribution')
    
    # Plot 3: Monthly Volume Trend with Current Order Contribution
    ax3 = axes[1, 0]
    if not pd.isna(row.get('current_month_total_qty', np.nan)):
        current_month_total = row['current_month_total_qty']
        current_order_qty = row.get('Sales Order item qty', 0)
        p05 = row.get('monthly_qty_p05', current_month_total * 0.8)
        p95 = row.get('monthly_qty_p95', current_month_total * 1.2)
        
        # Calculate other orders quantity (total - current order)
        other_orders_qty = max(0, current_month_total - current_order_qty)
        
        months = ['Previous Range', 'Current Month']
        
        # First bar: Previous Range (with error bars)
        ax3.bar([months[0]], [(p05 + p95) / 2], yerr=[(p95 - p05) / 2], 
               capsize=10, color='lightblue', label='Previous Range')
        
        # Second bar: Current Month as stacked bar
        # Bottom part: Other orders in current month
        if other_orders_qty > 0:
            ax3.bar([months[1]], [other_orders_qty], color='lightcoral', 
                   label=f'Other Orders: {other_orders_qty:.0f}')
        
        # Top part: Current order (stacked on top)
        if current_order_qty > 0:
            ax3.bar([months[1]], [current_order_qty], bottom=[other_orders_qty] if other_orders_qty > 0 else 0,
                   color='darkred', label=f'Current Order: {current_order_qty:.0f}')
        
        ax3.set_ylabel('Monthly Quantity')
        ax3.set_title(f'Monthly Volume Analysis\nCurrent Order: {current_order_qty:.0f} / {current_month_total:.0f} ({(current_order_qty/current_month_total*100):.1f}%)')
        ax3.legend(fontsize=10)
    
    # Plot 4: Feature Importance (Anomaly Contributors as Percentages)
    ax4 = axes[1, 1]
    
    # Use pre-computed SHAP if available, otherwise try on-demand computation or CSV SHAP
    shap_text = pre_computed_shap if pre_computed_shap else row.get('shap_explanation', '')
    
    if compute_shap_on_demand and not pre_computed_shap:
        try:
            # Import model service for on-demand SHAP computation
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from model_service import model_service
            
            if model_service.loaded:
                computed_shap = model_service.compute_shap_for_sample(row, features_df)
                if computed_shap:
                    shap_text = computed_shap
                    ax4.text(0.02, 0.02, 'SHAP computed on-demand', 
                            transform=ax4.transAxes, fontsize=8, style='italic', color='green')
        except Exception as e:
            ax4.text(0.02, 0.02, f'SHAP computation failed: {str(e)[:30]}...', 
                    transform=ax4.transAxes, fontsize=8, style='italic', color='red')
    
    # Extract feature importances from SHAP explanation if available
    if shap_text and 'Top contributors:' in shap_text:
        # Parse SHAP explanation
        contributors = []
        shap_values = []
        
        # Simple parsing of SHAP text
        parts = shap_text.split('Top contributors:')[1].split(';')[:5]  # Top 5 features
        for part in parts:
            if ':' in part and '(' in part:
                feature_name = part.split(':')[0].strip()
                # Extract the impact value
                if 'increases' in part:
                    impact = float(part.split('by')[1].split(')')[0].strip())
                else:
                    impact = -float(part.split('by')[1].split(')')[0].strip())
                
                contributors.append(format_feature_name(feature_name)[:20])  # Truncate long names
                shap_values.append(impact)
        
        if contributors:
            # Focus on anomaly contributors only (negative SHAP values)
            anomaly_contributors = []
            anomaly_values = []
            
            for contrib, val in zip(contributors, shap_values):
                if val < 0:  # Negative SHAP values increase anomaly score
                    anomaly_contributors.append(contrib)
                    anomaly_values.append(abs(val))  # Use absolute value for percentage calculation
            
            if anomaly_contributors:
                # Calculate percentage contributions
                total_anomaly_influence = sum(anomaly_values)
                percentages = [(val / total_anomaly_influence * 100) for val in anomaly_values]
                
                # Create horizontal bar chart with percentages
                bars = ax4.barh(anomaly_contributors, percentages, color='red', alpha=0.7)
                
                # Add percentage labels on bars
                for i, (bar, pct) in enumerate(zip(bars, percentages)):
                    width = bar.get_width()
                    ax4.text(width + 1, bar.get_y() + bar.get_height()/2, 
                            f'{pct:.1f}%', ha='left', va='center', fontweight='bold')
                
                ax4.set_xlabel('% Contribution to Anomaly Warning')
                ax4.set_title('Anomaly Contributors')
                ax4.set_xlim(0, max(percentages) * 1.15)  # Add space for labels
                
                # Add note about total
                ax4.text(0.02, 0.98, f'Total contributors: {len(anomaly_contributors)}', 
                        transform=ax4.transAxes, va='top', fontsize=8, style='italic')
            else:
                # No negative SHAP values found
                ax4.text(0.5, 0.5, 'No anomaly contributors\nfound in SHAP data', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Anomaly Contributors')
        else:
            ax4.text(0.5, 0.5, 'No SHAP data available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Feature Importance')
    else:
        # Fallback: Show boolean feature flags
        bool_features = ['is_first_time_cust_material_order', 'is_rare_material', 
                        'is_suspected_duplicate_order', 'is_unusual_ship_to_for_sold_to',
                        'is_qty_outside_typical_range', 'is_unusual_uom']
        
        feature_names = []
        feature_values = []
        for feat in bool_features:
            if feat in row and row[feat]:
                feature_names.append(format_feature_name(feat.replace('is_', '')).title()[:20])
                feature_values.append(1)
        
        if feature_names:
            ax4.barh(feature_names, feature_values, color='orange')
            ax4.set_xlim(0, 1.2)
            ax4.set_xlabel('Active Anomaly Flags')
            ax4.set_title('Detected Anomaly Patterns')
        else:
            ax4.text(0.5, 0.5, 'No anomaly flags detected', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Anomaly Patterns')
    
    plt.tight_layout()
    
    # Prepare return values based on what was computed
    return_values = [fig]
    
    # Save complete analysis for AI analysis if requested
    if save_for_ai_analysis:
        # Create a temporary directory for this analysis
        temp_dir = tempfile.mkdtemp(prefix="anomaly_plots_")
        
        # Save only the complete figure for AI analysis
        main_plot_path = os.path.join(temp_dir, "complete_analysis.png")
        fig.savefig(main_plot_path, dpi=150, bbox_inches='tight')
        
        # Return only the main plot path in a list for compatibility
        image_paths = [main_plot_path]
        return_values.append(image_paths)
    
    # Add on-demand SHAP text if it was computed
    if compute_shap_on_demand and 'shap_text' in locals() and shap_text != row.get('shap_explanation', ''):
        # Only return the on-demand SHAP if it's different from what's in the CSV
        return_values.append(shap_text)
    
    # Return appropriate values
    if len(return_values) == 1:
        return return_values[0]
    else:
        return tuple(return_values)


def create_feature_analysis_plot(results_df: pd.DataFrame, feature_columns: List[str], model_type: str) -> None:
    """
    Create detailed feature analysis visualization.
    
    Args:
        results_df: Results with predictions
        feature_columns: Feature column names
        model_type: Type of model used for unique filename
    """
    # Select top boolean features for detailed analysis
    boolean_features = [f for f in feature_columns if f.startswith('is_') and f in results_df.columns][:8]
    
    if boolean_features:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, feature in enumerate(boolean_features):
            if i < len(axes):
                # Calculate anomaly rates for this feature
                feature_analysis = results_df.groupby(feature)['predicted_anomaly'].agg(['mean', 'count'])
                
                bars = axes[i].bar(feature_analysis.index.astype(str), feature_analysis['mean'], 
                                  color=['lightblue', 'lightcoral'])
                axes[i].set_title(f'{feature.replace("is_", "").replace("_", " ").title()}')
                axes[i].set_ylabel('Anomaly Rate')
                axes[i].set_xlabel('Feature Value')
                
                # Add count labels on bars
                for j, bar in enumerate(bars):
                    height = bar.get_height()
                    count = feature_analysis['count'].iloc[j]
                    axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                               f'n={count}', ha='center', va='bottom', fontsize=8)
                
                axes[i].grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(boolean_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Anomaly Rates by Feature Values', fontsize=16)
        plt.tight_layout()
        
        # Save with model-specific filename
        safe_model_type = model_type.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('-', '_')
        feature_filename = f'feature_analysis_{safe_model_type}.png'
        plt.savefig(os.path.join(settings.RESULTS_DIR, feature_filename), 
                    dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()


def create_top_features_vs_score_plot(
    results_df: pd.DataFrame, 
    anomaly_scores: np.ndarray, 
    anomaly_labels: np.ndarray, 
    feature_columns: List[str], 
    model_type: str
) -> None:
    """
    Create comprehensive visualization showing anomaly scores vs top 10 features.
    Prefers SHAP global importance when available; falls back to normalized mean
    difference otherwise.
    """
    print(f"Creating top features vs anomaly score visualization...")
    
    # Prefer SHAP global importance if present on results_df
    feature_importance = []
    shap_available = hasattr(results_df, 'attrs') and 'shap_explanations' in results_df.attrs and results_df.attrs['shap_explanations'] is not None
    if shap_available:
        shap_data = results_df.attrs['shap_explanations']
        shap_values = shap_data.get('shap_values', None)
        if shap_values is not None and hasattr(shap_values, 'values'):
            mean_abs = np.abs(shap_values.values).mean(axis=0)
            feature_importance = list(zip(feature_columns, mean_abs))
    
    # Fallback to normalized mean diff
    if not feature_importance:
        for feature in feature_columns:
            if feature in results_df.columns:
                anomaly_mean = results_df[results_df['predicted_anomaly'] == 1][feature].mean()
                normal_mean = results_df[results_df['predicted_anomaly'] == 0][feature].mean()
                if not np.isnan(anomaly_mean) and not np.isnan(normal_mean):
                    std = results_df[feature].std()
                    if std > 0:
                        importance = abs(anomaly_mean - normal_mean) / std
                        feature_importance.append((feature, importance))
    
    if not feature_importance:
        print("No suitable features found for visualization")
        return
    
    # Sort and take top 10
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    top_features = [feat[0] for feat in feature_importance[:10]]
    
    # Create figure
    fig, axes = plt.subplots(5, 2, figsize=(12, 24))
    axes = axes.flatten()
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for idx, feature in enumerate(top_features):
        ax = axes[idx]
        x = results_df[feature]
        y = anomaly_scores
        normal_mask = results_df['predicted_anomaly'] == 0
        anomaly_mask = results_df['predicted_anomaly'] == 1
        ax.scatter(x[normal_mask], y[normal_mask], c=colors[0], alpha=0.5, s=15, label='Normal', edgecolors='none')
        ax.scatter(x[anomaly_mask], y[anomaly_mask], c=colors[1], alpha=0.8, s=25, label='Anomaly', edgecolors='white', linewidth=0.5)
        
        # Threshold line
        if settings.CONTAMINATION_RATE == 'auto':
            threshold = 0.5 if 'hana' in model_type.lower() else 0.0
        else:
            threshold = np.percentile(anomaly_scores, (1-float(settings.CONTAMINATION_RATE))*100) if 'hana' in model_type.lower() else np.percentile(anomaly_scores, settings.CONTAMINATION_RATE*100)
        ax.axhline(threshold, color=colors[2], linestyle='--', alpha=0.7, linewidth=1.5)
        
        feature_display_name = feature.replace('is_', '').replace('_', ' ').title()
        if len(feature_display_name) > 25:
            feature_display_name = feature_display_name[:22] + '...'
        ax.set_xlabel(feature_display_name, fontsize=10)
        ax.set_ylabel('Anomaly Score', fontsize=10)
        ax.set_title(f'{idx+1}. {feature_display_name}', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
        
        importance_score = feature_importance[idx][1]
        ax.text(0.02, 0.98, f'Importance: {importance_score:.2f}' + (' (SHAP)' if shap_available else ''),
                transform=ax.transAxes, fontsize=9, 
                verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Save file
    import os
    fig.tight_layout()
    safe_model_type = model_type.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('-', '_')
    filename = os.path.join(settings.RESULTS_DIR, f'top_features_vs_score_{safe_model_type}.png')
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
