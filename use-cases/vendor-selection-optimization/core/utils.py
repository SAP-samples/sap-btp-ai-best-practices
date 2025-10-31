"""
Utility functions for the Vendor Performance Dashboard.
Contains helper functions for data formatting, calculations, and other common operations.
"""

import pandas as pd
import numpy as np
import gc
from config import settings


def format_percentage(value, decimal_places=1):
    """Format a value as a percentage with specified decimal places"""
    return f"{value:.{decimal_places}%}"


def format_currency(value, decimal_places=2, currency_symbol='$'):
    """Format a value as currency with specified decimal places and symbol"""
    return f"{currency_symbol}{value:.{decimal_places}f}"


def format_number(value, decimal_places=1):
    """Format a number with specified decimal places"""
    return f"{value:.{decimal_places}f}"


def format_display_dataframe(df):
    """Format a DataFrame for display with memory optimization"""
    # Memory optimization: Use view instead of copy when possible, copy only at the end
    display_df = df.copy()
    
    # Memory-efficient vectorized formatting using pandas string methods where possible
    
    # Format percentage columns using vectorized operations
    percentage_columns = ['OnTimeRate', 'InFullRate', 'Avg_OTIF_Rate', 'OnTimeRate_raw', 'InFullRate_raw']
    for col in percentage_columns:
        if col in display_df.columns:
            # Vectorized percentage formatting
            display_df[col] = (display_df[col] * 100).round(1).astype(str) + '%'
    
    # Format tariff percentage column (already in percentage format)
    if 'TariffImpact_raw_percent' in display_df.columns:
        display_df['TariffImpact_raw_percent'] = display_df['TariffImpact_raw_percent'].round(1).astype(str) + '%'
    
    # Format numeric columns using vectorized operations
    numeric_format_columns = ['MedianLeadTimeDays', 'AvgLeadTimeDays_raw']
    for col in numeric_format_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(1).astype(str)
    
    # Format cost columns efficiently
    cost_columns = [col for col in display_df.columns if col.startswith('cost_') or col == 'EffectiveCostPerUnit_USD']
    for col in cost_columns:
        if col in display_df.columns:
            # Vectorized currency formatting
            display_df[col] = '$' + display_df[col].round(2).astype(str)
    
    # Memory-efficient column renaming - build rename dict only for existing columns
    rename_dict = {}
    
    # Add cost column renames
    for col in cost_columns:
        if col.startswith('cost_') and col in display_df.columns:
            display_name = col.replace('cost_', '').replace('_', ' ').title()
            rename_dict[col] = f"Cost: {display_name}"
    
    # Add standard column renames for existing columns only
    if hasattr(settings, 'COLUMN_RENAMES'):
        for old_col, new_col in settings.COLUMN_RENAMES.items():
            if old_col in display_df.columns:
                rename_dict[old_col] = new_col
    
    # Apply all renames in one operation
    if rename_dict:
        display_df = display_df.rename(columns=rename_dict)
    
    return display_df


def get_active_cost_columns(df, costs_config=None):
    """Get list of active cost columns based on configuration and available columns"""
    if costs_config:
        active_cost_columns = [col for col, active in costs_config.items() 
                              if active == "True" and col in df.columns]
    else:
        # If no costs config, use all available cost columns
        active_cost_columns = [col for col in df.columns if col.startswith('cost_')]
    
    return active_cost_columns


def normalize_min_max(series, lower_is_better=False):
    """Normalize a series using min-max scaling, with option for inverting scale"""
    min_val, max_val = series.min(), series.max()
    if max_val == min_val:
        return pd.Series([0.5] * len(series), index=series.index)
    if lower_is_better:
        return (max_val - series) / (max_val - min_val)
    else:
        return (series - min_val) / (max_val - min_val)


def filter_dataframe(df, filters):
    """Apply filters to a DataFrame with memory optimization"""
    # Memory optimization: Use boolean indexing to avoid intermediate copies
    mask = pd.Series(True, index=df.index)
    
    # Apply text filters using boolean indexing
    for column, value in filters.get('text_filters', {}).items():
        if value != 'All' and column in df.columns:
            mask = mask & (df[column] == value)
    
    # Apply range filters using boolean indexing
    for column, value_range in filters.get('range_filters', {}).items():
        if column in df.columns:
            min_val, max_val = value_range
            mask = mask & (df[column] >= min_val) & (df[column] <= max_val)
    
    # Apply minimum value filters using boolean indexing
    for column, min_value in filters.get('min_filters', {}).items():
        if column in df.columns:
            mask = mask & (df[column] >= min_value)
    
    # Only create copy at the end with filtered data
    return df[mask].copy()


def get_memory_usage_info(df):
    """Get memory usage information for a DataFrame"""
    memory_usage = df.memory_usage(deep=True)
    total_memory = memory_usage.sum()
    
    return {
        'total_memory_mb': total_memory / (1024 * 1024),
        'rows': len(df),
        'columns': len(df.columns),
        'memory_per_row': total_memory / len(df) if len(df) > 0 else 0
    }


def optimize_dataframe_memory(df):
    """Optimize DataFrame memory usage by downcasting numeric types"""
    optimized_df = df.copy()
    
    # Downcast integer columns
    int_columns = optimized_df.select_dtypes(include=['int64']).columns
    for col in int_columns:
        optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')
    
    # Downcast float columns
    float_columns = optimized_df.select_dtypes(include=['float64']).columns
    for col in float_columns:
        optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
    
    # Convert object columns to category if they have few unique values
    object_columns = optimized_df.select_dtypes(include=['object']).columns
    for col in object_columns:
        num_unique = optimized_df[col].nunique()
        num_total = len(optimized_df[col])
        
        # Convert to category if less than 50% unique values
        if num_unique / num_total < 0.5:
            optimized_df[col] = optimized_df[col].astype('category')
    
    return optimized_df