"""
Customer stratification utilities for pharmaceutical anomaly detection.

This module handles customer tier assignment based on order volume patterns.
"""

import pandas as pd
from typing import Dict
from config.settings import CUSTOMER_COL, KEY_COL, LARGE_CUSTOMER_THRESHOLD, MEDIUM_CUSTOMER_THRESHOLD


def assign_customer_tiers(df_full: pd.DataFrame) -> Dict[str, str]:
    """
    Assign customer tiers based on historical order volume.
    
    Args:
        df_full: Full dataset to analyze customer patterns
        
    Returns:
        Dictionary mapping customer_id -> tier ('large', 'medium', 'small')
    """
    print(f"\nAssigning customer tiers based on order volume...")
    
    # Calculate order counts per customer
    customer_orders = df_full.groupby(CUSTOMER_COL)[KEY_COL].nunique().reset_index()
    customer_orders.columns = ['customer_id', 'order_count']
    
    # Assign tiers
    customer_tiers = {}
    large_customers = []
    medium_customers = []
    small_customers = []
    
    for _, row in customer_orders.iterrows():
        customer_id = str(row['customer_id'])  # Convert to string for JSON serialization
        order_count = row['order_count']
        
        if order_count >= LARGE_CUSTOMER_THRESHOLD:
            tier = 'large'
            large_customers.append(customer_id)
        elif order_count >= MEDIUM_CUSTOMER_THRESHOLD:
            tier = 'medium'
            medium_customers.append(customer_id)
        else:
            tier = 'small'
            small_customers.append(customer_id)
            
        customer_tiers[customer_id] = tier
    
    print(f"Customer tier distribution:")
    print(f"  Large customers (≥{LARGE_CUSTOMER_THRESHOLD} orders): {len(large_customers):,}")
    print(f"  Medium customers ({MEDIUM_CUSTOMER_THRESHOLD}-{LARGE_CUSTOMER_THRESHOLD-1} orders): {len(medium_customers):,}")
    print(f"  Small customers (<{MEDIUM_CUSTOMER_THRESHOLD} orders): {len(small_customers):,}")
    
    return customer_tiers