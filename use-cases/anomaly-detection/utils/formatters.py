"""
Formatting utilities for feature names and display text.
"""

import re


def format_feature_name(feature_name: str) -> str:
    """Format feature name by replacing underscores with spaces"""
    if not feature_name:
        return feature_name
    return feature_name.replace('_', ' ')


def format_ai_response_feature_names(ai_response: str) -> str:
    """
    Format feature names in AI responses by replacing underscores with spaces.
    
    This function identifies feature names with underscores in AI-generated text
    and formats them for better readability.
    
    Args:
        ai_response: The AI-generated response text
        
    Returns:
        The response text with formatted feature names
    """
    if not ai_response:
        return ai_response
    
    # Common feature name patterns found in pharmaceutical anomaly detection
    feature_patterns = {
        # Order quantity patterns
        r'\bsales_order_item_qty\b': 'Sales Order Item Qty',
        r'\bactual_sales_order_item_qty\b': 'Actual Sales Order Item Qty',
        r'\btypical_qty_p05\b': 'Typical Qty P05',
        r'\btypical_qty_p95\b': 'Typical Qty P95',
        r'\bqty_z_score\b': 'Qty Z Score',
        r'\bqty_deviation_from_mean\b': 'Qty Deviation From Mean',
        
        # Price patterns
        r'\bunit_price\b': 'Unit Price',
        r'\bactual_unit_price\b': 'Actual Unit Price',
        r'\btypical_price_p05\b': 'Typical Price P05',
        r'\btypical_price_p95\b': 'Typical Price P95',
        
        # Order value patterns  
        r'\border_item_value\b': 'Order Item Value',
        r'\bactual_order_item_value\b': 'Actual Order Item Value',
        r'\bexpected_order_item_value\b': 'Expected Order Item Value',
        r'\bexpected_order_item_value_model\b': 'Expected Order Item Value Model',
        
        # Monthly patterns
        r'\bcurrent_month_total_qty\b': 'Current Month Total Qty',
        r'\bactual_current_month_total_qty_for_customer_material\b': 'Actual Current Month Total Qty For Customer Material',
        r'\btypical_monthly_qty_p05\b': 'Typical Monthly Qty P05',
        r'\btypical_monthly_qty_p95\b': 'Typical Monthly Qty P95',
        
        # Fulfillment patterns
        r'\bfulfillment_duration_days\b': 'Fulfillment Duration Days',
        r'\bactual_fulfillment_duration_days\b': 'Actual Fulfillment Duration Days',
        r'\btypical_fulfillment_p05\b': 'Typical Fulfillment P05',
        r'\btypical_fulfillment_p95\b': 'Typical Fulfillment P95',
        
        # Boolean flag patterns
        r'\bis_first_time_cust_material_order\b': 'Is First Time Customer Material Order',
        r'\bflag_is_first_time_cust_material_order\b': 'Flag Is First Time Customer Material Order',
        r'\bis_rare_material\b': 'Is Rare Material',
        r'\bflag_is_rare_material\b': 'Flag Is Rare Material',
        r'\bis_suspected_duplicate_order\b': 'Is Suspected Duplicate Order',
        r'\bflag_is_suspected_duplicate_order\b': 'Flag Is Suspected Duplicate Order',
        r'\bis_unusual_unit_price\b': 'Is Unusual Unit Price',
        r'\bflag_is_unusual_unit_price\b': 'Flag Is Unusual Unit Price',
        r'\bis_qty_outside_typical_range\b': 'Is Qty Outside Typical Range',
        r'\bflag_is_qty_outside_typical_range\b': 'Flag Is Qty Outside Typical Range',
        r'\bis_unusual_fulfillment_time\b': 'Is Unusual Fulfillment Time',
        r'\bflag_is_unusual_fulfillment_time\b': 'Flag Is Unusual Fulfillment Time',
        r'\bis_monthly_qty_outside_typical_range\b': 'Is Monthly Qty Outside Typical Range',
        r'\bflag_is_monthly_qty_outside_typical_range\b': 'Flag Is Monthly Qty Outside Typical Range',
        r'\bis_value_mismatch_price_qty\b': 'Is Value Mismatch Price Qty',
        r'\bflag_is_value_mismatch_price_qty\b': 'Flag Is Value Mismatch Price Qty',
        r'\bis_unusual_uom\b': 'Is Unusual UOM',
        r'\bflag_is_unusual_uom\b': 'Flag Is Unusual UOM',
        r'\bis_unusual_ship_to_for_sold_to\b': 'Is Unusual Ship To For Sold To',
        r'\bflag_is_unusual_ship_to_for_sold_to\b': 'Flag Is Unusual Ship To For Sold To',
        
        # Model and score patterns
        r'\bmodel_used\b': 'Model Used',
        r'\bmodel_anomaly_score\b': 'Model Anomaly Score',
        r'\bmodel_predicted_anomaly\b': 'Model Predicted Anomaly',
        r'\banomaly_score\b': 'Anomaly Score',
        r'\bpredicted_anomaly\b': 'Predicted Anomaly',
        
        # Order identifier patterns
        r'\border_sales_document_number\b': 'Order Sales Document Number',
        r'\border_sales_document_item\b': 'Order Sales Document Item',
        r'\border_customer_po_number\b': 'Order Customer PO Number',
        r'\border_material_number\b': 'Order Material Number',
        r'\border_material_description\b': 'Order Material Description',
        r'\border_sold_to_number\b': 'Order Sold To Number',
        r'\border_ship_to_party\b': 'Order Ship To Party',
        r'\border_created_date\b': 'Order Created Date',
        
        # Historical patterns
        r'\bhistorical_median_qty_for_material\b': 'Historical Median Qty For Material',
        r'\bnum_historical_orders_for_material\b': 'Num Historical Orders For Material',
        r'\btypical_qty_z_score_range\b': 'Typical Qty Z Score Range',
    }
    
    # Apply specific pattern replacements
    formatted_response = ai_response
    for pattern, replacement in feature_patterns.items():
        formatted_response = re.sub(pattern, replacement, formatted_response, flags=re.IGNORECASE)
    
    # Generic fallback: replace any remaining underscore patterns that look like feature names
    # This catches any feature names we might have missed in our specific patterns
    def generic_underscore_replacer(match):
        feature_name = match.group(0)
        # Only format if it looks like a feature name (contains underscores and is mostly lowercase)
        if '_' in feature_name and feature_name.islower():
            return format_feature_name(feature_name)
        return feature_name
    
    # Find remaining potential feature names with underscores (word boundaries, lowercase with underscores)
    formatted_response = re.sub(r'\b[a-z][a-z0-9_]*[a-z0-9]\b', generic_underscore_replacer, formatted_response)
    
    return formatted_response