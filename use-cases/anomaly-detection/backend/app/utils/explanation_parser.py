"""
Utility module for parsing and structuring explanations for better UI presentation.

This module provides functions to parse text-based explanations into structured
formats suitable for enhanced display in the Streamlit UI.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union


# Mapping of rule patterns to their corresponding model features
RULE_TO_FEATURE_MAPPING = {
    # Quantity-based rules
    r'Qty \(actual: [\d.]+\) outside typical range': 'is_qty_outside_typical_range',
    r'Monthly volume \(current: [\d.]+\) outside typical range': 'is_monthly_qty_outside_typical_range',
    
    # Duplicate detection rules
    r'Suspected duplicate: Material': 'is_suspected_duplicate_order',
    
    # Ship-to rules (not commonly used in current models)
    r'Unusual ship-to:': 'is_unusual_ship_to_for_sold_to',
    
    # Unit of measure rules
    r'Unusual UoM:': 'is_unusual_uom',
    
    # Pricing rules
    r'Unusual unit price:': 'is_unusual_unit_price',
    r'Order item value mismatch': 'is_value_mismatch_price_qty',
    
    # Fulfillment time rules
    r'Unusual fulfillment time:': 'is_unusual_fulfillment_time',
    
    # Order history rules
    r'First-time customer-material order': 'is_first_time_cust_material_order',
    
    # Material rarity rules
    r'Rare material \(appears less than': 'is_rare_material',
}


def get_active_model_features(results_dir: str = None) -> Optional[List[str]]:
    """
    Load the active features from the current model metadata.
    
    Args:
        results_dir: Path to results directory. If None, attempts to find latest results.
        
    Returns:
        List of active feature names, or None if not found
    """
    try:
        if results_dir is None:
            # Try to find the latest results directory
            ui_dir = Path(__file__).parent.parent
            results_base = ui_dir / "results"
            
            if not results_base.exists():
                return None
                
            # Find the most recent results directory
            result_dirs = [d for d in results_base.iterdir() if d.is_dir() and d.name.startswith('anomaly_detection_results')]
            if not result_dirs:
                return None
                
            # Sort by modification time and get the latest
            latest_dir = max(result_dirs, key=lambda d: d.stat().st_mtime)
            results_dir = str(latest_dir)
        
        # Load stratified model metadata first, fallback to regular metadata
        metadata_files = [
            Path(results_dir) / "models" / "stratified_models_metadata.json",
            Path(results_dir) / "models" / "sklearn_model_metadata.json",
            Path(results_dir) / "models" / "hana_model_metadata.json"
        ]
        
        for metadata_file in metadata_files:
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    return metadata.get('feature_columns', [])
                    
        return None
        
    except Exception as e:
        print(f"Warning: Could not load model features: {e}")
        return None


def is_rule_active(rule_text: str, active_features: List[str]) -> bool:
    """
    Check if a rule corresponds to an active feature in the current model.
    
    Args:
        rule_text: The rule text to check
        active_features: List of active feature names from the model
        
    Returns:
        bool: True if rule should be shown, False if it should be filtered out
    """
    if not active_features:
        # If we can't determine active features, show all rules (fallback)
        return True
    
    # Check each rule pattern mapping
    for pattern, feature_name in RULE_TO_FEATURE_MAPPING.items():
        if re.search(pattern, rule_text):
            return feature_name in active_features
    
    # If no pattern matches, include the rule (unknown rules are shown by default)
    return True


def parse_rule_based_explanation(explanation_text: str, active_features: List[str] = None) -> List[Dict[str, Any]]:
    """
    Parse rule-based explanation text into structured format, filtering by active model features.
    
    Args:
        explanation_text: Semi-colon separated explanation string
        active_features: List of active feature names from the current model. 
                        If None, will attempt to load from latest model metadata.
        
    Returns:
        List of structured explanation dictionaries (filtered by active features)
    """
    # Handle None, NaN, or non-string values
    if not explanation_text or explanation_text == 'N/A' or explanation_text == '' or pd.isna(explanation_text):
        return []
    
    # Convert to string if needed
    explanation_text = str(explanation_text)
    
    # Load active features if not provided
    if active_features is None:
        active_features = get_active_model_features()
    
    # Split by semicolon to get individual rules
    rules = explanation_text.split(';')
    structured_rules = []
    
    for rule in rules:
        rule = rule.strip()
        if not rule:
            continue
        
        # Check if this rule corresponds to an active feature
        if not is_rule_active(rule, active_features):
            continue  # Skip rules for inactive features
            
        # Parse different rule patterns
        structured_rule = parse_single_rule(rule)
        if structured_rule:
            structured_rules.append(structured_rule)
    
    return structured_rules


def parse_single_rule(rule_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single rule text into structured format.
    
    Args:
        rule_text: Single rule explanation text
        
    Returns:
        Structured rule dictionary or None if parsing fails
    """
    # Pattern for quantity-based rules
    qty_pattern = r'Qty \(actual: ([\d.]+)\) outside typical range \[([\d.]+)-([\d.]+)\]'
    match = re.match(qty_pattern, rule_text)
    if match:
        return {
            'feature_display_name': 'Quantity',
            'actual_value': float(match.group(1)),
            'status': 'outside typical range',
            'expected_value': f'[{match.group(2)} - {match.group(3)}]',
            'raw_reason': rule_text
        }
    
    # Pattern for monthly volume
    monthly_pattern = r'Monthly volume \(current: ([\d.]+)\) outside typical range \[([\d.]+)-([\d.]+)\]'
    match = re.match(monthly_pattern, rule_text)
    if match:
        return {
            'feature_display_name': 'Monthly Volume',
            'actual_value': float(match.group(1)),
            'status': 'outside typical range',
            'expected_value': f'[{match.group(2)} - {match.group(3)}]',
            'raw_reason': rule_text
        }
    
    # Pattern for suspected duplicate (new format)
    duplicate_pattern_new = r"Suspected duplicate: Material '([^']+)' qty ([\d.]+) \(previous order on ([^,]+), ([\d.]+)h prior\)"
    match = re.match(duplicate_pattern_new, rule_text)
    if match:
        return {
            'feature_display_name': 'Suspected Duplicate',
            'actual_value': f"Material: {match.group(1)}, Qty: {match.group(2)}",
            'status': 'detected',
            'expected_value': f'Previous order on {match.group(3)}, {match.group(4)}h prior',
            'raw_reason': rule_text
        }

    # Pattern for suspected duplicate (old format)
    # duplicate_pattern = r"Suspected duplicate: PO '([^']+)' \(see original Doc: '([^']+)' on ([^,]+), ([\d.]+)h prior\)"
    # match = re.match(duplicate_pattern, rule_text)
    # if match:
    #     return {
    #         'feature_display_name': 'Suspected Duplicate',
    #         'actual_value': f"PO: {match.group(1)}",
    #         'status': 'detected',
    #         'expected_value': f'Original Doc: {match.group(2)} on {match.group(3)}',
    #         'raw_reason': rule_text
    #     }
    
    # Pattern for unusual ship-to
    shipto_pattern = r"Unusual ship-to: '([^']+)' for sold-to '([^']+)' \(usage: ([\d.]+)%\)"
    match = re.match(shipto_pattern, rule_text)
    if match:
        return {
            'feature_display_name': 'Ship-To Location',
            'actual_value': match.group(1),
            'status': 'unusual for customer',
            'expected_value': f'Usage: {match.group(3)}% (low)',
            'raw_reason': rule_text
        }
    
    # Pattern for unusual UoM
    uom_pattern = r"Unusual UoM: '([^']+)' \(expected '([^']+)'\)"
    match = re.match(uom_pattern, rule_text)
    if match:
        return {
            'feature_display_name': 'Unit of Measure',
            'actual_value': match.group(1),
            'status': 'unusual',
            'expected_value': match.group(2),
            'raw_reason': rule_text
        }
    
    # Pattern for unit price
    price_pattern = r'Unusual unit price: ([\d.]+) \(expected range \[([\d.]+)-([\d.]+)\]\)'
    match = re.match(price_pattern, rule_text)
    if match:
        return {
            'feature_display_name': 'Unit Price',
            'actual_value': float(match.group(1)),
            'status': 'outside expected range',
            'expected_value': f'[${match.group(2)} - ${match.group(3)}]',
            'raw_reason': rule_text
        }
    
    # Pattern for value mismatch
    mismatch_pattern = r'Order item value mismatch \(actual: ([\d.]+), expected: ([\d.]+)\)'
    match = re.match(mismatch_pattern, rule_text)
    if match:
        return {
            'feature_display_name': 'Order Value',
            'actual_value': f'{float(match.group(1)):,.2f}',
            'status': 'mismatch with price × quantity',
            'expected_value': f'{float(match.group(2)):,.2f}',
            'raw_reason': rule_text
        }
    
    # Pattern for fulfillment time
    fulfillment_pattern = r'Unusual fulfillment time: ([\d.]+) days \(expected range \[([\d.]+)-([\d.]+) days\]\)'
    match = re.match(fulfillment_pattern, rule_text)
    if match:
        return {
            'feature_display_name': 'Fulfillment Time',
            'actual_value': f'{int(float(match.group(1)))} days',
            'status': 'outside typical range',
            'expected_value': f'[{int(float(match.group(2)))} - {int(float(match.group(3)))} days]',
            'raw_reason': rule_text
        }
    
    # Pattern for first-time order
    if 'First-time customer-material order' in rule_text:
        return {
            'feature_display_name': 'Order History',
            'actual_value': 'First-time',
            'status': 'new combination',
            'expected_value': 'Previous orders exist',
            'raw_reason': rule_text
        }
    
    # Pattern for rare material
    rare_pattern = r'Rare material \(appears less than (\d+) times\): ([^\(]+) \(([^\)]+)\)'
    match = re.match(rare_pattern, rule_text)
    if match:
        return {
            'feature_display_name': 'Material Rarity',
            'actual_value': match.group(2).strip(),
            'status': 'rare material',
            'expected_value': f'Appears < {match.group(1)} times',
            'raw_reason': rule_text
        }
    
    # Default fallback - return raw text
    return {
        'feature_display_name': 'Anomaly Flag',
        'actual_value': rule_text,
        'status': 'detected',
        'expected_value': '',
        'raw_reason': rule_text
    }


def parse_shap_explanation(explanation_text: str) -> List[Dict[str, Any]]:
    """
    Parse SHAP explanation text into structured format.
    
    Args:
        explanation_text: SHAP explanation string
        
    Returns:
        List of structured SHAP contribution dictionaries
    """
    # Handle None, NaN, or non-string values
    if not explanation_text or explanation_text == 'N/A' or pd.isna(explanation_text):
        return []
    
    # Convert to string if needed
    explanation_text = str(explanation_text)
    
    if not explanation_text.startswith('Top contributors:'):
        return []
    
    # Remove "Top contributors: " prefix
    contributors_text = explanation_text.replace('Top contributors: ', '')
    
    # Split by semicolon to get individual contributions
    contributions = contributors_text.split(';')
    structured_contributions = []
    
    for contrib in contributions:
        contrib = contrib.strip()
        if not contrib:
            continue
            
        # Parse pattern: "feature_name: value (increases/decreases anomaly score by amount)"
        pattern = r'([^:]+): ([\d.-]+) \((increases|decreases) anomaly score by ([\d.]+)\)'
        match = re.match(pattern, contrib)
        
        if match:
            feature_name = match.group(1).strip()
            feature_value = float(match.group(2))
            direction = match.group(3)
            impact = float(match.group(4))
            
            # SHAP value is negative if it decreases anomaly score (makes more anomalous)
            shap_value = -impact if direction == 'decreases' else impact
            
            structured_contributions.append({
                'feature_name': feature_name,
                'feature_value': feature_value,
                'shap_value': shap_value,
                'abs_contribution': abs(shap_value)
            })
    
    return structured_contributions


def format_feature_value(value: Any, feature_name: str = None) -> str:
    """
    Format feature value for display based on type and feature name.
    
    Args:
        value: The value to format
        feature_name: Optional feature name for context-aware formatting
        
    Returns:
        Formatted string representation
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    
    if isinstance(value, bool):
        return "Yes" if value else "No"
    
    if isinstance(value, (int, float)):
        # Boolean formatting for features starting with "is_"
        if feature_name and feature_name.lower().startswith("is_"):
            return "Yes" if value else "No"
        # Percentage formatting
        elif feature_name and 'percentage' in feature_name.lower():
            return f"{value:.1%}"
        # Z-score formatting
        elif feature_name and 'z_score' in feature_name.lower():
            return f"{value:.3f}"
        # General number formatting
        else:
            return f"{value:,.2f}"
    
    return str(value)


def parse_rule_based_explanation_with_model_filtering(explanation_text: str, results_dir: str = None) -> List[Dict[str, Any]]:
    """
    Convenience function to parse rule-based explanations with automatic model feature filtering.
    
    This function automatically loads the active features from the specified (or latest) model
    and filters the rule explanations to only show rules corresponding to features that were
    selected in the Fine Tuning tab for the current model.
    
    Args:
        explanation_text: Semi-colon separated explanation string
        results_dir: Path to specific results directory. If None, uses latest results.
        
    Returns:
        List of structured explanation dictionaries (filtered by active model features)
        
    Example:
        # Use latest model features automatically
        parsed_rules = parse_rule_based_explanation_with_model_filtering(explanation)
        
        # Use specific model features
        parsed_rules = parse_rule_based_explanation_with_model_filtering(
            explanation, 
            "/path/to/results/anomaly_detection_results_backend_sklearn_contamination_auto"
        )
    """
    # Load active features for the specified model
    active_features = get_active_model_features(results_dir)
    
    # Parse with feature filtering
    return parse_rule_based_explanation(explanation_text, active_features)


def get_feature_filtering_info(results_dir: str = None) -> Dict[str, Any]:
    """
    Get information about which features are active and how rules will be filtered.
    
    Args:
        results_dir: Path to results directory. If None, uses latest results.
        
    Returns:
        Dictionary with filtering information for debugging/display
    """
    active_features = get_active_model_features(results_dir)
    
    if not active_features:
        return {
            'status': 'error',
            'message': 'Could not load model features',
            'active_features': [],
            'total_rules': len(RULE_TO_FEATURE_MAPPING),
            'active_rules': 0
        }
    
    # Count which rule types will be active
    active_rule_features = []
    for feature_name in RULE_TO_FEATURE_MAPPING.values():
        if feature_name in active_features:
            active_rule_features.append(feature_name)
    
    return {
        'status': 'success',
        'active_features': active_features,
        'total_features_in_fine_tuning': len(RULE_TO_FEATURE_MAPPING),
        'active_rule_features': active_rule_features,
        'inactive_rule_features': [f for f in RULE_TO_FEATURE_MAPPING.values() if f not in active_features],
        'filtering_enabled': True
    }


# Import pandas for NaN checking
import pandas as pd