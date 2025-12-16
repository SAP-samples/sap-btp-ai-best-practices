"""
Feature selection and engineering module for pharmaceutical anomaly detection.

This module handles feature selection, feature engineering,
and feature preparation for model training.
"""

import pandas as pd
from typing import Tuple, Dict, Any


def select_features(df: pd.DataFrame) -> list:
    """
    Select appropriate features for Isolation Forest training.
    
    Features are selected based on their numerical nature and relevance to anomaly detection.
    Boolean flags and statistical measures are ideal for isolation forest.
    
    Args:
        df: Input dataframe containing all features
        
    Returns:
        List of selected feature column names
    """
    print("\n" + "="*80)
    print("FEATURE SELECTION FOR ISOLATION FOREST")
    print("="*80)
    
    # SELECTED FEATURES (with rationale)
    selected_features = [
        # Quantity-based anomalies (numerical, high anomaly signal)
        'qty_z_score',
        'qty_deviation_from_mean',
        'Sales Order item qty',
        'qty_trend_slope_lastN',
        'current_month_total_qty',
        'month_rolling_z',            # New numeric month context (not a boolean)
        'order_share_of_month',       # New per-order share context
        
        # Pricing anomalies (robust, contextual)
        'price_z_vs_customer',        # Customer-specific price z
        
        # Boolean anomaly flags (binary, high anomaly signal)
        'is_first_time_cust_material_order',
        'is_rare_material',
        'is_qty_outside_typical_range',
        'is_suspected_duplicate_order',
        'is_unusual_unit_price',
        'is_value_mismatch_price_qty',
        'is_unusual_fulfillment_time',
        'is_order_qty_high_z',        # High order-level z flag
    ]
    
    print("SELECTED FEATURES:")
    for i, feature in enumerate(selected_features, 1):
        if feature in df.columns:
            non_null_count = df[feature].notna().sum()
            unique_values = df[feature].nunique()
            print(f"{i:2d}. {feature:<35} - {non_null_count:>6} values, {unique_values:>4} unique")
        else:
            print(f"{i:2d}. {feature:<35} - NOT FOUND IN DATASET")
    
    print("\n" + "-"*80)
    print("EXCLUDED FEATURES (with rationale):")
    print("-"*80)
    
    excluded_features = [
        # Identifiers (not predictive)
        ('Sales Document Number', 'Primary key - not predictive of anomalies'),
        ('Sales Document Item', 'Item sequence number - not predictive'),
        ('Customer PO number', 'External reference - high cardinality, not suitable for isolation'),
        ('Sold To number', 'Customer ID - high cardinality, would cause overfitting'),
        ('Ship-To Party', 'Shipping location ID - high cardinality identifier'),
        ('Material Number', 'Product ID - high cardinality, covered by is_rare_material flag'),
        
        # Text/Description fields (categorical, high cardinality)
        ('Material Description', 'Free text - not suitable for numerical isolation forest'),
        ('Sales unit', 'Categorical - covered by is_unusual_uom boolean flag'),
        ('historical_common_uom', 'Categorical reference - not directly predictive'),
        
        # Date/Time fields (temporal, not suitable for isolation)
        ('Sales Document Created Date', 'Used for train/test split - not a feature'),
        ('Entry time', 'Time of day - low anomaly signal, would need preprocessing'),
        ('Actual GI Date', 'Delivery date - covered by fulfillment_duration_days'),
        
        # Statistical reference values (metadata, not features)
        ('hist_mean', 'Historical statistic - metadata, not a feature itself'),
        ('hist_std', 'Historical statistic - used to calculate z-score feature'),
        ('p05', 'Historical percentile - used in range calculations'),
        ('p95', 'Historical percentile - used in range calculations'), 
        ('monthly_qty_p05', 'Historical monthly percentile - replaced by rolling context'),
        ('monthly_qty_p95', 'Historical monthly percentile - replaced by rolling context'),
        ('price_p05', 'Historical price percentile - used in range calculations'),
        ('price_p95', 'Historical price percentile - used in range calculations'),
        ('fulfillment_p05', 'Historical fulfillment percentile - used in range calculations'),
        ('fulfillment_p95', 'Historical fulfillment percentile - used in range calculations'),
        ('Unit Price', 'Raw price is scale-dominant; use robust/contextual variants'),
        ('fulfillment_duration_days', 'Raw duration is scale-dominant; use robust flag instead'),
        
        # Deprecated monthly boolean (replaced by rolling z and share)
        ('is_monthly_qty_outside_typical_range', 'Deprecated - do not use boolean at row level'),
        
        # Explanatory field (output, not input)
        ('anomaly_explanation', 'Human-readable explanation - output field, not feature'),
    ]
    
    for feature_name, reason in excluded_features:
        if feature_name in df.columns:
            print(f"× {feature_name:<35} - {reason}")
    
    # Verify all selected features exist in the dataset
    available_features = [f for f in selected_features if f in df.columns]
    missing_features = [f for f in selected_features if f not in df.columns]
    
    if missing_features:
        print(f"\nWarning: {len(missing_features)} selected features not found in dataset:")
        for feature in missing_features:
            print(f"  - {feature}")
    
    print(f"\nFinal feature set: {len(available_features)} features selected for Isolation Forest")
    print("="*80)
    
    return available_features


def prepare_features(
    train_data: pd.DataFrame, 
    test_data: pd.DataFrame, 
    feature_columns: list
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Prepare feature matrices for training and testing.
    Handle missing values and perform basic preprocessing.
    
    Args:
        train_data: Training dataset
        test_data: Test dataset  
        feature_columns: List of feature column names
        
    Returns:
        tuple: (X_train, X_test, feature_info)
    """
    print(f"\nPreparing features for Isolation Forest...")
    
    # Extract feature matrices
    X_train = train_data[feature_columns].copy()
    X_test = test_data[feature_columns].copy()
    
    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    
    # Coerce numeric-like columns to numeric (protects against accidental strings)
    for col in feature_columns:
        if col.startswith('is_'):
            # keep boolean flags as-is
            continue
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
    
    # Handle missing values
    print("\nHandling missing values...")
    missing_info = {}
    
    for col in feature_columns:
        train_missing = X_train[col].isnull().sum()
        test_missing = X_test[col].isnull().sum()
        
        if train_missing > 0 or test_missing > 0:
            missing_info[col] = {'train': train_missing, 'test': test_missing}
            
            # Fill missing values based on data type
            if col.startswith('is_'):
                fill_value = False
                print(f"  {col}: Filling {train_missing + test_missing} missing values with {fill_value}")
            else:
                fill_value = X_train[col].median(skipna=True)
                if pd.isna(fill_value):
                    fill_value = 0.0
                print(f"  {col}: Filling {train_missing + test_missing} missing values with median ({fill_value})")
            
            X_train[col] = X_train[col].fillna(fill_value)
            X_test[col] = X_test[col].fillna(fill_value)
    
    if not missing_info:
        print("  No missing values found.")
    
    # Feature statistics
    print(f"\nFeature statistics:")
    feature_info = {}
    for col in feature_columns:
        if col.startswith('is_'):
            # skip stats for booleans
            continue
        train_mean = X_train[col].mean()
        train_std = X_train[col].std()
        feature_info[col] = {'mean': train_mean, 'std': train_std}
        print(f"  {col:<35}: mean={train_mean:8.3f}, std={train_std:8.3f}")
    
    return X_train, X_test, feature_info