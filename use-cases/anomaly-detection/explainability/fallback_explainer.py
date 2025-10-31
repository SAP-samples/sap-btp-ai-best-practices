"""
Fallback explanations module for pharmaceutical anomaly detection.

This module provides statistical deviation-based explanations when
SHAP is not available or fails.
"""

import pandas as pd
from typing import Dict, List
from config.settings import KEY_COL


def create_fallback_explanations(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    results_df: pd.DataFrame, 
    feature_columns: List[str], 
    n_samples: int
) -> Dict:
    """
    Fallback explanation method using statistical deviation analysis for all samples.
    
    Args:
        X_train: Training features for statistical baseline
        X_test: Test features
        results_df: Results with predictions
        feature_columns: Feature column names
        n_samples: Number of samples to process
        
    Returns:
        Dictionary with fallback explanations
    """
    print("Using fallback feature deviation analysis for all samples...")
    
    explanations = []
    
    # Process all samples in results_df
    for idx, row in results_df.iterrows():
        # Check if sample is in test set
        if idx not in X_test.index:
            explanations.append({
                'index': idx,
                'document_number': row.get(KEY_COL, 'Unknown'),
                'anomaly_score': row['anomaly_score'],
                'shap_explanation': "Sample not in test set",
                'rule_explanation': row.get('anomaly_explanation', 'N/A')
            })
            continue
            
        # Calculate feature deviations from training distribution
        feature_deviations = []
        instance = X_test.loc[idx]
        
        for feature in feature_columns:
            if feature in X_train.columns:
                train_mean = X_train[feature].mean()
                train_std = X_train[feature].std()
                
                if train_std > 0:
                    z_score = (instance[feature] - train_mean) / train_std
                    feature_deviations.append({
                        'feature': feature,
                        'value': instance[feature],
                        'z_score': z_score,
                        'contribution': abs(z_score)
                    })
        
        # Sort by contribution
        feature_deviations.sort(key=lambda x: x['contribution'], reverse=True)
        
        # Create explanation text
        top_contributors = feature_deviations[:5]
        explanation_parts = []
        for contrib in top_contributors:
            explanation_parts.append(
                f"{contrib['feature']}: {contrib['value']:.3f} "
                f"(z-score: {contrib['z_score']:.2f})"
            )
        
        explanations.append({
            'index': idx,
            'document_number': row.get(KEY_COL, 'Unknown'),
            'anomaly_score': row['anomaly_score'],
            'shap_explanation': "Statistical deviation: " + "; ".join(explanation_parts),
            'rule_explanation': row.get('anomaly_explanation', 'N/A')
        })
    
    return {'explanations': explanations}