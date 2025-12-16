"""
Main visualization dashboard for pharmaceutical anomaly detection.

This module creates comprehensive visualizations for model results.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import config.settings as settings
from visualization.feature_analysis import create_feature_analysis_plot, create_top_features_vs_score_plot


def create_visualizations(
    results_df: pd.DataFrame, 
    anomaly_scores: np.ndarray, 
    anomaly_labels: np.ndarray, 
    feature_columns: List[str], 
    model_type: str
) -> None:
    """
    Create comprehensive visualizations for model results.
    
    Args:
        results_df: Results with predictions
        anomaly_scores: Anomaly scores
        anomaly_labels: Binary predictions
        feature_columns: Feature column names
        model_type: Type of model used ('SAP HANA ML' or 'scikit-learn')
    """
    print(f"\nCreating visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Anomaly Score Distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(anomaly_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Calculate actual threshold used by the model
    if 'hana' in model_type.lower():
        threshold = 0.5 if settings.CONTAMINATION_RATE == 'auto' else np.percentile(anomaly_scores, (1-float(settings.CONTAMINATION_RATE))*100)
    else:
        threshold = 0.0 if settings.CONTAMINATION_RATE == 'auto' else np.percentile(anomaly_scores, settings.CONTAMINATION_RATE*100)
    
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Threshold')
    plt.title('Anomaly Score Distribution')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.legend()
    
    # 2. Feature Importance (prefer SHAP if available)
    plt.subplot(2, 2, 3)
    shap_available = hasattr(results_df, 'attrs') and 'shap_explanations' in results_df.attrs and results_df.attrs['shap_explanations'] is not None
    feature_importance = []
    if shap_available:
        shap_data = results_df.attrs['shap_explanations']
        shap_values = shap_data.get('shap_values', None)
        if shap_values is not None and hasattr(shap_values, 'values'):
            # Global SHAP importance = mean(|SHAP|) per feature
            mean_abs = np.abs(shap_values.values).mean(axis=0)
            feature_importance = list(zip(feature_columns, mean_abs))
    
    if not feature_importance:
        # Fallback to normalized mean difference
        for feature in feature_columns[:15]:  # Top 15 features
            if feature in results_df.columns:
                anomaly_mean = results_df[results_df['predicted_anomaly'] == 1][feature].mean()
                normal_mean = results_df[results_df['predicted_anomaly'] == 0][feature].mean()
                if not np.isnan(anomaly_mean) and not np.isnan(normal_mean):
                    std = results_df[feature].std()
                    if std > 0:
                        importance = abs(anomaly_mean - normal_mean) / std
                        feature_importance.append((feature, importance))
    
    if feature_importance:
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        features, importances = zip(*feature_importance[:10])
        plt.barh(range(len(features)), importances, color='lightcoral')
        plt.yticks(range(len(features)), [f.replace('is_', '').replace('_', ' ') for f in features])
        plt.xlabel('Feature Importance' + (' (Global SHAP)' if shap_available else ' (Normalized Difference)'))
        plt.title('Top Features for Anomaly Detection')
        plt.grid(axis='x', alpha=0.3)
    
    # 4. Anomaly Score vs Top Feature
    plt.subplot(2, 2, 4)
    if feature_importance:
        top_feature = feature_importance[0][0]
        if top_feature in results_df.columns:
            plt.scatter(results_df[top_feature], anomaly_scores, 
                       c=anomaly_labels, cmap='coolwarm', alpha=0.6)
            plt.xlabel(f'{top_feature.replace("_", " ").title()}')
            plt.ylabel('Anomaly Score')
            plt.title(f'Anomaly Score vs {top_feature.replace("_", " ").title()}')
            plt.colorbar(label='Predicted Anomaly')
            plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save with model-specific filename
    safe_model_type = model_type.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('-', '_')
    overview_filename = os.path.join(settings.RESULTS_DIR, f'overview_{safe_model_type}.png')
    plt.savefig(overview_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed feature analysis plot
    create_feature_analysis_plot(results_df, feature_columns, model_type)
    
    # Create top features vs anomaly score plot
    create_top_features_vs_score_plot(results_df, anomaly_scores, anomaly_labels, feature_columns, model_type)