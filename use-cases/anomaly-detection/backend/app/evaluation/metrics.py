"""
Model evaluation and metrics module for pharmaceutical anomaly detection.

This module handles performance evaluation and analysis of anomaly detection results.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Optional, Any
import config.settings as settings
from visualization.dashboard import create_visualizations


def evaluate_model(
    test_data: pd.DataFrame, 
    anomaly_scores: np.ndarray, 
    anomaly_labels: np.ndarray, 
    feature_columns: List[str], 
    model_type: str, 
    model: Optional[Any] = None, 
    X_train: Optional[pd.DataFrame] = None, 
    X_test: Optional[pd.DataFrame] = None, 
    enable_shap: bool = False, 
    model_assignments: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Evaluate model performance and create comprehensive analysis.
    
    Args:
        test_data: Test dataset
        anomaly_scores: Anomaly scores from model
        anomaly_labels: Binary anomaly predictions
        feature_columns: Feature column names
        model_type: Type of model used ('SAP HANA ML' or 'scikit-learn')
        model: Trained model (for SHAP explanations)
        X_train: Training features (for SHAP background)
        X_test: Test features (for SHAP explanations)
        enable_shap: Whether to generate SHAP explanations
        model_assignments: Which model was used for each prediction (for stratified models)
        
    Returns:
        DataFrame with evaluation results
    """
    print(f"\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    # Check if this is a stratified model evaluation (model is a dict of models)
    if isinstance(model, dict) and model_assignments is not None:
        return evaluate_stratified_models(
            test_data, anomaly_scores, anomaly_labels, feature_columns, 
            model_type, model, X_train, X_test, enable_shap, model_assignments
        )
    
    # Create results dataframe
    results_df = test_data.copy()
    results_df['anomaly_score'] = anomaly_scores
    results_df['predicted_anomaly'] = anomaly_labels
    
    # Add model assignment information if using stratified models
    if model_assignments is not None:
        results_df['model_used'] = model_assignments
    
    # Since we don't have ground truth labels, we'll analyze the predicted anomalies
    anomaly_count = anomaly_labels.sum()
    total_count = len(anomaly_labels)
    anomaly_rate = anomaly_count / total_count
    
    print(f"Total samples analyzed: {total_count:,}")
    print(f"Anomalies detected: {anomaly_count:,} ({anomaly_rate:.2%})")
    print(f"Normal samples: {total_count - anomaly_count:,} ({1-anomaly_rate:.2%})")
    
    # Analyze detected anomalies
    if anomaly_count > 0:
        print(f"\nANOMALY ANALYSIS")
        print("-" * 40)
        
        anomaly_samples = results_df[results_df['predicted_anomaly'] == 1]
        
        # Check if we have rule-based anomaly explanations
        if 'anomaly_explanation' in anomaly_samples.columns:
            print("\nRule-based anomaly patterns in detected anomalies:")
            # Filter out NaN values before counting
            explanations = anomaly_samples['anomaly_explanation'].dropna().value_counts()
            for explanation, count in explanations.head(10).items():
                if explanation and str(explanation) != 'nan':
                    print(f"  {count:3d} samples: {explanation}")
        
        # Feature analysis for anomalies
        print(f"\nFeature characteristics of detected anomalies:")
        for feature in feature_columns[:10]:  # Show top 10 features
            if feature in anomaly_samples.columns:
                anomaly_mean = anomaly_samples[feature].mean()
                normal_mean = results_df[results_df['predicted_anomaly'] == 0][feature].mean()
                if not np.isnan(anomaly_mean) and not np.isnan(normal_mean):
                    ratio = anomaly_mean / normal_mean if normal_mean != 0 else float('inf')
                    print(f"  {feature:<35}: anomaly={anomaly_mean:8.3f}, normal={normal_mean:8.3f}, ratio={ratio:6.2f}")
    
    # Create visualizations
    create_visualizations(results_df, anomaly_scores, anomaly_labels, feature_columns, model_type)
    
    # Generate SHAP explanations if enabled and model/training data are available
    if enable_shap and model is not None and X_train is not None and X_test is not None:
        # Import here to avoid circular imports
        from explainability.shap_explainer import create_shap_explanations
        
        # Check if we have stratified models (model_assignments indicates stratified approach)
        if model_assignments is not None:
            # For stratified models, we need the stratified SHAP function
            # This should be called from main() with the proper models and training data
            print("\\nNote: SHAP explanations with stratified models - each prediction uses its corresponding model")
            print("Warning: Stratified SHAP requires models and training data to be passed separately")
            results_df['shap_explanation'] = 'Customer-stratified SHAP: requires models and X_train_dict from main()'
        else:
            # Standard single-model SHAP
            shap_results = create_shap_explanations(model, X_train, X_test, results_df, feature_columns, model_type)
            if shap_results:
                results_df.attrs['shap_explanations'] = shap_results
                
                # Add SHAP explanations as a column to the DataFrame
                results_df['shap_explanation'] = 'No SHAP explanation available'
                
                # Update with actual explanations for anomalies that have them
                if 'explanations' in shap_results:
                    for exp in shap_results['explanations']:
                        idx = exp['index']
                        if idx in results_df.index:
                            results_df.loc[idx, 'shap_explanation'] = exp['shap_explanation']
    else:
        # If SHAP disabled or no model provided, still add the column for consistency  
        if enable_shap:
            print("SHAP explanations requested but model/data not available - skipping")
        results_df['shap_explanation'] = 'SHAP explanations disabled' if not enable_shap else 'No SHAP explanation available'
    
    # Save detailed results (after SHAP explanations are added)
    output_file = os.path.join(settings.RESULTS_DIR, 'anomaly_detection_results.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    return results_df


def evaluate_stratified_models(
    test_data: pd.DataFrame, 
    anomaly_scores: np.ndarray, 
    anomaly_labels: np.ndarray, 
    feature_columns: List[str], 
    model_type: str, 
    models: dict, 
    X_train: Optional[pd.DataFrame] = None, 
    X_test: Optional[pd.DataFrame] = None, 
    enable_shap: bool = False, 
    model_assignments: List[str] = None
) -> pd.DataFrame:
    """
    Evaluate stratified models by evaluating each tier independently.
    
    Args:
        test_data: Test dataset
        anomaly_scores: Combined anomaly scores from all tiers
        anomaly_labels: Combined anomaly predictions from all tiers
        feature_columns: Feature column names
        model_type: Type of model used
        models: Dictionary of models by tier
        X_train: Training features (not used in stratified case)
        X_test: Test features 
        enable_shap: Whether to generate SHAP explanations
        model_assignments: Which model was used for each prediction
        
    Returns:
        Combined DataFrame with evaluation results from all tiers
    """
    print("Evaluating stratified models by tier...")
    
    # Create results dataframe with predictions
    results_df = test_data.copy()
    results_df['anomaly_score'] = anomaly_scores
    results_df['predicted_anomaly'] = anomaly_labels
    results_df['model_used'] = model_assignments
    
    # Analyze overall results first
    anomaly_count = anomaly_labels.sum()
    total_count = len(anomaly_labels)
    anomaly_rate = anomaly_count / total_count
    
    print(f"Total samples analyzed: {total_count:,}")
    print(f"Anomalies detected: {anomaly_count:,} ({anomaly_rate:.2%})")
    print(f"Normal samples: {total_count - anomaly_count:,} ({1-anomaly_rate:.2%})")
    
    # Analyze by tier
    print(f"\nTier-specific analysis:")
    unique_tiers = set(model_assignments)
    for tier in ['large tier model', 'medium tier model', 'global (small customer)']:
        if tier in unique_tiers:
            tier_mask = np.array(model_assignments) == tier
            tier_count = tier_mask.sum()
            tier_anomalies = anomaly_labels[tier_mask].sum()
            tier_rate = tier_anomalies / tier_count if tier_count > 0 else 0
            print(f"  {tier}: {tier_count:,} samples, {tier_anomalies:,} anomalies ({tier_rate:.2%})")
    
    # Analyze detected anomalies
    if anomaly_count > 0:
        print(f"\nANOMALY ANALYSIS")
        print("-" * 40)
        
        anomaly_samples = results_df[results_df['predicted_anomaly'] == 1]
        
        # Check if we have rule-based anomaly explanations
        if 'anomaly_explanation' in anomaly_samples.columns:
            print("\nRule-based anomaly patterns in detected anomalies:")
            explanations = anomaly_samples['anomaly_explanation'].dropna().value_counts()
            for explanation, count in explanations.head(10).items():
                if explanation and str(explanation) != 'nan':
                    print(f"  {count:3d} samples: {explanation}")
        
        # Feature analysis for anomalies
        print(f"\nFeature characteristics of detected anomalies:")
        for feature in feature_columns[:10]:  # Show top 10 features
            if feature in anomaly_samples.columns:
                anomaly_mean = anomaly_samples[feature].mean()
                normal_mean = results_df[results_df['predicted_anomaly'] == 0][feature].mean()
                if not np.isnan(anomaly_mean) and not np.isnan(normal_mean):
                    ratio = anomaly_mean / normal_mean if normal_mean != 0 else float('inf')
                    print(f"  {feature:<35}: anomaly={anomaly_mean:8.3f}, normal={normal_mean:8.3f}, ratio={ratio:6.2f}")
    
    # Create tier-specific visualizations
    print(f"\nCreating tier-specific visualizations...")
    for tier in ['large tier model', 'medium tier model', 'global (small customer)']:
        if tier in unique_tiers:
            tier_mask = np.array(model_assignments) == tier
            tier_scores = anomaly_scores[tier_mask]
            tier_labels = anomaly_labels[tier_mask]
            tier_results = results_df[tier_mask]
            
            print(f"  Creating visualization for {tier}: {len(tier_scores)} samples")
            
            # Create tier-specific visualization with modified model type
            tier_model_type = f"{model_type} ({tier})"
            create_visualizations(tier_results, tier_scores, tier_labels, feature_columns, tier_model_type)
    
    # Also create an overall combined visualization
    print(f"  Creating combined overview visualization...")
    create_visualizations(results_df, anomaly_scores, anomaly_labels, feature_columns, f"{model_type} (Combined)")
    
    # Handle SHAP explanations for stratified models
    if enable_shap and models is not None:
        print(f"\nGenerating tier-specific SHAP explanations...")
        
        # We need to import here to avoid circular imports
        from explainability.shap_explainer import create_shap_explanations
        
        # For each tier, generate SHAP explanations with the appropriate model
        all_explanations = []
        
        for tier in ['large tier model', 'medium tier model', 'global (small customer)']:
            if tier in unique_tiers:
                # Get tier mask and data
                tier_mask = np.array(model_assignments) == tier
                tier_indices = np.where(tier_mask)[0]
                
                if len(tier_indices) == 0:
                    continue
                
                # Get the appropriate model
                if tier == 'large tier model':
                    tier_model = models.get('large')
                elif tier == 'medium tier model':
                    tier_model = models.get('medium')
                else:  # global (small customer)
                    tier_model = models.get('global')
                
                if tier_model is None:
                    print(f"  Warning: No model found for {tier}")
                    continue
                
                # Get tier-specific test data and features  
                tier_X_test = X_test.iloc[tier_indices] if X_test is not None else None
                tier_results = results_df.iloc[tier_indices]
                
                print(f"  Generating SHAP for {tier}: {len(tier_indices)} samples...")
                
                # Use the global X_train as background (could be improved to use tier-specific)
                shap_results = create_shap_explanations(
                    tier_model, X_train, tier_X_test, tier_results, 
                    feature_columns, f"{model_type} ({tier})"
                )
                
                if shap_results and 'explanations' in shap_results:
                    # Map back to original indices
                    for exp in shap_results['explanations']:
                        # Use the positional index to get the original test dataset index
                        original_test_idx = tier_indices[exp['index']]
                        exp['original_test_index'] = original_test_idx
                        # Keep the original DataFrame index for final mapping
                        exp['original_df_index'] = exp.get('original_df_index', exp['index'])
                        all_explanations.append(exp)
        
        # Update results with SHAP explanations
        results_df['shap_explanation'] = 'No SHAP explanation available'
        for exp in all_explanations:
            # Use the original DataFrame index to locate the row in results_df
            original_df_idx = exp['original_df_index']
            if original_df_idx in results_df.index:
                results_df.loc[original_df_idx, 'shap_explanation'] = exp['shap_explanation']
    
    else:
        # Add SHAP column for consistency
        if enable_shap:
            print("SHAP explanations requested but models not available - skipping")
        results_df['shap_explanation'] = 'SHAP explanations disabled' if not enable_shap else 'No SHAP explanation available'
    
    # Save detailed results
    output_file = os.path.join(settings.RESULTS_DIR, 'anomaly_detection_results.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    return results_df