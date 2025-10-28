"""
SHAP explanations module for pharmaceutical anomaly detection.

This module provides SHAP-based explanations for model predictions,
including stratified explanations for customer tiers.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import config.settings as settings
from explainability.text_generator import generate_textual_explanation
from explainability.fallback_explainer import create_fallback_explanations

if settings.SHAP_AVAILABLE:
    import shap


def create_shap_explanations(
    model: Any, 
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    results_df: pd.DataFrame, 
    feature_columns: List[str], 
    model_type: str, 
    n_samples: Optional[int] = None,
    results_dir: str = None
) -> Optional[Dict]:
    """
    Generate SHAP explanations for all test samples to understand feature contributions.
    
    Args:
        model: Trained model (HANA or scikit-learn)
        X_train: Training features  
        X_test: Test features
        results_df: Results with predictions
        feature_columns: Feature column names
        model_type: Type of model used
        n_samples: Legacy parameter, not used
        results_dir: Legacy parameter, not used
        
    Returns:
        Dictionary with SHAP explanations
    """
    if not settings.SHAP_AVAILABLE:
        print("SHAP not available - skipping explanations")
        return None
        
    print(f"\nGenerating SHAP explanations for all {len(X_test)} test samples...")
    print("This may take a few minutes for large datasets...")
    
    try:
        # For HANA models, we need a prediction wrapper since SHAP expects a callable
        if 'hana' in model_type.lower():
            # Create a wrapper function that mimics sklearn's decision_function
            def hana_predict_wrapper(X):
                """Wrapper to make HANA model compatible with SHAP"""
                # Handle different input types from SHAP
                if isinstance(X, np.ndarray):
                    X_df = pd.DataFrame(X, columns=feature_columns)
                elif isinstance(X, pd.DataFrame):
                    X_df = X.copy()
                else:
                    # Convert to DataFrame if other type
                    X_df = pd.DataFrame(X, columns=feature_columns)
                
                # Ensure feature order matches training
                X_df = X_df[feature_columns]
                
                try:
                    # Extract HANA model components - model should be the tuple from HanaAnomalyModel.train()
                    if isinstance(model, tuple) and len(model) >= 4:
                        hana_model, cc, _, model_feature_names = model  # Don't need hdf_train for prediction
                    else:
                        raise ValueError("HANA model should be a tuple (model, connection_context, hdf_train, feature_names)")
                    
                    # Prepare data for HANA prediction
                    hana_test_data = X_df.copy()
                    hana_test_data.insert(0, 'ID', range(len(hana_test_data)))
                    
                    # Create temporary table for SHAP prediction
                    import uuid
                    temp_table_name = f"PHARMA_SHAP_TEMP_{uuid.uuid4().hex[:8].upper()}"
                    
                    try:
                        cc.drop_table(temp_table_name)
                    except:
                        pass
                    
                    # Upload to HANA
                    from hana_ml.dataframe import create_dataframe_from_pandas
                    hdf_temp = create_dataframe_from_pandas(
                        connection_context=cc,
                        pandas_df=hana_test_data,
                        table_name=temp_table_name,
                        force=True,
                        replace=True,
                        primary_key='ID'
                    )
                    
                    # Generate predictions using the trained HANA model
                    results_hdf = hana_model.predict(
                        data=hdf_temp,
                        key='ID',
                        features=model_feature_names
                    )
                    
                    if results_hdf:
                        df_results = results_hdf.collect()
                        
                        # Extract scores - HANA scores are 0-1 range (higher = more anomalous)
                        # Convert to sklearn-like scores (lower = more anomalous) for SHAP consistency
                        hana_scores = df_results['SCORE'].values
                        sklearn_like_scores = -hana_scores  # Invert for sklearn compatibility
                        
                        # Cleanup temporary table
                        try:
                            cc.drop_table(temp_table_name)
                        except:
                            pass
                        
                        return sklearn_like_scores
                    else:
                        raise ValueError("HANA prediction returned no results")
                        
                except Exception as e:
                    print(f"Warning: HANA prediction failed in SHAP wrapper: {e}")
                    
                    # Cleanup temporary table on error
                    try:
                        if 'temp_table_name' in locals() and 'cc' in locals():
                            cc.drop_table(temp_table_name)
                    except:
                        pass
                    
                    # Re-raise the exception since we can't provide fake scores
                    raise Exception(f"HANA SHAP prediction failed: {e}. Cannot generate SHAP explanations.")
                
            prediction_function = hana_predict_wrapper
        else:
            # For scikit-learn, use decision_function directly
            prediction_function = model.decision_function
        
        # Create SHAP explainer
        # Use a sample of training data as background for faster computation
        background_size = min(100, len(X_train))
        background = X_train.sample(n=background_size, random_state=42)
        
        # Ensure data types are compatible with SHAP
        background = background.astype(np.float64)
        # Remove any infinite values
        background = background.replace([np.inf, -np.inf], np.nan)
        background = background.fillna(0)
        
        # Initialize SHAP explainer
        explainer = shap.Explainer(prediction_function, background)
        
        # Prepare all test data for SHAP processing
        X_explain = X_test.copy()
        
        # Ensure data types are compatible with SHAP
        X_explain = X_explain.astype(np.float64)
        # Remove any infinite values
        X_explain = X_explain.replace([np.inf, -np.inf], np.nan)
        X_explain = X_explain.fillna(0)
        
        # For large datasets, process in batches to avoid memory issues
        batch_size = 1000  # Process 1000 samples at a time
        all_shap_values = []
        
        print(f"Processing {len(X_explain)} samples in batches of {batch_size}...")
        
        for i in range(0, len(X_explain), batch_size):
            end_idx = min(i + batch_size, len(X_explain))
            batch = X_explain.iloc[i:end_idx]
            
            print(f"Processing batch {i//batch_size + 1}/{(len(X_explain)-1)//batch_size + 1} (samples {i+1}-{end_idx})...")
            
            # Generate SHAP values for this batch
            batch_shap_values = explainer(batch)
            all_shap_values.append(batch_shap_values)
        
        # Combine all SHAP values
        print("Combining SHAP results...")
        # Concatenate SHAP values from all batches
        if len(all_shap_values) > 1:
            combined_values = np.concatenate([sv.values for sv in all_shap_values], axis=0)
            combined_base_values = np.concatenate([sv.base_values for sv in all_shap_values], axis=0)
            # Create a combined SHAP values object
            shap_values = shap.Explanation(
                values=combined_values,
                base_values=combined_base_values,
                data=X_explain.values,
                feature_names=feature_columns
            )
        else:
            shap_values = all_shap_values[0]
        
        # Create visualizations (only for top anomalies to keep manageable)
        anomalies_df = results_df[results_df['predicted_anomaly'] == 1].copy()
        if len(anomalies_df) > 0:
            # Get top 10 anomalies for visualization
            if 'hana' in model_type.lower():
                top_anomalies = anomalies_df.nlargest(10, 'anomaly_score')
            else:
                top_anomalies = anomalies_df.nsmallest(10, 'anomaly_score')
            
            # SHAP visualizations are no longer generated in the current UI workflow
            pass
        
        # Generate textual explanations for ALL samples
        explanations = []
        print("Generating textual explanations for all samples...")
        
        for i, (idx, row) in enumerate(results_df.iterrows()):
            # Find the position of this sample in the test set
            test_position = list(X_test.index).index(idx) if idx in X_test.index else None
            
            if test_position is not None:
                explanation = generate_textual_explanation(
                    shap.Explanation(
                        values=shap_values.values[test_position],
                        base_values=shap_values.base_values[test_position],
                        data=X_explain.iloc[test_position].values,
                        feature_names=feature_columns
                    ), 
                    X_explain.iloc[test_position], feature_columns, 
                    row['anomaly_score'], row.get('anomaly_explanation', 'N/A')
                )
            else:
                explanation = "Sample not in test set"
            
            explanations.append({
                'index': i,  # Use positional index within the subset
                'original_df_index': idx,  # Keep original DataFrame index for reference
                'document_number': row.get(settings.KEY_COL, 'Unknown'),
                'anomaly_score': row['anomaly_score'],
                'shap_explanation': explanation,
                'rule_explanation': row.get('anomaly_explanation', 'N/A')
            })
        
        print(f"SHAP explanations generated for {len(explanations)} samples")
        return {
            'explanations': explanations,
            'shap_values': shap_values,
            'shap_values': shap_values  # Keep SHAP values for potential future use
        }
        
    except Exception as e:
        print(f"Error generating SHAP explanations: {e}")
        # Fallback to feature deviation analysis
        return create_fallback_explanations(X_train, X_test, results_df, feature_columns, len(results_df))


