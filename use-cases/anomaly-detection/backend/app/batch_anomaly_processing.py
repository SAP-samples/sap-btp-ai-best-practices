#!/usr/bin/env python3
"""
Batch Anomaly Processing Script for Pharmaceutical Sales Orders - Extended Analysis

This script processes sales orders for a two-week period with extended analysis:
1. Runs ML anomaly detection using pre-trained stratified models
2. Generates SHAP explanations for orders with anomaly scores <= 0.05 (includes borderline cases)
3. For extended analysis cases, runs AI binary classification to catch potential missed anomalies
4. Calculates comprehensive metrics including borderline case discovery rates
5. Provides detailed analysis of ML vs AI anomaly detection performance
"""

import os
import json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import required modules from the UI codebase
from data.features import select_features, prepare_features
from models.stratified_model import StratifiedAnomalyModel
from explainability.shap_explainer import create_shap_explanations
from explainability.ai_explanation_generator import generate_ai_binary_classification_with_images
from visualization.feature_analysis import create_feature_plots
from config.settings import DATE_COL, CUSTOMER_COL, KEY_COL

# Business rules integration (import from project root)
try:
    from business_rules import (
        add_90day_customer_material_check,
        add_weekly_average_checks,
        add_focused_anomaly_explanations,
    )
    _BUSINESS_RULES_AVAILABLE = True
except Exception as e:
    print(f"Warning: business_rules not available: {e}")
    _BUSINESS_RULES_AVAILABLE = False

# Configuration
MODEL_DIR = "results/anomaly_detection_results_backend_sklearn_contamination_0_045_shap_v3/models"
DATA_FILE = "results/anomaly_detection_results_backend_sklearn_contamination_0_045_shap_v3/merged_with_features_selected_ordered.csv"
OUTPUT_DIR = "results/anomaly_detection_results_backend_sklearn_contamination_0_045_shap_v3"
DAYS_TO_PROCESS = None  # Set to None to process full dataset, or specify number of days

def load_models_and_metadata(model_dir: str) -> Tuple[Dict, Dict]:
    """Load pre-trained stratified models and customer tier mappings."""
    print("Loading pre-trained models...")
    
    # Resolve available model files
    global_path = os.path.join(model_dir, "stratified_model_global.joblib")
    medium_path = os.path.join(model_dir, "stratified_model_medium.joblib")
    large_path = os.path.join(model_dir, "stratified_model_large.joblib")
    single_path = os.path.join(model_dir, "sklearn_model.joblib")
    tiers_path = os.path.join(model_dir, "customer_tiers.json")
    
    # Load customer tier mappings when available (optional for single-model flows)
    customer_tiers = {}
    if os.path.exists(tiers_path):
        try:
            with open(tiers_path, 'r') as f:
                customer_tiers = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load customer_tiers.json: {e}. Proceeding without tiers.")
    else:
        print("customer_tiers.json not found; proceeding without tiers (all customers treated as small/new)")
    
    models = None
    # Prefer stratified models when all are present
    if os.path.exists(global_path) and os.path.exists(medium_path) and os.path.exists(large_path):
        models = {
            'global': joblib.load(global_path),
            'medium': joblib.load(medium_path),
            'large': joblib.load(large_path)
        }
        print("Detected stratified model set: global, medium, large")
    # Fallback: single sklearn model
    elif os.path.exists(single_path):
        single_model = joblib.load(single_path)
        # Duplicate across tiers so downstream stratified pipeline can operate unchanged
        models = {
            'global': single_model,
            'medium': single_model,
            'large': single_model
        }
        print("Detected single sklearn model; using it for all tiers (global/medium/large)")
    else:
        # Helpful error with directory listing
        try:
            available = os.listdir(model_dir)
        except Exception:
            available = []
        raise FileNotFoundError(
            "No compatible model files found. Expected stratified_model_*.joblib or sklearn_model.joblib in '"
            f"{model_dir}'. Found: {available}"
        )
    
    print(f"Loaded {len(models)} models and {len(customer_tiers)} customer tier mappings")
    return models, customer_tiers

def load_and_filter_data(data_file: str, days: int = None) -> pd.DataFrame:
    """Load CSV data and filter for specified date range.
    
    Args:
        data_file: Path to the CSV file
        days: Number of days to process. If None, process full dataset.
    """
    print(f"Loading data from: {data_file}")
    
    # Load the processed CSV data directly
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} total records")
    
    # Convert date column to datetime
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    
    if days is None:
        # Process full dataset
        print("Processing full dataset (DAYS_TO_PROCESS is None)")
        print(f"Date range: {df[DATE_COL].min().date()} to {df[DATE_COL].max().date()}")
        print(f"Records to process: {len(df)}")
        return df.copy()
    else:
        # Filter for last N days from the most recent date
        end_date = df[DATE_COL].max()
        start_date = end_date - timedelta(days=days)
        
        # Filter for date range
        mask = (df[DATE_COL] >= start_date) & (df[DATE_COL] <= end_date)
        filtered_df = df[mask].copy()
        
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        print(f"Records in range: {len(filtered_df)}")
        
        return filtered_df

def run_ml_anomaly_detection(data: pd.DataFrame, models: Dict, customer_tiers: Dict, feature_columns: List[str]) -> pd.DataFrame:
    """Run ML anomaly detection using stratified models."""
    print("\nRunning ML anomaly detection...")
    
    # Get the expected feature order from the trained model (fallback to provided features)
    if hasattr(models['global'], 'feature_names_in_'):
        expected_features = models['global'].feature_names_in_
    else:
        expected_features = feature_columns
    print(f"Using model's expected feature order: {len(expected_features)} features")
    
    # Ensure we use the same feature order as the trained model
    model_feature_columns = [feat for feat in expected_features if feat in data.columns]
    
    # Initialize stratified model
    stratified_model = StratifiedAnomalyModel()
    
    # Generate predictions
    anomaly_scores, anomaly_labels, model_assignments = stratified_model.predict(
        models, customer_tiers, data, model_feature_columns
    )
    
    # Add results to dataframe
    result_data = data.copy()
    result_data['anomaly_score'] = anomaly_scores
    result_data['predicted_anomaly'] = anomaly_labels
    result_data['model_used'] = model_assignments
    
    # Add extended analysis flag (score <= 0.05)
    result_data['extended_analysis'] = result_data['anomaly_score'] <= 0.05
    
    ml_anomalies = anomaly_labels.sum()
    extended_cases = (result_data['anomaly_score'] <= 0.05).sum()
    borderline_cases = ((result_data['anomaly_score'] > 0) & (result_data['anomaly_score'] <= 0.05)).sum()
    
    print(f"ML detected {ml_anomalies} anomalies out of {len(data)} orders ({ml_anomalies/len(data):.2%})")
    print(f"Extended analysis will include {extended_cases} cases ({extended_cases/len(data):.2%}):")
    print(f"  - ML anomalies (score <= 0): {ml_anomalies}")
    print(f"  - Borderline cases (0 < score <= 0.05): {borderline_cases}")
    
    return result_data

def generate_shap_explanations_for_extended_analysis(data: pd.DataFrame, models: Dict, feature_columns: List[str]) -> pd.DataFrame:
    """Generate SHAP explanations for orders with anomaly scores <= 0.05 (includes both ML anomalies and borderline cases)."""
    print("\nGenerating SHAP explanations for extended analysis (anomaly_score <= 0.05)...")
    
    # Initialize the shap_explanation column
    data['shap_explanation'] = ''
    
    # Filter for extended analysis: anomaly score <= 0.05
    extended_analysis_cases = data[data['anomaly_score'] <= 0.05]
    
    if len(extended_analysis_cases) == 0:
        print("No cases found with anomaly_score <= 0.05, skipping SHAP generation")
        return data
    
    # Count ML anomalies vs borderline cases
    ml_anomalies = extended_analysis_cases[extended_analysis_cases['predicted_anomaly'] == 1]
    borderline_cases = extended_analysis_cases[extended_analysis_cases['predicted_anomaly'] == 0]
    
    print(f"Generating SHAP explanations for {len(extended_analysis_cases)} cases:")
    print(f"  - ML-detected anomalies (score <= 0): {len(ml_anomalies)}")
    print(f"  - Borderline cases (0 < score <= 0.05): {len(borderline_cases)}")
    
    # For stratified models, use the global model for SHAP consistency
    model = models['global']
    
    # Create sample training data for SHAP background (use first 100 samples from full data)
    X_train_sample = data[feature_columns].head(100).fillna(0)
    X_test_extended = extended_analysis_cases[feature_columns].fillna(0)
    
    try:
        # Generate SHAP explanations for extended analysis cases
        shap_results = create_shap_explanations(
            model=model,
            X_train=X_train_sample,
            X_test=X_test_extended,
            results_df=extended_analysis_cases,
            feature_columns=feature_columns,
            model_type='sklearn'
        )
        
        if shap_results and 'explanations' in shap_results:
            # Map SHAP explanations back to the extended analysis cases
            shap_dict = {}
            for exp in shap_results['explanations']:
                original_idx = exp.get('original_df_index')
                if original_idx is not None:
                    shap_dict[original_idx] = exp.get('shap_explanation', '')
            
            # Update extended analysis records with SHAP explanations
            for idx, explanation in shap_dict.items():
                if idx in data.index:
                    data.at[idx, 'shap_explanation'] = explanation
            
            print(f"Successfully generated SHAP explanations for {len(shap_dict)} extended analysis cases")
        else:
            print("Warning: SHAP explanations could not be generated for extended analysis cases")
    except Exception as e:
        print(f"Error generating SHAP explanations: {e}")
    
    return data

def run_ai_classification_for_extended_analysis(data: pd.DataFrame) -> pd.DataFrame:
    """Run AI binary classification for extended analysis cases (anomaly_score <= 0.05)."""
    print("\nRunning AI binary classification for extended analysis (anomaly_score <= 0.05)...")
    
    # Filter for extended analysis cases
    extended_analysis_cases = data[data['anomaly_score'] <= 0.05].copy()
    
    # Count ML anomalies vs borderline cases
    ml_anomalies = extended_analysis_cases[extended_analysis_cases['predicted_anomaly'] == 1]
    borderline_cases = extended_analysis_cases[extended_analysis_cases['predicted_anomaly'] == 0]
    
    print(f"Processing {len(extended_analysis_cases)} cases for AI classification:")
    print(f"  - ML-detected anomalies (score <= 0): {len(ml_anomalies)}")
    print(f"  - Borderline cases (0 < score <= 0.05): {len(borderline_cases)}")
    
    # Initialize AI results column
    data['ai_anomaly_result'] = pd.NA
    
    # For cases not in extended analysis, set AI result to False (normal)
    non_extended_mask = data['anomaly_score'] > 0.05
    data.loc[non_extended_mask, 'ai_anomaly_result'] = False
    
    # Process each extended analysis case
    success_count = 0
    for idx, (data_idx, row) in enumerate(extended_analysis_cases.iterrows()):
        order_id = row.get(KEY_COL, 'Unknown')
        case_type = "ML anomaly" if row.get('predicted_anomaly') == 1 else "borderline case"
        print(f"Processing {case_type} {idx + 1}/{len(extended_analysis_cases)} - Order: {order_id}")
        
        try:
            # Generate visualization for this order
            # Pass the pre-computed SHAP explanation if available
            pre_computed_shap = row.get('shap_explanation', None)
            
            fig, image_paths = create_feature_plots(
                row=row,
                features_df=data,  # Use all data as historical context
                save_for_ai_analysis=True,
                pre_computed_shap=pre_computed_shap  # Use the SHAP we already computed
            )
            
            # Close the figure to free memory
            import matplotlib.pyplot as plt
            plt.close(fig)
            
            # Run AI binary classification
            ai_result = generate_ai_binary_classification_with_images(
                row=row,
                features_df=data,
                image_paths=image_paths
            )
            
            # Store result
            if ai_result == "True":
                data.at[data_idx, 'ai_anomaly_result'] = True
                success_count += 1
            elif ai_result == "False":
                data.at[data_idx, 'ai_anomaly_result'] = False
                success_count += 1
            else:
                print(f"Warning: Invalid AI result for order {order_id}: {ai_result}")
                
        except Exception as e:
            print(f"Error processing order {order_id}: {str(e)}")
            continue
    
    print(f"Successfully processed {success_count}/{len(extended_analysis_cases)} extended analysis cases with AI")
    return data

def apply_business_rules(data: pd.DataFrame) -> pd.DataFrame:
    """Apply business checks and add summary columns.
    
    Adds two columns:
    - 'Blocked by Business Rules' (bool): True if any business check is triggered
    - 'Business Rules' (str): Human-readable details of triggered rules
    """
    print("\nApplying business rules...")
    work = data.copy()
    
    # Ensure date column is datetime
    if DATE_COL in work.columns:
        work[DATE_COL] = pd.to_datetime(work[DATE_COL], errors='coerce')
    
    if not _BUSINESS_RULES_AVAILABLE:
        # Fallback when business rules module is not importable
        work['Blocked by Business Rules'] = False
        work['Business Rules'] = ''
        return work
    
    try:
        # Compute business rule features and explanations
        work = add_90day_customer_material_check(work)
        work = add_weekly_average_checks(work)
        work = add_focused_anomaly_explanations(work)
    except Exception as e:
        print(f"Error while computing business rules: {e}")
        if 'focused_anomaly_explanation' not in work.columns:
            work['focused_anomaly_explanation'] = ''
    
    # Determine if an order is blocked by any business rule
    rule_flags = [
        'is_outside_cm_90d_threshold',
        'is_outside_cm_weekly_threshold',
        'is_outside_mat_weekly_threshold'
    ]
    
    def compute_blocked(row):
        blocked = False
        for f in rule_flags:
            try:
                blocked = blocked or bool(row.get(f, False))
            except Exception:
                blocked = blocked or False
        return blocked
    
    work['Blocked by Business Rules'] = work.apply(compute_blocked, axis=1)
    work['Business Rules'] = work['focused_anomaly_explanation'].fillna('') if 'focused_anomaly_explanation' in work.columns else ''
    
    # Log summary
    blocked_count = work['Blocked by Business Rules'].sum()
    print(f"Business rules flagged {blocked_count} orders out of {len(work)}")
    
    return work

def calculate_metrics(data: pd.DataFrame) -> Dict:
    """Calculate contamination rates and false positive rates for extended analysis."""
    print("\nCalculating metrics...")
    
    total_orders = len(data)
    
    # ML metrics (traditional anomaly detection)
    ml_anomalies = data['predicted_anomaly'].sum()
    ml_contamination_rate = ml_anomalies / total_orders if total_orders > 0 else 0
    
    # Extended analysis metrics (score <= 0.05)
    extended_analysis_mask = data['anomaly_score'] <= 0.05
    extended_analysis_cases = extended_analysis_mask.sum()
    extended_analysis_rate = extended_analysis_cases / total_orders if total_orders > 0 else 0
    
    # Borderline cases (0 < score <= 0.05)
    borderline_mask = (data['anomaly_score'] > 0) & (data['anomaly_score'] <= 0.05)
    borderline_cases = borderline_mask.sum()
    borderline_rate = borderline_cases / total_orders if total_orders > 0 else 0
    
    # AI metrics (only count definitive results)
    ai_true_mask = data['ai_anomaly_result'] == True
    ai_anomalies = ai_true_mask.sum()
    ai_contamination_rate = ai_anomalies / total_orders if total_orders > 0 else 0
    
    # AI results on borderline cases specifically
    ai_borderline_true_mask = borderline_mask & (data['ai_anomaly_result'] == True)
    ai_borderline_anomalies = ai_borderline_true_mask.sum()
    ai_borderline_discovery_rate = ai_borderline_anomalies / borderline_cases if borderline_cases > 0 else 0
    
    # False positives: ML said anomaly but AI said normal
    false_positive_mask = (data['predicted_anomaly'] == 1) & (data['ai_anomaly_result'] == False)
    false_positives = false_positive_mask.sum()
    false_positive_rate = false_positives / ml_anomalies if ml_anomalies > 0 else 0
    
    metrics = {
        'total_orders': int(total_orders),
        'ml_anomalies': int(ml_anomalies),
        'ml_contamination_rate': float(ml_contamination_rate),
        'extended_analysis_cases': int(extended_analysis_cases),
        'extended_analysis_rate': float(extended_analysis_rate),
        'borderline_cases': int(borderline_cases),
        'borderline_rate': float(borderline_rate),
        'ai_anomalies': int(ai_anomalies),
        'ai_contamination_rate': float(ai_contamination_rate),
        'ai_borderline_anomalies': int(ai_borderline_anomalies),
        'ai_borderline_discovery_rate': float(ai_borderline_discovery_rate),
        'false_positives': int(false_positives),
        'false_positive_rate': float(false_positive_rate)
    }
    
    return metrics

def save_results(data: pd.DataFrame, metrics: Dict, output_dir: str):
    """Save results to CSV and metrics to JSON."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save enhanced dataset
    output_file = os.path.join(output_dir, f"anomaly_detection_results_{timestamp}.csv")
    
    # Select key columns to save
    columns_to_save = [
        KEY_COL, 'Sales Document Item', 'Customer PO number',
        'Material Number', 'Material Description', CUSTOMER_COL, 'Ship-To Party',
        DATE_COL, 'Sales Order item qty', 'Unit Price', 'Order item value',
        'anomaly_score', 'predicted_anomaly', 'extended_analysis', 'ai_anomaly_result',
        'model_used', 'shap_explanation',
        'Blocked by Business Rules', 'Business Rules'
    ]
    
    # Filter to existing columns
    columns_to_save = [col for col in columns_to_save if col in data.columns]
    
    # Save to CSV
    data[columns_to_save].to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Save metrics
    metrics_file = os.path.join(output_dir, f"metrics_{timestamp}.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_file}")
    
    return output_file, metrics_file

def print_summary(metrics: Dict):
    """Print summary of results."""
    print("\n" + "="*70)
    print("BATCH PROCESSING SUMMARY - EXTENDED ANALYSIS")
    print("="*70)
    print(f"Total orders processed: {metrics['total_orders']:,}")
    print(f"\nML Anomaly Detection (Traditional):")
    print(f"  - Anomalies detected: {metrics['ml_anomalies']:,}")
    print(f"  - Contamination rate: {metrics['ml_contamination_rate']:.2%}")
    print(f"\nExtended Analysis (Score <= 0.05):")
    print(f"  - Total cases analyzed: {metrics['extended_analysis_cases']:,}")
    print(f"  - Extended analysis rate: {metrics['extended_analysis_rate']:.2%}")
    print(f"  - Borderline cases (0 < score <= 0.05): {metrics['borderline_cases']:,}")
    print(f"  - Borderline discovery rate: {metrics['borderline_rate']:.2%}")
    print(f"\nAI Verification Results:")
    print(f"  - Total AI-confirmed anomalies: {metrics['ai_anomalies']:,}")
    print(f"  - AI contamination rate: {metrics['ai_contamination_rate']:.2%}")
    print(f"  - AI discoveries in borderline cases: {metrics['ai_borderline_anomalies']:,}")
    print(f"  - AI borderline discovery rate: {metrics['ai_borderline_discovery_rate']:.2%}")
    print(f"\nFalse Positive Analysis:")
    print(f"  - False positives (ML=True, AI=False): {metrics['false_positives']:,}")
    print(f"  - False positive rate: {metrics['false_positive_rate']:.2%}")
    print("="*70)

def main():
    """Main execution function."""
    print("Starting batch anomaly processing with extended analysis (score <= 0.05)...")
    print("This will analyze both ML-detected anomalies and borderline cases for potential missed anomalies.")
    
    # Load and filter data
    data = load_and_filter_data(DATA_FILE, DAYS_TO_PROCESS)
    
    # Get feature columns
    feature_columns = select_features(data)
    print(f"Using {len(feature_columns)} features for anomaly detection")
    
    # Load pre-trained models
    models, customer_tiers = load_models_and_metadata(MODEL_DIR)
    
    # Run ML anomaly detection
    data = run_ml_anomaly_detection(data, models, customer_tiers, feature_columns)
    
    # Generate SHAP explanations for extended analysis (use model feature columns)
    if hasattr(models['global'], 'feature_names_in_'):
        expected_features = models['global'].feature_names_in_
    else:
        expected_features = feature_columns
    model_feature_columns = [feat for feat in expected_features if feat in data.columns]
    data = generate_shap_explanations_for_extended_analysis(data, models, model_feature_columns)
    
    # Run AI classification for extended analysis
    data = run_ai_classification_for_extended_analysis(data)
    
    # Apply business rules and add result columns
    data = apply_business_rules(data)
    
    # Calculate metrics
    metrics = calculate_metrics(data)
    
    # Print summary
    print_summary(metrics)
    
    # Save results
    save_results(data, metrics, OUTPUT_DIR)
    
    print(f"\nExtended batch processing completed successfully!")

if __name__ == "__main__":
    main()