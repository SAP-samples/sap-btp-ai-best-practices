#!/usr/bin/env python3
"""
Training pipeline for UI-based pharmaceutical anomaly detection.

This script orchestrates model training from the UI with configurable features.

Usage:
    python training_pipeline.py [--backend sklearn] [--contamination auto|0.05] [--shap] [--customer-stratified] [--n-estimators 400] [--max-samples 2048]
    
Environment Variables:
    SELECTED_FEATURES: Comma-separated list of features to use
    N_ESTIMATORS: Number of estimators for Isolation Forest (overridden by --n-estimators)
    MAX_SAMPLES: Max samples per tree (overridden by --max-samples)

Author: Francisco Robledo
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Import all necessary modules from the UI package
from config.cli import parse_arguments, validate_contamination_rate
from config.settings import HANA_AVAILABLE
from utils.directories import generate_results_folder_name, setup_directories
from data.loader import load_and_preprocess_data
from data.features import prepare_features, select_features
from utils.customer_tiers import assign_customer_tiers
from models.hana_model import HanaAnomalyModel
from models.sklearn_model import SklearnAnomalyModel
from models.stratified_model import StratifiedAnomalyModel, StratifiedHanaAnomalyModel
from models.persistence import (
    save_sklearn_model, save_stratified_models, save_hana_model_metadata,
    load_sklearn_model, load_stratified_models, check_model_compatibility,
    find_latest_model_dir
)
from evaluation.metrics import evaluate_model
from reporting.summary import generate_summary_report

# Global variables that will be updated
import config.settings as settings

def get_ui_selected_features():
    """Get features selected from the UI via environment variables"""
    selected_features_env = os.environ.get('SELECTED_FEATURES', '')
    if selected_features_env:
        return selected_features_env.split(',')
    return None

def configure_model_params(args):
    """Configure model parameters from command line arguments or environment variables"""
    # Priority: command line args > environment variables > defaults
    if args.n_estimators is not None:
        n_estimators = args.n_estimators
    else:
        n_estimators = int(os.environ.get('N_ESTIMATORS', 150))
    
    if args.max_samples is not None:
        max_samples = args.max_samples
    else:
        max_samples = os.environ.get('MAX_SAMPLES', 512)
    
    # Convert max_samples to int if not 'auto'
    if max_samples != 'auto':
        max_samples = int(max_samples)
    
    return n_estimators, max_samples


def main():
    """
    Main execution function for pharmaceutical anomaly detection.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Set contamination rate from arguments
    try:
        contamination_rate = validate_contamination_rate(args.contamination)
        settings.CONTAMINATION_RATE = contamination_rate
    except ValueError as e:
        print(f"Error: {e}")
        print("Using auto mode instead.")
        settings.CONTAMINATION_RATE = 'auto'
    
    print("="*80)
    print("PHARMACEUTICAL ANOMALY DETECTION WITH ISOLATION FOREST")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Backend selected: {args.backend.upper()}")
    print(f"Contamination mode: {settings.CONTAMINATION_RATE}")
    print(f"SHAP explanations: {'Enabled' if args.shap else 'Disabled'}")
    print(f"Customer stratification: {'Enabled' if args.customer_stratified else 'Disabled'}")
    print(f"Load saved models: {'Enabled' if args.load_models else 'Disabled'}")
    
    # Validate backend selection
    if args.backend == 'hana' and not HANA_AVAILABLE:
        print(f"\nWARNING: HANA ML backend requested but not available!")
        print("HANA ML dependencies not found. Please install hana-ml package.")
        print("Falling back to scikit-learn backend...")
        args.backend = 'sklearn'
    
    print(f"Using backend: {args.backend.upper()}")
    
    # Generate dynamic results directory for UI (save in ui/results/)
    results_dir = os.path.join("results", generate_results_folder_name(
        backend=args.backend,
        contamination=settings.CONTAMINATION_RATE,
        use_shap=args.shap,
        customer_stratified=args.customer_stratified
    ))
    settings.RESULTS_DIR = results_dir
    print(f"Results will be saved to: {results_dir}")
    
    # Display dataset information
    if args.file:
        print(f"Using custom dataset: {args.file}")
    else:
        print("Using default dataset from settings")
    
    # Setup directories
    setup_directories(results_dir)
    
    try:
        # Load and preprocess data
        df_full, train_data, test_data, feature_columns = load_and_preprocess_data(args.file)
        
        if len(train_data) == 0 or len(test_data) == 0:
            raise ValueError("Insufficient data for train/test split")
        
        # Override feature selection if specified from UI
        ui_selected_features = get_ui_selected_features()
        if ui_selected_features:
            print(f"\n" + "="*80)
            print("USING UI-SELECTED FEATURES")
            print("="*80)
            print(f"Features from UI: {len(ui_selected_features)}")
            for i, feature in enumerate(ui_selected_features, 1):
                print(f"{i:2d}. {feature}")
            
            # Validate features exist in the dataset
            available_features = [f for f in ui_selected_features if f in df_full.columns]
            missing_features = [f for f in ui_selected_features if f not in df_full.columns]
            
            if missing_features:
                print(f"\nWarning: {len(missing_features)} selected features not found in dataset:")
                for feature in missing_features:
                    print(f"  - {feature}")
            
            feature_columns = available_features
            print(f"\nFinal feature set: {len(feature_columns)} features")
        
        # Customer stratification setup
        customer_tiers = None
        if args.customer_stratified:
            customer_tiers = assign_customer_tiers(df_full)
        
        # Prepare features
        X_train, X_test, _ = prepare_features(train_data, test_data, feature_columns)
        
        # Configure model parameters from command line or environment variables
        n_estimators, max_samples = configure_model_params(args)
        print(f"\nUsing model parameters:")
        print(f"  n_estimators: {n_estimators}")
        print(f"  max_samples: {max_samples}")
        if args.n_estimators is not None or args.max_samples is not None:
            print("  (Parameters set via command line arguments)")
        else:
            print("  (Parameters from environment variables or defaults)")
        
        # Initialize models
        hana_model = HanaAnomalyModel()
        sklearn_model = SklearnAnomalyModel()
        stratified_model = StratifiedAnomalyModel()
        
        # Try to load models if requested
        loaded_models = None
        loaded_customer_tiers = None
        
        if args.load_models:
            print(f"\n" + "="*80)
            print("ATTEMPTING TO LOAD SAVED MODELS")
            print("="*80)
            
            # Try to find existing model directory
            if args.customer_stratified:
                pattern = f"anomaly_detection_results_backend_{args.backend}_contamination_auto_customer_stratified"
            else:
                pattern = f"anomaly_detection_results_backend_{args.backend}_contamination_auto"
            
            if args.shap:
                pattern += "_shap"
            
            latest_dir = find_latest_model_dir(pattern)
            
            if latest_dir:
                print(f"Found potential model directory: {latest_dir}")
                try:
                    if args.customer_stratified and args.backend == 'sklearn':
                        loaded_models, loaded_customer_tiers, metadata = load_stratified_models(latest_dir)
                        if check_model_compatibility(metadata, feature_columns, settings.CONTAMINATION_RATE):
                            print("Stratified models loaded successfully and are compatible!")
                            customer_tiers = loaded_customer_tiers  # Use loaded customer tiers
                        else:
                            print("Model compatibility issues found. Will train new models.")
                            loaded_models = None
                            loaded_customer_tiers = None
                    elif args.backend == 'sklearn':
                        loaded_model, metadata = load_sklearn_model(latest_dir)
                        if check_model_compatibility(metadata, feature_columns, settings.CONTAMINATION_RATE):
                            print("Sklearn model loaded successfully and is compatible!")
                            loaded_models = loaded_model
                        else:
                            print("Model compatibility issues found. Will train new model.")
                            loaded_models = None
                    else:
                        print("Model loading not supported for HANA backend (models cannot be serialized).")
                        print("Will train new HANA models.")
                
                except Exception as e:
                    print(f"Error loading models: {e}")
                    print("Will train new models instead.")
                    loaded_models = None
                    loaded_customer_tiers = None
            else:
                print(f"No existing model directory found matching pattern: {pattern}")
                print("Will train new models.")
        
        # Train model based on backend selection and stratification
        if args.customer_stratified and args.backend == 'sklearn':
            # Customer-stratified scikit-learn approach
            if loaded_models is not None:
                print("\nUsing loaded stratified models for predictions...")
                models = loaded_models
                model_type = "sklearn (customer-stratified, loaded)"
            else:
                print("\nTraining new stratified models...")
                models = stratified_model.train(train_data, customer_tiers, feature_columns, n_estimators=n_estimators, max_samples=max_samples, contamination=settings.CONTAMINATION_RATE)
                model_type = stratified_model.get_model_type()
            
            
            # Generate stratified predictions
            anomaly_scores, anomaly_labels, model_assignments = stratified_model.predict(
                models, customer_tiers, test_data, feature_columns)
            
            # For stratified models, we'll evaluate each tier independently
            model = models  # Pass the models dict for stratified evaluation
            
        elif args.customer_stratified and args.backend == 'hana':
            # Customer-stratified HANA ML approach
            if loaded_models is not None:
                print("\\nModel loading not supported for HANA backend (models cannot be serialized).")
                print("Training new stratified HANA models...")
            else:
                print("\\nTraining new stratified HANA models...")
            
            # Use the stratified HANA model
            stratified_hana_model = StratifiedHanaAnomalyModel()
            models = stratified_hana_model.train(train_data, customer_tiers, feature_columns)
            model_type = stratified_hana_model.get_model_type()
            
            # Generate stratified predictions
            anomaly_scores, anomaly_labels, model_assignments = stratified_hana_model.predict(
                models, customer_tiers, test_data, feature_columns)
            
            # For stratified models, we'll evaluate each tier independently
            model = models  # Pass the models dict for stratified evaluation
            
        elif args.backend == 'hana':
            try:
                model_data = hana_model.train(X_train, train_data)
                model_type = hana_model.get_model_type()
                
                # Generate predictions
                anomaly_scores, anomaly_labels = hana_model.predict(model_data, X_test, test_data=test_data)
                
                # Cleanup HANA resources
                try:
                    cc = model_data[1]  # Connection context
                    cc.drop_table("PHARMA_ANOMALY_TRAIN")
                    cc.close()
                except:
                    pass
                    
                model = model_data  # Pass the full tuple for SHAP compatibility
                model_assignments = None
                
            except Exception as e:
                print(f"HANA ML training failed: {e}")
                print("Falling back to scikit-learn...")
                model = sklearn_model.train(X_train, n_estimators=n_estimators, max_samples=max_samples, contamination=settings.CONTAMINATION_RATE)
                model_type = "scikit-learn (fallback)"
                anomaly_scores, anomaly_labels = sklearn_model.predict(model, X_test)
                model_assignments = None
        
        elif args.backend == 'sklearn':
            if loaded_models is not None:
                print("\nUsing loaded sklearn model for predictions...")
                model = loaded_models
                model_type = "sklearn (loaded)"
            else:
                print("\nTraining new sklearn model...")
                model = sklearn_model.train(X_train, n_estimators=n_estimators, max_samples=max_samples, contamination=settings.CONTAMINATION_RATE)
                model_type = sklearn_model.get_model_type()
            
            anomaly_scores, anomaly_labels = sklearn_model.predict(model, X_test)
            model_assignments = None
        
        else:
            raise ValueError(f"Invalid backend: {args.backend}")
        
        # Evaluate results
        results_df = evaluate_model(
            test_data, anomaly_scores, anomaly_labels, feature_columns, 
            model_type, model, X_train, X_test, args.shap, model_assignments
        )
        
        
        # Save models if they were trained (not loaded)
        if not args.load_models or loaded_models is None:
            print(f"\n" + "="*80)
            print("SAVING TRAINED MODELS")
            print("="*80)
            
            training_info = {
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'anomalies_detected': anomaly_labels.sum(),
                'anomaly_rate': anomaly_labels.sum() / len(anomaly_labels)
            }
            
            try:
                if args.customer_stratified and args.backend == 'sklearn':
                    save_stratified_models(
                        models, model_type, results_dir, feature_columns, 
                        settings.CONTAMINATION_RATE, customer_tiers, training_info
                    )
                elif args.customer_stratified and args.backend == 'hana':
                    # For stratified HANA models, save metadata only (models can't be serialized)
                    save_hana_model_metadata(
                        models, model_type, results_dir, feature_columns,
                        settings.CONTAMINATION_RATE, training_info
                    )
                    # Also save customer tiers for stratified HANA
                    import json
                    customer_tiers_file = os.path.join(results_dir, 'models', 'customer_tiers.json')
                    with open(customer_tiers_file, 'w') as f:
                        json.dump(customer_tiers, f, indent=2)
                    print(f"Customer tiers saved to: {customer_tiers_file}")
                elif args.backend == 'sklearn':
                    save_sklearn_model(
                        model, model_type, results_dir, feature_columns,
                        settings.CONTAMINATION_RATE, training_info
                    )
                elif args.backend == 'hana' and 'model_data' in locals():
                    save_hana_model_metadata(
                        model_data, model_type, results_dir, feature_columns,
                        settings.CONTAMINATION_RATE, training_info
                    )
                
                print("Models saved successfully!")
                print(f"To reuse these models, run with --load-models flag")
                
            except Exception as e:
                print(f"Warning: Failed to save models: {e}")
        else:
            print(f"\n" + "="*80)
            print("MODELS WERE LOADED (NO SAVING NEEDED)")
            print("="*80)
        
        # Generate summary report
        generate_summary_report(results_df, feature_columns, model_type, 
                               n_estimators=n_estimators, max_samples=max_samples, 
                               contamination_rate=settings.CONTAMINATION_RATE)
        
        print(f"\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Backend used: {model_type}")
        print(f"Results saved to: {results_dir}/")
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        raise


if __name__ == "__main__":
    main()