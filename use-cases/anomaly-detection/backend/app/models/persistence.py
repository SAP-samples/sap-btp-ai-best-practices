"""
Model persistence module for pharmaceutical anomaly detection.

This module handles saving and loading of trained models for both sklearn and HANA backends.
"""

import os
import json
import pickle
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import config.settings as settings


def save_sklearn_model(model: Any, model_type: str, results_dir: str, feature_columns: List[str], 
                      contamination_rate: Any, training_info: Optional[Dict] = None) -> str:
    """
    Save sklearn model to disk.
    
    Args:
        model: Trained sklearn model
        model_type: Type of model (e.g., 'scikit-learn')
        results_dir: Directory to save model in
        feature_columns: List of feature column names
        contamination_rate: Contamination rate used for training
        training_info: Additional training metadata
        
    Returns:
        Path to saved model file
    """
    model_dir = os.path.join(results_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the actual model
    model_file = os.path.join(model_dir, 'sklearn_model.joblib')
    joblib.dump(model, model_file)
    
    # Save metadata
    metadata = {
        'model_type': model_type,
        'feature_columns': feature_columns,
        'contamination_rate': contamination_rate,
        'n_features': len(feature_columns),
        'saved_at': datetime.now().isoformat(),
        'sklearn_params': {
            'contamination': getattr(model, 'contamination', None),
            'n_estimators': getattr(model, 'n_estimators', None),
            'max_samples': getattr(model, 'max_samples', None),
            'random_state': getattr(model, 'random_state', None)
        }
    }
    
    if training_info:
        # Convert any numpy types to Python native types for JSON serialization
        serializable_training_info = {}
        for key, value in training_info.items():
            if isinstance(value, (np.integer, np.floating, np.ndarray)):
                if hasattr(value, 'item'):
                    serializable_training_info[key] = value.item()
                else:
                    serializable_training_info[key] = value.tolist() if hasattr(value, 'tolist') else value
            else:
                serializable_training_info[key] = value
        metadata.update(serializable_training_info)
    
    metadata_file = os.path.join(model_dir, 'sklearn_model_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Sklearn model saved to: {model_file}")
    print(f"Model metadata saved to: {metadata_file}")
    
    return model_file


def save_stratified_models(models: Dict[str, Any], model_type: str, results_dir: str, 
                          feature_columns: List[str], contamination_rate: Any,
                          customer_tiers: Dict[str, str], training_info: Optional[Dict] = None) -> str:
    """
    Save stratified models (multiple models for different customer tiers).
    
    Args:
        models: Dictionary of trained models by tier
        model_type: Type of model (e.g., 'sklearn (customer-stratified)')
        results_dir: Directory to save models in
        feature_columns: List of feature column names
        contamination_rate: Contamination rate used for training
        customer_tiers: Customer tier assignments
        training_info: Additional training metadata
        
    Returns:
        Path to saved models directory
    """
    model_dir = os.path.join(results_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Save each tier model
    saved_models = {}
    for tier, model in models.items():
        if model is not None:
            model_file = os.path.join(model_dir, f'stratified_model_{tier}.joblib')
            joblib.dump(model, model_file)
            saved_models[tier] = model_file
            print(f"Stratified model ({tier}) saved to: {model_file}")
    
    # Save customer tier assignments
    tiers_file = os.path.join(model_dir, 'customer_tiers.json')
    with open(tiers_file, 'w') as f:
        json.dump(customer_tiers, f, indent=2)
    
    # Save stratified metadata
    metadata = {
        'model_type': model_type,
        'feature_columns': feature_columns,
        'contamination_rate': contamination_rate,
        'n_features': len(feature_columns),
        'saved_at': datetime.now().isoformat(),
        'saved_models': saved_models,
        'customer_tiers_file': tiers_file,
        'tier_counts': {tier: len([c for c, t in customer_tiers.items() if t == tier]) 
                       for tier in ['small', 'medium', 'large']},
        'sklearn_params': {}
    }
    
    # Extract parameters from first available model
    for tier, model in models.items():
        if model is not None:
            metadata['sklearn_params'] = {
                'contamination': getattr(model, 'contamination', None),
                'n_estimators': getattr(model, 'n_estimators', None),
                'max_samples': getattr(model, 'max_samples', None),
                'random_state': getattr(model, 'random_state', None)
            }
            break
    
    if training_info:
        # Convert any numpy types to Python native types for JSON serialization
        serializable_training_info = {}
        for key, value in training_info.items():
            if isinstance(value, (np.integer, np.floating, np.ndarray)):
                if hasattr(value, 'item'):
                    serializable_training_info[key] = value.item()
                else:
                    serializable_training_info[key] = value.tolist() if hasattr(value, 'tolist') else value
            else:
                serializable_training_info[key] = value
        metadata.update(serializable_training_info)
    
    metadata_file = os.path.join(model_dir, 'stratified_models_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Stratified models metadata saved to: {metadata_file}")
    print(f"Customer tiers saved to: {tiers_file}")
    
    return model_dir


def save_hana_model_metadata(model_data: Tuple, model_type: str, results_dir: str, 
                           feature_columns: List[str], contamination_rate: Any,
                           training_info: Optional[Dict] = None) -> str:
    """
    Save HANA model metadata (cannot save actual HANA model objects).
    
    Args:
        model_data: HANA model tuple (model, connection_context, hdf_train, feature_names)
        model_type: Type of model (e.g., 'SAP HANA ML')
        results_dir: Directory to save metadata in
        feature_columns: List of feature column names
        contamination_rate: Contamination rate used for training
        training_info: Additional training metadata
        
    Returns:
        Path to saved metadata file
    """
    model_dir = os.path.join(results_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Extract model information
    model, cc, hdf_train, feature_names = model_data
    
    # Save metadata (cannot serialize HANA objects)
    metadata = {
        'model_type': model_type,
        'feature_columns': feature_columns,
        'contamination_rate': contamination_rate,
        'n_features': len(feature_columns),
        'saved_at': datetime.now().isoformat(),
        'hana_info': {
            'model_name': getattr(model, 'model_name', None),
            'training_table': getattr(hdf_train, 'name', None) if hdf_train else None,
            'feature_names': feature_names,
            'warning': 'HANA models cannot be serialized. Re-training required for reuse.'
        }
    }
    
    if training_info:
        # Convert any numpy types to Python native types for JSON serialization
        serializable_training_info = {}
        for key, value in training_info.items():
            if isinstance(value, (np.integer, np.floating, np.ndarray)):
                if hasattr(value, 'item'):
                    serializable_training_info[key] = value.item()
                else:
                    serializable_training_info[key] = value.tolist() if hasattr(value, 'tolist') else value
            else:
                serializable_training_info[key] = value
        metadata.update(serializable_training_info)
    
    metadata_file = os.path.join(model_dir, 'hana_model_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"HANA model metadata saved to: {metadata_file}")
    print("Note: HANA models cannot be serialized and saved. Only metadata is saved.")
    
    return metadata_file


def load_sklearn_model(results_dir: str) -> Tuple[Any, Dict]:
    """
    Load sklearn model from disk.
    
    Args:
        results_dir: Directory containing saved model
        
    Returns:
        Tuple of (model, metadata)
    """
    model_dir = os.path.join(results_dir, 'models')
    
    # Load metadata first
    metadata_file = os.path.join(model_dir, 'sklearn_model_metadata.json')
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Model metadata not found: {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Load the actual model
    model_file = os.path.join(model_dir, 'sklearn_model.joblib')
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    model = joblib.load(model_file)
    
    print(f"Sklearn model loaded from: {model_file}")
    print(f"Model trained at: {metadata.get('saved_at', 'Unknown')}")
    print(f"Features: {metadata.get('n_features', 'Unknown')}")
    
    return model, metadata


def load_stratified_models(results_dir: str) -> Tuple[Dict[str, Any], Dict[str, str], Dict]:
    """
    Load stratified models from disk.
    
    Args:
        results_dir: Directory containing saved models
        
    Returns:
        Tuple of (models_dict, customer_tiers, metadata)
    """
    model_dir = os.path.join(results_dir, 'models')
    
    # Load metadata first
    metadata_file = os.path.join(model_dir, 'stratified_models_metadata.json')
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Stratified models metadata not found: {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Load customer tiers
    tiers_file = os.path.join(model_dir, 'customer_tiers.json')
    if not os.path.exists(tiers_file):
        raise FileNotFoundError(f"Customer tiers file not found: {tiers_file}")
    
    with open(tiers_file, 'r') as f:
        customer_tiers = json.load(f)
    
    # Load each tier model
    models = {}
    saved_models = metadata.get('saved_models', {})
    
    for tier, model_file in saved_models.items():
        if os.path.exists(model_file):
            models[tier] = joblib.load(model_file)
            print(f"Stratified model ({tier}) loaded from: {model_file}")
        else:
            print(f"Warning: Model file not found for tier {tier}: {model_file}")
    
    print(f"Stratified models trained at: {metadata.get('saved_at', 'Unknown')}")
    print(f"Features: {metadata.get('n_features', 'Unknown')}")
    print(f"Customer tiers loaded: {len(customer_tiers)} customers")
    
    return models, customer_tiers, metadata


def check_model_compatibility(metadata: Dict, current_feature_columns: List[str], 
                            current_contamination: Any) -> bool:
    """
    Check if saved model is compatible with current configuration.
    
    Args:
        metadata: Loaded model metadata
        current_feature_columns: Current feature columns
        current_contamination: Current contamination rate
        
    Returns:
        True if compatible, False otherwise
    """
    # Check feature compatibility
    saved_features = metadata.get('feature_columns', [])
    if saved_features != current_feature_columns:
        print(f"Warning: Feature mismatch!")
        print(f"  Saved model features: {len(saved_features)}")
        print(f"  Current features: {len(current_feature_columns)}")
        print(f"  Missing from current: {set(saved_features) - set(current_feature_columns)}")
        print(f"  New in current: {set(current_feature_columns) - set(saved_features)}")
        return False
    
    # Check contamination rate compatibility (warning only)
    saved_contamination = metadata.get('contamination_rate')
    if saved_contamination != current_contamination:
        print(f"Warning: Contamination rate mismatch!")
        print(f"  Saved model: {saved_contamination}")
        print(f"  Current: {current_contamination}")
        print("  This may affect model performance but is not critical.")
    
    return True


def find_latest_model_dir(base_pattern: str) -> Optional[str]:
    """
    Find the most recent model directory matching a pattern.
    
    Args:
        base_pattern: Base pattern to match (e.g., 'anomaly_detection_results_backend_sklearn')
        
    Returns:
        Path to most recent matching directory or None
    """
    current_dir = os.getcwd()
    matching_dirs = []
    
    for item in os.listdir(current_dir):
        if os.path.isdir(item) and item.startswith(base_pattern):
            model_dir = os.path.join(item, 'models')
            if os.path.exists(model_dir):
                matching_dirs.append(item)
    
    if not matching_dirs:
        return None
    
    # Sort by modification time (most recent first)
    matching_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return matching_dirs[0]