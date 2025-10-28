"""
Model loading service for UI on-demand SHAP computation.

This module handles loading pre-trained models and computing SHAP values on demand.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
import streamlit as st

# Local imports
from models.persistence import load_sklearn_model, load_stratified_models, check_model_compatibility
from explainability.shap_explainer import create_shap_explanations
from data.features import prepare_features


class ModelService:
    """Service for loading models and computing SHAP explanations on demand."""
    
    def __init__(self):
        self.models = None
        self.model_type = None
        self.customer_tiers = None
        self.feature_columns = None
        self.is_stratified = False
        self.loaded = False
        self.results_dir = None
        
    def find_best_results_directory(self):
        """Find the best available results directory, supporting both stratified and non-stratified models"""
        from pathlib import Path
        
        results_base = Path("results")
        if not results_base.exists():
            return None
        
        directories = []
        for item in results_base.iterdir():
            if item.is_dir() and item.name.startswith("anomaly_detection_results_backend_sklearn"):
                models_dir = item / "models"
                results_file = item / "anomaly_detection_results.csv"
                if models_dir.exists() and results_file.exists():
                    is_stratified = "customer_stratified" in item.name
                    has_shap = item.name.endswith("_shap")
                    priority = (1 if not is_stratified else 2, 0 if not has_shap else 1)
                    directories.append((priority, item.stat().st_mtime, str(item)))
        
        if directories:
            directories.sort(key=lambda x: (x[0], -x[1]))
            return directories[0][2]
        
        return None

    def load_models(self, results_dir: str = None) -> bool:
        """
        Load pre-trained models from the results directory.
        """
        try:
            if results_dir is None:
                results_dir = self.find_best_results_directory()
                if not results_dir:
                    st.error("No suitable model directories found")
                    return False

            results_path = Path(results_dir)
            if not results_path.exists():
                st.error(f"Results directory not found: {results_dir}")
                return False

            self.results_dir = str(results_path)

            # Try loading single sklearn model first (preferred for current setup)
            try:
                model, metadata = load_sklearn_model(str(results_path))
                if model:
                    self.models = model
                    self.customer_tiers = None
                    self.model_type = "sklearn"
                    self.feature_columns = metadata.get('feature_columns', [])
                    self.is_stratified = False
                    self.loaded = True
                    print(f"[DEBUG] Loaded sklearn model with {len(self.feature_columns)} features")
                    return True
            except Exception as e:
                print(f"[DEBUG] Sklearn model load failed: {e}")

            # Fallback: Attempt to load stratified models
            try:
                models, customer_tiers, metadata = load_stratified_models(str(results_path))
                if models and customer_tiers:
                    self.models = models
                    self.customer_tiers = customer_tiers
                    self.model_type = "sklearn (customer-stratified)"
                    self.feature_columns = metadata.get('feature_columns', [])
                    self.is_stratified = True
                    self.loaded = True
                    print(f"[DEBUG] Loaded stratified models with {len(self.feature_columns)} features")
                    return True
            except Exception as e:
                print(f"[DEBUG] Stratified model load failed: {e}")

            st.error("No compatible models found in the results directory")
            return False

        except Exception as e:
            st.error(f"Error loading models: {e}")
            return False
    
    def compute_shap_for_sample(self, sample_row: pd.Series, features_df: pd.DataFrame = None) -> Optional[str]:
        """
        Compute SHAP explanation for a single sample on demand.
        """
        if not self.loaded:
            st.error("Models not loaded. Please load models first.")
            return None

        try:
            sample_df = pd.DataFrame([sample_row])

            # Derive available features and skip missing ones without mutating global list
            available_features = [f for f in self.feature_columns if f in sample_df.columns]
            missing_features = [f for f in self.feature_columns if f not in sample_df.columns]
            if missing_features:
                print(f"[WARNING] Missing features for SHAP: {missing_features}")
                print(f"[INFO] Proceeding with {len(available_features)} features")

            if not available_features:
                st.error("No overlapping features between model and dataset for SHAP computation")
                return None

            # Background data preparation
            if features_df is not None and len(features_df) > 0:
                background_data = features_df[available_features].sample(n=min(100, len(features_df)), random_state=42)
            else:
                background_data = pd.DataFrame({col: [0] * 50 for col in available_features})

            background_data = background_data.fillna(0)

            # Prepare sample features
            X_sample = sample_df[available_features].copy()
            X_sample = X_sample.fillna(0)

            if self.is_stratified and self.customer_tiers:
                customer_id = str(sample_row.get('Sold To number', ''))
                tier = self.customer_tiers.get(customer_id, 'small')
                model_to_use = self.models.get(tier, self.models.get('global'))
                tier_label = f'{tier} tier model'
            else:
                model_to_use = self.models
                tier_label = self.model_type

            if model_to_use is None:
                st.error("Model not available for SHAP computation")
                return None

            # Generate predictions for context
            scores = model_to_use.decision_function(X_sample)
            labels = model_to_use.predict(X_sample)
            labels = (labels == -1).astype(int)

            results_row = sample_df.copy()
            results_row['anomaly_score'] = scores[0]
            results_row['predicted_anomaly'] = labels[0]

            shap_results = create_shap_explanations(
                model=model_to_use,
                X_train=background_data,
                X_test=X_sample,
                results_df=results_row,
                feature_columns=available_features,
                model_type=tier_label,
                n_samples=1,
                results_dir=self.results_dir
            )

            if shap_results and 'explanations' in shap_results and shap_results['explanations']:
                return shap_results['explanations'][0].get('shap_explanation', '')

            return "SHAP computation completed but no explanation generated"

        except Exception as e:
            st.error(f"Error computing SHAP: {e}")
            return None
    
    def _predict_single_sample(self, X_sample: pd.DataFrame, tier: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate prediction for a single sample.
        
        Args:
            X_sample: Features for the sample
            tier: Customer tier (for stratified models)
            
        Returns:
            tuple: (anomaly_scores, anomaly_labels)
        """
        if self.is_stratified and tier:
            if tier == 'small':
                model_to_use = self.models['global']
            else:
                model_to_use = self.models[tier]
        else:
            model_to_use = self.models
        
        # Use sklearn model predict method
        scores = model_to_use.decision_function(X_sample)
        labels = model_to_use.predict(X_sample)
        labels = (labels == -1).astype(int)  # Convert to 1 for anomaly, 0 for normal
        
        return scores, labels
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        if not self.loaded:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "model_type": self.model_type,
            "is_stratified": self.is_stratified,
            "feature_count": len(self.feature_columns) if self.feature_columns else 0,
            "features": self.feature_columns,
            "results_dir": self.results_dir
        }


# Global model service instance
model_service = ModelService()