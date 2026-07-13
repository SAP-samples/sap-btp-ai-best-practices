"""
Customer-stratified model implementation for pharmaceutical anomaly detection.

This module implements customer-aware anomaly detection with tier-specific models.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from models.sklearn_model import SklearnAnomalyModel
from models.hana_model import HanaAnomalyModel
from config.settings import CUSTOMER_COL, MIN_SAMPLES_FOR_TRAINING, MEDIUM_CUSTOMER_THRESHOLD
from data.features import prepare_features


class StratifiedAnomalyModel:
    """Customer-stratified Isolation Forest implementation."""
    
    def __init__(self):
        self.sklearn_model = SklearnAnomalyModel()
    
    def train(self, train_data: pd.DataFrame, customer_tiers: Dict[str, str], feature_columns: List[str], n_estimators=None, max_samples=None, contamination=None) -> Dict[str, Any]:
        """
        Train customer-stratified Isolation Forest models using scikit-learn.
        
        Args:
            train_data: Training dataset
            customer_tiers: customer_id -> tier mapping
            feature_columns: Feature column names
            n_estimators: Number of estimators for each model
            max_samples: Max samples per tree for each model
            contamination: Contamination rate for each model
            
        Returns:
            Dictionary containing trained models for each tier
        """
        print(f"\n" + "="*80)
        print("TRAINING CUSTOMER-STRATIFIED ISOLATION FOREST MODELS")
        print("="*80)
        
        # Add customer tier to training data
        train_data = train_data.copy()
        # Convert customer IDs to string to match customer_tiers dictionary keys
        train_data['customer_tier'] = train_data[CUSTOMER_COL].astype(str).map(customer_tiers)
        
        models = {}
        training_stats = {}
        
        # Train global model (baseline for small customers)
        print(f"\n1. Training GLOBAL model (for small customers <{MEDIUM_CUSTOMER_THRESHOLD} orders)...")
        X_train_global, _, _ = prepare_features(train_data, train_data, feature_columns)
        models['global'] = self.sklearn_model.train(X_train_global, n_estimators=n_estimators, max_samples=max_samples, contamination=contamination)
        training_stats['global'] = {'samples': len(X_train_global), 'customers': train_data[CUSTOMER_COL].nunique()}
        
        # Train tier-specific models
        for tier in ['medium', 'large']:
            tier_data = train_data[train_data['customer_tier'] == tier]
            
            if len(tier_data) >= MIN_SAMPLES_FOR_TRAINING:
                print(f"\n2. Training {tier.upper()} tier model...")
                print(f"   Training samples: {len(tier_data):,}")
                print(f"   Unique customers: {tier_data[CUSTOMER_COL].nunique():,}")
                
                X_train_tier, _, _ = prepare_features(tier_data, tier_data, feature_columns)
                models[tier] = self.sklearn_model.train(X_train_tier, n_estimators=n_estimators, max_samples=max_samples, contamination=contamination)
                training_stats[tier] = {'samples': len(X_train_tier), 'customers': tier_data[CUSTOMER_COL].nunique()}
            else:
                print(f"\n   Skipping {tier.upper()} tier - insufficient samples ({len(tier_data)} < {MIN_SAMPLES_FOR_TRAINING})")
                print(f"   {tier.upper()} customers will use GLOBAL model")
                models[tier] = models['global']  # Fallback to global model
                training_stats[tier] = {'samples': 0, 'customers': 0, 'fallback': True}
        
        print(f"\n" + "="*80)
        print("STRATIFIED MODEL TRAINING SUMMARY")
        print("="*80)
        for tier, stats in training_stats.items():
            if stats.get('fallback'):
                print(f"{tier.upper()} tier: FALLBACK to global model (insufficient data)")
            else:
                print(f"{tier.upper()} tier: {stats['samples']:,} samples, {stats['customers']:,} customers")
        
        return models
    
    def predict(self, models: Dict[str, Any], customer_tiers: Dict[str, str], test_data: pd.DataFrame, feature_columns: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate predictions using customer-stratified models.
        
        Args:
            models: Dictionary of trained models by tier
            customer_tiers: customer_id -> tier mapping
            test_data: Test dataset
            feature_columns: Feature column names
            
        Returns:
            tuple: (anomaly_scores, anomaly_labels, model_assignments)
        """
        print(f"\n" + "="*80)
        print("GENERATING CUSTOMER-STRATIFIED PREDICTIONS")
        print("="*80)
        
        # Add customer tier to test data
        test_data = test_data.copy()
        # Convert customer IDs to string to match customer_tiers dictionary keys
        test_data['customer_tier'] = test_data[CUSTOMER_COL].astype(str).map(customer_tiers)
        
        # Initialize results arrays
        anomaly_scores = np.zeros(len(test_data))
        anomaly_labels = np.zeros(len(test_data), dtype=int)
        model_assignments = ['unknown'] * len(test_data)
        
        # Count customers by tier
        tier_counts = test_data['customer_tier'].value_counts(dropna=False)
        print(f"Test data customer distribution:")
        for tier in ['large', 'medium', 'small', None]:
            count = tier_counts.get(tier, 0)
            if tier is None:
                print(f"  New customers (no tier): {count:,} samples")
            else:
                print(f"  {tier.title()} tier customers: {count:,} samples")
        
        # Process each tier
        for tier in ['large', 'medium', 'small']:
            if tier == 'small':
                # Small customers use global model
                tier_mask = (test_data['customer_tier'] == tier) | (test_data['customer_tier'].isna())
                model_to_use = models['global']
                tier_label = 'small/new (global model)'
            else:
                # Medium and large customers use tier-specific models
                tier_mask = test_data['customer_tier'] == tier
                model_to_use = models[tier]
                tier_label = f'{tier} tier model'
            
            if tier_mask.sum() > 0:
                print(f"\nProcessing {tier_mask.sum():,} samples with {tier_label}...")
                
                # Prepare features for this tier
                tier_data = test_data[tier_mask]
                X_test_tier, _, _ = prepare_features(tier_data, tier_data, feature_columns)
                
                # Generate predictions
                tier_scores, tier_labels = self.sklearn_model.predict(model_to_use, X_test_tier)
                
                # Store results
                anomaly_scores[tier_mask] = tier_scores
                anomaly_labels[tier_mask] = tier_labels
                
                # Record which model was used
                if tier == 'small':
                    for i, idx in enumerate(test_data[tier_mask].index):
                        customer_id = test_data.loc[idx, CUSTOMER_COL]
                        list_index = test_data.index.get_loc(idx)
                        if customer_id in customer_tiers:
                            model_assignments[list_index] = 'global (small customer)'
                        else:
                            model_assignments[list_index] = 'global (new customer)'
                else:
                    for i, idx in enumerate(test_data[tier_mask].index):
                        list_index = test_data.index.get_loc(idx)
                        model_assignments[list_index] = f'{tier} tier model'
        
        print(f"\n" + "="*80)
        print("STRATIFIED PREDICTION SUMMARY")
        print("="*80)
        
        # Summary statistics by model type
        model_assignment_counts = {}
        for assignment in model_assignments:
            if assignment not in model_assignment_counts:
                model_assignment_counts[assignment] = 0
            model_assignment_counts[assignment] += 1
        
        total_anomalies = anomaly_labels.sum()
        total_samples = len(anomaly_labels)
        
        print(f"Overall results:")
        print(f"  Total samples: {total_samples:,}")
        print(f"  Total anomalies: {total_anomalies:,} ({total_anomalies/total_samples:.2%})")
        print(f"\\nModel usage breakdown:")
        for model_type, count in model_assignment_counts.items():
            type_anomalies = sum(1 for i, assignment in enumerate(model_assignments) if assignment == model_type and anomaly_labels[i] == 1)
            print(f"  {model_type}: {count:,} samples, {type_anomalies:,} anomalies ({type_anomalies/count if count > 0 else 0:.2%})")
        
        return anomaly_scores, anomaly_labels, model_assignments
    
    def get_model_type(self) -> str:
        """Get the model type identifier."""
        return "sklearn (customer-stratified)"


class StratifiedHanaAnomalyModel:
    """Customer-stratified HANA ML Isolation Forest implementation."""
    
    def __init__(self):
        self.hana_model = HanaAnomalyModel()
    
    def train(self, train_data: pd.DataFrame, customer_tiers: Dict[str, str], feature_columns: List[str]) -> Dict[str, Any]:
        """
        Train customer-stratified Isolation Forest models using HANA ML.
        
        Args:
            train_data: Training dataset
            customer_tiers: customer_id -> tier mapping
            feature_columns: Feature column names
            
        Returns:
            Dictionary containing trained models for each tier
        """
        print(f"\n" + "="*80)
        print("TRAINING CUSTOMER-STRATIFIED HANA ML ISOLATION FOREST MODELS")
        print("="*80)
        
        # Add customer tier to training data
        train_data = train_data.copy()
        # Convert customer IDs to string to match customer_tiers dictionary keys
        train_data['customer_tier'] = train_data[CUSTOMER_COL].astype(str).map(customer_tiers)
        
        models = {}
        training_stats = {}
        
        # Train global model (baseline for small customers)
        print(f"\n1. Training GLOBAL model (for small customers <{MEDIUM_CUSTOMER_THRESHOLD} orders)...")
        X_train_global, _, _ = prepare_features(train_data, train_data, feature_columns)
        models['global'] = self.hana_model.train(X_train_global, train_data)
        training_stats['global'] = {'samples': len(X_train_global), 'customers': train_data[CUSTOMER_COL].nunique()}
        
        # Train tier-specific models
        for tier in ['medium', 'large']:
            tier_data = train_data[train_data['customer_tier'] == tier]
            
            if len(tier_data) >= MIN_SAMPLES_FOR_TRAINING:
                print(f"\n2. Training {tier.upper()} tier model...")
                print(f"   Training samples: {len(tier_data):,}")
                print(f"   Unique customers: {tier_data[CUSTOMER_COL].nunique():,}")
                
                X_train_tier, _, _ = prepare_features(tier_data, tier_data, feature_columns)
                models[tier] = self.hana_model.train(X_train_tier, tier_data)
                training_stats[tier] = {'samples': len(X_train_tier), 'customers': tier_data[CUSTOMER_COL].nunique()}
            else:
                print(f"\n   Skipping {tier.upper()} tier - insufficient samples ({len(tier_data)} < {MIN_SAMPLES_FOR_TRAINING})")
                print(f"   {tier.upper()} customers will use GLOBAL model")
                models[tier] = models['global']  # Fallback to global model
                training_stats[tier] = {'samples': 0, 'customers': 0, 'fallback': True}
        
        print(f"\n" + "="*80)
        print("STRATIFIED HANA ML MODEL TRAINING SUMMARY")
        print("="*80)
        for tier, stats in training_stats.items():
            if stats.get('fallback'):
                print(f"{tier.upper()} tier: FALLBACK to global model (insufficient data)")
            else:
                print(f"{tier.upper()} tier: {stats['samples']:,} samples, {stats['customers']:,} customers")
        
        return models
    
    def predict(self, models: Dict[str, Any], customer_tiers: Dict[str, str], test_data: pd.DataFrame, feature_columns: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate predictions using customer-stratified HANA ML models.
        
        Args:
            models: Dictionary of trained models by tier
            customer_tiers: customer_id -> tier mapping
            test_data: Test dataset
            feature_columns: Feature column names
            
        Returns:
            tuple: (anomaly_scores, anomaly_labels, model_assignments)
        """
        print(f"\n" + "="*80)
        print("GENERATING CUSTOMER-STRATIFIED HANA ML PREDICTIONS")
        print("="*80)
        
        # Add customer tier to test data
        test_data = test_data.copy()
        # Convert customer IDs to string to match customer_tiers dictionary keys
        test_data['customer_tier'] = test_data[CUSTOMER_COL].astype(str).map(customer_tiers)
        
        # Initialize results arrays
        anomaly_scores = np.zeros(len(test_data))
        anomaly_labels = np.zeros(len(test_data), dtype=int)
        model_assignments = ['unknown'] * len(test_data)
        
        # Count customers by tier
        tier_counts = test_data['customer_tier'].value_counts(dropna=False)
        print(f"Test data customer distribution:")
        for tier in ['large', 'medium', 'small', None]:
            count = tier_counts.get(tier, 0)
            if tier is None:
                print(f"  New customers (no tier): {count:,} samples")
            else:
                print(f"  {tier.title()} tier customers: {count:,} samples")
        
        # Process each tier
        for tier in ['large', 'medium', 'small']:
            if tier == 'small':
                # Small customers use global model
                tier_mask = (test_data['customer_tier'] == tier) | (test_data['customer_tier'].isna())
                model_to_use = models['global']
                tier_label = 'small/new (global model)'
            else:
                # Medium and large customers use tier-specific models
                tier_mask = test_data['customer_tier'] == tier
                model_to_use = models[tier]
                tier_label = f'{tier} tier model'
            
            if tier_mask.sum() > 0:
                print(f"\nProcessing {tier_mask.sum():,} samples with {tier_label}...")
                
                # Prepare features for this tier
                tier_data = test_data[tier_mask]
                X_test_tier, _, _ = prepare_features(tier_data, tier_data, feature_columns)
                
                # Generate predictions
                tier_scores, tier_labels = self.hana_model.predict(model_to_use, X_test_tier, test_data=tier_data)
                
                # Store results
                anomaly_scores[tier_mask] = tier_scores
                anomaly_labels[tier_mask] = tier_labels
                
                # Record which model was used
                if tier == 'small':
                    for i, idx in enumerate(test_data[tier_mask].index):
                        customer_id = test_data.loc[idx, CUSTOMER_COL]
                        list_index = test_data.index.get_loc(idx)
                        if customer_id in customer_tiers:
                            model_assignments[list_index] = 'global (small customer)'
                        else:
                            model_assignments[list_index] = 'global (new customer)'
                else:
                    for i, idx in enumerate(test_data[tier_mask].index):
                        list_index = test_data.index.get_loc(idx)
                        model_assignments[list_index] = f'{tier} tier model'
        
        print(f"\n" + "="*80)
        print("STRATIFIED HANA ML PREDICTION SUMMARY")
        print("="*80)
        
        # Summary statistics by model type
        model_assignment_counts = {}
        for assignment in model_assignments:
            if assignment not in model_assignment_counts:
                model_assignment_counts[assignment] = 0
            model_assignment_counts[assignment] += 1
        
        total_anomalies = anomaly_labels.sum()
        total_samples = len(anomaly_labels)
        
        print(f"Overall results:")
        print(f"  Total samples: {total_samples:,}")
        print(f"  Total anomalies: {total_anomalies:,} ({total_anomalies/total_samples:.2%})")
        print(f"\\nModel usage breakdown:")
        for model_type, count in model_assignment_counts.items():
            type_anomalies = sum(1 for i, assignment in enumerate(model_assignments) if assignment == model_type and anomaly_labels[i] == 1)
            print(f"  {model_type}: {count:,} samples, {type_anomalies:,} anomalies ({type_anomalies/count if count > 0 else 0:.2%})")
        
        return anomaly_scores, anomaly_labels, model_assignments
    
    def get_model_type(self) -> str:
        """Get the model type identifier."""
        return "hana (customer-stratified)"