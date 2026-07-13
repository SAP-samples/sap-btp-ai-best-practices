"""
Scikit-learn model implementation for pharmaceutical anomaly detection.

This module implements the Isolation Forest using scikit-learn backend.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Any
from sklearn.ensemble import IsolationForest as SklearnIsolationForest
from models.base import AnomalyDetectionModel
from config.settings import N_ESTIMATORS, MAX_SAMPLES, RANDOM_STATE, CONTAMINATION_RATE


class SklearnAnomalyModel(AnomalyDetectionModel):
    """Scikit-learn Isolation Forest implementation."""
    
    def train(self, X_train: pd.DataFrame, n_estimators=None, max_samples=None, contamination=None, **kwargs) -> Any:
        """
        Train Isolation Forest using scikit-learn (fallback option).
        
        Args:
            X_train: Training features
            n_estimators: Number of estimators (overrides default)
            max_samples: Max samples per tree (overrides default)  
            contamination: Contamination rate (overrides default)
            
        Returns:
            Trained Isolation Forest model
        """
        # Use provided parameters or defaults
        n_est = n_estimators if n_estimators is not None else N_ESTIMATORS
        max_samp = max_samples if max_samples is not None else MAX_SAMPLES
        contam = contamination if contamination is not None else CONTAMINATION_RATE
        
        print(f"\nTraining Isolation Forest with scikit-learn...")
        print(f"Parameters: n_estimators={n_est}, max_samples={max_samp}, contamination={contam}")
        
        # Handle max_samples
        if max_samp == 'auto':
            max_samples_param = 'auto'
        else:
            max_samples_param = min(int(max_samp), len(X_train))
        
        # Configure Isolation Forest
        iforest = SklearnIsolationForest(
            n_estimators=n_est,
            max_samples=max_samples_param,
            contamination=contam,  # 'auto' uses 0.5 as threshold on decision_function
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        # Train model
        iforest.fit(X_train)
        
        print(f"Model training completed with {n_est} trees")
        if contam == 'auto':
            print(f"Auto contamination: will use 0.5 threshold on decision_function scores")
        
        return iforest
    
    def predict(self, model: Any, X_test: pd.DataFrame, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions using scikit-learn model.
        
        Args:
            model: Trained sklearn Isolation Forest model
            X_test: Test features
            
        Returns:
            tuple: (anomaly_scores, anomaly_labels)
            
        Note:
            Scikit-learn Isolation Forest scoring:
            - decision_function() returns normalized path lengths
            - Score range: approximately [-1, +1] 
            - Lower scores = more anomalous (shorter isolation paths)
            - Contamination parameter determines bottom percentile as anomalies
            - Formula: score = 2^(-E(h(x))/c(n)) where E(h(x)) is average path length
        """
        print(f"\nGenerating predictions with scikit-learn...")
        
        # Get anomaly scores (lower = more anomalous)
        anomaly_scores = model.decision_function(X_test)
        
        # Get binary predictions (-1 for anomaly, 1 for normal)
        predictions = model.predict(X_test)
        anomaly_labels = (predictions == -1).astype(int)  # Convert to 0/1
        
        print(f"Score range: [{anomaly_scores.min():.3f}, {anomaly_scores.max():.3f}]")
        print(f"Detected {anomaly_labels.sum()} anomalies out of {len(X_test)} samples ({anomaly_labels.mean():.2%})")
        
        if CONTAMINATION_RATE == 'auto':
            print(f"Auto-detected contamination rate: {anomaly_labels.mean():.2%}")
            print(f"Anomaly threshold: score ≤ 0.0 (auto mode)")
        else:
            print(f"Anomaly threshold: score ≤ {np.percentile(anomaly_scores, CONTAMINATION_RATE*100):.3f}")
        
        return anomaly_scores, anomaly_labels
    
    def get_model_type(self) -> str:
        """Get the model type identifier."""
        return "sklearn"