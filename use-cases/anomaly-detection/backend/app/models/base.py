"""
Base model interface for pharmaceutical anomaly detection.

This module defines the abstract base class for anomaly detection models.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Any
import pandas as pd
import numpy as np


class AnomalyDetectionModel(ABC):
    """
    Abstract base class for anomaly detection models.
    """
    
    @abstractmethod
    def train(self, X_train: pd.DataFrame, **kwargs) -> Any:
        """
        Train the anomaly detection model.
        
        Args:
            X_train: Training features
            **kwargs: Additional training parameters
            
        Returns:
            Trained model object
        """
        pass
    
    @abstractmethod
    def predict(self, model: Any, X_test: pd.DataFrame, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate anomaly predictions.
        
        Args:
            model: Trained model object
            X_test: Test features
            **kwargs: Additional prediction parameters
            
        Returns:
            tuple: (anomaly_scores, anomaly_labels)
        """
        pass
    
    @abstractmethod
    def get_model_type(self) -> str:
        """
        Get the model type identifier.
        
        Returns:
            String identifier for the model type
        """
        pass