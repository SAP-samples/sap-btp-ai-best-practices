"""
Base CatBoost Predictor.

Provides the abstract base class for all CatBoost predictors in the forecasting system.
Handles common functionality like feature preparation, categorical encoding, and model persistence.

Feature lists are imported from model_config.py (single source of truth).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from app.regressor.configs.model_config import (
    ModelConfig,
    FeatureConfig,
    # Single source of truth for feature lists
    CATEGORICAL_FEATURES,
    EXCLUDE_FEATURES,
    BM_ONLY_FEATURES,
    WEB_ONLY_FEATURES,
)


@dataclass
class PredictionResult:
    """Container for prediction results from multi-objective models."""

    log_sales: np.ndarray
    log_aov: np.ndarray
    log_orders: np.ndarray  # Direct prediction
    logit_conversion: Optional[np.ndarray] = None
    derived_log_orders: Optional[np.ndarray] = None  # Sales - AOV

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame construction."""
        result = {
            "pred_log_sales": self.log_sales,
            "pred_log_aov": self.log_aov,
            "pred_log_orders": self.log_orders,
        }
        if self.logit_conversion is not None:
            result["pred_logit_conversion"] = self.logit_conversion
        if self.derived_log_orders is not None:
            result["pred_derived_log_orders"] = self.derived_log_orders
        return result


class BaseCatBoostPredictor(ABC):
    """
    Base class for CatBoost predictors.

    Provides common functionality for:
    - Feature preparation and filtering by channel
    - Categorical feature handling
    - Model persistence (save/load)

    Feature lists are imported from model_config.py (single source of truth).

    Subclasses must implement:
    - fit(): Train the model
    - predict(): Generate predictions
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        iterations: int = 5000,
        learning_rate: float = 0.05,
        depth: int = 6,
        channel: Optional[str] = None,
    ):
        """
        Initialize the predictor.

        Parameters
        ----------
        config : ModelConfig, optional
            Full model configuration. If provided, other parameters are ignored.
        iterations : int, optional
            Number of boosting iterations (used if config not provided).
        learning_rate : float, optional
            Learning rate (used if config not provided).
        depth : int, optional
            Tree depth (used if config not provided).
        channel : str, optional
            Channel type ('B&M' or 'WEB'). Determines which features to exclude.
        """
        if config is not None:
            self.config = config
            self.iterations = config.hyperparams.iterations
            self.learning_rate = config.hyperparams.learning_rate
            self.depth = config.hyperparams.depth
            self.channel = config.channel
            self._feature_config = config.features
        else:
            self.config = None
            self.iterations = iterations
            self.learning_rate = learning_rate
            self.depth = depth
            self.channel = channel
            # Use default FeatureConfig which pulls from canonical constants
            self._feature_config = FeatureConfig()

        self.model: Optional[CatBoostRegressor] = None
        self.feature_names: List[str] = []
        self.cat_features: List[str] = []

    def _get_categorical_features(self) -> List[str]:
        """Get list of categorical features to encode."""
        if self._feature_config is not None:
            return self._feature_config.categorical_features
        # Fallback to canonical constant (should not happen with default FeatureConfig)
        return CATEGORICAL_FEATURES.copy()

    def _get_exclude_features(self) -> List[str]:
        """Get list of features to exclude from training."""
        if self._feature_config is not None:
            return self._feature_config.exclude_features
        # Fallback to canonical constant
        return EXCLUDE_FEATURES.copy()

    def _get_bm_only_features(self) -> List[str]:
        """Get list of B&M-only features."""
        if self._feature_config is not None:
            return self._feature_config.bm_only_features
        # Fallback to canonical constant
        return BM_ONLY_FEATURES.copy()

    def _get_web_only_features(self) -> List[str]:
        """Get list of WEB-only features."""
        if self._feature_config is not None:
            return self._feature_config.web_only_features
        # Fallback to canonical constant
        return WEB_ONLY_FEATURES.copy()

    def _prepare_data(
        self,
        df: pd.DataFrame,
        channel: Optional[str] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare feature matrix and identify categorical columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with features.
        channel : str, optional
            Channel type ('B&M' or 'WEB'). If provided, excludes features
            that are not relevant to this channel:
            - For B&M: excludes WEB_ONLY_FEATURES
            - For WEB: excludes BM_ONLY_FEATURES
            If None, uses self.channel or includes all features.

        Returns
        -------
        Tuple[pd.DataFrame, List[str]]
            Feature matrix X and list of categorical column names.
        """
        # Determine channel for feature filtering
        effective_channel = channel or self.channel

        # Build exclusion list
        exclude_cols = set(self._get_exclude_features())

        if effective_channel == "B&M":
            # B&M model: exclude WEB-only features
            exclude_cols.update(self._get_web_only_features())
        elif effective_channel == "WEB":
            # WEB model: exclude B&M-only features
            exclude_cols.update(self._get_bm_only_features())

        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols].copy()

        # Handle categorical features
        categorical_features = self._get_categorical_features()
        cat_cols = [c for c in categorical_features if c in X.columns]
        for c in cat_cols:
            X[c] = X[c].astype(str)

        # Fill NaNs in numeric features with 0
        numeric_cols = [c for c in X.columns if c not in cat_cols]
        X[numeric_cols] = X[numeric_cols].fillna(0)

        # Reorder columns to match model's expected feature order
        # This ensures consistent column ordering regardless of how data was generated
        if self.model is not None and hasattr(self.model, 'feature_names_'):
            model_features = self.model.feature_names_
            # Only include columns that exist in both X and model's expected features
            ordered_cols = [c for c in model_features if c in X.columns]
            X = X[ordered_cols]
            # Update cat_cols to reflect actual columns in X
            cat_cols = [c for c in cat_cols if c in ordered_cols]

        return X, cat_cols

    def save_model(self, path: Union[str, Path]) -> None:
        """
        Persist underlying CatBoost model to disk.

        Parameters
        ----------
        path : str or Path
            Path to save the model file.

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """
        if self.model is None:
            raise ValueError("Model not fitted; nothing to save.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    @classmethod
    def load_model(cls, path: Union[str, Path]) -> "BaseCatBoostPredictor":
        """
        Load a persisted CatBoost model into a new predictor instance.

        Parameters
        ----------
        path : str or Path
            Path to the saved model file.

        Returns
        -------
        BaseCatBoostPredictor
            New predictor instance with loaded model.
        """
        obj = cls()
        obj.model = CatBoostRegressor()
        obj.model.load_model(str(path))
        return obj

    @abstractmethod
    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None
    ) -> "BaseCatBoostPredictor":
        """
        Train the model.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data with features and labels.
        val_df : pd.DataFrame, optional
            Validation data for early stopping.

        Returns
        -------
        BaseCatBoostPredictor
            Self, for method chaining.
        """
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame):
        """
        Generate predictions.

        Parameters
        ----------
        df : pd.DataFrame
            Input data with features.

        Returns
        -------
        Predictions (type depends on subclass).
        """
        pass

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.

        Returns
        -------
        pd.DataFrame
            DataFrame with feature names and importance scores.

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """
        if self.model is None:
            raise ValueError("Model not fitted; no feature importance available.")

        importance = self.model.get_feature_importance()
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False)
