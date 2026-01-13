"""
WEB (E-commerce) Channel Predictors.

This module provides predictors specialized for the WEB channel:
- WEBMultiObjectivePredictor: Predicts Sales, AOV, and Orders simultaneously
- WEBPredictor: Facade for WEB channel predictions

Note: WEB channel does not have conversion predictions since there is no
physical foot traffic to measure.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from app.regressor.configs.model_config import ModelConfig, get_web_multi_config
from .base import BaseCatBoostPredictor, PredictionResult


class WEBMultiObjectivePredictor(BaseCatBoostPredictor):
    """
    WEB Multi-Objective Predictor.

    Predicts Sales, AOV, and Orders simultaneously using CatBoost's MultiRMSE loss.
    Automatically excludes B&M-only features (conversion, cannibalization) from training.

    Parameters
    ----------
    config : ModelConfig, optional
        Full model configuration. Defaults to WEB multi-objective config.
    iterations : int, optional
        Number of boosting iterations (used if config not provided).
    learning_rate : float, optional
        Learning rate (used if config not provided).
    depth : int, optional
        Tree depth (used if config not provided).
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        iterations: int = 5000,
        learning_rate: float = 0.05,
        depth: int = 6,
    ):
        if config is None:
            config = get_web_multi_config()
            config.hyperparams.iterations = iterations
            config.hyperparams.learning_rate = learning_rate
            config.hyperparams.depth = depth

        # Force channel to WEB
        config.channel = "WEB"
        super().__init__(config=config)

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None
    ) -> "WEBMultiObjectivePredictor":
        """
        Train the multi-objective model.

        Requires 'label_log_sales', 'label_log_aov', and 'order_count' columns.
        'label_log_orders' is computed on the fly from order_count.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data with features and target labels.
        val_df : pd.DataFrame, optional
            Validation data for early stopping.

        Returns
        -------
        WEBMultiObjectivePredictor
            Self, for method chaining.
        """
        print(f"Training WEBMultiObjectivePredictor on {len(train_df)} samples...")

        # 1. Prepare Targets
        # Filter valid rows (must have Sales and AOV)
        mask = train_df["label_log_sales"].notna() & train_df["label_log_aov"].notna()
        # For Orders, we need order_count > 0
        mask = mask & (train_df["order_count"] > 0)

        train_subset = train_df[mask].copy()

        # Create log orders target
        train_subset["label_log_orders"] = np.log(train_subset["order_count"])

        target_cols = ["label_log_sales", "label_log_aov", "label_log_orders"]
        y_train = train_subset[target_cols].values

        # 2. Prepare Features
        X_train, self.cat_features = self._prepare_data(train_subset)
        self.feature_names = X_train.columns.tolist()

        train_pool = Pool(X_train, y_train, cat_features=self.cat_features)

        # Prepare validation data if provided
        val_pool = None
        if val_df is not None:
            mask_val = (
                val_df["label_log_sales"].notna() &
                val_df["label_log_aov"].notna() &
                (val_df["order_count"] > 0)
            )
            val_subset = val_df[mask_val].copy()
            val_subset["label_log_orders"] = np.log(val_subset["order_count"])

            y_val = val_subset[target_cols].values
            X_val, _ = self._prepare_data(val_subset)
            val_pool = Pool(X_val, y_val, cat_features=self.cat_features)

        # 3. Train Model
        early_stopping = self.config.hyperparams.early_stopping_rounds if val_pool else None

        self.model = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            loss_function="MultiRMSE",
            eval_metric="MultiRMSE",
            random_seed=self.config.hyperparams.random_seed if self.config else 42,
            early_stopping_rounds=early_stopping,
            verbose=self.config.hyperparams.verbose if self.config else 100,
        )

        self.model.fit(train_pool, eval_set=val_pool, use_best_model=bool(val_pool))
        return self

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        """
        Predict Sales, AOV, and Orders.

        Parameters
        ----------
        df : pd.DataFrame
            Input data with features.

        Returns
        -------
        PredictionResult
            Container with log_sales, log_aov, and log_orders predictions.
            Note: logit_conversion is always None for WEB channel.
        """
        X, _ = self._prepare_data(df)

        preds = self.model.predict(X)  # [n_samples, 3]

        log_sales = preds[:, 0]
        log_aov = preds[:, 1]
        log_orders = preds[:, 2]

        return PredictionResult(
            log_sales=log_sales,
            log_aov=log_aov,
            log_orders=log_orders,
            logit_conversion=None,  # WEB has no conversion
            derived_log_orders=log_sales - log_aov,
        )


@dataclass
class WEBPredictionResult:
    """Complete prediction result for WEB channel."""

    # Multi-objective predictions
    log_sales: np.ndarray
    log_aov: np.ndarray
    log_orders: np.ndarray

    # RMSE values (for quantile computation)
    rmse_sales: float = 0.0
    rmse_aov: float = 0.0

    def to_prediction_result(self) -> PredictionResult:
        """Convert to PredictionResult format."""
        return PredictionResult(
            log_sales=self.log_sales,
            log_aov=self.log_aov,
            log_orders=self.log_orders,
            logit_conversion=None,
            derived_log_orders=self.log_sales - self.log_aov,
        )


class WEBPredictor:
    """
    Facade for WEB channel predictions.

    Wraps WEBMultiObjectivePredictor for consistent interface with BMPredictor.

    Parameters
    ----------
    config : ModelConfig, optional
        Configuration for multi-objective model.
    iterations : int, optional
        Default iterations for the model.
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        iterations: int = 5000,
    ):
        self.multi_predictor = WEBMultiObjectivePredictor(
            config=config,
            iterations=iterations,
        )

        # RMSE values (computed during training)
        self.rmse_sales = 0.0
        self.rmse_aov = 0.0

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None
    ) -> "WEBPredictor":
        """
        Train the multi-objective model.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data.
        val_df : pd.DataFrame, optional
            Validation data.

        Returns
        -------
        WEBPredictor
            Self, for method chaining.
        """
        # Train multi-objective model
        self.multi_predictor.fit(train_df, val_df)

        # Compute RMSE on validation set (if provided)
        if val_df is not None:
            self._compute_rmse(val_df)

        return self

    def _compute_rmse(self, df: pd.DataFrame) -> None:
        """Compute RMSE values from validation data."""
        preds = self.multi_predictor.predict(df)

        valid_sales = df["label_log_sales"].notna()
        if valid_sales.any():
            self.rmse_sales = np.sqrt(np.mean(
                (preds.log_sales[valid_sales] - df.loc[valid_sales, "label_log_sales"])**2
            ))

        valid_aov = df["label_log_aov"].notna()
        if valid_aov.any():
            self.rmse_aov = np.sqrt(np.mean(
                (preds.log_aov[valid_aov] - df.loc[valid_aov, "label_log_aov"])**2
            ))

    def predict(self, df: pd.DataFrame) -> WEBPredictionResult:
        """
        Generate WEB predictions.

        Parameters
        ----------
        df : pd.DataFrame
            Input data with features.

        Returns
        -------
        WEBPredictionResult
            Prediction result with log_sales, log_aov, log_orders.
        """
        preds = self.multi_predictor.predict(df)

        return WEBPredictionResult(
            log_sales=preds.log_sales,
            log_aov=preds.log_aov,
            log_orders=preds.log_orders,
            rmse_sales=self.rmse_sales,
            rmse_aov=self.rmse_aov,
        )

    def save_models(self, checkpoint_dir: Union[str, Path]) -> None:
        """
        Save model to checkpoint directory.

        Parameters
        ----------
        checkpoint_dir : str or Path
            Directory to save model checkpoint.
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.multi_predictor.save_model(checkpoint_dir / "web_multi.cbm")

    def load_models(self, checkpoint_dir: Union[str, Path]) -> "WEBPredictor":
        """
        Load model from checkpoint directory.

        Parameters
        ----------
        checkpoint_dir : str or Path
            Directory containing saved model checkpoint.

        Returns
        -------
        WEBPredictor
            Self, for method chaining.
        """
        checkpoint_dir = Path(checkpoint_dir)

        self.multi_predictor = WEBMultiObjectivePredictor.load_model(
            checkpoint_dir / "web_multi.cbm"
        )

        return self

    def set_rmse(self, sales: float, aov: float) -> None:
        """
        Set RMSE values for quantile computation.

        Use this when loading pre-trained models with known RMSE values.

        Parameters
        ----------
        sales : float
            RMSE for sales predictions.
        aov : float
            RMSE for AOV predictions.
        """
        self.rmse_sales = sales
        self.rmse_aov = aov
