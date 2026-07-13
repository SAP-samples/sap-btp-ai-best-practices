"""
B&M (Brick & Mortar) Channel Predictors.

This module provides predictors specialized for the B&M channel:
- BMMultiObjectivePredictor: Predicts Sales, AOV, and Orders simultaneously
- BMConversionPredictor: Predicts Conversion (requires foot traffic data)
- BMPredictor: Facade combining both predictors
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from app.regressor.configs.model_config import ModelConfig, get_bm_multi_config, get_bm_conversion_config
from .base import BaseCatBoostPredictor, PredictionResult
from .traffic import TrafficEstimator, TrafficResult, TRAFFIC_ESTIMATION_ENABLED


class BMMultiObjectivePredictor(BaseCatBoostPredictor):
    """
    B&M Multi-Objective Predictor.

    Predicts Sales, AOV, and Orders simultaneously using CatBoost's MultiRMSE loss.
    Automatically excludes WEB-only features from training.

    Parameters
    ----------
    config : ModelConfig, optional
        Full model configuration. Defaults to B&M multi-objective config.
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
            config = get_bm_multi_config()
            config.hyperparams.iterations = iterations
            config.hyperparams.learning_rate = learning_rate
            config.hyperparams.depth = depth

        # Force channel to B&M
        config.channel = "B&M"
        super().__init__(config=config)

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None
    ) -> "BMMultiObjectivePredictor":
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
        BMMultiObjectivePredictor
            Self, for method chaining.
        """
        print(f"Training BMMultiObjectivePredictor on {len(train_df)} samples...")

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
            derived_log_orders=log_sales - log_aov,  # Alternative: Sales / AOV = Orders
        )


class BMConversionPredictor(BaseCatBoostPredictor):
    """
    B&M Conversion Predictor.

    Predicts logit-transformed Conversion rate for B&M stores with foot traffic data.
    Conversion is B&M-only since it requires physical store traffic.

    Parameters
    ----------
    config : ModelConfig, optional
        Full model configuration. Defaults to B&M conversion config.
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
            config = get_bm_conversion_config()
            config.hyperparams.iterations = iterations
            config.hyperparams.learning_rate = learning_rate
            config.hyperparams.depth = depth

        # Force channel to B&M (conversion is B&M-only)
        config.channel = "B&M"
        super().__init__(config=config)

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None
    ) -> "BMConversionPredictor":
        """
        Train the conversion model.

        Requires 'label_logit_conversion' and 'has_traffic_data'==1.
        Only B&M stores with valid traffic data are included.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data with features and target labels.
        val_df : pd.DataFrame, optional
            Validation data for early stopping.

        Returns
        -------
        BMConversionPredictor
            Self, for method chaining.
        """
        print(f"Training BMConversionPredictor on {len(train_df)} samples...")

        # Filter valid conversion rows (B&M with traffic data)
        mask = train_df["label_logit_conversion"].notna() & (train_df["has_traffic_data"] == 1)
        train_subset = train_df[mask].copy()

        y_train = train_subset["label_logit_conversion"].values
        X_train, self.cat_features = self._prepare_data(train_subset)
        self.feature_names = X_train.columns.tolist()

        train_pool = Pool(X_train, y_train, cat_features=self.cat_features)

        # Prepare validation data if provided
        val_pool = None
        if val_df is not None:
            mask_val = val_df["label_logit_conversion"].notna() & (val_df["has_traffic_data"] == 1)
            val_subset = val_df[mask_val].copy()
            y_val = val_subset["label_logit_conversion"].values
            X_val, _ = self._prepare_data(val_subset)
            val_pool = Pool(X_val, y_val, cat_features=self.cat_features)

        # Train Model
        early_stopping = self.config.hyperparams.early_stopping_rounds if val_pool else None

        self.model = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            loss_function="RMSE",  # Single target
            eval_metric="RMSE",
            random_seed=self.config.hyperparams.random_seed if self.config else 42,
            early_stopping_rounds=early_stopping,
            verbose=self.config.hyperparams.verbose if self.config else 100,
        )

        self.model.fit(train_pool, eval_set=val_pool, use_best_model=bool(val_pool))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict Logit Conversion.

        Parameters
        ----------
        df : pd.DataFrame
            Input data with features.

        Returns
        -------
        np.ndarray
            Logit-transformed conversion predictions.
        """
        X, _ = self._prepare_data(df)
        return self.model.predict(X)


@dataclass
class BMPredictionResult:
    """Complete prediction result for B&M channel."""

    # Multi-objective predictions
    log_sales: np.ndarray
    log_aov: np.ndarray
    log_orders: np.ndarray

    # Conversion prediction
    logit_conversion: np.ndarray

    # Traffic predictions (if computed)
    traffic: Optional[TrafficResult] = None

    # RMSE values (for quantile computation)
    rmse_sales: float = 0.0
    rmse_aov: float = 0.0
    rmse_conv: float = 0.0

    def to_prediction_result(self) -> PredictionResult:
        """Convert to PredictionResult format."""
        return PredictionResult(
            log_sales=self.log_sales,
            log_aov=self.log_aov,
            log_orders=self.log_orders,
            logit_conversion=self.logit_conversion,
            derived_log_orders=self.log_sales - self.log_aov,
        )


class BMPredictor:
    """
    Facade for B&M channel predictions.

    Combines MultiObjectivePredictor and ConversionPredictor to provide
    complete B&M forecasts including traffic estimation.

    Parameters
    ----------
    multi_config : ModelConfig, optional
        Configuration for multi-objective model.
    conv_config : ModelConfig, optional
        Configuration for conversion model.
    iterations : int, optional
        Default iterations for both models.
    """

    def __init__(
        self,
        multi_config: Optional[ModelConfig] = None,
        conv_config: Optional[ModelConfig] = None,
        iterations: int = 5000,
    ):
        self.multi_predictor = BMMultiObjectivePredictor(
            config=multi_config,
            iterations=iterations,
        )
        self.conv_predictor = BMConversionPredictor(
            config=conv_config,
            iterations=iterations,
        )
        # Only create traffic estimator if enabled (memory optimization)
        # Controlled by TRAFFIC_ESTIMATION_ENABLED env var
        self.traffic_estimator = TrafficEstimator() if TRAFFIC_ESTIMATION_ENABLED else None

        # RMSE values (computed during training)
        self.rmse_sales = 0.0
        self.rmse_aov = 0.0
        self.rmse_conv = 0.0

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None
    ) -> "BMPredictor":
        """
        Train both multi-objective and conversion models.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data.
        val_df : pd.DataFrame, optional
            Validation data.

        Returns
        -------
        BMPredictor
            Self, for method chaining.
        """
        # Train multi-objective model
        self.multi_predictor.fit(train_df, val_df)

        # Train conversion model
        self.conv_predictor.fit(train_df, val_df)

        # Compute RMSE on validation set (if provided)
        if val_df is not None:
            self._compute_rmse(val_df)

        return self

    def _compute_rmse(self, df: pd.DataFrame) -> None:
        """Compute RMSE values from validation data."""
        # Multi-objective predictions
        preds_multi = self.multi_predictor.predict(df)

        valid_sales = df["label_log_sales"].notna()
        if valid_sales.any():
            self.rmse_sales = np.sqrt(np.mean(
                (preds_multi.log_sales[valid_sales] - df.loc[valid_sales, "label_log_sales"])**2
            ))

        valid_aov = df["label_log_aov"].notna()
        if valid_aov.any():
            self.rmse_aov = np.sqrt(np.mean(
                (preds_multi.log_aov[valid_aov] - df.loc[valid_aov, "label_log_aov"])**2
            ))

        # Conversion predictions
        preds_conv = self.conv_predictor.predict(df)
        valid_conv = df["label_logit_conversion"].notna()
        if valid_conv.any():
            self.rmse_conv = np.sqrt(np.mean(
                (preds_conv[valid_conv] - df.loc[valid_conv, "label_logit_conversion"])**2
            ))

    def predict(
        self,
        df: pd.DataFrame,
        estimate_traffic: bool = True,
    ) -> BMPredictionResult:
        """
        Generate complete B&M predictions.

        Parameters
        ----------
        df : pd.DataFrame
            Input data with features.
        estimate_traffic : bool, optional
            Whether to estimate traffic quantiles. Default True.

        Returns
        -------
        BMPredictionResult
            Complete prediction result including traffic.
        """
        # Multi-objective predictions
        preds_multi = self.multi_predictor.predict(df)

        # Conversion predictions
        preds_conv = self.conv_predictor.predict(df)

        # Traffic estimation (only if enabled and estimator exists)
        traffic = None
        if estimate_traffic and self.rmse_sales > 0 and self.traffic_estimator is not None:
            traffic = self.traffic_estimator.estimate(
                log_sales_pred=preds_multi.log_sales,
                log_aov_pred=preds_multi.log_aov,
                logit_conv_pred=preds_conv,
                sales_rmse=self.rmse_sales,
                aov_rmse=self.rmse_aov,
                conv_rmse=self.rmse_conv,
            )

        return BMPredictionResult(
            log_sales=preds_multi.log_sales,
            log_aov=preds_multi.log_aov,
            log_orders=preds_multi.log_orders,
            logit_conversion=preds_conv,
            traffic=traffic,
            rmse_sales=self.rmse_sales,
            rmse_aov=self.rmse_aov,
            rmse_conv=self.rmse_conv,
        )

    def save_models(self, checkpoint_dir: Union[str, Path]) -> None:
        """
        Save both models to checkpoint directory.

        Parameters
        ----------
        checkpoint_dir : str or Path
            Directory to save model checkpoints.
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.multi_predictor.save_model(checkpoint_dir / "bm_multi.cbm")
        self.conv_predictor.save_model(checkpoint_dir / "bm_conversion.cbm")

    def load_models(self, checkpoint_dir: Union[str, Path]) -> "BMPredictor":
        """
        Load both models from checkpoint directory.

        Parameters
        ----------
        checkpoint_dir : str or Path
            Directory containing saved model checkpoints.

        Returns
        -------
        BMPredictor
            Self, for method chaining.
        """
        checkpoint_dir = Path(checkpoint_dir)

        self.multi_predictor = BMMultiObjectivePredictor.load_model(
            checkpoint_dir / "bm_multi.cbm"
        )
        self.conv_predictor = BMConversionPredictor.load_model(
            checkpoint_dir / "bm_conversion.cbm"
        )

        return self

    def set_rmse(self, sales: float, aov: float, conv: float) -> None:
        """
        Set RMSE values for quantile computation.

        Use this when loading pre-trained models with known RMSE values.

        Parameters
        ----------
        sales : float
            RMSE for sales predictions.
        aov : float
            RMSE for AOV predictions.
        conv : float
            RMSE for conversion predictions.
        """
        self.rmse_sales = sales
        self.rmse_aov = aov
        self.rmse_conv = conv
