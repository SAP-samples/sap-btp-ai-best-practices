"""
Surrogate Explainability Model.

Trains a lightweight Model A (business levers only) to approximate Model B predictions,
enabling SHAP-based interpretation of what drives forecasts.

The surrogate approach separates:
- Model B: Full autoregressive features for maximum accuracy
- Model A (Surrogate): Business-actionable features only for interpretability

SHAP analysis on the surrogate reveals which actionable levers drive Model B's predictions.

Memory Optimization Notes:
- SHAP computation can be batched to manage memory for large DataFrames
- Batch size configurable via SHAP_BATCH_SIZE environment variable
- Per-horizon SHAP computation can be disabled via COMPUTE_PER_HORIZON_SHAP
"""

import gc
import json
import os
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Union

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import r2_score

# Memory-efficient defaults for SHAP computation
SHAP_BATCH_SIZE = int(os.getenv("SHAP_BATCH_SIZE", "500"))
SHAP_ENABLED = os.getenv("SHAP_ENABLED", "true").lower() == "true"
COMPUTE_PER_HORIZON_SHAP = os.getenv("COMPUTE_PER_HORIZON_SHAP", "true").lower() == "true"

from app.regressor.configs.model_config import ModelConfig, get_surrogate_config
from app.regressor.features.model_views import (
    MODEL_A_FEATURES,
    MODEL_A_BM_ONLY_FEATURES,
)


class SurrogateExplainer:
    """
    Surrogate model for SHAP-based explainability.

    Trains on Model A features (business levers) to predict Model B outputs,
    enabling interpretation of what drives the production model's forecasts.

    Attributes
    ----------
    model : CatBoostRegressor
        The trained surrogate model.
    feature_names : List[str]
        Names of features used in training.
    r2_score : float
        R2 score measuring how well surrogate fits Model B predictions.
    """

    # Categorical features for Model A
    CATEGORICAL_FEATURES = [
        "is_outlet", "is_comp_store",
        "is_new_store", "is_holiday", "is_xmas_window",
        "is_black_friday_window", "is_pre_holiday_1wk",
        "is_pre_holiday_2wk", "is_pre_holiday_3wk"
    ]

    def __init__(self, config: Optional[ModelConfig] = None, channel: str = "B&M"):
        """
        Initialize the surrogate explainer.

        Parameters
        ----------
        config : ModelConfig, optional
            Model configuration. Defaults to surrogate-specific config.
        channel : str, optional
            Channel for channel-specific feature selection ("B&M" or "WEB").
            WEB channel excludes B&M-only physical store features.
            Default is "B&M".
        """
        if config is None:
            config = get_surrogate_config()

        self.config = config
        self.channel = channel.upper()
        self.model: Optional[CatBoostRegressor] = None
        self.feature_names: List[str] = []
        self.r2_score_value: float = 0.0
        self._cat_cols_cached: List[str] = []
        self._missing_feature_cache: set[str] = set()

    def _allowed_features_for_channel(self) -> List[str]:
        """
        Return the ordered Model A allow-list for the configured channel.

        Model A features intentionally include certain lag/roll metrics for
        actionable levers (e.g., pct_omni_channel_lag_1). Those are retained,
        while physical-store-only fields are removed for WEB.
        """
        allowed = list(MODEL_A_FEATURES)
        if self.channel == "WEB":
            allowed = [f for f in allowed if f not in MODEL_A_BM_ONLY_FEATURES]
        return allowed

    def _log_missing_features(self, missing: List[str]) -> None:
        """Warn once per newly-missing feature."""
        if not missing:
            return
        new_missing = set(missing) - self._missing_feature_cache
        if not new_missing:
            return
        self._missing_feature_cache.update(new_missing)
        warnings.warn(
            f"SurrogateExplainer: Missing expected Model A features for {self.channel}: "
            f"{sorted(new_missing)}",
            UserWarning,
        )

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features (Model A only - business levers).

        Applies channel-specific exclusions:
        - B&M: Uses all Model A features
        - WEB: Excludes B&M-only physical store features (merchandising_sf,
          cannibalization_pressure, min_dist_new_store_km, etc.)

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.

        Returns
        -------
        pd.DataFrame
            Prepared feature matrix with only actionable business levers.
        """
        # If model has been loaded/trained, use saved feature order for consistency
        if self.feature_names:
            # Use saved feature order, only include columns that exist in input
            feature_cols = [c for c in self.feature_names if c in df.columns]
            missing = [c for c in self.feature_names if c not in df.columns]
        else:
            allowed = self._allowed_features_for_channel()
            feature_cols = [c for c in allowed if c in df.columns]
            missing = [c for c in allowed if c not in df.columns]

        self._log_missing_features(missing)

        if not feature_cols:
            raise ValueError(
                "No valid Model A features found in dataframe. Ensure model_a.csv "
                "was generated with actionable levers."
            )

        X = df[feature_cols].copy()

        # Handle categorical features - convert to string for CatBoost
        cat_cols = [c for c in self.CATEGORICAL_FEATURES if c in X.columns]
        for c in cat_cols:
            X[c] = X[c].astype(str)

        # Fill NaNs in numeric features with 0
        num_cols = [c for c in X.columns if c not in cat_cols]
        X[num_cols] = X[num_cols].fillna(0)

        return X

    def fit(
        self,
        model_a_df: pd.DataFrame,
        model_b_predictions: np.ndarray
    ) -> "SurrogateExplainer":
        """
        Fit surrogate model to approximate Model B predictions.

        Parameters
        ----------
        model_a_df : pd.DataFrame
            Model A features (business levers).
        model_b_predictions : np.ndarray
            Model B predictions to fit against.

        Returns
        -------
        SurrogateExplainer
            Self, for method chaining.
        """
        print("Training Surrogate Model for Explainability...")

        X = self._prepare_data(model_a_df)
        self.feature_names = X.columns.tolist()
        cat_cols = [c for c in self.CATEGORICAL_FEATURES if c in X.columns]
        self._cat_cols_cached = cat_cols

        train_pool = Pool(X, model_b_predictions, cat_features=cat_cols)

        # High depth/iterations to overfit and match Model B decision boundary closely
        hp = self.config.hyperparams
        self.model = CatBoostRegressor(
            iterations=hp.iterations,
            learning_rate=hp.learning_rate,
            depth=hp.depth,
            loss_function="RMSE",
            verbose=hp.verbose,
            random_seed=hp.random_seed,
        )

        self.model.fit(train_pool)

        # Calculate R2 (Goodness of Fit to Model B)
        surrogate_preds = self.model.predict(train_pool)
        self.r2_score_value = r2_score(model_b_predictions, surrogate_preds)
        print(f"Surrogate Model R2 (Fit to Model B): {self.r2_score_value:.4f}")

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the surrogate model.

        Parameters
        ----------
        df : pd.DataFrame
            Input data with Model A features.

        Returns
        -------
        np.ndarray
            Surrogate model predictions.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        X = self._prepare_data(df)
        return self.model.predict(X)

    def save_model(
        self,
        model_path: Union[str, Path],
        meta_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Persist surrogate model and metadata.

        Parameters
        ----------
        model_path : str or Path
            Path to save the CatBoost model.
        meta_path : str or Path, optional
            Path to save metadata JSON. Defaults to model_path + ".meta.json".
        """
        if self.model is None:
            raise ValueError("Surrogate model is not trained.")

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(model_path))

        # Save metadata
        meta = {
            "feature_names": self.feature_names,
            "r2_score": self.r2_score_value,
            "cat_cols": self._cat_cols_cached,
            "channel": self.channel,
        }

        if meta_path is None:
            meta_path = Path(str(model_path) + ".meta.json")
        else:
            meta_path = Path(meta_path)

        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def load_model(
        self,
        model_path: Union[str, Path],
        meta_path: Optional[Union[str, Path]] = None
    ) -> "SurrogateExplainer":
        """
        Load a persisted surrogate model and metadata.

        Parameters
        ----------
        model_path : str or Path
            Path to the saved CatBoost model.
        meta_path : str or Path, optional
            Path to metadata JSON. Defaults to model_path + ".meta.json".

        Returns
        -------
        SurrogateExplainer
            Self, for method chaining.
        """
        model_path = Path(model_path)

        self.model = CatBoostRegressor()
        self.model.load_model(str(model_path))

        # Load metadata
        if meta_path is None:
            meta_path = Path(str(model_path) + ".meta.json")
        else:
            meta_path = Path(meta_path)

        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            self.feature_names = meta.get("feature_names", [])
            self.r2_score_value = meta.get("r2_score", 0.0)
            self._cat_cols_cached = meta.get("cat_cols", [])
            self.channel = meta.get("channel", "B&M")  # Default to B&M for backward compatibility

        return self

    def compute_shap_values(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute SHAP values for the input data.

        Parameters
        ----------
        df : pd.DataFrame
            Input data with Model A features.

        Returns
        -------
        np.ndarray
            SHAP values for each sample and feature.
        """
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP is required for explainability. Install with: pip install shap")

        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        X = self._prepare_data(df)
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)

        # Handle shap.Explanation object
        if hasattr(shap_values, "values"):
            return shap_values.values
        return shap_values

    def compute_shap_values_batched(
        self,
        df: pd.DataFrame,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute SHAP values in batches to manage memory.

        For large DataFrames, computing SHAP in one pass can exceed memory limits.
        This method processes in batches and concatenates results, with garbage
        collection between batches to free memory.

        Parameters
        ----------
        df : pd.DataFrame
            Input data with Model A features.
        batch_size : int, optional
            Number of rows per batch. Default is controlled by SHAP_BATCH_SIZE
            environment variable (500).

        Returns
        -------
        np.ndarray
            SHAP values for each sample and feature.

        Notes
        -----
        Memory usage is approximately: batch_size * n_features * 8 bytes per batch.
        With default batch_size=500 and 50 features: ~200 KB per batch.
        """
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP is required for explainability. Install with: pip install shap")

        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Use default batch size if not specified
        if batch_size is None:
            batch_size = SHAP_BATCH_SIZE

        X = self._prepare_data(df)
        n_samples = len(X)

        # For small DataFrames, use standard method (no batching overhead)
        if n_samples <= batch_size:
            return self.compute_shap_values(df)

        explainer = shap.TreeExplainer(self.model)
        shap_results = []

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_X = X.iloc[start_idx:end_idx]

            batch_shap = explainer.shap_values(batch_X)

            # Handle shap.Explanation object
            if hasattr(batch_shap, "values"):
                batch_shap = batch_shap.values

            shap_results.append(batch_shap)

            # Force garbage collection between batches to free memory
            gc.collect()

        return np.vstack(shap_results)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained surrogate model.

        Returns
        -------
        pd.DataFrame
            DataFrame with feature names and importance scores.
        """
        if self.model is None:
            raise ValueError("Model not trained.")

        importance = self.model.get_feature_importance()
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False)

    def get_top_features(
        self,
        shap_values: np.ndarray,
        top_n: int = 10
    ) -> List[str]:
        """
        Get top features by mean absolute SHAP value.

        Parameters
        ----------
        shap_values : np.ndarray
            SHAP values array.
        top_n : int, optional
            Number of top features to return. Default is 10.

        Returns
        -------
        List[str]
            Names of top features.
        """
        top_n = min(top_n, shap_values.shape[1])
        importance = np.abs(shap_values).mean(axis=0)
        ranked_indices = np.argsort(importance)[::-1][:top_n]
        return [self.feature_names[i] for i in ranked_indices]

    def build_contributor_strings(
        self,
        df: pd.DataFrame,
        shap_values: np.ndarray,
        top_k: int = 3
    ) -> pd.Series:
        """
        Build per-row strings of top contributors with signed SHAP values.

        Parameters
        ----------
        df : pd.DataFrame
            Input data with Model A features.
        shap_values : np.ndarray
            SHAP values array.
        top_k : int, optional
            Number of top contributors per row. Default is 3.

        Returns
        -------
        pd.Series
            String series with format "feature=value:+/-shap; ..."
        """
        top_k = min(top_k, shap_values.shape[1])
        shap_df = pd.DataFrame(shap_values, columns=self.feature_names)
        X = self._prepare_data(df).reset_index(drop=True)

        contributor_strings = []
        for i, row in shap_df.iterrows():
            abs_sorted = row.abs().sort_values(ascending=False)
            parts = []
            for feat in abs_sorted.index[:top_k]:
                shap_val = row[feat]
                raw_val = X.at[i, feat]
                if isinstance(raw_val, float):
                    val_str = f"{raw_val:.3g}"
                else:
                    val_str = str(raw_val)
                parts.append(f"{feat}={val_str}:{shap_val:+.3f}")
            contributor_strings.append("; ".join(parts))

        return pd.Series(contributor_strings)

    def explain(
        self,
        df: pd.DataFrame,
        output_dir: Union[str, Path],
        name: str,
        keys: Optional[List[str]] = None,
        target_label: str = "pred_log_sales",
        top_k_contributors: int = 3,
        cohort_cols: Optional[List[str]] = None,
        dependence_top_n: int = 8,
        color_feature: Optional[str] = None,
        generate_plots: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Generate SHAP explanation with plots and contributor strings.

        Parameters
        ----------
        df : pd.DataFrame
            Input data with Model A features and Model B predictions.
        output_dir : str or Path
            Directory to save plots.
        name : str
            Name prefix for output files.
        keys : List[str], optional
            Key columns for merging contributor results.
        target_label : str, optional
            Target column name for contributor label. Default "pred_log_sales".
        top_k_contributors : int, optional
            Number of top contributors per row. Default 3.
        cohort_cols : List[str], optional
            Columns for cohort analysis plots.
        dependence_top_n : int, optional
            Number of top features for dependence plots. Default 8.
        color_feature : str, optional
            Feature to use for coloring dependence plots.
        generate_plots : bool, optional
            Whether to generate SHAP plots. Default True.

        Returns
        -------
        pd.DataFrame or None
            DataFrame with key columns and contributor strings, or None.
        """
        try:
            import shap
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("SHAP and matplotlib are required. Install with: pip install shap matplotlib")

        print(f"Generating SHAP explanation for {name}...")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        X = self._prepare_data(df)
        shap_values = self.compute_shap_values(df)
        shap_array = shap_values

        top_features = self.get_top_features(shap_array, dependence_top_n)
        cohort_cols = cohort_cols or ["profit_center_nbr", "dma", "is_outlet", "is_comp_store"]
        color_feat = color_feature or ("horizon" if "horizon" in X.columns else None)

        if generate_plots:
            # Summary Plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X, show=False)
            plt.title(f"SHAP Summary: {name} (Surrogate R2={self.r2_score_value:.3f})")
            plt.tight_layout()
            plt.savefig(output_dir / f"shap_summary_{name}.png")
            plt.close()

            # Bar Plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
            plt.title(f"Feature Importance: {name}")
            plt.tight_layout()
            plt.savefig(output_dir / f"shap_importance_{name}.png")
            plt.close()

            # Dependence Plots (Top-N)
            color_feat_valid = color_feat if color_feat in X.columns else None
            for feat in top_features:
                plt.figure(figsize=(10, 8))
                shap.dependence_plot(feat, shap_values, X, interaction_index=color_feat_valid, show=False)
                color_label = color_feat_valid if color_feat_valid else feat
                plt.title(f"Dependence: {feat} vs SHAP (color={color_label})")
                plt.tight_layout()
                plt.savefig(output_dir / f"shap_dependence_{name}_{feat}.png")
                plt.close()

            # Horizon Sensitivity Plot
            if "horizon" in df.columns and not df["horizon"].isna().all():
                self._plot_horizon_sensitivity(shap_array, df, top_features, output_dir, name)

            # Cohort Bars
            self._plot_cohort_bars(shap_array, df, cohort_cols, top_features, output_dir, name)

            # Fit Scatter Plot
            if "pred_log_sales" in df.columns:
                self._plot_fit_scatter(df, output_dir, name)

        # Build contributor strings
        contributor_df = None
        if top_k_contributors and top_k_contributors > 0:
            contributor_series = self.build_contributor_strings(df, shap_array, top_k=top_k_contributors)
            contributor_col = f"top_features_{target_label}"
            if keys:
                missing_keys = [k for k in keys if k not in df.columns]
                if missing_keys:
                    print(f"Warning: Missing keys for contributor merge: {missing_keys}")
                contributor_df = df[keys].copy()
                contributor_df[contributor_col] = contributor_series.values
            else:
                contributor_df = pd.DataFrame({contributor_col: contributor_series.values})

        return contributor_df

    def _plot_horizon_sensitivity(
        self,
        shap_array: np.ndarray,
        df: pd.DataFrame,
        top_features: List[str],
        output_dir: Path,
        name: str
    ) -> None:
        """Plot mean |SHAP| by horizon for top features."""
        import matplotlib.pyplot as plt

        shap_abs_df = pd.DataFrame(np.abs(shap_array), columns=self.feature_names)
        shap_abs_df["horizon"] = df["horizon"].values

        horizon_group = shap_abs_df.groupby("horizon").mean().sort_index()
        if horizon_group.empty:
            return

        plt.figure(figsize=(12, 8))
        for feat in top_features:
            if feat in horizon_group.columns:
                plt.plot(horizon_group.index, horizon_group[feat], marker="o", label=feat)
        plt.xlabel("Horizon")
        plt.ylabel("Mean |SHAP|")
        plt.title(f"Horizon Sensitivity: {name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"shap_horizon_sensitivity_{name}.png")
        plt.close()

    def _plot_cohort_bars(
        self,
        shap_array: np.ndarray,
        df: pd.DataFrame,
        cohort_cols: List[str],
        top_features: List[str],
        output_dir: Path,
        name: str
    ) -> None:
        """Plot cohort-level bars of total top-feature |SHAP|."""
        import matplotlib.pyplot as plt

        shap_abs_df = pd.DataFrame(np.abs(shap_array), columns=self.feature_names)

        for cohort in cohort_cols:
            if cohort not in df.columns:
                continue

            cohort_df = shap_abs_df.copy()
            group_col = cohort if cohort not in shap_abs_df.columns else f"{cohort}_cohort"
            cohort_df[group_col] = df[cohort].astype(str).values
            grouped = cohort_df.groupby(group_col).mean()
            if grouped.empty:
                continue

            available_feats = [f for f in top_features if f in grouped.columns]
            if not available_feats:
                continue

            grouped["total_top_shap"] = grouped[available_feats].sum(axis=1)
            top_cohorts = grouped["total_top_shap"].sort_values(ascending=False).head(15)

            plt.figure(figsize=(12, 8))
            plt.barh(top_cohorts.index.astype(str), top_cohorts.values)
            plt.gca().invert_yaxis()
            plt.xlabel("Mean |SHAP| (Top Features)")
            plt.title(f"Cohort Sensitivity ({cohort}): {name}")
            plt.tight_layout()
            plt.savefig(output_dir / f"shap_cohort_{cohort}_{name}.png")
            plt.close()

    def _plot_fit_scatter(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        name: str
    ) -> None:
        """Plot surrogate fit vs Model B predictions."""
        import matplotlib.pyplot as plt

        target = df["pred_log_sales"]
        preds = self.predict(df)

        plt.figure(figsize=(10, 10))
        plt.scatter(target, preds, alpha=0.1, s=1)

        # Diagonal line
        min_val = min(target.min(), preds.min())
        max_val = max(target.max(), preds.max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--")

        plt.xlabel("Model B Prediction (Log Sales)")
        plt.ylabel("Surrogate Model Prediction")
        plt.title(f"Surrogate Fit Analysis: R2={self.r2_score_value:.4f}")
        plt.savefig(output_dir / f"surrogate_fit_scatter_{name}.png")
        plt.close()
