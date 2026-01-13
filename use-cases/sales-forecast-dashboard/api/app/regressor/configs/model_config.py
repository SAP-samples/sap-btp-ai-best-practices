"""
Model Configuration - Single Source of Truth for Feature Definitions.

This module defines:
1. CatBoost hyperparameters
2. Feature configurations (categorical, exclude, channel-specific)
3. Pre-configured model configs for common use cases

All feature exclusion lists are defined HERE and imported by other modules.
Do NOT define duplicate feature lists elsewhere.
"""

from dataclasses import dataclass, field
from typing import List, Optional


# =============================================================================
# CANONICAL FEATURE LISTS - Single Source of Truth
# =============================================================================

# Categorical features to be encoded natively by CatBoost
CATEGORICAL_FEATURES = [
    "profit_center_nbr",
    "dma",
    "is_outlet",
    "is_comp_store",
    "is_new_store",
    "is_holiday",
    "is_xmas_window",
    "is_black_friday_window",
    "is_pre_holiday_1wk",
    "is_pre_holiday_2wk",
    "is_pre_holiday_3wk",
]

# Features to ALWAYS exclude from training (metadata, targets, leakage)
EXCLUDE_FEATURES = [
    # Metadata / Keys
    "channel",
    "origin_week_date",
    "target_week_date",
    # Target variables (labels)
    "label_log_sales",
    "label_log_aov",
    "label_logit_conversion",
    "label_log_orders",
    # Raw values that should not be features
    "has_traffic_data",
    "total_sales",
    "order_count",
    "store_traffic",
    # Prediction outputs (if present in data)
    "predicted_log_sales",
    "predicted_log_aov",
    "predicted_logit_conversion",
    "predicted_log_orders",
    # Store DNA - not predictive
    "store_design_sf",
    # Data leakage - derived from target variables (total_sales/order_count)
    "aur",
]

# B&M-only features (excluded when training WEB models)
BM_ONLY_FEATURES = [
    # Conversion features - only applicable to B&M with foot traffic
    "ConversionRate_lag_1",
    "ConversionRate_lag_4",
    "ConversionRate_roll_mean_4",
    "ConversionRate_roll_mean_8",
    "ConversionRate_roll_mean_13",
    # Cannibalization features - physical store competition
    "cannibalization_pressure",
    "min_dist_new_store_km",
    "num_new_stores_within_10mi_last_52wk",
    "num_new_stores_within_20mi_last_52wk",
    # Store size - physical space (only relevant for B&M)
    "merchandising_sf",
]

# WEB-only features (excluded when training B&M models)
WEB_ONLY_FEATURES = [
    # Web traffic features from Ecomm Traffic
    "allocated_web_traffic_lag_1",
    "allocated_web_traffic_lag_4",
    "allocated_web_traffic_lag_13",
    "allocated_web_traffic_roll_mean_4",
    "allocated_web_traffic_roll_mean_8",
    "allocated_web_traffic_roll_mean_13",
    # WEB sales autoregressive features
    "log_web_sales_lag_1",
    "log_web_sales_lag_4",
    "log_web_sales_lag_13",
    "log_web_sales_roll_mean_4",
    "log_web_sales_roll_mean_8",
    "log_web_sales_roll_mean_13",
    "vol_log_web_sales_13",
    # WEB AOV features
    "web_aov_roll_mean_4",
    "web_aov_roll_mean_8",
    # Legacy single feature
    "log_web_sales_roll_mean_web_4",
    # DMA market feature
    "dma_web_penetration_pct",
]


# =============================================================================
# Helper Functions
# =============================================================================

def get_exclude_features_for_channel(channel: str) -> List[str]:
    """
    Get the complete list of features to exclude for a given channel.

    Parameters
    ----------
    channel : str
        'B&M' or 'WEB'

    Returns
    -------
    List[str]
        Complete exclusion list for the channel
    """
    exclude = set(EXCLUDE_FEATURES)

    if channel == "B&M":
        exclude.update(WEB_ONLY_FEATURES)
    elif channel == "WEB":
        exclude.update(BM_ONLY_FEATURES)

    return list(exclude)


# =============================================================================
# Dataclass Configurations
# =============================================================================

@dataclass
class CatBoostHyperparams:
    """CatBoost model hyperparameters."""

    iterations: int = 5000
    learning_rate: float = 0.05
    depth: int = 6
    loss_function: str = "MultiRMSE"  # or "RMSE" for single-target
    eval_metric: Optional[str] = None  # Defaults to loss_function if None
    early_stopping_rounds: Optional[int] = 50
    random_seed: int = 42
    verbose: int = 100

    def __post_init__(self):
        if self.eval_metric is None:
            self.eval_metric = self.loss_function


@dataclass
class FeatureConfig:
    """
    Feature configuration for model training.

    Uses module-level constants as defaults to ensure single source of truth.
    Override only when you need non-standard behavior.
    """

    categorical_features: List[str] = field(
        default_factory=lambda: CATEGORICAL_FEATURES.copy()
    )
    exclude_features: List[str] = field(
        default_factory=lambda: EXCLUDE_FEATURES.copy()
    )
    bm_only_features: List[str] = field(
        default_factory=lambda: BM_ONLY_FEATURES.copy()
    )
    web_only_features: List[str] = field(
        default_factory=lambda: WEB_ONLY_FEATURES.copy()
    )


@dataclass
class ModelConfig:
    """Complete model configuration combining hyperparameters and features."""

    name: str = "default_model"
    channel: Optional[str] = None  # 'B&M', 'WEB', or None
    target: str = "sales"  # 'sales', 'aov', 'orders', 'conversion', 'multi'
    hyperparams: CatBoostHyperparams = field(default_factory=CatBoostHyperparams)
    features: FeatureConfig = field(default_factory=FeatureConfig)

    def __post_init__(self):
        if self.channel is not None and self.channel not in ("B&M", "WEB"):
            raise ValueError(f"Invalid channel '{self.channel}'. Must be 'B&M', 'WEB', or None.")

        valid_targets = ("sales", "aov", "orders", "conversion", "multi")
        if self.target not in valid_targets:
            raise ValueError(f"Invalid target '{self.target}'. Must be one of {valid_targets}.")

    def get_exclude_features_for_channel(self, channel: str) -> List[str]:
        """Get the complete list of features to exclude for a given channel."""
        exclude = set(self.features.exclude_features)

        if channel == "B&M":
            exclude.update(self.features.web_only_features)
        elif channel == "WEB":
            exclude.update(self.features.bm_only_features)

        return list(exclude)


# =============================================================================
# Pre-configured Model Configs
# =============================================================================

def get_bm_multi_config() -> ModelConfig:
    """Get configuration for B&M multi-objective model (Sales, AOV, Orders)."""
    return ModelConfig(
        name="bm_multi_objective",
        channel="B&M",
        target="multi",
        hyperparams=CatBoostHyperparams(
            iterations=5000,
            learning_rate=0.05,
            depth=6,
            loss_function="MultiRMSE",
        ),
    )


def get_bm_conversion_config() -> ModelConfig:
    """Get configuration for B&M conversion model."""
    return ModelConfig(
        name="bm_conversion",
        channel="B&M",
        target="conversion",
        hyperparams=CatBoostHyperparams(
            iterations=5000,
            learning_rate=0.05,
            depth=6,
            loss_function="RMSE",
        ),
    )


def get_web_multi_config() -> ModelConfig:
    """Get configuration for WEB multi-objective model (Sales, AOV, Orders)."""
    return ModelConfig(
        name="web_multi_objective",
        channel="WEB",
        target="multi",
        hyperparams=CatBoostHyperparams(
            iterations=5000,
            learning_rate=0.05,
            depth=6,
            loss_function="MultiRMSE",
        ),
    )


def get_surrogate_config() -> ModelConfig:
    """Get configuration for surrogate explainability model."""
    return ModelConfig(
        name="surrogate_explainer",
        channel=None,
        target="multi",
        hyperparams=CatBoostHyperparams(
            iterations=5000,
            learning_rate=0.1,  # Higher learning rate
            depth=8,  # Higher depth to overfit
            loss_function="RMSE",
            early_stopping_rounds=None,  # No early stopping
            verbose=0,
        ),
    )
