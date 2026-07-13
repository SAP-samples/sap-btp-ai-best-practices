"""
Model View Builder - Dual Architecture (Model A vs Model B)

Partitions features into two model variants to maintain interpretability
while preserving accuracy:

Model B (Production/Full):
- Complete feature set including autoregressive lags/rolls
- Used for production forecasts
- Highest accuracy but attribution dominated by lags

Model A (Actionable/Explainability):
- Business levers + known-in-advance/static context ONLY
- NO target lags/rolls (prevents lag dominance)
- Used for SHAP explanations and what-if analyses
- NOT for production forecasts

Rationale: Autoregressive lags overshadow operational levers.
Example: Week-1 omnichannel push shows up in lag_1 next week, stealing credit.

Author: EPIC 4 Feature Engineering
Status: Phase 3 - Model Partitioning
"""

import pandas as pd
from typing import List, Optional, Set
import warnings


# Model A Features: Actionable business levers + seasonality for SHAP explainability
# 29 features (B&M) / 22 features (WEB) - includes seasonality to prevent feature aliasing
# Pruned: Only roll_mean_4 versions of business levers (no lag_1, lag_4, roll_mean_8)
# These are the features business users can actually influence or understand
MODEL_A_FEATURES = [
    # --------------------------------------------------
    # Categorical Features (5)
    # --------------------------------------------------
    'is_outlet',
    'is_comp_store',
    'is_new_store',
    'is_xmas_window',
    'is_black_friday_window',

    # --------------------------------------------------
    # Store DNA (4)
    # --------------------------------------------------
    'weeks_since_open',
    'weeks_since_open_capped_13',
    'weeks_since_open_capped_52',
    'merchandising_sf',

    # --------------------------------------------------
    # Market Signals (2) - 4-week rolling mean for stability
    # --------------------------------------------------
    'brand_awareness_dma_roll_mean_4',
    'brand_consideration_dma_roll_mean_4',

    # --------------------------------------------------
    # Cannibalization (4)
    # --------------------------------------------------
    'cannibalization_pressure',
    'min_dist_new_store_km',
    'num_new_stores_within_10mi_last_52wk',
    'num_new_stores_within_20mi_last_52wk',

    # --------------------------------------------------
    # Seasonality/Calendar (4) - Prevent feature aliasing
    # Required for differential SHAP (cancels out in Scenario - Baseline)
    # Using sin/cos encoding for woy to preserve cyclical nature (week 52 ≈ week 1)
    # --------------------------------------------------
    'dma_seasonal_weight',
    'sin_woy',
    'cos_woy',
    'is_holiday',

    # --------------------------------------------------
    # Staffing Levers - B&M ONLY (2)
    # Key actionable lever affecting conversion
    # --------------------------------------------------
    'staffing_unique_associates_roll_mean_4',
    'staffing_hours_roll_mean_4',

    # --------------------------------------------------
    # Operational Levers - ACTIONABLE (4)
    # Pruned: Keep only roll_mean_4 versions
    # --------------------------------------------------
    'pct_omni_channel_roll_mean_4',
    'pct_value_product_roll_mean_4',
    'pct_premium_product_roll_mean_4',
    'pct_white_glove_roll_mean_4',

    # --------------------------------------------------
    # Financing Levers - ACTIONABLE (3)
    # Pruned: Keep only roll_mean_4 versions
    # --------------------------------------------------
    'pct_primary_financing_roll_mean_4',
    'pct_secondary_financing_roll_mean_4',
    'pct_tertiary_financing_roll_mean_4',

    # --------------------------------------------------
    # Horizon (1)
    # --------------------------------------------------
    'horizon',
]


# B&M-only features to exclude from WEB Model A (explainability)
# These are physical store metrics that are not applicable to the digital channel
MODEL_A_BM_ONLY_FEATURES = [
    'merchandising_sf',           # Physical store square footage
    'cannibalization_pressure',   # Physical store competition metric
    'min_dist_new_store_km',      # Physical store proximity
    'num_new_stores_within_10mi_last_52wk',  # Physical store competition
    'num_new_stores_within_20mi_last_52wk',  # Physical store competition
    'staffing_unique_associates_roll_mean_4',  # Physical store staffing
    'staffing_hours_roll_mean_4',              # Physical store staffing
]


# Model B Features: Model A + Autoregressive lags/rolls + additional Calendar features
# Complete feature set for production forecasts
# Note: dma_seasonal_weight, woy, is_holiday are now in Model A (shared)
MODEL_B_ADDITIONAL_FEATURES = [
    # --------------------------------------------------
    # Sales & AOV Autoregressive (TIER 1)
    # --------------------------------------------------
    'log_sales_lag_1',
    'log_sales_lag_4',
    'log_sales_lag_13',
    'log_sales_lag_52',
    'log_sales_roll_mean_4',
    'log_sales_roll_mean_8',
    'log_sales_roll_mean_13',
    'AOV_roll_mean_8',
    'AOV_roll_mean_13',
    'vol_log_sales_13',
    'log_web_sales_roll_mean_web_4',

    # --------------------------------------------------
    # Conversion Dynamics (TIER 1 for B&M)
    # --------------------------------------------------
    'ConversionRate_lag_1',
    'ConversionRate_lag_4',
    'ConversionRate_roll_mean_4',
    'ConversionRate_roll_mean_8',
    'ConversionRate_roll_mean_13',

    # --------------------------------------------------
    # Additional Calendar/Seasonality (Model B only)
    # Note: dma_seasonal_weight, woy, is_holiday moved to Model A
    # --------------------------------------------------
    'sin_woy',
    'cos_woy',
    'month',
    'quarter',
    'fiscal_year',
    'fiscal_period',
    'weeks_to_holiday',

    # --------------------------------------------------
    # Web Traffic (Model B only)
    # --------------------------------------------------
    'allocated_web_traffic_roll_mean_4',
    'allocated_web_traffic_roll_mean_8',
    'allocated_web_traffic_roll_mean_13',
    'allocated_web_traffic_lag_1',
    'allocated_web_traffic_lag_4',
    'allocated_web_traffic_lag_13',

    # --------------------------------------------------
    # Additional Store DNA (Model B only)
    # --------------------------------------------------
    'sq_ft',
    'store_design_sf',

    # --------------------------------------------------
    # CRM Demographic Features (Tier 3 Cross-Sectional)
    # Pooled rho >= 0.15, FE rho < 0.10
    # Static cross-sectional predictors (not time-varying levers)
    # Only included when --include-crm flag is used
    # --------------------------------------------------
    'crm_dwelling_single_family_dwelling_unit_pct',
    'crm_dwelling_multi_family_dwelling_unit_pct',
    'crm_owner_renter_owner_pct',
    'crm_income_150k_plus_pct',
    'crm_income_under_50k_pct',
    'crm_marital_married_pct',
    'crm_age_25_34_pct',
    'crm_age_55_64_pct',
    'crm_age_65_80_pct',
    'crm_education_college_pct',
    'crm_education_high_school_pct',
    'crm_children_y_pct',
]


def build_model_b_features(
    canonical_df: pd.DataFrame,
    categorical_features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Build Model B (Production/Full) feature view.

    Model B uses the complete feature set including autoregressive lags/rolls.
    Used for production forecasts (highest accuracy).

    Features:
    - All TIER 1/2/3 features
    - Autoregressive lags/rolls of Sales, AOV, Conversion
    - Known-in-advance features
    - Static features
    - Operational levers

    Parameters
    ----------
    canonical_df : pd.DataFrame
        Canonical table with all features attached
    categorical_features : Optional[List[str]]
        Additional categorical features (e.g., region, format, store_id, dma)
        These will be marked for CatBoost's native categorical handling

    Returns
    -------
    pd.DataFrame
        Model B feature view (all features + essential columns preserved)

    Examples
    --------
    >>> categorical_cols = ['region', 'format', 'profit_center_nbr', 'dma', 'channel']
    >>> model_b_df = build_model_b_features(canonical_df, categorical_cols)
    >>> print(f"Model B features: {len(model_b_df.columns)}")
    """
    # Define essential columns to preserve (keys, labels, metadata, raw values)
    # Use sorted list for deterministic ordering, set for fast lookup
    essential_cols_list = sorted([
        # Keys
        'profit_center_nbr', 'dma', 'channel', 'origin_week_date',
        'target_week_date', 'horizon',
        # Labels
        'label_log_sales', 'label_log_aov', 'label_logit_conversion',
        # Metadata
        'has_traffic_data',
        # Raw values (optional, may not be present)
        'total_sales', 'order_count', 'store_traffic', 'aur', 'allocated_web_traffic'
    ])
    essential_cols = set(essential_cols_list)  # Keep set for fast lookup

    # Model B = Model A + Autoregressive features
    # Sort for deterministic column ordering
    all_features_sorted = sorted(set(MODEL_A_FEATURES + MODEL_B_ADDITIONAL_FEATURES))

    if categorical_features:
        # Add categorical features, but remove any that are already in essential_cols
        # to avoid treating them as features to filter
        cat_features_to_add = sorted([f for f in categorical_features if f not in essential_cols])
        all_features_sorted = sorted(set(all_features_sorted + cat_features_to_add))

    # Filter to available features (exclude essential columns from filtering)
    # Use sorted list for deterministic ordering
    available_features = [f for f in all_features_sorted
                          if f in canonical_df.columns and f not in essential_cols]
    missing_features = [f for f in all_features_sorted
                        if f not in canonical_df.columns and f not in essential_cols]

    if len(missing_features) > 0:
        warnings.warn(
            f"Model B: {len(missing_features)} expected features not found in dataframe. "
            f"Examples: {missing_features[:5]}"
        )

    # Preserve essential columns + filtered features
    # Use sorted list for deterministic column order
    essential_to_keep = [c for c in essential_cols_list if c in canonical_df.columns]
    columns_to_keep = essential_to_keep + available_features

    # Return dataframe with essential columns + Model B features
    return canonical_df[columns_to_keep]


def build_model_a_features(
    canonical_df: pd.DataFrame,
    categorical_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build Model A (Actionable/Explainability) feature view.

    Model A contains only actionable business levers for SHAP explanations.
    Used for what-if analyses (NOT production forecasts).

    Features Included (23 total):
    - Categorical: is_outlet, is_comp_store, is_new_store, is_xmas_window, is_black_friday_window
    - Store DNA: weeks_since_open, weeks_since_open_capped_*, merchandising_sf
    - Market Signals: brand_awareness_dma_roll_mean_4, brand_consideration_dma_roll_mean_4
    - Cannibalization: cannibalization_pressure, min_dist_new_store_km, num_new_stores_*
    - Operational Levers: pct_omni_channel_*, pct_value_product_*, pct_premium_product_*, pct_white_glove_*
    - horizon

    Features Excluded:
    - Calendar/Seasonality: woy, sin_woy, cos_woy, month, quarter, etc.
    - Holiday features: is_holiday (is_pre_holiday_* removed from Model B)
    - Web traffic: allocated_web_traffic_*
    - Target lags: log_sales_lag_*, AOV_*, ConversionRate_*

    Parameters
    ----------
    canonical_df : pd.DataFrame
        Canonical table with all features attached
    categorical_features : Optional[List[str]]
        Additional categorical features

    Returns
    -------
    pd.DataFrame
        Model A feature view (actionable only + essential columns preserved)

    Examples
    --------
    >>> model_a_df = build_model_a_features(canonical_df)
    >>> print(f"Model A features: {len(model_a_df.columns)}")  # Should be ~31 columns
    """
    # Define essential columns to preserve (keys and labels only)
    # Use sorted list for deterministic ordering, set for fast lookup
    essential_cols_list = sorted([
        # Keys
        'profit_center_nbr', 'dma', 'channel', 'origin_week_date',
        'target_week_date',
        # Labels (needed for training surrogate)
        'label_log_sales', 'label_log_aov', 'label_logit_conversion',
    ])
    essential_cols = set(essential_cols_list)  # Keep set for fast lookup

    # Sort for deterministic column ordering
    features_sorted = sorted(set(MODEL_A_FEATURES))

    if categorical_features:
        # Add categorical features, but remove any that are already in essential_cols
        cat_features_to_add = sorted([f for f in categorical_features if f not in essential_cols])
        features_sorted = sorted(set(features_sorted + cat_features_to_add))

    # Filter to available features (exclude essential columns from filtering)
    # Use sorted list for deterministic ordering
    available_features = [f for f in features_sorted
                          if f in canonical_df.columns and f not in essential_cols]
    missing_features = [f for f in features_sorted
                        if f not in canonical_df.columns and f not in essential_cols]

    if len(missing_features) > 0:
        warnings.warn(
            f"Model A: {len(missing_features)} expected features not found in dataframe. "
            f"Examples: {missing_features[:5]}"
        )

    # Verify no TARGET lags/rolls in Model A (operational lever lags ARE allowed)
    lag_roll_features = [f for f in available_features
                         if 'lag_' in f or 'roll_mean_' in f or 'roll' in f]
    # Only check for target variable lags (sales, AOV, conversion) and web traffic
    # Operational levers (pct_omni_channel, pct_value_product, etc.) ARE allowed
    target_lags = [f for f in lag_roll_features
                   if any(x in f for x in ['log_sales', 'AOV', 'ConversionRate', 'log_aov', 'allocated_web_traffic'])]

    if len(target_lags) > 0:
        raise ValueError(
            f"Model A contains {len(target_lags)} target lag/roll features - "
            f"this violates the actionable-only constraint! Features: {target_lags}"
        )

    # Preserve essential columns + filtered features
    # Use sorted list for deterministic column order
    essential_to_keep = [c for c in essential_cols_list if c in canonical_df.columns]
    columns_to_keep = essential_to_keep + available_features

    # Return dataframe with essential columns + Model A features
    return canonical_df[columns_to_keep]


def get_feature_importance_comparison(
    model_a_importance: pd.DataFrame,
    model_b_importance: pd.DataFrame,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Compare feature importance between Model A and Model B.

    Useful for identifying how autoregressive features affect attribution.

    Parameters
    ----------
    model_a_importance : pd.DataFrame
        Feature importance from Model A with columns: feature, importance
    model_b_importance : pd.DataFrame
        Feature importance from Model B with columns: feature, importance
    top_n : int, default=20
        Number of top features to compare

    Returns
    -------
    pd.DataFrame
        Comparison table with columns:
        - feature
        - importance_a (Model A importance)
        - importance_b (Model B importance)
        - rank_a (Model A rank)
        - rank_b (Model B rank)
        - rank_change (rank_a - rank_b, negative = gained importance in B)

    Examples
    --------
    >>> comparison = get_feature_importance_comparison(
    ...     model_a_shap_values,
    ...     model_b_shap_values,
    ...     top_n=20
    ... )
    >>> # Features that lose importance in Model B (dominated by lags)
    >>> dominated = comparison[comparison['rank_change'] > 5]
    >>> print("Features dominated by lags:")
    >>> print(dominated[['feature', 'rank_a', 'rank_b']])
    """
    # Ensure column names are consistent
    if 'feature' not in model_a_importance.columns:
        model_a_importance = model_a_importance.reset_index()
        model_a_importance.columns = ['feature', 'importance']

    if 'feature' not in model_b_importance.columns:
        model_b_importance = model_b_importance.reset_index()
        model_b_importance.columns = ['feature', 'importance']

    # Rank features
    model_a_importance['rank_a'] = model_a_importance['importance'].rank(
        ascending=False,
        method='first'
    ).astype(int)

    model_b_importance['rank_b'] = model_b_importance['importance'].rank(
        ascending=False,
        method='first'
    ).astype(int)

    # Merge
    comparison = model_a_importance[['feature', 'importance', 'rank_a']].merge(
        model_b_importance[['feature', 'importance', 'rank_b']],
        on='feature',
        how='outer',
        suffixes=('_a', '_b')
    )

    # Compute rank change
    comparison['rank_change'] = comparison['rank_a'] - comparison['rank_b']

    # Sort by Model B rank
    comparison = comparison.sort_values('rank_b')

    # Keep top N
    comparison = comparison.head(top_n)

    return comparison


def get_model_a_feature_list() -> List[str]:
    """Get list of Model A feature names."""
    return MODEL_A_FEATURES.copy()


def get_model_b_additional_feature_list() -> List[str]:
    """Get list of features ONLY in Model B (autoregressive)."""
    return MODEL_B_ADDITIONAL_FEATURES.copy()


def get_model_b_feature_list() -> List[str]:
    """Get complete list of Model B feature names."""
    return MODEL_A_FEATURES + MODEL_B_ADDITIONAL_FEATURES


def get_model_a_bm_only_features() -> List[str]:
    """Get list of B&M-only features to exclude from WEB Model A."""
    return MODEL_A_BM_ONLY_FEATURES.copy()


def get_model_a_features_for_channel(channel: str) -> List[str]:
    """
    Get Model A feature list for a specific channel.

    Parameters
    ----------
    channel : str
        Channel name ("B&M" or "WEB")

    Returns
    -------
    List[str]
        Feature list appropriate for the channel.
        - B&M: All 23 Model A features
        - WEB: 18 features (excludes B&M-only physical store metrics)
    """
    if channel.upper() == "WEB":
        return [f for f in MODEL_A_FEATURES if f not in MODEL_A_BM_ONLY_FEATURES]
    return MODEL_A_FEATURES.copy()
