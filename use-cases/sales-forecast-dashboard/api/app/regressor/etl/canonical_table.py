from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Iterable, Optional
import warnings

from app.regressor.data_ingestion.written_sales import load_written_sales_with_flags
from app.regressor.data_ingestion.store_master import load_store_master
from app.regressor.data_ingestion.ecomm_traffic import load_ecomm_traffic
from app.regressor.sister_store import build_sister_store_map
from app.regressor.features.time_varying_features import attach_time_varying_features
from app.regressor.features.static_features import attach_static_store_features
from app.regressor.features.dynamics_features import (
    attach_sales_aov_dynamics_bm,
    attach_web_dynamics,
    attach_conversion_omnichannel_features,
    attach_product_mix_features,
    attach_financing_features,
    attach_staffing_features,
)
from app.regressor.features.cannibalization import compute_cannibalization_pressure
from app.regressor.features.awareness_features import attach_awareness_features
from app.regressor.features.crm_features import attach_crm_features
from app.regressor.features.model_views import (
    build_model_a_features,
    build_model_b_features
)


def _safe_log(series: pd.Series, floor: float = 1e-6) -> pd.Series:
    return np.log(series.clip(lower=floor))


def _safe_logit(series: pd.Series, eps: float = 1e-6) -> pd.Series:
    s = series.clip(lower=eps, upper=1 - eps)
    return np.log(s / (1 - s))


def _prepare_sales_with_dma() -> pd.DataFrame:
    """
    Load sales data from correct sources and attach DMA info.

    Data Sources per ENGINEERING_INSTRUCTIONS:
    - B&M Sales: Written Sales Data.csv (channel != 'WEB')
    - WEB Sales: Ecomm Traffic.csv (merch_amt is authoritative)
    - WEB Orders: Written Sales Data.csv WEB rows (for AOV calculation only)

    Returns unified dataframe with columns:
    - profit_center_nbr, dma, channel, origin_week_date
    - total_sales, order_count, store_traffic, aur (for target calculation)
    - has_traffic_data (B&M only)
    """
    # Load store master for DMA mapping
    stores = load_store_master()[["profit_center_nbr", "market_city"]]

    # === B&M DATA (from Written Sales) ===
    written_sales = load_written_sales_with_flags()
    written_sales["channel"] = written_sales["channel"].astype(str).str.upper().str.strip()

    # Filter to B&M only (exclude WEB rows for sales amounts per ENGINEERING_INSTRUCTIONS line 32-33)
    bm_sales = written_sales[written_sales["channel"] != "WEB"].copy()
    bm_sales = bm_sales.merge(stores, on="profit_center_nbr", how="left")
    bm_sales = bm_sales.rename(columns={"market_city": "dma"})
    bm_sales["origin_week_date"] = pd.to_datetime(bm_sales["fiscal_start_date_week"], errors="coerce")

    # === WEB DATA (from Ecomm Traffic for sales, Written Sales for orders) ===
    ecomm = load_ecomm_traffic()
    ecomm = ecomm.rename(columns={"market_city": "dma"})
    ecomm["channel"] = "WEB"
    ecomm["origin_week_date"] = pd.to_datetime(ecomm["fiscal_start_date_week"], errors="coerce")
    # Map merch_amt to total_sales (authoritative WEB sales source)
    ecomm["total_sales"] = ecomm["merch_amt"]

    # Get WEB order counts from Written Sales WEB rows (for AOV calculation)
    web_written = written_sales[written_sales["channel"] == "WEB"][
        ["profit_center_nbr", "fiscal_start_date_week", "order_count", "aur"]
    ].copy()
    web_written["fiscal_start_date_week"] = pd.to_datetime(web_written["fiscal_start_date_week"], errors="coerce")

    # Merge WEB sales (ecomm) with WEB orders (written)
    web_sales = ecomm.merge(
        web_written,
        on=["profit_center_nbr", "fiscal_start_date_week"],
        how="left",
        suffixes=("", "_written")
    )

    # WEB has no physical traffic (set to NaN), no traffic flag
    web_sales["store_traffic"] = np.nan
    web_sales["has_traffic_data"] = 0

    # === COMBINE B&M and WEB ===
    # Standardize columns to match
    keep_cols = [
        "profit_center_nbr", "dma", "channel", "origin_week_date",
        "total_sales", "order_count", "store_traffic", "aur",
        "has_traffic_data"
    ]

    # Ensure both dataframes have all required columns
    for col in keep_cols:
        if col not in bm_sales.columns:
            bm_sales[col] = np.nan
        if col not in web_sales.columns:
            web_sales[col] = np.nan

    # Concatenate B&M and WEB
    combined = pd.concat([
        bm_sales[keep_cols],
        web_sales[keep_cols]
    ], ignore_index=True)

    combined = _impute_dma(combined, stores)
    return combined


def _impute_dma(sales_df: pd.DataFrame, store_master: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing DMA values using sister-store mapping; if still missing, leave as NaN and warn.
    """
    if "dma" not in sales_df.columns:
        return sales_df

    missing_mask = sales_df["dma"].isna()
    if not missing_mask.any():
        return sales_df

    sister_map = build_sister_store_map()
    if sister_map.empty:
        warnings.warn("DMA imputation: sister_store_map is empty; leaving missing DMA values as NaN.")
        return sales_df

    sister_dma = (
        sister_map[["profit_center_nbr", "sister_profit_center_nbr"]]
        .merge(
            store_master[["profit_center_nbr", "market_city"]],
            left_on="sister_profit_center_nbr",
            right_on="profit_center_nbr",
            suffixes=("", "_sister"),
            how="left",
        )
        .rename(columns={"market_city": "dma_imputed"})
        [["profit_center_nbr", "dma_imputed"]]
    )

    out = sales_df.merge(sister_dma, on="profit_center_nbr", how="left")
    out["dma"] = out["dma"].fillna(out["dma_imputed"])
    remaining_missing = out["dma"].isna().sum()
    if remaining_missing > 0:
        warnings.warn(f"DMA imputation: {remaining_missing} rows still missing DMA after sister-store fallback.")
    return out.drop(columns=["dma_imputed"])


def explode_history(
    sales_df: pd.DataFrame,
    horizons: Iterable[int] = range(1, 53),
) -> pd.DataFrame:
    """Create (store, dma, channel, origin_week_date, horizon, target_week_date) rows."""
    h_df = pd.DataFrame({"horizon": list(horizons)})
    base = sales_df[["profit_center_nbr", "dma", "channel", "origin_week_date"]].dropna(subset=["origin_week_date"])
    exploded = base.merge(h_df, how="cross")
    exploded["target_week_date"] = exploded["origin_week_date"] + pd.to_timedelta(exploded["horizon"], unit="W")
    return exploded


def attach_targets(
    exploded_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    drop_missing_targets: bool = True,
) -> pd.DataFrame:
    """
    Attach target values and labels (log sales, log AOV, logit conversion) at target_week_date.

    Target Calculation Rules:
    - AOV: total_sales / order_count (not AUR which is sales/units)
    - Conversion: ONLY for B&M channel with has_traffic_data=1
      - WEB channel: conversion set to NaN (no physical traffic concept)
      - B&M with has_traffic_data=0: conversion set to NaN (unreliable traffic data)
      - B&M with has_traffic_data=1: conversion = orders / traffic

    Drop Missing Targets Logic (if drop_missing_targets=True):
    - Drop rows missing Sales or AOV (both channels)
    - Drop B&M rows with has_traffic_data=1 but missing Conversion
    - Keep WEB rows (conversion expected to be NaN)
    - Keep B&M rows with has_traffic_data=0 (for Sales/AOV training only)
    """
    target_cols = [
        "profit_center_nbr",
        "channel",
        "origin_week_date",
        "total_sales",
        "aur",
        "order_count",
        "store_traffic",
        "has_traffic_data",
    ]
    targets = sales_df[target_cols].rename(columns={"origin_week_date": "target_week_date"})
    merged = exploded_df.merge(
        targets,
        on=["profit_center_nbr", "channel", "target_week_date"],
        how="left",
    )

    # === SALES TARGET (all channels) ===
    merged["label_log_sales"] = _safe_log(merged["total_sales"])

    # === AOV TARGET (all channels) ===
    orders = merged["order_count"].replace(0, np.nan)
    aov = merged["total_sales"] / orders
    merged["label_log_aov"] = _safe_log(aov)

    # === CONVERSION TARGET (B&M only, with valid traffic) ===
    # Initialize conversion as NaN for all rows
    merged["label_logit_conversion"] = np.nan

    # Calculate conversion ONLY for B&M rows with has_traffic_data=1
    is_bm = merged["channel"] != "WEB"
    has_valid_traffic = merged["has_traffic_data"] == 1

    bm_with_traffic_mask = is_bm & has_valid_traffic

    # Calculate conversion for valid B&M rows
    traffic = merged.loc[bm_with_traffic_mask, "store_traffic"].replace(0, np.nan)
    conversion = merged.loc[bm_with_traffic_mask, "order_count"] / traffic
    conversion = conversion.replace([np.inf, -np.inf], np.nan)
    merged.loc[bm_with_traffic_mask, "label_logit_conversion"] = _safe_logit(conversion)

    # === DROP MISSING TARGETS (channel-aware) ===
    if drop_missing_targets:
        # Always require Sales and AOV
        valid_sales_aov = merged["label_log_sales"].notna() & merged["label_log_aov"].notna()

        # For B&M with valid traffic, also require Conversion
        # For WEB or B&M without valid traffic, Conversion can be NaN
        needs_conversion = is_bm & has_valid_traffic
        valid_conversion = merged["label_logit_conversion"].notna() | ~needs_conversion

        merged = merged[valid_sales_aov & valid_conversion]

    return merged


def attach_features(
    base_df: pd.DataFrame,
    features_t0: Optional[pd.DataFrame] = None,
    features_tfuture: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Attach feature blocks:
    - features_t0 keyed by (profit_center_nbr, channel, origin_week_date)
    - features_tfuture keyed by (profit_center_nbr, channel, target_week_date)
    """
    df = base_df
    if features_t0 is not None and not features_t0.empty:
        df = df.merge(
            features_t0,
            on=["profit_center_nbr", "channel", "origin_week_date"],
            how="left",
        )
    if features_tfuture is not None and not features_tfuture.empty:
        df = df.merge(
            features_tfuture,
            on=["profit_center_nbr", "channel", "target_week_date"],
            how="left",
        )
    return df


def attach_all_features(
    exploded_df: pd.DataFrame,
    model_variant: str = 'B',
    include_crm: bool = False
) -> pd.DataFrame:
    """
    Orchestrate all feature engineering steps in priority order (EPIC 4).

    This function coordinates all [FE-01] through [FE-08] feature engineering tasks,
    applying them in the correct order to prevent dependency issues.

    Steps:
    1. Load Layer 0 artifacts (seasonality, holiday calendar)
    2. Load additional data sources (awareness, CRM, etc.)
    3. [FE-01] Time-varying known-in-advance features (seasonality, holidays, calendar)
    4. [FE-02] Sales/AOV dynamics (B&M channel)
    5. [FE-03] Web-specific dynamics
    6. [FE-04] Conversion & omnichannel features
    7. [FE-05] Static store DNA features
    8. [FE-06] Cannibalization pressure
    9. [FE-07] DMA awareness & consideration
    10. [FE-08] CRM demographics (optional)
    11. Product mix & service blend
    12. Build model view (A or B) to filter features

    Parameters
    ----------
    exploded_df : pd.DataFrame
        Canonical table with keys and targets
        Must have: profit_center_nbr, channel, origin_week_date, target_week_date, horizon
    model_variant : str, default='B'
        Which model variant to build:
        - 'B': Full feature set including autoregressive lags/rolls (production forecasts)
        - 'A': Actionable levers only, no target lags/rolls (SHAP/explainability)
    include_crm : bool, default=False
        Whether to include CRM demographic features (TIER 3)
        Set to False initially to avoid dependency on CRM data loader

    Returns
    -------
    pd.DataFrame
        Canonical table with all features attached and filtered to model variant

    Examples
    --------
    >>> # Build full canonical table with Model B features
    >>> canonical_with_features = attach_all_features(canonical_df, model_variant='B')
    >>>
    >>> # Build actionable-only table for SHAP
    >>> canonical_model_a = attach_all_features(canonical_df, model_variant='A')

    Notes
    -----
    - Features at t0 use data ≤ origin_week_date (leakage prevention)
    - Features at t0+h use target_week_date (known-in-advance only)
    - Model A excludes autoregressive target lags/rolls
    - Model B includes complete feature set for production
    """
    df = exploded_df.copy()

    print("EPIC 4 Feature Engineering: Starting feature attachment...")
    initial_cols = len(df.columns)

    try:
        # ===================================================================
        # Step 1: Load Layer 0 Artifacts and Additional Data
        # ===================================================================
        print("[1/14] Loading Layer 0 artifacts...")
        from app.regressor.seasonality import compute_dma_seasonality, compute_web_seasonality
        from app.regressor.io_utils import load_holiday_calendar

        # Compute DMA seasonality (requires sales history)
        # Using N-1 logic and B&M channel filter
        sales_history = load_written_sales_with_flags()
        dma_seasonality = compute_dma_seasonality(channel_filter='B&M')
        
        # Compute WEB seasonality (Global N-1 logic)
        web_seasonality = compute_web_seasonality()

        # Load holiday calendar
        holiday_calendar = load_holiday_calendar()

        # Load store master for static features and cannibalization
        store_master = load_store_master()

        # Load ecomm traffic for web dynamics
        ecomm_traffic = load_ecomm_traffic()

        # ===================================================================
        # Step 2: [FE-01] Time-Varying Known-in-Advance Features
        # ===================================================================
        print("[2/14] [FE-01] Attaching time-varying features (seasonality, holidays, calendar)...")
        df = attach_time_varying_features(
            df,
            dma_seasonality,
            holiday_calendar,
            web_seasonality_df=web_seasonality
        )
        print(f"        Added {len(df.columns) - initial_cols} features")

        # ===================================================================
        # Step 3: [FE-05] Static Store DNA (do early for reference)
        # ===================================================================
        print("[3/14] [FE-05] Attaching static store DNA features...")
        cols_before = len(df.columns)
        df = attach_static_store_features(
            df,
            store_master
        )
        print(f"        Added {len(df.columns) - cols_before} features")

        # ===================================================================
        # Step 4: [FE-02] Sales & AOV Dynamics (B&M)
        # ===================================================================
        print("[4/14] [FE-02] Attaching B&M sales/AOV dynamics...")
        cols_before = len(df.columns)
        df = attach_sales_aov_dynamics_bm(
            df,
            sales_history
        )
        print(f"        Added {len(df.columns) - cols_before} features")

        # ===================================================================
        # Step 5: [FE-03] Web-Specific Dynamics
        # ===================================================================
        print("[5/14] [FE-03] Attaching WEB-specific dynamics...")
        cols_before = len(df.columns)
        df = attach_web_dynamics(
            df,
            ecomm_traffic
        )
        print(f"        Added {len(df.columns) - cols_before} features")

        # ===================================================================
        # Step 6: [FE-04] Conversion & Omnichannel Features
        # ===================================================================
        print("[6/14] [FE-04] Attaching conversion & omnichannel features...")
        cols_before = len(df.columns)
        df = attach_conversion_omnichannel_features(
            df,
            sales_history
        )
        print(f"        Added {len(df.columns) - cols_before} features")

        # ===================================================================
        # Step 7: Product Mix & Service Blend
        # ===================================================================
        print("[7/14] Attaching product mix & service blend features...")
        cols_before = len(df.columns)
        df = attach_product_mix_features(
            df,
            sales_history
        )
        print(f"        Added {len(df.columns) - cols_before} features")

        # ===================================================================
        # Step 8: Financing Mix Features
        # ===================================================================
        print("[8/14] Attaching financing mix features...")
        cols_before = len(df.columns)
        df = attach_financing_features(
            df,
            sales_history
        )
        print(f"        Added {len(df.columns) - cols_before} features")

        # ===================================================================
        # Step 9: Staffing Features (B&M Only)
        # ===================================================================
        print("[9/14] Attaching staffing features (B&M only)...")
        cols_before = len(df.columns)
        df = attach_staffing_features(
            df,
            sales_history
        )
        print(f"        Added {len(df.columns) - cols_before} features")

        # ===================================================================
        # Step 10: [FE-06] Cannibalization Pressure
        # ===================================================================
        print("[10/14] [FE-06] Computing cannibalization pressure...")
        cols_before = len(df.columns)
        df = compute_cannibalization_pressure(
            df,
            store_master
        )
        print(f"        Added {len(df.columns) - cols_before} features")

        # ===================================================================
        # Step 11: [FE-07] DMA Awareness & Consideration
        # ===================================================================
        print("[11/14] [FE-07] Attaching awareness & consideration features...")
        try:
            from app.regressor.data_ingestion.awareness import load_awareness_with_mapping
            from app.regressor.data_ingestion.store_master import load_yougov_dma_map

            awareness_df = load_awareness_with_mapping()
            yougov_map = load_yougov_dma_map()

            cols_before = len(df.columns)
            df = attach_awareness_features(
                df,
                awareness_df,
                yougov_map,
                store_master,
                target_col='target_week_date'
            )
            print(f"        Added {len(df.columns) - cols_before} features")
        except Exception as e:
            warnings.warn(f"[FE-07] Awareness features failed: {e}. Skipping.")

        # ===================================================================
        # Step 12: [FE-08] CRM Demographics (Optional)
        # ===================================================================
        if include_crm:
            print("[12/14] [FE-08] Attaching CRM demographic features...")
            try:
                from app.regressor.data_ingestion.crm_mix import load_demographics_with_typing

                _, crm_df = load_demographics_with_typing()
                cols_before = len(df.columns)
                df = attach_crm_features(
                    df,
                    crm_df
                )
                print(f"        Added {len(df.columns) - cols_before} features")
            except Exception as e:
                warnings.warn(f"[FE-08] CRM features failed: {e}. Skipping.")
        else:
            print("[12/14] [FE-08] CRM features skipped (include_crm=False)")

        # ===================================================================
        # Step 13: Filter to Model Variant (A or B)
        # ===================================================================
        print(f"[13/14] Building Model {model_variant} feature view...")
        total_features_before = len(df.columns)

        categorical_features = ['profit_center_nbr', 'dma', 'channel']
        if 'region' in df.columns:
            categorical_features.append('region')
        if 'format' in df.columns:
            categorical_features.append('format')

        if model_variant.upper() == 'B':
            # Model B: Full feature set (production)
            df = build_model_b_features(df, categorical_features)
        elif model_variant.upper() == 'A':
            # Model A: Actionable only (SHAP/explainability)
            df = build_model_a_features(df, categorical_features)
        else:
            raise ValueError(f"Invalid model_variant '{model_variant}'. Use 'A' or 'B'.")

        features_after_filter = len(df.columns)
        print(f"        Filtered from {total_features_before} to {features_after_filter} columns for Model {model_variant}")

        # ===================================================================
        # Step 14: Complete
        # ===================================================================
        print(f"[14/14] Feature engineering complete!")
        print(f"        Total features attached: {len(df.columns) - initial_cols}")
        print(f"        Final dataframe: {len(df)} rows × {len(df.columns)} columns")

        return df

    except Exception as e:
        warnings.warn(f"Feature engineering failed: {e}. Returning dataframe without features.")
        raise


def build_canonical_training_table(
    horizons: Iterable[int] = range(1, 53),
    features_t0: Optional[pd.DataFrame] = None,
    features_tfuture: Optional[pd.DataFrame] = None,
    drop_missing_targets: bool = True,
    validate: bool = True,
    include_features: bool = False,
    model_variant: str = 'B',
    include_crm: bool = False,
) -> pd.DataFrame:
    """
    Build the canonical training table per Epic 3 & 4 (ENGINEERING_INSTRUCTIONS.md).

    This is the single source of truth for training, providing:
    - Keys: Unique identifier for each prediction task (store, channel, origin week, horizon, target week)
    - Targets: Transformed labels (log sales, log AOV, logit conversion)
    - Features (EPIC 4): Complete feature engineering pipeline with ~100-120 features

    Data Sources (per ENGINEERING_INSTRUCTIONS):
    - B&M Sales: Written Sales Data.csv
    - WEB Sales: Ecomm Traffic.csv (merch_amt is authoritative)
    - WEB Orders: Written Sales Data.csv WEB rows (for AOV calculation only)

    Target Calculation Rules:
    - Sales: log(total_sales) - both channels
    - AOV: log(total_sales / order_count) - both channels
    - Conversion: logit(orders / traffic) - B&M ONLY with has_traffic_data=1
      - WEB: conversion = NaN (no physical traffic)
      - B&M without valid traffic: conversion = NaN (unreliable data)

    Args:
        horizons: Forecast horizons to generate (default: 1-52 weeks)
        features_t0: DEPRECATED - Use include_features=True instead
                     Optional DataFrame with features observed at origin_week_date (t0).
        features_tfuture: DEPRECATED - Use include_features=True instead
                          Optional DataFrame with known-in-advance features at target_week_date (t0+h).
        drop_missing_targets: If True, drop rows with missing required targets (channel-aware logic)
        validate: If True, validate output against schema and raise warnings for issues
        include_features: If True, run EPIC 4 feature engineering pipeline (recommended)
                          If False, only generate keys and targets (Epic 3 only)
        model_variant: Which model variant to build when include_features=True:
                       - 'B' (default): Full feature set for production forecasts
                       - 'A': Actionable levers only for SHAP/explainability
        include_crm: Whether to include CRM demographic features (TIER 3, optional)

    Returns:
        DataFrame with columns (see schema.py for full specification):
        - Keys: profit_center_nbr, dma, channel, origin_week_date, horizon, target_week_date
        - Targets: label_log_sales, label_log_aov, label_logit_conversion
        - Metadata: has_traffic_data
        - Features (if include_features=True): ~100-120 features from [FE-01] through [FE-08]

    Example:
        >>> # Epic 3 only (keys + targets)
        >>> df = build_canonical_training_table(horizons=range(1, 13))
        >>>
        >>> # Epic 4 with Model B features (production)
        >>> df = build_canonical_training_table(
        ...     horizons=range(1, 53),
        ...     include_features=True,
        ...     model_variant='B'
        ... )
        >>>
        >>> # Epic 4 with Model A features (SHAP/explainability)
        >>> df = build_canonical_training_table(
        ...     include_features=True,
        ...     model_variant='A'
        ... )

    Schema Version: {schema_version}

    See Also:
        - forecasting.regressor.etl.schema: Full schema definition and validation
        - forecasting.regressor.features: EPIC 4 feature engineering modules
        - ENGINEERING_INSTRUCTIONS.md: Section 4 (Canonical Table), Section 5 (Features)
        - PROJECT_PLAN.md: Epic 3 (ETL) & Epic 4 (Feature Engineering)
    """
    # Load and prepare data with correct sources
    sales = _prepare_sales_with_dma()

    # Explode history to create (store, channel, origin, horizon, target) rows
    exploded = explode_history(sales, horizons=horizons)

    # Attach targets with channel-aware calculation logic
    with_targets = attach_targets(exploded, sales, drop_missing_targets=drop_missing_targets)

    # EPIC 4: Attach features if requested
    if include_features:
        # Use new EPIC 4 feature engineering pipeline
        full = attach_all_features(
            with_targets,
            model_variant=model_variant,
            include_crm=include_crm
        )
    else:
        # Legacy: Use old attach_features() for backward compatibility
        full = attach_features(with_targets, features_t0=features_t0, features_tfuture=features_tfuture)

    # Order columns for readability
    key_cols = [
        "profit_center_nbr",
        "dma",
        "channel",
        "origin_week_date",
        "horizon",
        "target_week_date",
    ]
    label_cols = ["label_log_sales", "label_log_aov", "label_logit_conversion"]
    metadata_cols = ["has_traffic_data"]
    raw_cols = ["total_sales", "order_count", "store_traffic", "aur"]

    # Preserve all columns but order them logically
    existing_cols = set(full.columns)
    ordered_cols = key_cols + label_cols
    ordered_cols += [c for c in metadata_cols if c in existing_cols]
    ordered_cols += [c for c in raw_cols if c in existing_cols]
    feature_cols = [c for c in full.columns if c not in ordered_cols]
    ordered_cols += feature_cols

    result = full[ordered_cols]

    # Validate schema if requested
    if validate:
        from app.regressor.etl.schema import validate_canonical_table
        errors = validate_canonical_table(result, strict=False, check_ranges=True)
        if errors:
            warning_msg = (
                f"Canonical table validation found {len(errors)} issue(s):\n"
                + "\n".join(f"  - {err}" for err in errors)
            )
            warnings.warn(warning_msg, UserWarning)

    return result


# Update docstring to include schema version (lazy import to avoid circular dependency)
def _update_docstring():
    from app.regressor.etl.schema import SCHEMA_VERSION
    build_canonical_training_table.__doc__ = build_canonical_training_table.__doc__.format(
        schema_version=SCHEMA_VERSION
    )

_update_docstring()
