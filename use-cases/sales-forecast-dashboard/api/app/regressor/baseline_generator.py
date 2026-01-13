"""
Baseline Generator Module

Generates "business as usual" forecast data for future horizons using
seasonal naive approach. Produces both Model A and Model B compatible outputs.

Usage:
    from app.regressor.baseline_generator import BaselineGenerator, BaselineConfig

    config = BaselineConfig(
        model_b_path=Path("final_data/model_b.csv"),
        output_dir=Path("output/baselines"),
    )
    generator = BaselineGenerator(config)
    model_a_df, model_b_df = generator.generate()

Author: EPIC 4 Feature Engineering
Status: Step 3 - Baseline Generator
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import warnings

import pandas as pd
import numpy as np

from app.regressor.io_utils import load_holiday_calendar, load_store_master
from app.regressor.seasonality import compute_dma_seasonality
from app.regressor.features.model_views import (
    MODEL_A_FEATURES,
    MODEL_A_BM_ONLY_FEATURES,
    build_model_a_features,
    build_model_b_features,
)


# Autoregressive features that should be frozen at t0 (origin_week_date)
AR_FEATURES = [
    # Sales dynamics (B&M)
    "log_sales_lag_1", "log_sales_lag_4", "log_sales_lag_13", "log_sales_lag_52",
    "log_sales_roll_mean_4", "log_sales_roll_mean_8", "log_sales_roll_mean_13",
    "vol_log_sales_13",
    # AOV dynamics
    "AOV_roll_mean_8", "AOV_roll_mean_13",
    # Conversion dynamics (B&M only)
    "ConversionRate_lag_1", "ConversionRate_lag_4",
    "ConversionRate_roll_mean_4", "ConversionRate_roll_mean_8", "ConversionRate_roll_mean_13",
    # Web traffic dynamics (WEB only)
    "allocated_web_traffic_lag_1", "allocated_web_traffic_lag_4", "allocated_web_traffic_lag_13",
    "allocated_web_traffic_roll_mean_4", "allocated_web_traffic_roll_mean_8", "allocated_web_traffic_roll_mean_13",
    # Web sales dynamics (WEB only)
    "log_web_sales_lag_1", "log_web_sales_lag_4", "log_web_sales_lag_13",
    "log_web_sales_roll_mean_4", "log_web_sales_roll_mean_8", "log_web_sales_roll_mean_13",
    "vol_log_web_sales_13",
    "log_web_sales_roll_mean_web_4",
    # Cannibalization (frozen at origin per user decision)
    "cannibalization_pressure", "min_dist_new_store_km",
    "num_new_stores_within_10mi_last_52wk", "num_new_stores_within_20mi_last_52wk",
]

# Business lever features that use seasonal naive approach (target - 52 weeks)
BUSINESS_LEVER_FEATURES = [
    "pct_primary_financing_roll_mean_4",
    "pct_secondary_financing_roll_mean_4",
    "pct_tertiary_financing_roll_mean_4",
    "pct_omni_channel_roll_mean_4",
    "pct_value_product_roll_mean_4",
    "pct_premium_product_roll_mean_4",
    "pct_white_glove_roll_mean_4",
    "staffing_unique_associates_roll_mean_4",
    "staffing_hours_roll_mean_4",
]

# Market signal features that use seasonal naive approach
MARKET_SIGNAL_FEATURES = [
    "brand_awareness_dma_roll_mean_4",
    "brand_consideration_dma_roll_mean_4",
]

# Static store features that come from store master
STATIC_FEATURES = [
    "is_outlet", "is_comp_store", "is_new_store",
    "merchandising_sf", "sq_ft", "store_design_sf",
    "weeks_since_open", "weeks_since_open_capped_13", "weeks_since_open_capped_52",
]


@dataclass
class BaselineConfig:
    """Configuration for baseline generation."""
    model_b_path: Path  # Path to existing model_b.csv
    output_dir: Path = field(default_factory=lambda: Path("output/baselines"))
    horizons: range = field(default_factory=lambda: range(1, 53))  # 1-52 weeks
    channels: List[str] = field(default_factory=lambda: ["B&M", "WEB"])
    origin_date: Optional[str] = None  # If None, use last available per store
    store_ids: Optional[List[int]] = None  # Filter to specific stores (early filtering)
    dmas: Optional[List[str]] = None  # Filter to specific DMAs (early filtering)


class BaselineGenerator:
    """
    Generate baseline forecasts using seasonal naive approach.

    The generator creates future forecast rows where:
    - Autoregressive features: Frozen at the last known origin date (t0)
    - Calendar features: Computed fresh for each target week (t0+h)
    - Business levers: Seasonal naive (use values from target - 52 weeks)
    - Market signals: Seasonal naive (use values from target - 52 weeks)
    - Cannibalization: Frozen at origin (current competitive state)
    """

    def __init__(self, config: BaselineConfig):
        self.config = config
        self._model_b_df: Optional[pd.DataFrame] = None
        self._holiday_calendar: Optional[pd.DataFrame] = None
        self._store_master: Optional[pd.DataFrame] = None

    def generate(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate baseline forecasts.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (model_a_df, model_b_df) - Baseline datasets for explainability and prediction
        """
        print("Loading data...")
        self._load_data()

        print("Finding last origin dates per store-channel...")
        origin_dates = self._find_last_origin_dates()
        print(f"  Found {len(origin_dates)} store-channel combinations")

        print("Creating baseline skeleton...")
        skeleton = self._create_baseline_skeleton(origin_dates)
        print(f"  Created {len(skeleton):,} rows ({len(self.config.horizons)} horizons)")

        print("Attaching autoregressive features (frozen at t0)...")
        df = self._attach_autoregressive_features(skeleton)

        print("Attaching static store features...")
        df = self._attach_static_features(df)

        print("Attaching calendar features (fresh for t0+h)...")
        df = self._attach_calendar_features(df)

        print("Attaching business levers (seasonal naive from t-52)...")
        df = self._attach_seasonal_naive_features(df)

        print("Attaching market signals (seasonal naive from t-52)...")
        df = self._attach_market_signals(df)

        print("Handling missing values with DMA fallback...")
        df = self._handle_missing_values(df)

        print("Building model A and B views...")
        model_a_df = build_model_a_features(df)
        model_b_df = build_model_b_features(df)

        print(f"Model A: {len(model_a_df.columns)} columns, {len(model_a_df):,} rows")
        print(f"Model B: {len(model_b_df.columns)} columns, {len(model_b_df):,} rows")

        return model_a_df, model_b_df

    def _load_data(self) -> None:
        """Load all required data sources."""
        # Load model_b.csv
        self._model_b_df = pd.read_csv(self.config.model_b_path)

        # Ensure datetime columns
        for col in ["origin_week_date", "target_week_date"]:
            if col in self._model_b_df.columns:
                self._model_b_df[col] = pd.to_datetime(self._model_b_df[col])

        # Load holiday calendar
        self._holiday_calendar = load_holiday_calendar()

        # Load store master for static features
        self._store_master = load_store_master()

    def _find_last_origin_dates(self) -> pd.DataFrame:
        """
        Find last available origin_week_date per store-channel.

        Applies early filtering based on store_ids and dmas from config
        to avoid generating rows for stores that will be filtered out later.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: profit_center_nbr, channel, dma, last_origin_date
        """
        # Start with the full dataset
        df = self._model_b_df

        # Apply early filtering by store_ids if specified
        if self.config.store_ids:
            df = df[df["profit_center_nbr"].isin(self.config.store_ids)]

        # Apply early filtering by dmas if specified
        if self.config.dmas:
            df = df[df["dma"].isin(self.config.dmas)]

        # Apply channel filter
        if self.config.channels:
            df = df[df["channel"].isin(self.config.channels)]

        if self.config.origin_date:
            # Use user-specified origin date for all stores
            origin_dt = pd.to_datetime(self.config.origin_date)
            stores = df[["profit_center_nbr", "channel", "dma"]].drop_duplicates()
            stores["last_origin_date"] = origin_dt
            return stores

        # Otherwise find max origin_week_date per store-channel
        return (
            df
            .groupby(["profit_center_nbr", "channel", "dma"], as_index=False)["origin_week_date"]
            .max()
            .rename(columns={"origin_week_date": "last_origin_date"})
        )

    def _create_baseline_skeleton(self, origin_dates: pd.DataFrame) -> pd.DataFrame:
        """
        Create skeleton with all horizon rows.

        Parameters
        ----------
        origin_dates : pd.DataFrame
            DataFrame with store-channel origin dates

        Returns
        -------
        pd.DataFrame
            Skeleton with columns: profit_center_nbr, channel, dma, origin_week_date, horizon, target_week_date
        """
        rows = []
        for _, row in origin_dates.iterrows():
            origin = row["last_origin_date"]
            for h in self.config.horizons:
                target = origin + pd.Timedelta(weeks=h)
                rows.append({
                    "profit_center_nbr": row["profit_center_nbr"],
                    "channel": row["channel"],
                    "dma": row["dma"],
                    "origin_week_date": origin,
                    "horizon": h,
                    "target_week_date": target,
                })
        return pd.DataFrame(rows)

    def _attach_autoregressive_features(self, skeleton: pd.DataFrame) -> pd.DataFrame:
        """
        Attach autoregressive features frozen at t0.

        These features represent the "state of the world" at forecast time and should
        be constant across all horizons for the same store-channel.
        """
        # Get the most recent horizon=1 row for each store-channel
        # (horizon=1 is closest to origin and has the AR values we want)
        ar_source = self._model_b_df[self._model_b_df["horizon"] == 1].copy()

        # Get the most recent origin for each store-channel
        ar_source = (
            ar_source
            .sort_values(["profit_center_nbr", "channel", "origin_week_date"])
            .groupby(["profit_center_nbr", "channel"], as_index=False)
            .last()
        )

        # Select only AR columns that exist
        ar_cols = [c for c in AR_FEATURES if c in ar_source.columns]
        ar_subset = ar_source[["profit_center_nbr", "channel"] + ar_cols]

        return skeleton.merge(ar_subset, on=["profit_center_nbr", "channel"], how="left")

    def _attach_static_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attach static store features.

        Most static features come from the store master or the last known model_b row.
        weeks_since_open is computed dynamically based on target_week_date.
        """
        # Get static features from the most recent model_b rows
        static_source = self._model_b_df[self._model_b_df["horizon"] == 1].copy()
        static_source = (
            static_source
            .sort_values(["profit_center_nbr", "channel", "origin_week_date"])
            .groupby(["profit_center_nbr", "channel"], as_index=False)
            .last()
        )

        # Get static columns that exist
        static_cols = [c for c in STATIC_FEATURES if c in static_source.columns]
        static_subset = static_source[["profit_center_nbr", "channel"] + static_cols]

        df = df.merge(static_subset, on=["profit_center_nbr", "channel"], how="left")

        # Recompute weeks_since_open based on target_week_date
        if "date_opened" not in df.columns:
            # Try to get date_opened from store master
            store_dates = self._store_master[["profit_center_nbr", "date_opened"]].drop_duplicates()
            df = df.merge(store_dates, on="profit_center_nbr", how="left")

        if "date_opened" in df.columns:
            df["date_opened"] = pd.to_datetime(df["date_opened"])
            df["weeks_since_open"] = (
                (df["target_week_date"] - df["date_opened"]).dt.days / 7
            ).clip(lower=0).fillna(0).astype(int)
            df["weeks_since_open_capped_13"] = df["weeks_since_open"].clip(upper=13)
            df["weeks_since_open_capped_52"] = df["weeks_since_open"].clip(upper=52)
            df["is_new_store"] = (df["weeks_since_open"] < 52).astype(int)
            df["is_comp_store"] = (df["weeks_since_open"] >= 60).astype(int)
            df = df.drop(columns=["date_opened"])

        return df

    def _attach_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attach fresh calendar features for each target_week_date.

        These are known-in-advance features that vary by horizon.
        """
        # Week of year with cyclical encoding
        df["woy"] = df["target_week_date"].dt.isocalendar().week.astype(int)
        df["sin_woy"] = np.sin(2 * np.pi * df["woy"] / 52)
        df["cos_woy"] = np.cos(2 * np.pi * df["woy"] / 52)

        # Calendar components
        df["month"] = df["target_week_date"].dt.month
        df["quarter"] = df["target_week_date"].dt.quarter
        df["fiscal_year"] = df["target_week_date"].dt.year
        df["fiscal_period"] = ((df["woy"] - 1) // 4 + 1).clip(upper=13)

        # Holiday features
        df = self._attach_holiday_features(df)

        # DMA seasonality
        df = self._attach_dma_seasonality(df)

        return df

    def _attach_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach holiday-related features from the holiday calendar."""
        if self._holiday_calendar is None or self._holiday_calendar.empty:
            # Create empty holiday features if no calendar available
            df["is_holiday"] = 0
            df["is_xmas_window"] = 0
            df["is_black_friday_window"] = 0
            df["weeks_to_holiday"] = 52
            return df

        # Ensure week_start is datetime in holiday calendar
        hcal = self._holiday_calendar.copy()
        if "week_start" in hcal.columns:
            hcal["week_start"] = pd.to_datetime(hcal["week_start"])

        # Create holiday lookup by week
        holiday_lookup = hcal.set_index("week_start")

        # Map target_week_date to nearest Monday for lookup
        df["_week_start"] = df["target_week_date"] - pd.to_timedelta(
            df["target_week_date"].dt.dayofweek, unit="D"
        )

        # Map holiday features
        for col in ["is_holiday_week", "is_pre_holiday_1wk", "is_pre_holiday_2wk", "is_pre_holiday_3wk"]:
            if col in holiday_lookup.columns:
                df[col] = df["_week_start"].map(holiday_lookup[col]).fillna(0).astype(int)

        # Rename is_holiday_week to is_holiday
        if "is_holiday_week" in df.columns:
            df = df.rename(columns={"is_holiday_week": "is_holiday"})
        else:
            df["is_holiday"] = 0

        # Compute is_xmas_window (Christmas week +/- 2 weeks)
        df["_xmas_week"] = (df["woy"] >= 51) | (df["woy"] <= 1)
        df["is_xmas_window"] = df["_xmas_week"].astype(int)

        # Compute is_black_friday_window (Thanksgiving week +/- 1 week, typically woy 47-48)
        df["is_black_friday_window"] = df["woy"].isin([46, 47, 48]).astype(int)

        # weeks_to_holiday: simplified as distance to next major holiday
        # For baseline, use a simple approximation based on week of year
        # Major holidays: Thanksgiving (woy 47), Christmas (woy 52), Memorial Day (woy 22), July 4 (woy 27)
        major_holidays = [22, 27, 47, 52]
        def weeks_to_next_holiday(woy):
            min_dist = 52
            for h in major_holidays:
                dist = (h - woy) % 52
                if dist == 0:
                    dist = 0  # We are on the holiday
                min_dist = min(min_dist, dist)
            return min_dist
        df["weeks_to_holiday"] = df["woy"].apply(weeks_to_next_holiday)

        # Clean up temporary columns
        df = df.drop(columns=["_week_start", "_xmas_week"], errors="ignore")

        return df

    def _attach_dma_seasonality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attach DMA seasonality weights.

        Uses historical seasonality patterns from the model_b data.
        """
        # Get seasonality from existing model_b data
        if "dma_seasonal_weight" in self._model_b_df.columns:
            # Extract unique DMA-woy seasonality weights
            seas = (
                self._model_b_df[["dma", "channel", "target_week_date", "dma_seasonal_weight"]]
                .copy()
            )
            seas["woy"] = seas["target_week_date"].dt.isocalendar().week.astype(int)

            # Average by DMA-channel-woy
            seas_avg = (
                seas
                .groupby(["dma", "channel", "woy"], as_index=False)["dma_seasonal_weight"]
                .mean()
            )

            # Merge with baseline
            df = df.merge(seas_avg, on=["dma", "channel", "woy"], how="left")

        # Fill missing with 1/52 (uniform)
        if "dma_seasonal_weight" not in df.columns:
            df["dma_seasonal_weight"] = 1.0 / 52
        else:
            df["dma_seasonal_weight"] = df["dma_seasonal_weight"].fillna(1.0 / 52)

        return df

    def _attach_seasonal_naive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attach business levers from target - 52 weeks (seasonal naive).

        For each target week, we use the business lever values from the same
        week one year ago as the "business as usual" baseline.
        """
        # Create lookback date (target - 52 weeks)
        df["_lookback_date"] = df["target_week_date"] - pd.Timedelta(weeks=52)

        # Get available business lever columns from model_b
        lever_cols = [c for c in BUSINESS_LEVER_FEATURES if c in self._model_b_df.columns]

        if not lever_cols:
            warnings.warn("No business lever features found in model_b.csv")
            df = df.drop(columns=["_lookback_date"])
            return df

        # Extract historical values for lookback
        historical = self._model_b_df[
            ["profit_center_nbr", "channel", "target_week_date"] + lever_cols
        ].copy()
        historical = historical.drop_duplicates(subset=["profit_center_nbr", "channel", "target_week_date"])

        # Merge on store-channel and lookback date
        df = df.merge(
            historical.rename(columns={"target_week_date": "_lookback_date"}),
            on=["profit_center_nbr", "channel", "_lookback_date"],
            how="left",
            suffixes=("", "_seasonal")
        )

        # Use seasonal values where available
        for feat in lever_cols:
            seasonal_col = f"{feat}_seasonal"
            if seasonal_col in df.columns:
                # Only use seasonal value if current value is missing
                if feat not in df.columns:
                    df[feat] = df[seasonal_col]
                else:
                    df[feat] = df[feat].fillna(df[seasonal_col])
                df = df.drop(columns=[seasonal_col])

        df = df.drop(columns=["_lookback_date"])
        return df

    def _attach_market_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attach market signals (awareness/consideration) using seasonal naive approach.
        """
        # Create lookback date (target - 52 weeks)
        df["_lookback_date"] = df["target_week_date"] - pd.Timedelta(weeks=52)

        # Get available market signal columns
        signal_cols = [c for c in MARKET_SIGNAL_FEATURES if c in self._model_b_df.columns]

        if not signal_cols:
            warnings.warn("No market signal features found in model_b.csv")
            df = df.drop(columns=["_lookback_date"])
            return df

        # Extract historical values for lookback
        historical = self._model_b_df[
            ["profit_center_nbr", "channel", "target_week_date"] + signal_cols
        ].copy()
        historical = historical.drop_duplicates(subset=["profit_center_nbr", "channel", "target_week_date"])

        # Merge on store-channel and lookback date
        df = df.merge(
            historical.rename(columns={"target_week_date": "_lookback_date"}),
            on=["profit_center_nbr", "channel", "_lookback_date"],
            how="left",
            suffixes=("", "_seasonal")
        )

        # Use seasonal values
        for feat in signal_cols:
            seasonal_col = f"{feat}_seasonal"
            if seasonal_col in df.columns:
                if feat not in df.columns:
                    df[feat] = df[seasonal_col]
                else:
                    df[feat] = df[feat].fillna(df[seasonal_col])
                df = df.drop(columns=[seasonal_col])

        df = df.drop(columns=["_lookback_date"])
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply DMA average fallback for missing seasonal naive values.

        For stores with less than 52 weeks of history, fill missing values
        with the DMA average for the same week-of-year.
        """
        # Features to fill with DMA fallback
        fallback_features = BUSINESS_LEVER_FEATURES + MARKET_SIGNAL_FEATURES

        for feat in fallback_features:
            if feat not in df.columns:
                continue

            # Count missing before
            missing_before = df[feat].isna().sum()
            if missing_before == 0:
                continue

            # Compute DMA-channel-woy averages from historical data
            if feat in self._model_b_df.columns:
                hist = self._model_b_df.copy()
                hist["_woy"] = hist["target_week_date"].dt.isocalendar().week.astype(int)

                dma_avg = (
                    hist
                    .groupby(["dma", "channel", "_woy"], as_index=False)[feat]
                    .mean()
                    .rename(columns={feat: f"_dma_avg_{feat}"})
                )

                # Merge and fill missing
                df["_woy"] = df["woy"]
                df = df.merge(dma_avg, on=["dma", "channel", "_woy"], how="left")
                df[feat] = df[feat].fillna(df[f"_dma_avg_{feat}"])
                df = df.drop(columns=[f"_dma_avg_{feat}", "_woy"], errors="ignore")

                # Count missing after
                missing_after = df[feat].isna().sum()
                if missing_after > 0:
                    # Final fallback: channel median
                    channel_median = self._model_b_df.groupby("channel")[feat].median()
                    df[feat] = df.apply(
                        lambda row: channel_median.get(row["channel"], 0) if pd.isna(row[feat]) else row[feat],
                        axis=1
                    )
                    print(f"  {feat}: Filled {missing_before - missing_after} with DMA avg, "
                          f"{missing_after} with channel median")

        return df

    def save(self, model_a_df: pd.DataFrame, model_b_df: pd.DataFrame) -> Tuple[Path, Path]:
        """
        Save baseline datasets to output directory.

        Returns
        -------
        Tuple[Path, Path]
            Paths to (model_a_csv, model_b_csv)
        """
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        model_a_path = self.config.output_dir / "baseline_model_a.csv"
        model_b_path = self.config.output_dir / "baseline_model_b.csv"

        model_a_df.to_csv(model_a_path, index=False)
        model_b_df.to_csv(model_b_path, index=False)

        return model_a_path, model_b_path
