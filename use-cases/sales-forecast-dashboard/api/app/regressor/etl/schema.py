"""
Canonical Training Table Schema Definition (ETL-01).

This module defines the exact schema for the canonical training table,
including column specifications, data types, validation rules, and documentation.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ColumnSpec:
    """Specification for a single column in the canonical table."""

    name: str
    dtype: str
    description: str
    required: bool = True
    channel_specific: Optional[str] = None  # "B&M", "WEB", or None for both
    value_range: Optional[tuple] = None


# === CANONICAL TABLE SCHEMA ===

SCHEMA_VERSION = "1.0.0"

# === KEY COLUMNS ===
KEY_COLUMNS = [
    ColumnSpec(
        name="profit_center_nbr",
        dtype="int64",
        description="Store identifier (unique profit center number)",
        required=True,
    ),
    ColumnSpec(
        name="dma",
        dtype="object",
        description="Designated Market Area (geographic market identifier)",
        required=True,
    ),
    ColumnSpec(
        name="channel",
        dtype="object",
        description="Sales channel: 'B&M' (Brick-and-Mortar) or 'WEB' (E-commerce)",
        required=True,
    ),
    ColumnSpec(
        name="origin_week_date",
        dtype="datetime64[ns]",
        description="Origin week (t0): Last week of known data at prediction time",
        required=True,
    ),
    ColumnSpec(
        name="horizon",
        dtype="int64",
        description="Forecast horizon in weeks (1 to 52)",
        required=True,
        value_range=(1, 52),
    ),
    ColumnSpec(
        name="target_week_date",
        dtype="datetime64[ns]",
        description="Target week (t0 + h): Week being forecasted",
        required=True,
    ),
]

# === TARGET COLUMNS (Labels) ===
TARGET_COLUMNS = [
    ColumnSpec(
        name="label_log_sales",
        dtype="float64",
        description="Natural log of total weekly sales ($). Both channels.",
        required=True,
    ),
    ColumnSpec(
        name="label_log_aov",
        dtype="float64",
        description="Natural log of Average Order Value (sales / orders). Both channels.",
        required=True,
    ),
    ColumnSpec(
        name="label_logit_conversion",
        dtype="float64",
        description=(
            "Logit-transformed conversion rate (orders / traffic). "
            "B&M ONLY with has_traffic_data=1. NaN for WEB or B&M without valid traffic."
        ),
        required=False,  # Not required for WEB or B&M without traffic
        channel_specific="B&M",
    ),
]

# === METADATA COLUMNS ===
METADATA_COLUMNS = [
    ColumnSpec(
        name="has_traffic_data",
        dtype="int64",
        description=(
            "Binary flag: 1 if store has reliable traffic data (B&M only), 0 otherwise. "
            "Used to filter Conversion Model training. Always 0 for WEB."
        ),
        required=True,
        value_range=(0, 1),
    ),
]

# === RAW TARGET COLUMNS (for debugging/validation) ===
RAW_TARGET_COLUMNS = [
    ColumnSpec(
        name="total_sales",
        dtype="float64",
        description="Raw total weekly sales ($) before log transformation",
        required=False,
    ),
    ColumnSpec(
        name="order_count",
        dtype="float64",
        description="Number of transactions/orders",
        required=False,
    ),
    ColumnSpec(
        name="store_traffic",
        dtype="float64",
        description="Physical visitor count (B&M only). NaN for WEB.",
        required=False,
        channel_specific="B&M",
    ),
    ColumnSpec(
        name="aur",
        dtype="float64",
        description="Average Unit Retail (sales / units) - NOT used for AOV",
        required=False,
    ),
]

# === FULL SCHEMA ===
CANONICAL_TABLE_SCHEMA: List[ColumnSpec] = (
    KEY_COLUMNS + TARGET_COLUMNS + METADATA_COLUMNS + RAW_TARGET_COLUMNS
)


def get_schema_dict() -> Dict[str, ColumnSpec]:
    """Return schema as a dictionary keyed by column name."""
    return {col.name: col for col in CANONICAL_TABLE_SCHEMA}


def get_required_columns() -> List[str]:
    """Return list of required column names."""
    return [col.name for col in CANONICAL_TABLE_SCHEMA if col.required]


def get_channel_specific_columns(channel: str) -> List[str]:
    """Return columns specific to a channel (B&M or WEB)."""
    return [
        col.name
        for col in CANONICAL_TABLE_SCHEMA
        if col.channel_specific == channel or col.channel_specific is None
    ]


# === VALIDATION FUNCTIONS ===


def validate_canonical_table(
    df: pd.DataFrame,
    strict: bool = False,
    check_ranges: bool = True,
) -> List[str]:
    """
    Validate a canonical training table against the schema.

    Args:
        df: DataFrame to validate
        strict: If True, require ALL schema columns (including optional). If False, only check required.
        check_ranges: If True, validate value ranges (horizon, has_traffic_data, etc.)

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    schema_dict = get_schema_dict()

    # Check required columns present
    required_cols = get_required_columns() if not strict else [col.name for col in CANONICAL_TABLE_SCHEMA]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {sorted(missing_cols)}")

    # Check data types for present columns
    for col_name in df.columns:
        if col_name in schema_dict:
            spec = schema_dict[col_name]
            expected_dtype = spec.dtype
            actual_dtype = str(df[col_name].dtype)

            # Allow some flexibility in dtype matching
            dtype_compatible = False
            if expected_dtype == "int64" and actual_dtype in ["int64", "Int64"]:
                dtype_compatible = True
            elif expected_dtype == "float64" and actual_dtype in ["float64", "float32"]:
                dtype_compatible = True
            elif expected_dtype == "object" and actual_dtype == "object":
                dtype_compatible = True
            elif expected_dtype == "datetime64[ns]" and "datetime64" in actual_dtype:
                dtype_compatible = True

            if not dtype_compatible:
                errors.append(f"Column '{col_name}': expected dtype {expected_dtype}, got {actual_dtype}")

    # Check value ranges
    if check_ranges:
        for spec in CANONICAL_TABLE_SCHEMA:
            if spec.name in df.columns and spec.value_range is not None:
                min_val, max_val = spec.value_range
                col_data = df[spec.name].dropna()
                if len(col_data) > 0:
                    actual_min = col_data.min()
                    actual_max = col_data.max()
                    if actual_min < min_val or actual_max > max_val:
                        errors.append(
                            f"Column '{spec.name}': values out of range [{min_val}, {max_val}]. "
                            f"Found [{actual_min}, {actual_max}]"
                        )

    # Check key uniqueness (no duplicate rows for same store/channel/origin/horizon)
    key_cols = [col.name for col in KEY_COLUMNS]
    if all(col in df.columns for col in key_cols):
        duplicates = df[key_cols].duplicated().sum()
        if duplicates > 0:
            errors.append(f"Found {duplicates} duplicate key combinations (store/channel/origin/horizon)")

    # Check channel-specific rules
    if "channel" in df.columns:
        for channel in df["channel"].unique():
            channel_df = df[df["channel"] == channel]

            # WEB should have NaN conversion
            if channel == "WEB" and "label_logit_conversion" in channel_df.columns:
                non_nan_conversion = channel_df["label_logit_conversion"].notna().sum()
                if non_nan_conversion > 0:
                    errors.append(
                        f"Channel 'WEB' has {non_nan_conversion} non-NaN conversion values. "
                        "Conversion should be NaN for WEB (no physical traffic)."
                    )

            # B&M with has_traffic_data=0 should have NaN conversion
            if channel != "WEB" and "has_traffic_data" in channel_df.columns:
                no_traffic = channel_df[channel_df["has_traffic_data"] == 0]
                if len(no_traffic) > 0 and "label_logit_conversion" in no_traffic.columns:
                    non_nan_conv = no_traffic["label_logit_conversion"].notna().sum()
                    if non_nan_conv > 0:
                        errors.append(
                            f"Channel '{channel}' with has_traffic_data=0 has {non_nan_conv} non-NaN conversion values. "
                            "Conversion should be NaN when traffic data is unreliable."
                        )

    # Check for infinite values in targets
    target_cols = [col.name for col in TARGET_COLUMNS if col.name in df.columns]
    for col_name in target_cols:
        inf_count = np.isinf(df[col_name]).sum()
        if inf_count > 0:
            errors.append(f"Column '{col_name}' contains {inf_count} infinite values")

    return errors


def print_schema_documentation():
    """Print human-readable schema documentation."""
    print("=" * 80)
    print(f"CANONICAL TRAINING TABLE SCHEMA (Version {SCHEMA_VERSION})")
    print("=" * 80)
    print()

    sections = [
        ("KEY COLUMNS", KEY_COLUMNS),
        ("TARGET COLUMNS (Labels)", TARGET_COLUMNS),
        ("METADATA COLUMNS", METADATA_COLUMNS),
        ("RAW TARGET COLUMNS (Optional, for debugging)", RAW_TARGET_COLUMNS),
    ]

    for section_name, columns in sections:
        print(f"\n{section_name}")
        print("-" * 80)
        for col in columns:
            required_str = "[REQUIRED]" if col.required else "[OPTIONAL]"
            channel_str = f" [{col.channel_specific} only]" if col.channel_specific else ""
            range_str = f" Range: {col.value_range}" if col.value_range else ""

            print(f"\n  {col.name} ({col.dtype}) {required_str}{channel_str}{range_str}")
            print(f"    {col.description}")

    print("\n" + "=" * 80)
    print("VALIDATION RULES")
    print("=" * 80)
    print("""
  1. WEB channel: label_logit_conversion must be NaN (no physical traffic)
  2. B&M channel with has_traffic_data=0: label_logit_conversion must be NaN
  3. B&M channel with has_traffic_data=1: label_logit_conversion should be valid
  4. horizon must be in range [1, 52]
  5. has_traffic_data must be 0 or 1
  6. No duplicate (store, channel, origin_week_date, horizon) combinations
  7. No infinite values in target columns
    """)


if __name__ == "__main__":
    # Print schema documentation when run as script
    print_schema_documentation()
