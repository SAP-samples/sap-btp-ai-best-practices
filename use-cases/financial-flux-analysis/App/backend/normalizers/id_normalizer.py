"""
ID Normalizer — safe string normalization for SAP identifier columns.

CRITICAL RULES:
- SAP identifiers (G/L Account, Profit Center, Cost Center, etc.) are STRINGS.
- NEVER cast to int or float.
- NEVER strip non-digit characters globally.
- ALWAYS preserve leading zeros.
- ALWAYS strip surrounding whitespace only.
"""

import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


def normalize_id_column(series: pd.Series, col_name: str = "") -> pd.Series:
    """
    Normalize a SAP identifier column to clean string dtype.

    - Non-null values → str, strip surrounding whitespace only.
    - NaN / None / pd.NA → remain NaN (never converted to "nan").
    - Leading zeros are PRESERVED.
    - No digit stripping, no regex cleanup, no numeric casting.
    - Logs a WARNING if any value changed (e.g. float suffix ".0" detected).
    """
    if series.empty:
        return series

    non_null = series.notna()
    result = series.copy().astype(object)

    if non_null.any():
        raw_str   = series[non_null].astype(str)
        stripped  = raw_str.str.strip()

        # Detect float artefacts: "61002000.0" → should be "61002000"
        # Fix: remove trailing ".0" only when the value is otherwise all-digits
        def _fix_float_artefact(val: str) -> str:
            if val.endswith(".0") and val[:-2].isdigit():
                fixed = val[:-2]
                logger.warning(
                    "ID column '%s': float artefact detected — '%s' → '%s'. "
                    "HANA column should be NVARCHAR, not numeric.",
                    col_name, val, fixed,
                )
                return fixed
            return val

        cleaned = stripped.apply(_fix_float_artefact)

        # Log any remaining changes
        changed = raw_str != cleaned
        if changed.any():
            samples = list(zip(
                series[non_null][changed].head(5).tolist(),
                cleaned[changed].head(5).tolist(),
            ))
            logger.warning(
                "ID column '%s': %d value(s) changed during normalization. "
                "Samples (original → normalized): %s",
                col_name, int(changed.sum()), samples,
            )

        result[non_null] = cleaned

    return result


def normalize_id_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Apply normalize_id_column to a list of columns in a DataFrame.
    Skips columns that don't exist in the DataFrame.
    Returns a copy with normalized columns.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = normalize_id_column(df[col], col)
        else:
            logger.debug("normalize_id_columns: column '%s' not found — skipped.", col)
    return df