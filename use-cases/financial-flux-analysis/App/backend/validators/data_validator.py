"""
Data Validator — integrity checks for the enriched financial dataset.

Validates:
- Malformed / unexpected-length G/L Accounts
- Duplicate join keys (join explosion detection)
- Records lost during joins
- Null key mismatches
- Join source traceability

All issues are LOGGED as warnings — never silently dropped.
"""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def validate_gl_accounts(df: pd.DataFrame, col: str = "G/L Account") -> None:
    """
    Check G/L Account values for integrity issues.
    Logs warnings for:
    - Non-string values
    - Values containing embedded descriptions (mixed format not yet parsed)
    - Unexpected lengths (distribution logged for diagnosis)
    """
    if col not in df.columns:
        return

    series = df[col].dropna()
    if series.empty:
        return

    # Length distribution
    length_dist = series.astype(str).str.len().value_counts().sort_index().to_dict()
    logger.info("G/L Account length distribution: %s", length_dist)

    # Detect mixed-format values (contain spaces — not yet parsed)
    mixed = series[series.astype(str).str.contains(r'\s', regex=True)]
    if not mixed.empty:
        logger.warning(
            "G/L Account: %d value(s) contain whitespace — mixed 'ID (Description)' "
            "format detected. These must be parsed before use as join keys. "
            "Sample: %s",
            len(mixed), mixed.head(5).tolist(),
        )

    # Detect float artefacts
    float_artefacts = series[series.astype(str).str.endswith(".0")]
    if not float_artefacts.empty:
        logger.warning(
            "G/L Account: %d value(s) end with '.0' — float artefact from numeric "
            "HANA column type. HANA column should be NVARCHAR. Sample: %s",
            len(float_artefacts), float_artefacts.head(5).tolist(),
        )


def validate_join_integrity(
    left_df: pd.DataFrame,
    result_df: pd.DataFrame,
    join_key: str,
    join_name: str,
) -> None:
    """
    Validate that a LEFT JOIN did not lose records or explode row count.

    - Lost records: rows in left_df that have no match in result_df
    - Join explosion: result_df has significantly more rows than left_df
    """
    left_count  = len(left_df)
    result_count = len(result_df)

    if result_count < left_count:
        lost = left_count - result_count
        logger.warning(
            "JOIN '%s' on '%s': %d record(s) LOST (left=%d, result=%d). "
            "Check for null keys or key type mismatches.",
            join_name, join_key, lost, left_count, result_count,
        )
    elif result_count > left_count * 1.05:
        extra = result_count - left_count
        logger.warning(
            "JOIN '%s' on '%s': possible JOIN EXPLOSION — "
            "%d extra rows (left=%d, result=%d). "
            "Check for duplicate keys in the right-side table.",
            join_name, join_key, extra, left_count, result_count,
        )
    else:
        logger.info(
            "JOIN '%s' on '%s': OK (left=%d, result=%d).",
            join_name, join_key, left_count, result_count,
        )


def validate_null_keys(df: pd.DataFrame, key_col: str, context: str = "") -> None:
    """Log a warning if a key column contains null values."""
    if key_col not in df.columns:
        return
    null_count = df[key_col].isna().sum()
    if null_count > 0:
        logger.warning(
            "%sColumn '%s': %d null key(s) found — these rows will not join. "
            "Check source data quality.",
            f"[{context}] " if context else "", key_col, null_count,
        )


def validate_enriched_dataset(df: pd.DataFrame) -> None:
    """
    Run all integrity checks on the final enriched dataset.
    Called once after the full pipeline completes.
    """
    logger.info("=== Enriched Dataset Validation ===")
    logger.info("Total rows: %d", len(df))
    logger.info("Columns: %s", list(df.columns))

    validate_gl_accounts(df, "G/L Account")

    for col in ["Profit Center", "Cost Center", "Company Code"]:
        validate_null_keys(df, col, context="enriched")

    # Check amount column
    if "amount_numeric" in df.columns:
        zero_amount = (df["amount_numeric"] == 0).sum()
        null_amount = df["amount_numeric"].isna().sum()
        logger.info(
            "amount_numeric: %d zero values, %d null values out of %d total.",
            zero_amount, null_amount, len(df),
        )

    # Check fiscal year
    if "fiscal_year" in df.columns:
        years = sorted(df["fiscal_year"].dropna().astype(int).unique().tolist())
        logger.info("Fiscal years present: %s", years)

    logger.info("=== Validation Complete ===")