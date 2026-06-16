"""
Contributor Extraction Utilities.

Provides functions for extracting and formatting top SHAP contributors
for each prediction row.
"""

from typing import List, Optional

import numpy as np
import pandas as pd


def build_contributor_strings(
    shap_values: np.ndarray,
    feature_names: List[str],
    feature_values: pd.DataFrame,
    top_k: int = 3,
) -> pd.Series:
    """
    Build per-row strings of top contributors with signed SHAP values.

    For each row, identifies the top-k features by absolute SHAP value
    and formats them as "feature=value:+/-shap; ..." strings.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values array of shape (n_samples, n_features).
    feature_names : List[str]
        Names of features corresponding to columns in shap_values.
    feature_values : pd.DataFrame
        Feature values matrix (same shape as shap_values).
    top_k : int, optional
        Number of top contributors per row. Default is 3.

    Returns
    -------
    pd.Series
        String series with format "feature=value:+/-shap; feature=value:+/-shap; ..."

    Examples
    --------
    >>> contributor_strings = build_contributor_strings(
    ...     shap_values, feature_names, X, top_k=3
    ... )
    >>> print(contributor_strings[0])
    "horizon=4:+0.523; brand_awareness=0.75:-0.312; is_outlet=1:+0.187"
    """
    top_k = min(top_k, shap_values.shape[1])
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    X_reset = feature_values.reset_index(drop=True)

    contributor_strings = []
    for i, row in shap_df.iterrows():
        # Sort by absolute SHAP value
        abs_sorted = row.abs().sort_values(ascending=False)
        parts = []

        for feat in abs_sorted.index[:top_k]:
            shap_val = row[feat]
            raw_val = X_reset.at[i, feat]

            # Format value appropriately
            if isinstance(raw_val, float):
                if np.isnan(raw_val):
                    val_str = "NaN"
                elif abs(raw_val) >= 1000:
                    val_str = f"{raw_val:.0f}"
                elif abs(raw_val) >= 1:
                    val_str = f"{raw_val:.2f}"
                else:
                    val_str = f"{raw_val:.3g}"
            else:
                val_str = str(raw_val)

            parts.append(f"{feat}={val_str}:{shap_val:+.3f}")

        contributor_strings.append("; ".join(parts))

    return pd.Series(contributor_strings)


def get_top_contributors_dataframe(
    shap_values: np.ndarray,
    feature_names: List[str],
    feature_values: pd.DataFrame,
    top_k: int = 3,
    key_columns: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Create a DataFrame with top contributors for each row.

    Returns a structured DataFrame with separate columns for each
    contributor rank (top_1_feature, top_1_value, top_1_shap, etc.).

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values array of shape (n_samples, n_features).
    feature_names : List[str]
        Names of features.
    feature_values : pd.DataFrame
        Feature values matrix.
    top_k : int, optional
        Number of top contributors per row. Default is 3.
    key_columns : pd.DataFrame, optional
        Key columns to include in the output DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns for each contributor rank:
        - top_1_feature, top_1_value, top_1_shap
        - top_2_feature, top_2_value, top_2_shap
        - etc.
    """
    top_k = min(top_k, shap_values.shape[1])
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    X_reset = feature_values.reset_index(drop=True)

    # Initialize result columns
    result_data = {}
    for rank in range(1, top_k + 1):
        result_data[f"top_{rank}_feature"] = []
        result_data[f"top_{rank}_value"] = []
        result_data[f"top_{rank}_shap"] = []

    for i, row in shap_df.iterrows():
        abs_sorted = row.abs().sort_values(ascending=False)

        for rank, feat in enumerate(abs_sorted.index[:top_k], start=1):
            result_data[f"top_{rank}_feature"].append(feat)
            result_data[f"top_{rank}_value"].append(X_reset.at[i, feat])
            result_data[f"top_{rank}_shap"].append(row[feat])

    result_df = pd.DataFrame(result_data)

    if key_columns is not None:
        result_df = pd.concat([key_columns.reset_index(drop=True), result_df], axis=1)

    return result_df


def aggregate_feature_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
    group_by: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Aggregate SHAP importance by feature, optionally grouped.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values array.
    feature_names : List[str]
        Names of features.
    group_by : pd.Series, optional
        Series to group by before aggregating.

    Returns
    -------
    pd.DataFrame
        DataFrame with mean |SHAP| per feature (and per group if specified).
    """
    shap_abs = np.abs(shap_values)

    if group_by is None:
        # Overall importance
        mean_importance = shap_abs.mean(axis=0)
        return pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_importance,
        }).sort_values("mean_abs_shap", ascending=False)
    else:
        # Grouped importance
        df = pd.DataFrame(shap_abs, columns=feature_names)
        df["group"] = group_by.values
        grouped = df.groupby("group").mean()

        # Melt to long format
        result = grouped.reset_index().melt(
            id_vars=["group"],
            var_name="feature",
            value_name="mean_abs_shap"
        )
        return result.sort_values(["group", "mean_abs_shap"], ascending=[True, False])


def filter_contributors_by_direction(
    shap_values: np.ndarray,
    feature_names: List[str],
    direction: str = "positive",
    top_k: int = 3,
) -> List[List[str]]:
    """
    Get top contributors filtered by SHAP direction.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values array.
    feature_names : List[str]
        Names of features.
    direction : str, optional
        "positive" for features increasing prediction,
        "negative" for features decreasing prediction.
        Default is "positive".
    top_k : int, optional
        Number of top contributors per row. Default is 3.

    Returns
    -------
    List[List[str]]
        List of top feature names per row, filtered by direction.
    """
    top_k = min(top_k, shap_values.shape[1])
    result = []

    for row in shap_values:
        if direction == "positive":
            # Filter to positive SHAP values
            mask = row > 0
        else:
            # Filter to negative SHAP values
            mask = row < 0

        filtered_indices = np.where(mask)[0]
        filtered_values = np.abs(row[mask])

        # Sort by absolute value
        sorted_indices = filtered_indices[np.argsort(filtered_values)[::-1][:top_k]]
        result.append([feature_names[i] for i in sorted_indices])

    return result
