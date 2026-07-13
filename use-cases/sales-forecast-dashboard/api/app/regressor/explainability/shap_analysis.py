"""
SHAP Analysis Utilities.

Provides standalone functions for SHAP-based model interpretation.
These functions can be used with any tree-based model that is compatible
with SHAP's TreeExplainer.
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd


def compute_shap_values(
    model,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Compute SHAP values for the input features.

    Parameters
    ----------
    model : CatBoostRegressor or similar
        Trained model compatible with SHAP TreeExplainer.
    X : pd.DataFrame
        Feature matrix.

    Returns
    -------
    np.ndarray
        SHAP values array of shape (n_samples, n_features).
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP is required. Install with: pip install shap")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Handle shap.Explanation object
    if hasattr(shap_values, "values"):
        return shap_values.values
    return shap_values


def get_top_features_by_shap(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_n: int = 10
) -> List[str]:
    """
    Get top features ranked by mean absolute SHAP value.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values array of shape (n_samples, n_features).
    feature_names : List[str]
        Names of features corresponding to columns in shap_values.
    top_n : int, optional
        Number of top features to return. Default is 10.

    Returns
    -------
    List[str]
        Names of top features ordered by importance.
    """
    top_n = min(top_n, shap_values.shape[1])
    importance = np.abs(shap_values).mean(axis=0)
    ranked_indices = np.argsort(importance)[::-1][:top_n]
    return [feature_names[i] for i in ranked_indices]


def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    output_path: Union[str, Path],
    title: Optional[str] = None,
    max_display: int = 20,
) -> None:
    """
    Generate and save SHAP summary plot.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values array.
    X : pd.DataFrame
        Feature matrix.
    output_path : str or Path
        Path to save the plot.
    title : str, optional
        Plot title.
    max_display : int, optional
        Maximum features to display. Default is 20.
    """
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("SHAP and matplotlib are required.")

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_shap_importance(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    output_path: Union[str, Path],
    title: Optional[str] = None,
    max_display: int = 20,
) -> None:
    """
    Generate and save SHAP feature importance bar plot.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values array.
    X : pd.DataFrame
        Feature matrix.
    output_path : str or Path
        Path to save the plot.
    title : str, optional
        Plot title.
    max_display : int, optional
        Maximum features to display. Default is 20.
    """
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("SHAP and matplotlib are required.")

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", max_display=max_display, show=False)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_shap_dependence(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature: str,
    output_path: Union[str, Path],
    interaction_feature: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """
    Generate and save SHAP dependence plot for a feature.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values array.
    X : pd.DataFrame
        Feature matrix.
    feature : str
        Feature name to plot.
    output_path : str or Path
        Path to save the plot.
    interaction_feature : str, optional
        Feature to use for coloring interaction effects.
    title : str, optional
        Plot title.
    """
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("SHAP and matplotlib are required.")

    interaction_idx = interaction_feature if interaction_feature in X.columns else None

    plt.figure(figsize=(10, 8))
    shap.dependence_plot(feature, shap_values, X, interaction_index=interaction_idx, show=False)
    if title:
        plt.title(title)
    else:
        color_label = interaction_idx if interaction_idx else feature
        plt.title(f"Dependence: {feature} (color={color_label})")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_horizon_sensitivity(
    shap_values: np.ndarray,
    feature_names: List[str],
    horizons: pd.Series,
    top_features: List[str],
    output_path: Union[str, Path],
    title: Optional[str] = None,
) -> None:
    """
    Plot mean absolute SHAP value by forecast horizon for top features.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values array.
    feature_names : List[str]
        Names of features.
    horizons : pd.Series
        Horizon values for each sample.
    top_features : List[str]
        Features to include in the plot.
    output_path : str or Path
        Path to save the plot.
    title : str, optional
        Plot title.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required.")

    if horizons.isna().all():
        return

    shap_abs_df = pd.DataFrame(np.abs(shap_values), columns=feature_names)
    shap_abs_df["horizon"] = horizons.values

    horizon_group = shap_abs_df.groupby("horizon").mean().sort_index()
    if horizon_group.empty:
        return

    plt.figure(figsize=(12, 8))
    for feat in top_features:
        if feat in horizon_group.columns:
            plt.plot(horizon_group.index, horizon_group[feat], marker="o", label=feat)

    plt.xlabel("Horizon")
    plt.ylabel("Mean |SHAP|")
    if title:
        plt.title(title)
    else:
        plt.title("Horizon Sensitivity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_cohort_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
    cohort_values: pd.Series,
    top_features: List[str],
    output_path: Union[str, Path],
    cohort_name: str = "cohort",
    title: Optional[str] = None,
    top_cohorts: int = 15,
) -> None:
    """
    Plot mean absolute SHAP value by cohort for top features.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values array.
    feature_names : List[str]
        Names of features.
    cohort_values : pd.Series
        Cohort values for each sample.
    top_features : List[str]
        Features to include in the aggregation.
    output_path : str or Path
        Path to save the plot.
    cohort_name : str, optional
        Name of the cohort for title. Default "cohort".
    title : str, optional
        Plot title.
    top_cohorts : int, optional
        Number of top cohorts to display. Default is 15.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required.")

    shap_abs_df = pd.DataFrame(np.abs(shap_values), columns=feature_names)
    shap_abs_df["cohort"] = cohort_values.astype(str).values

    grouped = shap_abs_df.groupby("cohort").mean()
    if grouped.empty:
        return

    available_feats = [f for f in top_features if f in grouped.columns]
    if not available_feats:
        return

    grouped["total_top_shap"] = grouped[available_feats].sum(axis=1)
    top_data = grouped["total_top_shap"].sort_values(ascending=False).head(top_cohorts)

    plt.figure(figsize=(12, 8))
    plt.barh(top_data.index.astype(str), top_data.values)
    plt.gca().invert_yaxis()
    plt.xlabel("Mean |SHAP| (Top Features)")
    if title:
        plt.title(title)
    else:
        plt.title(f"Cohort Sensitivity: {cohort_name}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_fit_scatter(
    actual: np.ndarray,
    predicted: np.ndarray,
    output_path: Union[str, Path],
    r2: Optional[float] = None,
    title: Optional[str] = None,
    xlabel: str = "Actual",
    ylabel: str = "Predicted",
) -> None:
    """
    Plot scatter of actual vs predicted values with diagonal line.

    Parameters
    ----------
    actual : np.ndarray
        Actual values.
    predicted : np.ndarray
        Predicted values.
    output_path : str or Path
        Path to save the plot.
    r2 : float, optional
        R2 score to display in title.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label. Default "Actual".
    ylabel : str, optional
        Y-axis label. Default "Predicted".
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required.")

    plt.figure(figsize=(10, 10))
    plt.scatter(actual, predicted, alpha=0.1, s=1)

    # Diagonal line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if title:
        plt.title(title)
    elif r2 is not None:
        plt.title(f"Fit Analysis: R2={r2:.4f}")
    else:
        plt.title("Fit Analysis")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
