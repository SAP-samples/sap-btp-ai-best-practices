from __future__ import annotations

"""Data loading service for the Streamlit application.

This module centralizes the logic for loading primary datasets, applying
business rules, and preparing data used across different application views.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

from services import rule_engine


DATA_CACHE_VERSION = "20250930_no_date_filter"
DEFAULT_DATA_PATH = Path("datasets/all_data.csv")


@dataclass
class LoadedData:
    """Container for loaded datasets."""

    results: pd.DataFrame
    features: pd.DataFrame
    data_path: Path
    results_directory: Optional[Path]


def _rename_business_rule_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename business rule columns to expected snake_case names."""
    column_mapping = {
        "Blocked by Business Rules": "blocked_by_business_rules",
        "Business Rules": "business_rules_explanation",
    }
    return df.rename(columns=column_mapping)


@st.cache_data
def load_dataset(data_path: Optional[Path] = None) -> LoadedData:
    """Load the primary dataset used by the Streamlit application.

    Parameters
    ----------
    data_path: Optional[Path]
        Optional override for the dataset path; defaults to DEFAULT_DATA_PATH.

    Returns
    -------
    LoadedData
        Dataclass containing preprocessed results and features dataframes.
    """
    target_path = data_path or DEFAULT_DATA_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Required dataset not found: {target_path}")

    df = pd.read_csv(target_path)
    df = _rename_business_rule_columns(df)

    results_dir = find_best_results_directory()

    # Ensure expected columns exist with defaults.
    if "anomaly_score" not in df.columns:
        df["anomaly_score"] = 0.0
    df["anomaly_score"] = df["anomaly_score"].fillna(0.0)

    if "predicted_anomaly" not in df.columns:
        df["predicted_anomaly"] = 0
    df["predicted_anomaly"] = df["predicted_anomaly"].fillna(0).astype(int)

    if "shap_explanation" not in df.columns:
        df["shap_explanation"] = "N/A"
    df["shap_explanation"] = df["shap_explanation"].fillna("N/A")

    if "model_used" not in df.columns:
        df["model_used"] = "sklearn_isolation_forest"
    df["model_used"] = df["model_used"].fillna("sklearn_isolation_forest")

    if "ai_anomaly_result" not in df.columns:
        df["ai_anomaly_result"] = False
    df["ai_anomaly_result"] = df["ai_anomaly_result"].fillna(False).astype(bool)

    df = rule_engine.ensure_business_rule_columns(df)

    # Prepare datetime columns.
    for col in ["Sales Document Created Date", "Actual GI Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return LoadedData(results=df.copy(), features=df.copy(), data_path=target_path, results_directory=results_dir)


def find_best_results_directory(results_base: Path = Path("results")) -> Optional[Path]:
    """Return the preferred results directory if available."""
    if not results_base.exists():
        return None

    directories = []
    for item in results_base.iterdir():
        if item.is_dir() and item.name.startswith("anomaly_detection_results_backend_sklearn"):
            models_dir = item / "models"
            results_file = item / "anomaly_detection_results.csv"
            if models_dir.exists() and results_file.exists():
                is_stratified = "customer_stratified" in item.name
                has_shap = item.name.endswith("_shap")
                priority = (1 if not is_stratified else 2, 0 if not has_shap else 1)
                directories.append((priority, item.stat().st_mtime, item))

    if directories:
        directories.sort(key=lambda x: (x[0], -x[1]))
        return directories[0][2]

    return None
