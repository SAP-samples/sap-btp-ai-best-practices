from __future__ import annotations

"""Utilities for SHAP explanation parsing and presentation."""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from utils import formatting
from utils.explanation_parser import parse_shap_explanation


@dataclass
class ShapContribution:
    feature: str
    value: str
    contribution: str


def build_shap_dataframe(
    shap_text: str,
    features_df: pd.DataFrame,
    row: pd.Series,
) -> Optional[pd.DataFrame]:
    """Convert SHAP explanation text into a formatted dataframe."""
    if not shap_text or shap_text == "N/A":
        return None

    shap_contributions = parse_shap_explanation(shap_text)
    if not shap_contributions:
        return None

    shap_df = pd.DataFrame(shap_contributions)
    shap_df = shap_df[shap_df["shap_value"] < 0].copy()  # Show only anomalous drivers

    if shap_df.empty:
        return None

    shap_df["Contribution"] = shap_df.apply(
        lambda r: formatting.describe_contribution(
            feature_name=r["feature_name"],
            feature_value=r["feature_value"],
            features_df=features_df,
            context_row=row,
        ),
        axis=1,
    )

    shap_df["Feature Value"] = shap_df.apply(
        lambda r: formatting.format_feature_value(r["feature_value"], r["feature_name"]),
        axis=1,
    )

    shap_df = shap_df.sort_values("shap_value", ascending=True)

    display_df = shap_df[["feature_name", "Feature Value", "Contribution"]].copy()
    display_df["feature_name"] = display_df["feature_name"].apply(formatting.format_feature_name)
    display_df.columns = ["Feature", "Value", "Contribution"]

    return display_df
