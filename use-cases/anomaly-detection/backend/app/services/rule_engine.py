from __future__ import annotations

"""Business rule utilities for the Streamlit application."""

from typing import Optional

import pandas as pd

try:
    from business_rules import (
        add_90day_customer_material_check,
        add_weekly_average_checks,
        add_focused_anomaly_explanations,
    )

    BUSINESS_RULES_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    BUSINESS_RULES_AVAILABLE = False


BUSINESS_RULE_FLAGS = [
    "is_outside_cm_90d_threshold",
    "is_outside_cm_weekly_threshold",
    "is_outside_mat_weekly_threshold",
]


def ensure_business_rule_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure business rule columns exist and have consistent types."""
    if "blocked_by_business_rules" not in df.columns:
        df["blocked_by_business_rules"] = False
    df["blocked_by_business_rules"] = df["blocked_by_business_rules"].fillna(False).astype(bool)

    if "business_rules_explanation" not in df.columns:
        df["business_rules_explanation"] = ""
    df["business_rules_explanation"] = df["business_rules_explanation"].fillna("")

    if BUSINESS_RULES_AVAILABLE:
        df = df.copy()
        if "Sales Document Created Date" in df.columns:
            df["Sales Document Created Date"] = pd.to_datetime(df["Sales Document Created Date"], errors="coerce")

        df = add_90day_customer_material_check(df)
        df = add_weekly_average_checks(df)
        df = add_focused_anomaly_explanations(df)
        df["blocked_by_business_rules"] = df[BUSINESS_RULE_FLAGS].any(axis=1)
        if "focused_anomaly_explanation" in df.columns:
            df["business_rules_explanation"] = df["focused_anomaly_explanation"].fillna("")

    return df
