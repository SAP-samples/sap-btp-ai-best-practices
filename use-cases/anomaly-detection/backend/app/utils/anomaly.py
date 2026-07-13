from __future__ import annotations

"""Anomaly-related utility helpers."""

from typing import Optional, Tuple


def get_anomaly_category(score: float, model_type: str) -> Tuple[str, str, str, Optional[int]]:
    if model_type.lower().startswith("hana"):
        if score > 0.85:
            percentage = int(85 + ((score - 0.85) / 0.15) * 10)
            return "High Probability", "anomaly-high", "HANA Score > 0.85", min(percentage, 95)
        if score > 0.65:
            percentage = int(65 + ((score - 0.65) / 0.20) * 20)
            return "Medium Probability", "anomaly-medium", "HANA Score 0.65 - 0.85", percentage
        if score >= 0.5:
            percentage = int(50 + ((score - 0.5) / 0.15) * 15)
            return "Low Probability", "anomaly-low", "HANA Score 0.5 - 0.65", percentage
        return "Normal", "anomaly-low", "HANA Score < 0.5 (Normal behavior)", None

    if score < -0.15:
        percentage = int(80 + min(abs(score + 0.15) * 100, 15))
        return "High Probability", "anomaly-high", "Score < -0.15 (Strong anomaly)", percentage
    if score < -0.05:
        percentage = int(60 + (abs(score + 0.05) / 0.1) * 20)
        return "Medium Probability", "anomaly-medium", "Score -0.15 to -0.05 (Moderate anomaly)", percentage
    if score < 0:
        percentage = int(50 + (abs(score) / 0.05) * 10)
        return "Low Probability", "anomaly-low", "Score -0.05 to 0 (Weak anomaly)", percentage
    return "Normal", "anomaly-low", "Score ≥ 0 (Normal behavior)", None
