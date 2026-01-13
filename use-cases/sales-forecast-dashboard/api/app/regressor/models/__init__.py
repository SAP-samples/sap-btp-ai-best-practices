"""
Models Module.

Provides CatBoost-based predictors for sales forecasting:
- B&M channel: BMPredictor (Sales, AOV, Orders, Conversion)
- WEB channel: WEBPredictor (Sales, AOV, Orders)
- Explainability: SurrogateExplainer (SHAP-based interpretation)
- Traffic: TrafficEstimator (Monte Carlo simulation)

Usage:
    from app.regressor.models import (
        BMPredictor,
        WEBPredictor,
        SurrogateExplainer,
        estimate_traffic_quantiles,
        get_predictor,
    )

    # Train B&M model
    bm_predictor = BMPredictor(iterations=5000)
    bm_predictor.fit(train_df, val_df)
    predictions = bm_predictor.predict(test_df)

    # Train WEB model
    web_predictor = WEBPredictor(iterations=5000)
    web_predictor.fit(train_df, val_df)
    predictions = web_predictor.predict(test_df)

    # Factory function
    predictor = get_predictor("B&M")
    predictor.fit(train_df)
"""

from typing import Union

from .base import BaseCatBoostPredictor, PredictionResult
from .bm_predictor import (
    BMMultiObjectivePredictor,
    BMConversionPredictor,
    BMPredictor,
    BMPredictionResult,
)
from .web_predictor import (
    WEBMultiObjectivePredictor,
    WEBPredictor,
    WEBPredictionResult,
)
from .surrogate import SurrogateExplainer
from .traffic import (
    TrafficEstimator,
    TrafficResult,
    estimate_traffic_quantiles,
)


def get_predictor(channel: str) -> Union[BMPredictor, WEBPredictor]:
    """
    Factory function to get the appropriate predictor for a channel.

    Parameters
    ----------
    channel : str
        Channel type: 'B&M' or 'WEB'.

    Returns
    -------
    BMPredictor or WEBPredictor
        Predictor instance for the specified channel.

    Raises
    ------
    ValueError
        If the channel is not recognized.
    """
    channel = channel.upper() if channel else channel

    if channel in ("B&M", "BM"):
        return BMPredictor()
    elif channel == "WEB":
        return WEBPredictor()
    else:
        raise ValueError(f"Unknown channel: {channel}. Must be 'B&M' or 'WEB'.")


__all__ = [
    # Base classes
    "BaseCatBoostPredictor",
    "PredictionResult",
    # B&M predictors
    "BMMultiObjectivePredictor",
    "BMConversionPredictor",
    "BMPredictor",
    "BMPredictionResult",
    # WEB predictors
    "WEBMultiObjectivePredictor",
    "WEBPredictor",
    "WEBPredictionResult",
    # Explainability
    "SurrogateExplainer",
    # Traffic estimation
    "TrafficEstimator",
    "TrafficResult",
    "estimate_traffic_quantiles",
    # Factory
    "get_predictor",
]
