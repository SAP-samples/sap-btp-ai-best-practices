from __future__ import annotations

"""Model runtime services including lazy loading and SHAP computation."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from functools import lru_cache

from services.data_loader import find_best_results_directory
from services.model_service import model_service

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    loaded: bool
    info: dict
    results_directory: Optional[Path]


@lru_cache(maxsize=1)
def load_models(results_directory: Optional[Path] = None) -> ModelInfo:
    directory = results_directory or find_best_results_directory()
    if not directory:
        return ModelInfo(loaded=False, info={"error": "No suitable model directory found"}, results_directory=None)

    success = model_service.load_models(str(directory))
    if not success:
        return ModelInfo(loaded=False, info={"error": "Failed to load models"}, results_directory=directory)

    info = model_service.get_model_info()
    return ModelInfo(loaded=True, info=info, results_directory=directory)


def compute_shap(row: pd.Series, features_df: pd.DataFrame, results_directory: Optional[Path] = None):
    model_info = load_models(results_directory)
    if not model_info.loaded:
        return None
    try:
        return model_service.compute_shap_for_sample(row, features_df)
    except Exception:
        return None
