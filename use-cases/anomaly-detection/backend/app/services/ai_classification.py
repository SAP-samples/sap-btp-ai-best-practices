from __future__ import annotations

"""AI classification utilities."""

from typing import Optional, Tuple

import pandas as pd

from explainability.ai_explanation_generator import (
    format_binary_result_for_display,
    generate_ai_binary_classification_with_images,
    generate_explanation_with_cache_enhanced,
)
from utils.formatters import format_ai_response_feature_names
from visualization.feature_analysis import create_feature_plots


def run_binary_classification(row: pd.Series, features_df: pd.DataFrame) -> Tuple[str, Optional[list]]:
    row_for_binary = row.copy()

    image_paths = None
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend for server environments
        
        fig, image_paths = create_feature_plots(row_for_binary, features_df, save_for_ai_analysis=True)
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        image_paths = None

    binary_result = generate_ai_binary_classification_with_images(row_for_binary, features_df, image_paths)
    formatted_result = format_binary_result_for_display(binary_result)
    formatted_result = format_ai_response_feature_names(formatted_result)
    return formatted_result, image_paths

def generate_full_explanation(row: pd.Series, features_df: pd.DataFrame) -> str:
    row_for_ai = row.copy()
    explanation = generate_explanation_with_cache_enhanced(row_for_ai, features_df, use_visual_analysis=True)
    return format_ai_response_feature_names(explanation)
