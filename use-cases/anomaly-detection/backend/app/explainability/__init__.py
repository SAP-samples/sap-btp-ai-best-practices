"""
Explainability module for anomaly detection.

This module provides explainability features for the anomaly detection models,
including SHAP-based explanations, AI-powered explanations, and fallback rule-based explanations.
"""

# Import based on what exists in the module
try:
    from .ai_explanation_generator import generate_explanation_with_cache, generate_ai_explanation
except ImportError:
    generate_explanation_with_cache = None
    generate_ai_explanation = None

__all__ = []

# Only add to __all__ if successfully imported
if generate_explanation_with_cache is not None:
    __all__.extend(['generate_explanation_with_cache', 'generate_ai_explanation'])