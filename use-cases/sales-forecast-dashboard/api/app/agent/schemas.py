"""
Type definitions and constants for the forecasting agent.

This module provides type aliases and constants used across the 17 core tools.
Tool implementations use the @tool decorator with typed function parameters
rather than Pydantic input schemas.
"""

from __future__ import annotations

from typing import Literal


# =============================================================================
# TYPE ALIASES
# =============================================================================

# Channel types
CHANNELS = Literal["B&M", "WEB"]

# Metric types for comparison and visualization
METRICS = Literal["sales", "aov", "orders", "traffic"]

# Feature categories for filtering
FEATURE_CATEGORIES = Literal[
    "financing",
    "staffing",
    "product_mix",
    "awareness",
    "cannibalization",
    "store_dna",
]


# =============================================================================
# CONSTANTS
# =============================================================================

# Valid channels
VALID_CHANNELS = ["B&M", "WEB"]

# Default horizon (13 weeks = 1 quarter)
DEFAULT_HORIZON_WEEKS = 13

# Maximum horizon
MAX_HORIZON_WEEKS = 52

# Default checkpoint directory
DEFAULT_CHECKPOINT_DIR = "final_data/checkpoints"

# Cannibalization formula constants
LAMBDA_KM = 8.0  # Distance decay in km
TAU_WEEKS = 13.0  # Time decay in weeks


__all__ = [
    # Type aliases
    "CHANNELS",
    "METRICS",
    "FEATURE_CATEGORIES",
    # Constants
    "VALID_CHANNELS",
    "DEFAULT_HORIZON_WEEKS",
    "MAX_HORIZON_WEEKS",
    "DEFAULT_CHECKPOINT_DIR",
    "LAMBDA_KM",
    "TAU_WEEKS",
]
