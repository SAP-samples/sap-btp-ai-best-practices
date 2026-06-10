"""Configuration package for the eligibility analysis system."""

from .eligibility_config import (
    EligibilitySettings,
    default_settings,
    get_database_path,
    get_output_directory,
)

__all__ = [
    "EligibilitySettings",
    "default_settings",
    "get_database_path",
    "get_output_directory",
]
