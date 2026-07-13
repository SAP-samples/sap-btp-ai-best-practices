"""
Eligibility Configuration Module

This module defines configuration parameters for the invoice eligibility analysis system.
All threshold values are configurable via environment variables with sensible defaults.
"""

import os
from pathlib import Path
from typing import List


class EligibilitySettings:
    """
    Configuration settings for eligibility rules.

    All values can be overridden via environment variables or API query parameters.
    The API parameters take precedence over environment variables.
    """

    def __init__(
        self,
        nddt: int = None,
        teih: int = None,
        isspur: int = None,
        eligible_currencies: List[str] = None,
    ):
        """
        Initialize eligibility settings.

        Args:
            nddt: Minimum days from purchase date to due date (Rule R1)
            teih: Maximum tenor days from issuance to due date (Rule R16)
            isspur: Minimum days from issuance to purchase date (Rule R17)
            eligible_currencies: List of allowed currency codes (Rule R11)
        """
        # R1: Due Date - Purchase Date >= NDDT (minimum days to due date)
        self.nddt = nddt if nddt is not None else int(os.getenv("ELIGIBILITY_NDDT", "6"))

        # R16: Due Date - Issuance Date < TEIH (maximum tenor days)
        self.teih = teih if teih is not None else int(os.getenv("ELIGIBILITY_TEIH", "15"))

        # R17: Purchase Date - Issuance Date >= ISSPUR (minimum days since issuance)
        self.isspur = isspur if isspur is not None else int(os.getenv("ELIGIBILITY_ISSPUR", "0"))

        # R11: Currency must be in allowed list
        if eligible_currencies is not None:
            self.eligible_currencies = eligible_currencies
        else:
            env_currencies = os.getenv("ELIGIBILITY_CURRENCIES", "EUR,USD")
            self.eligible_currencies = [c.strip().upper() for c in env_currencies.split(",")]

    def to_dict(self) -> dict:
        """Return settings as a dictionary for API responses."""
        return {
            "nddt": self.nddt,
            "teih": self.teih,
            "isspur": self.isspur,
            "eligible_currencies": self.eligible_currencies,
        }


# Database and output paths
def get_database_path() -> Path:
    """Get the path to the SQLite database for customer logs."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "customer_logs.db"


def get_output_directory() -> Path:
    """Get the directory for generated Excel output files."""
    output_dir = Path(__file__).resolve().parent.parent / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# Default settings instance (can be overridden per request)
default_settings = EligibilitySettings()
