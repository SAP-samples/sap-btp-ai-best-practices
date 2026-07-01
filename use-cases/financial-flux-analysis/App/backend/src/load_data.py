"""
Module for loading financial data from SAP HANA view v_delec_fin.
fiscal_year and fiscal_period are derived from "Posting Date" (the only date source).
"""

import logging
import sys
import os
from typing import Optional, Union

import pandas as pd

_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from services.hana_financial_service import HANAFinancialService

logger = logging.getLogger(__name__)

_service = HANAFinancialService()


def load_acdocu(file_path: str, period_label: str) -> pd.DataFrame:
    """
    Compatibility stub for the legacy Streamlit app (app.py).
    Loads an ACDOCU Excel file and returns a DataFrame.

    Args:
        file_path:    Path to the Excel file.
        period_label: Period identifier (e.g. "Q3_24") — stored in a 'Period' column.

    Returns:
        pandas DataFrame with all columns from the Excel file plus a 'Period' column.
    """
    logger.info("load_acdocu | file=%s period=%s", file_path, period_label)
    df = pd.read_excel(file_path)
    df["Period"] = period_label
    return df


def load_financial_data(
    fiscal_year: Optional[Union[int, str]] = None,
    fiscal_period: Optional[Union[int, str]] = None,
    company_code: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load financial data from SAP HANA view v_delec_fin.

    fiscal_year and fiscal_period are derived from "Posting Date":
      fiscal_year   = YEAR("Posting Date")
      fiscal_period = MONTH("Posting Date")  (1-12)

    Args:
        fiscal_year:   Filter by year (int or str, e.g. 2026 or "2026").
        fiscal_period: Filter by month (int or str, e.g. 2 or "2").
        company_code:  Filter by Company Code.

    Returns:
        Normalized pandas DataFrame with fiscal_year, fiscal_period, amount_numeric columns.
    """
    # Normalize types
    year = int(fiscal_year) if fiscal_year not in (None, "", "null") else None
    period = int(fiscal_period) if fiscal_period not in (None, "", "null") else None

    logger.info(
        "load_financial_data | year=%s period=%s company=%s",
        year, period, company_code,
    )
    return _service.get_financial_data(
        fiscal_year=year,
        fiscal_period=period,
        company_code=company_code,
    )