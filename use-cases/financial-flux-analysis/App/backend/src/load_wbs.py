"""
Module for loading WBS (Work Breakdown Structure) data from SAP HANA.
Replaces the previous Excel-based implementation.

The WBS table/view name is configured via the HANA_WBS_TABLE environment variable.
If the variable is not set or the query fails, an empty DataFrame is returned
so that the rest of the application continues to function.
"""

import logging
import os
import sys
from typing import Dict, Optional

import pandas as pd

# Ensure the backend root is on sys.path so sibling packages are importable
_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from database.hana_connection import HANAConnection, HDBCLI_AVAILABLE

logger = logging.getLogger(__name__)

# Optional: name of the HANA table/view that holds WBS data.
# Leave empty to disable WBS data loading from HANA.
HANA_WBS_TABLE: str = os.getenv("HANA_WBS_TABLE", "")


def create_wbs_lookup() -> Dict[str, Dict]:
    """
    Build a WBS lookup dictionary from HANA WBS data.

    Returns a dict compatible with DrillDownAnalyzer and explain_llm:
        {
            'wbs_descriptions': {wbs_element: description, ...},
            'wbs_programs':     {wbs_element: {'Project_Program': ..., 'Detailed_Program': ...}, ...}
        }

    Returns empty dicts when HANA_WBS_TABLE is not configured or data is unavailable.
    """
    df = load_wbs_data()

    wbs_descriptions: Dict[str, str] = {}
    wbs_programs: Dict[str, Dict] = {}

    if df.empty:
        logger.info("create_wbs_lookup: no WBS data available — returning empty lookup.")
        return {"wbs_descriptions": wbs_descriptions, "wbs_programs": wbs_programs}

    # Build description lookup — use first text-like column found
    desc_col = next(
        (c for c in df.columns if "desc" in c.lower() or "text" in c.lower() or "name" in c.lower()),
        None,
    )
    wbs_col = next(
        (c for c in df.columns if "wbs" in c.lower() or "element" in c.lower()),
        df.columns[0] if len(df.columns) > 0 else None,
    )

    if wbs_col:
        for _, row in df.iterrows():
            key = str(row[wbs_col]) if row[wbs_col] is not None else ""
            if not key:
                continue
            wbs_descriptions[key] = str(row[desc_col]) if desc_col and row[desc_col] is not None else ""
            wbs_programs[key] = {
                "Project_Program": str(row.get("Project_Program", row.get("project_program", ""))),
                "Detailed_Program": str(row.get("Detailed_Program", row.get("detailed_program", ""))),
            }

    logger.info("create_wbs_lookup: built lookup with %d entries.", len(wbs_descriptions))
    return {"wbs_descriptions": wbs_descriptions, "wbs_programs": wbs_programs}


def enrich_with_wbs_data(df: pd.DataFrame, wbs_lookup: Dict[str, Dict]) -> pd.DataFrame:
    """
    Enrich a DataFrame with WBS descriptions and program hierarchy.

    Adds the following columns (if not already present):
        - WBS_Description
        - Project_Program
        - Detailed_Program
        - High_Level_Program

    Parameters:
        df:         DataFrame containing a 'WBS_Element' column.
        wbs_lookup: Dict returned by create_wbs_lookup(), with keys
                    'wbs_descriptions' and 'wbs_programs'.

    Returns:
        Enriched DataFrame (copy).
    """
    if df.empty:
        return df

    df = df.copy()

    wbs_descriptions = wbs_lookup.get("wbs_descriptions", {})
    wbs_programs = wbs_lookup.get("wbs_programs", {})

    wbs_col = "WBS_Element" if "WBS_Element" in df.columns else None

    if wbs_col is None:
        logger.warning("enrich_with_wbs_data: 'WBS_Element' column not found — skipping enrichment.")
        df.setdefault("WBS_Description", "")
        df.setdefault("Project_Program", "")
        df.setdefault("Detailed_Program", "")
        df.setdefault("High_Level_Program", "")
        return df

    df["WBS_Description"] = df[wbs_col].astype(str).map(wbs_descriptions).fillna("")
    df["Project_Program"] = df[wbs_col].astype(str).apply(
        lambda x: wbs_programs.get(x, {}).get("Project_Program", "")
    )
    df["Detailed_Program"] = df[wbs_col].astype(str).apply(
        lambda x: wbs_programs.get(x, {}).get("Detailed_Program", "")
    )
    df["High_Level_Program"] = df[wbs_col].astype(str).apply(
        lambda x: wbs_programs.get(x, {}).get("High_Level_Program", "")
    )

    logger.info("enrich_with_wbs_data: enriched %d rows.", len(df))
    return df


def load_wbs_data() -> pd.DataFrame:
    """
    Load WBS data from SAP HANA.

    The table/view is configured via the HANA_WBS_TABLE environment variable.
    Returns an empty DataFrame if:
    - HANA_WBS_TABLE is not configured
    - hdbcli is not installed
    - The HANA connection or query fails

    Returns:
        pandas DataFrame with WBS data, or empty DataFrame on any error.
    """
    if not HANA_WBS_TABLE:
        logger.info(
            "HANA_WBS_TABLE not configured — WBS data loading skipped. "
            "Set HANA_WBS_TABLE env var to enable."
        )
        return pd.DataFrame()

    if not HDBCLI_AVAILABLE:
        logger.warning("hdbcli not installed — cannot load WBS data from HANA.")
        return pd.DataFrame()

    sql = f'SELECT * FROM "{HANA_WBS_TABLE}"'
    logger.info("Loading WBS data from HANA table: %s", HANA_WBS_TABLE)

    try:
        with HANAConnection() as hana:
            if hana.connection is None:
                logger.warning("No HANA connection available for WBS data.")
                return pd.DataFrame()
            df = hana.execute_query(sql)

        logger.info("Loaded %d WBS rows from HANA.", len(df))
        return df

    except Exception as exc:
        logger.error("Failed to load WBS data from HANA: %s", exc)
        return pd.DataFrame()