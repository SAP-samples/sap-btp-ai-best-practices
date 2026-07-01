"""
HANA Financial Service — ONLY loads raw data from v_delec_fin.
All processing is delegated to FinancialProcessor.
"""

import logging
import os
from typing import Dict

import pandas as pd

from database.hana_connection import HANAConnection, HDBCLI_AVAILABLE

logger = logging.getLogger(__name__)

VIEW_NAME = os.getenv("HANA_VIEW", "v_delec_fin")


class HANAFinancialService:
    """Loads raw data from SAP HANA view v_delec_fin. No filtering, no aggregation."""

    def load_all_data(self) -> pd.DataFrame:
        """
        Load ALL records from v_delec_fin with a simple SELECT *.
        Returns raw DataFrame — no transformations applied here.
        """
        logger.info("HANAFinancialService: loading all data from %s", VIEW_NAME)
        try:
            with HANAConnection() as hana:
                if not HDBCLI_AVAILABLE or hana.connection is None:
                    logger.warning("No HANA connection available.")
                    return pd.DataFrame()
                sql = 'SELECT * FROM "{}"'.format(VIEW_NAME)
                df = hana.execute_query(sql)
                logger.info("HANAFinancialService: loaded %d rows.", len(df))
                return df
        except Exception as exc:
            logger.error("load_all_data failed: %s", exc)
            return pd.DataFrame()

    def check_connectivity(self) -> Dict:
        """Lightweight connectivity check."""
        if not HDBCLI_AVAILABLE:
            return {"connected": False, "message": "hdbcli not installed."}
        try:
            with HANAConnection() as hana:
                if hana.connection is None:
                    return {"connected": False, "message": "Connection is None."}
                hana.execute_query('SELECT COUNT(*) FROM "{}"'.format(VIEW_NAME))
            return {"connected": True, "message": "SAP HANA connection successful."}
        except Exception as exc:
            return {"connected": False, "message": str(exc)}