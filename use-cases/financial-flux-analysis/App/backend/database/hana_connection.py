"""
Module for SAP HANA / SAP Datasphere connection.
Handles connection through hdbcli and environment variables.
"""

import logging
import os
import time
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

try:
    from hdbcli import dbapi
    HDBCLI_AVAILABLE = True
except ImportError:
    HDBCLI_AVAILABLE = False

load_dotenv()

logger = logging.getLogger(__name__)


class HANAConnection:
    """
    Encapsulates SAP HANA / SAP Datasphere connection.
    Credentials are loaded from environment variables.
    """

    def __init__(self) -> None:
        self.host: str = os.getenv("HANA_ADDRESS") or os.getenv("HANA_HOST", "")
        self.port: int = int(os.getenv("HANA_PORT", "443"))
        self.user: str = os.getenv("HANA_USER", "")
        self.password: str = os.getenv("HANA_PASSWORD", "")
        self.schema: str = os.getenv("HANA_SCHEMA", "")
        self.encrypt: bool = os.getenv("HANA_ENCRYPT", "True").lower() in ("true", "1", "yes")
        self.connection: Optional[object] = None

    def connect(self, max_retries: int = 3, retry_delay: float = 1.5) -> None:
        if not HDBCLI_AVAILABLE:
            logger.warning("hdbcli not installed.")
            return

        connect_kwargs = dict(
            address=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            encrypt=self.encrypt,
            sslValidateCertificate=False,
        )
        if self.schema:
            connect_kwargs["currentSchema"] = self.schema

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    "Connecting to SAP HANA %s:%s schema=%s (attempt %d/%d)",
                    self.host, self.port, self.schema, attempt, max_retries,
                )
                self.connection = dbapi.connect(**connect_kwargs)
                logger.info("SAP HANA connection established (schema=%s).", self.schema or "default")
                return
            except Exception as error:
                last_error = error
                logger.warning("Connection attempt %d failed: %s", attempt, error)
                if attempt < max_retries:
                    time.sleep(retry_delay * attempt)

        logger.error("All %d connection attempts failed. Last error: %s", max_retries, last_error)
        raise last_error

    def disconnect(self) -> None:
        if self.connection:
            try:
                self.connection.close()
                self.connection = None
                logger.info("SAP HANA connection closed.")
            except Exception as error:
                logger.error("Error closing connection: %s", error)

    def execute_query(self, sql: str) -> pd.DataFrame:
        if not HDBCLI_AVAILABLE or self.connection is None:
            logger.warning("No HANA connection available.")
            return pd.DataFrame()

        try:
            cursor = self.connection.cursor()
            cursor.execute(sql)

            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

            cursor.close()

            return pd.DataFrame(rows, columns=columns)

        except Exception as error:
            logger.error("Query execution error: %s", error)
            raise

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

        