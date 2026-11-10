import os
import importlib.util
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import logging
from typing import Dict, Any
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)


class HANAConnection:
    """
    Class to handle SAP HANA connections and DataFrame operations
    """

    def __init__(self):
        """
        Initialize connection with credentials from .env file
        """
        load_dotenv()

        self.address = os.getenv("HANA_ADDRESS")
        self.port = os.getenv("HANA_PORT")
        self.user = os.getenv("HANA_USER")
        self.password = os.getenv("HANA_PASSWORD")
        self.encrypt = os.getenv("HANA_ENCRYPT", "True").lower() == "true"

        self.engine = None
        self.connection = None

        # Validate credentials
        if not all([self.address, self.port, self.user, self.password]):
            logger.warning(
                "HANA credentials missing in .env file (HANA_ADDRESS, HANA_PORT, HANA_USER, HANA_PASSWORD)"
            )

    def connect(self):
        """
        Establish connection to HANA database
        """
        if not all([self.address, self.port, self.user, self.password]):
            raise ValueError("HANA credentials missing")

        try:
            # Validate that all credentials are available
            logger.info(f"Attempting to connect to HANA: {self.address}:{self.port}")

            # Force dialect registration in the active interpreter and fail with
            # a clear error if the server is not using the environment that has
            # sqlalchemy-hana installed.
            if importlib.util.find_spec("sqlalchemy_hana") is None:
                raise RuntimeError(
                    "sqlalchemy-hana is not installed in the active Python environment. "
                    "Start the API with the project's .venv, for example: "
                    "'../.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --reload'"
                )

            import sqlalchemy_hana  # noqa: F401

            # Create connection string for SAP HANA
            connection_string = (
                f"hana://{self.user}:{self.password}@{self.address}:{self.port}"
                f"?encrypt={'true' if self.encrypt else 'false'}"
            )

            logger.info("Creating SQLAlchemy engine...")
            self.engine = create_engine(connection_string, echo=False)

            logger.info("Establishing connection...")
            self.connection = self.engine.connect()

            return True

        except SQLAlchemyError as e:
            logger.error(f"❌ SQLAlchemy error connecting to HANA: {e}")
            raise e
        except Exception as e:
            logger.error(f"❌ Unexpected error connecting: {e}")
            raise e

    def disconnect(self):
        """
        Close database connection
        """
        try:
            if self.connection:
                self.connection.close()
            if self.engine:
                self.engine.dispose()
            logger.info("Connection closed successfully")
        except Exception as e:
            logger.error(f"Error closing connection: {e}")

    def test_connection(self) -> Dict[str, Any]:
        """Test the connection with a simple query."""
        try:
            if not self.connection:
                self.connect()

            test_result = self.connection.execute(text("SELECT 1 FROM DUMMY"))
            test_value = test_result.fetchone()[0]

            if test_value == 1:
                return {
                    "success": True,
                    "message": "Successfully connected to SAP HANA and executed 'SELECT 1 FROM DUMMY'",
                }
            else:
                return {
                    "success": False,
                    "message": "Test query did not return expected value",
                }
        except Exception as e:
            return {"success": False, "message": str(e)}
        finally:
            self.disconnect()
