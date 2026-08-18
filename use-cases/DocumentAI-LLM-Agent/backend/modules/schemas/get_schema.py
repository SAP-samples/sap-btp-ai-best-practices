"""
get_schema.py
-------------
Retrieves available schemas from SAP Document AI.

Endpoint:
    GET /document-information-extraction/v1/schemas?clientId=default

Documentation:
    https://help.sap.com/docs/document-information-extraction
"""

import logging
from typing import Any

import requests
from requests.exceptions import ConnectionError, HTTPError, Timeout

from modules.auth.get_token import AuthenticationError, TokenManager
from utils.config_loader import load_config

logger = logging.getLogger(__name__)

# Timeout for Document AI API requests (seconds)
API_REQUEST_TIMEOUT: int = 60

# Relative endpoint path for schemas resource
SCHEMAS_ENDPOINT: str = "/document-information-extraction/v1/schemas"

# Default client ID for SAP Document AI
DEFAULT_CLIENT_ID: str = "default"

# Page size for pagination (maximum per request)
PAGE_SIZE: int = 50


class DocumentAIError(Exception):
    """Raised when a SAP Document AI API request fails."""
    pass


class SchemaClient:
    """
    Client for the /schemas resource of SAP Document AI.

    Manages the HTTP session, Bearer authentication, and API error handling.

    Usage:
        client = SchemaClient()
        schemas = client.get_schemas()
    """

    def __init__(self, token_manager: TokenManager | None = None) -> None:
        """
        Initialize the SchemaClient.

        Args:
            token_manager: TokenManager instance. If None, creates a new one.
        """
        self._config = load_config()
        self._token_manager = token_manager or TokenManager(self._config)
        self._session = requests.Session()

        # Build service base URL
        self._base_url = self._config["service_url"].rstrip("/")

        logger.debug("SchemaClient initialized. Base URL: %s", self._base_url)

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers with the current Bearer token."""
        token = self._token_manager.get_token()
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _request_page(
        self,
        url: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform a single paginated GET request and return the JSON response.

        Raises:
            AuthenticationError: If the token is rejected (401).
            DocumentAIError: For any other HTTP error.
        """
        try:
            response = self._session.get(
                url,
                headers=self._build_headers(),
                params=params,
                timeout=API_REQUEST_TIMEOUT,
            )
            response.raise_for_status()

        except Timeout:
            raise DocumentAIError(
                f"Timeout connecting to SAP Document AI: {url} "
                f"(limit: {API_REQUEST_TIMEOUT}s)"
            )
        except ConnectionError as exc:
            raise DocumentAIError(
                f"Could not connect to SAP Document AI: {url}\n"
                f"Detail: {exc}"
            )
        except HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else "N/A"
            body = exc.response.text if exc.response is not None else ""

            if status_code == 401:
                logger.warning("Token rejected (401). Invalidating token cache.")
                self._token_manager.invalidate_token()
                raise AuthenticationError(
                    "Access token rejected by SAP Document AI (401 Unauthorized). "
                    "Try running again to obtain a fresh token."
                )
            elif status_code == 403:
                raise DocumentAIError(
                    "Access denied to schemas endpoint (403 Forbidden). "
                    "Check the service key permissions."
                )
            elif status_code == 404:
                raise DocumentAIError(
                    f"Endpoint not found (404): {url}\n"
                    "Check the service URL in docai.json."
                )
            elif status_code == 500:
                raise DocumentAIError(
                    f"SAP Document AI internal server error (500).\n"
                    f"Response: {body[:300]}"
                )
            else:
                raise DocumentAIError(
                    f"HTTP {status_code} error fetching schemas.\n"
                    f"URL: {url}\nResponse: {body[:300]}"
                )

        try:
            return response.json()
        except ValueError:
            raise DocumentAIError(
                "SAP Document AI response is not valid JSON.\n"
                f"Response received: {response.text[:300]}"
            )

    def get_schemas(self, client_id: str = DEFAULT_CLIENT_ID) -> dict[str, Any]:
        """
        Fetch ALL available schemas from SAP Document AI,
        handling pagination automatically.

        The API returns a subset by default (e.g. 5 of 12).
        This method iterates with offset until the total indicated
        in 'totalSchemaCount' is retrieved.

        Args:
            client_id: Document AI client ID. Default: "default".

        Returns:
            Dictionary with all schemas under the 'schemas' key
            and the actual total in 'totalSchemaCount'.

        Raises:
            AuthenticationError: If the token is invalid or expired.
            DocumentAIError: If the API returns an HTTP error.
        """
        url = f"{self._base_url}{SCHEMAS_ENDPOINT}"
        all_schemas: list[dict[str, Any]] = []
        offset = 0
        total_count = 0

        logger.info(
            "Fetching all schemas from: %s (clientId=%s)", url, client_id
        )

        while True:
            params: dict[str, Any] = {
                "clientId": client_id,
                "limit": PAGE_SIZE,
                "offset": offset,
            }

            logger.debug("Requesting page: offset=%d, limit=%d", offset, PAGE_SIZE)
            page_data = self._request_page(url, params)

            # Extract schemas from current page
            page_schemas = page_data.get("schemas") or []
            if not isinstance(page_schemas, list):
                page_schemas = []

            all_schemas.extend(page_schemas)

            total_count = page_data.get("totalSchemaCount", len(all_schemas))

            logger.debug(
                "Page received: %d schemas. Accumulated: %d / %d",
                len(page_schemas),
                len(all_schemas),
                total_count,
            )

            # Stop condition: no more pages
            if len(page_schemas) == 0 or len(all_schemas) >= total_count:
                break

            offset += PAGE_SIZE

        logger.info(
            "Schemas retrieved successfully. Total: %d / %d",
            len(all_schemas),
            total_count,
        )

        return {
            "schemas": all_schemas,
            "totalSchemaCount": total_count,
        }


def get_schemas(client_id: str = DEFAULT_CLIENT_ID) -> dict[str, Any]:
    """
    Convenience function to fetch schemas without instantiating SchemaClient.

    Args:
        client_id: Document AI client ID. Default: "default".

    Returns:
        Dictionary with the API JSON response.
    """
    client = SchemaClient()
    return client.get_schemas(client_id=client_id)