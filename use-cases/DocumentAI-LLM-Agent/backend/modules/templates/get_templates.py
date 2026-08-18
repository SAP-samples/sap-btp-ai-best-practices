"""
get_templates.py
----------------
Retrieves available templates from SAP Document AI.

Endpoint:
    GET /document-information-extraction/v1/templates?clientId=default

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

# Relative endpoint path — full URL built per-instance from docai.json service_url
TEMPLATES_ENDPOINT: str = "/document-information-extraction/v1/templates"

# Default client ID for SAP Document AI
DEFAULT_CLIENT_ID: str = "default"

# Page size for pagination (maximum per request)
PAGE_SIZE: int = 50


class DocumentAIError(Exception):
    """Raised when a SAP Document AI API request fails."""
    pass


class TemplateClient:
    """
    Client for the /templates resource of SAP Document AI.

    Manages the HTTP session, Bearer authentication, and API error handling.

    Usage:
        client = TemplateClient()
        templates = client.get_templates()
    """

    def __init__(self, token_manager: TokenManager | None = None) -> None:
        """
        Initialize the TemplateClient.

        Args:
            token_manager: TokenManager instance. If None, creates a new one.
        """
        self._config = load_config()
        self._token_manager = token_manager or TokenManager(self._config)
        self._session = requests.Session()

        # Build service base URL
        self._base_url = self._config["service_url"].rstrip("/")

        logger.debug("TemplateClient initialized. Base URL: %s", self._base_url)

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
                    f"Access denied to templates endpoint (403 Forbidden).\n"
                    f"URL: {url}\n"
                    f"Server response: {body}\n"
                    "Possible causes:\n"
                    "  1. Service key missing 'Document_Information_Extraction_Templates_Read' role\n"
                    "  2. Service instance does not have templates enabled\n"
                    "  3. clientId 'default' does not exist or lacks template permissions\n"
                    "Solution: Check roles assigned to the service key in SAP BTP Cockpit."
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
                    f"HTTP {status_code} error fetching templates.\n"
                    f"URL: {url}\nResponse: {body[:300]}"
                )

        try:
            return response.json()
        except ValueError:
            raise DocumentAIError(
                "SAP Document AI response is not valid JSON.\n"
                f"Response received: {response.text[:300]}"
            )

    def get_templates(self, client_id: str = DEFAULT_CLIENT_ID) -> dict[str, Any]:
        """
        Fetch ALL available templates from SAP Document AI,
        handling pagination automatically.

        Args:
            client_id: Document AI client ID. Default: "default".

        Returns:
            Dictionary with all templates under the 'templates' key.

        Raises:
            AuthenticationError: If the token is invalid or expired.
            DocumentAIError: If the API returns an HTTP error.
        """
        url = self._base_url + TEMPLATES_ENDPOINT
        all_templates: list[dict[str, Any]] = []
        offset = 0

        logger.info(
            "Fetching all templates from: %s (clientId=%s)", url, client_id
        )

        while True:
            params: dict[str, Any] = {
                "clientId": client_id,
                "limit": PAGE_SIZE,
                "offset": offset,
            }

            logger.debug("Requesting page: offset=%d, limit=%d", offset, PAGE_SIZE)
            page_data = self._request_page(url, params)

            # The API returns the list under different possible keys
            page_templates = (
                page_data.get("results")
                or page_data.get("templates")
                or page_data.get("value")
                or (page_data if isinstance(page_data, list) else [])
            )
            if not isinstance(page_templates, list):
                page_templates = []

            all_templates.extend(page_templates)

            # Look for total count under different possible keys
            total_count = (
                page_data.get("totalTemplateCount")
                or page_data.get("totalCount")
                or page_data.get("count")
                or len(all_templates)
            )

            logger.debug(
                "Page received: %d templates. Accumulated: %d / %s",
                len(page_templates),
                len(all_templates),
                total_count,
            )

            # Stop condition: empty page or total reached
            if len(page_templates) == 0 or len(all_templates) >= total_count:
                break

            offset += PAGE_SIZE

        logger.info(
            "Templates retrieved successfully. Total: %d",
            len(all_templates),
        )

        return {
            "templates": all_templates,
            "totalTemplateCount": len(all_templates),
        }


def get_templates(client_id: str = DEFAULT_CLIENT_ID) -> dict[str, Any]:
    """
    Convenience function to fetch templates without instantiating TemplateClient.

    Args:
        client_id: Document AI client ID. Default: "default".

    Returns:
        Dictionary with the API JSON response.
    """
    client = TemplateClient()
    return client.get_templates(client_id=client_id)
