"""
get_token.py
------------
OAuth2 authentication module for SAP Document AI.

Implements the client_credentials flow to obtain a Bearer Token
from the SAP BTP authorization server (UAA/XSUAA).

Features:
  - Token caching with automatic expiry detection
  - Configurable expiry safety margin
  - Thread-safe singleton via module-level manager
"""

import logging
import time
from typing import Optional

import requests
from requests.auth import HTTPBasicAuth
from requests.exceptions import ConnectionError, HTTPError, Timeout

from utils.config_loader import DocAIConfig, load_config

logger = logging.getLogger(__name__)

# Timeout for token requests (seconds)
TOKEN_REQUEST_TIMEOUT: int = 30

# Safety margin before considering the token expired (seconds)
TOKEN_EXPIRY_MARGIN: int = 60


class AuthenticationError(Exception):
    """Raised when authentication with SAP BTP fails."""
    pass


class TokenManager:
    """
    Manages the OAuth2 access token lifecycle.

    Fetches a new token when needed and reuses it while valid
    (with a configurable safety margin before expiry).

    Usage:
        manager = TokenManager()
        token = manager.get_token()
    """

    def __init__(self, config: Optional[DocAIConfig] = None) -> None:
        """
        Initialize the TokenManager.

        Args:
            config: Credentials configuration. If None, loads from docai.json.
        """
        self._config: DocAIConfig = config or load_config()
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._session = requests.Session()

    def _is_token_valid(self) -> bool:
        """Check whether the current token is still valid."""
        if not self._access_token:
            return False
        return time.time() < (self._token_expires_at - TOKEN_EXPIRY_MARGIN)

    def _fetch_token(self) -> str:
        """
        Perform the OAuth2 client_credentials request to the UAA server.

        Returns:
            access_token as string.

        Raises:
            AuthenticationError: If credentials are invalid or the server rejects the request.
        """
        token_url = self._config["token_url"]
        logger.info("Requesting new access token from: %s", token_url)

        payload = {
            "grant_type": "client_credentials",
            "response_type": "token",
        }

        auth = HTTPBasicAuth(
            self._config["clientid"],
            self._config["clientsecret"],
        )

        try:
            response = self._session.post(
                token_url,
                data=payload,
                auth=auth,
                timeout=TOKEN_REQUEST_TIMEOUT,
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()

        except Timeout:
            raise AuthenticationError(
                f"Timeout connecting to token server: {token_url} "
                f"(limit: {TOKEN_REQUEST_TIMEOUT}s)"
            )
        except ConnectionError as exc:
            raise AuthenticationError(
                f"Could not connect to token server: {token_url}\n"
                f"Detail: {exc}"
            )
        except HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else "N/A"
            body = exc.response.text if exc.response is not None else ""

            if status_code == 401:
                raise AuthenticationError(
                    "Invalid credentials (401 Unauthorized). "
                    "Check clientid and clientsecret in docai.json."
                )
            elif status_code == 403:
                raise AuthenticationError(
                    "Access denied (403 Forbidden). "
                    "The client does not have permission to obtain a token."
                )
            else:
                raise AuthenticationError(
                    f"HTTP {status_code} error obtaining token.\nResponse: {body}"
                )

        try:
            token_data = response.json()
        except ValueError:
            raise AuthenticationError(
                "Token server response is not valid JSON.\n"
                f"Response received: {response.text[:200]}"
            )

        access_token = token_data.get("access_token")
        if not access_token:
            raise AuthenticationError(
                "Server response does not contain 'access_token'.\n"
                f"Fields received: {list(token_data.keys())}"
            )

        # Calculate expiry time
        expires_in = token_data.get("expires_in", 3600)
        self._token_expires_at = time.time() + float(expires_in)

        logger.info("Token obtained successfully. Expires in %d seconds.", expires_in)

        return access_token

    def get_token(self) -> str:
        """
        Return a valid access token.

        Reuses the cached token if still valid,
        or requests a new one if expired.

        Returns:
            Bearer access token as string.

        Raises:
            AuthenticationError: If a valid token cannot be obtained.
        """
        if self._is_token_valid():
            logger.debug("Reusing cached token.")
            return self._access_token  # type: ignore[return-value]

        self._access_token = self._fetch_token()
        return self._access_token

    def invalidate_token(self) -> None:
        """Invalidate the cached token, forcing a new request on the next get_token() call."""
        logger.debug("Token manually invalidated.")
        self._access_token = None
        self._token_expires_at = 0.0


# Module-level shared instance (singleton per process)
_default_manager: Optional[TokenManager] = None


def get_token() -> str:
    """
    Convenience function to obtain an access token.

    Uses a global TokenManager instance to reuse tokens.

    Returns:
        Bearer access token as string.
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = TokenManager()
    return _default_manager.get_token()