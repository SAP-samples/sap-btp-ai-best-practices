"""
config_loader.py
----------------
Loads and normalizes SAP Document AI credentials from docai.json.

Automatically detects the following JSON structures:
  1. SAP BTP Service Key (flat with 'uaa' block)
  2. SAP BTP VCAP_SERVICES
  3. Direct flat structure

Returns a normalized DocAIConfig TypedDict with:
  - clientid
  - clientsecret
  - token_url
  - service_url
"""

import json
import logging
import os
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)

# Default path to the credentials file
DEFAULT_CREDENTIALS_PATH = Path(__file__).parent.parent / "docai.json"


class DocAIConfig(TypedDict):
    """Normalized SAP Document AI configuration."""
    clientid: str
    clientsecret: str
    token_url: str
    service_url: str


def _extract_from_uaa(credentials: dict) -> DocAIConfig:
    """
    Extract credentials from a structure containing a 'uaa' block.
    Typical format of SAP BTP Service Key.

    Priority for service_url:
      1. tenantuiurl  (tenant-specific URL, e.g. https://ar-demo-drxwlc9y.eu10.doc.cloud.sap)
      2. endpoints.backend.url
      3. url          (generic service URL)
    """
    uaa = credentials.get("uaa", {})

    clientid     = uaa.get("clientid") or credentials.get("clientid")
    clientsecret = uaa.get("clientsecret") or credentials.get("clientsecret")

    # token_url: uaa.url + /oauth/token is the standard SAP BTP format
    token_url = (
        uaa.get("tokenurl")
        or uaa.get("token_url")
        or (uaa.get("url", "").rstrip("/") + "/oauth/token" if uaa.get("url") else None)
    )

    # service_url: prefer tenantuiurl (tenant-specific URL with subdomain)
    endpoints_backend_url = (
        credentials.get("endpoints", {}).get("backend", {}).get("url")
    )
    service_url = (
        credentials.get("tenantuiurl")
        or endpoints_backend_url
        or credentials.get("url", "")
    )

    return DocAIConfig(
        clientid=clientid,
        clientsecret=clientsecret,
        token_url=token_url,
        service_url=service_url,
    )


def _extract_from_vcap(data: dict) -> DocAIConfig:
    """
    Extract credentials from a VCAP_SERVICES structure.
    Typical format of SAP BTP environment variables.
    """
    vcap = data.get("VCAP_SERVICES", {})

    # Search for the Document AI service under different possible keys
    service_keys = [
        "document-information-extraction",
        "document_information_extraction",
        "docai",
    ]

    credentials = None
    for key in service_keys:
        services = vcap.get(key, [])
        if services:
            credentials = services[0].get("credentials", {})
            logger.debug("Credentials found under VCAP_SERVICES['%s']", key)
            break

    if not credentials:
        raise ValueError(
            f"Document AI service not found in VCAP_SERVICES. "
            f"Keys searched: {service_keys}"
        )

    return _extract_from_uaa(credentials)


def _extract_flat(data: dict) -> DocAIConfig:
    """
    Extract credentials from a flat structure.
    Format: clientid, clientsecret, token_url/tokenurl/url directly in the JSON.
    """
    clientid     = data.get("clientid")
    clientsecret = data.get("clientsecret")

    token_url = (
        data.get("token_url")
        or data.get("tokenurl")
        or (data.get("url", "").rstrip("/") + "/oauth/token" if data.get("url") else None)
    )

    service_url = data.get("service_url") or data.get("url", "")

    return DocAIConfig(
        clientid=clientid,
        clientsecret=clientsecret,
        token_url=token_url,
        service_url=service_url,
    )


def _validate_config(config: DocAIConfig) -> None:
    """Validate that all required fields are present and non-empty."""
    required_fields = ["clientid", "clientsecret", "token_url"]
    missing = [field for field in required_fields if not config.get(field)]

    if missing:
        raise ValueError(
            f"Missing required fields in docai.json: {missing}. "
            "Check the structure of your credentials file."
        )


def load_config(credentials_path: Path | str | None = None) -> DocAIConfig:
    """
    Load and normalize SAP Document AI credentials.

    Args:
        credentials_path: Path to docai.json.
                          If None, uses DEFAULT_CREDENTIALS_PATH.

    Returns:
        DocAIConfig with clientid, clientsecret, token_url and service_url.

    Raises:
        FileNotFoundError: If the credentials file does not exist.
        ValueError: If the structure is invalid or required fields are missing.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    path = Path(credentials_path) if credentials_path else DEFAULT_CREDENTIALS_PATH

    # Validate file existence
    if not path.exists():
        raise FileNotFoundError(
            f"Credentials file not found: {path.resolve()}\n"
            "Ensure 'docai.json' exists in the project root directory."
        )

    logger.info("Loading credentials from: %s", path.resolve())

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Auto-detect structure
    if "VCAP_SERVICES" in data:
        logger.debug("Structure detected: VCAP_SERVICES")
        config = _extract_from_vcap(data)

    elif "uaa" in data:
        logger.debug("Structure detected: SAP BTP Service Key (with 'uaa' block)")
        config = _extract_from_uaa(data)

    elif "clientid" in data or "clientsecret" in data:
        logger.debug("Structure detected: flat")
        config = _extract_flat(data)

    else:
        raise ValueError(
            "Could not detect the structure of docai.json. "
            "Supported structures: VCAP_SERVICES, SAP BTP Service Key (with 'uaa'), flat."
        )

    _validate_config(config)
    logger.info("Credentials loaded successfully. clientid: %s...", config["clientid"][:10])

    return config