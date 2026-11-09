"""SAP CDC (Customer Data Cloud) service.

- JWT validation for tokens issued by SAP CDC / Gigya via OTP email login.
- Server-side account writes using admin credentials (user key + secret).
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, Optional

import jwt
import requests
from cryptography.hazmat.primitives.asymmetric import rsa

logger = logging.getLogger(__name__)

# Public/consumer email domains — excluded from company_domain tracking.
# Override via CDC_PUBLIC_EMAIL_DOMAINS env var (comma-separated).
DEFAULT_PUBLIC_EMAIL_DOMAINS = {
    "gmail.com", "hotmail.com", "outlook.com", "yahoo.com", "icloud.com",
    "aol.com", "proton.me", "protonmail.com", "mail.com", "gmx.com",
    "live.com", "msn.com", "me.com", "yandex.com", "zoho.com", "pm.me",
}


def _extract_email_domain(email: str) -> Optional[str]:
    """Extract domain from email address, lowercase."""
    parts = (email or "").strip().lower().split("@")
    return parts[1] if len(parts) == 2 and parts[1] else None


def _is_public_email_domain(domain: str) -> bool:
    """Check if domain is a public/consumer email provider."""
    public_domains_env = os.getenv("CDC_PUBLIC_EMAIL_DOMAINS")
    if public_domains_env:
        domains = {d.strip().lower() for d in public_domains_env.split(",") if d.strip()}
    else:
        domains = DEFAULT_PUBLIC_EMAIL_DOMAINS
    return (domain or "").lower() in domains


def _normalize_data_center(raw_data_center: Optional[str]) -> str:
    value = (raw_data_center or "us1.gigya.com").strip().lower()
    value = value.replace("https://", "").replace("http://", "")
    if value.startswith("accounts."):
        value = value[len("accounts."):]
    return value.strip("/") or "us1.gigya.com"


def get_cdc_base_url() -> str:
    data_center = _normalize_data_center(os.getenv("CDC_DATA_CENTER"))
    return f"https://accounts.{data_center}"


def _decode_base64_url(value: str) -> bytes:
    import base64

    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(f"{value}{padding}")


def _build_rsa_key(modulus: str, exponent: str):
    if not modulus or not exponent:
        return None

    try:
        n_int = int.from_bytes(_decode_base64_url(modulus), "big")
        e_int = int.from_bytes(_decode_base64_url(exponent), "big")
        return rsa.RSAPublicNumbers(e_int, n_int).public_key()
    except Exception as exc:
        logger.error("Failed to construct CDC public key: %s", exc)
        return None


def _get_public_key_by_kid(kid: Optional[str], keys_data: Dict[str, Any]):
    if not keys_data:
        return None

    if kid and keys_data.get("kid") == kid:
        return _build_rsa_key(keys_data.get("n", ""), keys_data.get("e", ""))

    for key in keys_data.get("keys", []):
        if key.get("kid") == kid:
            return _build_rsa_key(key.get("n", ""), key.get("e", ""))

    if not kid:
        if keys_data.get("n") and keys_data.get("e"):
            return _build_rsa_key(keys_data["n"], keys_data["e"])
        if keys_data.get("keys"):
            first_key = keys_data["keys"][0]
            return _build_rsa_key(first_key.get("n", ""), first_key.get("e", ""))

    return None


@lru_cache(maxsize=1)
def fetch_cdc_public_keys() -> Dict[str, Any]:
    """Fetch CDC/Gigya JWT public keys (cached)."""
    api_key = os.getenv("CDC_API_KEY")
    if not api_key:
        raise ValueError("CDC_API_KEY is not configured on the server.")

    url = f"{get_cdc_base_url()}/accounts.getJWTPublicKey"
    response = requests.get(url, params={"apiKey": api_key, "V2": "true"}, timeout=20)
    response.raise_for_status()
    payload = response.json()

    if payload.get("errorCode") != 0:
        raise ValueError(
            f"CDC public key fetch failed: {payload.get('errorMessage', 'unknown error')}"
        )

    return payload


def validate_cdc_jwt(token: str) -> Dict[str, Any]:
    """Validate a CDC JWT token and return its decoded claims.

    Raises ValueError on any validation failure.
    """
    if not token:
        raise ValueError("CDC token is missing.")

    try:
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")

        keys_data = fetch_cdc_public_keys()
        public_key = _get_public_key_by_kid(kid, keys_data)

        if not public_key:
            # Key rotation — clear cache and retry
            fetch_cdc_public_keys.cache_clear()
            keys_data = fetch_cdc_public_keys()
            public_key = _get_public_key_by_kid(kid, keys_data)

        if not public_key:
            raise ValueError("Could not find matching CDC signing key.")

        decoded = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            options={"verify_aud": False, "verify_iss": False},
        )

        # Verify the token's apiKey matches our configured key
        configured_api_key = os.getenv("CDC_API_KEY")
        token_api_key = decoded.get("apiKey")
        if configured_api_key and token_api_key and token_api_key != configured_api_key:
            raise ValueError("CDC token apiKey does not match server configuration.")

        return decoded
    except jwt.ExpiredSignatureError as exc:
        raise ValueError("Token has expired.") from exc
    except jwt.InvalidTokenError as exc:
        raise ValueError(f"Invalid token: {exc}") from exc


# ---------------------------------------------------------------------------
# Server-side CDC account writes (admin credentials)
# ---------------------------------------------------------------------------


def _get_cdc_credentials() -> Dict[str, str]:
    """Load CDC admin credentials from environment variables."""
    api_key = os.getenv("CDC_API_KEY")
    user_key = os.getenv("CDC_USER_KEY")
    secret_key = os.getenv("CDC_SECRET_KEY")
    if not all([api_key, user_key, secret_key]):
        raise ValueError("CDC admin credentials (CDC_API_KEY, CDC_USER_KEY, CDC_SECRET_KEY) are not fully configured.")
    return {"apiKey": api_key, "userKey": user_key, "secret": secret_key}


def _search_account_by_email(email: str) -> Optional[str]:
    """Search for a CDC account by email. Returns UID or None."""
    credentials = _get_cdc_credentials()
    escaped_email = email.replace("'", "''")
    query = f"SELECT UID FROM accounts WHERE profile.email = '{escaped_email}'"

    url = f"{get_cdc_base_url()}/accounts.search"
    params = {
        "apiKey": credentials["apiKey"],
        "userKey": credentials["userKey"],
        "secret": credentials["secret"],
        "query": query,
        "format": "json",
    }

    response = requests.post(url, data=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if payload.get("errorCode") != 0:
        logger.warning("CDC accounts.search failed: %s", payload.get("errorMessage"))
        return None

    results = payload.get("results", [])
    if results:
        return results[0].get("UID")
    return None


def _get_account_data(uid: str) -> Dict[str, Any]:
    """Get account data fields for a UID."""
    credentials = _get_cdc_credentials()

    url = f"{get_cdc_base_url()}/accounts.getAccountInfo"
    params = {
        "apiKey": credentials["apiKey"],
        "userKey": credentials["userKey"],
        "secret": credentials["secret"],
        "UID": uid,
        "include": "data",
        "format": "json",
    }

    response = requests.post(url, data=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if payload.get("errorCode") != 0:
        logger.warning("CDC accounts.getAccountInfo failed: %s", payload.get("errorMessage"))
        return {}

    return payload.get("data", {})


def _update_account(uid: str, data: Dict[str, Any], preferences: Dict[str, Any]) -> None:
    """Update an existing CDC account with data and preferences."""
    credentials = _get_cdc_credentials()

    url = f"{get_cdc_base_url()}/accounts.setAccountInfo"
    params = {
        "apiKey": credentials["apiKey"],
        "userKey": credentials["userKey"],
        "secret": credentials["secret"],
        "UID": uid,
        "format": "json",
    }
    if data:
        params["data"] = json.dumps(data)
    if preferences:
        params["preferences"] = json.dumps(preferences)

    response = requests.post(url, data=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if payload.get("errorCode") != 0:
        raise ValueError(
            f"CDC accounts.setAccountInfo failed: {payload.get('errorMessage', 'unknown error')}"
        )


def _create_account_and_set_data(
    email: str, data: Dict[str, Any], preferences: Dict[str, Any]
) -> None:
    """Create a full CDC account and set data + preferences.

    Uses the initRegistration → register two-step flow with admin credentials.
    Creates a full (non-lite) account with the given email as profile.
    """
    credentials = _get_cdc_credentials()

    # Step 1: Get a registration token
    init_url = f"{get_cdc_base_url()}/accounts.initRegistration"
    init_params = {
        "apiKey": credentials["apiKey"],
        "userKey": credentials["userKey"],
        "secret": credentials["secret"],
        "format": "json",
    }

    init_response = requests.post(init_url, data=init_params, timeout=30)
    init_response.raise_for_status()
    init_payload = init_response.json()

    reg_token = init_payload.get("regToken")
    if not reg_token:
        raise ValueError(
            f"CDC accounts.initRegistration failed: {init_payload.get('errorMessage', 'no regToken returned')}"
        )

    # Step 2: Register full account with profile, data, and preferences
    # A random password satisfies CDC site policy; users authenticate via OTP, not password.
    reg_url = f"{get_cdc_base_url()}/accounts.register"
    reg_params = {
        "apiKey": credentials["apiKey"],
        "userKey": credentials["userKey"],
        "secret": credentials["secret"],
        "regToken": reg_token,
        "email": email,
        "password": secrets.token_urlsafe(32),
        "profile": json.dumps({"email": email}),
        "finalizeRegistration": "true",
        "format": "json",
    }
    if data:
        reg_params["data"] = json.dumps(data)
    if preferences:
        reg_params["preferences"] = json.dumps(preferences)

    reg_response = requests.post(reg_url, data=reg_params, timeout=30)
    reg_response.raise_for_status()
    reg_payload = reg_response.json()

    if reg_payload.get("errorCode") != 0:
        raise ValueError(
            f"CDC accounts.register failed: {reg_payload.get('errorMessage', 'unknown error')}"
        )


def store_user_login_data(email: str) -> None:
    """Store consent preferences, login timestamps, and company domain for a user in CDC.

    Uses admin credentials (user key + secret) for server-side writes.
    - If account exists: update data fields
    - If account doesn't exist: create a full account, then set data
    Always writes: ideation_agent_last_login, company_domain, preferences
    Conditionally writes: ideation_agent_first_login (only if not already set)
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    preferences = {
        "privacy.ideation_agent": {"isConsentGranted": True},
        "terms.ideation_agent": {"isConsentGranted": True},
    }

    # Determine company_domain (empty string for public email domains)
    email_domain = _extract_email_domain(email)
    company_domain = email_domain if email_domain and not _is_public_email_domain(email_domain) else ""

    uid = _search_account_by_email(email)

    if uid:
        # Account exists — read existing data to check first_login
        existing_data = _get_account_data(uid)
        data = {
            "ideation_agent_last_login": now,
            "company_domain": company_domain,
        }
        if not existing_data.get("ideation_agent_first_login"):
            data["ideation_agent_first_login"] = now

        _update_account(uid, data, preferences)
        logger.info("CDC login tracked for existing account: %s", email)
    else:
        # No account — create full account with all data
        data = {
            "ideation_agent_first_login": now,
            "ideation_agent_last_login": now,
            "company_domain": company_domain,
        }
        _create_account_and_set_data(email, data, preferences)
        logger.info("CDC account created and login tracked: %s", email)
