"""Read BTP destinations through the user's `btp` CLI session and build request auth.

The connector never stores service keys. It reuses whatever session the user
opened with the SAP BTP CLI, for example:

    btp login --url https://cli.btp.cloud.sap --subdomain ai4u --sso
    btp target

Destination secrets (passwords, client secrets, `URL.headers.*` values) stay in
memory inside this process. They are only ever turned into HTTP headers or
query parameters for the destination's own host and are never returned to the
MCP client.

Environment variables:
    A2A_BTP_SUBACCOUNT  Subaccount ID to read; defaults to the `btp target`.
    A2A_BTP_SUBDOMAIN   Global-account subdomain shown in login hints (default: ai4u).
    BTP_CLI_PATH        Explicit path to the `btp` binary when it is not on PATH.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

LOGIN_COMMAND = (
    "btp login --url https://cli.btp.cloud.sap "
    f"--subdomain {os.environ.get('A2A_BTP_SUBDOMAIN', 'ai4u')} --sso"
)
# Authentication types this connector can perform from a local machine.
# OAuth2UserTokenExchange, SAML and principal propagation need a user JWT or a
# BTP runtime, so agents using them are listed but not callable.
SUPPORTED_AUTH = frozenset(
    {"NoAuthentication", "BasicAuthentication", "OAuth2ClientCredentials"}
)
# GUI apps (Claude Desktop) start with a minimal PATH, so look in the usual
# install locations when `shutil.which` finds nothing.
_FALLBACK_BTP_PATHS = ("/opt/homebrew/bin/btp", "/usr/local/bin/btp", "~/bin/btp")
_HEADER_PREFIX = "URL.headers."
_QUERY_PREFIX = "URL.queries."
# (token URL, client ID) -> (access token, expiry as time.monotonic() value)
_token_cache: dict[tuple[str, str], tuple[str, float]] = {}


class DestinationError(RuntimeError):
    """A user-facing error about the BTP CLI session or a destination.

    Messages are safe to show to the MCP client: they never contain secrets.
    """


@dataclass(frozen=True)
class Destination:
    """One subaccount destination as returned by `btp list connectivity/destination`.

    Attributes:
        name: Destination name, used as the agent identifier.
        url: Destination base URL.
        auth: Destination `Authentication` type.
        description: Destination description (may be empty).
        config: Full destination configuration, including secrets. Excluded
            from `repr` so it cannot leak through logs or error messages.
    """

    name: str
    url: str
    auth: str
    description: str
    config: dict = field(repr=False, compare=False)


def find_btp() -> str:
    """Return the path of the `btp` executable.

    `BTP_CLI_PATH` wins when set; otherwise PATH is searched, then the common
    install locations. Raises `DestinationError` when no binary is found.
    """

    explicit = os.environ.get("BTP_CLI_PATH")
    if explicit:
        return explicit
    found = shutil.which("btp")
    if found:
        return found
    for candidate in _FALLBACK_BTP_PATHS:
        path = os.path.expanduser(candidate)
        if os.access(path, os.X_OK):
            return path
    raise DestinationError(
        "The SAP BTP CLI (`btp`) was not found. Install it, or set BTP_CLI_PATH "
        "to its full path in the MCP server environment."
    )


def run_btp(arguments: list[str]) -> str:
    """Run one non-interactive `btp` command and return its standard output.

    Args:
        arguments: CLI arguments after the `btp` binary.

    Raises:
        DestinationError: on a non-zero exit. When the CLI output suggests a
            missing or expired session, the message names the login command.
    """

    try:
        result = subprocess.run(
            [find_btp(), *arguments],
            capture_output=True,
            text=True,
            timeout=60,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise DestinationError("The `btp` CLI did not answer within 60 seconds.") from error
    if result.returncode == 0:
        return result.stdout
    detail = (result.stderr or result.stdout).strip()[:500]
    if any(word in detail.lower() for word in ("login", "log in", "session", "expired", "logged")):
        raise DestinationError(
            f"The BTP CLI session is missing or expired ({detail}). "
            f"Run `{LOGIN_COMMAND}`, then `btp target` to choose the subaccount."
        )
    raise DestinationError(f"`btp {' '.join(arguments[:3])}` failed: {detail or 'no detail'}")


def list_destinations() -> list[Destination]:
    """Return every destination of the targeted (or configured) subaccount.

    Uses `A2A_BTP_SUBACCOUNT` when set, otherwise the subaccount chosen with
    `btp target`. Raises `DestinationError` when the CLI output is not the
    expected JSON list.
    """

    arguments = ["--format", "json", "list", "connectivity/destination"]
    subaccount = os.environ.get("A2A_BTP_SUBACCOUNT")
    if subaccount:
        arguments += ["--subaccount", subaccount]
    output = run_btp(arguments)
    try:
        entries = json.loads(output)
    except json.JSONDecodeError as error:
        raise DestinationError("`btp list connectivity/destination` did not return JSON.") from error
    if not isinstance(entries, list):
        raise DestinationError("`btp list connectivity/destination` returned an unexpected shape.")
    destinations = []
    for entry in entries:
        # Each entry wraps the properties in `destinationConfiguration`.
        config = entry.get("destinationConfiguration") if isinstance(entry, dict) else None
        if not isinstance(config, dict) or not config.get("Name") or not config.get("URL"):
            continue
        destinations.append(
            Destination(
                name=config["Name"],
                url=config["URL"],
                auth=config.get("Authentication", "NoAuthentication"),
                description=config.get("Description") or "",
                config=config,
            )
        )
    return destinations


def request_auth(destination: Destination) -> tuple[dict[str, str], dict[str, str]]:
    """Build the headers and query parameters that authenticate a request.

    `URL.headers.<Name>` and `URL.queries.<Name>` destination properties are
    added as-is; the destination `Authentication` type then adds an
    `Authorization` header when needed.

    Args:
        destination: Destination to authenticate against.

    Returns:
        `(headers, query)` dictionaries to apply to requests for this
        destination's host only.

    Raises:
        DestinationError: for unsupported authentication types or a failed
            OAuth token request.
    """

    config = destination.config
    headers = {
        key[len(_HEADER_PREFIX):]: str(value)
        for key, value in config.items()
        if key.startswith(_HEADER_PREFIX)
    }
    query = {
        key[len(_QUERY_PREFIX):]: str(value)
        for key, value in config.items()
        if key.startswith(_QUERY_PREFIX)
    }
    if destination.auth == "BasicAuthentication":
        pair = f"{config.get('User', '')}:{config.get('Password', '')}".encode()
        headers["Authorization"] = "Basic " + base64.b64encode(pair).decode()
    elif destination.auth == "OAuth2ClientCredentials":
        headers["Authorization"] = "Bearer " + _client_credentials_token(destination)
    elif destination.auth not in SUPPORTED_AUTH:
        raise DestinationError(
            f"Destination '{destination.name}' uses {destination.auth}, which needs a "
            "user token or a BTP runtime and cannot be used from a local machine."
        )
    return headers, query


def _client_credentials_token(destination: Destination) -> str:
    """Fetch (or reuse) an OAuth2 client-credentials access token.

    The token is cached in memory until 60 seconds before it expires.

    Args:
        destination: An `OAuth2ClientCredentials` destination with
            `tokenServiceURL`, `clientId`, and `clientSecret` properties.

    Returns:
        The bearer access token.
    """

    config = destination.config
    token_url = config.get("tokenServiceURL", "")
    client_id = config.get("clientId", "")
    cached = _token_cache.get((token_url, client_id))
    if cached and cached[1] > time.monotonic():
        return cached[0]
    if not token_url or not client_id or not config.get("clientSecret"):
        raise DestinationError(
            f"Destination '{destination.name}' lacks tokenServiceURL, clientId, or clientSecret."
        )
    form = {"grant_type": "client_credentials"}
    if config.get("scope"):
        form["scope"] = config["scope"]
    pair = f"{client_id}:{config['clientSecret']}".encode()
    request = urllib.request.Request(
        token_url,
        data=urllib.parse.urlencode(form).encode(),
        headers={
            "Authorization": "Basic " + base64.b64encode(pair).decode(),
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as error:
        # Only the status code is reported; token endpoints may echo request data.
        raise DestinationError(
            f"OAuth token request for '{destination.name}' failed with HTTP {error.code}."
        ) from None
    except (OSError, ValueError) as error:
        raise DestinationError(
            f"OAuth token request for '{destination.name}' failed: {type(error).__name__}."
        ) from None
    token = body.get("access_token") if isinstance(body, dict) else None
    if not token:
        raise DestinationError(f"OAuth token response for '{destination.name}' had no access_token.")
    lifetime = float(body.get("expires_in", 300))
    _token_cache[(token_url, client_id)] = (token, time.monotonic() + max(lifetime - 60, 0))
    return token
