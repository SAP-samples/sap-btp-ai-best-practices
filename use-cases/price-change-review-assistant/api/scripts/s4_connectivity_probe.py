"""Probe S/4 reachability from the deployed API app or local API folder.

Examples:
    cd api && ../.venv/bin/python scripts/s4_connectivity_probe.py
    cd api && BTP_DEST_DEBUG=true ../.venv/bin/python scripts/s4_connectivity_probe.py --skip-http
    cf run-task email-price-classifier-api --name s4-connectivity-probe --command \
      "python scripts/s4_connectivity_probe.py"
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is present in api/requirements.txt.

    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        """Fallback no-op dotenv loader when python-dotenv is unavailable.

        Returns:
            False because no environment file was loaded.
        """
        return False


API_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = API_ROOT.parent
for candidate in (API_ROOT, REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from app.price_changes.s4_btp_connectivity import load_s4_btp_connectivity_config_from_env
from app.price_changes.s4_lookup import (
    BUSINESS_PARTNER_SERVICE_NAME,
    S4Client,
    S4ConfigError,
    S4HTTPError,
    load_s4_config,
)


def load_probe_environment() -> None:
    """Load local environment files for direct CLI execution.

    Returns:
        None.
    """
    load_dotenv(API_ROOT / ".env", override=False)
    load_dotenv(REPO_ROOT / ".env", override=False)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Probe S/4 OData connectivity without printing secrets.")
    parser.add_argument(
        "--service",
        default=BUSINESS_PARTNER_SERVICE_NAME,
        help="S/4 OData service name to probe. Defaults to API_BUSINESS_PARTNER.",
    )
    parser.add_argument(
        "--path",
        default="/$metadata",
        help="Read-only service path to request. Defaults to /$metadata.",
    )
    parser.add_argument(
        "--skip-http",
        action="store_true",
        help="Only print mode and proxy preflight diagnostics without calling S/4.",
    )
    return parser


def tcp_probe(host: str, port: int, timeout: int = 8) -> tuple[bool, str]:
    """Probe TCP reachability for one host and port.

    Args:
        host: Hostname to connect to.
        port: TCP port.
        timeout: Connection timeout in seconds.

    Returns:
        Tuple with success flag and diagnostic message.
    """
    try:
        start = time.monotonic()
        with socket.create_connection((host, port), timeout=timeout):
            elapsed = time.monotonic() - start
        return True, f"tcp ok elapsed={elapsed:.2f}s"
    except Exception as exc:
        return False, f"tcp failed {exc.__class__.__name__}: {exc}"


def print_mode_diagnostics() -> None:
    """Print non-secret runtime mode and proxy diagnostics.

    Returns:
        None.
    """
    try:
        effective_config = load_s4_config()
    except S4ConfigError as exc:
        print(f"S4_PROBE config failed {exc.__class__.__name__}: {exc}")
        return
    if effective_config.runtime_context_provider is None:
        print("S4_PROBE mode=direct")
        parsed = urlparse(effective_config.base_url)
        if parsed.hostname:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            print(
                "S4_PROBE direct_target "
                f"scheme={parsed.scheme} host={parsed.hostname} port={port} verify={effective_config.verify}"
            )
        return

    btp_config = load_s4_btp_connectivity_config_from_env()
    if btp_config is None:
        print("S4_PROBE mode=btp-destination config=missing")
        return
    print(
        "S4_PROBE mode=btp-destination "
        f"destination={btp_config.destination_name} "
        f"proxy_host={btp_config.connectivity_proxy_host} "
        f"proxy_port={btp_config.connectivity_proxy_port} "
        f"verify={btp_config.verify}"
    )
    try:
        infos = socket.getaddrinfo(
            btp_config.connectivity_proxy_host,
            int(btp_config.connectivity_proxy_port),
            type=socket.SOCK_STREAM,
        )
        print(f"S4_PROBE proxy_dns ok count={len(infos)}")
    except Exception as exc:
        print(f"S4_PROBE proxy_dns failed {exc.__class__.__name__}: {exc}")
        return
    ok, message = tcp_probe(btp_config.connectivity_proxy_host, int(btp_config.connectivity_proxy_port))
    print(f"S4_PROBE proxy_{message}")
    if not ok:
        print("S4_PROBE proxy_tcp_warning=Connectivity proxy must be reachable from the target CF foundation.")


def probe_http(service_name: str, path: str) -> int:
    """Run a read-only S/4 OData HTTP probe.

    Args:
        service_name: S/4 OData service name.
        path: Read-only service path.

    Returns:
        Process exit code.
    """
    try:
        config = load_s4_config()
        parsed = urlparse(config.base_url) if config.base_url else None
        if parsed and parsed.hostname:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            print(
                "S4_PROBE direct_target "
                f"scheme={parsed.scheme} host={parsed.hostname} port={port} verify={config.verify}"
            )
        body, headers = S4Client(config).get_text(
            service_name,
            path=path,
            headers={"Accept": "application/xml"},
        )
    except S4ConfigError as exc:
        print(f"S4_PROBE config failed {exc.__class__.__name__}: {exc}")
        return 2
    except S4HTTPError as exc:
        print(f"S4_PROBE http failed status={exc.status_code} message={exc.message}")
        return 1
    except Exception as exc:
        print(f"S4_PROBE http failed {exc.__class__.__name__}: {exc}")
        return 1
    print(
        "S4_PROBE http ok "
        f"content_type={headers.get('content-type') or headers.get('Content-Type')} "
        f"bytes={len(body.encode('utf-8'))}"
    )
    return 0


def main() -> int:
    """Run the connectivity probe.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args()
    load_probe_environment()
    print_mode_diagnostics()
    if args.skip_http:
        return 0
    return probe_http(args.service, args.path)


if __name__ == "__main__":
    raise SystemExit(main())
