#!/usr/bin/env python3
"""
Streamlit launcher for the SAP RFQx application.

This script mirrors the SAP PoC template bootstrap while pointing to the
multi-page RFQx UI. Run with:

    python streamlit_app.py
    PORT=8080 python streamlit_app.py
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

# Configure logging similar to the template
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for launching Streamlit."""
    port = int(os.getenv("PORT", "8501"))
    host = "0.0.0.0"

    repo_root = Path(__file__).resolve().parent
    app_entry = repo_root / "RFQx.py"

    if not app_entry.exists():
        logger.error(f"Streamlit entry point not found at {app_entry}")
        sys.exit(1)

    logger.info(f"Starting SAP RFQx Streamlit app on {host}:{port}")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_entry),
        "--server.port",
        str(port),
        "--server.address",
        host,
        "--server.headless",
        "true",
        "--server.enableCORS",
        "false",
        "--server.enableXsrfProtection",
        "false",
        "--server.baseUrlPath",
        "",
        "--browser.gatherUsageStats",
        "false",
        "--theme.primaryColor",
        "#0a6ed1",
        "--theme.backgroundColor",
        "#FFFFFF",
        "--theme.secondaryBackgroundColor",
        "#f5f6f7",
        "--theme.textColor",
        "#32363a",
        "--theme.font",
        "sans serif",
    ]

    logger.info("Executing: %s", " ".join(cmd))

    env = os.environ.copy()
    python_path_parts = [
        str(repo_root),
        str(repo_root / "static"),
        env.get("PYTHONPATH", ""),
    ]
    python_path = os.pathsep.join(part for part in python_path_parts if part)
    env["PYTHONPATH"] = python_path

    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        logger.error("Streamlit failed to start: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Streamlit application interrupted")
        sys.exit(0)


if __name__ == "__main__":
    main()
