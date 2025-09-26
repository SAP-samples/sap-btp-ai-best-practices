#!/usr/bin/env python3
"""
Streamlit application entry point for Cloud Foundry deployment.
This runs the Streamlit UI as a standalone application.

Command examples:
python streamlit_app.py
export PORT=8080 && python streamlit_app.py
"""

import os
import sys
import subprocess
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for Streamlit application."""
    # Get port from environment (Cloud Foundry sets this)
    port = int(os.getenv("PORT", "8501"))

    # Cloud Foundry requires binding to 0.0.0.0
    host = "0.0.0.0"

    logger.info(f"Starting Streamlit on {host}:{port}")

    # Build Streamlit command
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "src/Home.py",
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

    logger.info(f"Executing: {' '.join(cmd)}")

    # Run Streamlit
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = "ui" + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        logger.error(f"Streamlit failed to start: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Streamlit application interrupted")
        sys.exit(0)


if __name__ == "__main__":
    main()
