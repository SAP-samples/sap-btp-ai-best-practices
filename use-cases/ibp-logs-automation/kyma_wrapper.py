"""
kyma_wrapper.py — Runs extract.py inside a Kyma Job/CronJob pod.

Streams progress back to the CF backend via POST /api/run/status
so the UI shows real-time progress just like the local agent.

Environment variables (set in kyma/cronjob.yaml):
  JOB_NAME         — IBP Application Job name to extract
  CF_BACKEND_URL   — CF backend base URL
  COOKIES_FILE     — path to session_cookies.json (mounted from Secret)
"""
import os
import sys
import subprocess
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

CF_BACKEND   = os.getenv("CF_BACKEND_URL",
                          "https://<your-cf-backend-host>")
JOB_NAME     = os.getenv("JOB_NAME", "")
COOKIES_FILE = os.getenv("COOKIES_FILE", "/app/session_cookies.json")
STATUS_URL   = f"{CF_BACKEND.rstrip('/')}/api/run/status"


def push_status(status: str, message: str = ""):
    """Push extraction status to CF backend (best-effort — never crash on failure)."""
    try:
        requests.post(STATUS_URL,
                      json={"status": status, "message": message},
                      timeout=10)
    except Exception:
        pass


def main():
    if not JOB_NAME:
        push_status("error", "JOB_NAME environment variable is not set.")
        sys.exit(1)

    push_status("running", f"Kyma job started: {JOB_NAME}")

    # Build command — use xvfb-run so Chrome gets a virtual display
    python  = sys.executable
    script  = Path(__file__).parent / "extract.py"
    cmd = [
        "xvfb-run", "--auto-servernum", "--server-args=-screen 0 1920x1080x24",
        python, str(script),
        "--job", JOB_NAME,
        "--backend", CF_BACKEND,
    ]

    # Override cookies file path for the subprocess via env
    env = os.environ.copy()
    env["COOKIES_FILE"] = COOKIES_FILE

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print(line, flush=True)
                push_status("running", line)
        proc.wait()

        if proc.returncode == 0:
            push_status("done", "Extraction completed successfully.")
        else:
            push_status("error", f"extract.py exited with code {proc.returncode}")
            sys.exit(proc.returncode)

    except Exception as e:
        push_status("error", f"Kyma wrapper error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
