"""
local_agent.py — Local extraction agent for IBP Optimizer Log Extractor.

Runs on the user's Mac/Windows on port 5001.
The Angular UI (CF or localhost) calls this to trigger the Chrome-based extraction.

Usage:
    python local_agent.py
"""
import json
import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / "backend" / ".env")

CF_BACKEND = os.getenv(
    "CF_BACKEND_URL",
    "https://<your-cf-backend-host>"
)
DEFAULT_JOB = os.getenv("JOB_NAME", "<your-ibp-job-name>")
EXTRACT_SCRIPT = Path(__file__).parent / "extract.py"

app = Flask(__name__)
# CORS origins can be overridden via CORS_ORIGINS (comma-separated).
# Default only allows local Angular dev server; add your CF UI URL in production.
_default_origins = "http://localhost:4200"
CORS(app, origins=[o.strip() for o in os.getenv("CORS_ORIGINS", _default_origins).split(",") if o.strip()])

_state = {
    "status": "idle",
    "log": [],
    "message": "",
    "jobName": DEFAULT_JOB,
}
_lock = threading.Lock()


def _set(** kwargs):
    with _lock:
        _state.update(kwargs)
        if "message" in kwargs and kwargs["message"]:
            _state["log"].append({
                "ts":  datetime.utcnow().isoformat() + "Z",
                "msg": kwargs["message"]
            })


def _run(job_name: str):
    _set(status="running", log=[], message=f"Starting extraction: {job_name}", jobName=job_name)
    cmd = [sys.executable, str(EXTRACT_SCRIPT),
           "--job", job_name,
           "--backend", CF_BACKEND]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print(line, flush=True)
                _set(message=line)
        proc.wait()
        if proc.returncode == 0:
            _set(status="done", message="Extraction completed successfully.")
        else:
            _set(status="error", message=f"extract.py exited with code {proc.returncode}")
    except Exception as e:
        _set(status="error", message=f"ERROR: {e}")


@app.get("/api/status")
def status():
    with _lock:
        return jsonify(dict(_state))


@app.get("/api/mode")
def mode():
    return jsonify({"canExtract": True, "platform": "local"})


@app.post("/api/run")
def run():
    with _lock:
        if _state["status"] == "running":
            return jsonify({"error": "An extraction is already in progress."}), 409
    body     = request.get_json(silent=True) or {}
    job_name = (body.get("jobName") or DEFAULT_JOB).strip()
    threading.Thread(target=_run, args=(job_name,), daemon=True).start()
    return jsonify({"started": True, "jobName": job_name})


@app.get("/api/session/status")
def session_status():
    """Proxy to CF backend session status."""
    import urllib.request
    try:
        url = f"{CF_BACKEND}/api/session/status"
        r   = urllib.request.urlopen(url, timeout=5)
        return jsonify(json.loads(r.read()))
    except Exception as e:
        return jsonify({"hasCookies": False, "cookieCount": 0, "error": str(e)})


if __name__ == "__main__":
    port = int(os.getenv("LOCAL_PORT", "5001"))

    # Kill anything already on this port before starting
    import subprocess, sys
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True
        )
        pids = result.stdout.strip().split()
        for pid in pids:
            if pid:
                subprocess.run(["kill", "-9", pid], capture_output=True)
                print(f"Killed existing process {pid} on port {port}")
    except Exception:
        pass

    print(f"IBP Local Agent running on http://localhost:{port}")
    print(f"CF Backend: {CF_BACKEND}")
    print(f"Default job: {DEFAULT_JOB}")
    app.run(host="0.0.0.0", port=port, debug=False)
