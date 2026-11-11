"""
IBP Optimizer Log Extractor — CF Backend (data API only)
No Playwright. Extraction runs via:
  - local_agent.py on the user's Mac/Windows (always available)
  - Kyma CronJob/Job (when KYMA_NAMESPACE env var is set)
"""
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import threading
import zipfile
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
# CORS origins can be overridden via CORS_ORIGINS (comma-separated).
# Default only allows local Angular dev server; add your CF UI URL in production.
_default_origins = "http://localhost:4200"
CORS(app,
     origins=[o.strip() for o in os.getenv("CORS_ORIGINS", _default_origins).split(",") if o.strip()],
     supports_credentials=True)

DOWNLOAD_DIR = Path(os.getenv("DOWNLOAD_DIR", "./downloads")).resolve()
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

JOB_NAME         = os.getenv("JOB_NAME", "")
HANA_ADDRESS     = os.getenv("HANA_ADDRESS", "")
HANA_PORT        = int(os.getenv("HANA_PORT", "443"))
HANA_USER        = os.getenv("HANA_USER", "")
HANA_PASSWORD    = os.getenv("HANA_PASSWORD", "")
HANA_ENCRYPT     = os.getenv("HANA_ENCRYPT", "True").lower() == "true"
HANA_SCHEMA      = os.getenv("HANA_SCHEMA", "AICOE")
HANA_TABLE       = "IBP_OPTIMIZER_LOGS"
COOKIES_FILE     = Path(os.getenv("COOKIES_FILE", "./session_cookies.json")).resolve()

# Kyma configuration — set these env vars to enable Kyma mode
KYMA_NAMESPACE   = os.getenv("KYMA_NAMESPACE", "")       # e.g. "ibp-agent"
KYMA_JOB_IMAGE   = os.getenv("KYMA_JOB_IMAGE",
                               "<your-registry>/ibp-extractor-agent:latest")
KYMA_COOKIES_SECRET = os.getenv("KYMA_COOKIES_SECRET", "ibp-cookies")
# URL Kyma pods use to POST progress back to this backend.
# When running on CF, set this to the public backend URL via `cf set-env`.
CF_BACKEND_URL   = os.getenv("CF_BACKEND_URL", "https://<your-cf-backend-host>")
KYMA_ENABLED     = bool(KYMA_NAMESPACE)

# ── Job state (shared between /api/run, /api/run/status, /api/status) ─────────
_job_state: dict = {
    "status":     "idle",
    "message":    "",
    "log":        [],
    "jobName":    JOB_NAME,
    "csvFiles":   [],
    "file":       None,
    "hanaStatus": None,
}
_job_lock = threading.Lock()


def _set_state(**kwargs):
    with _job_lock:
        _job_state.update(kwargs)
        if "message" in kwargs and kwargs["message"]:
            _job_state["log"].append({
                "ts":  datetime.utcnow().isoformat() + "Z",
                "msg": kwargs["message"],
            })

_lock = threading.Lock()


# ── HANA helpers ──────────────────────────────────────────────────────────────

def _hana_connect():
    from hdbcli import dbapi
    return dbapi.connect(
        address=HANA_ADDRESS, port=HANA_PORT,
        user=HANA_USER, password=HANA_PASSWORD,
        encrypt=HANA_ENCRYPT, sslValidateCertificate=False,
    )


def _table_for_csv(csv_filename: str) -> str:
    return f"IBP_{Path(csv_filename).stem.upper()}"


def _ensure_hana_table(conn, table_name: str, columns: list[str]):
    cursor = conn.cursor()
    meta_cols = [
        '"ID"           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY',
        '"EXTRACTED_AT" NVARCHAR(30)',
        '"SOURCE_FILE"  NVARCHAR(255)',
        '"CSV_FILE"     NVARCHAR(255)',
        '"JOB_NAME"     NVARCHAR(500)',
    ]
    data_cols = [f'"{c.upper()}" NVARCHAR(500)' for c in columns]
    cursor.execute(
        f'CREATE TABLE IF NOT EXISTS "{HANA_SCHEMA}"."{table_name}" '
        f'({", ".join(meta_cols + data_cols)})'
    )
    cursor.execute(
        f"SELECT COLUMN_NAME FROM SYS.TABLE_COLUMNS "
        f"WHERE SCHEMA_NAME='{HANA_SCHEMA}' AND TABLE_NAME='{table_name}'"
    )
    existing = {r[0].upper() for r in cursor.fetchall()}
    for col in columns:
        if col.upper() not in existing:
            cursor.execute(
                f'ALTER TABLE "{HANA_SCHEMA}"."{table_name}" '
                f'ADD ("{col.upper()}" NVARCHAR(500))'
            )
    conn.commit()
    cursor.close()


def _insert_csv_to_hana(conn, rows, source_zip, csv_filename, job_name):
    if not rows:
        return 0, _table_for_csv(csv_filename)
    table_name = _table_for_csv(csv_filename)
    columns    = list(rows[0].keys())
    _ensure_hana_table(conn, table_name, columns)
    cursor    = conn.cursor()
    ts        = datetime.utcnow().isoformat() + "Z"
    col_names = ", ".join(
        ['"EXTRACTED_AT"', '"SOURCE_FILE"', '"CSV_FILE"', '"JOB_NAME"'] +
        [f'"{c.upper()}"' for c in columns]
    )
    placeholders = ", ".join(["?"] * (4 + len(columns)))
    sql  = f'INSERT INTO "{HANA_SCHEMA}"."{table_name}" ({col_names}) VALUES ({placeholders})'
    data = [
        [ts, source_zip, csv_filename, job_name] + [str(row.get(c, "")) for c in columns]
        for row in rows
    ]
    cursor.executemany(sql, data)
    conn.commit()
    cursor.close()
    return len(data), table_name


def _process_csv_bytes(content_bytes, csv_filename, ts, source_file, csv_files_info, all_rows_by_file):
    out_path = DOWNLOAD_DIR / f"{ts}_{csv_filename}"
    out_path.write_bytes(content_bytes)
    text  = content_bytes.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    if lines and lines[0].strip().lower().startswith("sep="):
        text = "\n".join(lines[1:])
    reader = csv.DictReader(io.StringIO(text), delimiter=";")
    rows   = [dict(r) for r in reader]
    all_rows_by_file[csv_filename] = rows
    csv_files_info.append({
        "filename":    out_path.name,
        "csvName":     csv_filename,
        "rowCount":    len(rows),
        "columns":     list(rows[0].keys()) if rows else [],
        "sizeBytes":   len(content_bytes),
        "checksum":    hashlib.sha256(content_bytes).hexdigest(),
        "extractedAt": datetime.utcnow().isoformat() + "Z",
    })


# ── REST API ──────────────────────────────────────────────────────────────────

@app.get("/api/status")
def get_status():
    with _job_lock:
        return jsonify(dict(_job_state))


@app.get("/api/mode")
def get_mode():
    return jsonify({
        "canExtract": KYMA_ENABLED,
        "platform":   "kyma" if KYMA_ENABLED else "cf",
    })


@app.post("/api/run")
def trigger_run():
    if not KYMA_ENABLED:
        return jsonify({
            "error": "Kyma is not configured. Set KYMA_NAMESPACE to enable server-side extraction, "
                     "or run the local IBP-Agent on your machine."
        }), 503

    with _job_lock:
        if _job_state["status"] == "running":
            return jsonify({"error": "An extraction is already in progress."}), 409

    body     = request.get_json(silent=True) or {}
    job_name = (body.get("jobName") or JOB_NAME).strip()
    if not job_name:
        return jsonify({"error": "jobName is required."}), 400

    _set_state(status="running", log=[], file=None, csvFiles=[],
               hanaStatus=None, jobName=job_name,
               message=f"Triggering Kyma job for: {job_name}")

    threading.Thread(target=_trigger_kyma_job, args=(job_name,), daemon=True).start()
    return jsonify({"started": True, "jobName": job_name, "platform": "kyma"})


def _trigger_kyma_job(job_name: str):
    """Apply a one-off Kyma Job to run the extraction."""
    import time
    try:
        # Load the job template and customise it
        template_path = Path(__file__).parent.parent / "kyma" / "job-trigger.yaml"
        if not template_path.exists():
            # Fallback: inline minimal job spec
            yaml_content = _build_job_yaml(job_name)
        else:
            yaml_content = template_path.read_text()
            yaml_content = yaml_content.replace("__JOB_NAME__", job_name)

        # Give the job a unique timestamped name
        ts        = datetime.now().strftime("%Y%m%d%H%M%S")
        job_name_k8s = f"ibp-extractor-{ts}"
        yaml_content = yaml_content.replace(
            "name: ibp-extractor-manual", f"name: {job_name_k8s}"
        )

        # Apply via kubectl
        result = subprocess.run(
            ["kubectl", "apply", "-f", "-", "-n", KYMA_NAMESPACE],
            input=yaml_content, capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            _set_state(status="error",
                       message=f"kubectl apply failed: {result.stderr[:300]}")
        else:
            _set_state(message=f"Kyma job '{job_name_k8s}' created. Waiting for extraction...")
    except FileNotFoundError:
        _set_state(status="error",
                   message="kubectl not found. Make sure it is installed in the CF container.")
    except Exception as e:
        _set_state(status="error", message=f"Failed to trigger Kyma job: {e}")


def _build_job_yaml(job_name: str) -> str:
    """Build a minimal Kyma Job YAML inline (fallback when job-trigger.yaml is not present)."""
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"""apiVersion: batch/v1
kind: Job
metadata:
  name: ibp-extractor-{ts}
  labels:
    app: ibp-extractor
    trigger: manual
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: ibp-extractor
          image: {KYMA_JOB_IMAGE}
          imagePullPolicy: Always
          env:
            - name: JOB_NAME
              value: "{job_name}"
            - name: CF_BACKEND_URL
              value: "{CF_BACKEND_URL}"
            - name: COOKIES_FILE
              value: "/app/cookies/session_cookies.json"
          volumeMounts:
            - name: cookies-volume
              mountPath: /app/cookies
              readOnly: true
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "1000m"
      volumes:
        - name: cookies-volume
          secret:
            secretName: {KYMA_COOKIES_SECRET}
"""


@app.post("/api/run/status")
def update_run_status():
    """Called by kyma_wrapper.py to push progress back to the CF backend."""
    body = request.get_json(silent=True) or {}
    allowed = {k: v for k, v in body.items()
               if k in ("status", "message", "file", "csvFiles", "hanaStatus")}
    if allowed:
        _set_state(**allowed)
    return jsonify({"ok": True})


@app.post("/api/upload")
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400
    f         = request.files["file"]
    job_name  = request.form.get("jobName", JOB_NAME).strip()
    ts        = datetime.now().strftime("%Y%m%d_%H%M")
    orig_name = f.filename or "upload"
    save_name = f"IBP1_TS_OPTIMIZER_LOG_{ts}_{orig_name}"
    save_path = DOWNLOAD_DIR / save_name
    raw       = f.read()
    save_path.write_bytes(raw)

    csv_files_info   = []
    all_rows_by_file = {}
    is_zip = raw[:2] == b'PK'

    if is_zip:
        with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
            for entry in [n for n in zf.namelist()
                          if n.lower().endswith(".csv") and not n.startswith("__MACOSX")]:
                _process_csv_bytes(zf.read(entry), Path(entry).name,
                                   ts, save_name, csv_files_info, all_rows_by_file)
    else:
        _process_csv_bytes(raw, orig_name, ts, save_name, csv_files_info, all_rows_by_file)

    manifest = {
        "zipFile": save_name, "savedAt": datetime.utcnow().isoformat() + "Z",
        "sizeBytes": len(raw), "checksum": hashlib.sha256(raw).hexdigest(),
        "jobName": job_name, "csvFiles": csv_files_info,
    }
    (DOWNLOAD_DIR / (save_name.rsplit(".", 1)[0] + "_manifest.json")).write_text(
        json.dumps(manifest, indent=2))

    hana_status = {}
    if HANA_ADDRESS:
        try:
            conn = _hana_connect()
            for csv_filename, rows in all_rows_by_file.items():
                count, table_name = _insert_csv_to_hana(conn, rows, save_name, csv_filename, job_name)
                hana_status[csv_filename] = {"inserted": count, "table": table_name}
            conn.close()
        except Exception as e:
            hana_status["error"] = str(e)

    return jsonify({"saved": save_name, "hanaStatus": hana_status, "csvFiles": csv_files_info})


@app.get("/api/files")
def list_files():
    files, seen = [], set()
    for f in sorted(DOWNLOAD_DIR.glob("IBP1_TS_OPTIMIZER_LOG_*"), reverse=True):
        if f.suffix == ".json" or f.name in seen:
            continue
        mp = DOWNLOAD_DIR / (f.name.rsplit(".", 1)[0] + "_manifest.json")
        m  = json.loads(mp.read_text()) if mp.exists() else {}
        seen.add(f.name)
        files.append({
            "filename":   f.name,
            "sizeBytes":  f.stat().st_size,
            "modifiedAt": datetime.utcfromtimestamp(f.stat().st_mtime).isoformat() + "Z",
            "checksum":   m.get("checksum", ""),
            "jobName":    m.get("jobName", JOB_NAME),
            "csvFiles":   m.get("csvFiles", []),
        })
    return jsonify(files)


@app.get("/api/files/<filename>/csv/<csvname>")
def get_csv_data(filename: str, csvname: str):
    if not re.match(r'^IBP1_TS_OPTIMIZER_LOG_[\w_.]+$', filename):
        return jsonify({"error": "Invalid filename."}), 400
    if not re.match(r'^[\w_.]+\.csv$', csvname):
        return jsonify({"error": "Invalid CSV filename."}), 400
    m = re.match(r'^IBP1_TS_OPTIMIZER_LOG_(\d{8}_\d{4})_', filename)
    ts = m.group(1) if m else ""
    csv_path = DOWNLOAD_DIR / f"{ts}_{csvname}"
    if not csv_path.exists():
        return jsonify({"error": f"CSV file not found: {ts}_{csvname}"}), 404
    content = csv_path.read_bytes()
    text    = content.decode("utf-8-sig", errors="replace")
    lines   = text.splitlines()
    if lines and lines[0].strip().lower().startswith("sep="):
        text = "\n".join(lines[1:])
    reader = csv.DictReader(io.StringIO(text), delimiter=";")
    rows   = [dict(r) for r in reader]
    cols   = list(rows[0].keys()) if rows else []
    return jsonify({"filename": csvname, "columns": cols, "rows": rows[:500], "totalRows": len(rows)})


@app.get("/api/files/<filename>/download")
def download_file(filename: str):
    if not re.match(r'^IBP1_TS_OPTIMIZER_LOG_[\w_.]+$', filename):
        return jsonify({"error": "Invalid filename."}), 400
    return send_from_directory(DOWNLOAD_DIR, filename, as_attachment=True)


@app.get("/api/session/status")
def session_status():
    has_cookies  = COOKIES_FILE.exists() and COOKIES_FILE.stat().st_size > 10
    cookie_count = 0
    saved_at     = None
    if has_cookies:
        try:
            data = json.loads(COOKIES_FILE.read_text())
            cookies      = data.get("cookies", data) if isinstance(data, dict) else data
            cookie_count = len(cookies) if isinstance(cookies, list) else 0
            saved_at     = data.get("savedAt") if isinstance(data, dict) else None
        except Exception:
            pass
    return jsonify({"hasCookies": has_cookies and cookie_count > 0,
                    "cookieCount": cookie_count, "savedAt": saved_at})


@app.post("/api/session/upload")
def session_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400
    raw = request.files["file"].read()
    try:
        cookies = json.loads(raw)
        if isinstance(cookies, list):
            payload = {"cookies": cookies, "savedAt": datetime.utcnow().isoformat() + "Z",
                       "count": len(cookies)}
        elif isinstance(cookies, dict) and "cookies" in cookies:
            payload = cookies
            payload["savedAt"] = datetime.utcnow().isoformat() + "Z"
        else:
            return jsonify({"error": "Invalid cookie format."}), 400
        COOKIES_FILE.write_text(json.dumps(payload, indent=2))
        return jsonify({"uploaded": True, "count": len(payload["cookies"])})
    except json.JSONDecodeError:
        return jsonify({"error": "File is not valid JSON."}), 400


@app.get("/api/hana/preview")
def hana_preview():
    if not HANA_ADDRESS:
        return jsonify({"error": "HANA not configured."}), 503
    try:
        conn   = _hana_connect()
        cursor = conn.cursor()
        cursor.execute(
            f'SELECT TOP 100 * FROM "{HANA_SCHEMA}"."{HANA_TABLE}" ORDER BY "EXTRACTED_AT" DESC'
        )
        cols = [d[0] for d in cursor.description]
        rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
        cursor.close(); conn.close()
        return jsonify({"columns": cols, "rows": rows})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.delete("/api/hana/clear")
def hana_clear():
    if not HANA_ADDRESS:
        return jsonify({"error": "HANA not configured."}), 503
    try:
        conn = _hana_connect()
        cursor = conn.cursor()
        deleted = 0
        for table in ["IBP_OPTEXPLLOG1", "IBP_OPTEXPLLOG2", HANA_TABLE]:
            try:
                cursor.execute(f'DELETE FROM "{HANA_SCHEMA}"."{table}"')
                deleted += cursor.rowcount
            except Exception:
                pass
        conn.commit(); cursor.close(); conn.close()
        files_deleted = _clear_downloads()
        return jsonify({"deleted": deleted, "filesDeleted": files_deleted})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _clear_downloads() -> int:
    count = 0
    for f in DOWNLOAD_DIR.iterdir():
        if f.is_file():
            try:
                f.unlink(); count += 1
            except Exception:
                pass
    return count


if __name__ == "__main__":
    port  = int(os.getenv("FLASK_PORT", "8080"))
    debug = os.getenv("FLASK_ENV", "production") != "production"
    app.run(host="0.0.0.0", port=port, debug=debug)
