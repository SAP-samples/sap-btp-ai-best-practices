"""
Simple manual tester for SAP DOX schemas in tools/.

Purpose:
- Load each local schema JSON (e.g., `CGV_schema.json`) to get schema id/name/version
- Upload a representative PDF for that schema
- Wait for completion and print extracted header fields (name, value, confidence)

Notes:
- Uses the same `SapDoxClient` as the app, initialized from `service_key.json` by default
- Tries multiple fallbacks if the schema id/version is invalid in the tenant:
  1) schemaId + schemaVersion (from JSON)
  2) schemaId only (let DOX pick active version)
  3) schemaName only
- You can filter which schemas to test and override sample PDF paths with CLI options.

This script avoids the API router and calls the client directly: upload -> wait -> print.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make sure tools/ is importable when running this file directly
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in os.sys.path:
    os.sys.path.append(str(TOOLS_DIR))

from sap_dox_client import SapDoxClient, ServiceKey  # type: ignore


def resolve_project_root() -> Path:
    """Return the project root based on this file location.

    tools/ -> routers/ -> api/ -> project root (3 parents up)
    """
    return Path(__file__).resolve().parents[3]


def default_service_key_path() -> Path:
    """Return the default service key path if present under tools/."""
    for name in ("service_key.json", "service_key2.json"):
        candidate = TOOLS_DIR / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("service key not found in tools/; pass --service-key")


def load_local_schema_jsons(schema_dir: Path) -> Dict[str, Dict]:
    """Load known schema JSON files from tools/ and key them by base schema name.

    The key is inferred from the filename prefix before `_schema.json`.
    For example: `CGV_schema.json` -> key `CGV`.
    """
    results: Dict[str, Dict] = {}
    for path in sorted(schema_dir.glob("*_schema.json")):
        name = path.name
        if not name.endswith("_schema.json"):
            continue
        key = name[: -len("_schema.json")]
        try:
            with open(path, "r", encoding="utf-8") as f:
                results[key] = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load {name}: {e}")
    return results


def pick_sample_pdf(project_root: Path, schema_key: str, overrides: Dict[str, Path]) -> Optional[Path]:
    """Pick a representative PDF for the given schema.

    - Use overrides first when provided.
    - Otherwise, try a set of known sample documents under docs/.
    - Return None if nothing is found.
    """
    # Override provided by user
    if schema_key in overrides:
        override = overrides[schema_key]
        if override.is_file():
            return override
        print(f"[WARN] Override PDF for {schema_key} not found: {override}")

    docs_dir = project_root / "docs"
    documents_dir = docs_dir / "documents"
    datos_dir = docs_dir / "datos"

    candidates: List[Path] = []

    # Heuristic candidates per schema
    if schema_key.lower() in {"conoce_cliente", "conoce-cliente", "conocecliente"}:
        candidates += [
            documents_dir / "2025-05-21 Conoce a tu cliente.pdf",
        ]
    elif schema_key.lower() in {"comentarios_vendedor", "comentarios-vendedor", "comentariosvendedor"}:
        candidates += [
            documents_dir / "2025-05-23 Comentarios del vendedor copy.pdf",
        ]
    elif schema_key.lower() in {"csf", "constancia_fiscal", "constancia-fiscal"}:
        candidates += [
            documents_dir / "2025-05-21 CSF.pdf",
        ]
    elif schema_key.lower() in {"cgv"}:
        candidates += [
            documents_dir / "2025-06-06 FIRMADO Condiciones generales de venta I.M. FRUTAS SECAS.pdf",
            documents_dir / "2025-06-06 Condiciones generales de venta I.M. FRUTAS SECAS.pdf",
        ]
    elif schema_key.lower() in {"investigacion_comercial", "investigacion-comercial"}:
        candidates += [
            docs_dir / "datos" / "FaseDeInvestigación" / "2025-06-03 INFOMEX.pdf",
        ]
    elif schema_key.lower() in {"investigacion_legal", "investigacion-legal"}:
        candidates += [
            docs_dir / "datos" / "FaseDeInvestigación" / "2025-06-05 Investigación Legal FRUTAS SECAS DE CALIDAD.pdf",
            documents_dir / "Investigación_LivekDelCaribe 2025 (002).pdf",
        ]

    # Return the first existing candidate
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return None


def ensure_client(service_key_path: Path) -> SapDoxClient:
    """Create a SapDoxClient from the provided service key path."""
    with open(service_key_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    sk = ServiceKey.from_json(raw)
    return SapDoxClient(sk)


def try_upload(
    client: SapDoxClient,
    file_path: Path,
    client_id: str,
    schema_id: Optional[str],
    schema_name: Optional[str],
    schema_version: Optional[str],
) -> Tuple[Optional[Dict], Optional[str]]:
    """Attempt upload with progressive fallbacks.

    Returns a tuple (job_json, mode_used) where mode_used is one of:
      - "schema_id+version"
      - "schema_id"
      - "schema_name"
      - None (if all attempts fail)
    """
    # 1) id+version (only if both provided)
    if schema_id and schema_version:
        try:
            job = client.upload_document(
                file_path=str(file_path),
                client_id=client_id,
                schema_id=schema_id,
                schema_version=str(schema_version),
            )
            return job, "schema_id+version"
        except Exception as e:
            print(f"[WARN] Upload with schemaId+version failed: {e}")

    # 2) id only
    if schema_id:
        try:
            job = client.upload_document(
                file_path=str(file_path),
                client_id=client_id,
                schema_id=schema_id,
            )
            return job, "schema_id"
        except Exception as e:
            print(f"[WARN] Upload with schemaId failed: {e}")

    # 3) name only
    if schema_name:
        try:
            job = client.upload_document(
                file_path=str(file_path),
                client_id=client_id,
                schema_name=schema_name,
            )
            return job, "schema_name"
        except Exception as e:
            print(f"[WARN] Upload with schemaName failed: {e}")

    return None, None


def print_result(job_payload: Dict) -> None:
    """Print extraction header fields in a readable way."""
    fields = (((job_payload or {}).get("extraction") or {}).get("headerFields") or [])
    print("  Extracted header fields:")
    if not fields:
        print("    (none)")
        return
    for item in fields:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        value = item.get("value")
        conf = item.get("confidence")
        print(f"    - {name}: {value} (conf={conf})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual DOX schema tester: upload -> retrieve -> print")
    parser.add_argument(
        "--service-key",
        type=Path,
        help="Path to DOX service key JSON (defaults to tools/service_key.json)",
    )
    parser.add_argument(
        "--client-id",
        default=os.getenv("DOX_TENANT_CLIENT_ID") or os.getenv("DOX_TENANT_ID") or "default",
        help="DOX tenant clientId for jobs (not OAuth client id)",
    )
    parser.add_argument(
        "--only",
        help="Comma-separated schema keys to test (e.g., CGV,CSF)",
    )
    parser.add_argument(
        "--set-pdf",
        action="append",
        default=[],
        help="Override sample PDF for a schema, format KEY=/abs/path.pdf (repeatable)",
    )

    args = parser.parse_args()

    project_root = resolve_project_root()
    schema_dir = TOOLS_DIR
    service_key_path = args.service_key or default_service_key_path()

    # Parse overrides
    overrides: Dict[str, Path] = {}
    for pair in args.set_pdf:
        if "=" in pair:
            key, val = pair.split("=", 1)
        elif ":" in pair:
            key, val = pair.split(":", 1)
        else:
            print(f"[WARN] Ignoring invalid --set-pdf value: {pair}")
            continue
        overrides[key] = Path(val).expanduser().resolve()

    # Filter selection
    only_keys: Optional[List[str]] = None
    if args.only:
        only_keys = [s.strip() for s in args.only.split(",") if s.strip()]

    # Load local schemas
    local_schemas = load_local_schema_jsons(schema_dir)
    if not local_schemas:
        print(f"[ERROR] No *_schema.json files found under {schema_dir}")
        return

    # Establish client
    client = ensure_client(service_key_path)

    # Iterate schemas
    for schema_key, schema_json in local_schemas.items():
        if only_keys and schema_key not in only_keys:
            continue

        print(f"\n=== Testing schema: {schema_key} ===")
        schema_id = schema_json.get("id")
        schema_name = schema_json.get("name")
        schema_version = str(schema_json.get("version")) if schema_json.get("version") is not None else None
        print(f"  Local schema id={schema_id}, name={schema_name}, version={schema_version}")

        sample_pdf = pick_sample_pdf(project_root, schema_key, overrides)
        if not sample_pdf:
            print(f"  [SKIP] No sample PDF found for {schema_key}. Use --set-pdf {schema_key}=/abs/file.pdf")
            continue
        print(f"  Using PDF: {sample_pdf}")

        try:
            job, mode = try_upload(
                client=client,
                file_path=sample_pdf,
                client_id=args.client_id,
                schema_id=schema_id,
                schema_name=schema_name,
                schema_version=schema_version,
            )
            if not job:
                print("  [ERROR] All upload attempts failed.")
                continue

            job_id = job.get("id")
            print(f"  Upload OK via {mode}, jobId={job_id}")

            # Wait for completion
            result = client.wait_for_result(job_id=job_id, timeout_seconds=180, poll_interval_seconds=3)
            status = (result.get("status") or result.get("state") or "").upper()
            print(f"  Final status: {status}")
            print_result(result)
        except Exception as e:
            print(f"  [ERROR] Exception while processing {schema_key}: {e}")


if __name__ == "__main__":
    main()


