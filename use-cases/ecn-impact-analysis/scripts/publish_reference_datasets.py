"""Publish reference datasets and selected ECNs to a running ECN Impact API.

Reads a manifest (see scripts/manifests/q900v-2026-09.json), uploads every
workbook through the same Settings endpoint the UI uses, then imports only the
requested ECN numbers from a JSON export. All data lands in HANA through the
API, so table creation and validation stay identical to the UI flow.

Examples:
    # API running locally on port 8000, key taken from API_KEY or api/.env
    .venv/bin/python scripts/publish_reference_datasets.py \
        --manifest scripts/manifests/q900v-2026-09.json

    # Only re-import the ECNs (datasets already published)
    .venv/bin/python scripts/publish_reference_datasets.py \
        --manifest scripts/manifests/q900v-2026-09.json --skip-datasets

    # Assess BOM usage as of a specific date and target another API
    .venv/bin/python scripts/publish_reference_datasets.py \
        --manifest scripts/manifests/q900v-2026-09.json \
        --api-url http://127.0.0.1:8001 --api-key ... --assessment-date 2026-09-24
"""

import argparse
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

import httpx
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
XLSX_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def parse_args(argv=None):
    """Read CLI options.

    Args:
        argv: Optional argument list (defaults to sys.argv).

    Returns:
        argparse.Namespace with manifest, api_url, api_key, skip flags and assessment_date.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path, help="Manifest JSON path")
    parser.add_argument("--api-url", default=os.getenv("ECN_API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=os.getenv("API_KEY"), help="X-API-Key value (default: API_KEY env or api/.env)")
    parser.add_argument("--skip-datasets", action="store_true", help="Do not publish workbooks")
    parser.add_argument("--skip-ecns", action="store_true", help="Do not import ECNs")
    parser.add_argument("--assessment-date", default="", help="BOM usage date (YYYY-MM-DD); blank = upload date")
    return parser.parse_args(argv)


def select_ecns(export_path, numbers):
    """Return only the requested ECN records from a JSON export, as upload bytes.

    Args:
        export_path: Path to a JSON file holding a list of ECN records.
        numbers: ECN numbers to keep (matched on the "Number" field).

    Returns:
        UTF-8 JSON bytes containing the selected records, in export order.

    Raises:
        SystemExit: when any requested number is missing from the export.
    """
    records = json.loads(export_path.read_text(encoding="utf-8"))
    selected = [record for record in records if record.get("Number") in set(numbers)]
    missing = set(numbers) - {record["Number"] for record in selected}
    if missing:
        sys.exit(f"ECNs not found in {export_path.name}: {', '.join(sorted(missing))}")
    return json.dumps(selected).encode("utf-8")


def publish(manifest, client, args):
    """Upload every manifest workbook, then the selected ECNs.

    Args:
        manifest: Parsed manifest dictionary.
        client: httpx.Client bound to the API base URL and key.
        args: Parsed CLI options.

    Returns:
        List of (label, response JSON) tuples for the final report.
    """
    group = manifest["data_group"]
    results = []
    steps = ([] if args.skip_datasets else manifest["publications"]) + ([] if args.skip_ecns else [manifest["ecns"]])
    for item in tqdm(steps, desc="Publishing", unit="file"):
        path = ROOT / item["path"]
        if "dataset_type" in item:
            data = {"dataset_type": item["dataset_type"], "context_json": json.dumps({**item["context"], "data_group": group})}
            files = {"source_file": (path.name, path.read_bytes(), XLSX_TYPE)}
            response = client.post("/api/reference-datasets", data=data, files=files)
            label = item["dataset_type"]
        else:
            data = {"request_id": str(uuid4()), "data_group": group}
            if args.assessment_date:
                data["assessment_date"] = args.assessment_date
            files = {"ecn_file": ("selected-ecns.json", select_ecns(path, item["numbers"]), "application/json")}
            response = client.post("/api/uploads", data=data, files=files)
            label = "ECNs " + ", ".join(item["numbers"])
        if response.status_code >= 300:
            sys.exit(f"{label} failed: HTTP {response.status_code} {response.text[:800]}")
        results.append((label, response.json()))
    return results


def main(argv=None):
    """Run the publication and print a short per-file report."""
    # Same key file the API reads; parsed by python-dotenv, never shell-sourced.
    load_dotenv(ROOT / "api/.env")
    args = parse_args(argv)
    if not args.api_key:
        sys.exit("Provide --api-key or set API_KEY.")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    # Large workbooks are parsed and written to HANA synchronously by the API.
    with httpx.Client(base_url=args.api_url, headers={"X-API-Key": args.api_key}, timeout=900) as client:
        for label, body in publish(manifest, client, args):
            detail = body.get("dataset_id") or body.get("upload", {}).get("upload_id") or ""
            print(f"{label}: ok {detail}")


if __name__ == "__main__":
    main()
