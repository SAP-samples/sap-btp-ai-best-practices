#!/usr/bin/env python3
"""
Examples:
  python api/scripts/load_hana_seed_data.py \
    --customer-workbook /secure/path/customer_seed.xlsx \
    --program-codes-workbook /secure/path/program_seed.xlsx
  COA_CUSTOMER_SEED_WORKBOOK=/secure/path/customer_seed.xlsx \
    COA_PROGRAM_CODES_SEED_WORKBOOK=/secure/path/program_seed.xlsx \
    python api/scripts/load_hana_seed_data.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
API_ROOT = REPO_ROOT / "api"

for candidate in (str(REPO_ROOT), str(API_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

load_dotenv(API_ROOT / ".env")

from app.nbo.hana import HanaSettings, connect  # noqa: E402
from app.nbo.hana_loader import load_seed_datasets, recreate_and_load_tables  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """Parse explicit local seed workbook paths for a HANA load."""
    parser = argparse.ArgumentParser(
        description="Load explicit local seed workbooks into Customer Offer Advisor HANA tables."
    )
    parser.add_argument(
        "--customer-workbook",
        default=os.getenv("COA_CUSTOMER_SEED_WORKBOOK"),
        help="Path to the customer, segment, and active-offering workbook.",
    )
    parser.add_argument(
        "--program-codes-workbook",
        default=os.getenv("COA_PROGRAM_CODES_SEED_WORKBOOK"),
        help="Path to the program contract and sample account workbook.",
    )
    return parser.parse_args()


def _required_path(raw_path: str | None, label: str) -> Path:
    """Return an existing seed workbook path or raise a clear CLI error."""
    if not raw_path:
        raise SystemExit(
            f"{label} is required. Pass the CLI flag or set the matching COA_* env var."
        )
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"{label} does not exist: {path}")
    return path


def main() -> int:
    """Load the explicit seed workbooks into the configured HANA tables."""
    args = _parse_args()
    customer_workbook = _required_path(args.customer_workbook, "--customer-workbook")
    program_codes_workbook = _required_path(
        args.program_codes_workbook,
        "--program-codes-workbook",
    )
    settings = HanaSettings.from_env()
    datasets = load_seed_datasets(
        customer_workbook=customer_workbook,
        program_codes_workbook=program_codes_workbook,
    )

    connection = connect(settings)
    try:
        row_counts = recreate_and_load_tables(connection, datasets)
    finally:
        connection.close()

    for table_name, row_count in row_counts.items():
        print(f"{table_name}: loaded {row_count} rows")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
