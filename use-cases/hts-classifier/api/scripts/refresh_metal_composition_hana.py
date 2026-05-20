"""Refresh the denormalized metal-composition serving table in SAP HANA.

Examples:
    ./.venv/bin/python api/scripts/refresh_metal_composition_hana.py
    ./.venv/bin/python api/scripts/refresh_metal_composition_hana.py --workbook-path "data/new_data/GCC Tracker.xlsb"
    ./.venv/bin/python api/scripts/refresh_metal_composition_hana.py --workbook-path "outputs/anonymized_gcc_tracker_sample.xlsx"
    ./.venv/bin/python api/scripts/refresh_metal_composition_hana.py --hana-table METAL_COMPOSITION_SERVING
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path


API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.services.metal_composition.config import MetalCompositionSettings, get_settings
from app.services.metal_composition.hana_refresh import refresh_metal_composition_hana


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for a GCC Tracker HANA refresh.

    Returns:
        Parsed namespace containing optional workbook, sheet, schema, and table overrides.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook-path", type=Path, default=None)
    parser.add_argument("--sheet-name", type=str, default=None)
    parser.add_argument("--hana-schema", type=str, default=None)
    parser.add_argument("--hana-table", type=str, default=None)
    return parser.parse_args()


def build_settings(args: argparse.Namespace) -> MetalCompositionSettings:
    """Build refresh settings from environment configuration and CLI overrides.

    Args:
        args: Parsed CLI arguments from ``parse_args``.

    Returns:
        Metal composition settings configured for a HANA refresh run.
    """

    settings = get_settings()
    return replace(
        settings,
        workbook_path=(args.workbook_path or settings.workbook_path).resolve(),
        data_source="hana",
        hana_schema=(args.hana_schema if args.hana_schema is not None else settings.hana_schema).strip(),
        hana_table=(args.hana_table if args.hana_table is not None else settings.hana_table).strip(),
        sheet_name=args.sheet_name or settings.sheet_name,
    )


def main() -> None:
    """Run the GCC Tracker HANA refresh command and print the JSON result.

    Returns:
        None. The refresh result is written to standard output.
    """

    args = parse_args()
    settings = build_settings(args)
    result = refresh_metal_composition_hana(settings.workbook_path, settings=settings)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
