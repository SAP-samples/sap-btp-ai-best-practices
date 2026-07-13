"""Refresh the structured HTS catalog tables in SAP HANA from chapter CSV exports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.services.metal_composition.config import get_settings
from app.services.metal_composition.hts_catalog import refresh_hts_catalog_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, default=None)
    parser.add_argument("--code-map-path", type=Path, default=None)
    parser.add_argument("--hana-schema", type=str, default=None)
    parser.add_argument("--catalog-table", type=str, default=None)
    parser.add_argument("--code-map-table", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = refresh_hts_catalog_tables(
        settings=get_settings(),
        csv_dir=args.csv_dir,
        code_map_path=args.code_map_path,
        hana_schema=args.hana_schema,
        catalog_table=args.catalog_table,
        code_map_table=args.code_map_table,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
