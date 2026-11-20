"""Load mock ECC master data from a JSON seed file into the *_POAGENT HANA tables.

Examples (run from the api/ folder, credentials in api/.env):

    ../.venv/bin/python -m app.cli.seed_master_data
    ../.venv/bin/python -m app.cli.seed_master_data --file ../data/seed/master_data.json
    ../.venv/bin/python -m app.cli.seed_master_data --dry-run
    ../.venv/bin/python -m app.cli.seed_master_data --tables KNA1_POAGENT KNVP_POAGENT
    ../.venv/bin/python -m app.cli.seed_master_data --reset-transactions

Seed file format: ``{"_meta": {...}, "<TABLE>": [ {"<COLUMN>": value, ...}, ... ], ...}``.
Only master-data tables can be seeded (customer, material, unit tables); sales documents
(VBAK/VBKD/VBAP/VBPA) are only written by posting. Each seeded table is emptied and
reloaded, so the command is idempotent. Customer and material numbers are converted to
SAP ALPHA format (zero-padded) automatically.

``--reset-transactions`` additionally empties all POAGENT_* app tables and the mock sales
documents, returning the demo to a clean state (master data is kept).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text
from tqdm import tqdm

from app.db.engine import get_engine
from app.db.schema import APP_TABLES, ECC_TABLES, ensure_schema
from app.orders.matching import alpha_in

DEFAULT_FILE = Path(__file__).resolve().parents[3] / "data" / "seed" / "master_data.json"
SALES_DOCUMENT_TABLES = ("VBAK_POAGENT", "VBKD_POAGENT", "VBAP_POAGENT", "VBPA_POAGENT")
SEEDABLE = [name for name in ECC_TABLES if name not in SALES_DOCUMENT_TABLES]
ALPHA_LENGTH = {"KUNNR": 10, "KUNN2": 10, "MATNR": 18}


def validate_seed(seed: dict, tables: list[str]) -> list[str]:
    """Check table and column names of the seed against the schema catalogue.

    Args:
        seed: Parsed seed JSON.
        tables: Tables selected for loading.

    Returns:
        List of human-readable problems (empty when the seed is valid).
    """
    problems = []
    for table in tables:
        if table not in seed:
            problems.append(f"{table}: missing in seed file")
            continue
        max_len = {col: int(m.group(1)) for col, sql_type in ECC_TABLES[table].columns
                   if (m := re.match(r"NVARCHAR\((\d+)\)", sql_type))}
        for i, row in enumerate(seed[table]):
            unknown = set(row) - set(max_len) - set(ECC_TABLES[table].column_names)
            if unknown:
                problems.append(f"{table}[{i}]: unknown columns {sorted(unknown)}")
            for col, value in normalise_row(row).items():
                if value is not None and col in max_len and len(str(value)) > max_len[col]:
                    problems.append(f"{table}[{i}].{col}: {value!r} exceeds {max_len[col]} characters")
    return problems


def normalise_row(row: dict) -> dict:
    """Apply ALPHA conversion to customer/material number columns of one row."""
    return {key: alpha_in(value, ALPHA_LENGTH[key]) if key in ALPHA_LENGTH else value for key, value in row.items()}


def load_table(conn, table: str, rows: list[dict]) -> None:
    """Empty a table and insert the given rows (one executemany per table).

    Args:
        conn: Open SQLAlchemy connection inside a transaction.
        table: Target table.
        rows: Normalised rows; missing columns are inserted as NULL.
    """
    conn.execute(text(f'DELETE FROM "{table}"'))
    if not rows:
        return
    columns = ECC_TABLES[table].column_names
    sql = f'INSERT INTO "{table}" ({", ".join(chr(34) + c + chr(34) for c in columns)}) ' \
          f'VALUES ({", ".join(":" + c for c in columns)})'
    conn.execute(text(sql), [{c: row.get(c) for c in columns} for row in rows])


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Command-line arguments (defaults to sys.argv).

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE, help="seed JSON file")
    parser.add_argument("--tables", nargs="+", choices=SEEDABLE, default=SEEDABLE, help="subset of tables to load")
    parser.add_argument("--dry-run", action="store_true", help="validate the seed file only")
    parser.add_argument("--reset-transactions", action="store_true",
                        help="also empty POAGENT_* app tables and mock sales documents")
    args = parser.parse_args(argv)

    seed = json.loads(args.file.read_text(encoding="utf-8"))
    problems = validate_seed(seed, args.tables)
    if problems:
        print("Seed file is invalid:", *problems, sep="\n  ")
        return 1
    counts = {table: len(seed[table]) for table in args.tables}
    print(f"Seed file {args.file} is valid: {sum(counts.values())} rows in {len(counts)} tables")
    if args.dry_run:
        return 0

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    engine = get_engine()
    ensure_schema(engine)
    with engine.begin() as conn:
        for table in tqdm(args.tables, desc="Loading master data", unit="table"):
            load_table(conn, table, [normalise_row(row) for row in seed[table]])
        if args.reset_transactions:
            for table in tqdm([*SALES_DOCUMENT_TABLES, *APP_TABLES], desc="Clearing transactions", unit="table"):
                conn.execute(text(f'DELETE FROM "{table}"'))
    for table in args.tables:
        print(f"  {table:<22} {counts[table]:>5} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
