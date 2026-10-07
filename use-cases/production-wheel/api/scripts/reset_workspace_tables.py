"""Drop every Production Wheel table in the HANA user's default schema.

Use this to redeploy the workspace storage from scratch, for example after a
column-contract change in ``app/workspace/columns.json``. The application
recreates all tables with the current contract on its next start
(``WorkspaceRepository.ensure``, the jobs store and agent memory bootstrap), so
after a reset only the datasets have to be uploaded and published again.

Credentials come from ``api/.env`` (HANA_ADDRESS, HANA_PORT, HANA_USER,
HANA_PASSWORD, HANA_ENCRYPT), exactly as for the running API.

Examples::

    # Dry run: list the tables that would be dropped, change nothing
    .venv/bin/python api/scripts/reset_workspace_tables.py

    # Drop them (irreversible: all datasets, runs, jobs and agent memory are lost)
    .venv/bin/python api/scripts/reset_workspace_tables.py --confirm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

# Make the ``app`` package importable when the script is run from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.workspace.repository import PREFIX, connect_hana  # noqa: E402


def list_tables(cursor) -> list[str]:
    """Return the application's tables in the connected user's default schema.

    Args:
        cursor: Open hdbcli cursor.

    Returns:
        Sorted table names starting with the workspace prefix (``PRODUCTION_WHEEL_``).
    """

    cursor.execute(
        "SELECT TABLE_NAME FROM SYS.TABLES WHERE SCHEMA_NAME=CURRENT_SCHEMA "
        "AND TABLE_NAME LIKE ? ESCAPE '\\' ORDER BY TABLE_NAME",
        (PREFIX.replace("_", "\\_") + "%",),
    )
    return [row[0] for row in cursor.fetchall()]


def main() -> int:
    """List, and with ``--confirm`` drop, all workspace tables.

    Returns:
        Process exit code (0 on success).
    """

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--confirm", action="store_true", help="actually drop the listed tables"
    )
    arguments = parser.parse_args()

    connection = connect_hana()
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT CURRENT_SCHEMA FROM DUMMY")
        schema = cursor.fetchone()[0]
        tables = list_tables(cursor)
        print(f"Schema {schema}: {len(tables)} table(s) with prefix {PREFIX}")
        for name in tables:
            print(f"  {name}")
        if not arguments.confirm:
            print("Dry run only. Re-run with --confirm to drop these tables.")
            return 0
        for name in tqdm(tables, desc="Dropping tables", unit="table"):
            # Names come from the catalog and match the fixed prefix; quote them anyway.
            cursor.execute(f'DROP TABLE "{name}"')
        connection.commit()
        print(f"Dropped {len(tables)} table(s). Restart the API to recreate them.")
        return 0
    finally:
        cursor.close()
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
