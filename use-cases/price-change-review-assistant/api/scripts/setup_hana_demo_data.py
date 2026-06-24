"""Create and load HANA demo tables for the Gmail price-change PoC.

Examples:
    cd /Users/I760054/Documents/programs/Decathlon-Price-Change/api
    python scripts/setup_hana_demo_data.py --seed-dir seed_data

    cd /Users/I760054/Documents/programs/Decathlon-Price-Change/api
    python scripts/setup_hana_demo_data.py --seed-dir ../data --drop-existing
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.engine import Connection

from app.price_changes.db import create_hana_engine, hana_connection
from app.price_changes.schema import EMAIL_ATTACHMENTS_DDL, PROCESSING_EVENTS_DDL, PROCESSING_RUNS_DDL
from app.price_changes.settings import PriceChangeSettings


TABLES = [
    "price_change_processing_events",
    "price_change_processing_runs",
    "price_change_drafts",
    "email_extractions",
    "email_attachments",
    "gmail_emails",
    "gmail_fetch_state",
    "supplier_material_prices",
    "materials",
    "suppliers",
]


DDL_STATEMENTS = [
    (
        "suppliers",
        """
    CREATE COLUMN TABLE suppliers (
        supplier_id NVARCHAR(40) PRIMARY KEY,
        contact_name NVARCHAR(255),
        email NVARCHAR(320),
        company NVARCHAR(255),
        business_partner NVARCHAR(40),
        supplier NVARCHAR(40),
        purchasing_organization NVARCHAR(40),
        purchasing_group NVARCHAR(40),
        currency NVARCHAR(10)
    )
    """,
    ),
    (
        "materials",
        """
    CREATE COLUMN TABLE materials (
        material_code NVARCHAR(80) PRIMARY KEY,
        material_description NVARCHAR(1000),
        supplier_id NVARCHAR(40),
        current_price DECIMAL(15, 3),
        currency NVARCHAR(10),
        uom NVARCHAR(20)
    )
    """,
    ),
    (
        "supplier_material_prices",
        """
    CREATE COLUMN TABLE supplier_material_prices (
        supplier_id NVARCHAR(40),
        material_code NVARCHAR(80),
        current_price DECIMAL(15, 3),
        currency NVARCHAR(10),
        uom NVARCHAR(20),
        updated_at TIMESTAMP,
        PRIMARY KEY (supplier_id, material_code)
    )
    """,
    ),
    (
        "gmail_fetch_state",
        """
    CREATE COLUMN TABLE gmail_fetch_state (
        mailbox_id NVARCHAR(255) PRIMARY KEY,
        last_successful_fetch_at TIMESTAMP,
        last_started_at TIMESTAMP,
        last_finished_at TIMESTAMP,
        last_status NVARCHAR(40),
        summary_json NCLOB,
        error_message NCLOB
    )
    """,
    ),
    (
        "gmail_emails",
        """
    CREATE COLUMN TABLE gmail_emails (
        gmail_message_id NVARCHAR(255) PRIMARY KEY,
        thread_id NVARCHAR(255),
        sender_name NVARCHAR(255),
        sender_email NVARCHAR(320),
        subject NVARCHAR(1000),
        email_date TIMESTAMP,
        body NCLOB,
        fetched_at TIMESTAMP,
        processed_at TIMESTAMP,
        processing_status NVARCHAR(40),
        error_message NCLOB
    )
    """,
    ),
    (
        "email_attachments",
        EMAIL_ATTACHMENTS_DDL,
    ),
    (
        "price_change_processing_runs",
        PROCESSING_RUNS_DDL,
    ),
    (
        "price_change_processing_events",
        PROCESSING_EVENTS_DDL,
    ),
    (
        "email_extractions",
        """
    CREATE COLUMN TABLE email_extractions (
        extraction_id NVARCHAR(80) PRIMARY KEY,
        gmail_message_id NVARCHAR(255),
        model_name NVARCHAR(255),
        raw_model_output NCLOB,
        normalized_json NCLOB,
        is_price_request BOOLEAN,
        reason NCLOB,
        confidence DECIMAL(6, 5),
        status NVARCHAR(40),
        validation_errors NCLOB,
        created_at TIMESTAMP
    )
    """,
    ),
    (
        "price_change_drafts",
        """
    CREATE COLUMN TABLE price_change_drafts (
        draft_id NVARCHAR(80) PRIMARY KEY,
        gmail_message_id NVARCHAR(255),
        extraction_id NVARCHAR(80),
        item_index INTEGER,
        supplier_id NVARCHAR(40),
        supplier_name NVARCHAR(255),
        supplier_email NVARCHAR(320),
        material_code NVARCHAR(80),
        material_description NVARCHAR(1000),
        original_price DECIMAL(15, 3),
        requested_new_price DECIMAL(15, 3),
        currency NVARCHAR(10),
        uom NVARCHAR(20),
        price_change_mode NVARCHAR(40),
        price_change_value NVARCHAR(80),
        effective_from NVARCHAR(40),
        effective_to NVARCHAR(40),
        confidence DECIMAL(6, 5),
        status NVARCHAR(40),
        explanation NCLOB,
        candidate_materials_json NCLOB,
        candidate_suppliers_json NCLOB,
        validation_errors_json NCLOB,
        raw_agent_output_json NCLOB,
        created_at TIMESTAMP,
        updated_at TIMESTAMP
    )
    """,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create HANA tables and load demo supplier/material data.")
    parser.add_argument("--seed-dir", type=Path, default=Path("seed_data"))
    parser.add_argument("--drop-existing", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def table_exists(connection: Connection, table_name: str) -> bool:
    result = connection.execute(
        text(
            """
            SELECT table_name
            FROM sys.tables
            WHERE schema_name = CURRENT_SCHEMA
              AND table_name = :table_name
            """
        ),
        {"table_name": table_name.upper()},
    )
    return result.mappings().first() is not None


def drop_tables(connection: Connection) -> None:
    for table_name in TABLES:
        if table_exists(connection, table_name):
            connection.execute(text(f"DROP TABLE {table_name}"))


def create_tables(connection: Connection) -> None:
    for table_name, statement in DDL_STATEMENTS:
        if not table_exists(connection, table_name):
            connection.execute(text(statement))


def upsert_suppliers(connection: Connection, rows: Iterable[dict[str, str]]) -> None:
    statement = text(
        "UPSERT suppliers (supplier_id, contact_name, email, company, business_partner, supplier, "
        "purchasing_organization, purchasing_group, currency) VALUES (:supplier_id, :contact_name, :email, "
        ":company, :business_partner, :supplier, :purchasing_organization, :purchasing_group, :currency) "
        "WITH PRIMARY KEY"
    )
    for row in rows:
        params = {
            "supplier_id": row.get("supplier_id"),
            "contact_name": row.get("contact_name"),
            "email": row.get("email"),
            "company": row.get("company"),
            "business_partner": row.get("business_partner"),
            "supplier": row.get("supplier"),
            "purchasing_organization": row.get("purchasing_organization"),
            "purchasing_group": row.get("purchasing_group"),
            "currency": row.get("currency"),
        }
        connection.execute(statement, params)


def material_params(row: dict[str, str]) -> dict[str, Any]:
    return {
        "material_code": row.get("material_code"),
        "material_description": row.get("material_description"),
        "supplier_id": row.get("supplier_id"),
        "current_price": row.get("current_price"),
        "currency": row.get("currency"),
        "uom": row.get("uom"),
    }


def upsert_materials(connection: Connection, rows: Iterable[dict[str, str]]) -> None:
    material_statement = text(
        "UPSERT materials (material_code, material_description, supplier_id, current_price, currency, uom) "
        "VALUES (:material_code, :material_description, :supplier_id, :current_price, :currency, :uom) "
        "WITH PRIMARY KEY"
    )
    price_statement = text(
        "UPSERT supplier_material_prices (supplier_id, material_code, current_price, currency, uom, updated_at) "
        "VALUES (:supplier_id, :material_code, :current_price, :currency, :uom, CURRENT_TIMESTAMP) "
        "WITH PRIMARY KEY"
    )
    for row in rows:
        params = material_params(row)
        connection.execute(material_statement, params)
        connection.execute(price_statement, params)


def main() -> int:
    args = parse_args()
    load_dotenv()
    settings = PriceChangeSettings.from_env()
    engine = create_hana_engine(settings)
    suppliers = read_csv(args.seed_dir / "suppliers.csv")
    materials = read_csv(args.seed_dir / "materials.csv")
    with hana_connection(engine) as connection:
        if args.drop_existing:
            drop_tables(connection)
        create_tables(connection)
        upsert_suppliers(connection, suppliers)
        upsert_materials(connection, materials)
    print(f"Loaded {len(suppliers)} suppliers and {len(materials)} materials into HANA.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
