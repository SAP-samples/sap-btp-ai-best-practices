"""Create HANA tables and load anonymized sample shipment documents.

Supports two data source modes:
  1. Anonymized scenario-based (default): reads from data/anonymized/ and maps
     documents to synthetic sales order IDs and outbound delivery IDs.
  2. Explicit source override: reads from another scenario-shaped folder when
     --source-dir is provided.

Examples:
    cd api
    python scripts/setup_hana_document_request_data.py

    python scripts/setup_hana_document_request_data.py --source-dir ../data/anonymized

    python scripts/setup_hana_document_request_data.py --source-dir ../data/anonymized --drop-existing
"""

from __future__ import annotations

import argparse
import mimetypes
import re
import sys
from pathlib import Path
from typing import Any

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.engine import Connection

from app.document_requests.db import create_hana_engine, hana_connection
from app.document_requests.scenario_emails import (
    ParsedScenarioEmail,
    normalize_line_identifier,
    parse_scenario_email,
)
from app.document_requests.settings import DocumentRequestSettings

TABLES = [
    "document_email_drafts",
    "document_request_extractions",
    "document_request_emails",
    "document_request_fetch_state",
    "shipment_documents",
    "sales_order_item_details",
    "sales_order_status_lines",
    "scenario_source_emails",
    "order_status",
]

SEED_DATA_TABLES = [
    "shipment_documents",
    "sales_order_item_details",
    "sales_order_status_lines",
    "scenario_source_emails",
    "order_status",
]

DDL_STATEMENTS = [
    (
        "shipment_documents",
        """
        CREATE COLUMN TABLE shipment_documents (
            document_id NVARCHAR(255) PRIMARY KEY,
            client_country NVARCHAR(10),
            sales_order_id NVARCHAR(40),
            outbound_delivery_id NVARCHAR(40),
            document_type NVARCHAR(80),
            file_name NVARCHAR(500),
            mime_type NVARCHAR(255),
            file_size_bytes INTEGER,
            raw_text NCLOB,
            file_content BLOB,
            source_path NVARCHAR(1000),
            is_active BOOLEAN,
            created_at TIMESTAMP,
            updated_at TIMESTAMP
        )
        """,
    ),
    (
        "scenario_source_emails",
        """
        CREATE COLUMN TABLE scenario_source_emails (
            scenario_email_id NVARCHAR(255) PRIMARY KEY,
            scenario_id NVARCHAR(80),
            source_path NVARCHAR(1000),
            subject NVARCHAR(1000),
            sender NVARCHAR(500),
            email_date TIMESTAMP,
            body_preview NCLOB,
            created_at TIMESTAMP,
            updated_at TIMESTAMP
        )
        """,
    ),
    (
        "sales_order_status_lines",
        """
        CREATE COLUMN TABLE sales_order_status_lines (
            status_line_id NVARCHAR(255) PRIMARY KEY,
            scenario_id NVARCHAR(80),
            scenario_email_id NVARCHAR(255),
            order_date NVARCHAR(40),
            order_time NVARCHAR(40),
            client_country NVARCHAR(10),
            route_id NVARCHAR(80),
            sales_order_id NVARCHAR(40),
            outbound_delivery_id NVARCHAR(40),
            document_type_code NVARCHAR(80),
            line_number NVARCHAR(40),
            product_id NVARCHAR(80),
            batch_number NVARCHAR(80),
            quantity_ordered NVARCHAR(80),
            quantity_packed NVARCHAR(80),
            dangerous_goods NVARCHAR(40),
            pick_date NVARCHAR(80),
            pack_box NVARCHAR(120),
            status_pick NVARCHAR(80),
            status_gi NVARCHAR(80),
            created_at TIMESTAMP,
            updated_at TIMESTAMP
        )
        """,
    ),
    (
        "sales_order_item_details",
        """
        CREATE COLUMN TABLE sales_order_item_details (
            item_detail_id NVARCHAR(255) PRIMARY KEY,
            scenario_id NVARCHAR(80),
            scenario_email_id NVARCHAR(255),
            sales_order_id NVARCHAR(40),
            customer_ref NVARCHAR(255),
            order_type NVARCHAR(80),
            item_number NVARCHAR(40),
            sold_to_name NVARCHAR(500),
            quantity NVARCHAR(80),
            uom NVARCHAR(40),
            product_id NVARCHAR(80),
            lot_number NVARCHAR(80),
            description NVARCHAR(1000),
            line_weight_kg NVARCHAR(80),
            invoice_id NVARCHAR(80),
            invoice_date NVARCHAR(80),
            foreign_amount NVARCHAR(80),
            commodity_code NVARCHAR(80),
            commodity_description NVARCHAR(500),
            country_of_origin NVARCHAR(80),
            expiry_date NVARCHAR(80),
            manufacturing_date NVARCHAR(80),
            created_at TIMESTAMP,
            updated_at TIMESTAMP
        )
        """,
    ),
    (
        "order_status",
        """
        CREATE COLUMN TABLE order_status (
            sales_order_id NVARCHAR(40) PRIMARY KEY,
            client_country NVARCHAR(10),
            picking_status NVARCHAR(40),
            packing_status NVARCHAR(40),
            pgi_status NVARCHAR(40),
            shipped BOOLEAN,
            shipped_date NVARCHAR(40),
            back_order BOOLEAN,
            expected_completion NVARCHAR(40)
        )
        """,
    ),
    (
        "document_request_fetch_state",
        """
        CREATE COLUMN TABLE document_request_fetch_state (
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
        "document_request_emails",
        """
        CREATE COLUMN TABLE document_request_emails (
            gmail_message_id NVARCHAR(255) PRIMARY KEY,
            thread_id NVARCHAR(255),
            rfc_message_id NVARCHAR(500),
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
        "document_request_extractions",
        """
        CREATE COLUMN TABLE document_request_extractions (
            extraction_id NVARCHAR(80) PRIMARY KEY,
            gmail_message_id NVARCHAR(255),
            model_name NVARCHAR(255),
            raw_model_output NCLOB,
            normalized_json NCLOB,
            is_document_request BOOLEAN,
            client_country NVARCHAR(10),
            sales_order_id NVARCHAR(40),
            reason NCLOB,
            confidence DECIMAL(6, 5),
            status NVARCHAR(40),
            validation_errors NCLOB,
            created_at TIMESTAMP
        )
        """,
    ),
    (
        "document_email_drafts",
        """
        CREATE COLUMN TABLE document_email_drafts (
            draft_id NVARCHAR(80) PRIMARY KEY,
            gmail_message_id NVARCHAR(255),
            extraction_id NVARCHAR(80),
            gmail_draft_id NVARCHAR(255),
            gmail_draft_message_id NVARCHAR(255),
            sent_message_id NVARCHAR(255),
            client_country NVARCHAR(10),
            to_email NVARCHAR(320),
            subject NVARCHAR(1000),
            body NCLOB,
            status NVARCHAR(40),
            selected_documents_json NCLOB,
            confidence DECIMAL(6, 5),
            explanation NCLOB,
            validation_errors_json NCLOB,
            raw_agent_output_json NCLOB,
            created_at TIMESTAMP,
            updated_at TIMESTAMP
        )
        """,
    ),
]

SCENARIO_MAPPING: dict[str, dict[str, str | None]] = {
    "scenario1": {
        "sales_order_id": "6090004185",
        "outbound_delivery_id": "90000785",
        "client_country": "DE",
    },
    "scenario2": {
        "sales_order_id": "6090004185",
        "outbound_delivery_id": "90000785",
        "client_country": "DE",
    },
    "scenario3": {
        "sales_order_id": "6090005272",
        "outbound_delivery_id": None,
        "client_country": "PL",
    },
}

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for HANA setup."""
    parser = argparse.ArgumentParser(description="Create HANA tables and load anonymized sample documents.")
    parser.add_argument("--source-dir", type=Path, default=Path("../data/anonymized"))
    parser.add_argument("--drop-existing", action="store_true")
    parser.add_argument(
        "--keep-existing-seed-data",
        action="store_true",
        help="Append/upsert seed rows without deleting existing scenario/anonymized seed rows first.",
    )
    return parser.parse_args()


def infer_document_type(path: Path) -> str:
    """Infer normalized document type from a file name."""
    name = path.name.lower()
    if "billing" in name or "invoice" in name:
        return "billing_document"
    if "deliverynote" in name or "delivery_note" in name:
        return "delivery_note"
    if "packinglist" in name or "packing_list" in name:
        return "packing_list"
    if "outbound" in name and "status" in name:
        return "outbound_delivery_status"
    if "broker" in name or "com code" in name:
        return "broker_file"
    if "certificate" in name or "declaration" in name or "sterility" in name:
        return "certificate"
    if "preview" in name or "transportation" in name:
        return "transportation_order"
    return "other"


def slugify(value: str) -> str:
    """Convert a path fragment into a stable lowercase id fragment."""
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def document_id_for_path(path: Path) -> str:
    """Build a stable document id from the path."""
    return f"doc-{slugify(str(path.with_suffix('')))}-{slugify(path.suffix.lstrip('.'))}"


def table_exists(connection: Connection, table_name: str) -> bool:
    """Return True when a table exists in the current HANA schema."""
    result = connection.execute(
        text(
            "SELECT table_name FROM sys.tables "
            "WHERE schema_name = CURRENT_SCHEMA AND table_name = :table_name"
        ),
        {"table_name": table_name.upper()},
    ).mappings().first()
    return result is not None


def drop_tables(connection: Connection) -> None:
    """Drop feature tables in dependency order."""
    for table_name in TABLES:
        if table_exists(connection, table_name):
            connection.execute(text(f"DROP TABLE {table_name}"))


def clear_seed_data(connection: Connection) -> None:
    """Delete existing seeded order/document rows while preserving workflow history.

    Args:
        connection: Active HANA connection.
    """
    for table_name in SEED_DATA_TABLES:
        connection.execute(text(f"DELETE FROM {table_name}"))


def create_tables(connection: Connection) -> None:
    """Create feature tables that do not already exist."""
    for table_name, statement in DDL_STATEMENTS:
        if not table_exists(connection, table_name):
            connection.execute(text(statement))
    ensure_order_status_columns(connection)


def column_exists(connection: Connection, table_name: str, column_name: str) -> bool:
    """Return True when a column exists on a HANA table.

    Args:
        connection: Active HANA connection.
        table_name: Table name to inspect.
        column_name: Column name to inspect.

    Returns:
        True when the column exists.
    """
    result = connection.execute(
        text(
            "SELECT column_name FROM sys.table_columns WHERE schema_name = CURRENT_SCHEMA "
            "AND table_name = :table_name AND column_name = :column_name"
        ),
        {"table_name": table_name.upper(), "column_name": column_name.upper()},
    ).mappings().first()
    return result is not None


def ensure_order_status_columns(connection: Connection) -> None:
    """Add new order_status columns when upgrading an existing HANA schema.

    Args:
        connection: Active HANA connection.
    """
    if table_exists(connection, "order_status") and not column_exists(connection, "order_status", "client_country"):
        connection.execute(text("ALTER TABLE order_status ADD (client_country NVARCHAR(10))"))


def get_scenario_from_path(path: Path) -> str | None:
    """Extract scenario name from a file path (e.g. 'scenario1')."""
    for part in path.parts:
        if part in SCENARIO_MAPPING:
            return part
    return None


def iter_scenario_files(source_dir: Path) -> list[Path]:
    """Return PDF files from scenario directories, deduplicating by sales order + filename.

    Scenario 2 reuses the same documents as Scenario 1 (same SO/OBD). Its behavior
    is controlled by agent logic (line item → packing list only), not separate DB rows.
    """
    paths = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".pdf"
        and not path.name.startswith("~$")
        and get_scenario_from_path(path) is not None
    )
    seen: set[tuple[str, str]] = set()
    deduplicated: list[Path] = []
    for path in paths:
        scenario = get_scenario_from_path(path)
        if not scenario:
            continue
        mapping = SCENARIO_MAPPING[scenario]
        key = (str(mapping["sales_order_id"]), path.name)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(path)
    return deduplicated


def iter_scenario_email_files(source_dir: Path) -> list[Path]:
    """Return scenario source email files ordered deterministically.

    Args:
        source_dir: Base scenario-shaped source directory.

    Returns:
        Sorted .eml files under scenario directories.
    """
    return sorted(
        path
        for path in source_dir.glob("scenario*/*.eml")
        if path.is_file() and not path.name.startswith("~$")
    )


def upsert_scenario_document(connection: Connection, path: Path, source_dir: Path) -> None:
    """Load one scenario document into HANA with sales order mapping."""
    scenario = get_scenario_from_path(path)
    if not scenario:
        return

    mapping = SCENARIO_MAPPING[scenario]
    content = path.read_bytes()
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    rel_path = path.relative_to(source_dir)

    connection.execute(
        text(
            "UPSERT shipment_documents (document_id, client_country, sales_order_id, outbound_delivery_id, "
            "document_type, file_name, mime_type, file_size_bytes, raw_text, file_content, source_path, "
            "is_active, created_at, updated_at) VALUES (:document_id, :client_country, :sales_order_id, "
            ":outbound_delivery_id, :document_type, :file_name, :mime_type, :file_size_bytes, :raw_text, "
            ":file_content, :source_path, TRUE, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) WITH PRIMARY KEY"
        ),
        {
            "document_id": document_id_for_path(rel_path),
            "client_country": mapping["client_country"],
            "sales_order_id": mapping["sales_order_id"],
            "outbound_delivery_id": mapping["outbound_delivery_id"],
            "document_type": infer_document_type(path),
            "file_name": path.name,
            "mime_type": mime_type,
            "file_size_bytes": len(content),
            "raw_text": path.name,
            "file_content": content,
            "source_path": str(rel_path),
        },
    )


def upsert_scenario_email_context(connection: Connection, parsed: ParsedScenarioEmail, source_dir: Path) -> None:
    """Store one parsed scenario email and its normalized tables in HANA.

    Args:
        connection: Active HANA connection.
        parsed: Parsed scenario email.
        source_dir: Base source directory used for relative source paths.
    """
    source = parsed.source_email
    relative_source = str(Path(source.source_path).relative_to(source_dir))
    connection.execute(
        text(
            "UPSERT scenario_source_emails (scenario_email_id, scenario_id, source_path, subject, sender, "
            "email_date, body_preview, created_at, updated_at) VALUES (:scenario_email_id, :scenario_id, "
            ":source_path, :subject, :sender, :email_date, :body_preview, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) "
            "WITH PRIMARY KEY"
        ),
        {
            "scenario_email_id": source.scenario_email_id,
            "scenario_id": source.scenario_id,
            "source_path": relative_source,
            "subject": source.subject,
            "sender": source.sender,
            "email_date": source.email_date,
            "body_preview": source.body_preview,
        },
    )
    for row in parsed.status_lines:
        payload = dict(row)
        payload["scenario_email_id"] = source.scenario_email_id
        connection.execute(
            text(
                "UPSERT sales_order_status_lines (status_line_id, scenario_id, scenario_email_id, order_date, "
                "order_time, client_country, route_id, sales_order_id, outbound_delivery_id, document_type_code, "
                "line_number, product_id, batch_number, quantity_ordered, quantity_packed, dangerous_goods, "
                "pick_date, pack_box, status_pick, status_gi, created_at, updated_at) VALUES "
                "(:status_line_id, :scenario_id, :scenario_email_id, :order_date, :order_time, :client_country, "
                ":route_id, :sales_order_id, :outbound_delivery_id, :document_type_code, :line_number, "
                ":product_id, :batch_number, :quantity_ordered, :quantity_packed, :dangerous_goods, "
                ":pick_date, :pack_box, :status_pick, :status_gi, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) "
                "WITH PRIMARY KEY"
            ),
            payload,
        )
    for row in parsed.item_details:
        payload = dict(row)
        payload["scenario_email_id"] = source.scenario_email_id
        connection.execute(
            text(
                "UPSERT sales_order_item_details (item_detail_id, scenario_id, scenario_email_id, sales_order_id, "
                "customer_ref, order_type, item_number, sold_to_name, quantity, uom, product_id, lot_number, "
                "description, line_weight_kg, invoice_id, invoice_date, foreign_amount, commodity_code, "
                "commodity_description, country_of_origin, expiry_date, manufacturing_date, created_at, updated_at) "
                "VALUES (:item_detail_id, :scenario_id, :scenario_email_id, :sales_order_id, :customer_ref, "
                ":order_type, :item_number, :sold_to_name, :quantity, :uom, :product_id, :lot_number, "
                ":description, :line_weight_kg, :invoice_id, :invoice_date, :foreign_amount, :commodity_code, "
                ":commodity_description, :country_of_origin, :expiry_date, :manufacturing_date, "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) WITH PRIMARY KEY"
            ),
            payload,
        )


def seed_order_statuses_from_lines(connection: Connection, parsed_emails: list[ParsedScenarioEmail]) -> None:
    """Derive order header status rows from parsed scenario line data.

    Args:
        connection: Active HANA connection.
        parsed_emails: Parsed scenario emails.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for parsed in parsed_emails:
        for row in parsed.status_lines:
            sales_order_id = row.get("sales_order_id")
            if sales_order_id:
                grouped.setdefault(str(sales_order_id), []).append(row)
    for sales_order_id, rows in grouped.items():
        status = derive_order_status(sales_order_id, rows)
        connection.execute(
            text(
                "UPSERT order_status (sales_order_id, client_country, picking_status, packing_status, pgi_status, "
                "shipped, shipped_date, back_order, expected_completion) VALUES (:sales_order_id, "
                ":client_country, :picking_status, :packing_status, :pgi_status, :shipped, :shipped_date, "
                ":back_order, :expected_completion) WITH PRIMARY KEY"
            ),
            status,
        )


def derive_order_status(sales_order_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate line-level HANA data into an order-status row.

    Args:
        sales_order_id: Sales order id.
        rows: Normalized status-line rows for the order.

    Returns:
        Order status row ready for HANA upsert.
    """
    pick_values = [row.get("status_pick") for row in rows]
    gi_values = [row.get("status_gi") for row in rows]
    packed_values = [row.get("quantity_packed") for row in rows]
    gi_present = [value for value in gi_values if value is not None]
    pgi_completed = bool(gi_present) and all(is_completed_status(value) for value in gi_present)
    picking_status = aggregate_completion_status(pick_values)
    packing_status = aggregate_packed_status(packed_values)
    shipped_date = latest_date_prefix([row.get("pick_date") for row in rows]) if pgi_completed else None
    return {
        "sales_order_id": sales_order_id,
        "client_country": first_non_empty([row.get("client_country") for row in rows]),
        "picking_status": picking_status,
        "packing_status": packing_status,
        "pgi_status": "completed" if pgi_completed else "not_started",
        "shipped": pgi_completed,
        "shipped_date": shipped_date,
        "back_order": any(not is_completed_status(row.get("status_gi")) or not has_packed_quantity(row.get("quantity_packed")) for row in rows),
        "expected_completion": None,
    }


def is_completed_status(value: Any) -> bool:
    """Return True for status values representing completion.

    Args:
        value: Raw status code or label.

    Returns:
        True for completed status values.
    """
    return str(value or "").strip().upper() in {"9", "COMPLETED", "COMPLETE", "PICKED"}


def has_packed_quantity(value: Any) -> bool:
    """Return True when a quantity-packed field indicates packed units.

    Args:
        value: Raw quantity packed value.

    Returns:
        True when packed quantity is greater than zero.
    """
    raw = str(value or "").strip().replace(",", ".")
    if raw in {"", "-"}:
        return False
    try:
        return float(raw) > 0
    except ValueError:
        return False


def aggregate_completion_status(values: list[Any]) -> str:
    """Aggregate line completion values into a header status.

    Args:
        values: Raw per-line status values.

    Returns:
        completed, partially_completed, or not_started.
    """
    completed = [is_completed_status(value) for value in values]
    if completed and all(completed):
        return "completed"
    if any(completed):
        return "partially_completed"
    return "not_started"


def aggregate_packed_status(values: list[Any]) -> str:
    """Aggregate packed quantity values into a header packing status.

    Args:
        values: Raw packed quantity values.

    Returns:
        completed, partially_completed, or not_started.
    """
    packed = [has_packed_quantity(value) for value in values]
    if packed and all(packed):
        return "completed"
    if any(packed):
        return "partially_completed"
    return "not_started"


def first_non_empty(values: list[Any]) -> Any:
    """Return the first non-empty value.

    Args:
        values: Candidate values.

    Returns:
        First meaningful value, or None.
    """
    for value in values:
        if value not in {None, ""}:
            return value
    return None


def latest_date_prefix(values: list[Any]) -> str | None:
    """Return the latest date prefix from DD/MM/YYYY datetime strings.

    Args:
        values: Raw date or datetime strings.

    Returns:
        ISO date string when parseable, otherwise the first raw date prefix.
    """
    date_prefixes = [str(value).split()[0] for value in values if value]
    parsed: list[tuple[tuple[int, int, int], str]] = []
    for value in date_prefixes:
        parts = value.split("/")
        if len(parts) == 3 and all(part.isdigit() for part in parts):
            day, month, year = [int(part) for part in parts]
            parsed.append(((year, month, day), f"{year:04d}-{month:02d}-{day:02d}"))
    if parsed:
        return sorted(parsed)[-1][1]
    return date_prefixes[0] if date_prefixes else None


def main() -> None:
    """Run HANA table setup and scenario document loading."""
    args = parse_args()
    load_dotenv()
    settings = DocumentRequestSettings.from_env()
    engine = create_hana_engine(settings)
    with hana_connection(engine) as connection:
        if args.drop_existing:
            drop_tables(connection)
        create_tables(connection)
        if not args.keep_existing_seed_data:
            print("Clearing existing seeded order/document rows")
            clear_seed_data(connection)

        files = iter_scenario_files(args.source_dir)
        print(f"Found {len(files)} scenario document(s) to load")
        for path in files:
            scenario = get_scenario_from_path(path)
            print(f"  [{scenario}] {path.name}")
            upsert_scenario_document(connection, path, args.source_dir)

        email_files = iter_scenario_email_files(args.source_dir)
        parsed_emails = [parse_scenario_email(path) for path in email_files]
        print(f"Found {len(parsed_emails)} scenario source email(s) to load")
        for parsed in parsed_emails:
            print(
                f"  [{parsed.scenario_id}] {Path(parsed.source_email.source_path).name} "
                f"({len(parsed.status_lines)} status row(s), {len(parsed.item_details)} item detail row(s))"
            )
            upsert_scenario_email_context(connection, parsed, args.source_dir)

        seed_order_statuses_from_lines(connection, parsed_emails)
        print(f"Derived {len({row['sales_order_id'] for parsed in parsed_emails for row in parsed.status_lines})} order status row(s)")


if __name__ == "__main__":
    main()
