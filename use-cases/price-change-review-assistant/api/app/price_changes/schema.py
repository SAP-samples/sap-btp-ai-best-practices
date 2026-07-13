from __future__ import annotations

from threading import Lock

from sqlalchemy import text
from sqlalchemy.engine import Connection


EMAIL_ATTACHMENTS_TABLE = "email_attachments"
EMAIL_ATTACHMENTS_TABLE_UPPER = EMAIL_ATTACHMENTS_TABLE.upper()
EMAIL_ATTACHMENTS_DDL = """
    CREATE COLUMN TABLE email_attachments (
        attachment_id NVARCHAR(80) PRIMARY KEY,
        gmail_message_id NVARCHAR(255),
        source NVARCHAR(40),
        provider_attachment_id NVARCHAR(255),
        filename NVARCHAR(1000),
        mime_type NVARCHAR(255),
        file_extension NVARCHAR(20),
        size_bytes INTEGER,
        content_base64 NCLOB,
        text_preview NCLOB,
        created_at TIMESTAMP
    )
    """
EMAIL_ATTACHMENTS_REQUIRED_COLUMNS = {
    "ATTACHMENT_ID": "NVARCHAR",
    "GMAIL_MESSAGE_ID": "NVARCHAR",
    "SOURCE": "NVARCHAR",
    "PROVIDER_ATTACHMENT_ID": "NVARCHAR",
    "FILENAME": "NVARCHAR",
    "MIME_TYPE": "NVARCHAR",
    "FILE_EXTENSION": "NVARCHAR",
    "SIZE_BYTES": "INTEGER",
    "CONTENT_BASE64": "NCLOB",
    "TEXT_PREVIEW": "NCLOB",
    "CREATED_AT": "TIMESTAMP",
}
PROCESSING_RUNS_TABLE = "price_change_processing_runs"
PROCESSING_EVENTS_TABLE = "price_change_processing_events"
PROCESSING_RUNS_DDL = f"""
    CREATE COLUMN TABLE {PROCESSING_RUNS_TABLE} (
        processing_run_id NVARCHAR(120) PRIMARY KEY,
        source_type NVARCHAR(40),
        status NVARCHAR(40),
        started_at TIMESTAMP,
        finished_at TIMESTAMP,
        last_heartbeat_at TIMESTAMP,
        current_stage NVARCHAR(80),
        current_message NVARCHAR(1000),
        error_message NCLOB
    )
    """
PROCESSING_EVENTS_DDL = f"""
    CREATE COLUMN TABLE {PROCESSING_EVENTS_TABLE} (
        event_id NVARCHAR(80) PRIMARY KEY,
        processing_run_id NVARCHAR(120),
        sequence_number INTEGER,
        event_time TIMESTAMP,
        level NVARCHAR(20),
        stage NVARCHAR(80),
        message NVARCHAR(1000),
        metadata_json NCLOB
    )
    """
PROCESSING_RUNS_REQUIRED_COLUMNS = {
    "PROCESSING_RUN_ID": "NVARCHAR",
    "SOURCE_TYPE": "NVARCHAR",
    "STATUS": "NVARCHAR",
    "STARTED_AT": "TIMESTAMP",
    "FINISHED_AT": "TIMESTAMP",
    "LAST_HEARTBEAT_AT": "TIMESTAMP",
    "CURRENT_STAGE": "NVARCHAR",
    "CURRENT_MESSAGE": "NVARCHAR",
    "ERROR_MESSAGE": "NCLOB",
}
PROCESSING_EVENTS_REQUIRED_COLUMNS = {
    "EVENT_ID": "NVARCHAR",
    "PROCESSING_RUN_ID": "NVARCHAR",
    "SEQUENCE_NUMBER": "INTEGER",
    "EVENT_TIME": "TIMESTAMP",
    "LEVEL": "NVARCHAR",
    "STAGE": "NVARCHAR",
    "MESSAGE": "NVARCHAR",
    "METADATA_JSON": "NCLOB",
}

_email_attachments_schema_ready = False
_email_attachments_schema_lock = Lock()
_processing_progress_schema_ready = False
_processing_progress_schema_lock = Lock()


def reset_email_attachments_schema_cache() -> None:
    """Reset the process-local attachment schema readiness cache.

    Returns:
        None. The next schema guard call will query HANA again.
    """
    global _email_attachments_schema_ready
    with _email_attachments_schema_lock:
        _email_attachments_schema_ready = False


def reset_processing_progress_schema_cache() -> None:
    """Reset the process-local processing progress schema readiness cache.

    Returns:
        None. The next processing progress schema guard call will query HANA.
    """
    global _processing_progress_schema_ready
    with _processing_progress_schema_lock:
        _processing_progress_schema_ready = False


def table_exists(connection: Connection, table_name: str) -> bool:
    """Return whether a table exists in the current HANA schema.

    Args:
        connection: Active SQLAlchemy connection to SAP HANA.
        table_name: Unquoted table name to check.

    Returns:
        True when the current schema contains the table; otherwise False.
    """
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


def table_columns(connection: Connection, table_name: str) -> dict[str, str]:
    """Return column data types for a table in the current HANA schema.

    Args:
        connection: Active SQLAlchemy connection to SAP HANA.
        table_name: Unquoted table name to inspect.

    Returns:
        Mapping of uppercase column names to uppercase HANA data type names.
    """
    result = connection.execute(
        text(
            """
            SELECT column_name, data_type_name
            FROM sys.table_columns
            WHERE schema_name = CURRENT_SCHEMA
              AND table_name = :table_name
            """
        ),
        {"table_name": table_name.upper()},
    )
    return {
        str(row["column_name"]).upper(): str(row["data_type_name"]).upper()
        for row in result.mappings().all()
    }


def validate_email_attachments_table(connection: Connection) -> None:
    """Validate the existing attachment table has the required column contract.

    Args:
        connection: Active SQLAlchemy connection to SAP HANA.

    Returns:
        None when the table contract is compatible.

    Raises:
        RuntimeError: If the table exists but is missing required columns or data types.
    """
    columns = table_columns(connection, EMAIL_ATTACHMENTS_TABLE)
    missing = [
        column_name
        for column_name in EMAIL_ATTACHMENTS_REQUIRED_COLUMNS
        if column_name not in columns
    ]
    incompatible = [
        f"{column_name} expected {expected_type}, found {columns.get(column_name)}"
        for column_name, expected_type in EMAIL_ATTACHMENTS_REQUIRED_COLUMNS.items()
        if column_name in columns and columns[column_name] != expected_type
    ]
    if missing or incompatible:
        details = []
        if missing:
            details.append(f"missing columns: {', '.join(missing)}")
        if incompatible:
            details.append(f"incompatible columns: {', '.join(incompatible)}")
        raise RuntimeError(
            f"HANA table {EMAIL_ATTACHMENTS_TABLE_UPPER} exists with an incompatible schema: "
            + "; ".join(details)
        )


def validate_table_contract(
    connection: Connection,
    table_name: str,
    required_columns: dict[str, str],
) -> None:
    """Validate that a HANA table contains the expected columns and data types.

    Args:
        connection: Active SQLAlchemy connection to SAP HANA.
        table_name: Table to validate in the current schema.
        required_columns: Mapping of uppercase column names to expected HANA types.

    Returns:
        None when the table contract is compatible.

    Raises:
        RuntimeError: If the table exists with an incompatible schema.
    """
    columns = table_columns(connection, table_name)
    missing = [column_name for column_name in required_columns if column_name not in columns]
    incompatible = [
        f"{column_name} expected {expected_type}, found {columns.get(column_name)}"
        for column_name, expected_type in required_columns.items()
        if column_name in columns and columns[column_name] != expected_type
    ]
    if missing or incompatible:
        details = []
        if missing:
            details.append(f"missing columns: {', '.join(missing)}")
        if incompatible:
            details.append(f"incompatible columns: {', '.join(incompatible)}")
        raise RuntimeError(
            f"HANA table {table_name.upper()} exists with an incompatible schema: "
            + "; ".join(details)
        )


def ensure_email_attachments_table(connection: Connection) -> None:
    """Create or validate the attachment table once per API process.

    Args:
        connection: Active SQLAlchemy connection to SAP HANA.

    Returns:
        None. The table is present and compatible when the function returns.
    """
    global _email_attachments_schema_ready
    if _email_attachments_schema_ready:
        return
    with _email_attachments_schema_lock:
        if _email_attachments_schema_ready:
            return
        if table_exists(connection, EMAIL_ATTACHMENTS_TABLE):
            validate_email_attachments_table(connection)
        else:
            connection.execute(text(EMAIL_ATTACHMENTS_DDL))
        _email_attachments_schema_ready = True


def ensure_processing_progress_tables(connection: Connection) -> None:
    """Create or validate processing progress tables once per API process.

    Args:
        connection: Active SQLAlchemy connection to SAP HANA.

    Returns:
        None. The progress run and event tables are present and compatible.
    """
    global _processing_progress_schema_ready
    if _processing_progress_schema_ready:
        return
    with _processing_progress_schema_lock:
        if _processing_progress_schema_ready:
            return
        if table_exists(connection, PROCESSING_RUNS_TABLE):
            validate_table_contract(connection, PROCESSING_RUNS_TABLE, PROCESSING_RUNS_REQUIRED_COLUMNS)
        else:
            connection.execute(text(PROCESSING_RUNS_DDL))
        if table_exists(connection, PROCESSING_EVENTS_TABLE):
            validate_table_contract(connection, PROCESSING_EVENTS_TABLE, PROCESSING_EVENTS_REQUIRED_COLUMNS)
        else:
            connection.execute(text(PROCESSING_EVENTS_DDL))
        _processing_progress_schema_ready = True
