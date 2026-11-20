"""HANA table catalogue and automatic schema bootstrap.

Every table the application needs is declared here as data (column list + primary key)
so the same definition drives three things:

* ``CREATE COLUMN TABLE`` statements, run automatically at API start-up and by the
  seed CLI when a table is missing (customer landscapes need no manual DDL);
* the drift check that warns about columns missing in an existing table;
* the naming test (``tests/unit/test_schema_naming.py``).

Two families of tables exist:

* **Mock ECC** tables mirror SAP ECC SD/MM tables 1:1 in name and field names, with the
  suffix ``_POAGENT`` (e.g. ``KNA1_POAGENT``). They stand in for the real ECC system that
  is reached through SAP Integration Suite in production (see docs/ecc-integration.md).
* **App** tables, prefixed ``POAGENT_``, hold inbound messages, extracted orders,
  per-field confidence/issue metadata, agent runs and reviewer corrections.

Deliberate deviations from ECC are documented in docs/hana-data-model.md (no MANDT
client column, contact e-mail kept in KNVK instead of ADR6, unit aliases in a Z-table).

Example::

    from app.db.engine import get_engine
    from app.db.schema import ensure_schema
    summary = ensure_schema(get_engine())   # {"created_tables": [...], ...}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TableDef:
    """Declarative definition of one column table.

    Attributes:
        columns: Ordered (column name, HANA SQL type incl. constraints) pairs.
        primary_key: Column names forming the primary key.
        comment: One-line purpose, reused in documentation.
    """

    columns: tuple[tuple[str, str], ...]
    primary_key: tuple[str, ...]
    comment: str

    @property
    def column_names(self) -> list[str]:
        """Return the column names in declaration order."""
        return [name for name, _ in self.columns]


def _t(comment: str, pk: tuple[str, ...], *columns: tuple[str, str]) -> TableDef:
    """Shorthand constructor used by the catalogue below.

    Args:
        comment: Purpose of the table.
        pk: Primary-key column names.
        *columns: (name, type) pairs.

    Returns:
        The TableDef.
    """
    return TableDef(columns=tuple(columns), primary_key=pk, comment=comment)


# Reusable SAP data-element style types (lengths follow ECC).
KUNNR = "NVARCHAR(10)"   # customer number, ALPHA format (zero-padded)
MATNR = "NVARCHAR(18)"   # material number, ALPHA format for numeric values
VKORG = "NVARCHAR(4)"    # sales organisation
VTWEG = "NVARCHAR(2)"    # distribution channel
SPART = "NVARCHAR(2)"    # division
UNIT = "NVARCHAR(3)"     # SAP unit of measure (internal format, e.g. L, PC, GAL)
WERKS = "NVARCHAR(4)"    # plant
FLAG = "NVARCHAR(1)"     # 'X' or empty
UUID = "NVARCHAR(36)"    # application identifiers

# ---------------------------------------------------------------------------
# Mock ECC tables (SAP-native names + _POAGENT)
# ---------------------------------------------------------------------------
ECC_TABLES: dict[str, TableDef] = {
    "KNA1_POAGENT": _t(
        "Customer master: general data (KNA1)", ("KUNNR",),
        ("KUNNR", f"{KUNNR} NOT NULL"), ("KTOKD", "NVARCHAR(4)"), ("NAME1", "NVARCHAR(35) NOT NULL"),
        ("NAME2", "NVARCHAR(35)"), ("STRAS", "NVARCHAR(60)"), ("PSTLZ", "NVARCHAR(10)"),
        ("ORT01", "NVARCHAR(35)"), ("REGIO", "NVARCHAR(3)"), ("LAND1", "NVARCHAR(3) NOT NULL"),
        ("SPRAS", "NVARCHAR(1)"), ("TELF1", "NVARCHAR(16)"), ("STCEG", "NVARCHAR(20)"),
        ("AUFSD", "NVARCHAR(2)"), ("LOEVM", FLAG),
    ),
    "KNVV_POAGENT": _t(
        "Customer master: sales area data (KNVV)", ("KUNNR", "VKORG", "VTWEG", "SPART"),
        ("KUNNR", f"{KUNNR} NOT NULL"), ("VKORG", f"{VKORG} NOT NULL"), ("VTWEG", f"{VTWEG} NOT NULL"),
        ("SPART", f"{SPART} NOT NULL"), ("WAERS", "NVARCHAR(5)"), ("ZTERM", "NVARCHAR(4)"),
        ("INCO1", "NVARCHAR(3)"), ("INCO2", "NVARCHAR(28)"), ("VWERK", WERKS),
        ("AUFSD", "NVARCHAR(2)"), ("LOEVM", FLAG),
    ),
    "KNVP_POAGENT": _t(
        "Customer master: partner functions (KNVP); PARVW AG/WE/RE/RG", ("KUNNR", "VKORG", "VTWEG", "SPART", "PARVW", "PARZA"),
        ("KUNNR", f"{KUNNR} NOT NULL"), ("VKORG", f"{VKORG} NOT NULL"), ("VTWEG", f"{VTWEG} NOT NULL"),
        ("SPART", f"{SPART} NOT NULL"), ("PARVW", "NVARCHAR(2) NOT NULL"), ("PARZA", "NVARCHAR(3) NOT NULL"),
        ("KUNN2", f"{KUNNR} NOT NULL"), ("DEFPA", FLAG),
    ),
    "KNVK_POAGENT": _t(
        "Customer master: contact persons (KNVK, e-mail from ADR6 folded in)", ("PARNR",),
        ("PARNR", "NVARCHAR(10) NOT NULL"), ("KUNNR", f"{KUNNR} NOT NULL"), ("NAMEV", "NVARCHAR(35)"),
        ("NAME1", "NVARCHAR(35)"), ("TELF1", "NVARCHAR(16)"), ("SMTP_ADDR", "NVARCHAR(241)"),
    ),
    "MARA_POAGENT": _t(
        "Material master: general data (MARA); BISMT carries the sales code, MFRPN the manufacturer part number "
        "of traded goods", ("MATNR",),
        ("MATNR", f"{MATNR} NOT NULL"), ("MTART", "NVARCHAR(4)"), ("MATKL", "NVARCHAR(9)"),
        ("SPART", SPART), ("MEINS", f"{UNIT} NOT NULL"), ("BISMT", "NVARCHAR(18)"),
        ("VOLUM", "DECIMAL(13,3)"), ("VOLEH", UNIT), ("BRGEW", "DECIMAL(13,3)"), ("GEWEI", UNIT),
        ("MSTAE", "NVARCHAR(2)"), ("LVORM", FLAG), ("MFRPN", "NVARCHAR(40)"),
    ),
    "MAKT_POAGENT": _t(
        "Material descriptions per language (MAKT)", ("MATNR", "SPRAS"),
        ("MATNR", f"{MATNR} NOT NULL"), ("SPRAS", "NVARCHAR(1) NOT NULL"), ("MAKTX", "NVARCHAR(40) NOT NULL"),
    ),
    "MARM_POAGENT": _t(
        "Alternative units of measure (MARM): 1 MEINH = UMREZ/UMREN base units", ("MATNR", "MEINH"),
        ("MATNR", f"{MATNR} NOT NULL"), ("MEINH", f"{UNIT} NOT NULL"),
        ("UMREZ", "DECIMAL(5,0) NOT NULL"), ("UMREN", "DECIMAL(5,0) NOT NULL"),
    ),
    "MVKE_POAGENT": _t(
        "Material sales data per sales org / channel (MVKE)", ("MATNR", "VKORG", "VTWEG"),
        ("MATNR", f"{MATNR} NOT NULL"), ("VKORG", f"{VKORG} NOT NULL"), ("VTWEG", f"{VTWEG} NOT NULL"),
        ("VMSTA", "NVARCHAR(2)"), ("DWERK", WERKS), ("VRKME", UNIT),
    ),
    "KNMT_POAGENT": _t(
        "Customer-material info records (KNMT): customer's own material codes", ("VKORG", "VTWEG", "KUNNR", "MATNR"),
        ("VKORG", f"{VKORG} NOT NULL"), ("VTWEG", f"{VTWEG} NOT NULL"), ("KUNNR", f"{KUNNR} NOT NULL"),
        ("MATNR", f"{MATNR} NOT NULL"), ("KDMAT", "NVARCHAR(35)"), ("POSTX", "NVARCHAR(40)"), ("WERKS", WERKS),
    ),
    "A305_POAGENT": _t(
        "Customer-specific material price: condition table A305 (sales org / channel / customer / material) "
        "with its KONP rate folded in; KBETR per KPEIN KMEIN in KONWA", ("VKORG", "VTWEG", "KUNNR", "MATNR"),
        ("VKORG", f"{VKORG} NOT NULL"), ("VTWEG", f"{VTWEG} NOT NULL"), ("KUNNR", f"{KUNNR} NOT NULL"),
        ("MATNR", f"{MATNR} NOT NULL"), ("KBETR", "DECIMAL(15,2) NOT NULL"), ("KONWA", "NVARCHAR(5) NOT NULL"),
        ("KPEIN", "DECIMAL(5,0)"), ("KMEIN", UNIT),
    ),
    "ZUOM_ALIAS_POAGENT": _t(
        "Custom Z-table: free-text unit aliases (lower-case) to SAP units", ("ALIAS",),
        ("ALIAS", "NVARCHAR(30) NOT NULL"), ("MSEHI", f"{UNIT} NOT NULL"),
    ),
    "VBAK_POAGENT": _t(
        "Sales document header (VBAK)", ("VBELN",),
        ("VBELN", "NVARCHAR(10) NOT NULL"), ("AUART", "NVARCHAR(4) NOT NULL"), ("VKORG", VKORG),
        ("VTWEG", VTWEG), ("SPART", SPART), ("KUNNR", f"{KUNNR} NOT NULL"), ("BSTNK", "NVARCHAR(20)"),
        ("BSTDK", "DATE"), ("BSARK", "NVARCHAR(4)"), ("AUDAT", "DATE"), ("VDATU", "DATE"),
        ("WAERK", "NVARCHAR(5)"), ("NETWR", "DECIMAL(15,2)"), ("ERDAT", "DATE"), ("ERZET", "TIME"),
        ("ERNAM", "NVARCHAR(12)"),
    ),
    "VBKD_POAGENT": _t(
        "Sales document business data (VBKD); POSNR 000000 = header", ("VBELN", "POSNR"),
        ("VBELN", "NVARCHAR(10) NOT NULL"), ("POSNR", "NVARCHAR(6) NOT NULL"), ("BSTKD", "NVARCHAR(35)"),
        ("BSTDK", "DATE"), ("BSARK", "NVARCHAR(4)"), ("ZTERM", "NVARCHAR(4)"), ("INCO1", "NVARCHAR(3)"),
        ("INCO2", "NVARCHAR(28)"),
    ),
    "VBAP_POAGENT": _t(
        "Sales document items (VBAP); ZZCOLOR follows the customer-append convention", ("VBELN", "POSNR"),
        ("VBELN", "NVARCHAR(10) NOT NULL"), ("POSNR", "NVARCHAR(6) NOT NULL"), ("MATNR", f"{MATNR} NOT NULL"),
        ("ARKTX", "NVARCHAR(40)"), ("KDMAT", "NVARCHAR(35)"), ("KWMENG", "DECIMAL(15,3) NOT NULL"),
        ("VRKME", f"{UNIT} NOT NULL"), ("WERKS", WERKS), ("NETPR", "DECIMAL(11,2)"),
        ("WAERK", "NVARCHAR(5)"), ("ZZCOLOR", "NVARCHAR(20)"),
    ),
    "VBPA_POAGENT": _t(
        "Sales document partners (VBPA); POSNR 000000 = header", ("VBELN", "POSNR", "PARVW"),
        ("VBELN", "NVARCHAR(10) NOT NULL"), ("POSNR", "NVARCHAR(6) NOT NULL"),
        ("PARVW", "NVARCHAR(2) NOT NULL"), ("KUNNR", f"{KUNNR} NOT NULL"),
    ),
}

# ---------------------------------------------------------------------------
# Application tables (POAGENT_ prefix)
# ---------------------------------------------------------------------------
APP_TABLES: dict[str, TableDef] = {
    "POAGENT_INBOUND_MESSAGE": _t(
        "One inbound e-mail or upload batch", ("MESSAGE_ID",),
        ("MESSAGE_ID", f"{UUID} NOT NULL"), ("SOURCE_TYPE", "NVARCHAR(10) NOT NULL"),
        ("SENDER", "NVARCHAR(320)"), ("RECIPIENTS", "NVARCHAR(2000)"), ("SUBJECT", "NVARCHAR(1000)"),
        ("SENT_AT", "TIMESTAMP"), ("RECEIVED_AT", "TIMESTAMP NOT NULL"), ("BODY_TEXT", "NCLOB"),
        ("SALES_ORG_HINT", VKORG),
    ),
    "POAGENT_ATTACHMENT": _t(
        "Files of a message (attachments, inline images, forwarded e-mails)", ("ATTACHMENT_ID",),
        ("ATTACHMENT_ID", f"{UUID} NOT NULL"), ("MESSAGE_ID", f"{UUID} NOT NULL"), ("PARENT_ID", UUID),
        ("SEQ", "INTEGER NOT NULL"), ("FILE_NAME", "NVARCHAR(500) NOT NULL"), ("MIME_TYPE", "NVARCHAR(200)"),
        ("SIZE_BYTES", "BIGINT"), ("ROLE", "NVARCHAR(20) NOT NULL"), ("MODEL_INPUT", "NVARCHAR(10) NOT NULL"),
        ("MODEL_INPUT_REASON", "NVARCHAR(200)"), ("CONTENT", "BLOB"), ("EXTRACTED_TEXT", "NCLOB"),
    ),
    "POAGENT_ORDER_HEADER": _t(
        "Extracted / validated purchase order and its processing state", ("ORDER_ID",),
        ("ORDER_ID", f"{UUID} NOT NULL"), ("MESSAGE_ID", f"{UUID} NOT NULL"), ("STATUS", "NVARCHAR(20) NOT NULL"),
        ("CLASSIFICATION", "NVARCHAR(20)"), ("CLASSIFICATION_REASON", "NVARCHAR(1000)"),
        ("STATUS_MESSAGE", "NVARCHAR(2000)"), ("ATTEMPTS", "INTEGER DEFAULT 0 NOT NULL"),
        ("LOCKED_BY", "NVARCHAR(64)"), ("LOCKED_AT", "TIMESTAMP"), ("REVIEWER_NOTE", "NVARCHAR(2000)"),
        ("PO_NUMBER", "NVARCHAR(35)"), ("PO_DATE", "DATE"), ("REQ_DELIVERY_DATE", "DATE"),
        ("SOLD_TO", KUNNR), ("SOLD_TO_NAME", "NVARCHAR(200)"), ("SOLD_TO_ADDRESS", "NVARCHAR(500)"),
        ("SHIP_TO", KUNNR), ("SHIP_TO_NAME", "NVARCHAR(200)"), ("SHIP_TO_ADDRESS", "NVARCHAR(500)"),
        ("SALES_ORG", VKORG), ("DISTR_CHANNEL", VTWEG), ("DIVISION", SPART), ("ORDER_TYPE", "NVARCHAR(4)"),
        ("PO_TYPE", "NVARCHAR(4)"), ("CURRENCY", "NVARCHAR(5)"), ("NET_AMOUNT", "DECIMAL(15,2)"),
        ("GROSS_AMOUNT", "DECIMAL(15,2)"), ("PAYMENT_TERMS_TEXT", "NVARCHAR(200)"), ("PAYMENT_TERMS", "NVARCHAR(4)"),
        ("INCOTERMS", "NVARCHAR(3)"), ("INCOTERMS_PLACE", "NVARCHAR(28)"), ("HEADER_NOTE", "NVARCHAR(2000)"),
        ("AGENT_SUMMARY", "NCLOB"), ("SALES_DOC", "NVARCHAR(10)"), ("POSTING_PAYLOAD", "NCLOB"),
        ("POSTED_AT", "TIMESTAMP"), ("CREATED_AT", "TIMESTAMP DEFAULT CURRENT_UTCTIMESTAMP NOT NULL"),
        ("UPDATED_AT", "TIMESTAMP DEFAULT CURRENT_UTCTIMESTAMP NOT NULL"),
    ),
    "POAGENT_ORDER_ITEM": _t(
        "Order lines (MATERIAL / CHARGE / TEXT); ITEM_NO 1, 2, ... in document order", ("ORDER_ID", "ITEM_NO"),
        ("ORDER_ID", f"{UUID} NOT NULL"), ("ITEM_NO", "INTEGER NOT NULL"), ("LINE_TYPE", "NVARCHAR(10) NOT NULL"),
        ("CUSTOMER_ITEM_NO", "NVARCHAR(20)"), ("MATERIAL", MATNR), ("CUSTOMER_MATERIAL", "NVARCHAR(35)"),
        ("DESCRIPTION", "NVARCHAR(200)"), ("QUANTITY", "DECIMAL(15,3)"), ("UOM", UNIT),
        ("UNIT_PRICE", "DECIMAL(15,4)"), ("LINE_AMOUNT", "DECIMAL(15,2)"), ("COLOUR_CODE", "NVARCHAR(20)"),
        ("PLANT", WERKS),
    ),
    "POAGENT_FIELD_STATE": _t(
        "Per-field metadata: raw value, confidences, fixes, issues (ITEM_NO 0 = header)",
        ("ORDER_ID", "ITEM_NO", "FIELD_NAME"),
        ("ORDER_ID", f"{UUID} NOT NULL"), ("ITEM_NO", "INTEGER NOT NULL"), ("FIELD_NAME", "NVARCHAR(40) NOT NULL"),
        ("RAW_VALUE", "NVARCHAR(2000)"), ("SOURCE_REF", "NVARCHAR(200)"), ("EXTRACT_CONFIDENCE", "DECIMAL(4,3)"),
        ("MATCH_CONFIDENCE", "DECIMAL(4,3)"), ("CONFIDENCE", "DECIMAL(4,3)"),
        ("CHANGED_BY", "NVARCHAR(10) NOT NULL"), ("FIX_NOTE", "NVARCHAR(1000)"),
        ("ISSUE_SEVERITY", "NVARCHAR(10)"), ("ISSUE_CODE", "NVARCHAR(40)"), ("ISSUE_TEXT", "NVARCHAR(2000)"),
        ("UPDATED_AT", "TIMESTAMP"), ("EVIDENCE", "NVARCHAR(2000)"),
    ),
    "POAGENT_CORRECTION_LOG": _t(
        "Reviewer corrections (long-term agent memory)", ("CORRECTION_ID",),
        ("CORRECTION_ID", f"{UUID} NOT NULL"), ("ORDER_ID", f"{UUID} NOT NULL"), ("ITEM_NO", "INTEGER NOT NULL"),
        ("FIELD_NAME", "NVARCHAR(40) NOT NULL"), ("SOLD_TO", KUNNR), ("RAW_VALUE", "NVARCHAR(2000)"),
        ("AGENT_VALUE", "NVARCHAR(500)"), ("USER_VALUE", "NVARCHAR(500)"),
        ("CREATED_AT", "TIMESTAMP DEFAULT CURRENT_UTCTIMESTAMP NOT NULL"),
    ),
    "POAGENT_AGENT_RUN": _t(
        "One agent execution per processing attempt, with token usage and trace", ("RUN_ID",),
        ("RUN_ID", f"{UUID} NOT NULL"), ("ORDER_ID", f"{UUID} NOT NULL"), ("STARTED_AT", "TIMESTAMP NOT NULL"),
        ("FINISHED_AT", "TIMESTAMP"), ("OUTCOME", "NVARCHAR(20)"), ("MODEL", "NVARCHAR(60)"),
        ("LLM_CALLS", "INTEGER"), ("INPUT_TOKENS", "BIGINT"), ("CACHED_INPUT_TOKENS", "BIGINT"),
        ("OUTPUT_TOKENS", "BIGINT"), ("ERROR_TEXT", "NVARCHAR(2000)"), ("REVIEWER_NOTE", "NVARCHAR(2000)"),
        ("TRACE", "NCLOB"),
    ),
}

TABLES: dict[str, TableDef] = {**ECC_TABLES, **APP_TABLES}

# Mock number range for sales documents; clearly outside real ECC ranges.
SEQUENCES: dict[str, str] = {
    "VBELN_SEQ_POAGENT": "START WITH 9100000001 INCREMENT BY 1 NO CYCLE",
}


def create_table_sql(name: str, table: TableDef) -> str:
    """Render the CREATE COLUMN TABLE statement for one table.

    Args:
        name: Table name (upper case, unquoted-safe).
        table: Its definition.

    Returns:
        The DDL statement.
    """
    cols = ",\n  ".join(f'"{col}" {sql_type}' for col, sql_type in table.columns)
    pk = ", ".join(f'"{col}"' for col in table.primary_key)
    return f'CREATE COLUMN TABLE "{name}" (\n  {cols},\n  PRIMARY KEY ({pk})\n)'


def ensure_schema(engine: Engine) -> dict[str, list[str]]:
    """Create every missing table and sequence in the current schema.

    Existing tables are never dropped. Columns missing in an existing table are added when they
    are nullable (safe ``ALTER TABLE ... ADD``); missing NOT NULL columns are only reported (and
    logged) so an operator can migrate them.

    Args:
        engine: SQLAlchemy engine connected to HANA.

    Returns:
        Summary dict with keys ``created_tables``, ``created_sequences``, ``added_columns`` and
        ``missing_columns`` (entries formatted ``TABLE.COLUMN``).
    """
    summary: dict[str, list[str]] = {"created_tables": [], "created_sequences": [], "added_columns": [],
                                     "missing_columns": []}
    with engine.begin() as conn:
        existing = {
            row[0]
            for row in conn.execute(text("SELECT TABLE_NAME FROM SYS.TABLES WHERE SCHEMA_NAME = CURRENT_SCHEMA"))
        }
        existing_seq = {
            row[0]
            for row in conn.execute(text("SELECT SEQUENCE_NAME FROM SYS.SEQUENCES WHERE SCHEMA_NAME = CURRENT_SCHEMA"))
        }
        columns_by_table: dict[str, set[str]] = {}
        for table_name, column_name in conn.execute(
            text("SELECT TABLE_NAME, COLUMN_NAME FROM SYS.TABLE_COLUMNS WHERE SCHEMA_NAME = CURRENT_SCHEMA")
        ):
            columns_by_table.setdefault(table_name, set()).add(column_name)

        for name, table in TABLES.items():
            if name not in existing:
                conn.execute(text(create_table_sql(name, table)))
                summary["created_tables"].append(name)
                continue
            for col, sql_type in table.columns:
                if col in columns_by_table.get(name, set()):
                    continue
                if "NOT NULL" in sql_type:
                    summary["missing_columns"].append(f"{name}.{col}")
                else:
                    conn.execute(text(f'ALTER TABLE "{name}" ADD ("{col}" {sql_type})'))
                    summary["added_columns"].append(f"{name}.{col}")

        for name, options in SEQUENCES.items():
            if name not in existing_seq:
                conn.execute(text(f'CREATE SEQUENCE "{name}" {options}'))
                summary["created_sequences"].append(name)

    logger.info(
        "HANA schema check: %d tables created, %d sequences created, %d columns added, %d missing columns",
        len(summary["created_tables"]), len(summary["created_sequences"]), len(summary["added_columns"]),
        len(summary["missing_columns"]),
    )
    if summary["missing_columns"]:
        logger.warning("Columns missing in existing tables (migrate manually): %s", summary["missing_columns"])
    return summary


def clip_row(table: str, row: dict) -> dict:
    """Truncate string values to the NVARCHAR length declared for each column.

    Model output can be longer than ECC field lengths (e.g. a 40-character MAKTX); clipping
    avoids insert failures. Columns not declared in the table are dropped.

    Args:
        table: Table name in ``TABLES``.
        row: Column -> value mapping.

    Returns:
        A new dict containing only known columns, strings clipped to their length.
    """
    import re

    clipped = {}
    for column, sql_type in TABLES[table].columns:
        if column not in row:
            continue
        value = row[column]
        match = re.match(r"NVARCHAR\((\d+)\)", sql_type)
        if match and isinstance(value, str):
            value = value[: int(match.group(1))]
        clipped[column] = value
    return clipped
