from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from email.utils import parseaddr, parsedate_to_datetime
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from .models import GmailEmail


@dataclass
class ScenarioSourceEmail:
    """Metadata and body preview for one source scenario email.

    Attributes:
        scenario_email_id: Stable id derived from the scenario and file name.
        scenario_id: Scenario folder name such as scenario1.
        source_path: Original email file path.
        subject: Email subject.
        sender: Raw From header.
        email_date: Parsed email date.
        body_preview: Compact plain-text body preview for HANA audit rows.
    """

    scenario_email_id: str
    scenario_id: str
    source_path: str
    subject: str
    sender: str
    email_date: datetime | None
    body_preview: str


@dataclass
class ParsedScenarioEmail:
    """Parsed scenario email tables ready for HANA ingestion.

    Attributes:
        scenario_id: Scenario folder name.
        source_email: Source email metadata.
        status_lines: Normalized sales-order status rows.
        item_details: Normalized sales-order item detail rows.
    """

    scenario_id: str
    source_email: ScenarioSourceEmail
    status_lines: list[dict[str, Any]] = field(default_factory=list)
    item_details: list[dict[str, Any]] = field(default_factory=list)


def slugify(value: str) -> str:
    """Convert arbitrary text into a stable lowercase id fragment.

    Args:
        value: Text to normalize.

    Returns:
        Lowercase alphanumeric id fragment separated by hyphens.
    """
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def normalize_line_identifier(value: str | None) -> str | None:
    """Normalize customer-facing line identifiers to SAP four-digit item lines.

    Args:
        value: Raw line text such as "20" or "0020".

    Returns:
        Four-digit line value, or None when the input has no digits.
    """
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value))
    if not digits:
        return None
    return digits.zfill(4)


def normalize_product_identifier(value: str | None) -> str:
    """Normalize product ids for exact and suffix-aware comparisons.

    Args:
        value: Raw product id such as "3548121-US".

    Returns:
        Uppercase alphanumeric-only product id.
    """
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def product_matches(stored_product: str | None, requested_product: str | None) -> bool:
    """Return True when a requested product matches a stored HANA product id.

    Args:
        stored_product: Product id from HANA, possibly with suffix such as "-US".
        requested_product: Product id extracted from the request.

    Returns:
        True for exact matches or when the stored product's base part matches.
    """
    stored = normalize_product_identifier(stored_product)
    requested = normalize_product_identifier(requested_product)
    stored_base = normalize_product_identifier(str(stored_product or "").split("-", 1)[0])
    return bool(requested) and (stored == requested or stored_base == requested)


def parse_email_date(raw_date: str | None) -> datetime | None:
    """Parse an email Date header.

    Args:
        raw_date: Raw Date header value.

    Returns:
        Timezone-aware datetime or None when parsing fails.
    """
    if not raw_date:
        return None
    try:
        parsed = parsedate_to_datetime(raw_date)
    except (TypeError, ValueError):
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def extract_email_parts(path: Path) -> tuple[dict[str, str], str, str]:
    """Read source email headers plus plain and HTML body parts.

    Args:
        path: Path to an .eml file.

    Returns:
        Tuple of headers, plain body text, and HTML body text.
    """
    message = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
    plain_parts: list[str] = []
    html_parts: list[str] = []
    for part in message.walk():
        content_type = part.get_content_type()
        if content_type == "text/plain":
            plain_parts.append(str(part.get_content()))
        elif content_type == "text/html":
            html_parts.append(str(part.get_content()))
    return (
        {
            "subject": str(message.get("subject") or ""),
            "from": str(message.get("from") or ""),
            "date": str(message.get("date") or ""),
            "message_id": str(message.get("message-id") or ""),
        },
        "\n".join(plain_parts).strip(),
        "\n".join(html_parts).strip(),
    )


def html_to_text(html: str) -> str:
    """Convert HTML email content into compact readable text.

    Args:
        html: HTML body content.

    Returns:
        Plain text with collapsed whitespace.
    """
    return " ".join(BeautifulSoup(html, "html.parser").get_text(" ", strip=True).split())


def scenario_id_from_path(path: Path) -> str:
    """Extract scenario folder id from a source path.

    Args:
        path: Scenario email path.

    Returns:
        Scenario id such as scenario1.

    Raises:
        ValueError: If the path is not under a scenario folder.
    """
    for part in path.parts:
        if re.fullmatch(r"scenario\d+", part):
            return part
    raise ValueError(f"No scenario folder found in {path}")


def parse_scenario_email(path: Path) -> ParsedScenarioEmail:
    """Parse status and item-detail tables from a scenario .eml file.

    Args:
        path: Path to a scenario email.

    Returns:
        ParsedScenarioEmail with normalized HANA row dictionaries.
    """
    headers, plain_text, html = extract_email_parts(path)
    scenario_id = scenario_id_from_path(path)
    body_preview_source = html_to_text(html) if html else plain_text
    source_email = ScenarioSourceEmail(
        scenario_email_id=f"scenario-email-{scenario_id}-{slugify(path.stem)}",
        scenario_id=scenario_id,
        source_path=str(path),
        subject=headers["subject"],
        sender=headers["from"],
        email_date=parse_email_date(headers["date"]),
        body_preview=body_preview_source[:4000],
    )
    status_lines: list[dict[str, Any]] = []
    item_details_by_key: dict[str, dict[str, Any]] = {}
    for table_rows in extract_html_tables(html):
        if not table_rows:
            continue
        table_headers = table_rows[0]
        data_rows = table_rows[1:]
        header_map = {normalize_header(header): index for index, header in enumerate(table_headers)}
        if is_status_table(header_map):
            status_lines.extend(
                normalize_status_row(row, header_map, scenario_id, source_email.scenario_email_id)
                for row in data_rows
            )
        elif is_item_detail_table(header_map):
            for row in data_rows:
                item_detail = normalize_item_detail_row(row, header_map, scenario_id, source_email.scenario_email_id)
                key = str(item_detail["item_detail_id"])
                existing = item_details_by_key.get(key)
                item_details_by_key[key] = merge_non_empty(existing, item_detail) if existing else item_detail
    return ParsedScenarioEmail(
        scenario_id=scenario_id,
        source_email=source_email,
        status_lines=status_lines,
        item_details=list(item_details_by_key.values()),
    )


def scenario_email_to_gmail_email(path: Path) -> GmailEmail:
    """Convert a source scenario .eml file into a GmailEmail for processing.

    Args:
        path: Path to a scenario email.

    Returns:
        GmailEmail with stable scenario ids and readable body text.
    """
    headers, plain_text, html = extract_email_parts(path)
    scenario_id = scenario_id_from_path(path)
    sender_name, sender_email = parseaddr(headers["from"])
    body = plain_text or html_to_text(html)
    message_id = headers["message_id"] or f"<scenario-{scenario_id}-{slugify(path.stem)}@local>"
    return GmailEmail(
        gmail_message_id=f"scenario-{scenario_id}-{slugify(path.stem)}",
        thread_id=f"scenario-thread-{scenario_id}",
        rfc_message_id=message_id,
        sender_name=sender_name or None,
        sender_email=sender_email or None,
        subject=headers["subject"] or None,
        email_date=parse_email_date(headers["date"]),
        body=body,
    )


def load_scenario_emails(source_dir: Path) -> list[GmailEmail]:
    """Load all source scenario emails as processable GmailEmail rows.

    Args:
        source_dir: Directory containing scenario subfolders.

    Returns:
        GmailEmail rows ordered by path for deterministic processing.
    """
    return [scenario_email_to_gmail_email(path) for path in sorted(source_dir.glob("scenario*/*.eml"))]


def extract_html_tables(html: str) -> list[list[list[str]]]:
    """Extract visible rows and cells from HTML tables.

    Args:
        html: HTML email body.

    Returns:
        List of tables, each represented as rows of cell text.
    """
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    tables: list[list[list[str]]] = []
    for table in soup.find_all("table"):
        rows: list[list[str]] = []
        for tr in table.find_all("tr"):
            cells = [" ".join(cell.get_text(" ", strip=True).split()) for cell in tr.find_all(["th", "td"])]
            if cells and any(cells):
                rows.append(cells)
        if len(rows) > 1:
            tables.append(rows)
    return tables


def normalize_header(value: str) -> str:
    """Normalize an email table header to a compact matching key.

    Args:
        value: Raw header text.

    Returns:
        Uppercase alphanumeric header key.
    """
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def is_status_table(header_map: dict[str, int]) -> bool:
    """Return True when headers describe a sales-order status table.

    Args:
        header_map: Normalized table headers to column positions.

    Returns:
        True for status-line tables.
    """
    required = {"SOORDER", "DOCNO", "LINE", "PRODUCT", "STATUSGI"}
    return required.issubset(set(header_map))


def is_item_detail_table(header_map: dict[str, int]) -> bool:
    """Return True when headers describe an item-detail table.

    Args:
        header_map: Normalized table headers to column positions.

    Returns:
        True for sales-order item detail tables.
    """
    required = {"SOORDER", "ITEM", "PRODUCT", "LOT"}
    return required.issubset(set(header_map))


def row_value(row: list[str], header_map: dict[str, int], *keys: str) -> str | None:
    """Read a row value by the first matching normalized header key.

    Args:
        row: Table row cells.
        header_map: Normalized headers to positions.
        keys: Candidate normalized header keys.

    Returns:
        Cell value or None when unavailable.
    """
    for key in keys:
        index = header_map.get(key)
        if index is not None and index < len(row):
            value = row[index].strip()
            return None if value in {"", "-"} else value
    return None


def normalize_status_row(
    row: list[str], header_map: dict[str, int], scenario_id: str, scenario_email_id: str
) -> dict[str, Any]:
    """Normalize one status table row for HANA.

    Args:
        row: Raw table row.
        header_map: Normalized headers to positions.
        scenario_id: Scenario folder id.
        scenario_email_id: Source email id.

    Returns:
        HANA-ready status line dictionary.
    """
    sales_order_id = row_value(row, header_map, "SOORDER")
    outbound_delivery_id = row_value(row, header_map, "DOCNO")
    line_number = normalize_line_identifier(row_value(row, header_map, "LINE"))
    product_id = row_value(row, header_map, "PRODUCT")
    batch_number = row_value(row, header_map, "BATCHNO")
    status_line_id = "status-line-" + slugify(
        "-".join(str(item or "") for item in [sales_order_id, outbound_delivery_id, line_number, product_id, batch_number])
    )
    return {
        "status_line_id": status_line_id,
        "scenario_id": scenario_id,
        "scenario_email_id": scenario_email_id,
        "order_date": row_value(row, header_map, "ORDERDATE"),
        "order_time": row_value(row, header_map, "ORDERTIME"),
        "client_country": row_value(row, header_map, "COUNTRY"),
        "route_id": row_value(row, header_map, "ROUTEID"),
        "sales_order_id": sales_order_id,
        "outbound_delivery_id": outbound_delivery_id,
        "document_type_code": row_value(row, header_map, "DOCTYPE"),
        "line_number": line_number,
        "product_id": product_id,
        "batch_number": batch_number,
        "quantity_ordered": row_value(row, header_map, "QTYORDERED"),
        "quantity_packed": row_value(row, header_map, "QTYPACKED"),
        "dangerous_goods": row_value(row, header_map, "DG"),
        "pick_date": row_value(row, header_map, "PICKDATE"),
        "pack_box": row_value(row, header_map, "PACKBOX"),
        "status_pick": row_value(row, header_map, "STATUSPICK"),
        "status_gi": row_value(row, header_map, "STATUSGI"),
    }


def normalize_item_detail_row(
    row: list[str], header_map: dict[str, int], scenario_id: str, scenario_email_id: str
) -> dict[str, Any]:
    """Normalize one item detail table row for HANA.

    Args:
        row: Raw table row.
        header_map: Normalized headers to positions.
        scenario_id: Scenario folder id.
        scenario_email_id: Source email id.

    Returns:
        HANA-ready item detail dictionary.
    """
    sales_order_id = row_value(row, header_map, "SOORDER")
    item_number = normalize_line_identifier(row_value(row, header_map, "ITEM"))
    product_id = row_value(row, header_map, "PRODUCT")
    lot_number = row_value(row, header_map, "LOT")
    item_detail_id = "item-detail-" + slugify(
        "-".join(str(item or "") for item in [sales_order_id, item_number, product_id, lot_number])
    )
    return {
        "item_detail_id": item_detail_id,
        "scenario_id": scenario_id,
        "scenario_email_id": scenario_email_id,
        "sales_order_id": sales_order_id,
        "customer_ref": row_value(row, header_map, "CUSTOMERREF"),
        "order_type": row_value(row, header_map, "ORDERTYPE", "ORDERTYP"),
        "item_number": item_number,
        "sold_to_name": row_value(row, header_map, "SOLDTONAME"),
        "quantity": row_value(row, header_map, "QTY"),
        "uom": row_value(row, header_map, "UOM"),
        "product_id": product_id,
        "lot_number": lot_number,
        "description": row_value(row, header_map, "DESCRIPTION"),
        "line_weight_kg": row_value(row, header_map, "LINEWEIGHTKG"),
        "invoice_id": row_value(row, header_map, "INVOICE"),
        "invoice_date": row_value(row, header_map, "INVOICEDATE"),
        "foreign_amount": row_value(row, header_map, "FOREIGNAMOUNT"),
        "commodity_code": row_value(row, header_map, "COMMODITYCODE"),
        "commodity_description": row_value(row, header_map, "COMMODITYCODEDESCRIPTION", "COMMODITYCODEDESCRIPTIO"),
        "country_of_origin": row_value(row, header_map, "COO"),
        "expiry_date": row_value(row, header_map, "EXPIRYDATE"),
        "manufacturing_date": row_value(row, header_map, "MFGDATE"),
    }


def merge_non_empty(existing: dict[str, Any] | None, incoming: dict[str, Any]) -> dict[str, Any]:
    """Merge duplicate item rows without losing richer source values.

    Args:
        existing: Existing row for the same natural key.
        incoming: Newly parsed row.

    Returns:
        Merged row preferring non-empty values from either row.
    """
    if existing is None:
        return dict(incoming)
    merged = dict(existing)
    for key, value in incoming.items():
        if (merged.get(key) is None or merged.get(key) == "") and value not in {None, ""}:
            merged[key] = value
    return merged
