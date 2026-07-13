"""
Excel Parser Service

Handles parsing of offer Excel files with European number formats and date handling.
"""

import logging
import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from io import BytesIO
from typing import List, Optional, Tuple, Union

import pandas as pd

from ...models.eligibility import OfferInvoice

logger = logging.getLogger(__name__)

# Expected column mappings (Excel column name -> model field name)
COLUMN_MAPPINGS = {
    "PROGRAMA": "programa",
    "ID SELLER": "seller_id",
    "SELLER": "seller_name",
    "ID DEBTOR": "debtor_id",
    "DEBTOR": "debtor_name",
    "ORIGINAL CURRENCY": "original_currency",
    "ISSUANCE DATE": "issuance_date",
    "DUE DATE": "due_date",
}

REFERENCE_NUMBER_COLUMNS = (
    "REFERENCE NUMBER",
    "INVOICE REF",
    "REFERENCE",
    "XBLNR",
)

DOC_NUMBER_COLUMNS = (
    "DOC NUMBER",
    "DOCUMENT NUMBER",
    "DOC NO",
    "DOC NUMBER (BELNR)",
    "BELNR",
)

FISCAL_YEAR_COLUMNS = (
    "FISCAL YEAR",
    "FISCAL YEAR (GJAHR)",
    "GJAHR",
)

# Optional columns for extended offer metadata
OPTIONAL_COLUMNS = {
    "INSURER ID": "insurer_id",
    "GOODS SERVICES": "goods_services",
    "ITEM": "item",
    "TOTAL INVOICE AMOUNT (ORIGINAL CCY)": "total_invoice_amount_original",
    "TOTAL NET VALUE (ORIGINAL CCY)": "total_net_value_original",
    "DISCOUNT PERCENTAGE": "discount_percentage",
    "FUNDING CURRENCY": "funding_currency",
    "EXCHANGE RATE": "exchange_rate",
    "DESPATCH DATE": "despatch_date",
    "MARGIN": "margin",
}

# Optional amount columns
AMOUNT_COLUMNS = {
    "AMOUNT ORIGINAL": "amount_original",
    "AMOUNT EUR": "amount_eur",
    "AMOUNT (ORIGINAL)": "amount_original",
    "AMOUNT (EUR)": "amount_eur",
    "ORIGINAL AMOUNT": "amount_original",
    "EUR AMOUNT": "amount_eur",
    "TOTAL INVOICE AMOUNT (ORIGINAL CCY)": "amount_original",
    "TOTAL NET VALUE (ORIGINAL CCY)": "amount_original",
    "TOTAL INVOICE AMOUNT (EUR)": "amount_eur",
    "TOTAL NET VALUE (EUR)": "amount_eur",
}


def parse_european_number(value: Union[str, float, int, None]) -> Optional[Decimal]:
    """
    Parse a number in European format (e.g., '6.721,55') to Decimal.

    European format uses:
    - Period (.) as thousands separator
    - Comma (,) as decimal separator

    Examples:
        '6.721,55' -> Decimal('6721.55')
        '1.234.567,89' -> Decimal('1234567.89')
        '100' -> Decimal('100')
        '100,50' -> Decimal('100.50')

    Args:
        value: The value to parse (string, float, int, or None)

    Returns:
        Decimal representation of the number, or None if parsing fails
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        return Decimal(str(value))

    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() in ("nan", "none", "null", ""):
            return None

        # Remove currency symbols and whitespace
        value = re.sub(r"[^\d.,\-]", "", value)

        if not value:
            return None

        # Detect format: if last separator is comma, it's European format
        # European: 1.234,56 (comma is decimal)
        # US: 1,234.56 (period is decimal)
        last_comma = value.rfind(",")
        last_period = value.rfind(".")

        if last_comma > last_period:
            # European format: remove period (thousands), replace comma with period
            value = value.replace(".", "").replace(",", ".")
        elif last_period > last_comma and last_comma != -1:
            # US format: just remove commas (thousands)
            value = value.replace(",", "")
        # If only comma or only period, need to determine based on position
        elif last_comma != -1 and last_period == -1:
            # Only comma - check if it's thousands or decimal
            # If 3 digits after comma, it's thousands separator
            parts = value.split(",")
            if len(parts) == 2 and len(parts[1]) == 3:
                # Thousands separator
                value = value.replace(",", "")
            else:
                # Decimal separator
                value = value.replace(",", ".")

        try:
            return Decimal(value)
        except InvalidOperation:
            logger.warning(f"Failed to parse number: {value}")
            return None

    return None


def parse_date(value: Union[str, datetime, date, None]) -> Optional[date]:
    """
    Parse a date value from various formats.

    Handles:
    - datetime objects
    - date objects
    - Strings in DD/MM/YYYY, YYYY-MM-DD, or other common formats
    - pandas Timestamp

    Args:
        value: The value to parse

    Returns:
        date object, or None if parsing fails
    """
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    if pd.isna(value):
        return None

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None

        # Try common date formats
        formats = [
            "%d/%m/%Y",  # DD/MM/YYYY (European)
            "%Y-%m-%d",  # YYYY-MM-DD (ISO)
            "%d-%m-%Y",  # DD-MM-YYYY
            "%m/%d/%Y",  # MM/DD/YYYY (US)
            "%Y/%m/%d",  # YYYY/MM/DD
            "%d.%m.%Y",  # DD.MM.YYYY (German)
        ]

        for fmt in formats:
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue

        # Try pandas parsing as fallback
        try:
            parsed = pd.to_datetime(value, dayfirst=True)
            if pd.notna(parsed):
                return parsed.date()
        except Exception:
            pass

        logger.warning(f"Failed to parse date: {value}")
        return None

    # Handle pandas Timestamp
    if hasattr(value, "date"):
        return value.date()

    return None


def normalize_column_name(col: str) -> str:
    """Normalize column name for matching."""
    return col.strip().upper()


def normalize_identifier(value: Union[str, float, int, None]) -> Optional[str]:
    """Normalize identifier-like values (e.g., doc numbers) to stable strings."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip()


def _first_present(row: pd.Series, columns: Tuple[str, ...]) -> Optional[str]:
    for column in columns:
        if column in row.index and pd.notna(row[column]):
            return normalize_identifier(row[column])
    return None


def parse_offer_file(file_content: bytes, filename: str = "offer.xlsx") -> List[OfferInvoice]:
    """
    Parse an offer Excel file and return a list of OfferInvoice objects.

    Args:
        file_content: The raw bytes of the Excel file
        filename: Original filename (for logging)

    Returns:
        List of OfferInvoice objects

    Raises:
        ValueError: If required columns are missing or file cannot be read
    """
    logger.info(f"Parsing offer file: {filename}")

    try:
        # Read Excel file
        df = pd.read_excel(BytesIO(file_content), engine="openpyxl")
    except Exception as e:
        raise ValueError(f"Failed to read Excel file: {e}")

    if df.empty:
        logger.warning("Excel file is empty")
        return []

    # Normalize column names
    df.columns = [normalize_column_name(str(col)) for col in df.columns]

    # Check for required columns
    missing_columns = []
    for excel_col in COLUMN_MAPPINGS.keys():
        if excel_col not in df.columns:
            missing_columns.append(excel_col)
    if not any(col in df.columns for col in REFERENCE_NUMBER_COLUMNS):
        missing_columns.append("INVOICE REF / REFERENCE NUMBER")

    if missing_columns:
        available_cols = list(df.columns)
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: {available_cols}"
        )

    # Find amount columns (optional)
    amount_original_col = None
    amount_eur_col = None
    for excel_col, field in AMOUNT_COLUMNS.items():
        if excel_col in df.columns:
            if field == "amount_original" and amount_original_col is None:
                amount_original_col = excel_col
            elif field == "amount_eur" and amount_eur_col is None:
                amount_eur_col = excel_col

    invoices = []
    errors = []

    for idx, row in df.iterrows():
        row_num = idx + 2  # Excel rows are 1-indexed, plus header row

        try:
            # Parse required fields
            programa = str(row["PROGRAMA"]).strip() if pd.notna(row["PROGRAMA"]) else ""
            seller_id = str(row["ID SELLER"]).strip() if pd.notna(row["ID SELLER"]) else ""
            seller_name = str(row["SELLER"]).strip() if pd.notna(row["SELLER"]) else ""
            debtor_id = str(row["ID DEBTOR"]).strip() if pd.notna(row["ID DEBTOR"]) else ""
            debtor_name = str(row["DEBTOR"]).strip() if pd.notna(row["DEBTOR"]) else ""
            reference_number = _first_present(row, REFERENCE_NUMBER_COLUMNS)
            invoice_ref = reference_number or ""
            original_currency = (
                str(row["ORIGINAL CURRENCY"]).strip().upper()
                if pd.notna(row["ORIGINAL CURRENCY"])
                else ""
            )
            doc_number = _first_present(row, DOC_NUMBER_COLUMNS)
            fiscal_year = _first_present(row, FISCAL_YEAR_COLUMNS)

            insurer_id = (
                str(row["INSURER ID"]).strip()
                if "INSURER ID" in row.index and pd.notna(row["INSURER ID"])
                else None
            )
            goods_services = (
                str(row["GOODS SERVICES"]).strip()
                if "GOODS SERVICES" in row.index and pd.notna(row["GOODS SERVICES"])
                else None
            )
            item = (
                str(row["ITEM"]).strip()
                if "ITEM" in row.index and pd.notna(row["ITEM"])
                else None
            )
            total_invoice_amount_original = (
                parse_european_number(row["TOTAL INVOICE AMOUNT (ORIGINAL CCY)"])
                if "TOTAL INVOICE AMOUNT (ORIGINAL CCY)" in row.index
                else None
            )
            total_net_value_original = (
                parse_european_number(row["TOTAL NET VALUE (ORIGINAL CCY)"])
                if "TOTAL NET VALUE (ORIGINAL CCY)" in row.index
                else None
            )
            discount_percentage = (
                parse_european_number(row["DISCOUNT PERCENTAGE"])
                if "DISCOUNT PERCENTAGE" in row.index
                else None
            )
            funding_currency = (
                str(row["FUNDING CURRENCY"]).strip().upper()
                if "FUNDING CURRENCY" in row.index and pd.notna(row["FUNDING CURRENCY"])
                else None
            )
            exchange_rate = (
                parse_european_number(row["EXCHANGE RATE"])
                if "EXCHANGE RATE" in row.index
                else None
            )
            despatch_date = (
                parse_date(row["DESPATCH DATE"])
                if "DESPATCH DATE" in row.index
                else None
            )
            margin = (
                parse_european_number(row["MARGIN"])
                if "MARGIN" in row.index
                else None
            )

            # Parse dates
            issuance_date = parse_date(row["ISSUANCE DATE"])
            due_date = parse_date(row["DUE DATE"])

            # Parse optional amounts
            amount_original = None
            amount_eur = None
            if amount_original_col and amount_original_col in row.index:
                amount_original = parse_european_number(row[amount_original_col])
            if amount_eur_col and amount_eur_col in row.index:
                amount_eur = parse_european_number(row[amount_eur_col])

            # Validate required fields
            if not all([seller_id, invoice_ref, issuance_date, due_date]):
                errors.append(
                    f"Row {row_num}: Missing required fields "
                    f"(seller_id={seller_id}, invoice_ref={invoice_ref}, "
                    f"issuance_date={issuance_date}, due_date={due_date})"
                )
                continue

            invoice = OfferInvoice(
                programa=programa,
                seller_id=seller_id,
                seller_name=seller_name,
                debtor_id=debtor_id,
                debtor_name=debtor_name,
                insurer_id=insurer_id,
                doc_number=doc_number,
                fiscal_year=fiscal_year,
                reference_number=reference_number,
                invoice_ref=invoice_ref,
                goods_services=goods_services,
                item=item,
                total_invoice_amount_original=total_invoice_amount_original,
                total_net_value_original=total_net_value_original,
                discount_percentage=discount_percentage,
                original_currency=original_currency,
                funding_currency=funding_currency,
                exchange_rate=exchange_rate,
                despatch_date=despatch_date,
                issuance_date=issuance_date,
                due_date=due_date,
                margin=margin,
                amount_original=amount_original,
                amount_eur=amount_eur,
            )
            invoices.append(invoice)

        except Exception as e:
            errors.append(f"Row {row_num}: Error parsing row - {e}")
            continue

    if errors:
        logger.warning(f"Parsing completed with {len(errors)} errors: {errors[:5]}...")

    logger.info(f"Successfully parsed {len(invoices)} invoices from {filename}")
    return invoices
