"""Advice references: one normalization, conservative variants and the reviewer's document pick.

Pure functions, no S/4 access. Everything that turns the text a customer
printed on a remittance into something S/4 can be asked for lives here, so
lookup, validation and payload agree on one reading:

- `line_reference` decides which reference of an advice line is looked up.
- `is_document_number` tells whether a reference can be an FI or billing
  document number (ASCII digits only, at most 10).
- `reference_variants` proposes spellings to try when the exact reference is
  unknown in S/4 (spaces or separators removed, leading zeros stripped or
  added). A variant never replaces the exact reference silently: validation
  applies it only when the amount confirms it and always raises a warning.
- `parse_pick` / `pick_of` read and write the reviewer's explicit choice of an
  S/4 document, stored on the advice row as `s4_document`.

Example:
    line_reference({"invoice_reference": " inv-0012 "})   -> ("INV-0012", "invoice_reference")
    reference_variants("INV-0012")                         -> [("without separators", "INV0012")]
    parse_pick("1000/1800000012/2026/002")                 -> {"company_code": "1000", ...}
"""
from __future__ import annotations

import re
from typing import Any

XBLNR_LENGTH = 16
DOCUMENT_NUMBER_LENGTH = 10
# Shorter variants match unrelated documents too easily to be worth proposing.
MIN_VARIANT_LENGTH = 4
PICK_PATTERN = re.compile(r"^([A-Z0-9]{1,4})/([A-Z0-9]{1,10})/(\d{4})(?:/(\d{1,3}))?$")


def normalize_reference(value: Any) -> str:
    """Normalize an advice reference the way S/4 stores XBLNR: trimmed and upper case."""
    return str(value or "").strip().upper()


def line_reference(row: dict[str, Any]) -> tuple[str, str]:
    """Return the reference of an advice line and the field it came from.

    The invoice reference is used when present. Only when it is empty does the
    customer's own document reference take its place, so a line whose only
    reference was extracted into the other column is still looked up.

    Args:
        row: Advice line.

    Returns:
        `(reference, source)` with `source` either `invoice_reference` or
        `customer_document_reference`; the reference is "" when the line has none.
    """
    reference = normalize_reference(row.get("invoice_reference"))
    if reference:
        return reference, "invoice_reference"
    other = normalize_reference(row.get("customer_document_reference"))
    return (other, "customer_document_reference") if other else ("", "invoice_reference")


def is_document_number(reference: str) -> bool:
    """Return True when the reference can be an S/4 document number: ASCII digits only, at most 10."""
    return bool(reference) and reference.isascii() and reference.isdigit() and len(reference) <= DOCUMENT_NUMBER_LENGTH


def pad_document_number(reference: str) -> str:
    """Zero-pad a document number to the 10 digits S/4 stores."""
    return reference.zfill(DOCUMENT_NUMBER_LENGTH)


def reference_variants(reference: str) -> list[tuple[str, str]]:
    """Return alternative spellings of a reference that is unknown in S/4, most conservative first.

    Args:
        reference: Normalized advice reference.

    Returns:
        List of `(label, variant)`; the original and duplicates are left out, and
        only variants of 4 to 16 characters are proposed.
    """
    compact = re.sub(r"\s+", "", reference)
    bare = re.sub(r"[\s\-/._]", "", reference)
    proposals = [("without spaces", compact), ("without separators", bare),
                 ("without leading zeros", compact.lstrip("0")), ("without leading zeros", bare.lstrip("0"))]
    # A short document number may be stored zero-padded in the reference field.
    proposals += [("zero-padded", pad_document_number(value)) for value in (compact, bare) if is_document_number(value)]
    variants: list[tuple[str, str]] = []
    for label, value in proposals:
        if value != reference and MIN_VARIANT_LENGTH <= len(value) <= XBLNR_LENGTH and value not in [v for _, v in variants]:
            variants.append((label, value))
    return variants


def parse_pick(text: Any) -> dict[str, str] | None:
    """Parse a reviewer's pick `COMPANYCODE/DOCUMENT/YEAR[/ITEM]`.

    Args:
        text: Value of the row field `s4_document`.

    Returns:
        `{company_code, accounting_document (10 digits when numeric), fiscal_year, item (3 digits or "")}`,
        or None when the text is empty or malformed.
    """
    match = PICK_PATTERN.match(str(text or "").strip().upper())
    if not match:
        return None
    company_code, document, year, item = match.groups()
    return {"company_code": company_code, "accounting_document": document.zfill(10) if document.isdigit() else document,
            "fiscal_year": year, "item": item.zfill(3) if item else ""}


def pick_of(company_code: str, accounting_document: str, fiscal_year: str, item: str = "") -> str:
    """Format the pick text for one S/4 item (the inverse of `parse_pick`)."""
    return "/".join(part for part in (company_code, accounting_document, fiscal_year, item) if part)
