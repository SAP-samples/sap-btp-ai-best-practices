"""Canonical quote extraction schema used by prompts and normalization."""

from __future__ import annotations

import json
from typing import Any


HEADER_FIELDS = [
    "document_type",
    "vendor_name",
    "vendor_address",
    "vendor_phone",
    "vendor_email",
    "vendor_tax_id",
    "quote_number",
    "quote_date",
    "quote_expiration_date",
    "customer_name",
    "customer_number",
    "sold_to_name",
    "sold_to_address",
    "ship_to_name",
    "ship_to_address",
    "requester_name",
    "caller_name",
    "buyer_contact",
    "payment_terms",
    "incoterms",
    "freight_terms",
    "carrier",
    "currency",
    "subtotal_amount",
    "tax_amount",
    "freight_amount",
    "shipping_amount",
    "total_amount",
    "notes",
]


LINE_ITEM_FIELDS = [
    "line_number",
    "vendor_material_number",
    "manufacturer",
    "manufacturer_part_number",
    "description",
    "quantity",
    "unit_of_measure",
    "unit_price",
    "line_total",
    "expected_delivery_date",
    "origin_country",
    "carrier",
    "service_or_material",
]


def empty_quote_schema() -> dict[str, Any]:
    """Return the canonical JSON shape expected from extraction prompts."""

    return {
        "document_type": "vendor_quote",
        "header": {field: None for field in HEADER_FIELDS if field != "document_type"},
        "line_items": [{field: None for field in LINE_ITEM_FIELDS}],
        "pr_mapping": {
            "pr_exists_check_key": None,
            "sap_supplier_id": None,
            "sap_material_id": None,
            "plant": None,
            "purchasing_org": None,
            "purchasing_group": None,
            "account_assignment_category": None,
            "delivery_address": None,
            "requested_by": None,
            "need_by_date": None,
        },
        "evidence": {},
        "warnings": [],
    }


def schema_as_pretty_json() -> str:
    """Return the canonical schema as JSON text for prompts."""

    return json.dumps(empty_quote_schema(), indent=2, ensure_ascii=False)


def normalize_quote_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Coerce model output into the canonical top-level shape."""

    canonical = empty_quote_schema()
    if not isinstance(data, dict):
        return canonical

    if data.get("document_type"):
        canonical["document_type"] = data.get("document_type")

    source_header = data.get("header") if isinstance(data.get("header"), dict) else data
    for field in canonical["header"]:
        if field in source_header:
            canonical["header"][field] = source_header.get(field)

    line_items = data.get("line_items")
    if isinstance(line_items, list):
        canonical_items = []
        for item in line_items:
            if not isinstance(item, dict):
                continue
            row = {field: item.get(field) for field in LINE_ITEM_FIELDS}
            canonical_items.append(row)
        canonical["line_items"] = canonical_items

    pr_mapping = data.get("pr_mapping")
    if isinstance(pr_mapping, dict):
        for field in canonical["pr_mapping"]:
            if field in pr_mapping:
                canonical["pr_mapping"][field] = pr_mapping.get(field)

    evidence = data.get("evidence")
    if isinstance(evidence, dict):
        canonical["evidence"] = evidence

    warnings = data.get("warnings")
    if isinstance(warnings, list):
        canonical["warnings"] = warnings

    return canonical
