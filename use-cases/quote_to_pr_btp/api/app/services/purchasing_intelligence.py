"""Vendor intelligence and repeatable material-proposal workflows.

The default dataset is intentionally marked as demo data. The response shape
is also the contract for the future HANA Cloud repository, so the UI does not
need to change when customer tables become available.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


DEMO_INTELLIGENCE: dict[str, dict[str, Any]] = {
    "sample_facilities_quote.pdf": {
        "vendor": {
            "name": "NORTHWIND INDUSTRIAL SUPPLY",
            "supplier_id": "1000001",
            "preferred_status": "preferred",
            "preferred_label": "Preferred supplier",
            "reason": "Approved source for facilities and door hardware",
        },
        "decision": {
            "status": "ready_after_validation",
            "title": "Ready after validation",
            "message": "Supplier history and material candidates are available. Validate the proposed values before creating the PR.",
            "material_review_count": 0,
            "supplier_review_required": False,
        },
        "purchase_orders": [
            {"purchase_order": "4500000101", "item_description": "Aluminum push plate", "material": "DEMO-MAT-001", "net_unit_price": 18.95, "currency": "USD", "quantity": 12, "unit": "EA", "creation_date": "2026-02-10"},
            {"purchase_order": "4500000102", "item_description": "Oval pull plate", "material": "DEMO-MAT-002", "net_unit_price": 53.80, "currency": "USD", "quantity": 6, "unit": "EA", "creation_date": "2026-01-18"},
            {"purchase_order": "4500000103", "item_description": "Hydraulic door closer", "material": "DEMO-MAT-003", "net_unit_price": 335.00, "currency": "USD", "quantity": 4, "unit": "EA", "creation_date": "2025-11-22"},
        ],
        "purchase_requisitions": [
            {"purchase_requisition": "10000010", "status": "In approval", "creation_date": "2026-07-07"},
            {"purchase_requisition": "10000009", "status": "Converted to PO", "creation_date": "2026-05-21"},
        ],
    },
    "sample_service_quote.pdf": {
        "vendor": {
            "name": "CONTOSO MACHINE SERVICES",
            "supplier_id": "1000002",
            "preferred_status": "non_preferred",
            "preferred_label": "Non-preferred supplier",
            "reason": "Supplier is available but is not preferred for this commodity",
        },
        "decision": {
            "status": "material_review",
            "title": "Review 1 material match",
            "message": "A possible SAP material was found for the pump repair line. Confirm it or prepare a material request for master-data approval.",
            "material_review_count": 1,
            "supplier_review_required": False,
        },
        "purchase_orders": [
            {"purchase_order": "4500000104", "item_description": "Small water pump inspection and shaft repair", "material": "", "net_unit_price": 1240.00, "currency": "USD", "quantity": 1, "unit": "AU", "creation_date": "2025-10-18"},
        ],
        "purchase_requisitions": [
            {"purchase_requisition": "10000011", "status": "Created", "creation_date": "2026-07-07"},
        ],
    },
    "sample_new_supplier_quote.pdf": {
        "vendor": {
            "name": "ALPINE PIPE SUPPLY",
            "supplier_id": "",
            "preferred_status": "unknown",
            "preferred_label": "Supplier not found",
            "reason": "No reliable S/4 business-partner match is available",
        },
        "decision": {
            "status": "supplier_and_material_review",
            "title": "Review supplier and 9 materials",
            "message": "Create the PR without a fixed supplier if appropriate, and send a supplier-onboarding request to the procurement back office.",
            "material_review_count": 9,
            "supplier_review_required": True,
        },
        "purchase_orders": [],
        "purchase_requisitions": [],
    },
}


def _normalized(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", " ", str(value or "").upper()).strip()


def _stable_id(prefix: str, *values: Any, length: int = 12) -> str:
    source = "|".join(_normalized(value) for value in values)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest().upper()
    return f"{prefix}{digest[:length]}"


def purchasing_intelligence(normalized: dict[str, Any], document_name: str) -> dict[str, Any]:
    """Return the UI contract, currently backed by traceable demo records."""

    header = normalized.get("header") if isinstance(normalized.get("header"), dict) else {}
    vendor_name = str(header.get("vendor_name") or "").strip()
    scenario = DEMO_INTELLIGENCE.get(document_name)
    if scenario is None:
        scenario = {
            "vendor": {
                "name": vendor_name or "Unknown supplier",
                "supplier_id": "",
                "preferred_status": "unknown",
                "preferred_label": "Status unavailable",
                "reason": "No purchasing-intelligence record is available",
            },
            "decision": {
                "status": "review",
                "title": "Review required",
                "message": "Purchasing history is unavailable for this document.",
                "material_review_count": len(normalized.get("line_items") or []),
                "supplier_review_required": True,
            },
            "purchase_orders": [],
            "purchase_requisitions": [],
        }
    result = json.loads(json.dumps(scenario))
    result["source"] = {
        "type": "demo",
        "label": "Purchasing insights",
        "hana_ready": True,
        "notice": "",
    }
    result["document"] = document_name
    result["generated_at"] = datetime.now().isoformat(timespec="seconds")
    return result


def create_material_proposal(
    normalized: dict[str, Any],
    document_name: str,
    line_index: int,
    store_root: Path,
) -> dict[str, Any]:
    """Create or reopen one deterministic proposal without changing S/4."""

    line_items = normalized.get("line_items") if isinstance(normalized.get("line_items"), list) else []
    if line_index < 0 or line_index >= len(line_items):
        raise IndexError("Line item does not exist")
    item = line_items[line_index] if isinstance(line_items[line_index], dict) else {}
    header = normalized.get("header") if isinstance(normalized.get("header"), dict) else {}
    vendor_name = str(header.get("vendor_name") or "Unknown supplier")
    description = str(item.get("description") or item.get("service_description") or f"Quote item {line_index + 1}")
    part_number = str(item.get("manufacturer_part_number") or item.get("vendor_material_number") or "")
    proposal_id = _stable_id("MP-", vendor_name, description, part_number, length=10)
    suggested_product = _stable_id("Q2PR", vendor_name, description, part_number, length=10)
    folder = store_root / "material_proposals"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{proposal_id}.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        return {
            **existing,
            "reused": True,
            "message": "Existing material request reopened. It is awaiting master-data approval.",
        }

    proposal = {
        "proposal_id": proposal_id,
        "status": "draft",
        "document": document_name,
        "line_index": line_index,
        "vendor_name": vendor_name,
        "suggested_product_id": suggested_product,
        "description": description[:80],
        "long_description": description,
        "manufacturer_part_number": part_number,
        "base_unit": str(item.get("unit_of_measure") or "PC"),
        "material_group": "DEMO001",
        "plant": "1000",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "s4_material_created": False,
        "reused": False,
        "message": "Material request prepared for master-data approval. No SAP material has been created yet.",
    }
    path.write_text(json.dumps(proposal, indent=2), encoding="utf-8")
    return proposal


def list_material_proposals(store_root: Path, document_name: str | None = None) -> list[dict[str, Any]]:
    """Return persisted draft proposals without presenting them as S/4 materials."""

    folder = store_root / "material_proposals"
    if not folder.exists():
        return []
    proposals: list[dict[str, Any]] = []
    for path in folder.glob("MP-*.json"):
        try:
            proposal = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(proposal, dict):
            continue
        if document_name and proposal.get("document") != document_name:
            continue
        proposals.append(proposal)
    return sorted(
        proposals,
        key=lambda item: (str(item.get("created_at") or ""), str(item.get("proposal_id") or "")),
        reverse=True,
    )


def create_back_office_referral(
    normalized: dict[str, Any],
    document_name: str,
    store_root: Path,
) -> dict[str, Any]:
    """Create or reopen an idempotent supplier-onboarding referral."""

    header = normalized.get("header") if isinstance(normalized.get("header"), dict) else {}
    vendor_name = str(header.get("vendor_name") or "Unknown supplier")
    referral_id = _stable_id("BO-", vendor_name, document_name, length=10)
    folder = store_root / "back_office_referrals"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{referral_id}.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        return {**existing, "reused": True, "message": "Existing supplier-onboarding referral reopened."}
    referral = {
        "referral_id": referral_id,
        "status": "submitted",
        "workflow": "supplier_onboarding",
        "vendor_name": vendor_name,
        "document": document_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "reused": False,
        "message": "Supplier onboarding was sent to the procurement back office.",
    }
    path.write_text(json.dumps(referral, indent=2), encoding="utf-8")
    return referral
