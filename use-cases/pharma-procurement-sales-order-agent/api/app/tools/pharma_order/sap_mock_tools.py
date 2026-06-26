"""Pharma Procurement Sales Order Agent SAP-facing tool implementations backed by synthetic JSON data.

The functions mimic SAP API tools without calling a live S/4HANA system.
They are side-effect free so the same functions can be exposed through
LangGraph, MCP, FastAPI tests, and later replaced by real SAP clients.
"""

from __future__ import annotations

from typing import Any

from .data_store import search_many, search_records


def _response(
    tool: str,
    purpose: str,
    data: dict[str, Any],
    assumptions: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "tool": tool,
        "purpose": purpose,
        "data_status": "synthetic_demo_data",
        "assumptions": assumptions or [],
        "data": data,
    }


def get_pricing_for_customer_material(
    customer_name: str = "",
    material_name: str = "",
    quantity: int = 1,
    delivery_date: str = "",
) -> dict[str, Any]:
    data = search_many(
        ["pricing", "materials", "customer_compliance"],
        customer_name,
        material_name,
        quantity,
        delivery_date,
        limit_per_dataset=3,
    )
    return _response(
        "get_pricing_for_customer_material",
        "Simulate customer/material price lookup similar to sales order simulation and list price APIs.",
        data,
        ["No live S/4HANA pricing procedure is executed in this prototype."],
    )


def get_material_availability(
    material_name: str = "",
    requested_quantity: int = 1,
    requested_date: str = "",
    plant: str = "",
) -> dict[str, Any]:
    data = search_many(
        ["stock", "batches", "materials"],
        material_name,
        requested_quantity,
        requested_date,
        plant,
        limit_per_dataset=4,
    )
    return _response(
        "get_material_availability",
        "Check ATP-like stock, batch, expiry, cold-chain, and allocation data.",
        data,
    )


def get_order_status(
    sales_order: str = "", customer_name: str = "", po_number: str = ""
) -> dict[str, Any]:
    data = search_records("sales_orders", sales_order, customer_name, po_number, limit=6)
    return _response(
        "get_order_status",
        "Retrieve sales order status across header, items, partners, pricing elements, schedule lines, and text.",
        data,
    )


def lookup_customer_by_dea(dea_number: str = "", customer_name: str = "") -> dict[str, Any]:
    data = search_records("customer_compliance", dea_number, customer_name, limit=5)
    return _response(
        "lookup_customer_by_dea",
        "Lookup customer DEA, GTS, ship-to, sold-to, and compliance attributes.",
        data,
    )


def lookup_customer_recent_orders(
    customer_name: str = "", customer_id: str = "", dea_number: str = "", limit: int = 5
) -> dict[str, Any]:
    data = search_records(
        "sales_orders", customer_name, customer_id, dea_number, limit=max(1, min(limit, 10))
    )
    return _response(
        "lookup_customer_recent_orders",
        "Find recent customer orders and summarize fulfillment status.",
        data,
    )


def lookup_batch_expiry(batch_id: str = "", material_name: str = "") -> dict[str, Any]:
    data = search_records("batches", batch_id, material_name, limit=5)
    return _response(
        "lookup_batch_expiry",
        "Check batch, lot, expiry, quarantine, recall, and DSCSA-related attributes.",
        data,
    )


def lookup_material_by_ndc(ndc: str = "", material_name: str = "") -> dict[str, Any]:
    data = search_records("materials", ndc, material_name, limit=5)
    return _response(
        "lookup_material_by_ndc",
        "Map NDC or product name to material, GTIN, dosage, package, and pharma attributes.",
        data,
    )


def check_duplicate_po(
    customer_name: str = "", customer_id: str = "", po_number: str = ""
) -> dict[str, Any]:
    data = search_records("sales_orders", customer_name, customer_id, po_number, limit=10)
    return _response(
        "check_duplicate_po",
        "Detect potentially duplicated customer purchase orders before creating a new order.",
        data,
        ["This prototype only checks the local synthetic order history."],
    )


def list_blocked_orders(customer_name: str = "", block_reason: str = "") -> dict[str, Any]:
    data = search_records("sales_orders", customer_name, block_reason, "blocked block hold", limit=10)
    return _response(
        "list_blocked_orders",
        "List blocked sales orders and related block reasons.",
        data,
    )


def set_or_clear_order_block(
    sales_order: str = "",
    block_reason: str = "",
    action: str = "preview",
    note: str = "",
) -> dict[str, Any]:
    order_data = search_records("sales_orders", sales_order, block_reason, limit=3)
    return _response(
        "set_or_clear_order_block",
        "Preview a block/unblock operation. No SAP update is performed in the prototype.",
        {
            "requested_action": action,
            "requested_note": note,
            "matched_order_context": order_data,
            "execution_mode": "preview_only_no_update",
        },
        [
            "Write-back is intentionally disabled until a productive SAP integration and authorization model are agreed."
        ],
    )


def get_invoice_pdf(
    invoice_id: str = "", sales_order: str = "", customer_name: str = ""
) -> dict[str, Any]:
    data = search_records("invoices", invoice_id, sales_order, customer_name, limit=5)
    return _response(
        "get_invoice_pdf",
        "Find billing document PDF metadata. Prototype returns metadata only, not binary content.",
        data,
    )
