"""MCP-compatible Pharma Procurement Sales Order Agent tool server.

Run mode is stdio because the LangGraph agent starts this server as a local
subprocess. This keeps the prototype self-contained while preserving the
MCP contract shape for future remote tool servers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

api_root = Path(__file__).resolve().parents[2]
if str(api_root) not in sys.path:
    sys.path.insert(0, str(api_root))

from mcp.server.fastmcp import FastMCP

from app.tools.pharma_order.sap_mock_tools import (
    check_duplicate_po as _check_duplicate_po,
    get_invoice_pdf as _get_invoice_pdf,
    get_material_availability as _get_material_availability,
    get_order_status as _get_order_status,
    get_pricing_for_customer_material as _get_pricing_for_customer_material,
    list_blocked_orders as _list_blocked_orders,
    lookup_batch_expiry as _lookup_batch_expiry,
    lookup_customer_by_dea as _lookup_customer_by_dea,
    lookup_customer_recent_orders as _lookup_customer_recent_orders,
    lookup_material_by_ndc as _lookup_material_by_ndc,
    set_or_clear_order_block as _set_or_clear_order_block,
)

mcp = FastMCP("pharma_order_tools")


@mcp.tool()
def get_pricing_for_customer_material(
    customer_name: str = "",
    material_name: str = "",
    quantity: int = 1,
    delivery_date: str = "",
) -> dict[str, Any]:
    """Get customer-specific product pricing and relevant pharma context."""
    return _get_pricing_for_customer_material(customer_name, material_name, quantity, delivery_date)


@mcp.tool()
def get_material_availability(
    material_name: str = "",
    requested_quantity: int = 1,
    requested_date: str = "",
    plant: str = "",
) -> dict[str, Any]:
    """Check material availability, stock, batch, expiry, and cold-chain constraints."""
    return _get_material_availability(material_name, requested_quantity, requested_date, plant)


@mcp.tool()
def get_order_status(sales_order: str = "", customer_name: str = "", po_number: str = "") -> dict[str, Any]:
    """Get sales order status and related header/item/partner context."""
    return _get_order_status(sales_order, customer_name, po_number)


@mcp.tool()
def lookup_customer_by_dea(dea_number: str = "", customer_name: str = "") -> dict[str, Any]:
    """Lookup customer compliance information by DEA number or customer name."""
    return _lookup_customer_by_dea(dea_number, customer_name)


@mcp.tool()
def lookup_customer_recent_orders(
    customer_name: str = "",
    customer_id: str = "",
    dea_number: str = "",
    limit: int = 5,
) -> dict[str, Any]:
    """List recent customer orders."""
    return _lookup_customer_recent_orders(customer_name, customer_id, dea_number, limit)


@mcp.tool()
def lookup_batch_expiry(batch_id: str = "", material_name: str = "") -> dict[str, Any]:
    """Check batch, lot, expiry, recall, quarantine, and quality status."""
    return _lookup_batch_expiry(batch_id, material_name)


@mcp.tool()
def lookup_material_by_ndc(ndc: str = "", material_name: str = "") -> dict[str, Any]:
    """Map NDC/product names to SAP material and pharma catalog data."""
    return _lookup_material_by_ndc(ndc, material_name)


@mcp.tool()
def check_duplicate_po(customer_name: str = "", customer_id: str = "", po_number: str = "") -> dict[str, Any]:
    """Check if a PO number already exists for a customer."""
    return _check_duplicate_po(customer_name, customer_id, po_number)


@mcp.tool()
def list_blocked_orders(customer_name: str = "", block_reason: str = "") -> dict[str, Any]:
    """List blocked sales orders by customer or block reason."""
    return _list_blocked_orders(customer_name, block_reason)


@mcp.tool()
def set_or_clear_order_block(
    sales_order: str = "",
    block_reason: str = "",
    action: str = "preview",
    note: str = "",
) -> dict[str, Any]:
    """Preview a sales-order block or unblock action without mutating SAP data."""
    return _set_or_clear_order_block(sales_order, block_reason, action, note)


@mcp.tool()
def get_invoice_pdf(invoice_id: str = "", sales_order: str = "", customer_name: str = "") -> dict[str, Any]:
    """Find invoice PDF metadata for a billing document or sales order."""
    return _get_invoice_pdf(invoice_id, sales_order, customer_name)


if __name__ == "__main__":
    mcp.run()
