"""
LangChain-native tools for the API's LangGraph chatbot.

Implements one tool mirroring the tutorial examples:
- calculator: safely evaluates arithmetic expressions
"""

from __future__ import annotations

import ast
import time
import os
import httpx
from typing import Any, Dict, Optional, List

from langchain_core.tools import tool


def _safe_eval_arithmetic(expression: str) -> float:
    """Safely evaluate a basic arithmetic expression using Python's AST."""
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Num,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Constant,
        ast.Load,
        ast.Mod,
        ast.FloorDiv,
    )

    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError("Disallowed expression component")

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant):
            return float(n.value)
        if isinstance(n, ast.Num):
            return float(n.n)
        if isinstance(n, ast.BinOp):
            left = _eval(n.left)
            right = _eval(n.right)
            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
            if isinstance(n.op, ast.Div):
                return left / right
            if isinstance(n.op, ast.Pow):
                return left**right
            if isinstance(n.op, ast.Mod):
                return left % right
            if isinstance(n.op, ast.FloorDiv):
                return left // right
            raise ValueError("Unsupported operator")
        if isinstance(n, ast.UnaryOp):
            operand = _eval(n.operand)
            if isinstance(n.op, ast.UAdd):
                return +operand
            if isinstance(n.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary operator")
        raise ValueError("Unsupported expression type")

    return _eval(tree)


@tool("calculator")
def calculator_tool(expression: str) -> Dict[str, Any]:
    """Evaluate an arithmetic `expression` like `2*(3+4)` and return the result."""
    expr = str(expression).strip()
    if not expr:
        raise ValueError("calculator: 'expression' is required")
    value = _safe_eval_arithmetic(expr)
    # Add a wait/delay
    time.sleep(1)
    return {"expression": expr, "result": value}


# -----------------------------
# S/4 tools (HTTP to mock endpoints)
# -----------------------------


_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
_API_KEY_FOR_MOCK = os.getenv("API_KEY")
_HTTP_TIMEOUT = float(os.getenv("S4_TIMEOUT_SECONDS", "10"))


def _http_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if _API_KEY_FOR_MOCK:
        headers["X-API-Key"] = _API_KEY_FOR_MOCK
    return headers


async def _http_get_async(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """Async variant of _http_get using httpx.AsyncClient.

    Keep the sync version for existing tools; new async tools can await this.
    """
    url = _API_BASE_URL.rstrip("/") + path
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        resp = await client.get(url, params=params or {}, headers=_http_headers())
        resp.raise_for_status()
        return resp.json()


@tool("s4_list_business_partners")
async def s4_list_business_partners(q: Optional[str] = None) -> Dict[str, Any]:
    """List Business Partners from the S/4 mock.

    Args:
        q: Optional case-insensitive substring to filter by partner name.

    Returns:
        {"businessPartners": [ ... ]}
    """
    data = await _http_get_async(
        "/api/s4/business-partners", params={"q": q} if q else None
    )
    # Endpoint returns a list of partners
    return {"businessPartners": data}


@tool("s4_get_business_partner")
async def s4_get_business_partner(bp_id: str) -> Dict[str, Any]:
    """Get a single Business Partner by id from the S/4 mock.

    Args:
        bp_id: Business Partner id, e.g. "BP1000001"

    Returns:
        {"businessPartner": { ... }} if found, else raises ValueError
    """
    if not bp_id:
        raise ValueError("s4_get_business_partner: 'bp_id' is required")
    data = await _http_get_async(f"/api/s4/business-partners/{bp_id}")
    return {"businessPartner": data}


@tool("s4_list_purchase_orders")
async def s4_list_purchase_orders(
    vendor_id: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """List Purchase Orders from the S/4 mock with optional filters.

    Args:
        vendor_id: Filter by vendorId (Business Partner id)
        status: Filter by status (case-insensitive)

    Returns:
        {"purchaseOrders": [ ... ]}
    """
    params: Dict[str, Any] = {}
    if vendor_id:
        params["vendorId"] = vendor_id
    if status:
        params["status"] = status
    data = await _http_get_async("/api/s4/purchase-orders", params=params or None)
    return {"purchaseOrders": data}


@tool("s4_get_purchase_order")
async def s4_get_purchase_order(po_id: str) -> Dict[str, Any]:
    """Get a single Purchase Order by id from the S/4 mock.

    Args:
        po_id: Purchase Order id, e.g. "4500000001"

    Returns:
        {"purchaseOrder": { ... }} if found, else raises ValueError
    """
    if not po_id:
        raise ValueError("s4_get_purchase_order: 'po_id' is required")
    data = await _http_get_async(f"/api/s4/purchase-orders/{po_id}")
    return {"purchaseOrder": data}


@tool("s4_list_invoices")
async def s4_list_invoices(
    vendor_id: Optional[str] = None,
    po_id: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """List Invoices from the S/4 mock with optional filters.

    Args:
        vendor_id: Filter by vendorId (Business Partner id)
        po_id: Filter by Purchase Order id
        status: Filter by status (case-insensitive)

    Returns:
        {"invoices": [ ... ]}
    """
    params: Dict[str, Any] = {}
    if vendor_id:
        params["vendorId"] = vendor_id
    if po_id:
        params["poId"] = po_id
    if status:
        params["status"] = status
    data = await _http_get_async("/api/s4/invoices", params=params or None)
    return {"invoices": data}


@tool("s4_get_invoice")
async def s4_get_invoice(invoice_id: str) -> Dict[str, Any]:
    """Get a single Invoice by id from the S/4 mock.

    Args:
        invoice_id: Invoice id, e.g. "5100000001"

    Returns:
        {"invoice": { ... }} if found, else raises ValueError
    """
    if not invoice_id:
        raise ValueError("s4_get_invoice: 'invoice_id' is required")
    data = await _http_get_async(f"/api/s4/invoices/{invoice_id}")
    return {"invoice": data}


__all__ = [
    "calculator_tool",
    "s4_list_business_partners",
    "s4_get_business_partner",
    "s4_list_purchase_orders",
    "s4_get_purchase_order",
    "s4_list_invoices",
    "s4_get_invoice",
]
