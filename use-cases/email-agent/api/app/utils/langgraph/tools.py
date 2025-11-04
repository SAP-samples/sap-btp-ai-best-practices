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
from typing import Any, Dict, Optional, Tuple
import re

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
# HTTP helpers for mock APIs
# -----------------------------

_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
_API_KEY = os.getenv("API_KEY")
_HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT_SECONDS", "10"))
_TEMPLATES_DIR = os.getenv(
    "EMAIL_TEMPLATES_DIR",
    os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            "email-templates",
        )
    ),
)


def _http_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if _API_KEY:
        headers["X-API-Key"] = _API_KEY
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


def _paginate_list(
    items: Any, limit: Optional[int] = 50, offset: Optional[int] = 0
) -> Tuple[Any, Dict[str, Any]]:
    """Return a sliced list (if possible) and pagination metadata.

    This limits the amount of data returned to the LLM to avoid blowing context.
    """
    try:
        data_list = list(items)
    except Exception:
        return items, {"returned": 1, "total": 1, "offset": 0, "limit": 1}

    safe_limit = 50 if limit is None else max(1, min(int(limit), 200))
    safe_offset = 0 if offset is None else max(0, int(offset))
    total = len(data_list)
    sliced = data_list[safe_offset : safe_offset + safe_limit]
    meta = {
        "returned": len(sliced),
        "total": total,
        "offset": safe_offset,
        "limit": safe_limit,
        "truncated": total > safe_offset + len(sliced),
    }
    return sliced, meta


# -----------------------------
# Ariba tools
# -----------------------------


@tool("ariba_list_invoices")
async def ariba_list_invoices(
    q: Optional[str] = None,
    supplier: Optional[str] = None,
    status: Optional[str] = None,
    requester: Optional[str] = None,
    matched_order_id: Optional[str] = None,
    limit: Optional[int] = 50,
    offset: Optional[int] = 0,
) -> Dict[str, Any]:
    """List Ariba invoices with optional filters.

    Args:
        q: Generic search across supplier and invoice number
        supplier: Filter by supplier
        status: Filter by status
        requester: Filter by requester
        matched_order_id: Filter by matched order ID

    Returns:
        {"invoices": [ ... ]}
    """
    params: Dict[str, Any] = {}
    if q:
        params["q"] = q
    if supplier:
        params["supplier"] = supplier
    if status:
        params["status"] = status
    if requester:
        params["requester"] = requester
    if matched_order_id:
        params["matchedOrderId"] = matched_order_id
    data = await _http_get_async("/api/ariba/invoices", params=params or None)
    sliced, meta = _paginate_list(data, limit=limit, offset=offset)
    return {"invoices": sliced, "page": meta}


@tool("ariba_get_invoice")
async def ariba_get_invoice(invoice_number: str) -> Dict[str, Any]:
    """Get a single Ariba invoice by invoice number (or id fallback)."""
    if not invoice_number:
        raise ValueError("ariba_get_invoice: 'invoice_number' is required")
    data = await _http_get_async(f"/api/ariba/invoices/{invoice_number}")
    return {"invoice": data}


# -----------------------------
# Vendor List tools
# -----------------------------


@tool("vendor_list_list")
async def vendor_list_list(
    q: Optional[str] = None,
    supplier_name: Optional[str] = None,
    domain: Optional[str] = None,
    limit: Optional[int] = 50,
    offset: Optional[int] = 0,
) -> Dict[str, Any]:
    """List Vendor List entries with optional filters or domain lookup."""
    params: Dict[str, Any] = {}
    if q:
        params["q"] = q
    if supplier_name:
        params["supplierName"] = supplier_name
    if domain:
        params["domain"] = domain
    data = await _http_get_async("/api/vendor-list/", params=params or None)
    sliced, meta = _paginate_list(data, limit=limit, offset=offset)
    return {"vendorList": sliced, "page": meta}


# -----------------------------
# Relish tools
# -----------------------------


@tool("relish_list_records")
async def relish_list_records(
    q: Optional[str] = None,
    supplier_id: Optional[str] = None,
    supplier_name: Optional[str] = None,
    status: Optional[str] = None,
    inv_status: Optional[str] = None,
    company_code: Optional[str] = None,
    invoice_number: Optional[str] = None,
    payment_method: Optional[str] = None,
    limit: Optional[int] = 50,
    offset: Optional[int] = 0,
) -> Dict[str, Any]:
    """List Relish records with optional filters."""
    params: Dict[str, Any] = {}
    if q:
        params["q"] = q
    if supplier_id:
        params["supplierId"] = supplier_id
    if supplier_name:
        params["supplierName"] = supplier_name
    if status:
        params["status"] = status
    if inv_status:
        params["invStatus"] = inv_status
    if company_code:
        params["companyCode"] = company_code
    if invoice_number:
        params["invoiceNumber"] = invoice_number
    if payment_method:
        params["paymentMethod"] = payment_method
    data = await _http_get_async("/api/relish/records", params=params or None)
    sliced, meta = _paginate_list(data, limit=limit, offset=offset)
    return {"relishRecords": sliced, "page": meta}


@tool("relish_get_record")
async def relish_get_record(identifier: str) -> Dict[str, Any]:
    """Get a single Relish record by invoiceNumber, relishId, or uuid."""
    if not identifier:
        raise ValueError("relish_get_record: 'identifier' is required")
    data = await _http_get_async(f"/api/relish/records/{identifier}")
    return {"relishRecord": data}


# -----------------------------
# S/4 tools
# -----------------------------


@tool("s4_list_invoices")
async def s4_list_invoices(
    vendor_id: Optional[str] = None,
    po_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: Optional[int] = 50,
    offset: Optional[int] = 0,
) -> Dict[str, Any]:
    """List Invoices from the S/4 mock with optional filters."""
    params: Dict[str, Any] = {}
    if vendor_id:
        params["vendorId"] = vendor_id
    if po_id:
        params["poId"] = po_id
    if status:
        params["status"] = status
    data = await _http_get_async("/api/s4/invoices", params=params or None)
    sliced, meta = _paginate_list(data, limit=limit, offset=offset)
    return {"invoices": sliced, "page": meta}


@tool("s4_get_invoice")
async def s4_get_invoice(invoice_id: str) -> Dict[str, Any]:
    """Get a single Invoice by id from the S/4 mock."""
    if not invoice_id:
        raise ValueError("s4_get_invoice: 'invoice_id' is required")
    data = await _http_get_async(f"/api/s4/invoices/{invoice_id}")
    return {"invoice": data}


@tool("s4_list_records")
async def s4_list_records(
    q: Optional[str] = None,
    supplier_id: Optional[str] = None,
    supplier_name: Optional[str] = None,
    journal_entry: Optional[str] = None,
    purchasing_doc: Optional[str] = None,
    invoice_number: Optional[str] = None,
    limit: Optional[int] = 50,
    offset: Optional[int] = 0,
) -> Dict[str, Any]:
    """List raw S/4 records with optional filters."""
    params: Dict[str, Any] = {}
    if q:
        params["q"] = q
    if supplier_id:
        params["supplierId"] = supplier_id
    if supplier_name:
        params["supplierName"] = supplier_name
    if journal_entry:
        params["journalEntry"] = journal_entry
    if purchasing_doc:
        params["purchasingDocList"] = purchasing_doc
    if invoice_number:
        params["invoiceNumber"] = invoice_number
    data = await _http_get_async("/api/s4/records", params=params or None)
    sliced, meta = _paginate_list(data, limit=limit, offset=offset)
    return {"records": sliced, "page": meta}


__all__ = [
    "calculator_tool",
    # Ariba
    "ariba_list_invoices",
    "ariba_get_invoice",
    # Vendor List
    "vendor_list_list",
    # Relish
    "relish_list_records",
    "relish_get_record",
    # S/4
    "s4_list_invoices",
    "s4_get_invoice",
    "s4_list_records",
    # Templates
    "email_template_get",
]


# -----------------------------
# Email template tool
# -----------------------------


def _template_path(decision: str, locale: Optional[str]) -> Optional[str]:
    """Resolve template path by slugifying the decision string.

    Example: "ENABLED_REVIEW_IN_SBN" -> "enabled_review_in_sbn.en.json"
    """
    key_raw = (decision or "").strip()
    if not key_raw:
        return None

    # Normalize to a filesystem-friendly slug
    slug = re.sub(r"[^a-z0-9]+", "_", key_raw.lower()).strip("_")

    bases = []
    if slug:
        bases.append(slug)
    # Also try simple lowercase of the raw key as a fallback (often identical)
    lower_key = key_raw.lower()
    if lower_key and lower_key not in bases:
        bases.append(lower_key)

    for base in bases:
        candidate_names = []
        if locale:
            candidate_names.append(f"{base}.{locale}.json")
        candidate_names.append(f"{base}.en.json")
        for name in candidate_names:
            path = os.path.join(_TEMPLATES_DIR, name)
            if os.path.isfile(path):
                return path
    return None


@tool("email_template_get")
def email_template_get(decision: str, locale: Optional[str] = None) -> Dict[str, Any]:
    """Load an email template by decision and optional locale.

    Falls back to English if the requested locale is unavailable.
    Returns keys: decision, locale, subject, body
    """
    import json

    if not decision:
        raise ValueError("email_template_get: 'decision' is required")
    path = _template_path(decision, locale)
    if not path:
        raise FileNotFoundError(
            f"email template not found for decision='{decision}', locale='{locale or 'en'}'"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
