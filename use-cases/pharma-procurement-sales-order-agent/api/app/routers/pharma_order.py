"""FastAPI router for the Pharma Procurement Sales Order Agent agent."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.agents.pharma_order.graph import run_pharma_order_agent
from app.security import get_api_key

router = APIRouter()

PHARMA_ORDER_CAPABILITIES: list[dict[str, Any]] = [
    {
        "id": "pricing",
        "title": "Customer-specific pricing",
        "description": "Answer product pricing questions using customer, material, contract, and simulated sales-order pricing context.",
        "expected_tools": ["get_pricing_for_customer_material", "lookup_material_by_ndc", "lookup_customer_by_dea"],
        "source_structures": [
            "API_SALES_ORDER_SIMULATION_SRV",
            "ZAPI_DEL_LIST_PRICE_V4",
            "ZSD_EXTERNAL_INFO",
        ],
        "example_questions": [
            "What is the price for Northstar for Glycemor 10 mg?",
            "For Northstar, identify Glycemor 10 mg by NDC 90000-0100-30 and confirm the current net price.",
        ],
    },
    {
        "id": "availability",
        "title": "Availability and batch readiness",
        "description": "Check whether a product can be shipped by combining material stock, batch, expiry, allocation, and cold-chain context.",
        "expected_tools": ["get_material_availability", "lookup_batch_expiry", "lookup_material_by_ndc"],
        "source_structures": [
            "API_MATERIAL_STOCK_SRV",
            "API_BATCH_SRV",
            "ZAPI_DEL_LIST_PRICE_V4",
        ],
        "example_questions": [
            "Is Glycemor 10 mg available for shipment this week?",
            "For NDC 90000-0100-30, find the SAP material and tell me whether a usable batch is available this week.",
        ],
    },
    {
        "id": "sales_order_status",
        "title": "Sales order status",
        "description": "Summarize order status across sales order header, items, partners, pricing, schedule lines, and text.",
        "expected_tools": ["get_order_status", "lookup_customer_recent_orders"],
        "source_structures": [
            "API_SALES_ORDER_SRV/A_SalesOrder",
            "API_SALES_ORDER_SRV/A_SalesOrderItem",
            "API_SALES_ORDER_SRV/A_SalesOrderPartner",
            "API_SALES_ORDER_SRV/A_SalesOrderPrcgElmnt",
            "API_SALES_ORDER_SRV/A_SalesOrderScheduleLine",
            "API_SALES_ORDER_SRV/A_SalesOrderText",
            "API_SALES_ORDER_SRV/A_SalesOrderHdrBillPlan",
        ],
        "example_questions": [
            "What is the status of sales order 50214568?",
            "For Northstar, find the recent order for Glycemor and explain whether anything blocks shipment.",
        ],
    },
    {
        "id": "customer_compliance",
        "title": "Customer compliance lookup",
        "description": "Resolve customer identity and compliance context such as DEA, GTS, sold-to, and ship-to eligibility.",
        "expected_tools": ["lookup_customer_by_dea", "lookup_customer_recent_orders"],
        "source_structures": ["ZSD_EXTERNAL_INFO", "API_SALES_ORDER_SRV"],
        "example_questions": [
            "Is DEA number BC1234567 valid for Northstar?",
            "Identify Northstar by DEA number and show the latest relevant orders.",
        ],
    },
    {
        "id": "blocks_and_duplicate_po",
        "title": "Blocks and duplicate PO checks",
        "description": "Detect duplicate purchase orders and review blocked orders. Write-like changes remain preview-only in the prototype.",
        "expected_tools": ["check_duplicate_po", "list_blocked_orders", "set_or_clear_order_block"],
        "source_structures": ["API_SALES_ORDER_SRV"],
        "example_questions": [
            "Did we already receive PO PO-100456?",
            "Which MetroMed Wholesale orders are blocked and what would be required to clear the block?",
        ],
    },
    {
        "id": "invoice_pdf",
        "title": "Invoice PDF metadata",
        "description": "Find billing document and invoice PDF metadata for a sales order or customer.",
        "expected_tools": ["get_invoice_pdf", "get_order_status"],
        "source_structures": ["API_BILLING_DOCUMENT_SRV", "API_SALES_ORDER_SRV"],
        "example_questions": [
            "Can I get the invoice PDF for order 50214568?",
            "For the latest Northstar Glycemor order, find the invoice PDF metadata and summarize the order status.",
        ],
    },
    {
        "id": "complex_resolution_chain",
        "title": "Complex multi-step service representative question",
        "description": "Resolve names and identifiers first, then chain multiple tools to answer pricing, availability, order, compliance, or invoice questions.",
        "expected_tools": [
            "lookup_material_by_ndc",
            "lookup_customer_by_dea",
            "get_pricing_for_customer_material",
            "get_material_availability",
            "get_order_status",
            "get_invoice_pdf",
        ],
        "source_structures": [
            "ZAPI_DEL_LIST_PRICE_V4",
            "ZSD_EXTERNAL_INFO",
            "API_SALES_ORDER_SIMULATION_SRV",
            "API_MATERIAL_STOCK_SRV",
            "API_BATCH_SRV",
            "API_SALES_ORDER_SRV",
            "API_BILLING_DOCUMENT_SRV",
        ],
        "example_questions": [
            "For Northstar, identify Glycemor 10 mg by NDC 90000-0100-30, confirm the price, and tell me whether it can ship this week.",
            "For the latest Northstar Glycemor order, explain status, shipment risk, and invoice PDF availability.",
        ],
    },
]


class PharmaOrderRequest(BaseModel):
    question: str = Field(..., min_length=1, description="service representative question to answer")
    provider: str | None = Field(default=None, description="Optional LLM provider override")
    model: str | None = Field(default=None, description="Optional model override")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=900, ge=128, le=4000)
    prompt_variant: str = Field(default="joule")
    include_trace: bool = Field(default=False)


class PharmaOrderResponse(BaseModel):
    success: bool
    answer: str
    markdown: str
    model: str
    provider: str
    query: str
    tool_call_count: int = 0
    usage: dict[str, Any] = Field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


@router.get("/health")
async def pharma_order_health() -> dict[str, str]:
    return {"status": "ok", "agent": "pharma-order"}


@router.get("/capabilities")
async def pharma_order_capabilities() -> dict[str, Any]:
    return {
        "agent": "pharma-order",
        "mode": "test-ui",
        "ui_note": "This is a local test UI. The target user experience is Joule / Pro-Code, while this page is used to validate the backend agent and tool orchestration.",
        "target_story": "The agent should resolve names and identifiers first, chain the required tools, consolidate evidence, and answer complex pharmaceutical order support questions without exposing raw SAP payloads.",
        "default_question": "What is the price for Northstar for Glycemor 10 mg?",
        "capabilities": PHARMA_ORDER_CAPABILITIES,
    }


@router.post("", response_model=PharmaOrderResponse)
@router.post("/ask", response_model=PharmaOrderResponse)
async def ask_pharma_order(payload: PharmaOrderRequest, http_request: Request = None, api_key: str = Depends(get_api_key)) -> PharmaOrderResponse:
    try:
        correlation_id = None
        client_host = None
        if http_request is not None:
            correlation_id = http_request.headers.get("x-correlationid") or http_request.headers.get("x-correlation-id")
            client_host = http_request.headers.get("user-agent") or (http_request.client.host if http_request.client else None)
        result = await run_pharma_order_agent(
            question=payload.question,
            provider=payload.provider,
            model_name=payload.model,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
            prompt_variant=payload.prompt_variant,
            include_trace=payload.include_trace,
            route="/api/pharma-order/ask",
            method="POST",
            correlation_id=correlation_id,
            client_host=client_host,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    return PharmaOrderResponse(
        success=True,
        answer=result.get("answer", ""),
        markdown=result.get("markdown", result.get("answer", "")),
        model=result.get("model", payload.model or ""),
        provider=result.get("provider", payload.provider or ""),
        query=payload.question,
        tool_call_count=int(result.get("tool_call_count", 0)),
        usage=result.get("usage", {}),
        tool_calls=result.get("tool_calls", []),
        tool_results=result.get("tool_results", []),
    )




