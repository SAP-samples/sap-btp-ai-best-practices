"""
Dynamic SAP Search Routes

GET /api/customers/search?q=   — search Business Partners via OData contains()
GET /api/materials/search?q=   — search Materials via OData exact + contains()

Credentials: read from X-SAP-* request headers (sessionStorage) or .env fallback.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel

from matching.customer_api_matcher import search_customer_odata
from matching.product_api_matcher import search_material_odata

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["SAP Search"])


class CustomerSearchResult(BaseModel):
    business_partner: str
    customer_name: str
    score: float
    confidence: str


class CustomerSearchResponse(BaseModel):
    query: str
    count: int
    results: list[CustomerSearchResult]
    source: str = "SAP S/4HANA API_BUSINESS_PARTNER"


class MaterialSearchResult(BaseModel):
    product: str
    description: str
    score: float
    confidence: str


class MaterialSearchResponse(BaseModel):
    query: str
    count: int
    results: list[MaterialSearchResult]
    source: str = "SAP S/4HANA API_PRODUCT_SRV"


@router.get("/customers/search", response_model=CustomerSearchResponse)
async def search_customers(
    request: Request,
    q: str = Query(..., min_length=1),
    top: int = Query(default=10, ge=1, le=50),
) -> CustomerSearchResponse:
    logger.info("GET /api/customers/search | q=%r | top=%d", q, top)
    try:
        results = search_customer_odata(q, top=top, request=request)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Customer search error")
        raise HTTPException(status_code=500, detail=f"Search failed: {type(exc).__name__}") from exc

    return CustomerSearchResponse(
        query=q, count=len(results),
        results=[CustomerSearchResult(**r) for r in results[:top]],
    )


@router.get("/materials/search", response_model=MaterialSearchResponse)
async def search_materials(
    request: Request,
    q: str = Query(..., min_length=1),
    top: int = Query(default=10, ge=1, le=50),
) -> MaterialSearchResponse:
    logger.info("GET /api/materials/search | q=%r | top=%d", q, top)
    try:
        results = search_material_odata(q, top=top, request=request)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Material search error")
        raise HTTPException(status_code=500, detail=f"Search failed: {type(exc).__name__}") from exc

    return MaterialSearchResponse(
        query=q, count=len(results),
        results=[MaterialSearchResult(**r) for r in results[:top]],
    )