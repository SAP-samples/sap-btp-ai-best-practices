"""API models of the master-data value-help endpoints used by the review page."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class SalesOrgList(BaseModel):
    """Sales organisations found in customer sales areas."""

    items: list[str]


class CustomerHit(BaseModel):
    """One customer search result."""

    customer_number: str
    name: str
    address: str
    country: Optional[str] = None
    blocked: bool = False
    score: float
    matched_on: list[str] = []


class CustomerSearchResponse(BaseModel):
    """Response of GET /api/master-data/customers."""

    items: list[CustomerHit]


class Partner(BaseModel):
    """One partner function of a customer (ship-to, bill-to, ...)."""

    function: str
    sales_org: str
    customer_number: str
    name: str
    address: str
    default: bool = False
    blocked: bool = False


class SalesArea(BaseModel):
    """Customer sales-area data relevant for the order."""

    sales_org: str
    distribution_channel: str
    division: str
    currency: Optional[str] = None
    payment_terms: Optional[str] = None
    incoterms: Optional[str] = None
    incoterms_place: Optional[str] = None
    delivering_plant: Optional[str] = None


class CustomerPartnersResponse(BaseModel):
    """Response of GET /api/master-data/customers/{customer_number}/partners."""

    sales_areas: list[SalesArea]
    ship_tos: list[Partner]


class MaterialHit(BaseModel):
    """One material search result."""

    material_number: str
    description: Optional[str] = None
    sales_code: Optional[str] = None
    base_unit: Optional[str] = None
    allowed_units: list[str] = []
    score: float
    matched_on: list[str] = []


class MaterialSearchResponse(BaseModel):
    """Response of GET /api/master-data/materials."""

    items: list[MaterialHit]
