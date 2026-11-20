"""Master-data value help for the review page (customers, ship-to partners, materials, sales orgs).

Reads the mock ECC tables through ``app.ecc.master_data`` and ranks with the same matching
rules the agent uses, so a reviewer sees the same candidates the agent saw.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ..ecc import master_data
from ..models.master_data import (
    CustomerHit,
    CustomerPartnersResponse,
    CustomerSearchResponse,
    MaterialHit,
    MaterialSearchResponse,
    Partner,
    SalesArea,
    SalesOrgList,
)
from ..orders.matching import alpha_out, compact_id, rank_customers, rank_materials
from ..orders.quantity import valid_units
from ..security import get_api_key

router = APIRouter(dependencies=[Depends(get_api_key)])


def _name(row: dict) -> str:
    """NAME1 + NAME2."""
    return " ".join(filter(None, [row.get("NAME1"), row.get("NAME2")]))


def _address(row: dict) -> str:
    """One-line address."""
    return ", ".join(filter(None, [row.get("STRAS"), " ".join(filter(None, [row.get("PSTLZ"), row.get("ORT01")])),
                                   row.get("LAND1")]))


@router.get("/sales-orgs", response_model=SalesOrgList)
async def sales_orgs() -> SalesOrgList:
    """List sales organisations (for the upload dialog and the review form)."""
    return SalesOrgList(items=await asyncio.to_thread(master_data.list_sales_orgs))


@router.get("/customers", response_model=CustomerSearchResponse)
async def search_customers(q: str = Query(min_length=2), country: Optional[str] = None, limit: int = 10) -> CustomerSearchResponse:
    """Search sold-to customers by name, number or address."""
    def run() -> list[CustomerHit]:
        customers = [c for c in master_data.load_customers() if (c.get("KTOKD") or "0001") == "0001"]
        by_number = [c for c in customers if alpha_out(c["KUNNR"]) == q.strip().lstrip("0")]
        if by_number:
            ranked = [{**c, "SCORE": 1.0, "MATCHED_ON": ["customer_number"]} for c in by_number]
        else:
            address = q if any(ch.isdigit() for ch in q) else None
            ranked = rank_customers(customers, limit=limit, name=q, address=address, country=country)
        return [CustomerHit(customer_number=alpha_out(c["KUNNR"]), name=_name(c), address=_address(c),
                            country=c.get("LAND1"), blocked=bool(c.get("AUFSD")), score=c["SCORE"],
                            matched_on=c["MATCHED_ON"]) for c in ranked]
    return CustomerSearchResponse(items=await asyncio.to_thread(run))


@router.get("/customers/{customer_number}/partners", response_model=CustomerPartnersResponse)
async def customer_partners(customer_number: str, sales_org: Optional[str] = None) -> CustomerPartnersResponse:
    """Sales areas and ship-to partners of a sold-to (ship-to selection on the review page)."""
    def run() -> CustomerPartnersResponse | None:
        if master_data.get_customer(customer_number) is None:
            return None
        areas = [SalesArea(sales_org=a["VKORG"], distribution_channel=a["VTWEG"], division=a["SPART"],
                           currency=a.get("WAERS"), payment_terms=a.get("ZTERM"), incoterms=a.get("INCO1"),
                           incoterms_place=a.get("INCO2"), delivering_plant=a.get("VWERK"))
                 for a in master_data.get_sales_areas(customer_number) if not sales_org or a["VKORG"] == sales_org]
        ship_tos = [Partner(function="ship-to", sales_org=p["VKORG"], customer_number=alpha_out(p["KUNN2"]),
                            name=_name(p), address=_address(p), default=bool(p.get("DEFPA")),
                            blocked=bool(p.get("AUFSD") or p.get("LOEVM")))
                    for p in master_data.get_partners(customer_number, sales_org) if p["PARVW"] == "WE"]
        return CustomerPartnersResponse(sales_areas=areas, ship_tos=ship_tos)
    result = await asyncio.to_thread(run)
    if result is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return result


@router.get("/materials", response_model=MaterialSearchResponse)
async def search_materials(q: str = Query(min_length=2), sales_org: Optional[str] = None,
                           sold_to: Optional[str] = None, limit: int = 10) -> MaterialSearchResponse:
    """Search materials by number, sales code, customer code or description."""
    def run() -> list[MaterialHit]:
        materials = [m for m in master_data.load_materials()
                     if not sales_org or any(s["VKORG"] == sales_org for s in m.get("SALES", []))]
        codes = {}
        if sold_to:
            codes = {compact_id(k["KDMAT"]): k["MATNR"] for k in master_data.get_customer_material_codes(sold_to, sales_org)
                     if k.get("KDMAT")}
        ranked = rank_materials(materials, limit=limit, codes=[q], description=q, customer_codes=codes)
        return [MaterialHit(material_number=alpha_out(m["MATNR"]), description=m.get("MAKTX"), sales_code=m.get("BISMT"),
                            base_unit=m.get("MEINS"), allowed_units=sorted(valid_units(m)), score=m["SCORE"],
                            matched_on=m["MATCHED_ON"]) for m in ranked]
    return MaterialSearchResponse(items=await asyncio.to_thread(run))
