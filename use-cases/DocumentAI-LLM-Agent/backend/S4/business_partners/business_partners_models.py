"""
Business Partners Models

Pydantic models for S/4HANA Business Partner data.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class BusinessPartner(BaseModel):
    business_partner: str = Field(..., description="Business Partner number (customer code)")
    business_partner_name: str = Field(
        default="", description="Business Partner display name"
    )
    full_description: str = Field(
        default="", description="Full description: name / city state postal"
    )


class BusinessPartnersResponse(BaseModel):
    success: bool
    count: int
    limit: int
    business_partners: list[BusinessPartner]