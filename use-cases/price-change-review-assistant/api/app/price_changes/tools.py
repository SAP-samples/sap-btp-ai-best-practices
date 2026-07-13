from __future__ import annotations

import re
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from dateutil import parser as date_parser

from .models import AgentProposal, RawExtractionItem
from .repositories import PriceChangeRepository


def parse_decimal_text(value: str) -> Decimal:
    """Parse a decimal amount from supplier/email price text.

    Args:
        value: Numeric text that may include symbols, percent signs, commas, or
            ISO currency words such as EUR.

    Returns:
        Decimal value extracted from the text.
    """
    normalized = (
        value.strip()
        .replace("%", "")
        .replace("€", "")
        .replace("$", "")
        .replace(",", "")
    )
    normalized = re.sub(r"[A-Za-z]", "", normalized).strip()
    return Decimal(normalized)


def format_money(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def calculate_new_price(current_price: Decimal, price_mode: str | None, price_value: str | None) -> str | None:
    if price_mode is None or price_value is None:
        return None
    value = parse_decimal_text(price_value)
    if price_mode == "absolute":
        return format_money(value)
    if price_mode == "relative_percent":
        return format_money(current_price * (Decimal("1") + value / Decimal("100")))
    if price_mode == "relative_amount":
        return format_money(current_price + value)
    raise ValueError(f"Unsupported price mode: {price_mode}")


def resolve_effective_date(raw_date_text: str | None, email_date: datetime) -> str | None:
    if raw_date_text is None or not raw_date_text.strip():
        return None
    value = raw_date_text.strip().lower()
    base_date = email_date.date()
    if value in {"today", "immediately", "effective immediately", "starting today"}:
        return base_date.isoformat()
    if value == "tomorrow":
        return (base_date + timedelta(days=1)).isoformat()
    if value == "next monday":
        days_ahead = (7 - base_date.weekday()) % 7
        days_ahead = 7 if days_ahead == 0 else days_ahead
        return (base_date + timedelta(days=days_ahead)).isoformat()
    parsed = date_parser.parse(
        raw_date_text,
        default=email_date.replace(month=email_date.month, day=email_date.day),
    )
    return parsed.date().isoformat()


def validate_price_change_proposal(proposal: AgentProposal) -> list[str]:
    errors: list[str] = []
    required_fields = {
        "supplier_id": proposal.supplier_id,
        "material_number": proposal.material_number,
        "original_price": proposal.original_price,
        "requested_new_price": proposal.requested_new_price,
        "currency": proposal.currency,
        "uom": proposal.uom,
        "effective_from": proposal.effective_from,
    }
    for field_name, value in required_fields.items():
        if value is None or value == "":
            errors.append(f"{field_name} is required")
    return errors


class PriceChangeTools:
    def __init__(self, repository: PriceChangeRepository, lookup_repository: Any | None = None) -> None:
        """Create the tool facade used by the price-change agent.

        Args:
            repository: HANA-backed repository used for persistence.
            lookup_repository: Repository used for supplier/material/current-price
                lookups. Defaults to `repository` for tests and legacy callers.

        Returns:
            None.
        """
        self.repository = repository
        self.lookup_repository = lookup_repository or repository

    def find_supplier_by_email(self, email: str | None) -> dict[str, Any]:
        if not email:
            return {"status": "not_found", "candidates": [], "error": "email missing"}
        return self.lookup_repository.find_supplier_by_email(email)

    def find_supplier_by_id(self, supplier_id: str | None) -> dict[str, Any]:
        if not supplier_id:
            return {"status": "not_found", "candidates": [], "error": "supplier id missing"}
        return self.lookup_repository.find_supplier_by_id(supplier_id)

    def find_supplier_by_name(self, name_or_company: str | None) -> dict[str, Any]:
        if not name_or_company:
            return {"status": "not_found", "candidates": [], "error": "name missing"}
        return self.lookup_repository.find_supplier_by_name(name_or_company)

    def find_material_by_number(self, material_number: str | None) -> dict[str, Any]:
        if not material_number:
            return {"status": "not_found", "candidates": [], "error": "material number missing"}
        return self.lookup_repository.find_material_by_number(material_number)

    def search_materials_by_description(
        self,
        query: str | None,
        supplier_id: str | None = None,
    ) -> dict[str, Any]:
        if not query:
            return {"status": "not_found", "candidates": [], "error": "description query missing"}
        return self.lookup_repository.search_materials_by_description(query, supplier_id=supplier_id)

    def get_current_supplier_material_price(
        self,
        supplier_id: str | None,
        material_number: str | None,
    ) -> dict[str, Any]:
        if not supplier_id or not material_number:
            return {
                "status": "not_found",
                "price": None,
                "error": "supplier_id and material_number are required",
            }
        return self.lookup_repository.get_current_supplier_material_price(supplier_id, material_number)

    def find_supplier_material_price_candidates(
        self,
        supplier_candidates: list[dict[str, Any]],
        material_candidates: list[dict[str, Any]],
        purchasing_organizations: list[str] | None = None,
        info_record_categories: list[str] | None = None,
        plants: list[str] | None = None,
    ) -> dict[str, Any]:
        """Find possible S/4 price contexts for candidate suppliers/materials.

        Args:
            supplier_candidates: Supplier candidate rows returned by supplier
                lookup tools.
            material_candidates: Material candidate rows returned by material
                lookup tools.
            purchasing_organizations: Optional purchasing organizations to use
                as exact S/4 info-record filters.
            info_record_categories: Optional info-record categories to use as
                exact S/4 info-record filters.
            plants: Optional plants to use as exact S/4 info-record filters.

        Returns:
            Lookup repository response containing matching price-context
            candidates and status `found`, `ambiguous`, or `not_found`.
        """
        return self.lookup_repository.find_supplier_material_price_candidates(
            supplier_candidates=supplier_candidates,
            material_candidates=material_candidates,
            purchasing_organizations=purchasing_organizations,
            info_record_categories=info_record_categories,
            plants=plants,
        )

    def persist_price_change_draft(
        self,
        proposal: AgentProposal,
        extraction_id: str,
        item_index: int,
        raw_agent_output: dict[str, Any],
    ) -> str:
        return self.repository.insert_price_change_draft(
            proposal=proposal,
            extraction_id=extraction_id,
            item_index=item_index,
            raw_agent_output=raw_agent_output,
        )

    def extraction_item_to_price_parts(self, item: RawExtractionItem) -> tuple[str | None, str | None]:
        if item.requested_price is None:
            return None, None
        return item.requested_price.mode, item.requested_price.value
