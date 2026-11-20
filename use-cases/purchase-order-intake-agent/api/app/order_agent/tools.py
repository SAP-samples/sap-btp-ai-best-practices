"""LangChain tools of the purchase-order agent (passed to the runtime as ``extra_tools``).

| Tool                   | Purpose                                                              |
|------------------------|----------------------------------------------------------------------|
| search_customers       | Rank sold-to customers for the buyer described on the document       |
| get_customer_profile   | Sales areas, partners (ship-tos), contacts and past reviewer fixes   |
| search_materials       | Rank materials for many order lines in one call                      |
| convert_quantities     | Deterministic unit conversion to the material's sales unit           |
| submit_order           | Hand the order over: validate + persist; returns the issues to fix   |

All tools are synchronous (blocking DB access); the runtime's ToolNode runs them in worker
threads. Results are compact JSON strings; numbers are shown without ALPHA leading zeros.
``submit_order`` receives the order id from the graph state (``InjectedState("context_id")``),
so the model can neither see nor forge it - and it cannot post: posting is decided afterwards
by the deterministic gate.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal, Optional

from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field

from ..ecc import master_data
from ..orders import repository
from ..orders.matching import alpha_in, alpha_out, compact_id, rank_customers, rank_materials
from ..orders.quantity import convert_quantity, valid_units
from ..orders.service import apply_agent_submission, price_check
from .schemas import HEADER_FIELD_HELP, FieldValue, LineSubmission, OrderSubmission, ReviewFlag

PARTNER_FUNCTIONS = {"AG": "sold-to", "WE": "ship-to", "RE": "bill-to", "RG": "payer"}


def _json(data: Any) -> str:
    """Serialise a tool result compactly (dates and decimals as strings)."""
    return json.dumps(data, ensure_ascii=False, default=str, separators=(",", ":"))


def _address(row: dict[str, Any]) -> str:
    """One-line address of a KNA1-like row."""
    return ", ".join(filter(None, [row.get("STRAS"), " ".join(filter(None, [row.get("PSTLZ"), row.get("ORT01")])),
                                   row.get("LAND1")]))


def _name(row: dict[str, Any]) -> str:
    """Full name (NAME1 + NAME2) of a KNA1-like row."""
    return " ".join(filter(None, [row.get("NAME1"), row.get("NAME2")]))


@tool
def search_customers(
    name: Optional[str] = None,
    address: Optional[str] = None,
    country: Optional[str] = None,
    sender_email: Optional[str] = None,
    tax_id: Optional[str] = None,
    phone: Optional[str] = None,
    limit: int = 5,
) -> str:
    """Find the sold-to customer in ECC for the buyer of a purchase order.

    Pass everything the document and e-mail say about the buyer; strong identifiers (VAT/tax id,
    sender e-mail domain, phone) outrank name similarity.

    Args:
        name: Buyer company name as printed (any language).
        address: Buyer address as printed, one line.
        country: ISO 3166 alpha-2 country code of the buyer, if known.
        sender_email: E-mail address of the sender of the order e-mail.
        tax_id: VAT / GST / company registration number printed for the buyer.
        phone: Phone number printed in the buyer block.
        limit: Maximum number of candidates.
    """
    customers = [c for c in master_data.load_customers() if (c.get("KTOKD") or "0001") == "0001"]
    ranked = rank_customers(customers, limit=limit, learned=repository.learned_customers(), name=name,
                            address=address, country=country, sender_email=sender_email, tax_id=tax_id, phone=phone)
    return _json([{
        "customer_number": alpha_out(c["KUNNR"]), "name": _name(c), "address": _address(c),
        "blocked": bool(c.get("AUFSD")), "score": c["SCORE"], "matched_on": c["MATCHED_ON"],
        "ambiguous": c["AMBIGUOUS"], "sales_orgs": sorted({a["VKORG"] for a in c["SALES_AREAS"]}),
    } for c in ranked])


@tool
def get_customer_profile(customer_number: str, sales_org: Optional[str] = None) -> str:
    """Get a customer's sales areas, partner functions (ship-to addresses), contacts and past reviewer corrections.

    Always call this before choosing the ship-to, currency or sales organisation.

    Args:
        customer_number: Sold-to customer number.
        sales_org: Optional sales organisation filter.
    """
    customer = master_data.get_customer(customer_number)
    if customer is None:
        return _json({"error": f"customer {customer_number} not found"})
    areas = master_data.get_sales_areas(customer_number)
    partners = master_data.get_partners(customer_number, sales_org)
    corrections = repository.recent_corrections(alpha_in(customer_number, 10))
    return _json({
        "customer_number": alpha_out(customer["KUNNR"]), "name": _name(customer), "address": _address(customer),
        "blocked": bool(customer.get("AUFSD")), "deleted": bool(customer.get("LOEVM")),
        "sales_areas": [{
            "sales_org": a["VKORG"], "distribution_channel": a["VTWEG"], "division": a["SPART"], "currency": a["WAERS"],
            "payment_terms": a["ZTERM"], "incoterms": a["INCO1"], "incoterms_place": a["INCO2"],
            "delivering_plant": a["VWERK"], "blocked": bool(a.get("AUFSD")),
        } for a in areas if not sales_org or a["VKORG"] == sales_org],
        "partners": [{
            "function": PARTNER_FUNCTIONS.get(p["PARVW"], p["PARVW"]), "sales_org": p["VKORG"],
            "customer_number": alpha_out(p["KUNN2"]), "name": _name(p), "address": _address(p),
            "default": bool(p.get("DEFPA")), "blocked": bool(p.get("AUFSD") or p.get("LOEVM")),
        } for p in partners],
        "contacts": [{"name": " ".join(filter(None, [c.get("NAMEV"), c.get("NAME1")])), "email": c.get("SMTP_ADDR"),
                      "phone": c.get("TELF1")} for c in master_data.get_contacts(customer_number)],
        "recent_corrections": [{"field": c["FIELD_NAME"], "printed": c["RAW_VALUE"], "agent_value": c["AGENT_VALUE"],
                                "corrected_to": c["USER_VALUE"]} for c in corrections],
    })


class MaterialQuery(BaseModel):
    """One order line to resolve."""

    line_ref: str = Field(description="Your reference for the line, e.g. '1'.")
    codes: list[str] = Field(default_factory=list,
                             description="Every code printed on the line: material number, sales code, customer code, "
                                         "codes embedded in the description.")
    description: Optional[str] = Field(default=None, description="Line description as printed; on a retry an English translation "
                                       "with expanded abbreviations.")
    colour: Optional[str] = Field(default=None, description="Colour name or code on the line.")
    pack_text: Optional[str] = Field(default=None, description="Pack size stated on the line, e.g. '20L', '4 Ltr', 'QT'.")
    unit_price: Optional[float] = Field(default=None, description="Unit price printed on the line (plain number), "
                                                                   "compared with the customer's price.")


@tool
def search_materials(
    lines: list[MaterialQuery],
    sold_to: Optional[str] = None,
    sales_org: Optional[str] = None,
    limit_per_line: int = 8,
) -> str:
    """Find candidate ECC materials for several order lines in one call; you choose among them.

    Candidates come from material numbers, sales codes (tolerating a missing letter prefix, missing
    suffix, pack variants), the customer's own article codes, manufacturer part numbers, then
    description, colour and pack size. `score` only ranks candidates; compare each candidate's
    description with the line yourself. With `sold_to`, each candidate shows the customer's price
    and `price_check` against the printed `unit_price`.

    Args:
        lines: Lines to resolve.
        sold_to: Sold-to customer number (enables the customer's own article codes).
        sales_org: Sales organisation; only materials sold there are returned.
        limit_per_line: Candidates per line.
    """
    materials = [m for m in master_data.load_materials()
                 if not sales_org or any(s["VKORG"] == sales_org for s in m.get("SALES", []))]
    customer_codes, learned, prices = {}, {}, {}
    if sold_to:
        prices = master_data.get_customer_prices(sold_to, sales_org)
        learned = repository.learned_materials(alpha_in(sold_to, 10))
        customer_codes = {compact_id(k["KDMAT"]): k["MATNR"]
                          for k in master_data.get_customer_material_codes(sold_to, sales_org) if k.get("KDMAT")}
    result = {}
    for line in lines:
        ranked = rank_materials(materials, limit=limit_per_line, codes=line.codes, description=line.description,
                                colour=line.colour, pack_text=line.pack_text, customer_codes=customer_codes,
                                learned=learned)
        result[line.line_ref] = [{
            "material_number": alpha_out(m["MATNR"]), "description": m.get("MAKTX"), "sales_code": m.get("BISMT"),
            "base_unit": m.get("MEINS"), "allowed_units": sorted(valid_units(m)),
            "fill_volume_l": m.get("VOLUM"), "division": m.get("SPART"),
            "salable": not any(s.get("VMSTA") for s in m.get("SALES", []) if not sales_org or s["VKORG"] == sales_org),
            "manufacturer_part": m.get("MFRPN"),
            "customer_price": float(prices[m["MATNR"]]["KBETR"]) if m["MATNR"] in prices else None,
            "price_check": price_check({"UNIT_PRICE": line.unit_price}, prices.get(m["MATNR"]))[0],
            "score": m["SCORE"], "matched_on": m["MATCHED_ON"], "notes": m["NOTES"], "ambiguous": m["AMBIGUOUS"],
        } for m in ranked]
    return _json(result)


class QuantityItem(BaseModel):
    """One quantity to convert."""

    line_ref: str
    material_number: str
    quantity: float = Field(description="Quantity as stated on the document (plain number).")
    unit: Optional[str] = Field(default=None, description="Unit as stated, translated to English if needed (e.g. 'litre', 'gallon', 'pail').")
    pack_text: Optional[str] = Field(default=None, description="Pack size stated on the line, if any.")


@tool
def convert_quantities(items: list[QuantityItem]) -> str:
    """Convert document quantities to each material's sales unit using master data (deterministic).

    Volume units are divided by the material's fill volume (e.g. 4 400 L / 17.6 L = 250 pieces).
    `exact = false` means the result is not a whole number of pieces: check the document and flag it.

    Args:
        items: Quantities to convert.
    """
    aliases = master_data.load_unit_aliases()
    out = {}
    for item in items:
        material = master_data.get_material(item.material_number)
        if material is None:
            out[item.line_ref] = {"error": "MATERIAL_NOT_FOUND"}
            continue
        out[item.line_ref] = convert_quantity(material, item.quantity, item.unit, item.pack_text, aliases)
    return _json(out)


@tool
def submit_order(
    classification: Literal["STANDARD", "CONTRACT", "NOT_AN_ORDER"],
    classification_reason: str,
    header: dict[str, FieldValue],
    lines: list[LineSubmission],
    context_id: Annotated[str, InjectedState("context_id")],
    review_flags: Optional[list[ReviewFlag]] = None,
    multiple_orders_detected: bool = False,
) -> str:
    """Hand the extracted and resolved order over for validation and storage (does NOT post it).

    Call once when extraction and master-data resolution are complete; call again only to fix
    issues the documents support. The result lists blocking issues (ERROR, REVIEW), warnings and
    the values the system filled from master data.

    Args:
        classification: STANDARD, CONTRACT or NOT_AN_ORDER.
        classification_reason: One or two sentences citing the evidence.
        header: Header fields as {field_name: FieldValue} (allowed names in the schema description).
        lines: Order lines in document order.
        review_flags: Conflicts or doubts the reviewer must check.
        multiple_orders_detected: True if the message contains more than one purchase order.
    """
    submission = OrderSubmission(
        classification=classification, classification_reason=classification_reason, header=header, lines=lines,
        review_flags=review_flags or [], multiple_orders_detected=multiple_orders_detected,
    )
    return _json(apply_agent_submission(context_id, submission))


def build_order_tools() -> list[BaseTool]:
    """Return the order agent's tools (added to the runtime next to ``load_skill``)."""
    return [search_customers, get_customer_profile, search_materials, convert_quantities, submit_order]
