"""Deterministic order validation and system fills (pure; no database access).

One function, ``validate``, is used everywhere an order is judged: inside the agent's
``submit_order`` tool, when a reviewer saves or confirms, and in the final gate after the agent
run. The agent can therefore never post an order the rules would reject.

Severities:

* ``ERROR``   - the order cannot be posted (blocks auto-post and Confirm);
* ``REVIEW``  - a human must look at it (blocks auto-post; Confirm accepts it);
* ``WARNING`` - shown, never blocking;
* ``INFO``    - a system fill (value taken from master data or configuration), shown as an auto-fix.

``validate`` applies its fills to the given header/items dicts (documented mutation) so callers
persist exactly what was validated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from .quantity import valid_units

BLOCKING = {"ERROR", "REVIEW"}
# Fields whose low confidence blocks automatic posting (they end up in the sales order);
# low confidence on other fields (notes, amounts, terms text) only raises a warning.
GATED_FIELDS = {"PO_NUMBER", "PO_DATE", "REQ_DELIVERY_DATE", "SOLD_TO", "SHIP_TO", "SALES_ORG", "CURRENCY",
                "LINE_TYPE", "MATERIAL", "QUANTITY", "UOM", "COLOUR_CODE"}
SEVERITY_RANK = {"ERROR": 0, "REVIEW": 1, "WARNING": 2, "INFO": 3}


@dataclass
class Issue:
    """One validation finding; ``item_no`` 0 = header; fields starting with ``_`` are order-level."""

    severity: str
    item_no: int
    field: str
    code: str
    message: str


@dataclass
class MasterView:
    """Master data snapshot needed to validate one order.

    Attributes:
        customer: KNA1 row of the sold-to, or None.
        sales_areas: KNVV rows of the sold-to.
        partners: KNVP rows of the sold-to (with partner name/address).
        materials: MATNR -> material dict (incl. deleted ones, with SALES and UNITS).
        customer_materials: KNMT rows of the sold-to.
        ship_to_areas: Ship-to KUNNR -> its KNVV rows (delivering plant).
        duplicates: References of existing documents with the same sold-to + PO number.
        customer_prices: MATNR -> customer price row (A305: KBETR, KONWA, KPEIN, KMEIN) of the sold-to.
    """

    customer: dict[str, Any] | None = None
    sales_areas: list[dict[str, Any]] = field(default_factory=list)
    partners: list[dict[str, Any]] = field(default_factory=list)
    materials: dict[str, dict[str, Any]] = field(default_factory=dict)
    customer_materials: list[dict[str, Any]] = field(default_factory=list)
    ship_to_areas: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    duplicates: list[str] = field(default_factory=list)
    customer_prices: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class ValidationContext:
    """Order-independent inputs of the validation.

    Attributes:
        received_at: Document received date (requirement R5 fallback for the PO date).
        sales_org_hint: Sales organisation from the upload / inbox mapping.
        confidence_threshold: Minimum confidence before a field needs review.
        order_type, distr_channel, po_type: Configured defaults (requirements: TA / 10 / WWW).
        today: Reference date for plausibility checks.
    """

    received_at: datetime | None
    sales_org_hint: str | None
    confidence_threshold: float
    order_type: str
    distr_channel: str
    po_type: str
    today: date = field(default_factory=date.today)


@dataclass
class ValidationResult:
    """Outcome of ``validate``."""

    issues: list[Issue]
    auto_post_ok: bool

    @property
    def blocking(self) -> list[Issue]:
        """Issues that prevent automatic posting (ERROR and REVIEW)."""
        return [issue for issue in self.issues if issue.severity in BLOCKING]

    @property
    def errors(self) -> list[Issue]:
        """Issues that prevent posting even after human confirmation."""
        return [issue for issue in self.issues if issue.severity == "ERROR"]


def _as_date(value: Any) -> date | None:
    """Coerce a date/datetime/ISO string to a date (None if not parseable)."""
    if value is None or isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def validate(
    header: dict[str, Any],
    items: list[dict[str, Any]],
    field_meta: dict[tuple[int, str], dict[str, Any]],
    view: MasterView,
    ctx: ValidationContext,
    classification: str | None,
    multiple_orders: bool = False,
    review_flags: list[dict[str, Any]] | None = None,
) -> ValidationResult:
    """Validate an order against master data and business rules, applying system fills.

    Args:
        header: POAGENT_ORDER_HEADER-style dict (upper-case keys); fills are written into it.
        items: POAGENT_ORDER_ITEM-style dicts; fills (plant) are written into them.
        field_meta: (item_no, FIELD) -> {"CONFIDENCE", "CHANGED_BY", "AMBIGUOUS"} per field.
        view: Master data snapshot.
        ctx: Validation context.
        classification: STANDARD, CONTRACT or NOT_AN_ORDER.
        multiple_orders: The agent saw more than one order in the message.
        review_flags: Agent-raised review points ({"item_no", "field", "message"}).

    Returns:
        ValidationResult with all issues (incl. INFO fills) and the auto-post decision.
    """
    issues: list[Issue] = []

    def add(severity: str, item_no: int, fld: str, code: str, message: str) -> None:
        issues.append(Issue(severity, item_no, fld, code, message))

    def fill(item_no: int, fld: str, value: Any, note: str, target: dict[str, Any], silent: bool = False) -> None:
        target[fld] = value
        field_meta.setdefault((item_no, fld), {}).update({"CHANGED_BY": "SYSTEM", "CONFIDENCE": 1.0, "FIX_NOTE": note})
        if not silent:
            add("INFO", item_no, fld, "SYSTEM_FILL", note)

    if classification == "NOT_AN_ORDER":
        add("INFO", 0, "_CLASSIFICATION", "NOT_AN_ORDER", "The message was classified as not being a purchase order.")
        return ValidationResult(issues, auto_post_ok=False)

    # --- Configured constants (requirements: order type TA, channel 10, PO type WWW) --------------
    for fld, value in (("ORDER_TYPE", ctx.order_type), ("DISTR_CHANNEL", ctx.distr_channel), ("PO_TYPE", ctx.po_type)):
        if not header.get(fld):
            fill(0, fld, value, "configured default", header, silent=True)

    # --- PO number and date ---------------------------------------------------------------------
    po_number = (header.get("PO_NUMBER") or "").strip()
    if not po_number:
        add("ERROR", 0, "PO_NUMBER", "MISSING", "Customer PO number is missing.")
    elif len(po_number) > 20:
        add("WARNING", 0, "PO_NUMBER", "TRUNCATED", "PO number exceeds 20 characters; VBAK-BSTNK keeps 20 (VBKD-BSTKD keeps 35).")
    po_date = _as_date(header.get("PO_DATE"))
    if po_date is None and ctx.received_at:
        fill(0, "PO_DATE", ctx.received_at.date(), "PO date missing; document received date used", header)
    elif po_date and po_date > ctx.today + timedelta(days=1):
        add("WARNING", 0, "PO_DATE", "FUTURE_DATE", f"PO date {po_date} lies in the future.")

    # --- Sold-to --------------------------------------------------------------------------------
    customer = view.customer
    if not header.get("SOLD_TO"):
        add("ERROR", 0, "SOLD_TO", "MISSING", "Sold-to customer could not be determined.")
    elif customer is None:
        add("ERROR", 0, "SOLD_TO", "NOT_FOUND", f"Customer {header['SOLD_TO']} does not exist.")
    else:
        if customer.get("LOEVM"):
            add("ERROR", 0, "SOLD_TO", "DELETED", "Customer is flagged for deletion.")
        if customer.get("AUFSD"):
            add("ERROR", 0, "SOLD_TO", "BLOCKED", f"Customer has order block {customer['AUFSD']}.")
        if customer.get("KTOKD") and customer["KTOKD"] != "0001":
            add("ERROR", 0, "SOLD_TO", "NOT_SOLD_TO", "Account group is not a sold-to party (ship-to account used?).")

    # --- Sales area -----------------------------------------------------------------------------
    channel = header.get("DISTR_CHANNEL") or ctx.distr_channel
    areas = [a for a in view.sales_areas if a.get("VTWEG") == channel]
    material_lines = [i for i in items if i.get("LINE_TYPE", "MATERIAL") == "MATERIAL"]
    area = None
    if customer is not None:
        if header.get("SALES_ORG"):
            matches = [a for a in areas if a["VKORG"] == header["SALES_ORG"]]
            if not matches:
                add("ERROR", 0, "SALES_ORG", "NOT_MAINTAINED",
                    f"Customer is not maintained in sales org {header['SALES_ORG']} / channel {channel}.")
            area = matches[0] if len(matches) == 1 else None
        else:
            candidates = areas
            if ctx.sales_org_hint:
                candidates = [a for a in candidates if a["VKORG"] == ctx.sales_org_hint] or candidates
            known = [view.materials[i["MATERIAL"]] for i in material_lines if i.get("MATERIAL") in view.materials]
            if len(candidates) > 1 and known:
                candidates = [a for a in candidates
                              if all(any(s["VKORG"] == a["VKORG"] for s in m.get("SALES", [])) for m in known)] or candidates
            if len(candidates) == 1:
                area = candidates[0]
                fill(0, "SALES_ORG", area["VKORG"], "sales org from customer master", header)
            elif not candidates:
                add("ERROR", 0, "SALES_ORG", "NOT_MAINTAINED", f"Customer has no sales area for channel {channel}.")
            else:
                add("ERROR", 0, "SALES_ORG", "UNDETERMINED", "Customer has several sales areas; choose the sales org.")
        if area:
            if area.get("AUFSD"):
                add("ERROR", 0, "SALES_ORG", "BLOCKED", f"Customer is blocked in sales org {area['VKORG']}.")
            if not header.get("DIVISION") or header["DIVISION"] != area["SPART"]:
                fill(0, "DIVISION", area["SPART"], "division from customer sales area", header, silent=True)
            if not header.get("CURRENCY"):
                fill(0, "CURRENCY", area["WAERS"], "currency from customer master", header)
            elif area.get("WAERS") and header["CURRENCY"] != area["WAERS"]:
                add("REVIEW", 0, "CURRENCY", "MISMATCH",
                    f"Document currency {header['CURRENCY']} differs from customer currency {area['WAERS']}.")
            if area.get("ZTERM") and header.get("PAYMENT_TERMS") != area["ZTERM"]:
                fill(0, "PAYMENT_TERMS", area["ZTERM"], "payment terms from customer master", header, silent=True)
            if not header.get("INCOTERMS") and area.get("INCO1"):
                fill(0, "INCOTERMS", area["INCO1"], "incoterms from customer master", header)
                if not header.get("INCOTERMS_PLACE"):
                    fill(0, "INCOTERMS_PLACE", area.get("INCO2"), "incoterms place from customer master", header, silent=True)
            elif area.get("INCO1") and header.get("INCOTERMS") != area["INCO1"]:
                add("REVIEW", 0, "INCOTERMS", "MISMATCH",
                    f"Document incoterms {header['INCOTERMS']} differ from customer incoterms {area['INCO1']}.")

    # --- Ship-to (requirement R2: single WE partner is used; otherwise the document decides) ----
    vkorg = header.get("SALES_ORG")
    ship_tos = [p for p in view.partners if p.get("PARVW") == "WE" and (not vkorg or p.get("VKORG") == vkorg)]
    if customer is not None and vkorg:
        if header.get("SHIP_TO"):
            partner = next((p for p in ship_tos if p["KUNN2"] == header["SHIP_TO"]), None)
            if partner is None:
                add("ERROR", 0, "SHIP_TO", "NOT_A_PARTNER", f"{header['SHIP_TO']} is not a ship-to partner of the customer.")
            elif partner.get("AUFSD") or partner.get("LOEVM"):
                add("ERROR", 0, "SHIP_TO", "BLOCKED", "Ship-to party is blocked or deleted.")
        elif len(ship_tos) == 1:
            fill(0, "SHIP_TO", ship_tos[0]["KUNN2"], "customer has exactly one ship-to partner", header)
        elif not header.get("SHIP_TO_ADDRESS") and any(p.get("DEFPA") for p in ship_tos):
            default = next(p for p in ship_tos if p.get("DEFPA"))
            fill(0, "SHIP_TO", default["KUNN2"], "no delivery address on the document; default ship-to used", header)
        elif not ship_tos:
            add("ERROR", 0, "SHIP_TO", "NONE_MAINTAINED", "Customer has no ship-to partner in this sales org.")
        else:
            add("ERROR", 0, "SHIP_TO", "UNDETERMINED",
                "Several ship-to partners and the document address matches none of them; choose the ship-to.")

    # --- Lines ----------------------------------------------------------------------------------
    if not material_lines:
        add("ERROR", 0, "_NO_MATERIAL_LINES", "NO_LINES", "The order has no material line.")
    ship_to_areas = view.ship_to_areas.get(header.get("SHIP_TO") or "", [])
    for item in items:
        no = int(item["ITEM_NO"])
        line_type = item.get("LINE_TYPE", "MATERIAL")
        if line_type == "CHARGE":
            add("REVIEW", no, "LINE_TYPE", "CHARGE_LINE",
                "Charge line (e.g. freight) must be entered as a pricing condition; it is not posted as an item.")
            continue
        if line_type != "MATERIAL":
            continue
        matnr = item.get("MATERIAL")
        material = view.materials.get(matnr) if matnr else None
        if not matnr:
            add("ERROR", no, "MATERIAL", "MISSING", "Material could not be determined.")
        elif material is None:
            add("ERROR", no, "MATERIAL", "NOT_FOUND", f"Material {matnr} does not exist.")
        else:
            sales = [s for s in material.get("SALES", []) if not vkorg or (s["VKORG"] == vkorg and s["VTWEG"] == channel)]
            if material.get("LVORM"):
                add("ERROR", no, "MATERIAL", "DELETED", "Material is flagged for deletion.")
            elif material.get("MSTAE"):
                add("ERROR", no, "MATERIAL", "BLOCKED", f"Material has cross-plant status {material['MSTAE']}.")
            elif vkorg and not sales:
                add("ERROR", no, "MATERIAL", "NOT_LISTED", f"Material is not extended to sales org {vkorg} / channel {channel}.")
            elif any(s.get("VMSTA") for s in sales):
                add("ERROR", no, "MATERIAL", "NOT_SALABLE", "Material is blocked for sales (VMSTA).")
            if not item.get("PLANT"):
                knmt = next((k for k in view.customer_materials if k["MATNR"] == matnr and k.get("WERKS")), None)
                ship_plant = next((a.get("VWERK") for a in ship_to_areas if a.get("VKORG") == vkorg and a.get("VWERK")), None)
                sales_plant = next((s.get("DWERK") for s in sales if s.get("DWERK")), None)
                plant = (knmt or {}).get("WERKS") or ship_plant or sales_plant
                if plant:
                    source = "customer-material info" if knmt else "ship-to delivering plant" if ship_plant else "material sales data"
                    fill(no, "PLANT", plant, f"plant from {source}", item, silent=True)
                else:
                    add("ERROR", no, "PLANT", "UNDETERMINED", "Delivering plant cannot be determined.")
            uom = item.get("UOM")
            if not uom:
                add("ERROR", no, "UOM", "MISSING", "Unit of measure is missing.")
            elif uom not in valid_units(material):
                add("ERROR", no, "UOM", "INVALID",
                    f"Unit {uom} is not valid for the material (allowed: {', '.join(sorted(valid_units(material)))}); convert the quantity.")
        quantity = item.get("QUANTITY")
        if quantity is None or float(quantity) <= 0:
            add("ERROR", no, "QUANTITY", "INVALID", "Quantity must be greater than zero.")
        elif item.get("UOM") in {"PC", "EA", "ST"} and abs(float(quantity) - round(float(quantity))) > 1e-9:
            add("ERROR", no, "QUANTITY", "FRACTIONAL", f"{quantity} {item['UOM']} is not a whole number of pieces.")

    # --- Totals: printed line amounts must add up to the printed net amount --------------------
    amounts = [i.get("LINE_AMOUNT") for i in items if i.get("LINE_TYPE", "MATERIAL") in {"MATERIAL", "CHARGE"}]
    net = header.get("NET_AMOUNT")
    if net and amounts and all(a is not None for a in amounts):
        total = sum(float(a) for a in amounts)
        if abs(total - float(net)) > 0.01 * max(abs(float(net)), 1.0):
            add("REVIEW", 0, "NET_AMOUNT", "TOTAL_MISMATCH",
                f"Line amounts add up to {total:,.2f} but the document net amount is {float(net):,.2f}.")

    # --- Duplicates, classification, agent flags, confidence, ambiguity -------------------------
    if view.duplicates:
        add("REVIEW", 0, "_DUPLICATE", "DUPLICATE_PO",
            f"Same customer and PO number already exist: {', '.join(view.duplicates)}.")
    if classification == "CONTRACT":
        add("REVIEW", 0, "_CLASSIFICATION", "CONTRACT_ORDER",
            "Classified as contract-based order; contract orders are not posted automatically.")
    elif classification not in ("STANDARD", "CONTRACT"):
        add("REVIEW", 0, "_CLASSIFICATION", "UNCLASSIFIED", "The order has not been classified.")
    if multiple_orders:
        add("REVIEW", 0, "_MULTIPLE_ORDERS", "MULTIPLE_ORDERS", "The message seems to contain several orders.")
    for index, flag in enumerate(review_flags or []):
        # Each flag keeps its own row so it survives re-evaluation (see service.load_persisted_state).
        prefix = f"{flag['field']}: " if flag.get("field") and not str(flag.get("message", "")).startswith(str(flag["field"])) else ""
        add("REVIEW", int(flag.get("item_no") or 0), f"_AGENT_FLAG_{index + 1}", "AGENT_FLAG",
            prefix + (flag.get("message") or "Flagged by the agent."))
    for (item_no, fld), meta in field_meta.items():
        if meta.get("CHANGED_BY") == "USER" or meta.get("INFORMATIONAL"):
            continue
        confidence = meta.get("CONFIDENCE")
        has_error = any(i.item_no == item_no and i.field == fld and i.severity == "ERROR" for i in issues)
        if confidence is not None and confidence < ctx.confidence_threshold and not has_error:
            severity = "REVIEW" if fld in GATED_FIELDS else "WARNING"
            reason = f" {meta['CONFIDENCE_REASON']}" if meta.get("CONFIDENCE_REASON") else ""
            add(severity, item_no, fld, "LOW_CONFIDENCE",
                f"Confidence {confidence:.0%} is below {ctx.confidence_threshold:.0%}.{reason}")
        if meta.get("AMBIGUOUS"):
            add("REVIEW", item_no, fld, "AMBIGUOUS", "Several master-data records match almost equally well.")

    issues.sort(key=lambda issue: (SEVERITY_RANK[issue.severity], issue.item_no, issue.field))
    auto_post_ok = classification == "STANDARD" and not any(issue.severity in BLOCKING for issue in issues)
    return ValidationResult(issues, auto_post_ok)
