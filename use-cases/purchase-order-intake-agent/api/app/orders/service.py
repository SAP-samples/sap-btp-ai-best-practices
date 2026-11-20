"""Order evaluation and state transitions shared by the agent tool, the pipeline and the review UI.

``evaluate`` is the single place where an order (header, lines, per-field metadata) is turned
into validated, persisted state:

1. a master-data snapshot is loaded (``MasterView``);
2. *match confidence* is recomputed deterministically for sold-to and ship-to by re-scoring the
   printed values against master data (field confidence = min(agent confidence, match
   confidence)). Materials are chosen by the agent with structured evidence (matched attributes,
   conflicts, close alternatives); the code turns that evidence, the customer-price check and a
   weak-text guard into the confidence (``material_confidence``). An empty material scores 0;
3. ``validation.validate`` applies system fills and produces issues;
4. field-state rows are built (one row per field, worst issue first).

Entry points:

* ``apply_agent_submission`` - called by the agent's ``submit_order`` tool;
* ``reevaluate_persisted``  - re-runs validation on stored state (final gate, Save, Confirm);
* ``finalize_after_agent``  - sets the status after an agent run (posting is wired in by the pipeline).
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone
from typing import Any, Callable

from ..db.engine import get_engine
from ..ecc import master_data
from ..order_agent.schemas import HEADER_COLUMNS, HEADER_META_ONLY, LINE_COLUMNS, OrderSubmission
from . import repository
from .matching import (
    alpha_in,
    alpha_out,
    compact_id,
    rank_customers,
    score_customer,
    score_material,
    text_similarity,
    token_coverage,
)
from .settings import get_settings
from .validation import SEVERITY_RANK, Issue, MasterView, ValidationContext, ValidationResult, validate

# Header columns that only the system fills; cleared before every evaluation so they are re-derived.
SYSTEM_HEADER_COLUMNS = ("DIVISION", "PAYMENT_TERMS", "ORDER_TYPE", "DISTR_CHANNEL", "PO_TYPE")
DATE_COLUMNS = {"PO_DATE", "REQ_DELIVERY_DATE"}
NUMBER_COLUMNS = {"NET_AMOUNT", "GROSS_AMOUNT", "QUANTITY", "UNIT_PRICE", "LINE_AMOUNT"}
UPPER_COLUMNS = {"CURRENCY", "UOM", "INCOTERMS", "SALES_ORG"}
# Material confidence caps when the agent's evidence or the system checks raise doubts.
CONFLICT_CAP, PRICE_CAP, WEAK_TEXT_CAP = 0.6, 0.75, 0.7
WEAK_TEXT_SCORE = 0.35       # text score below which the chosen material hardly resembles the line
PRICE_MATCH_TOLERANCE, PRICE_CONFLICT_TOLERANCE = 0.02, 0.10


def _now() -> datetime:
    """Current naive UTC timestamp."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def coerce(column: str, value: Any) -> Any:
    """Convert a submitted value to the column's Python type (invalid values become None).

    Args:
        column: Target column name.
        value: Value from the agent or the UI.

    Returns:
        date for date columns, float for amounts/quantities, ALPHA-formatted numbers for
        customers/materials, upper-case keys for codes, stripped strings otherwise.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if column in DATE_COLUMNS:
        if isinstance(value, date):
            return value
        try:
            return date.fromisoformat(str(value).strip()[:10])
        except ValueError:
            return None
    if column in NUMBER_COLUMNS:
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip().replace(" ", "")
        for candidate in (text, text.replace(",", ""), text.replace(".", "").replace(",", ".")):
            try:
                return float(candidate)
            except ValueError:
                continue
        return None
    if column in {"SOLD_TO", "SHIP_TO"}:
        return alpha_in(value, 10)
    if column == "MATERIAL":
        return alpha_in(value, 18)
    if column in UPPER_COLUMNS:
        return str(value).strip().upper()
    return str(value).strip()


# ---------------------------------------------------------------------------
# Master data snapshot and deterministic match confidence
# ---------------------------------------------------------------------------
def load_master_view(order_id: str, header: dict[str, Any], items: list[dict[str, Any]]) -> tuple[MasterView, dict[str, dict]]:
    """Load the master data needed to validate one order.

    Args:
        order_id: Order id (excluded from the duplicate check).
        header: Header dict (SOLD_TO, PO_NUMBER).
        items: Line dicts.

    Returns:
        (MasterView, all materials by MATNR) - the full material map is reused for ambiguity checks.
    """
    sold_to = header.get("SOLD_TO")
    materials = {m["MATNR"]: m for m in master_data.load_materials(include_deleted=True)}
    view = MasterView(materials=materials)
    if sold_to:
        view.customer = master_data.get_customer(sold_to)
        if view.customer:
            view.sales_areas = master_data.get_sales_areas(sold_to)
            view.partners = master_data.get_partners(sold_to)
            view.customer_materials = master_data.get_customer_material_codes(sold_to)
            view.customer_prices = master_data.get_customer_prices(sold_to)
            for partner in view.partners:
                if partner["PARVW"] == "WE":
                    view.ship_to_areas[partner["KUNN2"]] = master_data.get_sales_areas(partner["KUNN2"])
        if header.get("PO_NUMBER"):
            view.duplicates = repository.find_duplicates(sold_to, header["PO_NUMBER"], order_id)
    return view, materials


def _meta_text(field_meta: dict, key: tuple[int, str]) -> str | None:
    """Raw printed text of a field, falling back to nothing."""
    return (field_meta.get(key) or {}).get("RAW_VALUE")


def price_check(item: dict[str, Any], price_row: dict[str, Any] | None) -> tuple[str | None, bool]:
    """Compare the printed unit price of a line with the customer's price (A305) for the material.

    Args:
        item: Line dict (UNIT_PRICE, or LINE_AMOUNT / QUANTITY; UOM).
        price_row: A305 row of the sold-to and material, or None.

    Returns:
        (text for the reviewer or None when nothing could be compared, True when the prices differ
        by more than ``PRICE_CONFLICT_TOLERANCE``).
    """
    printed = item.get("UNIT_PRICE")
    if printed is None and item.get("LINE_AMOUNT") and item.get("QUANTITY"):
        printed = float(item["LINE_AMOUNT"]) / float(item["QUANTITY"])
    if not price_row or printed is None or (price_row.get("KMEIN") and item.get("UOM")
                                             and price_row["KMEIN"] != item["UOM"]):
        return None, False
    price = float(price_row["KBETR"]) / float(price_row.get("KPEIN") or 1)
    if not price:
        return None, False
    deviation = float(printed) / price - 1
    currency = price_row.get("KONWA") or ""
    if abs(deviation) <= PRICE_MATCH_TOLERANCE:
        return f"Printed price {float(printed):.2f} matches the customer price {price:.2f} {currency}.", False
    return (f"Printed price {float(printed):.2f} differs from the customer price {price:.2f} {currency} "
            f"({deviation:+.0%})."), abs(deviation) > PRICE_CONFLICT_TOLERANCE


def material_confidence(meta: dict[str, Any], text_score: float, price_text: str | None, price_differs: bool) -> None:
    """Turn the agent's match evidence and the system checks into the material confidence (mutates ``meta``).

    Starts from the agent's confidence and caps it when the agent reported conflicts, when the
    printed price differs from the customer price, or when the printed line hardly resembles the
    material text. Close alternatives named by the agent make the field ambiguous (review).

    Args:
        meta: Field metadata of the MATERIAL field with ``EVIDENCE`` ({matched, conflicts, alternatives}).
        text_score: Deterministic score of the printed codes/description against the material.
        price_text: Result of ``price_check`` (None when no price could be compared).
        price_differs: True when the price deviation exceeds the tolerance.
    """
    evidence = meta["EVIDENCE"]
    confidence = float(meta.get("EXTRACT_CONFIDENCE") or 0.0)
    reasons = []
    if evidence.get("conflicts"):
        confidence = min(confidence, CONFLICT_CAP)
        reasons.append("Conflicts: " + "; ".join(evidence["conflicts"]) + ".")
    if price_differs:
        confidence = min(confidence, PRICE_CAP)
        reasons.append(price_text)
    if text_score < WEAK_TEXT_SCORE:
        confidence = min(confidence, WEAK_TEXT_CAP)
        reasons.append(f"The printed line resembles the material text only weakly ({text_score:.0%}).")
    evidence["price_check"] = price_text
    meta.update({"MATCH_CONFIDENCE": round(text_score, 3), "CONFIDENCE": round(confidence, 3),
                 "AMBIGUOUS": bool(evidence.get("alternatives")), "CONFIDENCE_REASON": " ".join(reasons) or None})


def apply_match_confidence(
    header: dict[str, Any],
    items: list[dict[str, Any]],
    field_meta: dict[tuple[int, str], dict[str, Any]],
    view: MasterView,
    sender: str | None,
) -> None:
    """Recompute MATCH_CONFIDENCE / CONFIDENCE / AMBIGUOUS for sold-to, ship-to and materials.

    Fields changed by a user keep confidence 1.0. Mutates ``field_meta``.

    Args:
        header: Header dict.
        items: Line dicts.
        field_meta: Per-field metadata to update.
        view: Master data snapshot.
        sender: Sender e-mail address of the message.
    """
    def set_match(key: tuple[int, str], score: float, ambiguous: bool) -> None:
        meta = field_meta.setdefault(key, {"CHANGED_BY": "AGENT"})
        if meta.get("CHANGED_BY") == "USER":
            return
        extract = meta.get("EXTRACT_CONFIDENCE")
        meta["MATCH_CONFIDENCE"] = round(score, 3)
        meta["CONFIDENCE"] = round(min(score, extract) if extract is not None else score, 3)
        meta["AMBIGUOUS"] = ambiguous

    # Sold-to: rank all sold-to accounts with what the document says about the buyer.
    if header.get("SOLD_TO") and view.customer:
        query = {
            "name": header.get("SOLD_TO_NAME") or _meta_text(field_meta, (0, "SOLD_TO")),
            "address": header.get("SOLD_TO_ADDRESS"),
            "sender_email": sender,
            "tax_id": _meta_text(field_meta, (0, "BUYER_TAX_ID")),
            "phone": _meta_text(field_meta, (0, "BUYER_PHONE")),
        }
        sold_tos = [c for c in master_data.load_customers() if (c.get("KTOKD") or "0001") == "0001"]
        ranked = rank_customers(sold_tos, limit=5, learned=repository.learned_customers(), **query)
        own = next((c for c in ranked if c["KUNNR"] == header["SOLD_TO"]), None)
        score = own["SCORE"] if own else score_customer({**view.customer, "EMAILS": []}, **query)[0]
        ambiguous = any(c["KUNNR"] != header["SOLD_TO"] and c["SCORE"] >= score - 0.05 for c in ranked)
        set_match((0, "SOLD_TO"), score, ambiguous)

    # Ship-to: compare the delivery address on the document with the chosen WE partner.
    doc_address = " ".join(filter(None, [header.get("SHIP_TO_NAME"), header.get("SHIP_TO_ADDRESS")]))
    if header.get("SHIP_TO") and header.get("SHIP_TO_ADDRESS"):
        def partner_score(partner: dict[str, Any]) -> float:
            master = " ".join(filter(None, [partner.get(k) for k in ("NAME1", "NAME2", "STRAS", "PSTLZ", "ORT01")]))
            return max(text_similarity(doc_address, master), token_coverage(doc_address, master))

        ship_tos = [p for p in view.partners if p["PARVW"] == "WE"]
        chosen = next((p for p in ship_tos if p["KUNN2"] == header["SHIP_TO"]), None)
        if chosen:
            score = partner_score(chosen)
            ambiguous = any(p["KUNN2"] != chosen["KUNN2"] and partner_score(p) >= score - 0.05 for p in ship_tos)
            set_match((0, "SHIP_TO"), score, ambiguous)

    # Materials: re-score printed codes / description against the material master.
    vkorg = header.get("SALES_ORG")
    customer_codes = {compact_id(k["KDMAT"]): k["MATNR"] for k in view.customer_materials if k.get("KDMAT")}
    learned = repository.learned_materials(header["SOLD_TO"]) if header.get("SOLD_TO") else {}
    candidates = [m for m in view.materials.values()
                  if not m.get("LVORM") and (not vkorg or any(s["VKORG"] == vkorg for s in m.get("SALES", [])))]
    for item in items:
        if item.get("LINE_TYPE", "MATERIAL") != "MATERIAL":
            continue
        matnr = item.get("MATERIAL")
        no = int(item["ITEM_NO"])
        if matnr not in view.materials:
            # No valid material chosen: the field must not show the confidence of reading the printed code.
            set_match((no, "MATERIAL"), 0.0, False)
            continue
        codes = [c for c in (_meta_text(field_meta, (no, "MATERIAL")), item.get("CUSTOMER_MATERIAL")) if c]
        line = {"codes": codes, "description": item.get("DESCRIPTION") or _meta_text(field_meta, (no, "DESCRIPTION")),
                "colour": item.get("COLOUR_CODE"), "customer_codes": customer_codes, "learned": learned}
        own_score = score_material(view.materials[matnr], **line)[0]
        meta = field_meta.get((no, "MATERIAL")) or {}
        if meta.get("EVIDENCE") is not None and meta.get("CHANGED_BY") != "USER":
            material_confidence(meta, own_score, *price_check(item, view.customer_prices.get(matnr)))
            continue
        # Without agent evidence (older orders): keep the deterministic score and rival-based ambiguity.
        rivals = [score_material(m, **line)[0] for m in candidates if m["MATNR"] != matnr]
        ambiguous = any(score >= own_score - 0.05 for score in rivals) and own_score < 1.0
        set_match((no, "MATERIAL"), own_score, ambiguous)


# ---------------------------------------------------------------------------
# Evaluation and field-state rows
# ---------------------------------------------------------------------------
def build_field_states(field_meta: dict[tuple[int, str], dict[str, Any]], issues: list[Issue]) -> list[dict[str, Any]]:
    """Merge field metadata and issues into POAGENT_FIELD_STATE rows (worst issue per field).

    Args:
        field_meta: Per-field metadata.
        issues: Validation issues, sorted most severe first.

    Returns:
        Rows ready for insertion.
    """
    now = _now()
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    for (item_no, name), meta in field_meta.items():
        rows[(item_no, name)] = {
            "ITEM_NO": item_no, "FIELD_NAME": name, "RAW_VALUE": meta.get("RAW_VALUE"), "SOURCE_REF": meta.get("SOURCE_REF"),
            "EXTRACT_CONFIDENCE": meta.get("EXTRACT_CONFIDENCE"), "MATCH_CONFIDENCE": meta.get("MATCH_CONFIDENCE"),
            "CONFIDENCE": meta.get("CONFIDENCE"), "CHANGED_BY": meta.get("CHANGED_BY") or "AGENT",
            "FIX_NOTE": meta.get("FIX_NOTE"), "UPDATED_AT": now,
            "EVIDENCE": json.dumps(meta["EVIDENCE"], ensure_ascii=False) if meta.get("EVIDENCE") else None,
        }
    for issue in issues:
        # Issue-only rows (no value metadata) are not system fills: CHANGED_BY stays AGENT so a later
        # re-evaluation does not clear the field (only real fills, which carry a FIX_NOTE, are re-derived).
        row = rows.setdefault((issue.item_no, issue.field), {
            "ITEM_NO": issue.item_no, "FIELD_NAME": issue.field, "CHANGED_BY": "AGENT", "UPDATED_AT": now})
        if not row.get("ISSUE_SEVERITY") or SEVERITY_RANK[issue.severity] < SEVERITY_RANK[row["ISSUE_SEVERITY"]]:
            row["ISSUE_SEVERITY"], row["ISSUE_CODE"] = issue.severity, issue.code
        row["ISSUE_TEXT"] = " | ".join(filter(None, [row.get("ISSUE_TEXT"), issue.message]))
    return list(rows.values())


def validation_context(message: dict[str, Any] | None) -> ValidationContext:
    """Build the validation context from the inbound message and settings."""
    settings = get_settings()
    return ValidationContext(
        received_at=(message or {}).get("RECEIVED_AT"),
        sales_org_hint=(message or {}).get("SALES_ORG_HINT"),
        confidence_threshold=settings.confidence_threshold,
        order_type=settings.default_order_type,
        distr_channel=settings.default_distr_channel,
        po_type=settings.default_po_type,
    )


def evaluate(
    order_id: str,
    header: dict[str, Any],
    items: list[dict[str, Any]],
    field_meta: dict[tuple[int, str], dict[str, Any]],
    classification: str | None,
    multiple_orders: bool,
    review_flags: list[dict[str, Any]],
    message: dict[str, Any] | None,
) -> tuple[ValidationResult, list[dict[str, Any]]]:
    """Recompute match confidence, validate and build field-state rows (no writes).

    Args:
        order_id: Order id.
        header: Header dict (system fills are applied to it).
        items: Line dicts (plant fills are applied to them).
        field_meta: Per-field metadata (updated in place).
        classification: STANDARD / CONTRACT / NOT_AN_ORDER.
        multiple_orders: Agent saw several orders.
        review_flags: Agent review flags with ``item_no``.
        message: Inbound message row (sender, received date, sales-org hint).

    Returns:
        (ValidationResult, field-state rows).
    """
    for column in SYSTEM_HEADER_COLUMNS:
        header.setdefault(column, None)
    view, _ = load_master_view(order_id, header, items)
    apply_match_confidence(header, items, field_meta, view, (message or {}).get("SENDER"))
    result = validate(header, items, field_meta, view, validation_context(message), classification,
                      multiple_orders, review_flags)
    return result, build_field_states(field_meta, result.issues)


# Header columns written by an evaluation (order content only - never status, locks or timestamps,
# which other actors may change concurrently).
CONTENT_COLUMNS = set(HEADER_COLUMNS.values()) | set(SYSTEM_HEADER_COLUMNS)


def persist_evaluation(conn, order_id: str, header: dict[str, Any], items: list[dict[str, Any]],
                       field_states: list[dict[str, Any]], extra: dict[str, Any] | None = None) -> None:
    """Write header content, lines and field states of an evaluated order in the caller's transaction."""
    content = {column: header.get(column) for column in CONTENT_COLUMNS}
    repository.update_header(conn, order_id, {**content, **(extra or {})})
    repository.replace_items_and_fields(conn, order_id, items, field_states)


def issue_dicts(result: ValidationResult, severities: set[str]) -> list[dict[str, Any]]:
    """Serialise issues of the given severities for tool results and API responses."""
    return [{"severity": i.severity, "item_no": i.item_no, "field": i.field, "code": i.code, "message": i.message}
            for i in result.issues if i.severity in severities]


# ---------------------------------------------------------------------------
# Agent submission
# ---------------------------------------------------------------------------
def apply_agent_submission(order_id: str, submission: OrderSubmission) -> dict[str, Any]:
    """Validate and persist the agent's hand-over; never posts.

    Args:
        order_id: Order being processed (injected, not chosen by the model).
        submission: Parsed ``submit_order`` arguments.

    Returns:
        Tool result: ``saved``, ``auto_post_eligible``, blocking/other ``issues``, ``system_fills``
        and ``item_numbers`` (line -> item number).
    """
    current = repository.get_header(order_id)
    if current is None or current["STATUS"] != "PROCESSING":
        return {"saved": False, "error": "This order is not being processed; submission rejected."}

    header: dict[str, Any] = {column: None for column in HEADER_COLUMNS.values()}
    field_meta: dict[tuple[int, str], dict[str, Any]] = {}

    def record(item_no: int, column: str, value: Any, target: dict[str, Any] | None) -> None:
        if value is None:
            return
        if target is not None:
            target[column] = coerce(column, value.value)
        field_meta[(item_no, column)] = {
            "RAW_VALUE": value.raw, "SOURCE_REF": value.source, "EXTRACT_CONFIDENCE": value.confidence,
            "CONFIDENCE": value.confidence, "CHANGED_BY": "AGENT", "FIX_NOTE": value.note,
            "INFORMATIONAL": target is None,
        }

    ignored: list[str] = []
    for name, value in submission.header.items():
        if name in HEADER_COLUMNS:
            record(0, HEADER_COLUMNS[name], value, header)
        elif name in HEADER_META_ONLY:
            record(0, HEADER_META_ONLY[name], value, None)
        else:
            ignored.append(f"header.{name}")

    items: list[dict[str, Any]] = []
    for index, line in enumerate(submission.lines, start=1):
        item = {"ITEM_NO": index, "LINE_TYPE": line.line_type, **{c: None for c in LINE_COLUMNS.values()}}
        for name, value in line.fields.items():
            if name in LINE_COLUMNS:
                record(index, LINE_COLUMNS[name], value, item)
            else:
                ignored.append(f"lines[{index}].{name}")
        if line.material_evidence is not None and (index, "MATERIAL") in field_meta:
            field_meta[(index, "MATERIAL")]["EVIDENCE"] = line.material_evidence.model_dump()
        items.append(item)

    flags = [{"item_no": flag.line or 0, "field": flag.field, "message": flag.message}
             for flag in submission.review_flags]
    message = repository.get_message(current["MESSAGE_ID"])
    result, field_states = evaluate(order_id, header, items, field_meta, submission.classification,
                                    submission.multiple_orders_detected, flags, message)
    with get_engine().begin() as conn:
        persist_evaluation(conn, order_id, header, items, field_states, {
            "CLASSIFICATION": submission.classification,
            "CLASSIFICATION_REASON": submission.classification_reason,
        })
    return {
        "saved": True,
        "auto_post_eligible": result.auto_post_ok,
        "issues": issue_dicts(result, {"ERROR", "REVIEW", "WARNING"}),
        "system_fills": issue_dicts(result, {"INFO"}),
        "item_numbers": {str(index): index for index in range(1, len(items) + 1)},
        "ignored_fields": ignored,
        "resolved": {"sold_to": alpha_out(header.get("SOLD_TO")), "ship_to": alpha_out(header.get("SHIP_TO")),
                     "sales_org": header.get("SALES_ORG"), "currency": header.get("CURRENCY")},
    }


# ---------------------------------------------------------------------------
# Re-evaluation of stored state (final gate, Save, Confirm)
# ---------------------------------------------------------------------------
def load_persisted_state(order_id: str) -> dict[str, Any] | None:
    """Rebuild the evaluation inputs from the stored order.

    System-filled values are cleared so validation re-derives them from current master data;
    agent review flags and the multiple-orders signal are recovered from their field-state rows.

    Args:
        order_id: Order id.

    Returns:
        Dict with header, items, field_meta, flags, multiple_orders, classification, message;
        or None when the order does not exist.
    """
    header = repository.get_header(order_id)
    if header is None:
        return None
    items = repository.get_items(order_id)
    field_meta: dict[tuple[int, str], dict[str, Any]] = {}
    flags: list[dict[str, Any]] = []
    multiple = False
    for state in repository.get_field_states(order_id):
        key = (state["ITEM_NO"], state["FIELD_NAME"])
        if state["FIELD_NAME"].startswith("_AGENT_FLAG_"):
            flags.append({"item_no": state["ITEM_NO"], "field": None, "message": state.get("ISSUE_TEXT")})
            continue
        if state["FIELD_NAME"] == "_MULTIPLE_ORDERS":
            multiple = True
            continue
        if state["FIELD_NAME"].startswith("_"):
            continue
        if state.get("CHANGED_BY") == "SYSTEM":
            target = header if key[0] == 0 else next((i for i in items if i["ITEM_NO"] == key[0]), None)
            if target is not None and key[1] in target and key[1] not in ("LINE_TYPE", "ITEM_NO"):
                target[key[1]] = None
            continue
        field_meta[key] = {
            "RAW_VALUE": state.get("RAW_VALUE"), "SOURCE_REF": state.get("SOURCE_REF"),
            "EXTRACT_CONFIDENCE": state.get("EXTRACT_CONFIDENCE"),
            "CONFIDENCE": 1.0 if state.get("CHANGED_BY") == "USER" else state.get("EXTRACT_CONFIDENCE"),
            "CHANGED_BY": state.get("CHANGED_BY"), "FIX_NOTE": state.get("FIX_NOTE"),
            "EVIDENCE": json.loads(state["EVIDENCE"]) if state.get("EVIDENCE") else None,
            "INFORMATIONAL": key[1] in HEADER_META_ONLY.values(),
        }
    for column in SYSTEM_HEADER_COLUMNS:
        header[column] = None
    return {"header": header, "items": items, "field_meta": field_meta, "flags": flags, "multiple_orders": multiple,
            "classification": header.get("CLASSIFICATION"), "message": repository.get_message(header["MESSAGE_ID"])}


def reevaluate_persisted(order_id: str) -> tuple[ValidationResult, dict[str, Any]] | None:
    """Re-validate the stored order against current master data and persist the result.

    Args:
        order_id: Order id.

    Returns:
        (ValidationResult, state) or None when the order does not exist.
    """
    state = load_persisted_state(order_id)
    if state is None:
        return None
    result, field_states = evaluate(order_id, state["header"], state["items"], state["field_meta"],
                                    state["classification"], state["multiple_orders"], state["flags"], state["message"])
    with get_engine().begin() as conn:
        persist_evaluation(conn, order_id, state["header"], state["items"], field_states)
    return result, state


def blocking_summary(result: ValidationResult) -> str:
    """One-line summary of blocking issues for STATUS_MESSAGE."""
    blocking = result.blocking
    if not blocking:
        return "All checks passed."
    first = blocking[0]
    more = f" (+{len(blocking) - 1} more)" if len(blocking) > 1 else ""
    return f"{len(blocking)} issue(s) need attention: {first.message}{more}"[:2000]


def finalize_after_agent(order_id: str, agent_summary: str | None,
                         post: Callable[[str], str] | None = None) -> str:
    """Decide the order status after an agent run (the deterministic final gate).

    Args:
        order_id: Order id.
        agent_summary: Final text of the agent (stored for the reviewer).
        post: Callable that posts the order and returns the sales document number; when None
            (or auto-posting is disabled) eligible orders stay in NEEDS_REVIEW.

    Returns:
        The new status.
    """
    header = repository.get_header(order_id)
    summary = (agent_summary or "").strip() or None
    if header is None:
        return "FAILED"
    if not header.get("CLASSIFICATION"):
        repository.set_status(order_id, "FAILED", "The agent finished without submitting the order.", AGENT_SUMMARY=summary)
        return "FAILED"
    if header["CLASSIFICATION"] == "NOT_AN_ORDER":
        repository.set_status(order_id, "NOT_AN_ORDER", header.get("CLASSIFICATION_REASON"), AGENT_SUMMARY=summary)
        return "NOT_AN_ORDER"
    result, _ = reevaluate_persisted(order_id)
    if result.auto_post_ok and get_settings().auto_post_enabled and post is not None:
        repository.set_status(order_id, "PROCESSING", "Posting to ECC...", AGENT_SUMMARY=summary)
        return "POSTED" if post(order_id) else "FAILED"   # post() records ECC rejections itself
    message = blocking_summary(result) if result.blocking else "Ready to post; automatic posting is disabled."
    repository.set_status(order_id, "NEEDS_REVIEW", message, AGENT_SUMMARY=summary)
    return "NEEDS_REVIEW"


def new_id() -> str:
    """Return a new UUID string (application identifiers)."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Reviewer actions: Save, Confirm, Reprocess
# ---------------------------------------------------------------------------
EDITABLE_STATUSES = {"NEEDS_REVIEW", "FAILED", "NOT_AN_ORDER"}
API_HEADER_FIELDS = dict(HEADER_COLUMNS)          # snake_case API name -> column
API_LINE_FIELDS = {**LINE_COLUMNS, "plant": "PLANT"}


class ReviewError(Exception):
    """A reviewer action was refused; ``status_code`` is the HTTP status to return."""

    def __init__(self, message: str, status_code: int, issues: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.issues = issues or []


def delete_order(order_id: str) -> None:
    """Delete an order so the same document can be tested again without a duplicate warning (PoC only).

    Args:
        order_id: Order id.

    Raises:
        ReviewError: 403 when deleting is disabled (ORDER_DELETE_ENABLED=false), 404 when the order
            does not exist, 409 while the agent is processing it.
    """
    if not get_settings().order_delete_enabled:
        raise ReviewError("Deleting orders is disabled (ORDER_DELETE_ENABLED=false).", 403)
    header = repository.get_header(order_id)
    if header is None:
        raise ReviewError(f"Order {order_id} not found.", 404)
    if header["STATUS"] == "PROCESSING":
        raise ReviewError("The agent is processing this order; delete it when processing has finished.", 409)
    with get_engine().begin() as conn:
        repository.delete_order(conn, header)


def _mark_user(field_meta: dict, key: tuple[int, str], corrections: list, sold_to: Any, old: Any, new: Any) -> None:
    """Record a reviewer change: confidence 1.0, CHANGED_BY USER, correction entry for learning."""
    meta = field_meta.setdefault(key, {})
    corrections.append({"ITEM_NO": key[0], "FIELD_NAME": key[1], "SOLD_TO": sold_to, "RAW_VALUE": meta.get("RAW_VALUE"),
                        "AGENT_VALUE": None if old is None else str(old), "USER_VALUE": None if new is None else str(new)})
    meta.update({"CHANGED_BY": "USER", "CONFIDENCE": 1.0, "FIX_NOTE": "edited by reviewer"})


def save_user_edits(order_id: str, header_changes: dict[str, Any] | None = None,
                    items: list[dict[str, Any]] | None = None,
                    classification: str | None = None) -> tuple[ValidationResult, list[dict[str, Any]]]:
    """Apply reviewer edits, re-validate and persist (status is kept; NOT_AN_ORDER turned into an order goes to review).

    Args:
        order_id: Order id.
        header_changes: {api field name: value} for changed header fields.
        items: Full new list of lines ({item_no?, line_type?, <api field>: value}); None keeps the lines.
        classification: New classification, if the reviewer changed it.

    Returns:
        (ValidationResult, corrections) - corrections are the changed fields (used by correction learning).

    Raises:
        ReviewError: 404 unknown order, 409 order not editable, 400 unknown field.
    """
    state = load_persisted_state(order_id)
    if state is None:
        raise ReviewError("Order not found", 404)
    header, field_meta = state["header"], state["field_meta"]
    if header["STATUS"] not in EDITABLE_STATUSES:
        raise ReviewError(f"Order in status {header['STATUS']} cannot be edited", 409)
    corrections: list[dict[str, Any]] = []
    for name, value in (header_changes or {}).items():
        column = API_HEADER_FIELDS.get(name)
        if column is None:
            raise ReviewError(f"Unknown header field {name!r}", 400)
        new = coerce(column, value)
        if new != header.get(column):
            _mark_user(field_meta, (0, column), corrections, header.get("SOLD_TO"), header.get(column), new)
            header[column] = new

    current_items = {item["ITEM_NO"]: item for item in state["items"]}
    new_items = state["items"]
    if items is not None:
        new_items, next_no = [], max(current_items, default=0)
        for line in items:
            item_no = line.get("item_no")
            if item_no not in current_items:
                next_no += 1
                item_no = next_no
            item = dict(current_items.get(item_no) or {"ITEM_NO": item_no, "LINE_TYPE": "MATERIAL",
                                                       **{c: None for c in API_LINE_FIELDS.values()}})
            item["LINE_TYPE"] = line.get("line_type") or item.get("LINE_TYPE") or "MATERIAL"
            for name, value in line.items():
                if name in ("item_no", "line_type"):
                    continue
                column = API_LINE_FIELDS.get(name)
                if column is None:
                    raise ReviewError(f"Unknown line field {name!r}", 400)
                new = coerce(column, value)
                if new != item.get(column):
                    _mark_user(field_meta, (item_no, column), corrections, header.get("SOLD_TO"), item.get(column), new)
                    item[column] = new
            new_items.append(item)
        kept = {item["ITEM_NO"] for item in new_items}
        for key in [k for k in field_meta if k[0] and k[0] not in kept]:
            del field_meta[key]

    status = header["STATUS"]
    if classification and classification != state["classification"]:
        state["classification"] = classification
        if status == "NOT_AN_ORDER" and classification != "NOT_AN_ORDER":
            status = "NEEDS_REVIEW"
    result, field_states = evaluate(order_id, header, new_items, field_meta, state["classification"],
                                    state["multiple_orders"], state["flags"], state["message"])
    for correction in corrections:          # learning is scoped to the (possibly corrected) sold-to
        correction["SOLD_TO"] = header.get("SOLD_TO")
    with get_engine().begin() as conn:
        persist_evaluation(conn, order_id, header, new_items, field_states, {
            "CLASSIFICATION": state["classification"], "STATUS": status,
            "STATUS_MESSAGE": blocking_summary(result) if result.blocking else "All checks passed; ready to confirm.",
        })
        repository.insert_corrections(conn, order_id, corrections)
    return result, corrections


def confirm_order(order_id: str, post: Callable[[str], str]) -> str:
    """Human confirmation: re-validate and post unless an ERROR remains (REVIEW issues are accepted).

    Args:
        order_id: Order id.
        post: Posting function (``orders.posting.post_order``).

    Returns:
        The sales document number.

    Raises:
        ReviewError: 404 / 409 for wrong state, 422 with the blocking ERROR issues.
        PostingError: raised by ``post`` (ECC rejection or concurrent posting).
    """
    header = repository.get_header(order_id)
    if header is None:
        raise ReviewError("Order not found", 404)
    if header["STATUS"] not in {"NEEDS_REVIEW", "FAILED"}:
        raise ReviewError(f"Order in status {header['STATUS']} cannot be confirmed", 409)
    if header.get("CLASSIFICATION") == "NOT_AN_ORDER":
        raise ReviewError("The message is classified as not an order", 409)
    result, _ = reevaluate_persisted(order_id)
    if result.errors:
        raise ReviewError("Required data is missing or invalid", 422, issue_dicts(result, {"ERROR"}))
    return post(order_id)


def request_reprocess(order_id: str, note: str | None) -> None:
    """Queue an order for a new agent run with an optional reviewer note.

    Raises:
        ReviewError: 404 unknown order, 409 when posted or already queued/processing.
    """
    header = repository.get_header(order_id)
    if header is None:
        raise ReviewError("Order not found", 404)
    if header["STATUS"] in {"POSTED", "QUEUED", "PROCESSING"}:
        raise ReviewError(f"Order in status {header['STATUS']} cannot be reprocessed", 409)
    repository.set_status(order_id, "QUEUED", "Queued for reprocessing", REVIEWER_NOTE=(note or None),
                          ATTEMPTS=0, CLASSIFICATION=None, CLASSIFICATION_REASON=None, LOCKED_BY=None, LOCKED_AT=None)
