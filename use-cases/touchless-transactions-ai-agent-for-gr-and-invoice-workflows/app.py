import os, json, re, threading
from typing import Any, Dict, List, Tuple, Optional
from flask import Flask, request, Response
from dotenv import load_dotenv
import requests

from botbuilder.core import (
    BotFrameworkAdapter,
    BotFrameworkAdapterSettings,
    TurnContext,
)
from botbuilder.core.teams import TeamsActivityHandler
from botbuilder.schema import (
    Activity,
    ActivityTypes,
    InvokeResponse,
    ChannelAccount,
    ConversationReference,
)

from api_client import InvoiceAPI
from cards import (
    get_unified_card,
    get_result_summary_card,
    get_welcome_card,
    get_user_orders_card,
    get_invoice_items_card,
    get_totals_card,
    get_invoice_status_card,
)

# Optional LLM backend (AI Core proxy)
try:
    from llm_client import ask_llm_simple as ask_llm_backend
except Exception:
    ask_llm_backend = None

# ---------- env / globals ----------
load_dotenv()
APP_ID = os.getenv("MICROSOFT_APP_ID")
APP_PWD = os.getenv("MICROSOFT_APP_PASSWORD")
APP_PUBLIC_URL = os.getenv("APP_PUBLIC_URL", "http://localhost:3978")
EXCEL_SOURCE = os.getenv("EXCEL_SOURCE", "dataset_enriched.xlsx")
EXCEL_SHEET = os.getenv("EXCEL_SHEET", "POC scienarios")
BRAND_IMAGE_URL = os.getenv(
    "BRAND_IMAGE_URL",
    "https://cdn.logojoy.com/wp-content/uploads/2018/12/14141838/ic_pen_tool.svg",
)
USER_NAME = os.getenv("USER_NAME", "Marta")

app = Flask(__name__)
adapter = BotFrameworkAdapter(BotFrameworkAdapterSettings(APP_ID, APP_PWD))
api = InvoiceAPI(excel_path=EXCEL_SOURCE, sheet_name=EXCEL_SHEET)

CONV_REFS: Dict[str, ConversationReference] = {}
REF_FILE = os.getenv("CONV_REF_FILE", "conversation_ref.json").strip()


# ---------- utils ----------
def _to_float(v: Any) -> float:
    """Robust to strings like '1.234,50 €' or '1,234.50 Euro'."""
    try:
        s = str(v).strip().replace("€", "").replace("Euro", "").replace("EUR", "")
        if "." in s and "," in s:
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", ".")
        return float(s)
    except Exception:
        return 0.0


def conv_id(ctx: TurnContext) -> str:
    return ctx.activity.conversation.id or "global"


def remember_reference(ctx: TurnContext):
    ref = TurnContext.get_conversation_reference(ctx.activity)
    CONV_REFS[conv_id(ctx)] = ref
    try:
        with open(REF_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {k: v.serialize() for k, v in CONV_REFS.items()},
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception:
        pass


def _load_refs():
    try:
        if not os.path.exists(REF_FILE):
            return
        with open(REF_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for cid, d in raw.items():
            CONV_REFS[cid] = ConversationReference().deserialize(d)
    except Exception:
        pass


_load_refs()

# ---------- intents / parsing ----------
MENTION_RE = re.compile(r"(?is)<at>.*?</at>")


def _clean_text(t: str) -> str:
    t = (t or "").strip()
    t = MENTION_RE.sub("", t)
    return " ".join(t.split())


INVOICE_ID_RE = re.compile(r"\b(\d{7,12})\b")
MY_ORDERS_PATTERNS = re.compile(
    r"(?is)\b(my\s+orders|list\s+my\s+orders|list\s+my\s+po\b|list\s+my\s+pos\b|show\s+my\s+(orders|po|pos))\b"
)
RECEIVED_Q = re.compile(r"(?is)\b(which\s+orders\s+are\s+received|received\s+orders)\b")
NOT_RECEIVED_Q = re.compile(
    r"(?is)\b(which\s+orders\s+are\s+not\s+received|not\s+received\s+orders)\b"
)
INVOICE_ITEMS_ASK_RE = re.compile(
    r"(?is)\b(line(\s*item)?\s*(name|description)?|item\s*(name|desc)|lines?)\b"
)
INVOICE_ONLY_RE = re.compile(r"(?is)\binvoice\s+(\d{7,12})\b")
ITEM_FILTER_RE = re.compile(r"(?is)\ball\s+po[s]?\s+for\s+([a-z0-9 \-_/]+)")
LIST_ALL_PO_RE = re.compile(r"(?is)\blist\s+all\s+po[s]?\b")
TOTALS_Q = re.compile(
    r"(?is)\bwhat\s+is\s+the\s+total\s+value\s+of\s+my\s+orders\b|\bshow\s+totals\s+ir\s+vs\s+po\b"
)
STATUS_Q = re.compile(r"(?is)\bwhat\s+is\s+the\s+status\s+of\s+invoice\s+(\d{7,12})\b")
DOMAIN_KEYWORDS = re.compile(
    r"(?is)\b(invoice|po|purchase\s*order|goods\s*receipt|gr|order)\b"
)


def _domain_relevant(text: str) -> bool:
    return bool(DOMAIN_KEYWORDS.search(text))


# ---------- business rules ----------
def _approx(a: float, b: float, eps: float = 1e-6) -> bool:
    return abs(a - b) <= eps


def classify_line(ir: float, po: float, gr: float) -> Dict[str, Any]:
    """
    Return scenario classification & UI hints.
    """
    diff = ir - po
    pct_thr = 0.05 * max(po, 0.0)
    amt_thr = 250.0

    within = diff > 0 and diff <= pct_thr and diff <= amt_thr
    exceed_percent_only = diff > pct_thr and diff <= amt_thr
    exceed_amount_only = diff > amt_thr and diff <= pct_thr
    exceed_both = diff > pct_thr and diff > amt_thr

    # S1
    if (ir <= po and _approx(gr, 0.0)) or (within and _approx(gr, 0.0)):
        return {
            "scenario": "s1",
            "edit_kind": "none",
            "suggest": 0.0,
            "button_hint": "received",
            "cond": (
                "IR ≤ PO, GR = 0" if ir <= po else "IR > PO (within tolerance), GR = 0"
            ),
        }
    # S2
    if ir <= po and ir > gr:
        return {
            "scenario": "s2",
            "edit_kind": "gr",
            "suggest": max(ir - gr, 0.0),
            "button_hint": "partial",
            "cond": "IR ≤ PO, IR > GR",
        }
    if within and ir > gr:
        return {
            "scenario": "s2",
            "edit_kind": "gr",
            "suggest": gr,
            "button_hint": "partial",
            "cond": "IR > PO (within tolerance), IR > GR",
        }
    # S3
    if exceed_both:
        return {
            "scenario": "s3",
            "edit_kind": "po",
            "suggest": ir,
            "button_hint": "change",
            "cond": "IR > PO (exceed % and €250)",
        }
    # S6
    if exceed_percent_only and ir > gr:
        return {
            "scenario": "s6",
            "edit_kind": "po",
            "suggest": ir,
            "button_hint": "changes",
            "cond": "IR > PO (exceed % only), IR > GR",
        }
    # S7
    if exceed_amount_only:
        return {
            "scenario": "s7",
            "edit_kind": "po",
            "suggest": ir,
            "button_hint": "change",
            "cond": "IR > PO (exceed €250 only)",
        }
    # Fallbacks
    if ir > gr:
        return {
            "scenario": "s2",
            "edit_kind": "gr",
            "suggest": max(ir - gr, 0.0),
            "button_hint": "partial",
            "cond": "IR > GR",
        }
    return {
        "scenario": "s1",
        "edit_kind": "none",
        "suggest": 0.0,
        "button_hint": "received",
        "cond": "IR ≤ PO, GR = 0",
    }


# ---------- LLM contexts ----------
def build_llm_context_for_invoice(invoice_number: str) -> str:
    try:
        rows = api.get_invoice_lines(invoice_number).get("rows", [])
    except Exception:
        rows = []
    if not rows:
        return f"No rows found for invoice {invoice_number}."
    lines, not_received, need_po_change = [], 0, 0
    for idx, r in enumerate(rows, start=1):
        ir = _to_float(r.get("IR_amount", 0))
        po = _to_float(r.get("PO_amount", 0))
        gr = _to_float(r.get("GR_amount", 0))
        cond = classify_line(ir, po, gr).get("cond", "")
        item = r.get("Item_Name", "")
        if ir > gr:
            not_received += 1
        if ir - po > 250 or (po > 0 and ir - po > 0.05 * po):
            need_po_change += 1
        lines.append(f"Line {idx}: Item={item}; IR={ir}; PO={po}; GR={gr} | {cond}")
    summary = f"Invoice {invoice_number}: {len(rows)} lines; not fully received={not_received}; suggested PO changes={need_po_change}."
    return summary + "\n" + "\n".join(lines)


def build_llm_context_for_user(user_name: str) -> str:
    try:
        rows = api.get_rows_for_user(user_name)
    except Exception:
        rows = []
    if not rows:
        return f"No rows found for user {user_name}."
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        inv = r.get("Invoice_Number", "N/A")
        grouped.setdefault(inv, []).append(r)

    lines, total_not_received, total_po_change = [], 0, 0
    for inv, inv_rows in grouped.items():
        nr = 0
        pc = 0
        for r in inv_rows:
            ir = _to_float(r.get("IR_amount", 0))
            po = _to_float(r.get("PO_amount", 0))
            gr = _to_float(r.get("GR_amount", 0))
            if ir > gr:
                nr += 1
            if ir - po > 250 or (po > 0 and ir - po > 0.05 * po):
                pc += 1
        total_not_received += nr
        total_po_change += pc
        item_names = ", ".join(
            sorted({rr.get("Item_Name", "") for rr in inv_rows if rr.get("Item_Name")})
        )
        lines.append(
            f"Invoice {inv}: lines={len(inv_rows)}, not received={nr}, need PO change={pc}; items=[{item_names}]"
        )
    summary = f"User {user_name}: invoices={len(grouped)}, not received total lines={total_not_received}, suggested PO changes total={total_po_change}."
    return summary + "\n" + "\n".join(lines)


# ---------- unified card builder ----------
def build_unified_card(
    invoice_number: str, structured: str = "", detailed: str = ""
) -> Dict[str, Any]:
    rows = api.get_invoice_lines(invoice_number).get("rows", [])

    # Guard: invoice not found / empty
    if not rows:
        msg = f"No items found for invoice {invoice_number}. Please check the number."
        return get_result_summary_card(
            [msg],
            title="Invoice not found",
            add_ai_prompt=True,
            invoice_number=invoice_number,
        )

    # Totals
    multi = len(rows) > 1
    sum_ir = sum(_to_float(r.get("IR_amount", 0)) for r in rows)
    sum_po = sum(_to_float(r.get("PO_amount", 0)) for r in rows)
    sum_gr = sum(_to_float(r.get("GR_amount", 0)) for r in rows)
    facts = {
        "Invoice #": invoice_number,
        "Invoice amount": f"{sum_ir:,.2f} €",
        "Purchase order Amount": f"{sum_po:,.2f} €",
        "Goods received Amount": f"{sum_gr:,.2f} €",
        "Requestor": USER_NAME,
        "Vendor Name": "Schneider Cables Ltd",
        "PO Order ID": rows[0].get("PO#", "") if rows else "N/A",
        "Document Type": "Invoice",
        "Company Code": "2020 CVS London",
    }

    # Line scenarios
    line_items: List[Dict[str, Any]] = []
    for idx, r in enumerate(rows):
        ir = _to_float(r.get("IR_amount", 0))
        po = _to_float(r.get("PO_amount", 0))
        gr = _to_float(r.get("GR_amount", 0))
        cl = classify_line(ir, po, gr)
        line_items.append(
            {
                "index": idx,
                "line_no": int(r.get("Line", idx + 1)),
                "name": r.get("Item_Name") or f"Item {idx+1}",
                "scenario_label": {
                    "s1": "Scenario 1",
                    "s2": "Scenario 2",
                    "s3": "Scenario 3",
                    "s6": "Scenario 6",
                    "s7": "Scenario 7",
                }[cl["scenario"]],
                "condition_label": cl.get("cond", ""),
                "ir": ir,
                "po": po,
                "gr": gr,
                "show_facts": multi,
                "edit_kind": cl["edit_kind"],  # 'po' | 'gr' | 'none'
                "suggest": cl["suggest"] if cl["edit_kind"] != "none" else 0.0,
                "button_hint": cl["button_hint"],
            }
        )

    if not line_items:
        return get_result_summary_card(
            [f"No actionable lines for invoice {invoice_number}."],
            title="Nothing to confirm",
            add_ai_prompt=True,
            invoice_number=invoice_number,
        )

    # Intro + button label
    if len(line_items) == 1:
        single = line_items[0]
        if single["edit_kind"] == "po":
            intro = (
                f"Hello {USER_NAME}, your open order has a PO amount of "
                f"{single['po']:,.0f}, and an invoice amount of {single['ir']:,.0f}.  "
                f"Would you like to conduct a PO amount change to {single['ir']:,.0f}?"
            )
        else:
            intro = (
                f"Hello {USER_NAME}, our records show you have an open order that has not been recorded as "
                '"received". Have you received this order?'
            )
    else:
        intro = (
            f"Hello {USER_NAME}, there are **{len(line_items)} line items** requiring your confirmation. "
            f"Have you received these items?"
        )

    if multi:
        kinds = {li["edit_kind"] for li in line_items}
        if kinds == {"none"}:
            label = "✅ Confirm received"
        elif kinds <= {"gr"}:
            label = "✅ Confirm partial received"
        elif "po" in kinds:
            label = "✅ Confirm change(s)"
        else:
            label = "✅ Confirm"
    else:
        hint = line_items[0]["button_hint"]
        label = {
            "received": "✅ Confirm received",
            "partial": "✅ Confirm partial received",
            "change": "✅ Confirm change",
            "changes": "✅ Confirm change",
        }.get(hint, "✅ Confirm Now")

    return get_unified_card(
        intro_text=intro,
        facts=facts,
        invoice_number=invoice_number,
        line_items=line_items,
        action_label=label,
        labels_override={
            "po": "New Purchase Order Amount",
            "gr": "New Goods Received Amount",
        },
    )


# ---------- follow-up push ----------
def schedule_follow_up(invoice_number: str):
    def _do():
        try:
            requests.post(
                f"{APP_PUBLIC_URL}/alerts",
                json={"invoice_number": invoice_number},
                timeout=10,
            )
        except Exception:
            pass

    threading.Timer(5.0, _do).start()


# ---------- helpers for adaptive actions ----------
def extract_submit_data(activity) -> Dict[str, Any]:
    """
    Teams + Action.Execute:
    - inputs могут прийти на верхнем уровне activity.value
    - часть полей в action.data
    Сливаем всё в один dict, не перетирая уже существующее.
    """
    val = getattr(activity, "value", None)
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except Exception:
            val = {}
    if not isinstance(val, dict):
        val = {}

    # 1) базовое (что было в action.data)
    data = {}
    if isinstance(val.get("action"), dict):
        data = val["action"].get("data") or {}
        if not isinstance(data, dict):
            data = {}
    # 2) добавить то, что в value.data
    extra = val.get("data") or {}
    if isinstance(extra, dict):
        for k, v in extra.items():
            if k not in data:
                data[k] = v
    # 3) добавить ВСЕ верхнеуровневые инпуты (value_0, value_1, invoice_number, и т.д.)
    for k, v in val.items():
        if k in ("data", "verb"):
            continue
        if k not in data:
            data[k] = v

    # Гарантировать, что data['action'] — строка-верб (для корректной маршрутизации)
    if not isinstance(data.get("action"), str) or not data.get("action"):
        # Попытка: достать из вложенного action.data.action (Action.Execute)
        if isinstance(val.get("action"), dict):
            nested = val["action"].get("data") or {}
            act_str = nested.get("action")
            if isinstance(act_str, str) and act_str:
                data["action"] = act_str
        # Fallback: если видно полезную нагрузку подтверждения — трактуем как unified_confirm
        if not isinstance(data.get("action"), str) or not data.get("action"):
            if ("row_count" in data) or any(
                isinstance(k, str) and (k.startswith("kind_") or k.startswith("line_"))
                for k in data.keys()
            ):
                data["action"] = "unified_confirm"

    return data


def extract_field_from_activity(activity, field: str) -> Optional[str]:
    val = getattr(activity, "value", None)
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except Exception:
            val = {}
    if isinstance(val, dict):
        return val.get(field)
    return None


# ---------- Bot ----------
class Bot(TeamsActivityHandler):
    async def on_members_added_activity(
        self, members_added: List[ChannelAccount], turn_context: TurnContext
    ):
        remember_reference(turn_context)
        # Don't send automatic welcome message to avoid double messages
        # User can type "hi" to get welcome card
        pass

    async def on_message_activity(self, turn_context: TurnContext):
        remember_reference(turn_context)
        data = extract_submit_data(turn_context.activity)

        act = (str(data.get("action") or "")).lower()
        if act == "unified_confirm":
            await self._confirm(turn_context, data)
            return
        if act == "validate_direct_prompt":
            await turn_context.send_activity(
                "Type `validate <invoice>` (e.g., `validate 5109058689`)."
            )
            return
        if act == "show_help":
            await turn_context.send_activity(
                "Examples: 'Which orders are not received?', 'Which orders are received?', "
                "'List my POs', 'Invoice 5109058689', 'all Po for Whiteboards', 'List all PO', "
                "'What is the status of invoice 5109058684', 'What is the total value of my orders?'."
            )
            return
        if act == "ask_ai":
            msg = (
                data.get("ai_prompt")
                or extract_field_from_activity(turn_context.activity, "ai_prompt")
                or ""
            )
            await self._route_and_reply(turn_context, msg)
            return

        # Plain chat message
        msg = turn_context.activity.text or ""
        await self._route_and_reply(turn_context, msg)

    async def on_invoke_activity(self, turn_context: TurnContext) -> InvokeResponse:
        remember_reference(turn_context)
        if (turn_context.activity.name or "").lower() == "adaptivecard/action":
            data = extract_submit_data(turn_context.activity)
            act = (str(data.get("action") or "")).lower()
            if act == "unified_confirm":
                await self._confirm(turn_context, data)
                return InvokeResponse(status=200)
            if act == "ask_ai":
                msg = data.get("ai_prompt") or ""
                await self._route_and_reply(turn_context, msg)
                return InvokeResponse(status=200)
        return await super().on_invoke_activity(turn_context)

    async def _route_and_reply(self, ctx: TurnContext, text_raw: str):
        text = _clean_text(text_raw)

        # validate <invoice>
        if text.lower().startswith("validate "):
            inv = text.split(" ", 1)[1].strip()
            await self._show_unified(ctx, inv)
            return

        # deterministic intents
        if MY_ORDERS_PATTERNS.search(text):
            await self._show_user_orders(ctx, USER_NAME, title="My POs")
            return

        if NOT_RECEIVED_Q.search(text):
            rows = [
                r
                for r in api.get_rows_for_user(USER_NAME)
                if (_to_float(r.get("IR_amount", 0)) > _to_float(r.get("GR_amount", 0)))
                and not bool(r.get("Received_Flag", False))
            ]
            await self._show_user_orders(
                ctx, USER_NAME, title="Orders not received", rows_override=rows
            )
            return

        if RECEIVED_Q.search(text):
            rows = [
                r
                for r in api.get_rows_for_user(USER_NAME)
                if _to_float(r.get("GR_amount", 0)) >= _to_float(r.get("IR_amount", 0))
                or bool(r.get("Received_Flag", False))
            ]
            await self._show_user_orders(
                ctx, USER_NAME, title="Orders received", rows_override=rows
            )
            return

        m = INVOICE_ONLY_RE.search(text)
        if m:
            await self._show_unified(ctx, m.group(1))
            return

        m = ITEM_FILTER_RE.search(text)
        if m:
            term = m.group(1).strip()
            rows = [
                r
                for r in api.get_rows_for_user(USER_NAME)
                if term.lower() in str(r.get("Item_Name", "")).lower()
            ]
            await self._show_user_orders(
                ctx, USER_NAME, title=f"POs for {term}", rows_override=rows
            )
            return

        if LIST_ALL_PO_RE.search(text):
            rows_user = api.get_rows_for_user(USER_NAME)
            rows_rec = [
                r
                for r in rows_user
                if _to_float(r.get("GR_amount", 0)) >= _to_float(r.get("IR_amount", 0))
                or bool(r.get("Received_Flag", False))
            ]
            rows_not = [r for r in rows_user if r not in rows_rec]
            await self._show_user_orders(
                ctx, USER_NAME, title="POs — Received", rows_override=rows_rec
            )
            await self._show_user_orders(
                ctx, USER_NAME, title="POs — Not received", rows_override=rows_not
            )
            return

        m = STATUS_Q.search(text)
        if m:
            inv = m.group(1)
            rows = api.get_invoice_lines(inv).get("rows", [])
            status_card = get_invoice_status_card(inv, rows)
            await ctx.send_activity(
                Activity(
                    type=ActivityTypes.message,
                    attachments=[
                        {
                            "contentType": "application/vnd.microsoft.card.adaptive",
                            "content": status_card,
                        }
                    ],
                )
            )
            items_card = get_invoice_items_card(inv, rows)
            await ctx.send_activity(
                Activity(
                    type=ActivityTypes.message,
                    attachments=[
                        {
                            "contentType": "application/vnd.microsoft.card.adaptive",
                            "content": items_card,
                        }
                    ],
                )
            )
            return

        if TOTALS_Q.search(text):
            rows = api.get_rows_for_user(USER_NAME)
            totals = get_totals_card(USER_NAME, rows)
            await ctx.send_activity(
                Activity(
                    type=ActivityTypes.message,
                    attachments=[
                        {
                            "contentType": "application/vnd.microsoft.card.adaptive",
                            "content": totals,
                        }
                    ],
                )
            )
            return

        # invoice items table (natural ask)
        m = INVOICE_ID_RE.search(text)
        if m and INVOICE_ITEMS_ASK_RE.search(text):
            await self._show_invoice_items(ctx, m.group(1))
            return

        # Check for simple greetings - show welcome card instead of domain guard
        simple_greetings = re.compile(
            r"(?is)^(hi|hello|hey|good morning|good afternoon|good evening|start|begin)$"
        )
        if simple_greetings.match(text.strip()):
            card = get_welcome_card(
                BRAND_IMAGE_URL, user_name=USER_NAME, include_ai=True
            )
            await ctx.send_activity(
                Activity(
                    type=ActivityTypes.message,
                    attachments=[
                        {
                            "contentType": "application/vnd.microsoft.card.adaptive",
                            "content": card,
                        }
                    ],
                )
            )
            return

        # domain guard for other non-relevant queries
        if not _domain_relevant(text):
            msg = (
                "This assistant only handles invoices, purchase orders (PO), and goods receipts (GR). "
                "Please provide an invoice number or ask about your orders."
            )
            card = get_result_summary_card(
                [msg],
                title=f"Info for {USER_NAME}",
                add_ai_prompt=True,
                invoice_number="",
            )
            await ctx.send_activity(
                Activity(
                    type=ActivityTypes.message,
                    attachments=[
                        {
                            "contentType": "application/vnd.microsoft.card.adaptive",
                            "content": card,
                        }
                    ],
                )
            )
            return

        # fallback to LLM
        m_any = INVOICE_ID_RE.search(text)
        inv_hint = m_any.group(1) if m_any else ""
        await self._ask_ai(ctx, {"ai_prompt": text_raw, "invoice_number": inv_hint})

    # ---- helpers to send cards ----
    async def _show_unified(self, ctx: TurnContext, invoice_number: str):
        card = build_unified_card(invoice_number)
        await ctx.send_activity(
            Activity(
                type=ActivityTypes.message,
                attachments=[
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": card,
                    }
                ],
            )
        )

    async def _show_user_orders(
        self,
        ctx: TurnContext,
        user_name: str,
        title: str = "My POs",
        rows_override: Optional[List[Dict[str, Any]]] = None,
    ):
        rows = (
            rows_override
            if rows_override is not None
            else api.get_rows_for_user(user_name)
        )
        card = get_user_orders_card(user_name, rows, title=title)
        await ctx.send_activity(
            Activity(
                type=ActivityTypes.message,
                attachments=[
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": card,
                    }
                ],
            )
        )

    async def _show_invoice_items(self, ctx: TurnContext, invoice_number: str):
        payload = api.get_invoice_lines(invoice_number)
        rows = payload.get("rows", [])
        card = get_invoice_items_card(invoice_number, rows)
        await ctx.send_activity(
            Activity(
                type=ActivityTypes.message,
                attachments=[
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": card,
                    }
                ],
            )
        )

    # ---------- confirmation ----------
    def _get_line_amounts(
        self, invoice_number: str, line_no: int
    ) -> Tuple[float, float]:
        rows = api.get_invoice_lines(invoice_number).get("rows", [])
        for r in rows:
            if int(r.get("Line", 0)) == int(line_no):
                ir = float(r.get("IR_amount", 0) or 0)
                gr = float(r.get("GR_amount", 0) or 0)
                return ir, gr
        return 0.0, 0.0

    async def _confirm(self, ctx: TurnContext, data: Dict[str, Any]):
        inv = data.get("invoice_number")
        n = int(data.get("row_count", 0))

        def _to_opt_float(x):
            try:
                return (
                    float(x)
                    if x
                    not in (
                        None,
                        "",
                    )
                    else None
                )
            except Exception:
                return None

        rows: List[Dict[str, Any]] = []
        for i in range(n):
            kind = (data.get(f"kind_{i}") or "none").lower()
            line = int(data.get(f"line_{i}", i + 1))
            cond = data.get(f"condition_{i}", "")
            val = _to_opt_float(data.get(f"value_{i}"))
            rows.append({"kind": kind, "line": line, "val": val, "cond": cond})

        # Fallback: if no rows parsed (some Teams clients omit data block for Action.Submit),
        # derive S1 rows from current invoice data so 'Confirm received' still sets flags.
        if not rows:
            inv_rows = api.get_invoice_lines(inv).get("rows", [])
            for rline in inv_rows:
                ir = _to_float(rline.get("IR_amount", 0))
                po = _to_float(rline.get("PO_amount", 0))
                gr = _to_float(rline.get("GR_amount", 0))
                cl = classify_line(ir, po, gr)
                if (cl.get("edit_kind") or "none") == "none":
                    line_no = int(rline.get("Line", 0) or 0)
                    cond = cl.get("cond", "")
                    rows.append(
                        {"kind": "none", "line": line_no, "val": None, "cond": cond}
                    )

        messages: List[str] = []
        po_changed = False

        for idx, r in enumerate(rows):
            ir, gr = self._get_line_amounts(inv, r["line"])  # текущие суммы

            if r["kind"] == "po":
                # S3/S6/S7 — меняем ТОЛЬКО PO; GR не трогаем
                api.update_po_amount(inv, r["line"], float(r["val"] or 0.0))
                messages.append(
                    f"{r['cond']}: PO Amount has been updated to {float(r['val'] or 0.0):,.0f} Euro.  "
                    f"New approval workflow has been initiated in Ariba, would you like to view the status of this order?"
                )
                po_changed = True

            elif r["kind"] == "gr":
                # 1) взять явный ввод; если пусто — fallback к suggest_i (если мы его передали из карточки)
                raw_val = r.get("val", None)
                if raw_val is None:
                    # поддержка fallback: suggest_0, suggest_1 ... если добавим это в карточку
                    raw_val = data.get(f"suggest_{idx}", None)

                if raw_val is None:
                    new_gr = gr  # нет ввода и нет suggest → не меняем сумму
                else:
                    new_gr = max(0.0, min(ir, float(raw_val)))  # clamp [0, IR]

                # 2) записать
                api.book_gr_amount(inv, r["line"], new_gr)

                # 3) требования кейса: в S2 флаг ставим даже при частичном приёме
                if new_gr > 0.0:
                    api.set_received_flag(inv, r["line"], True)

                messages.append(
                    f"{r['cond']}: Goods Receipt has been successfully submitted. View order status."
                )

            else:
                # S1 — НЕ менять суммы; только Received_Flag = True
                # (даже если GR < IR, флаг выставляем — таковы требования кейса S1)
                api.set_received_flag(inv, r["line"], True)
                messages.append(
                    f"{r['cond']}: Receipt status has been updated to 'Yes'."
                )

        if n == 1 and po_changed:
            # напоминание после изменения PO (как было)
            schedule_follow_up(inv)

        card = get_result_summary_card(
            messages,
            title="Confirmation results",
            add_ai_prompt=True,
            invoice_number=inv,
        )
        await ctx.send_activity(
            Activity(
                type=ActivityTypes.message,
                attachments=[
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": card,
                    }
                ],
            )
        )

    async def _ask_ai(self, ctx: TurnContext, data: Dict[str, Any]):
        inv = data.get("invoice_number") or ""
        user_prompt = (
            data.get("ai_prompt")
            or extract_field_from_activity(ctx.activity, "ai_prompt")
            or ""
        )
        user_prompt = (user_prompt or "").strip()
        if not user_prompt:
            await ctx.send_activity(
                f"{USER_NAME}, please type a question in the input field."
            )
            return

        context = (
            build_llm_context_for_invoice(inv)
            if inv
            else build_llm_context_for_user(USER_NAME)
        )
        guardrail = (
            "SYSTEM: You are a procurement assistant that ONLY answers questions "
            "about invoices, purchase orders (PO), and goods receipts (GR). "
            "If the user's question is not about invoices/PO/GR, reply exactly: "
            '"I can only help with invoices and purchase orders. Please provide an invoice number or ask about your orders."'
        )
        prompt = (
            f"{guardrail}\n\nAnswer concisely based only on the provided data. "
            f"If you don't have enough data, say so.\n\nUser: {USER_NAME}\nQuestion: {user_prompt}\n\nData:\n{context}"
        )

        if ask_llm_backend:
            try:
                answer = ask_llm_backend(prompt)
            except Exception as e:
                answer = f"(LLM error) {e}"
        else:
            answer = "(LLM unavailable) " + user_prompt

        card = get_result_summary_card(
            [answer],
            title=f"AI answer for {USER_NAME}",
            add_ai_prompt=True,
            invoice_number=inv,
        )
        await ctx.send_activity(
            Activity(
                type=ActivityTypes.message,
                attachments=[
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": card,
                    }
                ],
            )
        )


bot = Bot()


# ---------- Bot adapter plumbing ----------
async def run_activity(body, auth_header):
    activity = Activity().deserialize(body)

    async def aux(turn_context: TurnContext):
        await bot.on_turn(turn_context)

    return await adapter.process_activity(activity, auth_header, aux)


@app.post("/api/messages")
async def messages():
    if "application/json" not in request.headers.get("Content-Type", ""):
        return {"error": "Unsupported Media Type"}, 415
    body = request.json
    auth = request.headers.get("Authorization", "")
    await run_activity(body, auth)
    return Response(status=201)


# ---------- proactive alerts ----------
@app.post("/alerts")
async def alerts():
    data = request.json or {}
    inv = str(data.get("invoice_number") or data.get("invoice") or "")
    alert_type = str(data.get("type") or "generic")
    po_number = str(data.get("po_number") or "")

    if not inv:
        return {"error": "invoice required"}, 400
    if not CONV_REFS:
        return {"error": "No conversation yet"}, 400
    ref = list(CONV_REFS.values())[-1]

    # Special handling for case 2 (5109058677) - this is Scenario 2 (partial received)
    # We should NOT automatically mark as received, but let user confirm partial receipt
    # The logic will be handled in the unified card display and confirmation
    # No special pre-processing needed for this case

    card = build_unified_card(inv)

    async def aux(turn_context: TurnContext):
        await turn_context.send_activity(
            Activity(
                type=ActivityTypes.message,
                attachments=[
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": card,
                    }
                ],
            )
        )

    await adapter.continue_conversation(ref, aux, APP_ID)
    return {"ok": True}, 200
