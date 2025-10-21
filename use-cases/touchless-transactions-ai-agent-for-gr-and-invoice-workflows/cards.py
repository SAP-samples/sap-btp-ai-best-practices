from typing import Any, Dict, List, Optional

AC_VERSION = "1.4"


# ---------- Monospace table helpers (strict alignment) ----------

# ---- ColumnSet table helpers ----
# ---- ColumnSet table helpers ----
def _col(text: str, width: int | str, *, align: str = "Left", wrap: bool = False, header: bool = False) -> Dict[str, Any]:
    tb = {
        "type": "TextBlock",
        "text": text,
        "size": "Small",
        "horizontalAlignment": align,
        "wrap": bool(wrap),
    }
    if not wrap:
        tb["maxLines"] = 1
    if header:
        tb["weight"] = "Bolder"
    return {"type": "Column", "width": width, "items": [tb]}

def _header_cs(headers: List[str], weights: List[int | str], aligns: List[str]) -> Dict[str, Any]:
    return {
        "type": "ColumnSet",
        "spacing": "Small",
        "separator": True,
        "columns": [
            _col(headers[i], weights[i], align=aligns[i], wrap=False, header=True)
            for i in range(len(headers))
        ],
    }

def _row_cs(values: List[str], weights: List[int | str], aligns: List[str], nowrap_idx: List[int]) -> Dict[str, Any]:
    # жёсткая строка без переносов по всем колонкам
    return {
        "type": "ColumnSet",
        "spacing": "Small",
        "columns": [
            _col(values[i], weights[i], align=aligns[i], wrap=False, header=False)
            for i in range(len(values))
        ],
    }

def _clip_item(name: str, max_chars: int = 20) -> str:
    s = str(name or "").strip()
    return s if len(s) <= max_chars else s[: max(0, max_chars - 1)] + "…"





def _num_str(v: float) -> str:
    """Return decimal with dot only, 2 places, no thousands separators."""
    try:
        return f"{float(v):.2f}"  # e.g. 50000000.00
    except Exception:
        return "0.00"

def _max_len(values) -> int:
    return max((len(str(x)) for x in values if x is not None), default=0)

def _clip(s: str, width: int) -> str:
    s = str(s or "")
    return s[:width] if len(s) > width else s

def _lpad(s: str, width: int) -> str:
    return str(s).ljust(width, "\u00A0")

def _rpad(s: str, width: int) -> str:
    return str(s).rjust(width, "\u00A0")

def _mono_block(lines: list[str]) -> dict:
    """One monospace TextBlock with all lines."""
    return {
        "type": "TextBlock",
        "fontType": "Monospace",
        "size": "Small",
        "wrap": True,
        "text": "\n".join(lines) 
    }

def _compute_widths_for_orders(rows, *, max_item=18, max_num=14):
    """
    Compute column widths from data so header sits exactly above values.
    Returns (w_inv, w_item, w_num, w_rec, total_len)
    """
    inv_vals  = [str(r.get("Invoice_Number","")) for r in rows]
    item_vals = [str(r.get("Item_Name","")) for r in rows]
    ir_vals   = [_num_str(r.get("IR_amount",0)) for r in rows]
    po_vals   = [_num_str(r.get("PO_amount",0)) for r in rows]
    gr_vals   = [_num_str(r.get("GR_amount",0)) for r in rows]
    rec_vals  = [("Yes" if (r.get("Received_Flag") or False) else "No") for r in rows]

    w_inv  = max(len("Invoice"),  _max_len(inv_vals)) + 1
    w_item = max(len("Item"),     min(max_item, _max_len(item_vals))) + 1
    w_num  = max(len("IR"),       min(max_num, _max_len(ir_vals+po_vals+gr_vals))) + 1
    w_rec  = max(len("Received"), _max_len(rec_vals)) + 0  # без хвостового пробела

    total = w_inv + w_item + 3*w_num + w_rec + 4  # пробелы между колонками
    return w_inv, w_item, w_num, w_rec, total

def _row_orders(r, w_inv, w_item, w_num, w_rec):
    inv = _lpad(str(r.get("Invoice_Number","")), w_inv)
    item = _lpad(_clip(str(r.get("Item_Name","")), w_item-1), w_item)
    ir = _rpad(_num_str(r.get("IR_amount",0)), w_num)
    po = _rpad(_num_str(r.get("PO_amount",0)), w_num)
    gr = _rpad(_num_str(r.get("GR_amount",0)), w_num)
    rec = "Yes" if (r.get("Received_Flag") or False) else "No"
    rec = _lpad(rec, w_rec)
    return f"{inv} {item} {ir} {po} {gr} {rec}"

def _compute_widths_for_invoice_items(rows, *, max_item=18, max_num=14):
    line_vals = [str(r.get("Line","")) for r in rows]
    item_vals = [str(r.get("Item_Name","")) for r in rows]
    ir_vals   = [_num_str(r.get("IR_amount",0)) for r in rows]
    po_vals   = [_num_str(r.get("PO_amount",0)) for r in rows]
    gr_vals   = [_num_str(r.get("GR_amount",0)) for r in rows]
    rec_vals  = [("Yes" if ((r.get("GR_amount",0) or 0) >= (r.get("IR_amount",0) or 0) or bool(r.get("Received_Flag",False))) else "No") for r in rows]

    w_line = max(len("Line"), _max_len(line_vals)) + 1
    w_item = max(len("Item"), min(max_item, _max_len(item_vals))) + 1
    w_num  = max(len("IR"),   min(max_num, _max_len(ir_vals+po_vals+gr_vals))) + 1
    w_rec  = max(len("Received"), _max_len(rec_vals))

    total = w_line + w_item + 3*w_num + w_rec + 4
    return w_line, w_item, w_num, w_rec, total

def _row_invoice_item(r, w_line, w_item, w_num, w_rec):
    line = _lpad(str(int(r.get("Line",0))), w_line)
    item = _lpad(_clip(str(r.get("Item_Name","")), w_item-1), w_item)
    ir = _rpad(_num_str(r.get("IR_amount",0)), w_num)
    po = _rpad(_num_str(r.get("PO_amount",0)), w_num)
    gr = _rpad(_num_str(r.get("GR_amount",0)), w_num)
    rec = "Yes" if ((r.get("GR_amount",0) or 0) >= (r.get("IR_amount",0) or 0) or bool(r.get("Received_Flag",False))) else "No"
    rec = _lpad(rec, w_rec)
    return f"{line} {item} {ir} {po} {gr} {rec}"



def _euro(v: float) -> str:
    try:
        return f"{float(v):,.2f}\u00a0€"  # NBSP перед €
    except Exception:
        return "0.00\u00a0€"


def _textblock(text: str, **kwargs) -> Dict[str, Any]:
    base = {"type": "TextBlock", "wrap": True, "text": text}
    base.update(kwargs)
    return base


def _factset_from_dict(facts: Dict[str, str]) -> Dict[str, Any]:
    return {
        "type": "FactSet",
        "facts": [{"title": k, "value": str(v)} for k, v in facts.items()],
    }


def _clip_ljust(s: str, w: int) -> str:
    s = str(s or "")
    s = s[:w] if len(s) > w else s
    return s.ljust(w, "\u00A0")


def _num_rjust(v, w: int) -> str:
    try:
        x = float(v or 0)
    except Exception:
        x = 0.0
    # только точка, без тысячных запятых
    s = f"{x:.2f}"  # 50000000.00
    return s.rjust(w, " ")


def _mono_row(
    inv, item, ir, po, gr, rec, W_INV=10, W_ITEM=12, W_NUM=14, W_REC=8
) -> str:
    # item: обрезаем до 12, добиваем пробелами
    return (
        _clip_ljust(inv, W_INV)
        + " "
        + _clip_ljust(item, W_ITEM)
        + " "
        + _num_rjust(ir, W_NUM)
        + " "
        + _num_rjust(po, W_NUM)
        + " "
        + _num_rjust(gr, W_NUM)
        + " "
        + _clip_ljust(rec, W_REC)
    )


def _mono_table_block(lines: list[str]) -> dict:
    """
    Render each line as its own TextBlock to avoid Teams swallowing newlines.
    Monospace + wrap=False keeps columns fixed and prevents mid-line wraps.
    """
    items = []
    for i, line in enumerate(lines):
        tb = {
            "type": "TextBlock",
            "fontType": "Monospace",
            "size": "Small",
            "wrap": False,         # не переносить внутри строки
            "text": line,
        }
        if i == 0:                 # шапку делаем жирной
            tb["weight"] = "Bolder"
        items.append(tb)

    return {
        "type": "Container",
        "spacing": "Small",
        "items": items
    }



# ---------- helpers ----------
def _received_yes_no(row: Dict[str, Any]) -> str:
    try:
        rf = bool(row.get("Received_Flag", False))
        ir = float(row.get("IR_amount", 0) or 0)
        gr = float(row.get("GR_amount", 0) or 0)
        return "Yes" if (rf or gr >= ir) else "No"
    except Exception:
        return "No"


# ---------- Unified ----------
def _item_container(
    idx: int, item: Dict[str, Any], label_po: str, label_gr: str, multi_show_facts: bool
) -> Dict[str, Any]:
    blocks: List[Dict[str, Any]] = []
    title = (
        f"Item {idx + 1} — {item.get('name','Item')} (Line {item.get('line_no','?')})"
    )
    blocks.append(_textblock(title, weight="Bolder", size="Medium"))
    blocks.append(_textblock("Condition", weight="Bolder", spacing="Small"))
    blocks.append(_textblock(item.get("condition_label", "—")))

    if multi_show_facts or item.get("show_facts"):
        facts = {
            "PO Amount": _euro(item.get("po", 0)),
            "GR Amount": _euro(item.get("gr", 0)),
            "IR Amount": _euro(item.get("ir", 0)),
        }
        blocks.append(
            {
                "type": "Container",
                "items": [_factset_from_dict(facts)],
                "spacing": "Small",
            }
        )

    kind = (item.get("edit_kind") or "none").lower()
    suggested = float(item.get("suggest", 0.0) or 0.0)

    if kind == "po":
        blocks.append(
            {
                "type": "Input.Number",
                "id": f"value_{idx}",
                "label": label_po,
                "value": suggested,
                "min": 0,
            }
        )
    elif kind == "gr":
        blocks.append(
            {
                "type": "Input.Number",
                "id": f"value_{idx}",
                "label": label_gr,
                "value": suggested,
                "min": 0,
            }
        )
    else:
        blocks.append(
            {
                "type": "Input.Text",
                "id": f"value_{idx}",
                "value": str(suggested),
                "isVisible": False,
            }
        )

    return {
        "type": "Container",
        "items": blocks,
        "style": "default",
        "bleed": False,
        "spacing": "Medium",
    }


def get_unified_card(
    *,
    intro_text: str,
    facts: Dict[str, str],
    invoice_number: str,
    line_items: List[Dict[str, Any]],
    action_label: str,
    labels_override: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    labels_override = labels_override or {}
    label_po = labels_override.get("po", "New Purchase Order Amount")
    label_gr = labels_override.get("gr", "New Goods Received Amount")

    body: List[Dict[str, Any]] = []
    body.append(_textblock(intro_text, size="Medium", weight="Bolder"))
    body.append(_factset_from_dict(facts))

    multi = len(line_items) > 1
    if multi:
        body.append(
            _textblock("Line items", size="Medium", weight="Bolder", spacing="Medium")
        )

    for i, li in enumerate(line_items):
        body.append(_item_container(i, li, label_po, label_gr, multi_show_facts=multi))

    data: Dict[str, Any] = {
        "action": "unified_confirm",
        "invoice_number": invoice_number,
        "row_count": len(line_items),
    }
    for i, li in enumerate(line_items):
        data[f"kind_{i}"] = li.get("edit_kind") or "none"
        data[f"line_{i}"] = li.get("line_no", i + 1)
        data[f"scenario_{i}"] = li.get("scenario_label", "")
        data[f"condition_{i}"] = li.get("condition_label", "")

    actions = [{"type": "Action.Submit", "title": action_label, "data": data}]

    return {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": AC_VERSION,
        "body": body,
        "actions": actions,
    }


# ---------- Result summary ----------
def get_result_summary_card(
    messages: List[str],
    title: str = "Confirmation results",
    add_ai_prompt: bool = False,
    invoice_number: Optional[str] = None,
) -> Dict[str, Any]:
    items = []
    if title:
        items.append(_textblock(title, size="Medium", weight="Bolder"))
    for msg in messages:
        items.append(_textblock(f"• {msg}"))

    actions: List[Dict[str, Any]] = [
        {
            "type": "Action.OpenUrl",
            "title": "Open in Ariba",
            "url": "https://www.ariba.com/",
        }
    ]

    return {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": AC_VERSION,
        "body": [{"type": "Container", "items": items}],
        "actions": actions,
    }


# ---------- Welcome ----------
def get_welcome_card(
    brand_url: Optional[str] = None, user_name: str = "Marta", include_ai: bool = True
) -> Dict[str, Any]:
    body: List[Dict[str, Any]] = []
    if brand_url:
        body.append(
            {
                "type": "Image",
                "url": brand_url,
                "size": "Small",
                "style": "Person",
                "horizontalAlignment": "Left",
            }
        )
    body.append(_textblock(f"Welcome, {user_name}!", size="Large", weight="Bolder"))
    body.append(
        _textblock(
            "I handle invoice validation, goods receipt confirmation, and PO changes directly in Teams.",
            spacing="Small",
        )
    )

    actions: List[Dict[str, Any]] = [
        {
            "type": "Action.Submit",
            "title": "Validate invoice",
            "data": {"action": "validate_direct_prompt"},
        }
    ]

    return {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": AC_VERSION,
        "body": body,
        "actions": actions,
    }


# ---------- User orders table ----------
def get_user_orders_card(
    user_name: str,
    rows: List[Dict[str, Any]],
    *,
    title: str = "My POs",
    limit: int = 50,
) -> Dict[str, Any]:
    rows = list(rows or [])
    rows.sort(key=lambda r: (str(r.get("Invoice_Number", "")), int(r.get("Line", 0))))
    total = len(rows)
    shown = rows[:limit]

    body: List[Dict[str, Any]] = []
    body.append(_textblock(f"{title} — {user_name}", size="Medium", weight="Bolder"))
    body.append(
        _textblock(
            f"Total rows: {total}{' (showing first ' + str(limit) + ')' if total > limit else ''}",
            spacing="Small",
        )
    )

    # сетка колонок (Invoice шире на 3 за счёт Item)
    headers = ["Invoice", "Item", "IR", "PO", "GR", "Rec"]
    weights = [21, 29, 18, 18, 18, 10]  # было [18,32,18,18,18,10]
    aligns  = ["Left", "Left", "Right", "Right", "Right", "Center"]
    nowrap  = [0, 1, 2, 3, 4, 5]

    # шапка — СВЕРХУ
    body.append(_header_cs(headers, weights, aligns))

    for r in shown:
        inv = str(r.get("Invoice_Number", "")).strip()
        item = _clip_item(r.get("Item_Name", ""), max_chars=18)  # чуть уже
        ir = _num_str(r.get("IR_amount", 0))
        po = _num_str(r.get("PO_amount", 0))
        gr = _num_str(r.get("GR_amount", 0))
        rec = "Yes" if ((r.get("GR_amount", 0) or 0) >= (r.get("IR_amount", 0) or 0) or bool(r.get("Received_Flag", False))) else "No"

        body.append(_row_cs([inv, item, ir, po, gr, rec], weights, aligns, nowrap))

    return {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": AC_VERSION,
        "body": body,
    }





# ---------- Invoice items ----------
def get_invoice_items_card(
    invoice_number: str, rows: List[Dict[str, Any]], limit: int = 100
) -> Dict[str, Any]:
    rows = [r for r in (rows or []) if str(r.get("Invoice_Number", "")) == str(invoice_number)]
    rows.sort(key=lambda r: int(r.get("Line", 0)))
    total = len(rows)
    shown = rows[:limit]

    body: List[Dict[str, Any]] = []
    body.append(_textblock(f"Invoice {invoice_number} — items", size="Medium", weight="Bolder"))
    body.append(
        _textblock(
            f"Lines: {total}{' (showing first ' + str(limit) + ')' if total > limit else ''}",
            spacing="Small",
        )
    )

    headers = ["Line", "Item", "IR", "PO", "GR", "Rec"]
    weights = [12, 32, 18, 18, 18, 10]
    aligns  = ["Left", "Left", "Right", "Right", "Right", "Center"]
    nowrap  = [0, 1, 2, 3, 4, 5]

    body.append(_header_cs(headers, weights, aligns))

    for r in shown:
        line = str(int(r.get("Line", 0)))
        item = _clip_item(r.get("Item_Name", ""), max_chars=20)
        ir = _num_str(r.get("IR_amount", 0))
        po = _num_str(r.get("PO_amount", 0))
        gr = _num_str(r.get("GR_amount", 0))
        rec = "Yes" if ((r.get("GR_amount", 0) or 0) >= (r.get("IR_amount", 0) or 0) or bool(r.get("Received_Flag", False))) else "No"

        body.append(_row_cs([line, item, ir, po, gr, rec], weights, aligns, nowrap))

    return {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": AC_VERSION,
        "body": body,
    }




# ---------- Totals & Status ----------
def get_totals_card(user_name: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    delivered = [
        r
        for r in rows
        if (r.get("GR_amount", 0) or 0) >= (r.get("IR_amount", 0) or 0)
        or bool(r.get("Received_Flag", False))
    ]
    pending = [r for r in rows if r not in delivered]
    sum_delivered = sum(float(r.get("IR_amount", 0) or 0) for r in delivered)
    sum_pending = sum(float(r.get("IR_amount", 0) or 0) for r in pending)

    facts = {
        "Delivered (count)": f"{len(delivered)}",
        "Delivered (IR total)": _euro(sum_delivered),
        "Not delivered (count)": f"{len(pending)}",
        "Not delivered (IR total)": _euro(sum_pending),
        "All orders (IR total)": _euro(sum_delivered + sum_pending),
    }
    return {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": AC_VERSION,
        "body": [
            _textblock(f"Totals for {user_name}", size="Medium", weight="Bolder"),
            _factset_from_dict(facts),
        ],
    }


def get_invoice_status_card(
    invoice_number: str, rows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    rows = [r for r in rows if str(r.get("Invoice_Number", "")) == str(invoice_number)]
    sum_ir = sum(float(r.get("IR_amount", 0) or 0) for r in rows)
    sum_po = sum(float(r.get("PO_amount", 0) or 0) for r in rows)
    sum_gr = sum(float(r.get("GR_amount", 0) or 0) for r in rows)

    # Check delivery status - all lines must be received
    delivered = (
        all(
            ((r.get("GR_amount", 0) or 0) >= (r.get("IR_amount", 0) or 0))
            or bool(r.get("Received_Flag", False))
            for r in rows
        )
        if rows
        else False
    )

    # Count delivered vs pending lines
    delivered_lines = sum(
        1
        for r in rows
        if ((r.get("GR_amount", 0) or 0) >= (r.get("IR_amount", 0) or 0))
        or bool(r.get("Received_Flag", False))
    )
    pending_lines = len(rows) - delivered_lines

    facts = {
        "Invoice #": str(invoice_number),
        "Overall Status": "✅ Delivered" if delivered else "⏳ Pending",
        "Lines Delivered": f"{delivered_lines} of {len(rows)}",
        "Lines Pending": str(pending_lines),
        "Invoice amount (IR)": _euro(sum_ir),
        "PO amount": _euro(sum_po),
        "GR amount": _euro(sum_gr),
    }
    return {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": AC_VERSION,
        "body": [
            _textblock("Invoice status", size="Medium", weight="Bolder"),
            _factset_from_dict(facts),
        ],
    }
