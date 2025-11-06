"""
LangChain-native tools for the API's LangGraph chatbot.

Implements one tool mirroring the tutorial examples:
- calculator: safely evaluates arithmetic expressions
"""

from __future__ import annotations

import ast
import time
import os
import csv
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Any, Dict, Optional, List, Tuple, Iterable

from langchain_core.tools import tool


def _safe_eval_arithmetic(expression: str) -> float:
    """Safely evaluate a basic arithmetic expression using Python's AST."""
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Num,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Constant,
        ast.Load,
        ast.Mod,
        ast.FloorDiv,
    )

    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError("Disallowed expression component")

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant):
            return float(n.value)
        if isinstance(n, ast.Num):
            return float(n.n)
        if isinstance(n, ast.BinOp):
            left = _eval(n.left)
            right = _eval(n.right)
            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
            if isinstance(n.op, ast.Div):
                return left / right
            if isinstance(n.op, ast.Pow):
                return left**right
            if isinstance(n.op, ast.Mod):
                return left % right
            if isinstance(n.op, ast.FloorDiv):
                return left // right
            raise ValueError("Unsupported operator")
        if isinstance(n, ast.UnaryOp):
            operand = _eval(n.operand)
            if isinstance(n.op, ast.UAdd):
                return +operand
            if isinstance(n.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary operator")
        raise ValueError("Unsupported expression type")

    return _eval(tree)


@tool("calculator")
def calculator_tool(expression: str) -> Dict[str, Any]:
    """Evaluate an arithmetic `expression` like `2*(3+4)` and return the result."""
    expr = str(expression).strip()
    if not expr:
        raise ValueError("calculator: 'expression' is required")
    value = _safe_eval_arithmetic(expr)
    # Add a wait/delay
    time.sleep(1)
    return {"expression": expr, "result": value}


# -----------------------------
# PO CSV analytics tools
# -----------------------------


_DEFAULT_PO_CSV = (
    Path(__file__).resolve().parents[2] / "mock" / "data" / "po_data_seed.csv"
)
_PO_CSV_PATH = Path(os.getenv("PO_CSV_PATH", str(_DEFAULT_PO_CSV)))
_CSV_CACHE: Dict[str, Any] = {"path": None, "mtime": None, "rows": None}

# Scrapping CSV
_DEFAULT_SCRAP_CSV = (
    Path(__file__).resolve().parents[2] / "mock" / "data" / "scrapping_data_seed.csv"
)
_SCRAP_CSV_PATH = Path(os.getenv("SCRAP_CSV_PATH", str(_DEFAULT_SCRAP_CSV)))
_SCRAP_CACHE: Dict[str, Any] = {"path": None, "mtime": None, "rows": None}


def _get_today() -> date:
    override = os.getenv("PO_TODAY")
    if override:
        try:
            return datetime.strptime(override, "%Y-%m-%d").date()
        except Exception:
            pass
    return date.today()


def _parse_date(value: str) -> Optional[date]:
    v = (value or "").strip()
    if not v:
        return None
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(v, fmt).date()
        except Exception:
            continue
    return None


def _to_number(value: str) -> float:
    v = (value or "").strip()
    if not v:
        return 0.0
    try:
        return float(v)
    except Exception:
        return 0.0


def _load_po_rows() -> List[Dict[str, Any]]:
    path = str(_PO_CSV_PATH)
    try:
        mtime = os.path.getmtime(path)
    except Exception:
        mtime = None

    if (
        _CSV_CACHE.get("path") == path
        and _CSV_CACHE.get("mtime") == mtime
        and _CSV_CACHE.get("rows") is not None
    ):
        return _CSV_CACHE["rows"]  # type: ignore

    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Normalize header keys (strip whitespace and BOM if present)
            r = {(k or "").strip().lstrip("\ufeff"): v for k, v in r.items()}
            # Normalize keys and parse types we care about
            plant = (r.get("Plant") or "").strip()
            material = (r.get("Material Number") or "").strip()
            maktx = (r.get("MAKTX") or "").strip()
            date_str = (r.get("Delivery Date (Requested)") or "").strip()
            requested = _to_number(r.get("Purchase Order Quantity (Requested)") or "")
            delivered = _to_number(r.get("Purchase Order Quantity (Delivered)") or "")
            planned_input = _to_number(r.get("Planned Input (lbs)") or "")
            actual_input = _to_number(r.get("Actual Input (lbs)") or "")
            rows.append(
                {
                    "Plant": plant,
                    "Material Number": material,
                    "MAKTX": maktx,
                    "Delivery Date (Requested)": date_str,
                    "_date": _parse_date(date_str),
                    "Purchase Order Quantity (Requested)": requested,
                    "Purchase Order Quantity (Delivered)": delivered,
                    "Planned Input (lbs)": planned_input,
                    "Actual Input (lbs)": actual_input,
                }
            )

    _CSV_CACHE.update({"path": path, "mtime": mtime, "rows": rows})
    return rows


def _parse_scrap_date(value: str) -> Optional[date]:
    v = (value or "").strip()
    if not v:
        return None
    # Common formats in scrapping_data_seed.csv: "20-Jun-23", "3-May-24"
    fmts = ("%d-%b-%y", "%Y-%m-%d", "%m/%d/%Y")
    for fmt in fmts:
        try:
            return datetime.strptime(v, fmt).date()
        except Exception:
            continue
    return None


def _load_scrap_rows() -> List[Dict[str, Any]]:
    path = str(_SCRAP_CSV_PATH)
    try:
        mtime = os.path.getmtime(path)
    except Exception:
        mtime = None

    if (
        _SCRAP_CACHE.get("path") == path
        and _SCRAP_CACHE.get("mtime") == mtime
        and _SCRAP_CACHE.get("rows") is not None
    ):
        return _SCRAP_CACHE["rows"]  # type: ignore

    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r = {(k or "").strip().lstrip("\ufeff"): v for k, v in r.items()}
            plant = (r.get("Plant") or "").strip()
            date_str = (r.get("Date of Manufacture") or "").strip()
            material = (r.get("Material Number") or "").strip()
            maktx = (r.get("MAKTX") or "").strip()
            meins = (r.get("MEINS") or "").strip()
            qty = _to_number(r.get("Quantity") or "")
            rows.append(
                {
                    "Plant": plant,
                    "Date of Manufacture": date_str,
                    "_date": _parse_scrap_date(date_str),
                    "Material Number": material,
                    "MAKTX": maktx,
                    "MEINS": meins,
                    "Quantity": qty,
                }
            )

    _SCRAP_CACHE.update({"path": path, "mtime": mtime, "rows": rows})
    return rows


def _resolve_time_window(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    last_n_days: Optional[int] = None,
    last_n_months: Optional[int] = None,
) -> Tuple[Optional[date], Optional[date]]:
    today = _get_today()

    # Explicit dates take precedence
    start = _parse_date(start_date) if start_date else None
    end = _parse_date(end_date) if end_date else None

    if start or end:
        if start is None:
            start = date.min
        if end is None:
            end = today
        return (start, end)

    # Relative windows
    if last_n_days and last_n_days > 0:
        start = today - timedelta(days=last_n_days - 1)
        end = today
        return (start, end)

    if last_n_months and last_n_months > 0:
        # Start at first day of (today - last_n_months + 1) months
        year = today.year
        month = today.month - (last_n_months - 1)
        while month <= 0:
            month += 12
            year -= 1
        start = date(year, month, 1)
        end = today
        return (start, end)

    return (None, None)


def _within_window(
    d: Optional[date], start: Optional[date], end: Optional[date]
) -> bool:
    if d is None:
        return False
    if start and d < start:
        return False
    if end and d > end:
        return False
    return True


def _sort_by_percent_desc(
    rows: List[Dict[str, Any]], percent_key: str
) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda x: (x.get(percent_key) is None, x.get(percent_key, 0.0)),
        reverse=True,
    )


def _highest_lowest(
    rows: List[Dict[str, Any]], percent_key: str
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    valid = [r for r in rows if isinstance(r.get(percent_key), (int, float))]
    if not valid:
        return (None, None)
    highest = max(valid, key=lambda r: r[percent_key])
    lowest = min(valid, key=lambda r: r[percent_key])
    return (highest, lowest)


def _ym_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def _round2(value: float) -> float:
    return round(value, 2)


def _iter_months(start: date, end: date) -> Iterable[str]:
    """Yield YYYY-MM strings from the month of start to the month of end inclusive."""
    if start > end:
        return []
    y, m = start.year, start.month
    while True:
        yield f"{y:04d}-{m:02d}"
        if y == end.year and m == end.month:
            break
        m += 1
        if m > 12:
            m = 1
            y += 1


@tool("po_yield_by_plant_for_material")
async def po_yield_by_plant_for_material(
    material_number: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    last_n_days: Optional[int] = None,
    last_n_months: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute Yield% per Plant for a given material number over an optional time window.

    Yield% = Sum(Delivered) / Sum(Actual Input) * 100
    """
    material_number = (material_number or "").strip()
    if not material_number:
        return {"error": "material_number is required", "rows": []}

    start, end = _resolve_time_window(start_date, end_date, last_n_days, last_n_months)
    rows = _load_po_rows()

    groups: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if r["Material Number"] != material_number:
            continue
        if not _within_window(r.get("_date"), start, end):
            continue
        plant = r["Plant"]
        g = groups.setdefault(
            plant,
            {
                "plant": plant,
                "materialNumber": material_number,
                "delivered": 0.0,
                "actualInput": 0.0,
            },
        )
        g["delivered"] += r["Purchase Order Quantity (Delivered)"]
        g["actualInput"] += r["Actual Input (lbs)"]

    out: List[Dict[str, Any]] = []
    for g in groups.values():
        denom = g["actualInput"]
        if denom <= 0:
            continue
        pct = _round2((g["delivered"] / denom) * 100.0)
        g["yieldPct"] = pct
        out.append(g)

    out = _sort_by_percent_desc(out, "yieldPct")
    highest, lowest = _highest_lowest(out, "yieldPct")
    return {"rows": out, "highest": highest, "lowest": lowest}


@tool("po_efficiency_by_plant_for_material")
async def po_efficiency_by_plant_for_material(
    material_number: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    last_n_days: Optional[int] = None,
    last_n_months: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute Efficiency% per Plant for a given material number over an optional time window.

    Efficiency% = Sum(Delivered) / Sum(Requested) * 100
    """
    material_number = (material_number or "").strip()
    if not material_number:
        return {"error": "material_number is required", "rows": []}

    start, end = _resolve_time_window(start_date, end_date, last_n_days, last_n_months)
    rows = _load_po_rows()

    groups: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if r["Material Number"] != material_number:
            continue
        if not _within_window(r.get("_date"), start, end):
            continue
        plant = r["Plant"]
        g = groups.setdefault(
            plant,
            {
                "plant": plant,
                "materialNumber": material_number,
                "delivered": 0.0,
                "requested": 0.0,
            },
        )
        g["delivered"] += r["Purchase Order Quantity (Delivered)"]
        g["requested"] += r["Purchase Order Quantity (Requested)"]

    out: List[Dict[str, Any]] = []
    for g in groups.values():
        denom = g["requested"]
        if denom <= 0:
            continue
        pct = _round2((g["delivered"] / denom) * 100.0)
        g["efficiencyPct"] = pct
        out.append(g)

    out = _sort_by_percent_desc(out, "efficiencyPct")
    highest, lowest = _highest_lowest(out, "efficiencyPct")
    return {"rows": out, "highest": highest, "lowest": lowest}


@tool("po_yield_by_material_at_plant")
async def po_yield_by_material_at_plant(
    plant: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    last_n_days: Optional[int] = None,
    last_n_months: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute Yield% per Material Number at a given Plant over an optional time window.

    Yield% = Sum(Delivered) / Sum(Actual Input) * 100
    """
    plant = (plant or "").strip()
    if not plant:
        return {"error": "plant is required", "rows": []}

    start, end = _resolve_time_window(start_date, end_date, last_n_days, last_n_months)
    rows = _load_po_rows()

    groups: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if r["Plant"] != plant:
            continue
        if not _within_window(r.get("_date"), start, end):
            continue
        material = r["Material Number"]
        g = groups.setdefault(
            material,
            {
                "plant": plant,
                "materialNumber": material,
                "delivered": 0.0,
                "actualInput": 0.0,
            },
        )
        g["delivered"] += r["Purchase Order Quantity (Delivered)"]
        g["actualInput"] += r["Actual Input (lbs)"]

    out: List[Dict[str, Any]] = []
    for g in groups.values():
        denom = g["actualInput"]
        if denom <= 0:
            continue
        pct = _round2((g["delivered"] / denom) * 100.0)
        g["yieldPct"] = pct
        out.append(g)

    out = _sort_by_percent_desc(out, "yieldPct")
    highest, lowest = _highest_lowest(out, "yieldPct")
    return {"rows": out, "highest": highest, "lowest": lowest}


@tool("po_yield_by_month")
async def po_yield_by_month(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    last_n_days: Optional[int] = None,
    last_n_months: Optional[int] = 12,
) -> Dict[str, Any]:
    """Compute Yield% per Month over an optional time window (default last 12 months).

    Groups by YYYY-MM using Delivery Date (Requested).
    """
    start, end = _resolve_time_window(start_date, end_date, last_n_days, last_n_months)
    rows = _load_po_rows()

    groups: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        d = r.get("_date")
        if not _within_window(d, start, end):
            continue
        if d is None:
            continue
        key = _ym_key(d)
        g = groups.setdefault(key, {"month": key, "delivered": 0.0, "actualInput": 0.0})
        g["delivered"] += r["Purchase Order Quantity (Delivered)"]
        g["actualInput"] += r["Actual Input (lbs)"]

    out: List[Dict[str, Any]] = []
    for g in groups.values():
        denom = g["actualInput"]
        if denom <= 0:
            continue
        pct = _round2((g["delivered"] / denom) * 100.0)
        g["yieldPct"] = pct
        out.append(g)

    # Sort by month ascending but also compute highest/lowest by percent
    out = sorted(out, key=lambda r: r["month"]) if out else out
    highest, lowest = _highest_lowest(out, "yieldPct")
    return {"rows": out, "highest": highest, "lowest": lowest}


@tool("po_query")
async def po_query(
    plant: Optional[str] = None,
    material_number: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    last_n_days: Optional[int] = None,
    last_n_months: Optional[int] = None,
    group_by: Optional[str] = None,
    metric: Optional[str] = None,
    include_empty_months: Optional[bool] = False,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Generic PO data access with optional filters and aggregation.

    Filters:
    - plant: filter by Plant (exact)
    - material_number: filter by Material Number (exact)
    - time window: start_date/end_date (YYYY-MM-DD or MM/DD/YYYY) or last_n_days/last_n_months

    Aggregation (optional):
    - group_by: one of "plant", "material", "month"
    - metric: when grouping, choose "yield" (delivered / actualInput) or "efficiency" (delivered / requested)
    - include_empty_months: when group_by="month", fill missing months in window with zeros (yield/efficiency omitted)

    If group_by is omitted, returns filtered raw rows (selected fields). Otherwise returns aggregated rows.
    """
    start, end = _resolve_time_window(start_date, end_date, last_n_days, last_n_months)
    rows = _load_po_rows()

    def _passes_filters(r: Dict[str, Any]) -> bool:
        if plant and r.get("Plant") != plant:
            return False
        if material_number and r.get("Material Number") != material_number:
            return False
        if not _within_window(r.get("_date"), start, end):
            return False
        return True

    filtered = [r for r in rows if _passes_filters(r)]

    if not group_by:
        out_rows: List[Dict[str, Any]] = []
        for r in filtered:
            out_rows.append(
                {
                    "plant": r.get("Plant"),
                    "materialNumber": r.get("Material Number"),
                    "maktx": r.get("MAKTX"),
                    "date": r.get("_date").isoformat() if r.get("_date") else None,
                    "requested": r.get("Purchase Order Quantity (Requested)"),
                    "delivered": r.get("Purchase Order Quantity (Delivered)"),
                    "actualInput": r.get("Actual Input (lbs)"),
                }
            )
        if isinstance(limit, int) and limit > 0:
            out_rows = out_rows[:limit]
        return {"rows": out_rows}

    group_by = (group_by or "").lower()
    metric = (metric or "yield").lower()
    if group_by not in {"plant", "material", "month"}:
        return {"error": "group_by must be one of 'plant','material','month'"}
    if metric not in {"yield", "efficiency"}:
        return {"error": "metric must be 'yield' or 'efficiency'"}

    groups: Dict[str, Dict[str, Any]] = {}

    def _key_for(r: Dict[str, Any]) -> Optional[str]:
        if group_by == "plant":
            return r.get("Plant")
        if group_by == "material":
            return r.get("Material Number")
        d = r.get("_date")
        if group_by == "month" and isinstance(d, date):
            return _ym_key(d)
        return None

    for r in filtered:
        key = _key_for(r)
        if not key:
            continue
        g = groups.setdefault(
            key,
            {
                group_by: key,
                "plant": r.get("Plant"),
                "materialNumber": r.get("Material Number"),
                "delivered": 0.0,
                "requested": 0.0,
                "actualInput": 0.0,
            },
        )
        g["delivered"] += r.get("Purchase Order Quantity (Delivered)", 0.0)
        g["requested"] += r.get("Purchase Order Quantity (Requested)", 0.0)
        g["actualInput"] += r.get("Actual Input (lbs)", 0.0)

    # Optionally add empty months within window
    if group_by == "month" and include_empty_months and start and end:
        for ym in _iter_months(
            date(start.year, start.month, 1), date(end.year, end.month, 1)
        ):
            if ym not in groups:
                groups[ym] = {
                    "month": ym,
                    "delivered": 0.0,
                    "requested": 0.0,
                    "actualInput": 0.0,
                }

    out: List[Dict[str, Any]] = []
    for g in groups.values():
        denom = g["actualInput"] if metric == "yield" else g["requested"]
        if denom > 0:
            pct = _round2((g["delivered"] / denom) * 100.0)
            g["percent"] = pct
        out.append(g)

    # Sorting
    if any("percent" in r for r in out):
        out = sorted(out, key=lambda r: r.get("percent", -1), reverse=True)
    elif group_by == "month":
        out = sorted(out, key=lambda r: r.get("month"))
    else:
        out = sorted(out, key=lambda r: r.get(group_by))

    if isinstance(limit, int) and limit > 0:
        out = out[:limit]

    highest, lowest = (None, None)
    if any("percent" in r for r in out):
        highest, lowest = _highest_lowest(out, "percent")
    return {"rows": out, "highest": highest, "lowest": lowest}


@tool("po_date_range")
async def po_date_range(
    plant: Optional[str] = None,
    material_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the earliest and latest dates available, optionally filtered by plant/material.

    Returns: { minDate: YYYY-MM-DD | null, maxDate: YYYY-MM-DD | null, rowCount: int }
    """
    rows = _load_po_rows()
    dates: List[date] = []
    for r in rows:
        if plant and r.get("Plant") != plant:
            continue
        if material_number and r.get("Material Number") != material_number:
            continue
        d = r.get("_date")
        if isinstance(d, date):
            dates.append(d)
    if not dates:
        return {"minDate": None, "maxDate": None, "rowCount": 0}
    min_d = min(dates)
    max_d = max(dates)
    return {
        "minDate": min_d.isoformat(),
        "maxDate": max_d.isoformat(),
        "rowCount": len(dates),
    }


@tool("scrap_by_plant")
async def scrap_by_plant(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    last_n_days: Optional[int] = None,
    last_n_months: Optional[int] = None,
) -> Dict[str, Any]:
    """Aggregate scrapping quantities by Plant over an optional time window.

    Returns rows sorted by Quantity descending with highest/lowest entries.
    """
    start, end = _resolve_time_window(start_date, end_date, last_n_days, last_n_months)
    rows = _load_scrap_rows()

    groups: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        d = r.get("_date")
        if not _within_window(d, start, end):
            continue
        plant = r.get("Plant") or ""
        g = groups.setdefault(plant, {"plant": plant, "quantity": 0.0, "rows": 0})
        g["quantity"] += r.get("Quantity", 0.0)
        g["rows"] += 1

    out = sorted(groups.values(), key=lambda x: x.get("quantity", 0.0), reverse=True)
    highest = out[0] if out else None
    lowest = out[-1] if out else None
    return {"rows": out, "highest": highest, "lowest": lowest}


@tool("scrap_query")
async def scrap_query(
    plant: Optional[str] = None,
    material_number: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    last_n_days: Optional[int] = None,
    last_n_months: Optional[int] = None,
    group_by: Optional[str] = None,
    include_empty_months: Optional[bool] = False,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Generic scrapping data access with optional filters and aggregation.

    Filters:
    - plant, material_number, and time window

    Aggregation (optional):
    - group_by: one of "plant", "material", "month"
    - metric implicitly is total Quantity
    - include_empty_months: when group_by="month", fill missing months within window
    """
    start, end = _resolve_time_window(start_date, end_date, last_n_days, last_n_months)
    rows = _load_scrap_rows()

    def _passes_filters(r: Dict[str, Any]) -> bool:
        if plant and r.get("Plant") != plant:
            return False
        if material_number and r.get("Material Number") != material_number:
            return False
        if not _within_window(r.get("_date"), start, end):
            return False
        return True

    filtered = [r for r in rows if _passes_filters(r)]

    if not group_by:
        out_rows: List[Dict[str, Any]] = []
        for r in filtered:
            out_rows.append(
                {
                    "plant": r.get("Plant"),
                    "materialNumber": r.get("Material Number"),
                    "maktx": r.get("MAKTX"),
                    "meins": r.get("MEINS"),
                    "date": r.get("_date").isoformat() if r.get("_date") else None,
                    "quantity": r.get("Quantity"),
                }
            )
        if isinstance(limit, int) and limit > 0:
            out_rows = out_rows[:limit]
        return {"rows": out_rows}

    group_by = (group_by or "").lower()
    if group_by not in {"plant", "material", "month"}:
        return {"error": "group_by must be one of 'plant','material','month'"}

    groups: Dict[str, Dict[str, Any]] = {}

    def _key_for(r: Dict[str, Any]) -> Optional[str]:
        if group_by == "plant":
            return r.get("Plant")
        if group_by == "material":
            return r.get("Material Number")
        d = r.get("_date")
        if group_by == "month" and isinstance(d, date):
            return _ym_key(d)
        return None

    for r in filtered:
        key = _key_for(r)
        if not key:
            continue
        g = groups.setdefault(
            key,
            {
                group_by: key,
                "plant": r.get("Plant"),
                "materialNumber": r.get("Material Number"),
                "quantity": 0.0,
            },
        )
        g["quantity"] += r.get("Quantity", 0.0)

    if group_by == "month" and include_empty_months and start and end:
        for ym in _iter_months(
            date(start.year, start.month, 1), date(end.year, end.month, 1)
        ):
            if ym not in groups:
                groups[ym] = {"month": ym, "quantity": 0.0}

    out: List[Dict[str, Any]] = list(groups.values())
    out = sorted(out, key=lambda r: r.get("quantity", 0.0), reverse=True)
    if isinstance(limit, int) and limit > 0:
        out = out[:limit]
    highest = out[0] if out else None
    lowest = out[-1] if out else None
    return {"rows": out, "highest": highest, "lowest": lowest}


__all__ = [
    "calculator_tool",
    "po_yield_by_plant_for_material",
    "po_efficiency_by_plant_for_material",
    "po_yield_by_material_at_plant",
    "po_yield_by_month",
    "po_query",
    "po_date_range",
    "scrap_by_plant",
    "scrap_query",
]
