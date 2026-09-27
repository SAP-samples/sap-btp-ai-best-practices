"""Data loading — 100% S/4HANA OData. No CSV, no HANA, no fallback.

Source of truth is S/4HANA via standard OData v2 APIs:
  * API_MAINTENANCEORDER_SRV  → orders + operations  (load_orders)
  * API_WORK_CENTER_SRV       → work-center capacity  (load_capacity)

OData fields are mapped back to the legacy column names the scheduling engine
and the frontend already expect, so no downstream logic changes.
"""
from functools import lru_cache

import pandas as pd

from . import s4_client

DATE_COLS_ORDERS = [
    "BASIC_START_DATE", "BASIC_FINISH_DATE", "RELEASE_DATE",
    "EARLY_EXECTN_START_DATE", "LATEST_EXECTN_FINISH_DATE",
]

# ── S/4 code → legacy value maps ──────────────────────────────────────────────
# Maintenance plant code → the plant NAME the app/UI use everywhere.
PLANT_CODE_TO_NAME = {
    "3707": "Miami",
    "3720": "Sierrita",
}

# MaintPriority code → the priority TEXT the scheduler filters on (P3/P4 etc.).
PRIORITY_CODE_TO_TEXT = {
    "1": "Emergency",
    "2": "High",
    "3": "Medium",
    "4": "Low",
    "5": "Future Oppty",
}

# MaintOrdProcessSubPhaseCode → the ORDER_SUBPHASE text (opportunity filter keys
# off "Ready to Schedule"). E0009 is the standard RTS code for scope items 4HH/4HI.
SUBPHASE_CODE_TO_TEXT = {
    "E0009": "Ready to Schedule (Order)",
}


def _odata_ts(val):
    """Convert an OData `/Date(ms)/` value to a pandas Timestamp (or None)."""
    ms = s4_client.parse_odata_date(val)
    if ms is None:
        return None
    # S/4 PM dates are business dates (no meaningful time-of-day). Normalise to
    # midnight so week-window comparisons match the former CSV date-only semantics.
    if isinstance(ms, (int, float)):
        return pd.to_datetime(ms, unit="ms").normalize()
    return pd.to_datetime(ms, errors="coerce").normalize()


def _phase_from_status(system_status: str) -> str:
    """Derive the legacy WO_PHASE bucket from the S/4 SystemStatus string."""
    s = (system_status or "").upper()
    if "CLSD" in s or "TECO" in s:
        return "Closed"
    if "REL" in s:
        return "Released"
    return "Created"


@lru_cache(maxsize=1)
def load_orders() -> pd.DataFrame:
    """Fetch PM orders with operations from S/4 and flatten to one row per operation."""
    select = ",".join([
        "MaintenanceOrder", "MaintenanceOrderDesc", "OrderType", "MaintenancePlant",
        "MaintPriority", "MaintPlannerGroup", "MaintActivityType", "Equipment",
        "EquipmentName", "EquipmentCriticality", "FunctionalLocation",
        "MaintOrdBasicStartDate", "MaintOrdBasicEndDate", "LatestAcceptableCompletionDate",
        "MaintOrdProcessSubPhaseCode", "SystemStatus", "ReleaseDate",
    ])
    records = s4_client.odata_get(
        s4_client.SRV_MAINT_ORDER,
        "MaintenanceOrder",
        {"$expand": "to_MaintOrderOperation", "$select": select},
    )
    rows = [row for rec in records for row in _flatten_order(rec)]
    df = pd.DataFrame(rows)

    df["ORDER_NO"] = df["ORDER_NO"].astype(str)
    for col in DATE_COLS_ORDERS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    df["ACTIVITY_WORK_INVOLVE"] = pd.to_numeric(df["ACTIVITY_WORK_INVOLVE"], errors="coerce").fillna(0)
    df["ACTIVITY_NORMAL_DURATION"] = pd.to_numeric(df["ACTIVITY_NORMAL_DURATION"], errors="coerce").fillna(0)
    df["CAPACITY_COUNT"] = pd.to_numeric(df["CAPACITY_COUNT"], errors="coerce").fillna(1)
    df["ORDER_SUBPHASE"] = df["ORDER_SUBPHASE"].fillna("")
    return df


@lru_cache(maxsize=1)
def load_capacity() -> pd.DataFrame:
    """Fetch work-center capacity (one row per WC per day) from S/4."""
    select = ",".join([
        "WorkCenter", "WorkCenterPlant", "CapacityInternalID",
        "CapacityStartDate", "CapacityEndDate", "AvailableCapacity",
        "CapacityUtilizationRate", "NumberOfCapacities",
    ])
    records = s4_client.odata_get(
        s4_client.SRV_WORK_CENTER,
        "WorkCenterCapacity",
        {"$select": select},
    )
    rows = []
    for rec in records:
        plant_code = str(rec.get("WorkCenterPlant") or "")
        wc = rec.get("WorkCenter")
        rows.append({
            "PLANT_CODE": plant_code,
            "PLANT_NAME": PLANT_CODE_TO_NAME.get(plant_code, plant_code),
            "WORK_CENTER": wc,
            "OPER_WORK_CENTER": wc,
            "CAPACITY_ID": rec.get("CapacityInternalID"),
            "DATE": _odata_ts(rec.get("CapacityStartDate")),
            "SHIFTNAME": "",
            "AVAILABLETIME": rec.get("AvailableCapacity"),
            "MANPOWER": rec.get("NumberOfCapacities"),
            "CAPACITY": rec.get("AvailableCapacity"),
        })
    df = pd.DataFrame(rows)

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["CAPACITY"] = pd.to_numeric(df["CAPACITY"], errors="coerce")
    df["MANPOWER"] = pd.to_numeric(df["MANPOWER"], errors="coerce")
    return df


# ── S/4 order flattening ──────────────────────────────────────────────────────
def _flatten_order(rec: dict) -> list[dict]:
    """Expand one MaintenanceOrder (+ nested operations) into legacy order rows."""
    plant_code = str(rec.get("MaintenancePlant") or "")
    header = {
        "ORDER_NO": rec.get("MaintenanceOrder"),
        "PRODUCTION_ORDER_HDR_DESC": rec.get("MaintenanceOrderDesc"),
        "ORDER_TYPE_CODE": rec.get("OrderType"),
        "PLANT_CODE": plant_code,
        "PLANT_NAME": PLANT_CODE_TO_NAME.get(plant_code, plant_code),
        "PRIORITY": PRIORITY_CODE_TO_TEXT.get(str(rec.get("MaintPriority") or ""), rec.get("MaintPriority")),
        "PLANNER_GRP": rec.get("MaintPlannerGroup"),
        "MAINT_ACTIVITY_TYPE": rec.get("MaintActivityType"),
        "EQUIPMENT_NO": rec.get("Equipment"),
        "EQUIPMENT_DESC": rec.get("EquipmentName"),
        "EQUIPMENT_CRITICALITY": rec.get("EquipmentCriticality"),
        "FUNC_LOC_CODE": rec.get("FunctionalLocation"),
        "BASIC_START_DATE": _odata_ts(rec.get("MaintOrdBasicStartDate")),
        "BASIC_FINISH_DATE": _odata_ts(rec.get("MaintOrdBasicEndDate")),
        "LATEST_EXECTN_FINISH_DATE": _odata_ts(rec.get("LatestAcceptableCompletionDate")),
        "ORDER_SUBPHASE": SUBPHASE_CODE_TO_TEXT.get(
            str(rec.get("MaintOrdProcessSubPhaseCode") or ""), rec.get("MaintOrdProcessSubPhaseCode")
        ),
        "SYSTEM_STATUS": rec.get("SystemStatus"),
        "WO_PHASE": _phase_from_status(rec.get("SystemStatus")),
        "RELEASE_DATE": _odata_ts(rec.get("ReleaseDate")),
    }

    ops_container = rec.get("to_MaintOrderOperation") or {}
    ops = ops_container.get("results", []) if isinstance(ops_container, dict) else ops_container

    if not ops:
        return [dict(header, OPER_NO=None, OPER_SHORT_TEXT=None, OPER_WORK_CENTER=None,
                     WORK_CENTER=None, ACTIVITY_WORK_INVOLVE=0, WORK_UNIT=None,
                     CAPACITY_COUNT=1, ACTIVITY_NORMAL_DURATION=0, ACTIVITY_NORMAL_DURATION_UNIT=None)]

    rows = []
    for op in ops:
        wc = op.get("WorkCenter")
        rows.append(dict(
            header,
            OPER_NO=op.get("MaintenanceOrderOperation"),
            OPER_SHORT_TEXT=op.get("OperationDescription"),
            # OData exposes only the work-center code; use it on both orders and
            # capacity so the existing OPER_WORK_CENTER join stays consistent.
            OPER_WORK_CENTER=wc,
            WORK_CENTER=wc,
            ACTIVITY_WORK_INVOLVE=op.get("PlannedWork"),
            WORK_UNIT=op.get("WorkUnit"),
            CAPACITY_COUNT=op.get("CapacityCount"),
            ACTIVITY_NORMAL_DURATION=op.get("NormalDuration"),
            ACTIVITY_NORMAL_DURATION_UNIT=op.get("NormalDurationUnit"),
        ))
    return rows


def _safe(val):
    """Convert pandas/numpy scalar to a JSON-safe Python type."""
    if pd.isna(val):
        return None
    if hasattr(val, "isoformat"):
        return val.isoformat()
    if hasattr(val, "item"):          # numpy scalar
        return val.item()
    return val


def row_to_dict(row) -> dict:
    """Serialize a pandas Series or dict to a JSON-safe dict."""
    if hasattr(row, "to_dict"):
        d = row.to_dict()
    else:
        d = dict(row)
    return {k: _safe(v) for k, v in d.items() if not k.startswith("_")}
