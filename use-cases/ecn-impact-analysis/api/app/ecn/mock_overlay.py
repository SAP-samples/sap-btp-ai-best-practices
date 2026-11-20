"""Overlay mock prototype data onto snapshot dicts for demo ECNs.

Reads CSVs from prototype/mock_data/ and merges additional context, pricing,
work instruction, tooling, and phase schedule data into snapshots whose ECN
number matches. Data is parsed once and cached.

Remove this module once the underlying data lands in HANA.
"""
# MOCK_DATA_RETIREMENT_VALIDATION_DISABLED: SnapshotService no longer imports
# or invokes this temporary producer. It remains intact only for a reversible
# comparison during validation before the final deletion.
import csv
import functools
from pathlib import Path

_MOCK_DIR = Path(__file__).parents[3] / "prototype" / "mock_data"

_PHASE_LABEL = {"10": "Created", "20": "In Review", "30": "Eng. Approved",
                "40": "Phase-In", "50": "Active", "70": "Flagged Obsolete", "90": "Phase-Out"}


def _read_csv(filename):
    """Read a comment-stripped CSV file into a list of dicts."""
    path = _MOCK_DIR / filename
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        lines = [line for line in f if not line.startswith("#")]
    return list(csv.DictReader(lines))


@functools.lru_cache(maxsize=1)
def _load():
    """Parse all mock CSVs once and return keyed structures."""
    context_by_ecn = {r["ECN_NUMBER"]: r for r in _read_csv("mock_ecn_context.csv")}

    prices_by_ecn: dict[str, list] = {}
    for r in _read_csv("mock_purchase_prices.csv"):
        prices_by_ecn.setdefault(r["ECN_NUMBER"], []).append(r)

    wi_by_ecn: dict[str, list] = {}
    for r in _read_csv("mock_work_instructions.csv"):
        wi_by_ecn.setdefault(r["ECN_NUMBER"], []).append(r)

    tooling_by_plant: dict[str, list] = {}
    for r in _read_csv("mock_tooling_inventory.csv"):
        tooling_by_plant.setdefault(r["PLANT_CODE"], []).append(r)

    schedule_by_ecn: dict[str, list] = {}
    for r in _read_csv("mock_phase_schedule.csv"):
        schedule_by_ecn.setdefault(r["ECN_NUMBER"], []).append(r)

    recommendations_by_ecn: dict[str, list] = {}
    for r in _read_csv("mock_make_buy_recommendations.csv"):
        recommendations_by_ecn.setdefault(r["ECN_NUMBER"], []).append(r)

    return context_by_ecn, prices_by_ecn, wi_by_ecn, tooling_by_plant, schedule_by_ecn, recommendations_by_ecn


def merge_mock_data(snapshot: dict) -> dict:
    """Add mock overlay fields to a snapshot dict for demo ECNs.

    Mutates the dict in place (caller owns it) and returns it for chaining.
    Non-matching ECN numbers are left untouched.
    """
    ecn_number = snapshot.get("ecn_number")
    if not ecn_number:
        return snapshot

    context_map, prices_map, wi_map, tooling_map, schedule_map, recommendations_map = _load()

    # ── ECN context (owner / why / when / financial direction) ──────────────
    if ctx := context_map.get(ecn_number):
        snapshot["ecn_context"] = {
            **snapshot.get("ecn_context", {}),
            "owner_name": ctx.get("OWNER_NAME", ""),
            "category_label": ctx.get("CATEGORY_LABEL", ""),
            "business_driver": ctx.get("BUSINESS_DRIVER", ""),
            "cib_target_date": ctx.get("CIB_TARGET_DATE", ""),
            "financial_impact": ctx.get("FINANCIAL_IMPACT_ESTIMATE", "UNKNOWN"),
            "financial_notes": ctx.get("FINANCIAL_NOTES", ""),
            "risk_level": ctx.get("RISK_LEVEL", ""),
            "source": "DEMO",
        }

    # ── Purchase prices ──────────────────────────────────────────────────────
    if prices := prices_map.get(ecn_number):
        snapshot["purchase_prices"] = [
            {
                "material_number": r["MATERIAL_NUMBER"],
                "plant_code": r["PLANT_CODE"],
                "info_record": r.get("INFO_RECORD", ""),
                "supplier_name": r.get("SUPPLIER_NAME", ""),
                "net_price": r.get("NET_PRICE", ""),
                "currency": r.get("CURRENCY", ""),
                "price_per": r.get("PRICE_PER", ""),
                "unit_of_measure": r.get("UNIT_OF_MEASURE", ""),
                "valid_from": r.get("VALID_FROM", ""),
                "valid_to": r.get("VALID_TO", ""),
                "status": r.get("STATUS", ""),
                "notes": r.get("NOTES", ""),
                "source": "CURATED_DEMONSTRATION",
            }
            for r in prices
        ]
        missing = [r for r in prices if r.get("STATUS") == "MISSING"]
        existing_codes = {q.get("code") for q in snapshot.get("questions", [])}
        if missing and "MISSING_INFO_RECORD" not in existing_codes:
            snapshot.setdefault("questions", []).insert(0, {
                "code": "MISSING_INFO_RECORD",
                "question": "New parts have no purchase info record",
                "why": (
                    "These parts are new to this ECN but have no supplier or price information "
                    "in SAP. A purchase info record must be created before they can be ordered."
                ),
                "resolution": (
                    "Create an info record in SAP (ME21N) for each missing part before "
                    "the ECN cutover date."
                ),
                "affected_count": len(missing),
                "subjects": [
                    {
                        "number": r["MATERIAL_NUMBER"],
                        "details": [
                            f"Plant: {r['PLANT_CODE']}",
                            r.get("NOTES", "")[:120],
                        ],
                    }
                    for r in missing
                ],
            })

    # ── Work instructions ────────────────────────────────────────────────────
    if wi := wi_map.get(ecn_number):
        snapshot["work_instructions"] = [
            {
                "material_number": r["MATERIAL_NUMBER"],
                "plant_code": r["PLANT_CODE"],
                "instruction_id": r.get("INSTRUCTION_ID", ""),
                "instruction_title": r.get("INSTRUCTION_TITLE", ""),
                "status": r.get("STATUS", ""),
                "tooling_class": r.get("TOOLING_CLASS", ""),
                "est_time_min": r.get("EST_TIME_MIN", ""),
                "notes": r.get("NOTES", ""),
            }
            for r in wi
        ]
        missing_wi = [r for r in wi if r.get("STATUS") == "MISSING"]
        existing_codes = {q.get("code") for q in snapshot.get("questions", [])}
        if missing_wi and "MISSING_WORK_INSTRUCTION" not in existing_codes:
            snapshot.setdefault("questions", []).insert(0, {
                "code": "MISSING_WORK_INSTRUCTION",
                "question": "New parts have no work instruction",
                "why": (
                    "These parts require a work instruction document before production staff "
                    "can assemble them. The absence blocks line readiness at PL10."
                ),
                "resolution": (
                    "Create work instructions in the document management system and validate "
                    "with line supervisors before the production start date."
                ),
                "affected_count": len(missing_wi),
                "subjects": [
                    {
                        "number": r["MATERIAL_NUMBER"],
                        "details": [
                            f"Plant: {r['PLANT_CODE']}",
                            r.get("NOTES", "")[:120],
                        ],
                    }
                    for r in missing_wi
                ],
            })

    # ── Tooling inventory (collect plants referenced by WI + schedule) ───────
    plant_codes = {
        r.get("PLANT_CODE", "")
        for r in (wi_map.get(ecn_number) or []) + (schedule_map.get(ecn_number) or [])
        if r.get("PLANT_CODE")
    }
    if plant_codes:
        snapshot["tooling_inventory"] = [
            {
                "plant_code": t["PLANT_CODE"],
                "machine_id": t.get("MACHINE_ID", ""),
                "tooling_class": t.get("TOOLING_CLASS", ""),
                "machine_type": t.get("MACHINE_TYPE", ""),
                "current_util_pct": t.get("CURRENT_UTIL_PCT", ""),
                "next_available_date": t.get("NEXT_AVAILABLE_DATE", ""),
                "changeover_time_min": t.get("CHANGEOVER_TIME_MIN", ""),
                "notes": t.get("NOTES", ""),
            }
            for plant in plant_codes
            for t in tooling_map.get(plant, [])
        ]

    # ── Phase schedule ───────────────────────────────────────────────────────
    if schedule := schedule_map.get(ecn_number):
        snapshot["phase_schedule"] = [
            {
                "material_number": r["MATERIAL_NUMBER"],
                "plant_code": r["PLANT_CODE"],
                "description": r.get("DESCRIPTION", ""),
                "from_status": r.get("FROM_STATUS", ""),
                "from_label": _PHASE_LABEL.get(r.get("FROM_STATUS", ""), r.get("FROM_STATUS", "")),
                "to_status": r.get("TO_STATUS", ""),
                "to_label": _PHASE_LABEL.get(r.get("TO_STATUS", ""), r.get("TO_STATUS", "")),
                "effective_date": r.get("EFFECTIVE_DATE", ""),
                "planner_confirmed": r.get("PLANNER_CONFIRMED", "FALSE").upper() == "TRUE",
                "cutover_note": r.get("CUTOVER_NOTE", ""),
                "source": "CURATED_DEMONSTRATION",
            }
            for r in schedule
        ]

    # ── Make-or-buy recommendation ───────────────────────────────────────────
    if recommendations := recommendations_map.get(ecn_number):
        snapshot["make_buy_recommendations"] = [
            {
                "material_number": r["MATERIAL_NUMBER"],
                "material_description": r.get("MATERIAL_DESCRIPTION", ""),
                "plant_code": r["PLANT_CODE"],
                "recommendation": r.get("RECOMMENDATION", ""),
                "recommendation_summary": r.get("RECOMMENDATION_SUMMARY", ""),
                "owner": r.get("OWNER", ""),
                "make_cost": r.get("MAKE_COST", ""),
                "currency": r.get("CURRENCY", ""),
                "price_per": r.get("PRICE_PER", ""),
                "unit_of_measure": r.get("UNIT_OF_MEASURE", ""),
                "comparable_history": r.get("COMPARABLE_HISTORY", ""),
                "tooling": r.get("TOOLING", ""),
                "capacity": r.get("CAPACITY", ""),
                "raw_materials": r.get("RAW_MATERIALS", ""),
                "next_action": r.get("NEXT_ACTION", ""),
                "source": "CURATED_DEMONSTRATION",
            }
            for r in recommendations
        ]

    if any((prices_map.get(ecn_number), wi_map.get(ecn_number), schedule_map.get(ecn_number), recommendations_map.get(ecn_number))):
        snapshot["demo_assumptions"] = [
            "Cost, sourcing, and manufacturing timing values are illustrative demonstration data.",
        ]

    return snapshot
