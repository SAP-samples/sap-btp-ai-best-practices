#!/usr/bin/env python3
"""Deterministic PS4 measurement calculations.

The commands in this module evaluate source-backed input values against PS4 guideline
values. The sole geometry derivation is the approved standard bender-schedule mapping,
which has explicit row-pattern validation and provenance in its output.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Optional


MATERIAL_DESIGNATIONS = frozenset(
    {
        "Ø08TH1",
        "Ø10TH1",
        "Ø10TH1,5",
        "Ø12TH1",
        "Ø12TH1,5",
        "Ø15TH1,5",
        "Ø15TH2",
        "Ø16TH2",
        "Ø18TH1,5",
        "Ø18TH2",
        "Ø20TH2,5",
        "Ø22TH1,5",
        "Ø22TH2",
    }
)

# Values transcribed from Pipes Guideline PS4, "Mini straight lengths – Auto Crimping".
AUTO_CRIMP_TABLE: dict[str, dict[str, Decimal]] = {
    "8": {
        "bendRadiusMm": Decimal("15.5"),
        "l2MinimumMm": Decimal("12"),
        "l1BaseMinimumMm": Decimal("29.4"),
        "l4BaseMinimumMm": Decimal("30.4"),
        "l3MaximumTheta40MinimumMm": Decimal("46.9"),
        "l3MaximumTheta85MinimumMm": Decimal("62.4"),
        "l3MaximumTheta180MinimumMm": Decimal("77.9"),
    },
    "10": {
        "bendRadiusMm": Decimal("20.5"),
        "l2MinimumMm": Decimal("15"),
        "l1BaseMinimumMm": Decimal("32.9"),
        "l4BaseMinimumMm": Decimal("33.9"),
        "l3MaximumTheta40MinimumMm": Decimal("55.4"),
        "l3MaximumTheta85MinimumMm": Decimal("75.9"),
        "l3MaximumTheta180MinimumMm": Decimal("96.4"),
    },
    "12": {
        "bendRadiusMm": Decimal("25.8"),
        "l2MinimumMm": Decimal("18"),
        "l1BaseMinimumMm": Decimal("36.0"),
        "l4BaseMinimumMm": Decimal("37.0"),
        "l3MaximumTheta40MinimumMm": Decimal("63.8"),
        "l3MaximumTheta85MinimumMm": Decimal("89.6"),
        "l3MaximumTheta180MinimumMm": Decimal("115.4"),
    },
    "15": {
        "bendRadiusMm": Decimal("30.8"),
        "l2MinimumMm": Decimal("24"),
        "l1BaseMinimumMm": Decimal("42.6"),
        "l4BaseMinimumMm": Decimal("39.6"),
        "l3MaximumTheta40MinimumMm": Decimal("71.4"),
        "l3MaximumTheta85MinimumMm": Decimal("102.2"),
        "l3MaximumTheta180MinimumMm": Decimal("133.0"),
    },
    "16": {
        "bendRadiusMm": Decimal("33"),
        "l2MinimumMm": Decimal("24"),
        "l1BaseMinimumMm": Decimal("45.7"),
        "l4BaseMinimumMm": Decimal("47.7"),
        "l3MaximumTheta40MinimumMm": Decimal("81.7"),
        "l3MaximumTheta85MinimumMm": Decimal("114.7"),
        "l3MaximumTheta180MinimumMm": Decimal("147.7"),
    },
    "18": {
        "bendRadiusMm": Decimal("36"),
        "l2MinimumMm": Decimal("27"),
        "l1BaseMinimumMm": Decimal("46.2"),
        "l4BaseMinimumMm": Decimal("45.2"),
        "l3MaximumTheta40MinimumMm": Decimal("82.2"),
        "l3MaximumTheta85MinimumMm": Decimal("118.2"),
        "l3MaximumTheta180MinimumMm": Decimal("154.2"),
    },
    "20": {
        "bendRadiusMm": Decimal("41"),
        "l2MinimumMm": Decimal("30"),
        "l1BaseMinimumMm": Decimal("55.0"),
        "l4BaseMinimumMm": Decimal("56.0"),
        "l3MaximumTheta40MinimumMm": Decimal("98.0"),
        "l3MaximumTheta85MinimumMm": Decimal("139.0"),
        "l3MaximumTheta180MinimumMm": Decimal("180.0"),
    },
    "22": {
        "bendRadiusMm": Decimal("49.2"),
        "l2MinimumMm": Decimal("33"),
        "l1BaseMinimumMm": Decimal("52.6"),
        "l4BaseMinimumMm": Decimal("50.6"),
        "l3MaximumTheta40MinimumMm": Decimal("100.8"),
        "l3MaximumTheta85MinimumMm": Decimal("150.0"),
        "l3MaximumTheta180MinimumMm": Decimal("199.2"),
    },
}

ROBOT_PICKUP_L2_MINIMUM_MM = Decimal("34")
ROBOT_PICKUP_L1_ADDITION_MM = Decimal("34")


class CalculatorInputError(ValueError):
    """An input cannot be evaluated by the calculator."""


def parse_decimal(value: str, argument_name: str) -> Decimal:
    # Accept the numeric forms visibly used in PS4 drawings and tables so an
    # agent never has to manually convert a decimal comma or diameter symbol.
    source_text = value.strip().replace("Ø", "").replace("ø", "")
    normalized_text = source_text.replace(",", ".")
    try:
        parsed = Decimal(normalized_text)
    except (InvalidOperation, ValueError) as error:
        raise CalculatorInputError(
            f"{argument_name} must be a decimal number, received {value!r}."
        ) from error
    if not parsed.is_finite():
        raise CalculatorInputError(f"{argument_name} must be finite.")
    return parsed


def optional_decimal(value: Optional[str], argument_name: str) -> Optional[Decimal]:
    return None if value is None else parse_decimal(value, argument_name)


def decimal_key(value: Decimal) -> str:
    if value != value.to_integral_value():
        raise CalculatorInputError(
            "diameter-mm must match a whole-number PS4 table diameter."
        )
    return str(int(value))


def decimal_text(value: Decimal) -> str:
    normalized = value.normalize()
    text = format(normalized, "f")
    return "0" if text in {"-0", ""} else text


def table_row(diameter_mm: Decimal) -> dict[str, Decimal]:
    key = decimal_key(diameter_mm)
    try:
        return AUTO_CRIMP_TABLE[key]
    except KeyError as error:
        raise CalculatorInputError(
            f"Diameter {decimal_text(diameter_mm)} mm is not covered by the PS4 auto-crimp table."
        ) from error


def assessed_result(
    design_value: Optional[Decimal],
    required_value: Optional[Decimal],
    cannot_assess_reason: Optional[str] = None,
) -> dict[str, str]:
    if required_value is None:
        return {
            "status": "Cannot assess",
            "reason": cannot_assess_reason or "The applicable PS4 requirement is unavailable.",
        }
    if design_value is None:
        return {
            "status": "Cannot assess",
            "reason": "The source-backed design measurement is unavailable.",
        }
    return {
        "status": "Pass" if design_value >= required_value else "Fail",
        "reason": "The source-backed design measurement was compared with the calculated requirement.",
    }


def material_designation(args: argparse.Namespace) -> dict[str, Any]:
    diameter = parse_decimal(args.diameter_mm, "diameter-mm")
    thickness = parse_decimal(args.wall_thickness_mm, "wall-thickness-mm")
    diameter_part = "Ø08" if diameter == Decimal("8") else f"Ø{decimal_text(diameter)}"
    thickness_part = decimal_text(thickness).replace(".", ",")
    return {
        "operation": "material-designation",
        "diameterMm": decimal_text(diameter),
        "wallThicknessMm": decimal_text(thickness),
        "candidate": f"{diameter_part}TH{thickness_part}",
    }


def raw_material(args: argparse.Namespace) -> dict[str, Any]:
    designation = args.designation.strip()
    return {
        "operation": "raw-material",
        "designation": designation,
        "guidelineCoverage": "listed" if designation in MATERIAL_DESIGNATIONS else "not_listed",
    }


def tube_length(args: argparse.Namespace) -> dict[str, Any]:
    diameter = parse_decimal(args.diameter_mm, "diameter-mm")
    wall_thickness = parse_decimal(args.wall_thickness_mm, "wall-thickness-mm")
    total_length = parse_decimal(args.total_length_mm, "total-length-mm")
    key = decimal_key(diameter)
    if key not in AUTO_CRIMP_TABLE:
        return {
            "operation": "tube-length",
            "diameterMm": decimal_text(diameter),
            "wallThicknessMm": decimal_text(wall_thickness),
            "totalLengthMm": decimal_text(total_length),
            "status": "Cannot assess",
            "reason": "The diameter is not listed in the PS4 guideline table.",
        }
    maximum = (
        Decimal("1500")
        if key == "8" or (key == "10" and wall_thickness == Decimal("1"))
        else Decimal("3000")
    )
    minimum = Decimal("200")
    return {
        "operation": "tube-length",
        "diameterMm": decimal_text(diameter),
        "wallThicknessMm": decimal_text(wall_thickness),
        "totalLengthMm": decimal_text(total_length),
        "minimumMm": decimal_text(minimum),
        "maximumMm": decimal_text(maximum),
        "status": "Pass" if minimum <= total_length <= maximum else "Fail",
    }


def bend_angles(args: argparse.Namespace) -> dict[str, Any]:
    angles = [parse_decimal(value, "angle-deg") for value in args.angle_deg]
    results = [
        {
            "angleDeg": decimal_text(angle),
            "status": "Pass" if Decimal("0") < angle < Decimal("180") else "Fail",
        }
        for angle in angles
    ]
    return {
        "operation": "bend-angles",
        "results": results,
        "status": "Pass" if all(result["status"] == "Pass" for result in results) else "Fail",
    }


def l1_requirement(row: dict[str, Decimal], theta1: Optional[Decimal]) -> tuple[Optional[Decimal], str, Optional[str]]:
    if theta1 is None:
        return None, "", "The source-backed θ1 measurement is unavailable."
    if not Decimal("0") < theta1 < Decimal("180"):
        return None, "", "θ1 is outside the PS4 formula/table coverage."
    base = row["l1BaseMinimumMm"]
    if theta1 <= Decimal("90"):
        return base, "table L1 minimum for θ1 ≤ 90", None
    calculated = float(base) + float(row["bendRadiusMm"]) * math.tan(
        math.radians(float(theta1 - Decimal("90")))
    )
    # The guideline does not define a rounding rule. Preserve Python's shortest
    # decimal representation of the trigonometric result and apply no tolerance.
    return Decimal(str(calculated)), "table L1 minimum + R cintrage × tan(θ1 − 90)", None


def l4_requirement(row: dict[str, Decimal], theta4: Optional[Decimal]) -> tuple[Optional[Decimal], str, Optional[str]]:
    if theta4 is None:
        return None, "", "The source-backed θ4 measurement is unavailable."
    if not Decimal("0") < theta4 < Decimal("180"):
        return None, "", "θ4 is outside the PS4 formula/table coverage."
    base = row["l4BaseMinimumMm"]
    if theta4 <= Decimal("120"):
        return base, "table L4 minimum for θ4 ≤ 120", None
    calculated = float(base) + float(row["bendRadiusMm"]) * math.tan(
        math.radians(float(theta4 - Decimal("120")))
    )
    # The guideline does not define a rounding rule. Preserve Python's shortest
    # decimal representation of the trigonometric result and apply no tolerance.
    return Decimal(str(calculated)), "table L4 minimum + R cintrage × tan(θ4 − 120)", None


def l3_requirement(row: dict[str, Decimal], theta4: Optional[Decimal]) -> tuple[Optional[Decimal], str, Optional[str]]:
    if theta4 is None:
        return None, "", "The source-backed θ4 measurement is unavailable."
    if not Decimal("0") < theta4 < Decimal("180"):
        return None, "", "θ4 is outside the PS4 table coverage."
    if theta4 <= Decimal("40"):
        return row["l3MaximumTheta40MinimumMm"], "table L3 minimum for θ4 ≤ 40", None
    if theta4 <= Decimal("85"):
        return row["l3MaximumTheta85MinimumMm"], "table L3 minimum for 40 < θ4 ≤ 85", None
    return row["l3MaximumTheta180MinimumMm"], "table L3 minimum for 85 < θ4 < 180", None


def auto_crimp(args: argparse.Namespace) -> dict[str, Any]:
    diameter = parse_decimal(args.diameter_mm, "diameter-mm")
    row = table_row(diameter)
    theta1 = optional_decimal(args.theta1_deg, "theta1-deg")
    theta4 = optional_decimal(args.theta4_deg, "theta4-deg")
    l2_values = [parse_decimal(value, "l2-mm") for value in (args.l2_mm or [])]
    dimensions = {
        "l1": optional_decimal(args.l1_mm, "l1-mm"),
        "l3": optional_decimal(args.l3_mm, "l3-mm"),
        "l4": optional_decimal(args.l4_mm, "l4-mm"),
    }
    l1_minimum, l1_formula, l1_error = l1_requirement(row, theta1)
    l4_minimum, l4_formula, l4_error = l4_requirement(row, theta4)
    l3_minimum, l3_formula, l3_error = l3_requirement(row, theta4)
    requirement_data = {
        "l1": (l1_minimum, l1_formula, l1_error, row["l1BaseMinimumMm"]),
        "l3": (l3_minimum, l3_formula, l3_error, None),
        "l4": (l4_minimum, l4_formula, l4_error, row["l4BaseMinimumMm"]),
    }
    results: dict[str, Any] = {}
    for name, design_value in dimensions.items():
        requirement, formula, error, base = requirement_data[name]
        result = assessed_result(design_value, requirement, error)
        result["designValueMm"] = decimal_text(design_value) if design_value is not None else ""
        result["requiredMinimumMm"] = decimal_text(requirement) if requirement is not None else ""
        result["formulaOrLookup"] = formula
        if base is not None:
            result["baseMinimumMm"] = decimal_text(base)
        results[name] = result

    l2_segments: list[dict[str, str]] = []
    for index, l2_value in enumerate(l2_values, 1):
        l2_result = assessed_result(l2_value, row["l2MinimumMm"])
        l2_result["segmentIndex"] = str(index)
        l2_result["designValueMm"] = decimal_text(l2_value)
        l2_result["requiredMinimumMm"] = decimal_text(row["l2MinimumMm"])
        l2_result["formulaOrLookup"] = "table L2 minimum"
        l2_segments.append(l2_result)
    if not l2_segments:
        l2_status = "Cannot assess"
        l2_reason = "No source-backed L2 segment is available."
    elif any(segment["status"] == "Fail" for segment in l2_segments):
        l2_status = "Fail"
        l2_reason = "At least one source-backed L2 segment is below the table minimum."
    else:
        l2_status = "Pass"
        l2_reason = "Every supplied source-backed L2 segment meets the table minimum."
    results["l2"] = {
        "status": l2_status,
        "reason": l2_reason,
        "requiredMinimumMm": decimal_text(row["l2MinimumMm"]),
        "formulaOrLookup": "table L2 minimum",
        "segments": l2_segments,
    }

    result_statuses = [result["status"] for result in results.values()]
    if "Fail" in result_statuses:
        overall_status = "Fail"
    elif "Cannot assess" in result_statuses:
        overall_status = "Cannot assess"
    else:
        overall_status = "Pass"
    return {
        "operation": "auto-crimp",
        "diameterMm": decimal_text(diameter),
        "bendRadiusMm": decimal_text(row["bendRadiusMm"]),
        "theta1Deg": decimal_text(theta1) if theta1 is not None else "",
        "theta4Deg": decimal_text(theta4) if theta4 is not None else "",
        "overallStatus": overall_status,
        "results": results,
    }


def robot_pickup(args: argparse.Namespace) -> dict[str, Any]:
    l1 = optional_decimal(args.l1_mm, "l1-mm")
    l2 = optional_decimal(args.l2_mm, "l2-mm")
    diameter = optional_decimal(args.diameter_mm, "diameter-mm")

    l2_result = assessed_result(l2, ROBOT_PICKUP_L2_MINIMUM_MM)
    l2_result["designValueMm"] = decimal_text(l2) if l2 is not None else ""
    l2_result["requiredMinimumMm"] = decimal_text(ROBOT_PICKUP_L2_MINIMUM_MM)

    if l1 is not None and diameter is not None:
        row = table_row(diameter)
        l1_minimum = row["l1BaseMinimumMm"] + ROBOT_PICKUP_L1_ADDITION_MM
        l1_result = assessed_result(l1, l1_minimum)
        l1_result["designValueMm"] = decimal_text(l1)
        l1_result["requiredMinimumMm"] = decimal_text(l1_minimum)
        l1_result["baseL1MinimumMm"] = decimal_text(row["l1BaseMinimumMm"])
    elif l1 is None:
        l1_result = {
            "status": "Cannot assess",
            "reason": "The source-backed L1 measurement is unavailable.",
            "designValueMm": "",
            "requiredMinimumMm": "",
        }
    else:
        l1_result = {
            "status": "Cannot assess",
            "reason": "The source-backed diameter is unavailable for the L1 threshold lookup.",
            "designValueMm": decimal_text(l1),
            "requiredMinimumMm": "",
        }

    branch_statuses = {l1_result["status"], l2_result["status"]}
    if "Pass" in branch_statuses:
        condition_status = "Met"
    elif branch_statuses == {"Fail"}:
        condition_status = "Not met"
    else:
        condition_status = "Cannot assess"

    return {
        "operation": "robot-pickup",
        "diameterMm": decimal_text(diameter) if diameter is not None else "",
        "status": condition_status,
        "l2Branch": l2_result,
        "l1Branch": l1_result,
    }


def calculate_l3_from_terminal_values(
    terminal_straight_part: Decimal,
    terminal_bend_radius: Decimal,
    terminal_bend_angle: Decimal,
) -> dict[str, str]:
    if not Decimal("0") < terminal_bend_angle < Decimal("180"):
        raise CalculatorInputError(
            "terminal-bend-angle-deg must be strictly greater than 0 and strictly less than 180."
        )
    terminal_bend_arc = Decimal(
        str(float(terminal_bend_radius) * math.radians(float(terminal_bend_angle)))
    )
    l3 = terminal_straight_part + terminal_bend_arc
    return {
        "mappingRule": "terminal_straight_part_plus_final_bend_arc",
        "terminalStraightPartMm": decimal_text(terminal_straight_part),
        "terminalBendRadiusMm": decimal_text(terminal_bend_radius),
        "terminalBendAngleDeg": decimal_text(terminal_bend_angle),
        "terminalBendArcMm": decimal_text(terminal_bend_arc),
        "l3Mm": decimal_text(l3),
        "formula": "L3 = terminal straight part + terminal bend radius × θ4(rad)",
    }


def optional_schedule_decimal(value: Any, field_name: str) -> Optional[Decimal]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return parse_decimal(str(value), field_name)


def load_schedule_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    try:
        if args.schedule_file:
            with open(args.schedule_file, encoding="utf-8") as source:
                payload = json.load(source)
        else:
            payload = json.loads(args.schedule_json)
    except (OSError, json.JSONDecodeError) as error:
        raise CalculatorInputError(f"Cannot read schedule JSON: {error}") from error

    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise CalculatorInputError("Schedule JSON must be an array or an object with a rows array.")

    normalized_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise CalculatorInputError(f"Schedule row {index} must be an object.")
        point = str(row.get("point", "")).strip()
        if not point:
            raise CalculatorInputError(f"Schedule row {index} is missing point.")
        normalized_rows.append(
            {
                "point": point,
                "bendAngleDeg": optional_schedule_decimal(
                    row.get("bendAngleDeg"), f"schedule row {point} bendAngleDeg"
                ),
                "bendRadiusMm": optional_schedule_decimal(
                    row.get("bendRadiusMm"), f"schedule row {point} bendRadiusMm"
                ),
                "straightPartMm": optional_schedule_decimal(
                    row.get("straightPartMm"), f"schedule row {point} straightPartMm"
                ),
            }
        )
    return normalized_rows


def mapped_measurement(
    value: Decimal,
    source_point: str,
    end_point: str,
    source_field: str,
) -> dict[str, str]:
    return {
        "valueMm": decimal_text(value),
        "sourcePoint": source_point,
        "endPoint": end_point,
        "sourceField": source_field,
        "state": "mapped",
    }


def mapping_cannot_assess(reasons: list[str]) -> dict[str, Any]:
    return {
        "operation": "map-standard-bender-schedule",
        "mappingStatus": "Cannot assess",
        "mappingRule": "standard_ordered_bender_schedule/v1",
        "reasons": reasons,
    }


def map_standard_bender_schedule(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_schedule_rows(args)
    if len(rows) < 3:
        return mapping_cannot_assess(
            ["The standard mapping requires a start endpoint, at least one bend, and a terminal endpoint."]
        )

    first_row = rows[0]
    terminal_row = rows[-1]
    bend_rows = rows[1:-1]
    reasons: list[str] = []
    if first_row["bendAngleDeg"] is not None or first_row["bendRadiusMm"] is not None:
        reasons.append("The first schedule row is not an endpoint without bend angle and bend radius.")
    if terminal_row["bendAngleDeg"] is not None or terminal_row["bendRadiusMm"] is not None:
        reasons.append("The final schedule row is not an endpoint without bend angle and bend radius.")
    for row in bend_rows:
        if row["bendAngleDeg"] is None or row["bendRadiusMm"] is None:
            reasons.append(
                f"Schedule point {row['point']} is an intermediate row without both bend angle and bend radius."
            )
    for row, next_row in zip(rows[:-1], rows[1:]):
        if row["straightPartMm"] is None:
            reasons.append(
                f"Schedule point {row['point']} has no STRAIGHT PART for the run to point {next_row['point']}."
            )
    if reasons:
        return mapping_cannot_assess(reasons)

    first_bend = rows[1]
    terminal_bend = rows[-2]
    l2_segments = [
        {
            **mapped_measurement(
                row["straightPartMm"], row["point"], next_row["point"], "STRAIGHT PART"
            ),
            "segmentId": f"point-{row['point']}-to-point-{next_row['point']}",
        }
        for row, next_row in zip(rows[1:-2], rows[2:-1])
    ]
    l3 = calculate_l3_from_terminal_values(
        terminal_bend["straightPartMm"],
        terminal_bend["bendRadiusMm"],
        terminal_bend["bendAngleDeg"],
    )
    l3.update(
        {
            "state": "derived",
            "terminalBendPoint": terminal_bend["point"],
            "terminalEndpointPoint": terminal_row["point"],
        }
    )
    return {
        "operation": "map-standard-bender-schedule",
        "mappingStatus": "Mapped",
        "mappingRule": "standard_ordered_bender_schedule/v1",
        "sourcePointCount": len(rows),
        "l1": mapped_measurement(
            first_row["straightPartMm"], first_row["point"], first_bend["point"], "STRAIGHT PART"
        ),
        "theta1": {
            "valueDeg": decimal_text(first_bend["bendAngleDeg"]),
            "sourcePoint": first_bend["point"],
            "sourceField": "ANGLE",
            "state": "mapped",
        },
        "l2Segments": l2_segments,
        "robotPickupL2": l2_segments[0] if l2_segments else None,
        "l4": mapped_measurement(
            terminal_bend["straightPartMm"],
            terminal_bend["point"],
            terminal_row["point"],
            "STRAIGHT PART",
        ),
        "theta4": {
            "valueDeg": decimal_text(terminal_bend["bendAngleDeg"]),
            "sourcePoint": terminal_bend["point"],
            "sourceField": "ANGLE",
            "state": "mapped",
        },
        "l3": l3,
    }


def l3_from_terminal_schedule(args: argparse.Namespace) -> dict[str, Any]:
    terminal_straight_part = parse_decimal(
        args.terminal_straight_part_mm, "terminal-straight-part-mm"
    )
    terminal_bend_radius = parse_decimal(
        args.terminal_bend_radius_mm, "terminal-bend-radius-mm"
    )
    terminal_bend_angle = parse_decimal(
        args.terminal_bend_angle_deg, "terminal-bend-angle-deg"
    )
    result = calculate_l3_from_terminal_values(
        terminal_straight_part, terminal_bend_radius, terminal_bend_angle
    )
    return {"operation": "l3-from-terminal-schedule", **result}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministic PS4 guideline calculations. Output is JSON."
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)

    material = subparsers.add_parser("material-designation")
    material.add_argument("--diameter-mm", required=True)
    material.add_argument("--wall-thickness-mm", required=True)
    material.set_defaults(handler=material_designation)

    raw = subparsers.add_parser("raw-material")
    raw.add_argument("--designation", required=True)
    raw.set_defaults(handler=raw_material)

    length = subparsers.add_parser("tube-length")
    length.add_argument("--diameter-mm", required=True)
    length.add_argument("--wall-thickness-mm", required=True)
    length.add_argument("--total-length-mm", required=True)
    length.set_defaults(handler=tube_length)

    angles = subparsers.add_parser("bend-angles")
    angles.add_argument("--angle-deg", required=True, action="append")
    angles.set_defaults(handler=bend_angles)

    schedule_mapping = subparsers.add_parser("map-standard-bender-schedule")
    schedule_mapping_input = schedule_mapping.add_mutually_exclusive_group(required=True)
    schedule_mapping_input.add_argument("--schedule-json")
    schedule_mapping_input.add_argument("--schedule-file")
    schedule_mapping.set_defaults(handler=map_standard_bender_schedule)

    l3_mapping = subparsers.add_parser("l3-from-terminal-schedule")
    l3_mapping.add_argument("--terminal-straight-part-mm", required=True)
    l3_mapping.add_argument("--terminal-bend-radius-mm", required=True)
    l3_mapping.add_argument("--terminal-bend-angle-deg", required=True)
    l3_mapping.set_defaults(handler=l3_from_terminal_schedule)

    crimp = subparsers.add_parser("auto-crimp")
    crimp.add_argument("--diameter-mm", required=True)
    crimp.add_argument("--l1-mm")
    crimp.add_argument("--l2-mm", action="append")
    crimp.add_argument("--l3-mm")
    crimp.add_argument("--l4-mm")
    crimp.add_argument("--theta1-deg")
    crimp.add_argument("--theta4-deg")
    crimp.set_defaults(handler=auto_crimp)

    pickup = subparsers.add_parser("robot-pickup")
    pickup.add_argument("--diameter-mm")
    pickup.add_argument("--l1-mm")
    pickup.add_argument("--l2-mm")
    pickup.set_defaults(handler=robot_pickup)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
    except CalculatorInputError as error:
        print(json.dumps({"operation": args.operation, "error": str(error)}))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
