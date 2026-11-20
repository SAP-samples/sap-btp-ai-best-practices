"""Safe adapter for the packaged, policy-approved PS4 calculator CLI."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any


_CALCULATOR = Path(__file__).with_name("ps4_measurement_calculator.py")
_CALCULATOR_TIMEOUT_SECONDS = 15


class CalculatorExecutionError(RuntimeError):
    """Represent an unavailable or invalid deterministic calculator operation."""


def _argument(value: object) -> str:
    """Convert one controlled model field to a subprocess argument string."""
    if value is None:
        raise CalculatorExecutionError("calculator input was not available")
    return str(value)


async def run_calculator(operation: str, *arguments: str) -> dict[str, Any]:
    """Run one calculator command without a shell and parse its one JSON result.

    Args:
        operation: One documented calculator operation.
        arguments: Already-selected controlled option/value argument pairs.

    Returns:
        Parsed calculator JSON, preserving its values and provenance.

    Raises:
        CalculatorExecutionError: When the packaged deterministic operation
            cannot be evaluated. No model fallback is permitted.
    """
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(_CALCULATOR),
        operation,
        *arguments,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        stdout, _ = await asyncio.wait_for(
            process.communicate(), timeout=_CALCULATOR_TIMEOUT_SECONDS
        )
    except TimeoutError as exc:
        process.kill()
        await process.wait()
        raise CalculatorExecutionError("calculator operation timed out") from exc
    if process.returncode != 0:
        raise CalculatorExecutionError("calculator operation could not be evaluated")
    try:
        result = json.loads(stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalculatorExecutionError("calculator returned an invalid result") from exc
    if not isinstance(result, dict):
        raise CalculatorExecutionError("calculator returned an invalid result")
    return result


async def material_designation(diameter_mm: object, wall_thickness_mm: object) -> dict[str, Any]:
    """Derive the documented candidate designation from source-backed dimensions."""
    return await run_calculator(
        "material-designation",
        "--diameter-mm",
        _argument(diameter_mm),
        "--wall-thickness-mm",
        _argument(wall_thickness_mm),
    )


async def raw_material(designation: object) -> dict[str, Any]:
    """Look up whether a source-backed designation appears in the guideline."""
    return await run_calculator("raw-material", "--designation", _argument(designation))


async def tube_length(
    diameter_mm: object, wall_thickness_mm: object, total_length_mm: object
) -> dict[str, Any]:
    """Assess one source-backed tube length using the calculator."""
    return await run_calculator(
        "tube-length",
        "--diameter-mm",
        _argument(diameter_mm),
        "--wall-thickness-mm",
        _argument(wall_thickness_mm),
        "--total-length-mm",
        _argument(total_length_mm),
    )


async def bend_angles(angles_deg: list[object]) -> dict[str, Any]:
    """Assess every source-backed bend angle using repeated CLI options."""
    arguments: list[str] = []
    for angle in angles_deg:
        arguments.extend(("--angle-deg", _argument(angle)))
    if not arguments:
        raise CalculatorExecutionError("no source-backed bend angles were available")
    return await run_calculator("bend-angles", *arguments)


async def map_standard_bender_schedule(rows: list[dict[str, object]]) -> dict[str, Any]:
    """Validate/map the full ordered source schedule using only calculator logic."""
    return await run_calculator(
        "map-standard-bender-schedule", "--schedule-json", json.dumps(rows, separators=(",", ":"))
    )


async def auto_crimp(
    diameter_mm: object,
    l1_mm: object,
    l2_segments_mm: list[object],
    l3_mm: object,
    l4_mm: object,
    theta1_deg: object,
    theta4_deg: object,
) -> dict[str, Any]:
    """Assess automatic-crimp dimensions without model numeric reasoning."""
    arguments = [
        "--diameter-mm", _argument(diameter_mm),
        "--l1-mm", _argument(l1_mm),
        "--l3-mm", _argument(l3_mm),
        "--l4-mm", _argument(l4_mm),
        "--theta1-deg", _argument(theta1_deg),
        "--theta4-deg", _argument(theta4_deg),
    ]
    for segment in l2_segments_mm:
        arguments.extend(("--l2-mm", _argument(segment)))
    if not l2_segments_mm:
        raise CalculatorExecutionError("no source-backed L2 segments were available")
    return await run_calculator("auto-crimp", *arguments)


async def robot_pickup(diameter_mm: object, l1_mm: object, l2_mm: object) -> dict[str, Any]:
    """Assess the separate robot-pickup condition through the calculator."""
    return await run_calculator(
        "robot-pickup",
        "--diameter-mm", _argument(diameter_mm),
        "--l1-mm", _argument(l1_mm),
        "--l2-mm", _argument(l2_mm),
    )
