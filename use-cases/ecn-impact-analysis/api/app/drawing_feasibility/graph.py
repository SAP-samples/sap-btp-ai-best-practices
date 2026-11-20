"""Bounded LangGraph workflow and deterministic PS4 report assembly."""

from __future__ import annotations

import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.drawing_feasibility.calculator_tools import (
    CalculatorExecutionError,
    auto_crimp,
    bend_angles,
    map_standard_bender_schedule,
    material_designation,
    raw_material,
    robot_pickup,
    tube_length,
)
from app.drawing_feasibility.models import (
    BendScheduleRow,
    CandidateReport,
    CriterionResult,
    DrawingEvidence,
    DrawingFeasibilityReport,
    DrawingInput,
    DrawingSource,
    ExtractedFinding,
    ExtractedMeasurement,
    MAX_CRITERION_EVIDENCE,
    PipeCandidate,
    PipeSpecification,
    TelemetryContext,
)
from app.drawing_feasibility.policy import PolicyBundle
from app.drawing_feasibility.provider import DrawingExtractionProvider


_SCOPE_NOTICE = (
    "This is a PS4 technical-feasibility assessment based only on the submitted "
    "drawing and the assessed PS4 requirements. It is not a general manufacturing, "
    "quality, tooling, tolerance, cost, capacity, or release approval."
)
_ROBOT_NOTICE = (
    "Robot pickup is not explicitly required by the drawing, so this does not change "
    "the technical-feasibility result. However, the stated robot-pickup condition is "
    "not met; if robot pickup becomes required, this design would not meet that condition."
)
_PROHIBITED_FEATURES = (
    "banjo screw",
    "beads/form B",
    "tube diameter change",
    "end chamfers/machining",
    "tube+flexible/hose",
    "soldering/transplanting",
    "EO-2 soft seal",
    "metal seal",
)


class DrawingAssessmentState(TypedDict, total=False):
    """Partial, bounded data passed through the purpose-built graph nodes."""

    drawing: DrawingInput
    policy: PolicyBundle
    telemetry: TelemetryContext
    specification: PipeSpecification
    candidate_reports: list[CandidateReport]
    report: DrawingFeasibilityReport


def _evidence(
    *values: ExtractedMeasurement | ExtractedFinding | BendScheduleRow,
) -> list[DrawingEvidence]:
    """Return bounded, deduplicated evidence in source declaration order."""
    result: list[DrawingEvidence] = []
    seen: set[tuple[str, str, str]] = set()
    for value in values:
        for item in value.evidence:
            key = (item.page, item.source_label, item.raw_text)
            if key not in seen:
                result.append(item)
                seen.add(key)
                if len(result) == MAX_CRITERION_EVIDENCE:
                    return result
    return result


def _criterion(
    identifier: str,
    title: str,
    status: str,
    explanation: str,
    evidence: list[DrawingEvidence] | None = None,
    calculator: dict[str, object] | None = None,
) -> CriterionResult:
    """Create one controlled UI criterion without model-authored prose."""
    return CriterionResult(
        id=identifier,
        title=title,
        status=status,  # type: ignore[arg-type]
        explanation=explanation,
        evidence=list((evidence or [])[:MAX_CRITERION_EVIDENCE]),
        calculator=calculator,
    )


def _not_assessable(
    identifier: str, title: str, explanation: str, *inputs: ExtractedMeasurement | ExtractedFinding
) -> CriterionResult:
    """Create a consistent unknown-result criterion with available evidence."""
    return _criterion(identifier, title, "Cannot assess", explanation, _evidence(*inputs))


def _source_value(measurement: ExtractedMeasurement) -> str | None:
    """Return a directly stated/mapped value; never a missing or unreadable value."""
    return measurement.value if measurement.state in {"stated", "mapped", "derived"} else None


def _schedule_rows(candidate: PipeCandidate) -> list[dict[str, object]]:
    """Translate exact transcribed schedule rows into calculator's documented JSON shape."""
    return [
        {
            "point": row.point,
            "bendAngleDeg": row.bend_angle_deg,
            "bendRadiusMm": row.bend_radius_mm,
            "straightPartMm": row.straight_part_mm,
        }
        for row in candidate.bend_schedule
    ]


def _end_option_key(finding: ExtractedFinding) -> str:
    """Return a punctuation-insensitive key for a source-stated end option."""
    return _feature_key(finding.name)


def _split_end_option_labels(raw_text: str) -> list[str]:
    """Split a source label's explicit `or` alternatives without inventing values."""
    return [
        label.strip(" ,;")
        for label in re.split(r"\s+\bor\b\s+", raw_text, flags=re.IGNORECASE)
        if label.strip(" ,;")
    ]


def _end_option_labels(finding: ExtractedFinding) -> list[str]:
    """Return the stated alternatives from one source-backed end option finding."""
    return _split_end_option_labels(finding.name)


def _end_option_kinds_for_label(label: str) -> set[str]:
    """Classify one stated end label against the documented PS4 choices."""
    key = _feature_key(label)
    kinds: set[str] = set()
    if "free" in key or "b4" in key:
        kinds.add("supported")
    if any(marker in key for marker in ("eo2", "metal", "parkereo", "castb3")):
        kinds.add("unsupported")
    return kinds


def _end_option_kinds(finding: ExtractedFinding) -> set[str]:
    """Classify documented end options without treating an option as a selection."""
    kinds: set[str] = set()
    for label in _end_option_labels(finding):
        kinds.update(_end_option_kinds_for_label(label))
    return kinds


def _end_option_names(
    options: list[ExtractedFinding], kind: str
) -> list[str]:
    """Return distinct source labels for one PS4 end-compatibility disposition."""
    names: list[str] = []
    seen: set[str] = set()
    for option in options:
        for label in _end_option_labels(option):
            if kind not in _end_option_kinds_for_label(label):
                continue
            key = _feature_key(label)
            if key not in seen:
                names.append(label)
                seen.add(key)
    return names


def _format_end_options(disposition: str, names: list[str]) -> str:
    """Format source-stated compatible or non-compatible end labels for the report."""
    noun = "option" if len(names) == 1 else "options"
    return f"{disposition} {noun}: {', '.join(names)}."


def _present_end_options(candidate: PipeCandidate) -> list[ExtractedFinding]:
    """Return stated options, recovering a complete `or` callout from its evidence if needed."""
    options = [
        option
        for option in candidate.end_configurations
        if option.state == "present"
    ]
    if options:
        return options

    recovered: list[ExtractedFinding] = []
    recovered_kinds: set[str] = set()
    seen: set[tuple[str, str, str]] = set()
    for finding in candidate.prohibited_features:
        if finding.state != "present":
            continue
        for evidence in finding.evidence:
            raw_text = evidence.raw_text.strip()
            labels = _split_end_option_labels(raw_text) if len(raw_text) <= 160 else []
            kinds = {
                kind
                for label in labels
                for kind in _end_option_kinds_for_label(label)
            }
            key = (evidence.page, evidence.source_label, raw_text)
            if "or" not in raw_text.casefold() or not kinds or key in seen:
                continue
            recovered.append(
                ExtractedFinding(name=raw_text, state="present", evidence=[evidence])
            )
            recovered_kinds.update(kinds)
            seen.add(key)
    return recovered if {"supported", "unsupported"} <= recovered_kinds else []


def _has_supported_end_option(candidate: PipeCandidate) -> bool:
    """Identify whether the drawing lists at least one PS4-compatible end option."""
    return any("supported" in _end_option_kinds(option) for option in _present_end_options(candidate))


def _is_end_option_for_prohibited_feature(
    expected: str, candidate: PipeCandidate
) -> bool:
    """Match an incompatible end option to its PS4 prohibited-feature label."""
    expected_key = _feature_key(expected)
    for option in _present_end_options(candidate):
        label = _end_option_key(option)
        if expected_key == _feature_key("EO-2 soft seal") and "eo2" in label:
            return True
        if expected_key == _feature_key("metal seal") and any(
            marker in label for marker in ("metal", "parkereo", "castb3")
        ):
            return True
    return False


def _end_configuration(candidate: PipeCandidate) -> tuple[CriterionResult, list[str]]:
    """Assess stated alternatives; retain a warning when only some options are supported."""
    options = _present_end_options(candidate)
    supported = [option for option in options if "supported" in _end_option_kinds(option)]
    unsupported = [option for option in options if "unsupported" in _end_option_kinds(option)]
    evidence = _evidence(*options)
    if supported and unsupported:
        compatible_options = _end_option_names(options, "supported")
        non_compatible_options = _end_option_names(options, "unsupported")
        compatibility_detail = " ".join(
            (
                _format_end_options("Compatible", compatible_options),
                _format_end_options("Non-compatible", non_compatible_options),
            )
        )
        warning = f"Warning: Configuration-dependent end compatibility. {compatibility_detail}"
        return (
            _criterion(
                "end_configuration",
                "End configuration",
                "Warning",
                compatibility_detail,
                evidence,
            ),
            [warning],
        )
    if supported:
        return (
            _criterion(
                "end_configuration",
                "End configuration",
                "Pass",
                "All stated end configurations are free ends or B4 soft-seal ends.",
                evidence,
            ),
            [],
        )
    if unsupported:
        return (
            _criterion(
                "end_configuration",
                "End configuration",
                "Fail",
                "None of the stated end configurations is supported by the assessed PS4 requirements.",
                evidence,
            ),
            [],
        )
    return (
        _not_assessable(
            "end_configuration",
            "End configuration",
            "The drawing does not state an end configuration covered by the assessed PS4 requirements.",
            *candidate.end_configurations,
        ),
        [],
    )


def _feature_key(name: str) -> str:
    """Compare controlled prohibited-feature labels without relying on punctuation."""
    return "".join(character for character in name.casefold() if character.isalnum())


def _prohibited_feature_criteria(
    candidate: PipeCandidate, *, drawing_scope_complete: bool
) -> list[CriterionResult]:
    """Assess all named PS4 prohibited features using closed-world extraction states."""
    by_name = {_feature_key(finding.name): finding for finding in candidate.prohibited_features}
    criteria: list[CriterionResult] = []
    for expected in _PROHIBITED_FEATURES:
        finding = by_name.get(_feature_key(expected))
        if finding is None and drawing_scope_complete:
            criteria.append(_criterion(f"prohibited_{expected}", f"Prohibited feature: {expected}", "Pass", "The complete drawing was reviewed and the prohibited feature is absent by drawing omission."))
        elif finding is None:
            criteria.append(_not_assessable(f"prohibited_{expected}", f"Prohibited feature: {expected}", "The drawing scope is incomplete, so this prohibited feature cannot be determined."))
        elif finding.state == "present":
            if _has_supported_end_option(candidate) and _is_end_option_for_prohibited_feature(
                expected, candidate
            ):
                criteria.append(
                    _criterion(
                        f"prohibited_{expected}",
                        f"Prohibited feature: {expected}",
                        "Warning",
                        "The drawing lists this unsupported seal only as an alternative; a compatible free-end or B4 option is also stated and must be selected.",
                        _evidence(finding, *_present_end_options(candidate)),
                    )
                )
            else:
                criteria.append(_criterion(f"prohibited_{expected}", f"Prohibited feature: {expected}", "Fail", "The drawing indicates a prohibited feature.", _evidence(finding)))
        elif finding.state == "absent_by_drawing_omission":
            criteria.append(_criterion(f"prohibited_{expected}", f"Prohibited feature: {expected}", "Pass", "The complete drawing was reviewed and the prohibited feature is absent by drawing omission.", _evidence(finding)))
        else:
            criteria.append(_not_assessable(f"prohibited_{expected}", f"Prohibited feature: {expected}", "The prohibited feature could not be determined from the drawing.", finding))
    return criteria


async def _material_criterion(candidate: PipeCandidate) -> CriterionResult:
    """Derive the PS4 list key from dimensions; retain source material text as evidence."""
    diameter = _source_value(candidate.outer_diameter_mm)
    wall = _source_value(candidate.wall_thickness_mm)
    evidence = _evidence(
        candidate.raw_material_designation,
        candidate.outer_diameter_mm,
        candidate.wall_thickness_mm,
    )
    if diameter is None or wall is None:
        return _not_assessable(
            "raw_material",
            "Raw material",
            "Source-backed outer diameter and wall thickness are required to derive the PS4 material designation.",
            candidate.raw_material_designation,
            candidate.outer_diameter_mm,
            candidate.wall_thickness_mm,
        )
    try:
        derived = await material_designation(diameter, wall)
    except CalculatorExecutionError:
        return _not_assessable(
            "raw_material",
            "Raw material",
            "The documented PS4 material-designation calculation could not be evaluated.",
            candidate.raw_material_designation,
            candidate.outer_diameter_mm,
            candidate.wall_thickness_mm,
        )
    designation = str(derived.get("candidate") or "") or None
    if designation is None:
        return _not_assessable(
            "raw_material",
            "Raw material",
            "No PS4 material designation can be derived from the source-backed dimensions.",
            candidate.raw_material_designation,
            candidate.outer_diameter_mm,
            candidate.wall_thickness_mm,
        )
    try:
        result = await raw_material(designation)
    except CalculatorExecutionError:
        return _not_assessable(
            "raw_material",
            "Raw material",
            "The documented raw-material lookup could not be evaluated.",
            candidate.raw_material_designation,
            candidate.outer_diameter_mm,
            candidate.wall_thickness_mm,
        )
    calculator = {"material_designation": derived, "raw_material": result}
    if result.get("guidelineCoverage") == "listed":
        return _criterion(
            "raw_material",
            "Raw material",
            "Pass",
            f"The calculator-derived PS4 designation {designation} is listed in the guideline.",
            evidence,
            calculator,
        )
    return _criterion(
        "raw_material",
        "Raw material",
        "Cannot assess",
        f"The calculator-derived PS4 designation {designation} is not listed; the guideline gives no pass/fail disposition.",
        evidence,
        calculator,
    )


async def _tube_length_criterion(candidate: PipeCandidate) -> CriterionResult:
    """Run the calculator rather than manually comparing length limits."""
    diameter = _source_value(candidate.outer_diameter_mm)
    wall_thickness = _source_value(candidate.wall_thickness_mm)
    length = _source_value(candidate.total_length_mm)
    if diameter is None or wall_thickness is None or length is None:
        return _not_assessable(
            "tube_length",
            "Tube length",
            "Source-backed outer diameter, wall thickness, and total length are all required.",
            candidate.outer_diameter_mm,
            candidate.wall_thickness_mm,
            candidate.total_length_mm,
        )
    try:
        result = await tube_length(diameter, wall_thickness, length)
    except CalculatorExecutionError:
        return _not_assessable(
            "tube_length",
            "Tube length",
            "The documented tube-length calculation could not be evaluated.",
            candidate.outer_diameter_mm,
            candidate.wall_thickness_mm,
            candidate.total_length_mm,
        )
    status = result.get("status") if result.get("status") in {"Pass", "Fail", "Cannot assess"} else "Cannot assess"
    return _criterion(
        "tube_length",
        "Tube length",
        status,
        "Result is determined by the documented PS4 tube-length calculation.",
        _evidence(
            candidate.outer_diameter_mm,
            candidate.wall_thickness_mm,
            candidate.total_length_mm,
        ),
        {"tube_length": result},
    )


async def _bend_angle_criterion(candidate: PipeCandidate) -> CriterionResult:
    """Run calculator-owned bend-angle checks for all explicitly extracted angles."""
    values = [_source_value(angle) for angle in candidate.bend_angles_deg]
    if not values or any(value is None for value in values):
        return _not_assessable("bend_angles", "Bend angles", "All assessed bend angles must be source-backed and readable.", *candidate.bend_angles_deg)
    try:
        result = await bend_angles([value for value in values if value is not None])
    except CalculatorExecutionError:
        return _not_assessable("bend_angles", "Bend angles", "The documented bend-angle calculation could not be evaluated.", *candidate.bend_angles_deg)
    status = result.get("status") if result.get("status") in {"Pass", "Fail", "Cannot assess"} else "Cannot assess"
    return _criterion("bend_angles", "Bend angles", status, "All extracted bend angles were assessed by the documented calculator.", _evidence(*candidate.bend_angles_deg), {"bend_angles": result})


def _automatic_crimp_objective(
    geometry: CriterionResult,
) -> CriterionResult:
    """Project calculator-derived crimp geometry into the customer-facing objective."""
    if geometry.status == "Pass":
        status = "Pass"
        explanation = (
            "All calculator-assessed automatic-crimp geometry meets the PS4 minimums."
        )
    elif geometry.status == "Fail":
        status = "Fail"
        explanation = (
            "One or more calculator-assessed automatic-crimp dimensions does not meet "
            "the PS4 minimums."
        )
    else:
        status = "Warning"
        explanation = (
            "Automatic-crimp geometry cannot be fully assessed from the documented "
            "schedule."
        )
    return _criterion(
        "automatic_crimp_objective",
        "Automatic-crimp objective",
        status,
        explanation,
        geometry.evidence,
        geometry.calculator,
    )


async def _crimp_and_robot(
    candidate: PipeCandidate, *, drawing_scope_complete: bool
) -> tuple[CriterionResult, CriterionResult, CriterionResult, list[str]]:
    """Map schedule once and run calculator-only crimp/robot evaluations when valid."""
    limitations: list[str] = []
    if not candidate.bend_schedule:
        geometry = _criterion(
            "automatic_crimp_dimensions",
            "Automatic-crimp dimensions",
            "Warning",
            "No complete ordered bend schedule was extracted.",
        )
        robot = _not_assessable(
            "robot_pickup",
            "Robot pickup",
            "No complete ordered bend schedule was extracted.",
            candidate.robot_pickup_required,
        )
        return _automatic_crimp_objective(geometry), geometry, robot, limitations
    try:
        mapping = await map_standard_bender_schedule(_schedule_rows(candidate))
    except CalculatorExecutionError:
        mapping = {
            "mappingStatus": "Cannot assess",
            "reasons": ["The schedule mapping tool could not evaluate the supplied rows."],
        }
    schedule_evidence = _evidence(*candidate.bend_schedule)
    if mapping.get("mappingStatus") != "Mapped":
        reason = "The complete ordered schedule could not be mapped by the documented calculator."
        limitations.append(reason)
        geometry = _criterion(
            "automatic_crimp_dimensions",
            "Automatic-crimp dimensions",
            "Warning",
            reason,
            schedule_evidence,
            {"schedule_mapping": mapping},
        )
        robot = _criterion(
            "robot_pickup",
            "Robot pickup",
            "Cannot assess",
            reason,
            schedule_evidence,
            {"schedule_mapping": mapping},
        )
        return _automatic_crimp_objective(geometry), geometry, robot, limitations
    diameter = _source_value(candidate.outer_diameter_mm)
    l1 = ((mapping.get("l1") or {}).get("valueMm"))
    l2 = [
        segment.get("valueMm")
        for segment in mapping.get("l2Segments", [])
        if isinstance(segment, dict)
    ]
    l3 = ((mapping.get("l3") or {}).get("l3Mm"))
    l4 = ((mapping.get("l4") or {}).get("valueMm"))
    theta1 = ((mapping.get("theta1") or {}).get("valueDeg"))
    theta4 = ((mapping.get("theta4") or {}).get("valueDeg"))
    if not all((diameter, l1, l2, l3, l4, theta1, theta4)):
        reason = "The mapped schedule lacks one or more dimensions required by the documented calculator."
        limitations.append(reason)
        geometry = _criterion(
            "automatic_crimp_dimensions",
            "Automatic-crimp dimensions",
            "Warning",
            reason,
            schedule_evidence,
            {"schedule_mapping": mapping},
        )
        robot = _criterion(
            "robot_pickup",
            "Robot pickup",
            "Cannot assess",
            reason,
            schedule_evidence,
            {"schedule_mapping": mapping},
        )
        return _automatic_crimp_objective(geometry), geometry, robot, limitations
    try:
        crimp = await auto_crimp(diameter, l1, l2, l3, l4, theta1, theta4)
        calculator_status = crimp.get("overallStatus")
        geometry_status = (
            calculator_status
            if calculator_status in {"Pass", "Fail"}
            else "Warning"
        )
        geometry = _criterion(
            "automatic_crimp_dimensions",
            "Automatic-crimp dimensions",
            geometry_status,
            "Result is determined by the documented automatic-crimp calculator.",
            schedule_evidence,
            {"schedule_mapping": mapping, "auto_crimp": crimp},
        )
    except CalculatorExecutionError:
        geometry = _criterion(
            "automatic_crimp_dimensions",
            "Automatic-crimp dimensions",
            "Warning",
            "The documented automatic-crimp calculation could not be evaluated.",
            schedule_evidence,
            {"schedule_mapping": mapping},
        )
    pickup_l2 = mapping.get("robotPickupL2")
    if not isinstance(pickup_l2, dict) or not pickup_l2.get("valueMm"):
        robot = _criterion(
            "robot_pickup",
            "Robot pickup",
            "Cannot assess",
            "The mapped schedule has no eligible robot-pickup L2 segment.",
            schedule_evidence,
            {"schedule_mapping": mapping},
        )
    else:
        try:
            pickup = await robot_pickup(diameter, l1, pickup_l2["valueMm"])
            pickup_status = (
                pickup.get("status")
                if pickup.get("status") in {"Met", "Not met", "Cannot assess"}
                else "Cannot assess"
            )
            robot = _criterion(
                "robot_pickup",
                "Robot pickup",
                pickup_status,
                "Result is determined by the documented robot-pickup calculator.",
                schedule_evidence,
                {"schedule_mapping": mapping, "robot_pickup": pickup},
            )
        except CalculatorExecutionError:
            robot = _criterion(
                "robot_pickup",
                "Robot pickup",
                "Cannot assess",
                "The documented robot-pickup calculation could not be evaluated.",
                schedule_evidence,
                {"schedule_mapping": mapping},
            )
    return _automatic_crimp_objective(geometry), geometry, robot, limitations


def _candidate_result(criteria: list[CriterionResult], robot_required: ExtractedFinding) -> tuple[str, list[str]]:
    """Apply only stated gating rules to controlled criterion outcomes."""
    limitations: list[str] = []
    gated = [criterion for criterion in criteria if criterion.id not in {"automatic_crimp_objective", "automatic_crimp_dimensions", "robot_pickup"}]
    robot = next(criterion for criterion in criteria if criterion.id == "robot_pickup")
    if robot_required.state == "present":
        if robot.status == "Not met":
            gated.append(_criterion("robot_pickup_gating", "Robot pickup requirement", "Fail", "The drawing explicitly requires robot pickup, but the calculator condition is not met.", robot.evidence, robot.calculator))
        elif robot.status == "Cannot assess":
            gated.append(_criterion("robot_pickup_gating", "Robot pickup requirement", "Cannot assess", "The drawing explicitly requires robot pickup, but it cannot be assessed.", robot.evidence, robot.calculator))
    elif robot.status == "Not met":
        limitations.append(_ROBOT_NOTICE)
    statuses = {criterion.status for criterion in gated}
    if "Fail" in statuses:
        return "Not feasible against the PS4 requirements", limitations
    if "Cannot assess" in statuses:
        return "Inconclusive", limitations
    return "Feasible against the assessed PS4 requirements", limitations


async def evaluate_candidate(
    candidate: PipeCandidate, *, drawing_scope_complete: bool
) -> CandidateReport:
    """Evaluate one explicitly associated configuration without LLM arithmetic."""
    end_configuration, end_warnings = _end_configuration(candidate)
    criteria = [
        await _material_criterion(candidate),
        await _tube_length_criterion(candidate),
        await _bend_angle_criterion(candidate),
        end_configuration,
        *_prohibited_feature_criteria(candidate, drawing_scope_complete=drawing_scope_complete),
    ]
    crimp_objective, crimp_dimensions, robot, calculator_limitations = await _crimp_and_robot(
        candidate, drawing_scope_complete=drawing_scope_complete
    )
    criteria.extend((crimp_objective, crimp_dimensions, robot))
    result, result_limitations = _candidate_result(criteria, candidate.robot_pickup_required)
    return CandidateReport(
        id=candidate.id,
        label=candidate.label,
        completeness=candidate.completeness,
        result=result,  # type: ignore[arg-type]
        criteria=criteria,
        limitations=end_warnings + calculator_limitations + result_limitations,
    )


async def extract_pipe_specification(
    state: DrawingAssessmentState, provider: DrawingExtractionProvider
) -> dict[str, object]:
    """Call the only LLM stage to extract structured source-backed values."""
    return {
        "specification": await provider.extract(
            state["drawing"], state["policy"], state["telemetry"]
        )
    }


async def evaluate_ps4_requirements(state: DrawingAssessmentState) -> dict[str, object]:
    """Run deterministic calculator-backed assessment for each drawing candidate."""
    specification = state["specification"]
    return {
        "candidate_reports": [
            await evaluate_candidate(
                candidate, drawing_scope_complete=specification.drawing_scope_complete
            )
            for candidate in specification.candidates
        ]
    }


def render_report(state: DrawingAssessmentState) -> dict[str, object]:
    """Project validated workflow state into a safe transient UI report."""
    reports = state["candidate_reports"]
    results = {report.result for report in reports}
    overall = reports[0].result if len(reports) == 1 else "Configuration-dependent"
    return {
        "report": DrawingFeasibilityReport(
            correlation_id=state["telemetry"].correlation_id,
            source=DrawingSource(file_name=state["drawing"].file_name, media_type=state["drawing"].media_type, byte_size=len(state["drawing"].content)),
            overall_result=overall,  # type: ignore[arg-type]
            candidate_reports=reports,
            scope_notice=_SCOPE_NOTICE,
            extraction_limitations=state["specification"].extraction_limitations,
            guideline_limitations=[] if results else ["No candidate configuration was available for assessment."],
        )
    }


def create_drawing_assessment_graph(provider: DrawingExtractionProvider):
    """Compile the fixed finite graph; no agent loop or unbounded tool selection exists."""
    async def extraction_node(state: DrawingAssessmentState) -> dict[str, object]:
        """Bind the runtime-owned provider outside the serializable graph state."""
        return await extract_pipe_specification(state, provider)

    graph = StateGraph(DrawingAssessmentState)
    graph.add_node("extract_pipe_specification", extraction_node)
    graph.add_node("evaluate_ps4_requirements", evaluate_ps4_requirements)
    graph.add_node("render_report", render_report)
    graph.add_edge(START, "extract_pipe_specification")
    graph.add_edge("extract_pipe_specification", "evaluate_ps4_requirements")
    graph.add_edge("evaluate_ps4_requirements", "render_report")
    graph.add_edge("render_report", END)
    # The provider is captured by the node closure rather than retained in
    # serializable graph state or report output.
    return graph.compile()
