"""Strict request, extraction, calculator, and UI-report contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


BoundedText = StringConstraints(strip_whitespace=True, max_length=4_000)
MeasurementText = StringConstraints(strip_whitespace=True, min_length=1, max_length=100)
MAX_CRITERION_EVIDENCE = 60


class StrictModel(BaseModel):
    """Close every model to unrecognised fields for model and API safety."""

    model_config = ConfigDict(extra="forbid")


class DrawingEvidence(StrictModel):
    """Preserve one bounded source reference without interpreting it as instructions."""

    page: str = Field(min_length=1, max_length=40)
    source_label: str = Field(min_length=1, max_length=240)
    raw_text: str = Field(min_length=1, max_length=1_000)


class ExtractedMeasurement(StrictModel):
    """Carry a source-backed drawing value or explicitly record its absence."""

    value: str | None = Field(default=None, max_length=100)
    state: Literal["stated", "mapped", "derived", "not_stated", "unreadable"]
    evidence: list[DrawingEvidence] = Field(default_factory=list, max_length=12)


class ExtractedFinding(StrictModel):
    """Represent a feature observation under the policy closed-world rules."""

    name: str = Field(min_length=1, max_length=160)
    state: Literal["present", "absent_by_drawing_omission", "not_stated", "unreadable"]
    evidence: list[DrawingEvidence] = Field(default_factory=list, max_length=12)


class BendScheduleRow(StrictModel):
    """Keep an ordered schedule transcription for calculator-only mapping."""

    point: str = Field(min_length=1, max_length=100)
    bend_angle_deg: str | None = Field(default=None, max_length=100)
    bend_radius_mm: str | None = Field(default=None, max_length=100)
    straight_part_mm: str | None = Field(default=None, max_length=100)
    evidence: list[DrawingEvidence] = Field(default_factory=list, max_length=12)


class PipeCandidate(StrictModel):
    """One source-associated pipe configuration with its stated end-option set."""

    id: str = Field(min_length=1, max_length=100)
    label: str = Field(min_length=1, max_length=200)
    completeness: Literal["complete", "partial"]
    outer_diameter_mm: ExtractedMeasurement
    wall_thickness_mm: ExtractedMeasurement
    raw_material_designation: ExtractedMeasurement
    total_length_mm: ExtractedMeasurement
    bend_angles_deg: list[ExtractedMeasurement] = Field(default_factory=list, max_length=50)
    bend_schedule: list[BendScheduleRow] = Field(default_factory=list, max_length=80)
    end_configurations: list[ExtractedFinding] = Field(
        min_length=1,
        max_length=8,
        description=(
            "Every source-stated end/seal option, one item per option. For a callout such "
            "as 'Parker EO2 or CAST B4', emit separate present findings named 'Parker "
            "EO2' and 'CAST B4', each with the same source evidence. Do not leave this "
            "not_stated when a legible end-option callout exists."
        ),
    )
    robot_pickup_required: ExtractedFinding
    automatic_crimp_left: ExtractedFinding
    automatic_crimp_right: ExtractedFinding
    prohibited_features: list[ExtractedFinding] = Field(default_factory=list, max_length=20)


class PipeSpecification(StrictModel):
    """Structured model output from the drawing-evidence extraction stage."""

    drawing_scope_complete: bool
    candidates: list[PipeCandidate] = Field(min_length=1, max_length=20)
    extraction_limitations: list[str] = Field(default_factory=list, max_length=30)


class CriterionResult(StrictModel):
    """One UI-renderable rule outcome with evidence references and provenance."""

    id: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=160)
    status: Literal["Pass", "Warning", "Fail", "Cannot assess", "Met", "Not met"]
    explanation: str = Field(min_length=1, max_length=1_000)
    evidence: list[DrawingEvidence] = Field(
        default_factory=list, max_length=MAX_CRITERION_EVIDENCE
    )
    calculator: dict[str, object] | None = None


class CandidateReport(StrictModel):
    """Assessment report for one explicitly associated drawing configuration."""

    id: str = Field(min_length=1, max_length=100)
    label: str = Field(min_length=1, max_length=200)
    completeness: Literal["complete", "partial"]
    result: Literal[
        "Feasible against the assessed PS4 requirements",
        "Not feasible against the PS4 requirements",
        "Inconclusive",
    ]
    criteria: list[CriterionResult] = Field(min_length=1, max_length=40)
    limitations: list[str] = Field(default_factory=list, max_length=30)


class DrawingSource(StrictModel):
    """Safe drawing metadata returned to the browser; no source bytes are retained."""

    file_name: str = Field(min_length=1, max_length=500)
    media_type: str = Field(min_length=1, max_length=100)
    byte_size: int = Field(ge=1)


class DrawingFeasibilityReport(StrictModel):
    """Complete transient PS4 report returned by the standalone endpoint."""

    correlation_id: str = Field(min_length=1, max_length=100)
    source: DrawingSource
    overall_result: Literal[
        "Feasible against the assessed PS4 requirements",
        "Not feasible against the PS4 requirements",
        "Inconclusive",
        "Configuration-dependent",
    ]
    candidate_reports: list[CandidateReport] = Field(min_length=1, max_length=20)
    scope_notice: str = Field(min_length=1, max_length=500)
    extraction_limitations: list[str] = Field(default_factory=list, max_length=30)
    guideline_limitations: list[str] = Field(default_factory=list, max_length=30)


class DrawingInput(StrictModel):
    """Validated transient bytes and sanitized metadata for a single upload."""

    file_name: str = Field(min_length=1, max_length=500)
    media_type: str = Field(min_length=1, max_length=100)
    content: bytes = Field(min_length=1)


class TelemetryContext(StrictModel):
    """Non-sensitive caller context used only for safe usage telemetry."""

    correlation_id: str = Field(min_length=1, max_length=100)
    user_id: str | None = Field(default=None, max_length=500)
    client_host: str | None = Field(default=None, max_length=500)
