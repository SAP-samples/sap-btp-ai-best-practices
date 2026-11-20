"""Focused contracts for the standalone transient drawing-feasibility flow."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from app.drawing_feasibility.drawing_input import validate_drawing_metadata
from app.drawing_feasibility.errors import DrawingValidationError
from app.drawing_feasibility.graph import evaluate_candidate
from app.drawing_feasibility.models import (
    DrawingEvidence,
    DrawingFeasibilityReport,
    DrawingInput,
    ExtractedFinding,
    ExtractedMeasurement,
    PipeCandidate,
    PipeSpecification,
    TelemetryContext,
)
from app.drawing_feasibility.policy import load_policy_bundle
from app.drawing_feasibility.provider import NativeDrawingExtractionProvider
from app.drawing_feasibility.service import DrawingFeasibilityService
from app.drawing_feasibility.settings import (
    DrawingFeasibilitySettings,
    load_drawing_feasibility_settings,
)
from app.main import create_app
from fastapi.testclient import TestClient


@pytest.fixture
def evidence() -> DrawingEvidence:
    """Provide one non-sensitive synthetic source reference."""
    return DrawingEvidence(page="1", source_label="Schedule", raw_text="Synthetic test value")


def _measurement(value: str, evidence: DrawingEvidence) -> ExtractedMeasurement:
    """Build a directly stated synthetic dimension."""
    return ExtractedMeasurement(value=value, state="stated", evidence=[evidence])


def _candidate(evidence: DrawingEvidence) -> PipeCandidate:
    """Build one complete synthetic PS4 candidate with a valid mapped schedule."""
    omitted = lambda name: ExtractedFinding(name=name, state="absent_by_drawing_omission")
    return PipeCandidate(
        id="candidate-1",
        label="Primary configuration",
        completeness="complete",
        outer_diameter_mm=_measurement("10", evidence),
        wall_thickness_mm=_measurement("1", evidence),
        raw_material_designation=_measurement("Ø10TH1", evidence),
        total_length_mm=_measurement("1000", evidence),
        bend_angles_deg=[_measurement("90", evidence), _measurement("90", evidence)],
        bend_schedule=[
            {"point": "P0", "bend_angle_deg": None, "bend_radius_mm": None, "straight_part_mm": "100", "evidence": [evidence]},
            {"point": "B1", "bend_angle_deg": "90", "bend_radius_mm": "50", "straight_part_mm": "100", "evidence": [evidence]},
            {"point": "B2", "bend_angle_deg": "90", "bend_radius_mm": "50", "straight_part_mm": "100", "evidence": [evidence]},
            {"point": "P3", "bend_angle_deg": None, "bend_radius_mm": None, "straight_part_mm": None, "evidence": [evidence]},
        ],
        end_configurations=[
            ExtractedFinding(name="free end", state="present", evidence=[evidence])
        ],
        robot_pickup_required=ExtractedFinding(name="robot pickup", state="absent_by_drawing_omission"),
        automatic_crimp_left=ExtractedFinding(name="left automatic crimp", state="present", evidence=[evidence]),
        automatic_crimp_right=ExtractedFinding(name="right automatic crimp", state="present", evidence=[evidence]),
        prohibited_features=[omitted(name) for name in (
            "banjo screw", "beads/form B", "tube diameter change", "end chamfers/machining",
            "tube+flexible/hose", "soldering/transplanting", "EO-2 soft seal", "metal seal",
        )],
    )


@pytest.mark.asyncio
async def test_evaluate_candidate_uses_calculator_backed_ps4_results(
    evidence: DrawingEvidence,
) -> None:
    """Keep numeric checks in the packaged calculator and produce a safe report."""
    report = await evaluate_candidate(_candidate(evidence), drawing_scope_complete=True)

    assert report.result == "Feasible against the assessed PS4 requirements"
    criteria = {criterion.id: criterion for criterion in report.criteria}
    assert criteria["tube_length"].calculator is not None
    tube_length = criteria["tube_length"].calculator["tube_length"]
    assert tube_length["wallThicknessMm"] == "1"
    assert tube_length["maximumMm"] == "1500"
    assert criteria["automatic_crimp_dimensions"].status == "Pass"
    assert criteria["robot_pickup"].status == "Met"


@pytest.mark.asyncio
async def test_material_uses_dimension_derived_ps4_designation_over_source_grade_text(
    evidence: DrawingEvidence,
) -> None:
    """Avoid treating a source material grade or note as the PS4 lookup key."""
    candidate = _candidate(evidence)
    candidate.raw_material_designation = _measurement(
        "SFS-EN 10305-4 E235 Alternative/Additional materials. See notes.", evidence
    )

    report = await evaluate_candidate(candidate, drawing_scope_complete=True)

    criterion = {item.id: item for item in report.criteria}["raw_material"]
    assert criterion.status == "Pass"
    assert criterion.calculator == {
        "material_designation": {
            "operation": "material-designation",
            "diameterMm": "10",
            "wallThicknessMm": "1",
            "candidate": "Ø10TH1",
        },
        "raw_material": {
            "operation": "raw-material",
            "designation": "Ø10TH1",
            "guidelineCoverage": "listed",
        },
    }
    assert report.result == "Feasible against the assessed PS4 requirements"


@pytest.mark.asyncio
async def test_material_cannot_assess_without_source_backed_dimensions(
    evidence: DrawingEvidence,
) -> None:
    """Do not use a drawing material-grade field as a PS4 list key fallback."""
    candidate = _candidate(evidence)
    candidate.outer_diameter_mm = ExtractedMeasurement(state="not_stated")
    candidate.raw_material_designation = _measurement("Ø10TH1", evidence)

    report = await evaluate_candidate(candidate, drawing_scope_complete=True)

    criterion = {item.id: item for item in report.criteria}["raw_material"]
    assert criterion.status == "Cannot assess"
    assert criterion.calculator is None
    assert report.result == "Inconclusive"


@pytest.mark.asyncio
async def test_end_option_set_with_b4_reports_warning_not_failure(
    evidence: DrawingEvidence,
) -> None:
    """Keep an actionable warning when source-stated seal options include B4."""
    candidate = _candidate(evidence)
    candidate.end_configurations = [
        ExtractedFinding(
            name="Parker EO2 or CAST B4", state="present", evidence=[evidence]
        ),
        ExtractedFinding(
            name="Parker EO or CAST B3", state="present", evidence=[evidence]
        ),
    ]
    candidate.prohibited_features = [
        ExtractedFinding(name="EO-2 soft seal", state="present", evidence=[evidence]),
        ExtractedFinding(name="metal seal", state="present", evidence=[evidence]),
    ]

    report = await evaluate_candidate(candidate, drawing_scope_complete=True)

    criteria = {criterion.id: criterion for criterion in report.criteria}
    assert criteria["end_configuration"].status == "Warning"
    assert criteria["prohibited_EO-2 soft seal"].status == "Warning"
    assert criteria["prohibited_metal seal"].status == "Warning"
    assert criteria["end_configuration"].explanation == (
        "Compatible option: CAST B4. Non-compatible options: Parker EO2, Parker EO, CAST B3."
    )
    assert not any(criterion.status == "Fail" for criterion in report.criteria)
    assert report.result == "Feasible against the assessed PS4 requirements"
    assert report.limitations == [
        "Warning: Configuration-dependent end compatibility. Compatible option: CAST B4. "
        "Non-compatible options: Parker EO2, Parker EO, CAST B3."
    ]


@pytest.mark.asyncio
async def test_end_option_set_recovers_from_source_backed_prohibited_evidence(
    evidence: DrawingEvidence,
) -> None:
    """Avoid a false failure when extraction keeps the complete option callout as evidence."""
    candidate = _candidate(evidence)
    candidate.end_configurations = [
        ExtractedFinding(name="end configuration", state="not_stated")
    ]
    soft_seal_evidence = DrawingEvidence(
        page="1",
        source_label="End of the pipe / soft seal",
        raw_text="Parker EO2 or CAST B4",
    )
    metal_seal_evidence = DrawingEvidence(
        page="1",
        source_label="End of the pipe / metal seal",
        raw_text="Parker EO or CAST B3",
    )
    candidate.prohibited_features = [
        ExtractedFinding(
            name="EO-2 soft seal", state="present", evidence=[soft_seal_evidence]
        ),
        ExtractedFinding(
            name="metal seal", state="present", evidence=[metal_seal_evidence]
        ),
    ]

    report = await evaluate_candidate(candidate, drawing_scope_complete=True)

    criteria = {criterion.id: criterion for criterion in report.criteria}
    assert criteria["end_configuration"].status == "Warning"
    assert criteria["prohibited_EO-2 soft seal"].status == "Warning"
    assert criteria["prohibited_metal seal"].status == "Warning"
    assert criteria["end_configuration"].explanation == (
        "Compatible option: CAST B4. Non-compatible options: Parker EO2, Parker EO, CAST B3."
    )
    assert report.result == "Feasible against the assessed PS4 requirements"


@pytest.mark.asyncio
async def test_end_option_set_without_b4_or_free_end_fails(
    evidence: DrawingEvidence,
) -> None:
    """Do not convert a set of only incompatible end options into a warning."""
    candidate = _candidate(evidence)
    candidate.end_configurations = [
        ExtractedFinding(name="Parker EO2", state="present", evidence=[evidence]),
        ExtractedFinding(name="Parker EO", state="present", evidence=[evidence]),
        ExtractedFinding(name="CAST B3", state="present", evidence=[evidence]),
    ]

    report = await evaluate_candidate(candidate, drawing_scope_complete=True)

    criteria = {criterion.id: criterion for criterion in report.criteria}
    assert criteria["end_configuration"].status == "Fail"
    assert report.result == "Not feasible against the PS4 requirements"


@pytest.mark.asyncio
async def test_evaluate_candidate_fails_diameter_10_thickness_1_over_1500_mm(
    evidence: DrawingEvidence,
) -> None:
    """Apply the narrower tube-length range in the full deterministic assessment."""
    candidate = _candidate(evidence)
    candidate.total_length_mm = _measurement("1500.1", evidence)

    report = await evaluate_candidate(candidate, drawing_scope_complete=True)

    criteria = {criterion.id: criterion for criterion in report.criteria}
    tube_length = criteria["tube_length"]
    assert tube_length.status == "Fail"
    assert tube_length.calculator == {
        "tube_length": {
            "operation": "tube-length",
            "diameterMm": "10",
            "wallThicknessMm": "1",
            "totalLengthMm": "1500.1",
            "minimumMm": "200",
            "maximumMm": "1500",
            "status": "Fail",
        }
    }
    assert report.result == "Not feasible against the PS4 requirements"


@pytest.mark.asyncio
async def test_missing_prohibited_finding_passes_only_for_complete_drawing(
    evidence: DrawingEvidence,
) -> None:
    """Apply the documented closed-world omission rule without inventing evidence."""
    candidate = _candidate(evidence)
    candidate.prohibited_features = []

    complete = await evaluate_candidate(candidate, drawing_scope_complete=True)
    partial = await evaluate_candidate(candidate, drawing_scope_complete=False)
    complete_criteria = {item.id: item for item in complete.criteria}
    partial_criteria = {item.id: item for item in partial.criteria}

    assert complete_criteria["prohibited_banjo screw"].status == "Pass"
    assert partial_criteria["prohibited_banjo screw"].status == "Cannot assess"


@pytest.mark.asyncio
async def test_prohibited_feature_matching_does_not_depend_on_label_punctuation(
    evidence: DrawingEvidence,
) -> None:
    """Do not turn an extracted prohibited feature into an omission due to spelling style."""
    candidate = _candidate(evidence)
    candidate.prohibited_features = [
        ExtractedFinding(name="tube + flexible / hose", state="present", evidence=[evidence])
    ]

    report = await evaluate_candidate(candidate, drawing_scope_complete=True)
    criteria = {item.id: item for item in report.criteria}

    assert criteria["prohibited_tube+flexible/hose"].status == "Fail"
    assert report.result == "Not feasible against the PS4 requirements"


@pytest.mark.asyncio
async def test_automatic_crimp_supports_schedule_evidence_over_30_references(
    evidence: DrawingEvidence,
) -> None:
    """Keep a large, source-backed schedule from breaking report construction."""
    candidate = _candidate(evidence)
    schedule_evidence = [
        DrawingEvidence(
            page="1",
            source_label=f"Schedule row {index + 1}",
            raw_text=f"Source row {index + 1}",
        )
        for index in range(33)
    ]
    for row, start in zip(
        candidate.bend_schedule,
        range(0, len(candidate.bend_schedule) * 12, 12),
        strict=True,
    ):
        row.evidence = schedule_evidence[start : start + 12]

    report = await evaluate_candidate(candidate, drawing_scope_complete=True)

    criterion = {item.id: item for item in report.criteria}["automatic_crimp_dimensions"]
    assert criterion.status == "Pass"
    assert len(criterion.evidence) == 33
    assert criterion.evidence[0].source_label == "Schedule row 1"
    assert criterion.evidence[-1].source_label == "Schedule row 33"


@pytest.mark.asyncio
async def test_geometry_drives_automatic_crimp_objective_when_callout_is_omitted(
    evidence: DrawingEvidence,
) -> None:
    """Treat source-backed passing geometry as the crimp-objective evidence."""
    candidate = _candidate(evidence)
    candidate.automatic_crimp_right = ExtractedFinding(
        name="right automatic crimp", state="not_stated"
    )

    report = await evaluate_candidate(candidate, drawing_scope_complete=True)
    criteria = {item.id: item for item in report.criteria}

    assert criteria["automatic_crimp_dimensions"].status == "Pass"
    assert criteria["automatic_crimp_objective"].status == "Pass"
    assert report.result == "Feasible against the assessed PS4 requirements"


@pytest.mark.asyncio
async def test_failing_automatic_crimp_geometry_fails_the_objective(
    evidence: DrawingEvidence,
) -> None:
    """Keep the objective aligned with calculator-derived geometry failures."""
    candidate = _candidate(evidence)
    candidate.bend_schedule[0].straight_part_mm = "10"

    report = await evaluate_candidate(candidate, drawing_scope_complete=True)
    criteria = {item.id: item for item in report.criteria}

    assert criteria["automatic_crimp_dimensions"].status == "Fail"
    assert criteria["automatic_crimp_objective"].status == "Fail"
    assert report.result == "Feasible against the assessed PS4 requirements"


@pytest.mark.asyncio
async def test_unavailable_automatic_crimp_geometry_warns(
    evidence: DrawingEvidence,
) -> None:
    """Use warning rather than cannot-assess for an incomplete crimp geometry check."""
    candidate = _candidate(evidence)
    candidate.bend_schedule = []

    report = await evaluate_candidate(candidate, drawing_scope_complete=True)
    criteria = {item.id: item for item in report.criteria}

    assert criteria["automatic_crimp_dimensions"].status == "Warning"
    assert criteria["automatic_crimp_objective"].status == "Warning"
    assert report.result == "Feasible against the assessed PS4 requirements"


@pytest.mark.asyncio
async def test_service_runs_a_bounded_graph_with_a_local_fake_provider(
    evidence: DrawingEvidence,
) -> None:
    """Bind the provider in graph closure rather than serializable report state."""

    class ProviderFake:
        """Return an approved synthetic extraction without external I/O."""

        async def extract(self, _drawing, _policy, _telemetry):
            """Produce one complete synthetic drawing specification."""
            return PipeSpecification(
                drawing_scope_complete=True, candidates=[_candidate(evidence)]
            )

        async def close(self) -> None:
            """Implement the provider lifecycle contract."""

    service = DrawingFeasibilityService(
        DrawingFeasibilitySettings(
            model_name="synthetic-model", request_timeout_seconds=180, max_concurrency=1
        ),
        provider=ProviderFake(),
        policy=load_policy_bundle(),
    )
    report = await service.assess(
        DrawingInput(file_name="synthetic.png", media_type="image/png", content=b"test")
    )

    assert report.overall_result == "Feasible against the assessed PS4 requirements"
    assert report.source.byte_size == 4
    await service.close()


def test_policy_bundle_uses_exactly_the_packaged_markdown_allowlist() -> None:
    """Prevent accidental inclusion of reference PDFs, ZIPs, or directory scans."""
    policy = load_policy_bundle()
    assert tuple(policy.source_digests) == (
        "pipe-drawing-to-ps4-specification.md",
        "pipes-guideline-ps4.md",
        "ps4-pipe-evaluation-rubric.md",
        "tools-readme.md",
    )
    assert all(name.endswith(".md") for name in policy.source_digests)
    assert ".zip" not in policy.instructions.casefold()


def test_native_provider_builds_a_strict_local_multimodal_request(
    evidence: DrawingEvidence,
) -> None:
    """Verify SAP proxy payload shape without transmitting a drawing."""
    specification = PipeSpecification(
        drawing_scope_complete=True, candidates=[_candidate(evidence)]
    )

    class ResponsesFake:
        """Record one Responses request and return a synthetic strict result."""

        call = None

        async def create(self, **kwargs):
            """Return a response-shaped object without network activity."""
            self.call = kwargs
            return SimpleNamespace(output_text=specification.model_dump_json())

    responses = ResponsesFake()
    client = SimpleNamespace(responses=responses)
    provider = NativeDrawingExtractionProvider(
        DrawingFeasibilitySettings(
            model_name="synthetic-model", request_timeout_seconds=180, max_concurrency=1
        ), client=client
    )

    result = asyncio.run(
        provider.extract(
            DrawingInput(file_name="synthetic.png", media_type="image/png", content=b"image"),
            load_policy_bundle(),
            TelemetryContext(correlation_id="provider-test"),
        )
    )

    assert result == specification
    assert responses.call["store"] is False
    assert responses.call["text"]["format"]["strict"] is True
    end_options_schema = responses.call["text"]["format"]["schema"]["$defs"][
        "PipeCandidate"
    ]["properties"]["end_configurations"]
    assert "Parker EO2 or CAST B4" in end_options_schema["description"]
    content = responses.call["input"][0]["content"]
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/png;base64,")


def test_drawing_settings_inherit_the_primary_llm_runtime_configuration() -> None:
    """Keep drawing model, timeout, and capacity aligned with the ECN LLM runtime."""
    settings = load_drawing_feasibility_settings(
        {"DRAWING_MAX_UPLOAD_BYTES": "1024"},
        model_name="shared-multimodal-model",
        request_timeout_seconds=91,
        max_concurrency=3,
    )

    assert settings.model_name == "shared-multimodal-model"
    assert settings.request_timeout_seconds == 91
    assert settings.max_concurrency == 3
    assert settings.max_upload_bytes == 1024
    assert "enabled" not in DrawingFeasibilitySettings.model_fields


def test_drawing_metadata_rejects_mismatched_extension() -> None:
    """Do not trust a client-declared drawing media type."""
    with pytest.raises(DrawingValidationError) as error:
        validate_drawing_metadata("drawing.pdf", "image/png")
    assert error.value.code == "DRAWING_FILE_TYPE_UNSUPPORTED"



class _DrawingServiceFake:
    """Return a static safe report and retain only call metadata for route assertions."""

    def __init__(self) -> None:
        """Use a minimal immutable settings object for bounded reader tests."""
        self.settings = SimpleNamespace(max_upload_bytes=100)
        self.drawing = None

    async def assess(self, drawing, **_kwargs) -> DrawingFeasibilityReport:
        """Return a browser-safe report without external work."""
        self.drawing = drawing
        return DrawingFeasibilityReport(
            correlation_id="test-correlation",
            source={"file_name": drawing.file_name, "media_type": drawing.media_type, "byte_size": len(drawing.content)},
            overall_result="Inconclusive",
            candidate_reports=[{
                "id": "candidate-1", "label": "Primary configuration", "completeness": "partial",
                "result": "Inconclusive",
                "criteria": [{"id": "tube_length", "title": "Tube length", "status": "Cannot assess", "explanation": "Synthetic report"}],
            }],
            scope_notice="Synthetic test report.",
        )


def test_drawing_route_returns_transient_safe_report(monkeypatch) -> None:
    """Expose a protected multipart endpoint independent from ECN ingestion."""
    monkeypatch.setenv("API_KEY", "drawing-test-key")
    drawing_service = _DrawingServiceFake()

    @asynccontextmanager
    async def lifespan(_app):
        """Supply only the standalone service needed by this transport test."""
        yield SimpleNamespace(drawing_feasibility_service=drawing_service)

    app = create_app(service_lifespan_factory=lifespan)
    with TestClient(app) as client:
        response = client.post(
            "/api/drawing-feasibility/assessments",
            headers={"X-API-Key": "drawing-test-key"},
            files={"drawing_file": ("drawing.png", b"\x89PNG\r\n\x1a\nsynthetic", "image/png")},
        )

    assert response.status_code == 200
    assert response.json()["source"] == {
        "file_name": "drawing.png", "media_type": "image/png", "byte_size": 17,
    }
    assert "content" not in response.text
    assert drawing_service.drawing.content == b"\x89PNG\r\n\x1a\nsynthetic"


def test_drawing_route_rejects_extra_multipart_fields(monkeypatch) -> None:
    """Keep the transport contract to exactly one transient drawing file."""
    monkeypatch.setenv("API_KEY", "drawing-test-key")
    drawing_service = _DrawingServiceFake()

    @asynccontextmanager
    async def lifespan(_app):
        """Supply a service that must not be reached for invalid multipart input."""
        yield SimpleNamespace(drawing_feasibility_service=drawing_service)

    app = create_app(service_lifespan_factory=lifespan)
    with TestClient(app) as client:
        response = client.post(
            "/api/drawing-feasibility/assessments",
            headers={"X-API-Key": "drawing-test-key"},
            data={"unexpected": "metadata"},
            files={"drawing_file": ("drawing.png", b"\x89PNG\r\n\x1a\nsynthetic", "image/png")},
        )

    assert response.status_code == 400
    assert response.json()["code"] == "DRAWING_FIELDS_INVALID"
    assert drawing_service.drawing is None


def test_drawing_route_rejects_invalid_signature(monkeypatch) -> None:
    """Reject deceptive upload content before it reaches the assessment service."""
    monkeypatch.setenv("API_KEY", "drawing-test-key")
    drawing_service = _DrawingServiceFake()

    @asynccontextmanager
    async def lifespan(_app):
        """Supply the standalone service for the validation-only request."""
        yield SimpleNamespace(drawing_feasibility_service=drawing_service)

    app = create_app(service_lifespan_factory=lifespan)
    with TestClient(app) as client:
        response = client.post(
            "/api/drawing-feasibility/assessments",
            headers={"X-API-Key": "drawing-test-key"},
            files={"drawing_file": ("drawing.pdf", b"not a pdf", "application/pdf")},
        )

    assert response.status_code == 400
    assert response.json()["code"] == "DRAWING_CONTENT_INVALID"
    assert drawing_service.drawing is None
