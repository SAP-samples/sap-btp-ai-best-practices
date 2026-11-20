"""Regression tests for automatic reuse of the existing drawing assessment."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.ecn.services.automated_manufacturing_feasibility import (
    AutomatedManufacturingFeasibilityRuntime,
    FixtureDrawingResolver,
)


class RepositoryFake:
    """Keep only controlled assessment state written by the automatic runtime."""

    def __init__(self) -> None:
        self.created = []
        self.running = []
        self.completed = []
        self.failed = []

    def create_queued(self, record):
        self.created.append(record)
        return True

    def mark_running(self, assessment_id, _updated_at):
        self.running.append(assessment_id)
        return True

    def complete(self, assessment_id, report, _updated_at):
        self.completed.append((assessment_id, report))

    def fail(self, assessment_id, error_code, _updated_at):
        self.failed.append((assessment_id, error_code))


class DrawingServiceFake:
    """Record the automatic input while returning the same report-shaped payload."""

    def __init__(self) -> None:
        self.drawings = []

    async def assess(self, drawing, *, user_id=None, client_host=None):
        self.drawings.append((drawing, user_id, client_host))
        return SimpleNamespace(model_dump=lambda *, mode: {"overall_result": "Inconclusive"})


def test_packaged_fixture_resolver_includes_each_allowlisted_pdf() -> None:
    """The API deployment artifact contains the exact PDFs used by automatic assessment."""
    resolver = FixtureDrawingResolver()

    for material_number in ("QKX5632660", "QKX5525860", "QKX3896510", "QKX0971490"):
        resolved = resolver.resolve(material_number)
        assert resolved is not None
        fixture, content = resolved
        assert fixture.material_number == material_number
        assert content.startswith(b"%PDF")


@pytest.mark.asyncio
async def test_matching_fixture_runs_the_existing_drawing_service_and_persists_only_report(
    tmp_path,
) -> None:
    """A mapped material uses its exact PDF fixture with no manual upload path."""
    fixture_path = tmp_path / "QKX5525860_2_AFS.pdf"
    fixture_path.write_bytes(b"%PDF-1.7 fixture")
    drawing_service = DrawingServiceFake()
    repository = RepositoryFake()
    runtime = AutomatedManufacturingFeasibilityRuntime(
        drawing_service,
        repository,
        FixtureDrawingResolver(tmp_path),
    )

    runtime.schedule(("ecn-1",), {"ecn-1": ("QKX5525860",)})
    await runtime.wait_until_idle()

    assert len(drawing_service.drawings) == 1
    drawing, user_id, client_host = drawing_service.drawings[0]
    assert drawing.file_name == "QKX5525860_2_AFS.pdf"
    assert drawing.content == b"%PDF-1.7 fixture"
    assert user_id is None
    assert client_host == "automated-fixture"
    assert repository.created[0].material_number == "QKX5525860"
    assert repository.completed[0][1] == {"overall_result": "Inconclusive"}
    assert repository.failed == []


@pytest.mark.asyncio
async def test_unmatched_material_neither_reads_a_drawing_nor_creates_an_assessment(tmp_path) -> None:
    """No filename guessing means unrelated ECN materials never invoke Gen AI Hub."""
    drawing_service = DrawingServiceFake()
    repository = RepositoryFake()
    runtime = AutomatedManufacturingFeasibilityRuntime(
        drawing_service,
        repository,
        FixtureDrawingResolver(tmp_path),
    )

    runtime.schedule(("ecn-1",), {"ecn-1": ("UNMAPPED-123",)})
    await runtime.wait_until_idle()

    assert drawing_service.drawings == []
    assert repository.created == []
