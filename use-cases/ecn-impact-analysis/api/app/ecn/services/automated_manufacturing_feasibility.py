"""Automatic reuse of the existing PS4 drawing assessment for local PoC fixtures."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.ecn.services.blocking_io import run_blocking
from app.drawing_feasibility.errors import DrawingAnalysisError
from app.drawing_feasibility.models import DrawingInput
from app.drawing_feasibility.service import DrawingFeasibilityService
from app.persistence.manufacturing_feasibility_repository import (
    ManufacturingFeasibilityRecord,
    ManufacturingFeasibilityRepository,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FixtureDrawing:
    """One explicit local drawing fixture that may be assessed automatically."""

    material_number: str
    file_name: str


_FIXTURE_DRAWINGS = (
    FixtureDrawing("QKX5632660", "QKX5632660_2_AFS.pdf"),
    FixtureDrawing("QKX5525860", "QKX5525860_2_AFS.pdf"),
    FixtureDrawing("QKX3896510", "QKX3896510_2_IW_old.pdf"),
    FixtureDrawing("QKX0971490", "QKX0971490_2_IW_old.pdf"),
)


class FixtureDrawingResolver:
    """Resolve only explicit local fixture drawings; never infer a filename or revision."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = root or Path(__file__).resolve().parents[1] / "fixtures" / "drawings"
        self._by_material = {
            fixture.material_number: fixture for fixture in _FIXTURE_DRAWINGS
        }

    def available(self, material_number: str) -> FixtureDrawing | None:
        """Return an exact configured drawing only while its packaged file exists."""
        if not material_number:
            return None
        fixture = self._by_material.get(material_number.strip().upper())
        if fixture is None:
            return None
        path = self._root / fixture.file_name
        if not path.is_file():
            logger.warning("Configured drawing fixture is unavailable: %s", fixture.file_name)
            return None
        return fixture

    def resolve(self, material_number: str) -> tuple[FixtureDrawing, bytes] | None:
        """Return source bytes only for an exact configured material mapping."""
        fixture = self.available(material_number)
        return (fixture, (self._root / fixture.file_name).read_bytes()) if fixture else None


class AutomatedManufacturingFeasibilityRuntime:
    """Run existing transient PS4 analysis for matching fixture drawings after ECN intake."""

    def __init__(
        self,
        drawing_service: DrawingFeasibilityService,
        repository: ManufacturingFeasibilityRepository,
        resolver: FixtureDrawingResolver | None = None,
        maximum_concurrency: int = 1,
    ) -> None:
        self._drawing_service = drawing_service
        self._repository = repository
        self._resolver = resolver or FixtureDrawingResolver()
        self._capacity = asyncio.Semaphore(maximum_concurrency)
        self._tasks: set[asyncio.Task[None]] = set()
        self._closing = False

    def schedule(self, ecn_row_ids: tuple[str, ...], material_numbers: dict[str, tuple[str, ...]]) -> None:
        """Schedule exact fixture assessments after ECN publication; unmatched parts are ignored."""
        for ecn_row_id in ecn_row_ids:
            for material_number in material_numbers.get(ecn_row_id, ()):
                self._track(
                    asyncio.create_task(self._prepare_and_run(ecn_row_id, material_number))
                )

    async def _prepare_and_run(self, ecn_row_id: str, material_number: str) -> None:
        """Read one allowlisted drawing after publication and enqueue its unchanged analysis."""
        resolved = await asyncio.to_thread(self._resolver.resolve, material_number)
        if resolved is None:
            return
        fixture, content = resolved
        now = datetime.now(timezone.utc)
        record = ManufacturingFeasibilityRecord(
            assessment_id=str(uuid4()),
            ecn_row_id=ecn_row_id,
            material_number=material_number,
            drawing_file_name=fixture.file_name,
            drawing_sha256=hashlib.sha256(content).hexdigest(),
            state="QUEUED",
            report=None,
            error_code=None,
            created_at=now,
            updated_at=now,
        )
        created = await run_blocking(self._repository.create_queued, record)
        if created:
            await self._run(record, content)

    async def _run(self, record: ManufacturingFeasibilityRecord, content: bytes) -> None:
        """Invoke the unchanged drawing service and persist only its validated report JSON."""
        async with self._capacity:
            now = datetime.now(timezone.utc)
            if not await run_blocking(
                self._repository.mark_running, record.assessment_id, now
            ):
                return
            try:
                report = await self._drawing_service.assess(
                    DrawingInput(
                        file_name=record.drawing_file_name,
                        media_type="application/pdf",
                        content=content,
                    ),
                    client_host="automated-fixture",
                )
            except DrawingAnalysisError as exc:
                await run_blocking(
                    self._repository.fail,
                    record.assessment_id,
                    exc.code,
                    datetime.now(timezone.utc),
                )
            except Exception:  # noqa: BLE001 - isolate an automatic auxiliary task.
                logger.exception("Automatic drawing feasibility assessment failed")
                await run_blocking(
                    self._repository.fail,
                    record.assessment_id,
                    "DRAWING_ANALYSIS_FAILED",
                    datetime.now(timezone.utc),
                )
            else:
                await run_blocking(
                    self._repository.complete,
                    record.assessment_id,
                    report.model_dump(mode="json"),
                    datetime.now(timezone.utc),
                )

    def _track(self, task: asyncio.Task[None]) -> None:
        """Keep the task alive and consume contained failures on completion."""
        if self._closing:
            task.cancel()
            raise RuntimeError("automated manufacturing runtime is closing")
        self._tasks.add(task)

        def discard(completed: asyncio.Task[None]) -> None:
            self._tasks.discard(completed)
            if not completed.cancelled():
                completed.exception()

        task.add_done_callback(discard)

    async def wait_until_idle(self) -> None:
        """Wait for scheduled automatic assessments; used only by controlled tests."""
        while self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)

    async def close(self) -> None:
        """Cancel automatic work during application shutdown without retaining fixtures."""
        self._closing = True
        while self._tasks:
            tasks = tuple(self._tasks)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
