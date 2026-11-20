"""Typed application-scoped dependency container for ECN HTTP handlers."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import TypeAlias

from fastapi import FastAPI, Request
from sqlalchemy.engine import Engine

from app.ecn.llm.preflight import ProviderStatusService
from app.ecn.reference_data.service import ReferenceDatasetService
from app.ecn.rdf.service import RdfProjectionService
from app.ecn.services.automated_manufacturing_feasibility import (
    AutomatedManufacturingFeasibilityRuntime,
)
from app.ecn.services.history import HistoryService
from app.ecn.services.ingestion import IngestionService
from app.ecn.services.retry import RetryService
from app.ecn.services.runtime import GenerationRuntime
from app.ecn.services.snapshots import SnapshotService
from app.ecn.settings import EcnSettings
from app.drawing_feasibility.service import DrawingFeasibilityService
from app.persistence.analysis_repository import AnalysisRepository
from app.persistence.artifact_repository import ArtifactRepository
from app.persistence.ecn_repository import EcnRepository
from app.persistence.manufacturing_feasibility_repository import (
    ManufacturingFeasibilityRepository,
)
from app.persistence.plant_settings_repository import PlantSettingsRepository
from app.persistence.reference_dataset_repository import ReferenceDatasetRepository
from app.persistence.workspace_repository import WorkspaceRepository


@dataclass(frozen=True)
class ServiceContainer:
    """Hold every initialized dependency shared by the ECN API routes."""

    settings: EcnSettings
    ingestion_service: IngestionService
    history_service: HistoryService
    retry_service: RetryService
    ecn_repository: EcnRepository
    analysis_repository: AnalysisRepository
    runtime: GenerationRuntime
    engine: Engine
    assistant_service: object | None = None
    workspace_repository: WorkspaceRepository | None = None
    snapshot_service: SnapshotService | None = None
    artifact_repository: ArtifactRepository | None = None
    provider_status_service: ProviderStatusService | None = None
    reference_dataset_service: ReferenceDatasetService | None = None
    reference_dataset_repository: ReferenceDatasetRepository | None = None
    rdf_projection_service: RdfProjectionService | None = None
    plant_settings_repository: PlantSettingsRepository | None = None
    drawing_feasibility_service: DrawingFeasibilityService | None = None
    manufacturing_feasibility_runtime: AutomatedManufacturingFeasibilityRuntime | None = None


ServiceLifespanFactory: TypeAlias = Callable[
    [FastAPI], AbstractAsyncContextManager[ServiceContainer]
]


def get_services(request: Request) -> ServiceContainer:
    """Return the initialized application service container for one request.

    Args:
        request: Active FastAPI request whose app lifespan has started.

    Returns:
        Fully initialized immutable service container.
    """
    return request.app.state.services
