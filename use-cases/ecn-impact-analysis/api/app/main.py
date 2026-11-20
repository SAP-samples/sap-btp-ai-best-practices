"""FastAPI application factory and production ECN service lifespan."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

# Load the documented local API credentials before the module-level application
# can construct a HANA engine or SAP AI Core client. Existing process variables
# still take precedence, as required for Cloud Foundry bindings and test setup.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.engine import Engine

from app.dependencies import ServiceContainer, ServiceLifespanFactory
from app.ecn.errors import EcnImpactError
from app.ecn.assistant.discovery import ImpactDiscoveryService
from app.ecn.llm.preflight import AiCorePreflightConfig, ProviderStatusService
from app.ecn.llm.provider import EcnAnalysisProvider, create_impact_provider
from app.ecn.reference_data.service import ReferenceDatasetService
from app.ecn.rdf.service import RdfProjectionService
from app.ecn.services.automated_manufacturing_feasibility import (
    AutomatedManufacturingFeasibilityRuntime,
)
from app.ecn.services.blocking_io import run_blocking
from app.ecn.services.generation import AnalysisGenerationService, SystemClock
from app.ecn.services.history import HistoryService
from app.ecn.services.ingestion import IngestionService
from app.ecn.services.lifecycle import HistoryMutationGuard, IngestionPreparationGate
from app.ecn.services.retry import RetryService
from app.ecn.services.runtime import GenerationRuntime
from app.ecn.services.snapshots import SnapshotService
from app.ecn.settings import EcnSettings, load_ecn_settings
from app.drawing_feasibility.service import DrawingFeasibilityService
from app.drawing_feasibility.settings import load_drawing_feasibility_settings
from app.models.common import HealthResponse
from app.observability import emit_llm_usage_event
from app.persistence.analysis_repository import AnalysisRepository
from app.persistence.artifact_repository import ArtifactRepository
from app.persistence.ecn_repository import EcnRepository
from app.persistence.hana import create_hana_engine
from app.persistence.history_repository import HistoryRepository
from app.persistence.manufacturing_feasibility_repository import (
    ManufacturingFeasibilityRepository,
)
from app.persistence.plant_settings_repository import PlantSettingsRepository
from app.persistence.reference_dataset_repository import ReferenceDatasetRepository
from app.persistence.rdf_repository import RdfRepository
from app.persistence.schema import ensure_schema
from app.persistence.table_contracts import PersistenceTableNames
from app.persistence.upload_repository import UploadRepository
from app.persistence.workspace_repository import WorkspaceRepository
from app.routers import (
    analysis_progress,
    assistant,
    drawing_feasibility,
    ecns,
    history,
    plant_settings,
    provider_status,
    reference_datasets,
    rdf_projections,
    snapshots,
    uploads,
    workspace_actions,
)
from app.security import get_api_key

logger = logging.getLogger(__name__)


@asynccontextmanager
async def production_service_lifespan(
    app: FastAPI,
    engine_factory: Callable[[], Engine],
    provider_factory: Callable[[EcnSettings], EcnAnalysisProvider],
) -> AsyncIterator[ServiceContainer]:
    """Initialize services, recover interrupted work, and safely tear down.

    Args:
        app: FastAPI application being initialized.
        engine_factory: HANA engine constructor.
        provider_factory: Long-lived asynchronous provider constructor.

    Yields:
        Complete application-scoped service container.
    """
    del app
    settings = load_ecn_settings()
    drawing_settings = load_drawing_feasibility_settings(
        model_name=settings.llm_model_name,
        request_timeout_seconds=settings.llm_request_timeout_seconds,
        max_concurrency=settings.llm_max_concurrency,
    )
    engine = engine_factory()
    provider: EcnAnalysisProvider | None = None
    runtime: GenerationRuntime | None = None
    assistant_service = None
    drawing_feasibility_service: DrawingFeasibilityService | None = None
    manufacturing_feasibility_runtime: AutomatedManufacturingFeasibilityRuntime | None = None
    snapshot_service = None
    rdf_projection_service: RdfProjectionService | None = None
    rdf_reconciliation_task: asyncio.Task[object] | None = None
    try:
        drawing_feasibility_service = DrawingFeasibilityService(drawing_settings)
        await run_blocking(ensure_schema, engine)
        upload_repository = UploadRepository(engine)
        artifact_repository = ArtifactRepository(engine)
        ecn_repository = EcnRepository(engine)
        analysis_repository = AnalysisRepository(engine)
        history_repository = HistoryRepository(engine)
        reference_dataset_repository = ReferenceDatasetRepository(engine)
        plant_settings_repository = PlantSettingsRepository(engine)
        manufacturing_feasibility_repository = ManufacturingFeasibilityRepository(engine)
        rdf_repository = RdfRepository(engine)
        preparation_gate = IngestionPreparationGate()
        mutation_guard = HistoryMutationGuard()
        clock = SystemClock()
        rdf_projection_service = RdfProjectionService(rdf_repository, clock.now)
        await run_blocking(rdf_repository.ensure_capabilities)
        rdf_reconciliation_task = asyncio.create_task(
            run_blocking(rdf_projection_service.reconcile)
        )
        provider = provider_factory(settings)
        provider_status_service = ProviderStatusService(
            AiCorePreflightConfig.from_environ()
        )
        await run_blocking(provider_status_service.check)
        ingestion_service = IngestionService(
            upload_repository=upload_repository,
            artifact_repository=artifact_repository,
            settings=settings,
            preparation_gate=preparation_gate,
            mutation_guard=mutation_guard,
            clock=lambda: datetime.now(timezone.utc),
            uuid_factory=lambda: str(uuid4()),
            reference_repository=reference_dataset_repository,
            plant_settings_repository=plant_settings_repository,
        )
        reference_dataset_service = ReferenceDatasetService(
            reference_dataset_repository,
            settings,
            clock=lambda: datetime.now(timezone.utc),
            uuid_factory=lambda: str(uuid4()),
            mutation_guard=mutation_guard,
        )
        history_service = HistoryService(
            history_repository,
            mutation_guard,
            rdf_projection_service.clear_all,
            rdf_projection_service.delete_ecns,
        )
        retry_service = RetryService(analysis_repository, mutation_guard)
        snapshot_service = SnapshotService(
            ecn_repository,
            artifact_repository,
            reference_dataset_repository,
            engine,
            PersistenceTableNames(),
            manufacturing_feasibility_repository,
        )
        generation_service = AnalysisGenerationService(
            analysis_repository,
            provider,
            emit_llm_usage_event,
            clock,
            uuid4,
            snapshot_service=snapshot_service,
            discovery_service=ImpactDiscoveryService(
                rdf_projection_service,
                settings.llm_model_name,
            ),
            rdf_projection_service=rdf_projection_service,
        )
        manufacturing_feasibility_runtime = AutomatedManufacturingFeasibilityRuntime(
            drawing_feasibility_service,
            manufacturing_feasibility_repository,
        )
        runtime = GenerationRuntime(
            generation_service,
            analysis_repository,
            mutation_guard,
            min(settings.llm_max_concurrency, 2),
            clock,
            uuid4,
            provider_status_service=provider_status_service,
            provider_max_concurrency=settings.llm_max_concurrency,
        )
        from app.ecn.assistant.runtime import AgentRuntime

        workspace_repository = WorkspaceRepository(engine, guard=mutation_guard)
        assistant_service = AgentRuntime(
            snapshot_service,
            ecn_repository,
            rdf_projection_service,
            settings.llm_model_name,
            runtime.provider_capacity,
            workspace=workspace_repository,
        )
        await run_blocking(analysis_repository.fail_interrupted_jobs, clock.now())
        yield ServiceContainer(
            settings=settings,
            ingestion_service=ingestion_service,
            artifact_repository=artifact_repository,
            workspace_repository=workspace_repository,
            assistant_service=assistant_service,
            snapshot_service=snapshot_service,
            history_service=history_service,
            retry_service=retry_service,
            ecn_repository=ecn_repository,
            analysis_repository=analysis_repository,
            runtime=runtime,
            engine=engine,
            provider_status_service=provider_status_service,
            reference_dataset_service=reference_dataset_service,
            reference_dataset_repository=reference_dataset_repository,
            rdf_projection_service=rdf_projection_service,
            plant_settings_repository=plant_settings_repository,
            drawing_feasibility_service=drawing_feasibility_service,
            manufacturing_feasibility_runtime=manufacturing_feasibility_runtime,
        )
    finally:
        try:
            try:
                if rdf_reconciliation_task is not None:
                    rdf_reconciliation_task.cancel()
                    try:
                        await rdf_reconciliation_task
                    except asyncio.CancelledError:
                        pass
                    except Exception:  # noqa: BLE001 - retry remains durable for the next startup.
                        logger.exception("RDF projection startup reconciliation failed")
                if assistant_service is not None:
                    await assistant_service.close()
                if manufacturing_feasibility_runtime is not None:
                    await manufacturing_feasibility_runtime.close()
                if drawing_feasibility_service is not None:
                    await drawing_feasibility_service.close()
            finally:
                if runtime is not None:
                    await runtime.close()
                elif provider is not None:
                    await provider.close()
        finally:
            if snapshot_service is not None:
                snapshot_service.clear()
            await run_blocking(engine.dispose)


def _error_payload(
    code: str, message: str, details: object = None
) -> dict[str, object]:
    """Build the sole public error envelope shape."""
    return {"code": code, "message": message, "details": details}


def _configure_error_handlers(application: FastAPI) -> None:
    """Register domain, framework, authentication, and safe fallback handlers."""

    @application.exception_handler(EcnImpactError)
    async def handle_domain_error(
        _request: Request, exc: EcnImpactError
    ) -> JSONResponse:
        """Return the safe fields carried by a typed domain error."""
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(exc.code, exc.public_message, exc.details),
        )

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Describe invalid locations/types without echoing submitted values."""
        errors = [
            {"location": list(error.get("loc", ())), "type": error.get("type")}
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content=_error_payload(
                "REQUEST_VALIDATION_ERROR",
                "The request did not match the API contract",
                {"errors": errors},
            ),
        )

    @application.exception_handler(HTTPException)
    async def handle_http_error(_request: Request, exc: HTTPException) -> JSONResponse:
        """Normalize FastAPI HTTP errors into the stable public envelope."""
        code = "API_KEY_INVALID" if exc.status_code == 403 else "HTTP_ERROR"
        message = exc.detail if isinstance(exc.detail, str) else "The request failed"
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(code, message),
            headers=exc.headers,
        )

    @application.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        """Log an unexpected failure and expose no exception content."""
        logger.exception("Unhandled API error on %s", request.url.path, exc_info=exc)
        return JSONResponse(
            status_code=500,
            content=_error_payload(
                "INTERNAL_SERVER_ERROR", "An unexpected server error occurred"
            ),
        )


def create_app(
    engine_factory: Callable[[], Engine] = create_hana_engine,
    provider_factory: Callable[
        [EcnSettings], EcnAnalysisProvider
    ] = create_impact_provider,
    service_lifespan_factory: ServiceLifespanFactory | None = None,
) -> FastAPI:
    """Build the API with production factories or one complete test lifespan seam.

    Args:
        engine_factory: Production HANA engine constructor.
        provider_factory: Production native Responses provider constructor.
        service_lifespan_factory: Optional complete test-owned service context.

    Returns:
        Fully routed FastAPI application.
    """
    selected_factory = service_lifespan_factory

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        """Expose the selected container only while its context is active."""
        context = (
            selected_factory(application)
            if selected_factory is not None
            else production_service_lifespan(
                application, engine_factory, provider_factory
            )
        )
        async with context as services:
            application.state.services = services
            yield

    application = FastAPI(
        title="ECN Impact Analysis API",
        description="Evidence-grounded ECN impact ingestion and analysis API",
        version="1.0.0",
        lifespan=lifespan,
    )
    origins: list[str] = []
    production_origin = os.getenv("ALLOWED_ORIGIN")
    if production_origin:
        origins.append(production_origin)
    if os.getenv("APP_ENV") != "production":
        origins.extend(
            [
                "http://localhost:4173",
                *(f"http://localhost:{port}" for port in range(5173, 5178)),
                *(f"http://127.0.0.1:{port}" for port in range(5173, 5178)),
            ]
        )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    _configure_error_handlers(application)

    @application.get("/api/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Return public process liveness without HANA or provider calls."""
        return HealthResponse.healthy("api")

    protected = [Depends(get_api_key)]
    application.include_router(assistant.router, prefix="/api", dependencies=protected)
    application.include_router(
        drawing_feasibility.router, prefix="/api", dependencies=protected
    )
    application.include_router(
        workspace_actions.router, prefix="/api", dependencies=protected
    )
    application.include_router(snapshots.router, prefix="/api", dependencies=protected)
    application.include_router(uploads.router, prefix="/api", dependencies=protected)
    application.include_router(ecns.router, prefix="/api", dependencies=protected)
    application.include_router(
        rdf_projections.router, prefix="/api", dependencies=protected
    )
    application.include_router(
        analysis_progress.router, prefix="/api", dependencies=protected
    )
    application.include_router(history.router, prefix="/api", dependencies=protected)
    application.include_router(
        provider_status.router, prefix="/api", dependencies=protected
    )
    application.include_router(
        reference_datasets.router, prefix="/api", dependencies=protected
    )
    application.include_router(plant_settings.router, prefix="/api", dependencies=protected)
    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        log_level="info",
    )
