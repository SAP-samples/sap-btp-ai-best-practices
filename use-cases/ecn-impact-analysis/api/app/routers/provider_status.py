"""Safe cached and explicitly refreshed SAP AI Core provider status routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.dependencies import ServiceContainer, get_services
from app.ecn.errors import StartupConfigurationError
from app.ecn.llm.preflight import ProviderStatusSnapshot
from app.ecn.services.blocking_io import run_blocking


router = APIRouter()


@router.get("/provider-status", response_model=ProviderStatusSnapshot)
async def get_provider_status(
    services: Annotated[ServiceContainer, Depends(get_services)],
) -> ProviderStatusSnapshot:
    """Return the last safe preflight snapshot without contacting AI Core."""
    if services.provider_status_service is None:
        raise StartupConfigurationError(
            "PROVIDER_STATUS_UNAVAILABLE",
            "Provider diagnostics are unavailable",
            status_code=503,
        )
    return services.provider_status_service.get_status()


@router.post("/provider-status/check", response_model=ProviderStatusSnapshot)
async def check_provider_status(
    services: Annotated[ServiceContainer, Depends(get_services)],
) -> ProviderStatusSnapshot:
    """Run OAuth and exact deployment discovery outside the event loop."""
    if services.provider_status_service is None:
        raise StartupConfigurationError(
            "PROVIDER_STATUS_UNAVAILABLE",
            "Provider diagnostics are unavailable",
            status_code=503,
        )
    snapshot = await run_blocking(services.provider_status_service.check)
    if services.provider_status_service.generation_blocker() is None:
        queued = await run_blocking(
            services.analysis_repository.queue_provider_blocked_jobs
        )
        services.runtime.schedule_jobs(queued)
    return snapshot
