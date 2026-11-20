"""Authenticated Settings endpoints for the Plant Code Mapping.

``GET /api/plant-settings`` returns the brand suffix to plant rows, the rule
lifecycle states excluded from plant assignment and the states seen in rule
uploads. ``PUT /api/plant-settings`` replaces both lists atomically. New values
apply to ECNs imported afterwards; existing assessments keep the values frozen
at their import.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.dependencies import ServiceContainer, get_services
from app.ecn.errors import PersistenceStateError
from app.ecn.plant_settings import PlantSettings, normalize_plant_settings
from app.ecn.services.blocking_io import run_blocking

router = APIRouter()


class PlantMappingItem(BaseModel):
    """One brand suffix to plant row exchanged with the Settings page."""

    suffix: str = Field(max_length=20)
    plant_code: str = Field(max_length=40)
    plant_name: str = Field(max_length=200)


class PlantSettingsRequest(BaseModel):
    """Complete replacement of the plant-assignment settings."""

    mappings: list[PlantMappingItem] = Field(default_factory=list, max_length=200)
    excluded_rule_states: list[str] = Field(default_factory=list, max_length=50)


class PlantSettingsResponse(PlantSettingsRequest):
    """Saved settings plus the rule lifecycle states offered as choices."""

    available_rule_states: list[str]


def _response(settings: PlantSettings) -> PlantSettingsResponse:
    """Convert domain settings to the HTTP response model."""
    return PlantSettingsResponse(
        mappings=[PlantMappingItem(**mapping.as_json()) for mapping in settings.mappings],
        excluded_rule_states=list(settings.excluded_rule_states),
        available_rule_states=list(settings.available_rule_states),
    )


def _repository(services: ServiceContainer):
    """Return the plant settings repository or a 503 when a test container omits it."""
    repository = getattr(services, "plant_settings_repository", None)
    if repository is None:
        raise PersistenceStateError(
            "PLANT_SETTINGS_UNAVAILABLE", "Plant settings are unavailable.", status_code=503
        )
    return repository


@router.get("/plant-settings", response_model=PlantSettingsResponse)
async def get_plant_settings(
    services: Annotated[ServiceContainer, Depends(get_services)],
) -> PlantSettingsResponse:
    """Return the current Plant Code Mapping and rule-state filter."""
    return _response(await run_blocking(_repository(services).get))


@router.put("/plant-settings", response_model=PlantSettingsResponse)
async def replace_plant_settings(
    body: PlantSettingsRequest,
    services: Annotated[ServiceContainer, Depends(get_services)],
) -> PlantSettingsResponse:
    """Validate and atomically replace the Plant Code Mapping and rule-state filter."""
    repository = _repository(services)
    settings = normalize_plant_settings(
        [item.model_dump() for item in body.mappings], body.excluded_rule_states
    )
    return _response(await run_blocking(repository.replace, settings))
