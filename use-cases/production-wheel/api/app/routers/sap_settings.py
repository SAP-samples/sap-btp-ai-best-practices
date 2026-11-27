"""SAP settings HTTP interface.

Endpoints:
    GET  /api/sap-policy                       former policy rows (read-only) and the settings catalogue
    PUT  /api/sap-policy                       410: the rules now live in the plant profile
    POST /api/runs/{run_id}/sap-settings       generate for the selected scenario {"period_label": "26Q4"}
    GET  /api/runs/{run_id}/sap-settings       stored generation summary
    GET  /api/runs/{run_id}/sap-settings/readiness?point_index=  what generation would produce and what is missing
    GET  /api/runs/{run_id}/sap-settings/export  XLSX laid out like the SAP upload template
    GET  /api/plant-profiles/{profile_id}/sap-settings/suggestions?dataset_id=  values learnt from the plant's datasets
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import Field

from app.routers.workspace import call
from app.workspace.dependencies import get_service
from app.workspace.models import StrictModel
from app.workspace.sap_settings_inputs import PERIOD_LABEL_PATTERN
from app.workspace.sap_settings_service import POLICY_MOVED

router = APIRouter(prefix="/api")


class PolicyRows(StrictModel):
    """Complete replacement set of plant policy rows."""

    rows: list[dict[str, Any]] = Field(default_factory=list, max_length=2000)


@router.get("/sap-policy")
def get_sap_policy(service=Depends(get_service)):
    """Return the rows left in the former policy store (read-only) and the settings catalogue."""
    return call(service.sap_policy)


@router.put("/sap-policy")
def put_sap_policy(body: PolicyRows):
    """Refuse policy row writes: SAP settings rules are edited in the plant profile."""
    raise HTTPException(status_code=410, detail=POLICY_MOVED)


class SapGeneration(StrictModel):
    """Planning period written into the APO comment, e.g. a quarter label."""

    period_label: str = Field(pattern=PERIOD_LABEL_PATTERN)


@router.post("/runs/{run_id}/sap-settings")
def post_sap_settings(run_id: str, body: SapGeneration, service=Depends(get_service)):
    """Generate and persist SAP settings for the run's selected decision scenario."""
    return call(service.generate_sap_settings, run_id, body.period_label)


@router.get("/runs/{run_id}/sap-settings")
def get_sap_settings(run_id: str, service=Depends(get_service)):
    """Return the stored SAP settings summary (null before generation)."""
    return {"sap_settings": call(service.sap_settings, run_id)}


@router.get("/runs/{run_id}/sap-settings/readiness")
def get_sap_settings_readiness(run_id: str, point_index: int | None = Query(default=None, ge=0),
                               service=Depends(get_service)):
    """Report what generation would produce for a point (default: the stored selection) and what is missing."""
    return call(service.sap_settings_readiness, run_id, point_index)


@router.get("/plant-profiles/{profile_id}/sap-settings/suggestions")
def get_sap_settings_suggestions(profile_id: str, dataset_id: str | None = None, service=Depends(get_service)):
    """Suggest SAP settings values (calendar lot-size profiles) learnt from the plant's datasets."""
    return call(service.sap_settings_suggestions, profile_id, dataset_id)


@router.get("/runs/{run_id}/sap-settings/export")
def export_sap_settings_xlsx(run_id: str, service=Depends(get_service)):
    """Download the generated SAP settings as an XLSX laid out like the upload template."""
    from fastapi.responses import Response

    from app.workspace.sap_settings_export import export_sap_settings

    data = call(export_sap_settings, service, run_id)
    return Response(
        data,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="sap-settings-{run_id[:12]}.xlsx"'},
    )
