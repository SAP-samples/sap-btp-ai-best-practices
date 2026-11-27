"""Decision-scenario HTTP interface: the 7 key frontier points and the selection.

Endpoints:
    GET /api/runs/{run_id}/decision-scenarios?neighbours=3
    PUT /api/runs/{run_id}/decision-selection
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import Field

from app.routers.workspace import call
from app.workspace.dependencies import get_service
from app.workspace.models import StrictModel
from production_wheel.decision_scenarios import DEFAULT_NEIGHBOURS

router = APIRouter(prefix="/api")


class DecisionSelection(StrictModel):
    """One chosen frontier point; its subgroups already carry their cycle stock strategy."""

    point_index: int = Field(ge=1)
    neighbours: int = Field(default=DEFAULT_NEIGHBOURS, ge=1, le=10)


@router.get("/runs/{run_id}/decision-scenarios")
def get_decision_scenarios(
    run_id: str,
    neighbours: int = Query(DEFAULT_NEIGHBOURS, ge=1, le=10),
    service=Depends(get_service),
):
    """Return every usable point, labelled when it is the knee or one of its neighbours."""
    return call(service.decision_scenarios, run_id, neighbours)


@router.put("/runs/{run_id}/decision-selection")
def put_decision_selection(
    run_id: str, body: DecisionSelection, service=Depends(get_service)
):
    """Persist the chosen scenario on the run for SAP settings generation."""
    return call(
        service.select_decision,
        run_id,
        body.point_index,
        body.neighbours,
    )
