from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.models.nbo import BatchRunResponse
from app.security import get_api_key
from app.services.batch import batch_service


router = APIRouter(dependencies=[Depends(get_api_key)])


@router.post("/runs", response_model=BatchRunResponse, status_code=202)
async def run_batch() -> BatchRunResponse:
    return batch_service.run_batch()


@router.get("/runs/{run_id}", response_model=BatchRunResponse)
async def get_batch_run(run_id: str) -> BatchRunResponse:
    try:
        return batch_service.get_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown batch run: {run_id}") from exc


@router.get("/runs/{run_id}/artifacts/{artifact_name}")
async def download_batch_artifact(run_id: str, artifact_name: str):
    try:
        artifact_path = batch_service.get_artifact_path(run_id, artifact_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown artifact: {artifact_name}") from exc

    return FileResponse(path=artifact_path, filename=artifact_path.name)
