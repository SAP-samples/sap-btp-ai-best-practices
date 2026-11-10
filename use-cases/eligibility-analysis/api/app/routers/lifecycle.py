"""Authorized endpoints managing the HANA lifecycle history used as RPT-1 lifetime context.

Datasets are immutable versions of observed invoice lifetimes. One dataset is marked
active; every new lifetime preparation reads its rows from HANA. No local workbook is
read at runtime.
"""

from functools import lru_cache

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile

from ..models.workspace import ActivateLifecycleDataset
from ..security import get_api_key
from ..services.lifecycle.importer import import_workbook_bytes
from ..services.lifecycle.store import LifecycleStore
from .workspace import domain_call, get_workspace_service

router = APIRouter(prefix="/workspace/lifecycle", tags=["lifecycle"], dependencies=[Depends(get_api_key)])

# Upper bound for one uploaded history workbook; the reference dataset is a few MB.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@lru_cache(maxsize=1)
def get_lifecycle_store():
    """Return the HANA lifecycle store sharing the workspace backend (503 when HANA is unavailable)."""
    return LifecycleStore(get_workspace_service().runs.backend)


def _read_upload(file: UploadFile) -> bytes:
    """Read an uploaded workbook, rejecting empty or oversized files as validation errors."""
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise ValueError("The uploaded history workbook is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError(f"The uploaded history workbook exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    return content


def _with_active(store, metadata):
    """Attach the active flag to one dataset's metadata."""
    try:
        active = store.active_dataset()["dataset_id"]
    except LookupError:
        active = None
    return {**metadata, "active": metadata["dataset_id"] == active}


@router.post("/datasets")
def upload_dataset(file: UploadFile = File(...), dataset_id: str = Form(...),
                   fixed_reference: bool = Form(True), activate: bool = Form(False),
                   store=Depends(get_lifecycle_store)):
    """Import one history workbook as an immutable dataset and optionally activate it.

    Re-uploading identical bytes under the same dataset_id returns the saved metadata;
    different content under an existing dataset_id returns 409.
    """
    def run():
        content = _read_upload(file)
        purpose = "fixed_reference" if fixed_reference else "chronological_history"
        metadata = import_workbook_bytes(store, content, file.filename or "history.xlsx", dataset_id,
                                         reference_purpose=purpose)
        if activate:
            store.activate(dataset_id)
        return _with_active(store, metadata)
    return domain_call(run)


@router.get("/datasets")
def list_datasets(store=Depends(get_lifecycle_store)):
    """List saved datasets (newest first) with row counts, hashes and the active flag."""
    return {"items": domain_call(store.list_datasets)}


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str, store=Depends(get_lifecycle_store)):
    """Return one dataset's metadata: counts, exclusions, coverage, date range and provenance."""
    return _with_active(store, domain_call(store.get_dataset, dataset_id))


@router.get("/datasets/{dataset_id}/rows")
def get_dataset_rows(dataset_id: str, limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0),
                     store=Depends(get_lifecycle_store)):
    """Page through the normalized lifetime rows stored in HANA for one dataset."""
    return domain_call(store.rows, dataset_id, limit, offset)


@router.get("/active")
def get_active_dataset(store=Depends(get_lifecycle_store)):
    """Return the dataset future preparations use, or 404 when none is active."""
    return _with_active(store, domain_call(store.active_dataset))


@router.put("/active")
def activate_dataset(request: ActivateLifecycleDataset, store=Depends(get_lifecycle_store)):
    """Switch the active dataset; the next lifetime preparation reads the new version."""
    domain_call(store.activate, request.dataset_id)
    return _with_active(store, store.get_dataset(request.dataset_id))
