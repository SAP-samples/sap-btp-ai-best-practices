"""Dataset, draft, run and semantic query HTTP interfaces for the UI5 workspace."""

from __future__ import annotations

import csv
import io
import json
import logging
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile

import openai
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import Field

from app.workspace.dependencies import get_service
from app.workspace.models import QuerySpec, RunSubmission, StrictModel

router = APIRouter(prefix="/api")


class DraftCreate(StrictModel):
    """Select one published dataset for a new manual/agent draft."""

    dataset_id: str
    plant_profile_id: str | None = None
    title: str = Field(default="", max_length=255)


class DraftPatch(StrictModel):
    """Patch an exact draft revision with typed request and budget subobjects."""

    revision: int
    dataset_id: str | None = None
    request: dict | None = None
    budget: dict | None = None
    title: str | None = Field(default=None, max_length=255)


class Selection(StrictModel):
    """Identify a persisted solution point without local file paths."""

    run_id: str
    point_index: int = Field(ge=1)


class Comparison(StrictModel):
    """Compare two explicit solution selections."""

    left: Selection
    right: Selection


class AIModelSelection(StrictModel):
    """Choose one supported deployment against the last loaded settings revision."""

    model: str
    revision: int = Field(ge=0)


@router.get("/ai-model-settings")
def get_ai_model_settings(service=Depends(get_service)):
    """Expose the effective global model and ordered supported choices."""
    return call(service.get_ai_model_settings)


@router.put("/ai-model-settings")
def save_ai_model_settings(body: AIModelSelection, service=Depends(get_service)):
    """Persist a supported model for subsequent application LLM calls."""
    return call(service.save_ai_model_settings, body.model, body.revision)


def call(operation: Callable, *args, **kwargs):
    """Translate known domain errors while retaining unexpected server diagnostics."""
    try:
        return operation(*args, **kwargs)
    except KeyError as exc:
        raise HTTPException(
            404, "Unknown dataset, run, draft, or artifact identifier"
        ) from exc
    except ValueError as exc:
        status = (
            409
            if any(
                word in str(exc)
                for word in ("revision", "idempotency", "not available")
            )
            else 422
        )
        raise HTTPException(status, str(exc)) from exc


@router.get("/datasets")
def list_datasets(
    request: Request,
    plant: str | None = None,
    status: str | None = None,
    name: str | None = None,
    created_from: str | None = None,
    created_to: str | None = None,
    include_removed: bool = False,
    service=Depends(get_service),
):
    """List datasets with metadata and inclusive created_from/created_to ISO bounds."""
    return {
        "items": call(
            service.list_datasets,
            {k: v for k, v in request.query_params.items() if k != "include_removed"},
            include_removed=include_removed,
        )
    }


@router.post("/datasets")
def upload_dataset(
    primary: UploadFile = File(...),
    enrichment: UploadFile | None = File(None),
    name: str = Form("Dataset"),
    settings: str = Form("{}"),
    service=Depends(get_service),
):
    """Extract uploaded source files deterministically and stage a review snapshot."""
    from production_wheel.extraction.datasets import extract_workbooks

    try:
        parameters = json.loads(settings)
    except ValueError as exc:
        raise HTTPException(422, "settings must be a JSON object") from exc
    if not isinstance(parameters, dict):
        raise HTTPException(422, "settings must be an object")
    sources = []
    with TemporaryDirectory(prefix="wheel-upload-") as directory:
        paths = []
        for index, upload in enumerate((primary, enrichment)):
            if upload is None:
                paths.append(None)
                continue
            content = upload.file.read(50 * 1024 * 1024 + 1)
            if len(content) > 50 * 1024 * 1024:
                raise HTTPException(413, "Each workbook must be under 50 MiB")
            try:
                with ZipFile(io.BytesIO(content)) as archive:
                    if (
                        sum(entry.file_size for entry in archive.infolist())
                        > 512 * 1024 * 1024
                    ):
                        raise HTTPException(413, "Expanded workbook exceeds 512 MiB")
            except BadZipFile as exc:
                raise HTTPException(422, "Upload must be an XLSX workbook") from exc
            filename = Path(upload.filename or f"source-{index}.xlsx").name
            path = Path(directory) / f"{index}.xlsx"
            path.write_bytes(content)
            paths.append(path)
            sources.append((filename, content))
        extracted = call(extract_workbooks, paths[0], paths[1], parameters)
    try:
        return call(service.register_dataset, name, extracted, sources)
    except HTTPException:
        raise
    except Exception as exc:
        # Return through FastAPI's handled-error path so CORS headers survive.
        # Raw database details stay in the server log, not in browser messages.
        logging.getLogger(__name__).exception("Dataset persistence failed")
        raise HTTPException(
            500,
            "The workbook was extracted, but saving the snapshot failed. "
            "No new snapshot was registered. Check the backend log before retrying.",
        ) from exc


@router.post("/datasets/settings/interpret")
async def interpret_dataset_settings(body: dict, service=Depends(get_service)):
    """Translate planner free text into validated extraction overrides; saves nothing."""
    from app.workspace.dataset_settings_interpretation import interpret_dataset_settings as run

    try:
        return await run(service, body.get("text", ""))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/datasets/{dataset_id}")
def inspect_dataset(dataset_id: str, service=Depends(get_service)):
    """Return publication state, quality findings and immutable provenance."""
    return call(service.inspect_dataset, dataset_id)


@router.post("/datasets/{dataset_id}/publish")
def publish_dataset(dataset_id: str, service=Depends(get_service)):
    """Publish a valid reviewed version; does not modify extracted data."""
    return call(service.publish_dataset, dataset_id)


@router.post("/run-drafts")
def create_draft(body: DraftCreate, service=Depends(get_service)):
    """Create defaults for a selected published dataset."""
    return call(service.create_draft, body.dataset_id, body.plant_profile_id, body.title)


@router.get("/run-drafts/{draft_id}")
def get_draft(draft_id: str, service=Depends(get_service)):
    """Read authoritative values and revision for dashboard or agent."""
    return call(service.get_draft, draft_id)


@router.patch("/run-drafts/{draft_id}")
def update_draft(draft_id: str, body: DraftPatch, service=Depends(get_service)):
    """Update only explicitly supplied draft fields, using optimistic concurrency."""
    return call(
        service.update_draft,
        draft_id,
        body.revision,
        body.model_dump(exclude_none=True, exclude={"revision"}),
    )


@router.post("/run-drafts/{draft_id}/validate")
def validate_draft(draft_id: str, service=Depends(get_service)):
    """Validate/compile constraints against the actual source snapshot."""
    return call(service.validate_draft, draft_id)


@router.post("/runs")
def launch_run(body: RunSubmission, service=Depends(get_service)):
    """Register an idempotent durable job and return its compact status."""
    return call(service.submit_summary, body)


@router.get("/runs")
def list_runs(
    request: Request,
    dataset_id: str | None = None,
    status: str | None = None,
    parent_run_id: str | None = None,
    created_from: str | None = None,
    created_to: str | None = None,
    include_removed: bool = False,
    service=Depends(get_service),
):
    """Discover jobs with metadata and inclusive created_from/created_to ISO bounds."""
    return {
        "items": call(
            service.list_run_summaries,
            {k: v for k, v in request.query_params.items() if k != "include_removed"},
            include_removed=include_removed,
        )
    }


@router.get("/runs/{run_id}")
def get_run(run_id: str, service=Depends(get_service)):
    """Read worker-owned lifecycle and result readiness from HANA."""
    return call(service.run_status, run_id)


@router.get("/runs/{run_id}/configuration")
def get_run_configuration(run_id: str, service=Depends(get_service)):
    """Read compact frozen configuration without replay-only request/matrix objects."""
    return call(service.run_status, run_id)["configuration"]


@router.get("/runs/{run_id}/failure-diagnostics")
def get_run_failure_diagnostics(run_id: str, service=Depends(get_service)):
    """Read bounded checkpoint diagnostics for a failed or incomplete run."""
    return call(service.run_failure_diagnostics, run_id)


@router.get("/runs/{run_id}/matrix")
def get_run_matrix(
    run_id: str,
    offset: int = 0,
    limit: int = 50,
    status: str | None = None,
    service=Depends(get_service),
):
    """Page the frozen compatibility evidence for one optimizer run."""
    return call(service.run_matrix_page, run_id, offset=offset, limit=limit, status=status)


@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str, service=Depends(get_service)):
    """Request queued/running job cancellation; completed results stay immutable."""
    return call(service.cancel_run, run_id)


@router.get("/runs/{run_id}/results")
def get_results(run_id: str, service=Depends(get_service)):
    """Return compact frontier data after atomic publication."""
    return call(service.run_results_view, run_id)


@router.post("/runs/{run_id}/retry-publication")
def retry_results(run_id: str, service=Depends(get_service)):
    """Retry a saved result checkpoint without rerunning the solver."""
    from app.workspace.worker import retry_publication

    return call(retry_publication, service, run_id)


@router.post("/query")
def query_data(body: QuerySpec, service=Depends(get_service)):
    """Execute a registered read-only analytical query."""
    return call(service.query, body)


@router.post("/compare")
def compare_results(body: Comparison, service=Depends(get_service)):
    """Compare metrics and assignments with source-population caveats."""
    return call(service.compare, body.left.model_dump(), body.right.model_dump())


@router.get("/runs/{run_id}/export")
def export_result(
    run_id: str,
    view: str = "groups",
    point_index: int | None = None,
    service=Depends(get_service),
):
    """Download a complete scoped CSV assembled from HANA query pages."""
    spec = call(QuerySpec, view=view, run_id=run_id, point_index=point_index, limit=500)
    first = call(service.query, spec)

    def chunks():
        """Stream registered columns with identical query scope on every page."""
        result = first
        offset = 0
        output = io.StringIO()
        writer = csv.DictWriter(
            output, fieldnames=first["columns"], extrasaction="ignore"
        )
        writer.writeheader()
        yield output.getvalue()
        output.seek(0)
        output.truncate()
        while True:
            for row in result["rows"]:
                writer.writerow(
                    {
                        k: json.dumps(v) if isinstance(v, (dict, list)) else v
                        for k, v in row.items()
                    }
                )
            yield output.getvalue()
            output.seek(0)
            output.truncate()
            offset += len(result["rows"])
            if offset >= result["total"] or not result["rows"]:
                break
            result = service.query(spec.model_copy(update={"offset": offset}))

    return StreamingResponse(
        chunks(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{view}-{run_id[:32]}.csv"'
        },
    )


@router.delete("/runs/{run_id}")
def remove_run(run_id: str, service=Depends(get_service)):
    """Permanently delete a terminal run and its evidence."""
    return call(service.remove_run, run_id)


@router.delete("/datasets/{dataset_id}")
def remove_dataset(dataset_id: str, service=Depends(get_service)):
    """Permanently delete a snapshot, its source data, drafts and terminal runs."""
    return call(service.remove_dataset, dataset_id)


@router.get("/runs/{run_id}/wheel-export")
def export_wheel_point(run_id: str, point_index: int, service=Depends(get_service)):
    """Download a selected point with uploaded production-wheel headers and notation."""
    from fastapi.responses import Response

    from app.workspace.wheel_export import export_wheel

    if point_index < 1:
        raise HTTPException(422, "point_index must be positive")
    data = call(export_wheel, service, run_id, point_index)
    return Response(
        data,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f'attachment; filename="production-wheel-point-{point_index}.xlsx"'
        },
    )


@router.get("/configuration-help")
def configuration_help():
    """Serve the same static setting definitions used by the agent reference tool."""
    from app.workspace.reference import CONFIGURATION_HELP

    return CONFIGURATION_HELP


@router.get("/plant-profile-defaults")
def plant_profile_defaults(service=Depends(get_service)):
    """Return editable settings and symmetric operational matrix defaults."""
    return call(service.plant_profile_defaults)


@router.get("/plant-profile-help")
def plant_profile_help():
    """Read shared field explanations for the Settings page."""
    return json.loads((Path(__file__).resolve().parents[1] / 'workspace/plant_profile_help.json').read_text())


@router.post("/plant-profiles/matrix-template")
def profile_matrix_template(body: dict):
    """Download an Excel template built from the current accepted matrix."""
    from fastapi.responses import Response
    from app.workspace.profile_matrix import matrix_workbook
    if set(body) != {'matrix_rows'} or not isinstance(body['matrix_rows'], list) or len(body['matrix_rows']) > 10000:
        raise HTTPException(422, 'Expected a bounded matrix_rows list')
    return Response(call(matrix_workbook, body['matrix_rows']),
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers={'Content-Disposition': 'attachment; filename="volume-compatibility.xlsx"'})


@router.post("/plant-profiles/matrix-preview")
def profile_matrix_preview(file: UploadFile = File(...)):
    """Validate an Excel upload without changing the accepted profile matrix."""
    from app.workspace.profile_matrix import MAX_WORKBOOK_BYTES, read_matrix_workbook
    if not (file.filename or '').lower().endswith('.xlsx'):
        raise HTTPException(422, 'Upload an .xlsx compatibility template')
    rows = call(read_matrix_workbook, file.file.read(MAX_WORKBOOK_BYTES + 1))
    return {'matrix_rows': rows, 'volume_count': len({row['volume_a'] for row in rows})}


@router.get("/plant-profiles")
def plant_profiles(service=Depends(get_service)):
    """List active HANA profiles including their current matrix rows."""
    return {"profiles": call(service.list_plant_profiles)}


@router.post("/plant-profiles")
def create_plant_profile(body: dict, service=Depends(get_service)):
    """Persist a reviewed typed profile and its relational matrix."""
    return call(service.create_plant_profile, body)


@router.post("/plant-profiles/interpret")
async def interpret_plant_profile(body: dict, service=Depends(get_service)):
    """Preview a read-only structured interpretation without saving or launching."""
    from app.workspace.profile_interpretation import interpret_profile
    try:
        return await interpret_profile(service, body)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except openai.APIError as exc:
        # Surface provider failures as a handled error so CORS headers reach the browser.
        raise HTTPException(502, f"AI model request failed: {getattr(exc, 'message', exc)}") from exc


@router.get("/plant-profiles/{profile_id}")
def inspect_plant_profile(profile_id: str, service=Depends(get_service)):
    """Inspect one active profile and its revision."""
    return call(service.get_plant_profile, profile_id)


@router.patch("/plant-profiles/{profile_id}")
def update_plant_profile(profile_id: str, body: dict, service=Depends(get_service)):
    """Save reviewed profile changes against the exact current revision."""
    return call(service.update_plant_profile, profile_id, body)


class ProfileRemoval(StrictModel):
    """Require a revision and explicit affirmative removal confirmation."""
    revision: int = Field(ge=1)
    confirmed: bool


@router.delete("/plant-profiles/{profile_id}")
def remove_plant_profile(profile_id: str, body: ProfileRemoval, service=Depends(get_service)):
    """Soft-delete a confirmed revision, preserving historical run snapshots."""
    return call(service.remove_plant_profile, profile_id, body.revision, body.confirmed)
