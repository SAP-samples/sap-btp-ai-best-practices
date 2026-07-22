"""FastAPI router for the Purchase Requisition Extraction POC."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.security import get_api_key
from app.services.purchase_requisition_runner import (
    live_runner_available,
    load_runner_status,
    start_runner_for_session,
)
from app.services.purchasing_intelligence import (
    create_back_office_referral,
    create_material_proposal,
    list_material_proposals,
    purchasing_intelligence,
)
from app.services.s4_master_data import preflight_master_data_apis, suggest_master_data
from app.services.s4_pr_client import (
    S4PRError,
    build_pr_payload,
    create_purchase_requisition_for_poc,
    preflight_purchase_requisition_api,
)

router = APIRouter()
logger = logging.getLogger(__name__)

DEFAULT_RUN_ID = ""


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _s4_integration_enabled() -> bool:
    return _env_flag("S4_INTEGRATION_ENABLED", default=False)


class ResearchRequest(BaseModel):
    mode: str = Field(default="shortlist", description="shortlist or research")
    experiment_name: str = Field(default="quote_to_pr_extraction")
    document_names: list[str] = Field(default_factory=list)
    include_docai: bool = True
    include_llm: bool = True
    selected_llm_models: list[str] = Field(default_factory=list)
    selected_llm_scenarios: list[str] = Field(default_factory=list)
    selected_docai_scenarios: list[str] = Field(default_factory=list)
    approach_profile: str | None = None


class PRPayloadRequest(BaseModel):
    run_id: str | None = None
    document_name: str
    method_family: str | None = None
    model: str | None = None
    strategy: str | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)
    force_refresh: bool = False


class PRCreateRequest(PRPayloadRequest):
    confirm_create: bool = False


class MaterialProposalRequest(PRPayloadRequest):
    line_index: int = Field(ge=0)


class BackOfficeReferralRequest(PRPayloadRequest):
    pass




def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_").lower()
    return cleaned[:48] or "quote_to_pr_experiment"


def _completed_runs() -> list[str]:
    if not _runs_dir().exists():
        return []
    return sorted(
        [path.name for path in _runs_dir().iterdir() if path.is_dir() and (path / "comparison.json").exists()],
        reverse=True,
    )


def _prepared_experiments() -> list[dict[str, Any]]:
    if not _runs_dir().exists():
        return []
    experiments: list[dict[str, Any]] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for path in sorted([item for item in _runs_dir().iterdir() if item.is_dir()], reverse=True):
        manifest = path / "experiment.json"
        if not manifest.exists() or (path / "comparison.json").exists():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            key = (
                str(data.get("experiment_name") or ""),
                str(data.get("mode") or "shortlist"),
                tuple(sorted(str(item) for item in data.get("documents") or [])),
            )
            if key in seen:
                continue
            seen.add(key)
            data["run_id"] = path.name
            data["runner_status"] = load_runner_status(path)
            experiments.append(data)
        except Exception:
            continue
    return experiments

def _matching_prepared_experiment(payload: ResearchRequest, selected_docs: list[str]) -> dict[str, Any] | None:
    target_key = (
        payload.experiment_name,
        payload.mode,
        tuple(sorted(selected_docs)),
        payload.include_docai,
                payload.include_llm,
        tuple(sorted(payload.selected_llm_models)),
        tuple(sorted(payload.selected_llm_scenarios)),
        tuple(sorted(payload.selected_docai_scenarios)),
        payload.approach_profile or "",
    )
    for experiment in _prepared_experiments():
        key = (
            experiment.get("experiment_name"),
            experiment.get("mode"),
            tuple(sorted(str(item) for item in experiment.get("documents") or [])),
            bool(experiment.get("include_docai", True)),
                        bool(experiment.get("include_llm", True)),
            tuple(sorted(str(item) for item in experiment.get("selected_llm_models") or [])),
            tuple(sorted(str(item) for item in experiment.get("selected_llm_scenarios") or [])),
            tuple(sorted(str(item) for item in experiment.get("selected_docai_scenarios") or [])),
            str(experiment.get("approach_profile") or ""),
        )
        if key == target_key:
            experiment["reused"] = True
            return experiment
    return None


def _create_prepared_experiment(payload: ResearchRequest, selected_docs: list[str]) -> dict[str, Any]:
    _runs_dir().mkdir(parents=True, exist_ok=True)
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_slug(payload.experiment_name)}"
    run_dir = _runs_dir() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": run_id,
        "experiment_name": payload.experiment_name,
        "status": "prepared",
        "mode": payload.mode,
        "documents": selected_docs,
        "include_docai": payload.include_docai,
        "include_llm": payload.include_llm,
        "selected_llm_models": payload.selected_llm_models,
        "selected_llm_scenarios": payload.selected_llm_scenarios,
        "selected_docai_scenarios": payload.selected_docai_scenarios,
        "approach_profile": payload.approach_profile,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(_data_dir()),
        "next_step": "Benchmark runner will execute this session and publish comparison artifacts.",
    }
    (run_dir / "experiment.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _workspace_root() -> Path:
    configured = os.getenv("PR_WORKSPACE_ROOT")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[1] / "data" / "purchase_requisition"


def _runs_dir() -> Path:
    return _workspace_root() / "runs"


def _data_dir() -> Path:
    return _workspace_root() / "data"



def _document_path(document_name: str) -> Path:
    if "/" in document_name or "\\" in document_name or document_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid document name")
    root = _data_dir().resolve()
    path = (root / document_name).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid document path") from exc
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Document not found: {document_name}")
    return path


def _document_profile(document_name: str) -> dict[str, Any]:
    path = _document_path(document_name)
    profile: dict[str, Any] = {
        "document": document_name,
        "file_name": path.name,
        "size_bytes": path.stat().st_size,
        "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        "extension": path.suffix.lower(),
        "page_count": None,
        "text_character_count": None,
        "has_text_layer": None,
        "likely_scanned": None,
        "profile_note": "PDF profile is based on local file inspection, not on extraction quality scoring.",
    }
    if path.suffix.lower() != ".pdf":
        profile["profile_note"] = "Only PDF source profiling is currently implemented."
        return profile
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        profile["page_count"] = len(reader.pages)
        text_character_count = 0
        pages_checked = min(len(reader.pages), 5)
        for page in reader.pages[:pages_checked]:
            try:
                text_character_count += len(page.extract_text() or "")
            except Exception:
                continue
        profile["text_character_count"] = text_character_count
        profile["has_text_layer"] = text_character_count >= 100
        profile["likely_scanned"] = text_character_count < 100
        profile["profile_note"] = (
            "Very little embedded text was detected; OCR or vision-capable extraction is likely required."
            if profile["likely_scanned"]
            else "Embedded text was detected; text-based LLM extraction can use the document text layer."
        )
    except Exception as exc:
        profile["profile_note"] = f"PDF profile could not be fully inspected: {exc}"
    return profile

def _read_json(path: Path) -> Any:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _run_dir(run_id: str | None = None) -> Path:
    selected_run = run_id or DEFAULT_RUN_ID
    path = _runs_dir() / selected_run
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Run not found: {selected_run}")
    return path

def _run_dir_for_delete(run_id: str) -> Path:
    if not run_id or run_id == "_runner_work" or "/" in run_id or "\\" in run_id or run_id in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid run id")
    root = _runs_dir().resolve()
    path = (root / run_id).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid run path") from exc
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return path

def _comparison(run_id: str | None = None) -> dict[str, Any]:
    return _read_json(_run_dir(run_id) / "comparison.json")


def _summary(run_id: str | None = None) -> dict[str, Any]:
    path = _run_dir(run_id) / "summary.json"
    return _read_json(path) if path.exists() else {}


def _rows(run_id: str | None = None) -> list[dict[str, Any]]:
    data = _comparison(run_id)
    return list(data.get("rows") or [])


def _field_rows(run_id: str | None = None) -> list[dict[str, Any]]:
    data = _comparison(run_id)
    return list(data.get("field_rows") or [])


def _method_dirs(run_id: str | None = None) -> list[Path]:
    methods = _run_dir(run_id) / "methods"
    return sorted([path for path in methods.iterdir() if path.is_dir()]) if methods.exists() else []


def _method_metadata(method_dir: Path) -> dict[str, Any]:
    cache_key = _read_json(method_dir / "cache_key.json")
    normalized_path = method_dir / "normalized.json"
    metrics_path = method_dir / "metrics.json"
    return {
        "method_dir": method_dir.name,
        "cache_key": cache_key,
        "normalized": _read_json(normalized_path) if normalized_path.exists() else {},
        "metrics": _read_json(metrics_path) if metrics_path.exists() else {},
    }


def _find_method_result(row: dict[str, Any], run_id: str | None = None) -> dict[str, Any] | None:
    for method_dir in _method_dirs(run_id):
        meta = _method_metadata(method_dir)
        key = meta.get("cache_key") or {}
        if (
            key.get("document_name") == row.get("document")
            and key.get("method_family") == row.get("method_family")
            and key.get("scenario_key") == row.get("scenario")
            and key.get("model") == row.get("model")
        ):
            return meta
    return None


def _candidate_rows_for_document(document_name: str, run_id: str | None = None) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in _rows(run_id)
        if row.get("document") == document_name
    ]
    return sorted(
        candidates,
        key=lambda row: (
            str(row.get("status") or "").lower() == "success",
            isinstance(row.get("quality_score"), (int, float)),
            float(row.get("quality_score") or 0),
            float(row.get("extraction_quality") or 0),
            float(row.get("pr_readiness") or 0),
            float(row.get("confidence") or 0),
        ),
        reverse=True,
    )


def _best_row(document_name: str, run_id: str | None = None) -> dict[str, Any] | None:
    rows = _candidate_rows_for_document(document_name, run_id)
    return rows[0] if rows else None


def _matching_row(
    *,
    document_name: str,
    run_id: str | None = None,
    method_family: str | None = None,
    model: str | None = None,
    strategy: str | None = None,
) -> dict[str, Any] | None:
    candidates = _candidate_rows_for_document(document_name, run_id)
    for row in candidates:
        if method_family and row.get("method_family") != method_family:
            continue
        if model and row.get("model") != model:
            continue
        if strategy and row.get("scenario") != strategy:
            continue
        return row
    return candidates[0] if candidates else None


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "document": row.get("document"),
        "approach_family": row.get("method_family"),
        "model": row.get("model"),
        "strategy": row.get("scenario"),
        "status": row.get("status"),
        "overall_score": row.get("quality_score"),
        "extraction_quality": row.get("extraction_quality"),
        "pr_readiness": row.get("pr_readiness"),
        "confidence": row.get("confidence"),
        "latency_s": row.get("latency_s"),
        "tokens": row.get("tokens"),
        "cost": row.get("cost_display"),
        "cost_basis": row.get("cost_basis"),
        "error_code": row.get("error_code"),
        "error_explanation": row.get("error_explanation"),
        "recommendation": row.get("recommendation"),
        "risks": row.get("risks"),
    }


def _extract_header(normalized: dict[str, Any]) -> dict[str, Any]:
    header = normalized.get("header")
    return header if isinstance(header, dict) else {}


def _extract_suppliers(normalized: dict[str, Any]) -> list[dict[str, Any]]:
    header = _extract_header(normalized)
    suppliers: list[dict[str, Any]] = []
    vendor = {
        "role": "primary vendor",
        "name": header.get("vendor_name"),
        "address": header.get("vendor_address"),
        "phone": header.get("vendor_phone"),
        "email": header.get("vendor_email"),
        "tax_id": header.get("vendor_tax_id"),
    }
    if any(vendor.values()):
        suppliers.append(vendor)
    for item in normalized.get("suppliers") or []:
        if isinstance(item, dict):
            suppliers.append(item)
    return suppliers


def _presentation_fields(normalized: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[Any]]:
    """Return business-facing fields with small display-only cleanup."""

    header = dict(_extract_header(normalized))
    warnings = list(normalized.get("warnings") or [])
    line_items: list[dict[str, Any]] = []
    trailing_contact = re.compile(r"\b([A-Z]{2,}(?:\s+[A-Z]\.?)?)\s+(\d{3}-\d{3}-\d{4})\s*$")
    for item in normalized.get("line_items") or []:
        if not isinstance(item, dict):
            continue
        clean_item = dict(item)
        description = str(clean_item.get("description") or "")
        match = trailing_contact.search(description)
        if match:
            contact = f"{match.group(1).strip()} {match.group(2)}"
            if not header.get("buyer_contact"):
                header["buyer_contact"] = contact
            clean_item["description"] = description[: match.start()].strip(" -;,\n")
            warnings.append(
                {
                    "field": "line_items.description",
                    "message": f"Display cleanup moved trailing contact '{contact}' from line description to buyer contact.",
                }
            )
        line_items.append(clean_item)
    return header, line_items, warnings


def _best_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if isinstance(row.get("quality_score"), (int, float))]
    if not scored:
        return {
            "best_current_approach": "No scored approach yet",
            "main_risk": "Run research or load a saved run.",
            "recommended_next_action": "Open the saved demo project.",
        }
    best = sorted(
        scored,
        key=lambda row: (
            float(row.get("quality_score") or 0),
            float(row.get("extraction_quality") or 0),
            float(row.get("confidence") or 0),
        ),
        reverse=True,
    )[0]
    low_pr = [row for row in scored if float(row.get("pr_readiness") or 0) < 60]
    return {
        "best_current_approach": f"{best.get('model')} / {best.get('scenario')} on {best.get('document')}",
        "main_risk": (
            f"{len(low_pr)} approach(es) still need SAP PR enrichment before creation."
            if low_pr
            else "No major readiness issue in the scored shortlist."
        ),
        "recommended_next_action": best.get("recommendation") or "Review the best extraction with business owners.",
        "best_score": best.get("quality_score"),
        "best_extraction_quality": best.get("extraction_quality"),
        "best_pr_readiness": best.get("pr_readiness"),
        "confidence": best.get("confidence"),
    }


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "purchase-requisition-extraction"}


@router.get("/projects")
async def projects(api_key: str = Depends(get_api_key)) -> dict[str, Any]:
    runs = _completed_runs()
    documents = sorted({path.name for path in _data_dir().glob("*.PDF")} | {path.name for path in _data_dir().glob("*.pdf")})
    return {
        "projects": [
            {
                "id": "purchase_requisition_extraction",
                "name": "Quote-to-PR BTP AI",
                "description": "Quote-to-PR extraction workspace with upload, processing, and results review.",
                "default_run_id": DEFAULT_RUN_ID,
                "runs": runs,
                "documents": documents,
                "data_dir": str(_data_dir()),
                "live_runner_available": live_runner_available(),
                "prepared_experiments": _prepared_experiments(),
            }
        ]
    }




@router.post("/documents/upload")
async def upload_documents(files: list[UploadFile] = File(...), api_key: str = Depends(get_api_key)) -> dict[str, Any]:
    _data_dir().mkdir(parents=True, exist_ok=True)
    saved: list[dict[str, Any]] = []
    for upload in files:
        filename = Path(upload.filename or "").name
        if not filename or Path(filename).suffix.lower() != ".pdf":
            raise HTTPException(status_code=400, detail=f"Only PDF files are supported: {upload.filename}")
        target = (_data_dir() / filename).resolve()
        root = _data_dir().resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid upload file name") from exc
        content = await upload.read()
        target.write_bytes(content)
        saved.append({"file_name": filename, "size_bytes": len(content)})
    return {"status": "saved", "documents": saved, "data_dir": str(_data_dir())}


@router.get("/overview")
async def overview(run_id: str | None = None, api_key: str = Depends(get_api_key)) -> dict[str, Any]:
    rows = _rows(run_id)
    summary = _summary(run_id)
    documents = sorted({row.get("document") for row in rows if row.get("document")})
    family_counts = Counter(str(row.get("method_family") or "unknown") for row in rows)
    status_counts = Counter(str(row.get("status") or "unknown") for row in rows)
    top_by_document = {
        document: [_compact_row(row) for row in _candidate_rows_for_document(str(document), run_id)[:5]]
        for document in documents
    }
    return {
        "run_id": run_id or DEFAULT_RUN_ID,
        "business_summary": summary.get("business_summary") or "Saved research results are available for review.",
        "best": _best_summary(rows),
        "documents": documents,
        "method_count": len(rows),
        "family_counts": dict(family_counts),
        "status_counts": dict(status_counts),
        "top_by_document": top_by_document,
        "comparison_rows": [_compact_row(row) for row in rows],
        "field_rows": _field_rows(run_id),
    }



@router.get("/documents/{document_name}/profile")
async def document_profile(document_name: str, api_key: str = Depends(get_api_key)) -> dict[str, Any]:
    return _document_profile(document_name)


@router.get("/documents/{document_name}/file")
async def document_file(document_name: str, api_key: str = Depends(get_api_key)) -> FileResponse:
    path = _document_path(document_name)
    return FileResponse(path, media_type="application/pdf", filename=path.name)


@router.get("/documents/{document_name}")
async def document_detail(
    document_name: str,
    run_id: str | None = None,
    method_family: str | None = Query(default=None),
    model: str | None = Query(default=None),
    strategy: str | None = Query(default=None),
    api_key: str = Depends(get_api_key),
) -> dict[str, Any]:
    scored_rows = _candidate_rows_for_document(document_name, run_id)
    if not scored_rows:
        raise HTTPException(status_code=404, detail=f"No extraction found for {document_name}")
    top_rows = scored_rows[:5]
    best = _matching_row(
        document_name=document_name,
        run_id=run_id,
        method_family=method_family,
        model=model,
        strategy=strategy,
    )
    if best is None:
        raise HTTPException(status_code=404, detail=f"No matching extraction method found for {document_name}")
    method = _find_method_result(best, run_id)
    normalized = method.get("normalized") if method else {}
    if not isinstance(normalized, dict):
        normalized = {}
    header, line_items, warnings = _presentation_fields(normalized)
    return {
        "document": document_name,
        "best_approach": _compact_row(best),
        "top_approaches": [_compact_row(row) for row in top_rows],
        "extracted": {
            "header": header,
            "suppliers": _extract_suppliers(normalized),
            "line_items": line_items,
            "pr_mapping": normalized.get("pr_mapping") or {},
            "evidence": normalized.get("evidence") or [],
            "warnings": warnings,
        },
    }



def _normalized_for_pr_request(payload: PRPayloadRequest) -> tuple[dict[str, Any], dict[str, Any]]:
    row = _matching_row(
        document_name=payload.document_name,
        run_id=payload.run_id,
        method_family=payload.method_family,
        model=payload.model,
        strategy=payload.strategy,
    )
    if row is None:
        raise HTTPException(status_code=404, detail=f"No matching extraction found for {payload.document_name}")
    method = _find_method_result(row, payload.run_id)
    normalized = method.get("normalized") if method else {}
    if not isinstance(normalized, dict):
        normalized = {}
    header, line_items, warnings = _presentation_fields(normalized)
    business_normalized = dict(normalized)
    business_normalized["header"] = header
    business_normalized["line_items"] = line_items
    business_normalized["warnings"] = warnings
    return business_normalized, _compact_row(row)


@router.get("/s4/purchase-requisition/preflight")
async def s4_pr_preflight(api_key: str = Depends(get_api_key)) -> dict[str, Any]:
    try:
        return preflight_purchase_requisition_api()
    except S4PRError as exc:
        raise HTTPException(status_code=502, detail={"message": str(exc), "details": exc.details}) from exc


@router.get("/s4/master-data/preflight")
async def s4_master_data_preflight(api_key: str = Depends(get_api_key)) -> dict[str, Any]:
    try:
        return preflight_master_data_apis()
    except S4PRError as exc:
        raise HTTPException(status_code=502, detail={"message": str(exc), "details": exc.details}) from exc


@router.post("/s4/master-data/suggestions")
async def s4_master_data_suggestions(
    payload: PRPayloadRequest,
    api_key: str = Depends(get_api_key),
) -> dict[str, Any]:
    normalized, approach = _normalized_for_pr_request(payload)
    if not _s4_integration_enabled():
        return {
            "status": "disabled",
            "document": payload.document_name,
            "approach": approach,
            "message": "S/4HANA master-data matching is optional and is not configured in this deployment.",
        }
    try:
        result = suggest_master_data(normalized, force_refresh=payload.force_refresh)
    except S4PRError as exc:
        logger.warning("S/4 master-data suggestions unavailable: %s", exc)
        return {
            "status": "unavailable",
            "document": payload.document_name,
            "approach": approach,
            "message": "SAP master data could not be searched. Enter the SAP values manually.",
        }
    return {
        **result,
        "document": payload.document_name,
        "approach": approach,
    }


@router.post("/purchasing-intelligence")
async def purchasing_intelligence_for_document(
    payload: PRPayloadRequest,
    api_key: str = Depends(get_api_key),
) -> dict[str, Any]:
    normalized, approach = _normalized_for_pr_request(payload)
    return {
        **purchasing_intelligence(normalized, payload.document_name),
        "approach": approach,
    }


@router.post("/material-proposals")
async def prepare_material_proposal(
    payload: MaterialProposalRequest,
    api_key: str = Depends(get_api_key),
) -> dict[str, Any]:
    normalized, approach = _normalized_for_pr_request(payload)
    try:
        proposal = create_material_proposal(
            normalized,
            payload.document_name,
            payload.line_index,
            _workspace_root(),
        )
    except IndexError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {**proposal, "approach": approach}


@router.get("/material-proposals")
async def saved_material_proposals(
    document_name: str | None = Query(default=None),
    api_key: str = Depends(get_api_key),
) -> dict[str, Any]:
    proposals = list_material_proposals(_workspace_root(), document_name)
    return {"document": document_name, "proposals": proposals, "count": len(proposals)}


@router.post("/back-office-referrals")
async def prepare_back_office_referral(
    payload: BackOfficeReferralRequest,
    api_key: str = Depends(get_api_key),
) -> dict[str, Any]:
    normalized, approach = _normalized_for_pr_request(payload)
    referral = create_back_office_referral(
        normalized,
        payload.document_name,
        _workspace_root(),
    )
    return {**referral, "approach": approach}


@router.post("/s4/purchase-requisition/payload")
async def s4_pr_payload(payload: PRPayloadRequest, api_key: str = Depends(get_api_key)) -> dict[str, Any]:
    normalized, approach = _normalized_for_pr_request(payload)
    prepared = build_pr_payload(normalized, payload.overrides)
    return {
        "status": "prepared",
        "document": payload.document_name,
        "approach": approach,
        "s4_integration_enabled": _s4_integration_enabled(),
        "purchase_requisition": prepared,
    }


@router.post("/s4/purchase-requisition/create")
async def s4_pr_create(payload: PRCreateRequest, api_key: str = Depends(get_api_key)) -> dict[str, Any]:
    if not _s4_integration_enabled():
        raise HTTPException(
            status_code=503,
            detail="S/4HANA creation is disabled. Set S4_INTEGRATION_ENABLED=true after configuring the connection.",
        )
    if not payload.confirm_create:
        raise HTTPException(status_code=400, detail="Set confirm_create=true after reviewing the generated PR payload.")
    normalized, approach = _normalized_for_pr_request(payload)
    prepared = build_pr_payload(normalized, payload.overrides)
    if not prepared.get("ready_for_create") and not payload.overrides.get("allow_incomplete_payload"):
        raise HTTPException(
            status_code=400,
            detail={
                "message": "PR payload is missing SAP-required enrichment fields.",
                "missing_fields": prepared.get("missing_fields") or [],
                "hint": "Provide fallback constants or UI overrides, then retry.",
            },
        )
    try:
        created = create_purchase_requisition_for_poc(prepared["payload"])
    except S4PRError as exc:
        raise HTTPException(status_code=502, detail={"message": str(exc), "status_code": exc.status_code, "details": exc.details}) from exc
    return {
        "status": "created",
        "document": payload.document_name,
        "approach": approach,
        "prepared_payload": prepared,
        "s4_result": created,
    }


@router.delete("/runs/{run_id}")
async def delete_run(run_id: str, api_key: str = Depends(get_api_key)) -> dict[str, Any]:
    run_dir = _run_dir_for_delete(run_id)
    status = load_runner_status(run_dir)
    if status.get("status") == "running":
        raise HTTPException(status_code=409, detail="Cannot delete a run while the benchmark runner is still running.")
    shutil.rmtree(run_dir)
    return {"status": "deleted", "run_id": run_id}

@router.get("/runs/{run_id}/status")
async def runner_status(run_id: str, api_key: str = Depends(get_api_key)) -> dict[str, Any]:
    return load_runner_status(_run_dir(run_id))


@router.post("/runs/{run_id}/execute")
async def execute_run(run_id: str, api_key: str = Depends(get_api_key)) -> dict[str, Any]:
    run_dir = _run_dir(run_id)
    status = start_runner_for_session(run_dir, workspace_root=_workspace_root())
    return {"run_id": run_id, "runner_status": status, "live_runner_available": live_runner_available()}


@router.post("/research")
async def research(payload: ResearchRequest, api_key: str = Depends(get_api_key)) -> dict[str, Any]:
    available_docs = sorted({path.name for path in _data_dir().glob("*.PDF")} | {path.name for path in _data_dir().glob("*.pdf")})
    selected_docs = payload.document_names or available_docs
    missing_docs = [document for document in selected_docs if document not in available_docs]
    if missing_docs:
        raise HTTPException(status_code=400, detail=f"Documents not found in project data folder: {', '.join(missing_docs)}")

    manifest = _matching_prepared_experiment(payload, selected_docs) or _create_prepared_experiment(payload, selected_docs)
    run_dir = _run_dir(str(manifest["run_id"]))
    runner = start_runner_for_session(run_dir, workspace_root=_workspace_root()) if live_runner_available() else load_runner_status(run_dir)
    scope = "full research" if payload.mode == "research" else "default shortlist"
    return {
        "status": "prepared",
        "mode": payload.mode,
        "scope": scope,
        "message": (
            f"Prepared experiment '{payload.experiment_name}' with {len(selected_docs)} document(s). "
            "Benchmark runner has been started in the background."
            if live_runner_available()
            else f"Prepared experiment '{payload.experiment_name}' with {len(selected_docs)} document(s). Runner is disabled."
        ),
        "run_id": manifest["run_id"],
        "documents": selected_docs,
        "reused": bool(manifest.get("reused")),
        "include_docai": payload.include_docai,
        "include_llm": payload.include_llm,
        "selected_llm_models": payload.selected_llm_models,
        "selected_llm_scenarios": payload.selected_llm_scenarios,
        "selected_docai_scenarios": payload.selected_docai_scenarios,
        "approach_profile": payload.approach_profile,
        "live_runner_available": live_runner_available(),
        "runner_status": runner,
        "next_step": "Watch runner progress; completed charts will appear when status becomes completed.",
    }
