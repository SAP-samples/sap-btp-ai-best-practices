"""FastAPI router for metal composition prediction."""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response

from ..models.metal_composition import (
    ClassificationJobStatusResponse,
    ClassificationJobSubmissionResponse,
    ClassificationResetResponse,
    ClassificationStatsResponse,
    DocumentAssignmentRequest,
    HTSCatalogSourceDeleteResponse,
    HTSCatalogSourceListResponse,
    HTSCatalogSourceUploadResponse,
    ItemClassifyBatchRequest,
    ItemPredictBatchRequest,
    ItemPredictRequest,
    MetalCompositionAppSettings,
    MetalCompositionAppSettingsUpdateRequest,
    MetalCompositionItemDetail,
    MetalCompositionItemListResponse,
    Section232CancelDraftBatchResponse,
    Section232ClassificationResponse,
    Section232DraftBatchProcessResponse,
    Section232DraftRuleDeleteResponse,
    Section232DraftRuleListResponse,
    Section232DraftRuleBulkReviewRequest,
    Section232DraftRuleBulkReviewResponse,
    Section232DraftRuleReviewRequest,
    Section232DraftRuleReviewResponse,
    Section232DirectClassificationRequest,
    Section232EligibleCodeDetailListResponse,
    Section232EligibleCodeListResponse,
    Section232PublishDraftBatchRequest,
    Section232PublishDraftBatchResponse,
    Section232PublishedRuleDeleteRequest,
    Section232PublishedRuleDeleteResponse,
    Section232ResetResponse,
    Section232RulesetSummaryResponse,
    Section232ReviewWorkspaceResponse,
    Section232SourceListResponse,
)
from ..security import get_api_key
from ..services.metal_composition import (
    MetalCompositionService,
    get_metal_composition_service,
)
from ..services.metal_composition.documents import DocumentValidationError
from ..services.metal_composition.service import (
    ClassificationAlreadyRunningError,
    ExportReportUnavailableError,
    MissingDocumentsConfirmationRequiredError,
)


logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(get_api_key)])


def _form_bool(value: object, *, default: bool = False) -> bool:
    """Return a boolean parsed from a multipart form value.

    Inputs:
        value: Raw value from ``request.form()``.
        default: Fallback returned when the field is omitted or blank.

    Expected output:
        ``True`` for common truthy strings and ``False`` for common falsey
        strings or missing values.
    """

    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "on"}


@router.post("/section-232/classify", response_model=Section232ClassificationResponse)
async def classify_section_232(
    payload: Section232DirectClassificationRequest,
    service: MetalCompositionService = Depends(get_metal_composition_service),
) -> Section232ClassificationResponse:
    return service.classify_section_232(payload)


@router.get("/app-settings", response_model=MetalCompositionAppSettings)
async def get_app_settings(
    service: MetalCompositionService = Depends(get_metal_composition_service),
) -> MetalCompositionAppSettings:
    return service.get_app_settings()


@router.put("/app-settings", response_model=MetalCompositionAppSettings)
async def update_app_settings(
    payload: MetalCompositionAppSettingsUpdateRequest,
    service: MetalCompositionService = Depends(get_metal_composition_service),
) -> MetalCompositionAppSettings:
    return service.update_app_settings(
        use_material_master_metal_composition=payload.use_material_master_metal_composition,
    )


@router.get("/section-232/ruleset-summary", response_model=Section232RulesetSummaryResponse)
async def get_section_232_ruleset_summary(
    service: MetalCompositionService = Depends(get_metal_composition_service),
) -> Section232RulesetSummaryResponse:
    return service.get_section_232_ruleset_summary()


@router.get("/section-232/review", response_model=Section232ReviewWorkspaceResponse)
async def get_section_232_review_workspace(
    batch_id: Optional[str] = Query(None),
    version: Optional[str] = Query(None),
    hts_query: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    service: MetalCompositionService = Depends(get_metal_composition_service),
) -> Section232ReviewWorkspaceResponse:
    normalized_batch_id = (batch_id or "").strip() or None
    normalized_version = (version or "").strip() or None
    if bool(normalized_batch_id) == bool(normalized_version):
        raise HTTPException(status_code=422, detail="Provide exactly one of batch_id or version.")
    try:
        return service.get_section_232_review_workspace(
            batch_id=normalized_batch_id,
            version=normalized_version,
            hts_query=hts_query,
            limit=limit,
            offset=offset,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=exc.args[0] if exc.args else "Review workspace not found") from exc


@router.get("/section-232/sources", response_model=Section232SourceListResponse)
async def list_section_232_sources(
    service: MetalCompositionService = Depends(get_metal_composition_service),
) -> Section232SourceListResponse:
    return service.list_section_232_sources()


@router.post(
    "/section-232/draft-batches/process",
    response_model=Section232DraftBatchProcessResponse,
    status_code=201,
)
async def process_section_232_draft_batch(
    request: Request,
    service: MetalCompositionService = Depends(get_metal_composition_service),
) -> Section232DraftBatchProcessResponse:
    form = await request.form()
    upload_files: List[UploadFile] = []
    for field_name in ("files", "file"):
        upload_files.extend(
            [
                value
                for value in form.getlist(field_name)
                if hasattr(value, "read") and hasattr(value, "filename")
            ]
        )

    if not upload_files:
        raise HTTPException(status_code=422, detail="At least one PDF file is required.")

    uploads: List[tuple[str, bytes]] = []
    for upload_file in upload_files:
        if not (upload_file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(status_code=422, detail="Uploaded files must all be PDFs.")
        content = await upload_file.read()
        if not content.startswith(b"%PDF"):
            raise HTTPException(status_code=422, detail="Uploaded files must all be valid PDFs.")
        uploads.append((upload_file.filename or "section232-source.pdf", content))

    try:
        return service.process_section_232_draft_batch(
            uploads=uploads,
            include_token_usage=_form_bool(form.get("include_token_usage")),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get(
    "/section-232/draft-batches/{batch_id}/rules",
    response_model=Section232DraftRuleListResponse,
)
async def list_section_232_draft_rules(
    batch_id: str,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    service: MetalCompositionService = Depends(get_metal_composition_service),
) -> Section232DraftRuleListResponse:
    try:
        return service.list_section_232_draft_rules(
            batch_id=batch_id,
            limit=limit,
            offset=offset,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete(
    "/section-232/draft-batches/{batch_id}",
    response_model=Section232CancelDraftBatchResponse,
)
async def cancel_section_232_draft_batch(
    batch_id: str,
    service: MetalCompositionService = Depends(get_metal_composition_service),
) -> Section232CancelDraftBatchResponse:
    try:
        return service.cancel_section_232_draft_batch(batch_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.patch(
    "/section-232/draft-batches/{batch_id}/rules",
    response_model=Section232DraftRuleBulkReviewResponse,
)
async def review_section_232_draft_rules(
    batch_id: str,
    payload: Section232DraftRuleBulkReviewRequest,
    service: MetalCompositionService = Depends(get_metal_composition_service),
) -> Section232DraftRuleBulkReviewResponse:
    try:
        return service.review_section_232_draft_rules(
            batch_id=batch_id,
            candidate_ids=payload.candidate_ids,
            selection_mode=payload.selection_mode,
            excluded_candidate_ids=payload.excluded_candidate_ids,
            decision=payload.decision,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.patch(
    "/section-232/draft-batches/{batch_id}/rules/{candidate_id}",
    response_model=Section232DraftRuleReviewResponse,
)
async def review_section_232_draft_rule(
    batch_id: str,
    candidate_id: str,
    payload: Section232DraftRuleReviewRequest,
    service: MetalCompositionService = Depends(get_metal_composition_service),
) -> Section232DraftRuleReviewResponse:
    try:
        return service.review_section_232_draft_rule(
            batch_id=batch_id,
            candidate_id=candidate_id,
            decision=payload.decision,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.delete(
    "/section-232/draft-batches/{batch_id}/hts-codes/{hts_code}",
    response_model=Section232DraftRuleDeleteResponse,
)
async def delete_section_232_draft_hts_code(
    batch_id: str,
    hts_code: str,
    service: MetalCompositionService = Depends(get_metal_composition_service),
) -> Section232DraftRuleDeleteResponse:
    try:
        return service.delete_section_232_draft_hts_code(
            batch_id=batch_id,
            hts_code=hts_code,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/section-232/draft-batches/{batch_id}/publish",
    response_model=Section232PublishDraftBatchResponse,
)
async def publish_section_232_draft_batch(
    batch_id: str,
    payload: Section232PublishDraftBatchRequest,
    service: MetalCompositionService = Depends(get_metal_composition_service),
) -> Section232PublishDraftBatchResponse:
    try:
        return service.publish_section_232_draft_batch(
            batch_id=batch_id,
            published_by=payload.published_by,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/section-232/published/hts-codes/{hts_code}/delete",
    response_model=Section232PublishedRuleDeleteResponse,
)
async def delete_section_232_published_hts_code(
    hts_code: str,
    payload: Section232PublishedRuleDeleteRequest,
    service: MetalCompositionService = Depends(get_metal_composition_service),
) -> Section232PublishedRuleDeleteResponse:
    try:
        return service.delete_section_232_published_hts_code(
            hts_code=hts_code,
            published_by=payload.published_by,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/section-232/sources", response_model=Section232DraftBatchProcessResponse, status_code=201)
async def upload_section_232_sources(
    request: Request,
    service: MetalCompositionService = Depends(get_metal_composition_service),
) -> Section232DraftBatchProcessResponse:
    form = await request.form()
    upload_files: List[UploadFile] = []
    for field_name in ("files", "file"):
        upload_files.extend(
            [
                value
                for value in form.getlist(field_name)
                if hasattr(value, "read") and hasattr(value, "filename")
            ]
        )

    if not upload_files:
        raise HTTPException(status_code=422, detail="At least one PDF file is required.")

    uploads: List[tuple[str, bytes]] = []
    for upload_file in upload_files:
        if not (upload_file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(status_code=422, detail="Uploaded files must all be PDFs.")
        content = await upload_file.read()
        if not content.startswith(b"%PDF"):
            raise HTTPException(status_code=422, detail="Uploaded files must all be valid PDFs.")
        uploads.append((upload_file.filename or "section232-source.pdf", content))

    return service.process_section_232_draft_batch(
        uploads=uploads,
        include_token_usage=_form_bool(form.get("include_token_usage")),
    )


@router.get("/section-232/eligible-hts-codes", response_model=Section232EligibleCodeListResponse)
async def list_section_232_eligible_hts_codes(
    service: MetalCompositionService = Depends(get_metal_composition_service),
) -> Section232EligibleCodeListResponse:
    return service.list_section_232_eligible_hts_codes()


@router.get(
    "/section-232/eligible-hts-codes/details",
    response_model=Section232EligibleCodeDetailListResponse,
)
async def list_section_232_eligible_hts_code_details(
    query: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=250),
    offset: int = Query(0, ge=0),
    service: MetalCompositionService = Depends(get_metal_composition_service),
) -> Section232EligibleCodeDetailListResponse:
    return service.list_section_232_eligible_hts_code_details(
        query=query,
        limit=limit,
        offset=offset,
    )


@router.post("/section-232/reset", response_model=Section232ResetResponse)
async def reset_section_232_data(
    service: MetalCompositionService = Depends(get_metal_composition_service),
) -> Section232ResetResponse:
    return service.reset_section_232_data()


@router.get("/classifications/stats", response_model=ClassificationStatsResponse)
async def get_classification_stats(
    service: MetalCompositionService = Depends(get_metal_composition_service),
) -> ClassificationStatsResponse:
    return service.get_classification_stats()


@router.post("/classifications/reset", response_model=ClassificationResetResponse)
async def reset_classifications(
    service: MetalCompositionService = Depends(get_metal_composition_service),
) -> ClassificationResetResponse:
    return service.reset_classifications()


@router.get("/hts-catalog/sources", response_model=HTSCatalogSourceListResponse)
async def list_hts_catalog_sources(
    service: MetalCompositionService = Depends(get_metal_composition_service),
) -> HTSCatalogSourceListResponse:
    return service.list_hts_catalog_sources()


@router.post("/hts-catalog/sources", response_model=HTSCatalogSourceUploadResponse, status_code=201)
async def upload_hts_catalog_sources(
    request: Request,
    service: MetalCompositionService = Depends(get_metal_composition_service),
) -> HTSCatalogSourceUploadResponse:
    form = await request.form()
    upload_files: List[UploadFile] = []
    for field_name in ("files", "file"):
        upload_files.extend(
            [
                value
                for value in form.getlist(field_name)
                if hasattr(value, "read") and hasattr(value, "filename")
            ]
        )

    if not upload_files:
        raise HTTPException(status_code=422, detail="At least one CSV file is required.")

    uploads: List[tuple[str, bytes]] = []
    for upload_file in upload_files:
        if not (upload_file.filename or "").lower().endswith(".csv"):
            raise HTTPException(status_code=422, detail="Uploaded HTS catalog files must all be CSVs.")
        uploads.append((upload_file.filename or "chapter.csv", await upload_file.read()))

    try:
        return service.upload_hts_catalog_sources(uploads=uploads)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.delete("/hts-catalog/sources/{filename}", response_model=HTSCatalogSourceDeleteResponse)
async def delete_hts_catalog_source(
    filename: str,
    service: MetalCompositionService = Depends(get_metal_composition_service),
) -> HTSCatalogSourceDeleteResponse:
    try:
        return service.delete_hts_catalog_source(filename=filename)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/items", response_model=MetalCompositionItemListResponse)
async def list_metal_composition_items(
    priority: Optional[str] = Query(None),
    business_segment: Optional[str] = Query(None),
    product_code: Optional[str] = Query(None),
    pn_revised_standardized: Optional[str] = Query(None),
    new_part_description: Optional[str] = Query(None),
    part_description: Optional[str] = Query(None),
    has_documents: Optional[bool] = Query(None),
    is_classified: Optional[bool] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    service: MetalCompositionService = Depends(get_metal_composition_service),
) -> MetalCompositionItemListResponse:
    return service.list_items(
        priority=priority,
        business_segment=business_segment,
        product_code=product_code,
        pn_revised_standardized=pn_revised_standardized,
        new_part_description=new_part_description,
        part_description=part_description,
        has_documents=has_documents,
        is_classified=is_classified,
        limit=limit,
        offset=offset,
    )


@router.post("/items/predict", response_model=ClassificationJobSubmissionResponse, status_code=202)
async def predict_item(
    payload: ItemPredictRequest,
    service: MetalCompositionService = Depends(get_metal_composition_service),
) -> ClassificationJobSubmissionResponse:
    try:
        return service.submit_predict_item_job(
            payload.item_id,
            document_mode=payload.document_mode,
            include_token_usage=payload.include_token_usage,
        )
    except MissingDocumentsConfirmationRequiredError as exc:
        raise HTTPException(status_code=409, detail=exc.detail.model_dump()) from exc
    except ClassificationAlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=exc.detail.model_dump()) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/items/predict-batch", response_model=ClassificationJobSubmissionResponse, status_code=202)
async def predict_item_batch(
    payload: ItemPredictBatchRequest,
    service: MetalCompositionService = Depends(get_metal_composition_service),
) -> ClassificationJobSubmissionResponse:
    if len(payload.item_ids) > service.settings.batch_max_items:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Batch size {len(payload.item_ids)} exceeds maximum of "
                f"{service.settings.batch_max_items}."
            ),
        )
    try:
        return service.submit_predict_items_job(
            payload.item_ids,
            document_mode=payload.document_mode,
            include_token_usage=payload.include_token_usage,
        )
    except MissingDocumentsConfirmationRequiredError as exc:
        raise HTTPException(status_code=409, detail=exc.detail.model_dump()) from exc
    except ClassificationAlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=exc.detail.model_dump()) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/items/classify-batch", response_model=ClassificationJobSubmissionResponse, status_code=202)
async def classify_item_batch(
    payload: ItemClassifyBatchRequest,
    service: MetalCompositionService = Depends(get_metal_composition_service),
) -> ClassificationJobSubmissionResponse:
    if len(payload.item_ids) > service.settings.batch_max_items:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Batch size {len(payload.item_ids)} exceeds maximum of "
                f"{service.settings.batch_max_items}."
            ),
        )
    try:
        return service.submit_classify_items_job(
            payload.item_ids,
            confirm_missing_documents=payload.confirm_missing_documents,
        )
    except MissingDocumentsConfirmationRequiredError as exc:
        raise HTTPException(status_code=409, detail=exc.detail.model_dump()) from exc
    except ClassificationAlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=exc.detail.model_dump()) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/classification-jobs/{job_id}", response_model=ClassificationJobStatusResponse)
async def get_classification_job_status(
    job_id: str,
    service: MetalCompositionService = Depends(get_metal_composition_service),
) -> ClassificationJobStatusResponse:
    try:
        return service.get_classification_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/items/{item_id}", response_model=MetalCompositionItemDetail)
async def get_item_detail(
    item_id: str,
    service: MetalCompositionService = Depends(get_metal_composition_service),
) -> MetalCompositionItemDetail:
    try:
        return service.get_item_detail(item_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/items/{item_id}/export-report")
async def export_item_report(
    item_id: str,
    service: MetalCompositionService = Depends(get_metal_composition_service),
) -> Response:
    try:
        filename, pdf_bytes = service.export_classification_report(item_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ExportReportUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.put("/items/{item_id}/documents", response_model=MetalCompositionItemDetail)
async def replace_item_documents(
    item_id: str,
    payload: DocumentAssignmentRequest,
    service: MetalCompositionService = Depends(get_metal_composition_service),
) -> MetalCompositionItemDetail:
    try:
        return service.replace_item_documents(item_id, document_paths=payload.document_paths)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DocumentValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/items/{item_id}/documents/upload", response_model=MetalCompositionItemDetail)
async def upload_item_document(
    item_id: str,
    file: UploadFile = File(...),
    service: MetalCompositionService = Depends(get_metal_composition_service),
) -> MetalCompositionItemDetail:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Uploaded file must be a PDF.")
    content = await file.read()
    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=422, detail="Uploaded file must be a valid PDF.")
    try:
        return service.upload_item_document(
            item_id,
            filename=file.filename or "uploaded.pdf",
            content=content,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DocumentValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/items/{item_id}/classify", response_model=ClassificationJobSubmissionResponse, status_code=202)
async def classify_single_item(
    item_id: str,
    service: MetalCompositionService = Depends(get_metal_composition_service),
) -> ClassificationJobSubmissionResponse:
    try:
        return service.submit_classify_item_job(item_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except MissingDocumentsConfirmationRequiredError as exc:
        raise HTTPException(status_code=409, detail=exc.detail.model_dump()) from exc
    except ClassificationAlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=exc.detail.model_dump()) from exc
    except DocumentValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
