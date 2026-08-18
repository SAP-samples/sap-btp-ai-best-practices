"""
pa_routes.py
------------
FastAPI router for Payment Advice extraction and FI posting.

POST /api/v1/pa/extract  — extract from PDF via SAP Document AI
POST /api/v1/pa/post     — post to S/4HANA FI via API_PAYMENT_ADVICE_SRV
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, status

from PaymentAdviceProcess.pa_extractor import PaymentAdviceExtractor
from PaymentAdviceProcess.pa_models import (
    ExtractedPaymentAdvice,
    PostPaymentAdviceRequest,
    PostPaymentAdviceResponse,
)
from PaymentAdviceProcess.pa_poster import post_payment_advice

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/pa", tags=["Payment Advice"])


@router.post(
    "/extract",
    response_model=ExtractedPaymentAdvice,
    summary="Extract Payment Advice from PDF via SAP Document AI",
)
async def extract_payment_advice(
    file: UploadFile = File(..., description="Payment Advice PDF"),
    client_id: str = Query(default="default"),
) -> ExtractedPaymentAdvice:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        shutil.copyfileobj(file.file, tmp)
        tmp.close()
        extractor = PaymentAdviceExtractor()
        return extractor.extract(Path(tmp.name), client_id=client_id)
    except Exception as exc:
        logger.exception("PA extraction failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        Path(tmp.name).unlink(missing_ok=True)


@router.post(
    "/post",
    response_model=PostPaymentAdviceResponse,
    summary="Post Payment Advice to S/4HANA FI",
    description=(
        "Posts a Payment Advice to S/4HANA via API_PAYMENT_ADVICE_SRV/A_PaymentAdvice.\n\n"
        "Auto-matches the payer name to a Business Partner unless `payer_bp` is provided.\n\n"
        "**Credentials:** X-SAP-* request headers or .env fallback."
    ),
)
async def post_pa(
    body: PostPaymentAdviceRequest,
    request: Request,
) -> PostPaymentAdviceResponse:
    logger.info(
        "POST /api/v1/pa/post | payer=%r | amount=%s %s",
        body.payer_name, body.total_amount, body.currency,
    )
    try:
        result = post_payment_advice(body, request=request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error("PA post failed | error=%s", exc)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error posting Payment Advice")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    return PostPaymentAdviceResponse(**result)
