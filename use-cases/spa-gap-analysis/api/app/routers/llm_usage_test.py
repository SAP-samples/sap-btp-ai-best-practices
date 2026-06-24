"""Test route for SAP Cloud Logging-compatible LLM usage events."""

from fastapi import APIRouter, Depends, Request

from app.observability.llm_usage_logging import emit_llm_usage_event, extract_user_id_from_request
from app.security import require_api_key


router = APIRouter(prefix="/test", tags=["Observability"])


@router.get("/log-llm-usage", dependencies=[Depends(require_api_key)])
async def log_llm_usage_test(request: Request):
    """Emit one synthetic llm_usage event for Cloud Logging validation."""

    user_id = extract_user_id_from_request(request)
    emit_llm_usage_event(
        route="/api/test/log-llm-usage",
        method="GET",
        request=request,
        provider="sap-ai-core",
        model="test-model",
        llm_endpoint="test",
        input_tokens=1,
        output_tokens=1,
        total_tokens=2,
        outcome="success",
        latency_ms=0,
    )
    return {
        "status": "ok",
        "message": "llm_usage event emitted",
        "user_detected": bool(user_id),
    }

