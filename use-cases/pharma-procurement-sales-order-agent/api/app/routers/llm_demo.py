import logging
import time

from fastapi import APIRouter, Depends, Request
from gen_ai_hub.proxy.native.openai import chat

from ..models.llm import LLMRequest, LLMResponse
from ..observability.llm_usage_logging import (
    TokenUsage,
    emit_llm_usage_event,
    extract_client_host_from_request,
    extract_token_usage,
    extract_user_id_from_request,
    usage_dict_from_tokens,
)
from ..security import get_api_key

# Setup logging
logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(get_api_key)])

MODEL_NAME = "gpt-4.1"  # Using GPT-4 as default for this demo

SYSTEM_PROMPT = "You are a helpful assistant for a minimal demo."


@router.post("/chat", response_model=LLMResponse)
async def simple_chat(http_request: Request, request: LLMRequest) -> LLMResponse:
    """Call GPT-4.1 through SAP Gen AI Hub chat completions.

    Args:
        http_request: FastAPI request used to enrich Cloud Logging events.
        request: User prompt and generation settings for the model call.

    Returns:
        LLMResponse: Generated text, model name, success flag, and token usage.
    """
    usage = TokenUsage()
    start_time = time.perf_counter()
    route = str(http_request.url.path)
    method = http_request.method
    user_id = extract_user_id_from_request(http_request)
    client_host = extract_client_host_from_request(http_request)
    correlation_id = http_request.headers.get("x-correlation-id")

    try:
        logger.info(f"LLM demo request: {request.message[:50]}...")

        response = chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.message},
            ],
            model=MODEL_NAME,
            temperature=request.temperature if request.temperature is not None else 0.6,
            max_tokens=request.max_tokens if request.max_tokens is not None else 1000,
        )

        usage = extract_token_usage(response)
        text = response.choices[0].message.content or ""
        emit_llm_usage_event(
            route=route,
            method=method,
            user_id=user_id,
            client_host=client_host,
            provider="sap-ai-core",
            model=MODEL_NAME,
            llm_endpoint="chat.completions",
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            outcome="success",
            latency_ms=int((time.perf_counter() - start_time) * 1000),
            correlation_id=correlation_id,
        )

        return LLMResponse(
            text=text,
            model=MODEL_NAME,
            success=True,
            usage=usage_dict_from_tokens(usage),
        )

    except Exception as e:
        emit_llm_usage_event(
            route=route,
            method=method,
            user_id=user_id,
            client_host=client_host,
            provider="sap-ai-core",
            model=MODEL_NAME,
            llm_endpoint="chat.completions",
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            outcome="error",
            latency_ms=int((time.perf_counter() - start_time) * 1000),
            correlation_id=correlation_id,
        )
        logger.error(f"LLM demo error: {e}")
        return LLMResponse(
            text="",
            model=MODEL_NAME,
            success=False,
            usage=usage_dict_from_tokens(usage),
            error=str(e),
        )
