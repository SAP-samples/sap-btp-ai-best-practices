"""
Customer Onboarding Router using Sonar-Pro

Separate skill for researching new customer information using Perplexity Sonar.
Does NOT use tools/function calling - simple RAG-style queries.
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
import logging
import time

from gen_ai_hub.orchestration_v2.models.message import SystemMessage, UserMessage
from gen_ai_hub.orchestration_v2.models.template import Template, PromptTemplatingModuleConfig
from gen_ai_hub.orchestration_v2.models.llm_model_details import LLMModelDetails
from gen_ai_hub.orchestration_v2.models.config import ModuleConfig, OrchestrationConfig
from gen_ai_hub.orchestration_v2.service import OrchestrationService
from app.observability.llm_usage_logging import emit_llm_usage_event, extract_token_counts
from app.security import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/spa/onboarding", tags=["Customer Onboarding"])


def get_http_request(request: Request) -> Request:
    return request


class OnboardingQueryRequest(BaseModel):
    """Request for customer onboarding research"""
    query: str
    max_tokens: int = 500


class OnboardingQueryResponse(BaseModel):
    """Response from Sonar research"""
    response: str
    query: str


def _run_ai_core_orchestration(
    *,
    system_prompt: str,
    user_prompt: str,
    model_name: str = "sonar-pro",
    temperature: float = 0.2,
    max_tokens: int = 500,
):
    """Run a simple SAP AI Core Orchestration v2 prompt without LangChain proxy wrappers."""
    template = Template(
        template=[
            SystemMessage(content=system_prompt),
            UserMessage(content=user_prompt),
        ]
    )
    llm = LLMModelDetails(
        name=model_name,
        params={
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        },
    )
    prompt_template = PromptTemplatingModuleConfig(
        prompt=template,
        model=llm,
    )
    module_config = ModuleConfig(prompt_templating=prompt_template)
    config = OrchestrationConfig(modules=module_config)
    return OrchestrationService(config=config).run()


def _extract_orchestration_text(result) -> str:
    """Extract assistant text from an Orchestration v2 response."""
    return result.final_result.choices[0].message.content


@router.post(
    "/research",
    response_model=OnboardingQueryResponse,
    dependencies=[Depends(require_api_key)]
)
async def research_customer(
    request: OnboardingQueryRequest,
    http_request: Request = Depends(get_http_request),
):
    """
    Research customer information using Sonar-Pro (Perplexity).

    Use cases:
    - Find public information about a new customer
    - Research industry context
    - Find similar companies
    - General business intelligence

    Note: This endpoint does NOT use tools/function calling.
    """
    try:
        logger.info(f"Onboarding research query: {request.query[:100]}...")

        # Simple AI Core Orchestration v2 call (no tools / no LangChain proxy).
        start = time.perf_counter()
        try:
            response = _run_ai_core_orchestration(
                system_prompt="You are a helpful business research assistant.",
                user_prompt=request.query,
                model_name="sonar-pro",
                temperature=0.2,
                max_tokens=request.max_tokens,
            )
            latency_ms = int((time.perf_counter() - start) * 1000)
            input_tokens, output_tokens, total_tokens = extract_token_counts(response)
            emit_llm_usage_event(
                route="/api/spa/onboarding/research",
                method="POST",
                request=http_request,
                provider="sap-ai-core",
                model="sonar-pro",
                llm_endpoint="OrchestrationService.run",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                outcome="success",
                latency_ms=latency_ms,
            )
        except Exception:
            latency_ms = int((time.perf_counter() - start) * 1000)
            emit_llm_usage_event(
                route="/api/spa/onboarding/research",
                method="POST",
                request=http_request,
                provider="sap-ai-core",
                model="sonar-pro",
                llm_endpoint="OrchestrationService.run",
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                outcome="error",
                latency_ms=latency_ms,
            )
            raise

        return OnboardingQueryResponse(
            response=_extract_orchestration_text(response),
            query=request.query
        )

    except Exception as e:
        logger.error(f"Onboarding research failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Research failed: {str(e)}"
        )


@router.post(
    "/test",
    dependencies=[Depends(require_api_key)]
)
async def test_sonar(http_request: Request = Depends(get_http_request)):
    """Test endpoint to verify Sonar-Pro connectivity"""
    try:
        start = time.perf_counter()
        try:
            response = _run_ai_core_orchestration(
                system_prompt="You are a helpful assistant.",
                user_prompt="What is 1+1? Answer with just the number.",
                model_name="sonar-pro",
                temperature=0.2,
                max_tokens=100,
            )
            latency_ms = int((time.perf_counter() - start) * 1000)
            input_tokens, output_tokens, total_tokens = extract_token_counts(response)
            emit_llm_usage_event(
                route="/api/spa/onboarding/test",
                method="POST",
                request=http_request,
                provider="sap-ai-core",
                model="sonar-pro",
                llm_endpoint="OrchestrationService.run",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                outcome="success",
                latency_ms=latency_ms,
            )
        except Exception:
            latency_ms = int((time.perf_counter() - start) * 1000)
            emit_llm_usage_event(
                route="/api/spa/onboarding/test",
                method="POST",
                request=http_request,
                provider="sap-ai-core",
                model="sonar-pro",
                llm_endpoint="OrchestrationService.run",
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                outcome="error",
                latency_ms=latency_ms,
            )
            raise

        return {
            "status": "success",
            "model": "sonar-pro",
            "test_query": "What is 1+1?",
            "response": _extract_orchestration_text(response)
        }

    except Exception as e:
        logger.error(f"Sonar test failed: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "model": "sonar-pro",
            "error": str(e)
        }
