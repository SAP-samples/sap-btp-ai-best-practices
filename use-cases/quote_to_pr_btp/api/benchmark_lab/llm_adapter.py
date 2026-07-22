"""SAP AI Core / GenAI Hub LLM adapter with usage logging."""

from __future__ import annotations

import base64
import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from observability.llm_usage_logging import (
    emit_llm_usage_event,
    extract_token_usage,
    usage_dict_from_tokens,
)

from .extraction_schema import normalize_quote_payload, schema_as_pretty_json
from .json_utils import parse_json_object
from .prompts import PromptSpec, get_prompt
from .settings import AppPaths


class LLMAdapterError(RuntimeError):
    """Raised when an LLM call fails."""


class UnsupportedModelPayloadError(LLMAdapterError):
    """Raised when a selected model path does not support the requested payload."""


@dataclass
class LLMCallResult:
    """Result of a single LLM extraction call."""

    model: str
    scenario_key: str
    prompt_version: str
    provider_family: str
    raw_text: str
    parsed_json: dict[str, Any] | None
    normalized_json: dict[str, Any]
    usage: dict[str, int]
    latency_ms: int
    outcome: str
    error: str | None = None
    correlation_id: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def provider_family_for_model(model: str) -> str:
    """Classify model names into native client families."""

    if model.startswith("gemini-"):
        return "gemini"
    if model.startswith("gpt-"):
        return "openai"
    if model.startswith("anthropic--"):
        return "bedrock_anthropic"
    if model.startswith("sonar"):
        return "sonar"
    return "orchestration"


class GenAIHubLLMAdapter:
    """LLM extraction adapter with provider-aware PDF handling."""

    def __init__(self, paths: AppPaths | None = None) -> None:
        self.paths = paths or AppPaths.for_project()
        self._ocr_text_cache: dict[str, str] = {}
        load_dotenv(dotenv_path=self.paths.env_file)

    def extract_quote(
        self,
        *,
        pdf_path: Path,
        model: str,
        scenario_key: str,
        route: str = "streamlit.llm_extract",
        user_id: str | None = None,
    ) -> LLMCallResult:
        """Extract quote data from a PDF with the selected prompt scenario."""

        prompt = get_prompt(scenario_key)
        family = provider_family_for_model(model)
        correlation_id = str(uuid.uuid4())
        started = time.perf_counter()
        raw_text = ""
        response_obj: Any = None
        outcome = "success"
        error = None

        try:
            if scenario_key == "dynamic_prompt":
                prompt = self._build_document_specific_prompt(
                    pdf_path=pdf_path,
                    model=model,
                    family=family,
                    seed_prompt=prompt,
                    route=route,
                    user_id=user_id,
                )
                raw_text, response_obj = self._run_prompt_once(pdf_path, model, family, prompt)
            elif scenario_key == "dynamic_prompt_judge_loop":
                prompt = self._build_dynamic_judge_loop_prompt(prompt)
                raw_text, response_obj = self._run_dynamic_judge_loop(
                    pdf_path=pdf_path,
                    model=model,
                    family=family,
                    base_prompt=get_prompt("detailed_static_prompt"),
                    repair_prompt=prompt,
                    route=route,
                    user_id=user_id,
                )
            else:
                raw_text, response_obj = self._run_prompt_once(pdf_path, model, family, prompt)

            parsed = parse_json_object(raw_text)
            normalized = normalize_quote_payload(parsed)
        except Exception as exc:
            outcome = "error"
            error = str(exc)
            parsed = None
            normalized = normalize_quote_payload({})
        latency_ms = int((time.perf_counter() - started) * 1000)
        usage_tokens = extract_token_usage(response_obj)
        usage = usage_dict_from_tokens(usage_tokens)

        emit_llm_usage_event(
            route=route,
            method="INTERNAL",
            user_id=user_id,
            provider="sap-ai-core",
            model=model,
            llm_endpoint=family,
            input_tokens=usage_tokens.input_tokens,
            cached_input_tokens=usage_tokens.cached_input_tokens,
            output_tokens=usage_tokens.output_tokens,
            outcome=outcome,
            latency_ms=latency_ms,
            correlation_id=correlation_id,
        )

        return LLMCallResult(
            model=model,
            scenario_key=scenario_key,
            prompt_version=prompt.version,
            provider_family=family,
            raw_text=raw_text,
            parsed_json=parsed,
            normalized_json=normalized,
            usage=usage,
            latency_ms=latency_ms,
            outcome=outcome,
            error=error,
            correlation_id=correlation_id,
        )

    def _run_prompt_once(self, pdf_path: Path, model: str, family: str, prompt: PromptSpec) -> tuple[str, Any]:
        if family == "gemini":
            return self._run_gemini_pdf(pdf_path, model, prompt)
        if family == "openai":
            return self._run_openai_pdf(pdf_path, model, prompt)
        return self._run_orchestration_text(pdf_path, model, prompt)

    def _build_document_specific_prompt(
        self,
        *,
        pdf_path: Path,
        model: str,
        family: str,
        seed_prompt: PromptSpec,
        route: str,
        user_id: str | None,
    ) -> PromptSpec:
        """Run the dynamic prompt-design step, then return the generated extraction prompt."""

        started = time.perf_counter()
        correlation_id = str(uuid.uuid4())
        outcome = "success"
        response_obj: Any = None
        try:
            raw_text, response_obj = self._run_prompt_once(pdf_path, model, family, seed_prompt)
            parsed = parse_json_object(raw_text)
            generated_prompt = str(parsed.get("generated_prompt") or "").strip()
            if not generated_prompt:
                raise LLMAdapterError("Dynamic prompt generation did not return generated_prompt.")
            system_prompt = (
                "You are a precise extraction engine for vendor quotation documents. "
                "Use the document-specific instructions, return valid JSON only, and never invent values."
            )
            return PromptSpec(
                scenario_key="dynamic_prompt",
                version="v2",
                system_prompt=system_prompt,
                user_prompt=generated_prompt,
                summary="Dynamic prompt: generated a document-specific extraction prompt, then extracted the quote.",
            )
        except Exception:
            outcome = "error"
            raise
        finally:
            usage_tokens = extract_token_usage(response_obj)
            emit_llm_usage_event(
                route=f"{route}.dynamic_prompt_design",
                method="INTERNAL",
                user_id=user_id,
                provider="sap-ai-core",
                model=model,
                llm_endpoint=family,
                input_tokens=usage_tokens.input_tokens,
                cached_input_tokens=usage_tokens.cached_input_tokens,
                output_tokens=usage_tokens.output_tokens,
                outcome=outcome,
                latency_ms=int((time.perf_counter() - started) * 1000),
                correlation_id=correlation_id,
            )

    @staticmethod
    def _build_dynamic_judge_loop_prompt(prompt: PromptSpec) -> PromptSpec:
        return PromptSpec(
            scenario_key="dynamic_prompt_judge_loop",
            version="v3",
            system_prompt=prompt.system_prompt
            + " You are also a strict extraction quality judge. Repair omissions only when the source document supports the value.",
            user_prompt=(
                "Repair the previous extraction against the source document.\n\n"
                "Quality goals:\n"
                "- Improve quote header, line items, PR mapping readiness, evidence, and warnings.\n"
                "- Keep all unsupported values null; never invent SAP master data.\n"
                "- Add warnings for missing SAP PR mapping fields and unclear commercial terms.\n"
                "- Preserve every source-supported line item.\n"
                "- Return only valid JSON in the canonical schema below.\n\n"
                f"Canonical schema:\n{schema_as_pretty_json()}"
            ),
            summary="Dynamic prompt + judge loop: static extraction followed by source-grounded repair.",
        )

    def _run_dynamic_judge_loop(
        self,
        *,
        pdf_path: Path,
        model: str,
        family: str,
        base_prompt: PromptSpec,
        repair_prompt: PromptSpec,
        route: str,
        user_id: str | None,
    ) -> tuple[str, Any]:
        """Run static extraction, then a second source-grounded repair pass."""

        first_raw, first_response = self._run_prompt_once(pdf_path, model, family, base_prompt)
        repair_user_prompt = (
            f"{repair_prompt.user_prompt}\n\n"
            "Previous extraction JSON/text:\n"
            f"{first_raw}\n\n"
            "Judge the previous extraction privately, then return the repaired final JSON only."
        )
        repair_spec = PromptSpec(
            scenario_key=repair_prompt.scenario_key,
            version=repair_prompt.version,
            system_prompt=repair_prompt.system_prompt,
            user_prompt=repair_user_prompt,
            summary=repair_prompt.summary,
        )
        final_raw, final_response = self._run_prompt_once(pdf_path, model, family, repair_spec)
        combined_usage = _combined_usage([first_response, final_response])
        emit_llm_usage_event(
            route=f"{route}.dynamic_judge_repair",
            method="INTERNAL",
            user_id=user_id,
            provider="sap-ai-core",
            model=model,
            llm_endpoint=family,
            input_tokens=combined_usage.input_tokens,
            cached_input_tokens=combined_usage.cached_input_tokens,
            output_tokens=combined_usage.output_tokens,
            outcome="success",
            latency_ms=None,
            correlation_id=str(uuid.uuid4()),
        )
        return final_raw, {"usage": usage_dict_from_tokens(combined_usage)}

    def _run_openai_pdf(self, pdf_path: Path, model: str, prompt: PromptSpec) -> tuple[str, Any]:
        """Run OpenAI Responses API with a PDF file payload."""

        try:
            from gen_ai_hub.proxy.native.openai import responses
        except Exception as exc:
            return self._run_orchestration_text(pdf_path, model, prompt)

        if responses is None or not hasattr(responses, "create"):
            return self._run_orchestration_text(pdf_path, model, prompt)

        pdf_base64 = base64.b64encode(pdf_path.read_bytes()).decode("utf-8")
        try:
            response = responses.create(
                model=model,
                instructions=prompt.system_prompt,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_file",
                                "filename": pdf_path.name,
                                "file_data": f"data:application/pdf;base64,{pdf_base64}",
                            },
                            {"type": "input_text", "text": prompt.user_prompt},
                        ],
                    }
                ],
                reasoning={"effort": "low"},
            )
        except (AttributeError, TypeError) as exc:
            if "create" not in str(exc):
                raise
            return self._run_orchestration_text(pdf_path, model, prompt)
        return getattr(response, "output_text", str(response)), response

    def _run_gemini_pdf(self, pdf_path: Path, model: str, prompt: PromptSpec) -> tuple[str, Any]:
        """Run Gemini native client with inline PDF bytes."""

        try:
            from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
            from gen_ai_hub.proxy.native.google_genai.clients import Client
            from google.genai import types
        except Exception as exc:
            raise LLMAdapterError(f"Gemini native client is unavailable: {exc}") from exc

        proxy_client = get_proxy_client("gen-ai-hub")
        client = Client(proxy_client=proxy_client)
        pdf_part = types.Part.from_bytes(data=pdf_path.read_bytes(), mime_type="application/pdf")
        prompt_part = types.Part.from_text(text=f"{prompt.system_prompt}\n\n{prompt.user_prompt}")
        contents = [types.Content(role="user", parts=[pdf_part, prompt_part])]
        response = client.models.generate_content(model=model, contents=contents)
        return getattr(response, "text", str(response)), response

    def _run_orchestration_text(self, pdf_path: Path, model: str, prompt: PromptSpec) -> tuple[str, Any]:
        """Run orchestration text flow using extractable PDF text as context."""

        try:
            from gen_ai_hub.orchestration.models.config import OrchestrationConfig
            from gen_ai_hub.orchestration.models.llm import LLM
            from gen_ai_hub.orchestration.models.message import SystemMessage, UserMessage
            from gen_ai_hub.orchestration.models.template import Template
            from gen_ai_hub.orchestration.service import OrchestrationService
        except Exception as exc:
            raise LLMAdapterError(f"GenAI orchestration client is unavailable: {exc}") from exc

        text = self._extract_text_for_context(pdf_path, model)
        template = Template(
            messages=[
                SystemMessage(prompt.system_prompt),
                UserMessage(f"{prompt.user_prompt}\n\nDOCUMENT TEXT:\n{text}"),
            ]
        )
        llm = LLM(name=model, version="latest")
        result = OrchestrationService(config=OrchestrationConfig(template=template, llm=llm)).run()
        raw = result.orchestration_result.choices[0].message.content
        return raw, result.orchestration_result

    def _extract_text_for_context(self, pdf_path: Path, model: str | None = None) -> str:
        try:
            import pypdf  # type: ignore
        except Exception as exc:
            raise UnsupportedModelPayloadError(
                "Text fallback requires pypdf. Use a PDF-capable Gemini/OpenAI model or install pypdf."
            ) from exc

        parts: list[str] = []
        with pdf_path.open("rb") as handle:
            reader = pypdf.PdfReader(handle)
            for page in reader.pages:
                parts.append(page.extract_text() or "")
        text = "\n".join(parts).strip()
        if not text:
            return self._extract_ocr_text_for_context(pdf_path, model)
        return text[:60000]

    def _extract_ocr_text_for_context(self, pdf_path: Path, model: str | None = None) -> str:
        """Use a PDF-native Gemini model as an OCR bridge for text-only routes."""

        cache_key = str(pdf_path.resolve())
        if cache_key in self._ocr_text_cache:
            return self._ocr_text_cache[cache_key]

        ocr_prompt = PromptSpec(
            scenario_key="ocr_text_bridge",
            version="v1",
            system_prompt=(
                "You are an OCR and document transcription engine for purchase requisition automation. "
                "Transcribe the PDF faithfully. Preserve table rows, labels, values, totals, dates, addresses, "
                "phone numbers, emails, item numbers, quantities, units, prices, and footnotes. "
                "Do not summarize and do not infer missing values."
            ),
            user_prompt=(
                "Return clean Markdown text for this PDF. Keep reading order natural. "
                "If a value is unclear, write [unclear] next to it. Do not return JSON."
            ),
            summary="OCR text bridge for image-only PDFs.",
        )
        raw_text, _response = self._run_gemini_pdf(pdf_path, "gemini-2.5-flash", ocr_prompt)
        text = str(raw_text or "").strip()
        if not text:
            raise UnsupportedModelPayloadError(
                f"Model {model or 'selected model'} requires text fallback for {pdf_path.name}, "
                "but Gemini OCR bridge returned no text."
            )
        self._ocr_text_cache[cache_key] = text[:60000]
        return self._ocr_text_cache[cache_key]


def _combined_usage(responses: list[Any]):
    usages = [extract_token_usage(response) for response in responses]
    return type(usages[0])(
        input_tokens=sum(item.input_tokens for item in usages),
        cached_input_tokens=sum(item.cached_input_tokens for item in usages),
        output_tokens=sum(item.output_tokens for item in usages),
        total_tokens=sum(item.total_tokens for item in usages),
    )
