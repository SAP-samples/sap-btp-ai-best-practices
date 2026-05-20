"""LLM invocation abstraction for the metal-composition workflow."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from ..config import MetalCompositionSettings
from .normalize import _extract_json_payload, _response_text
from .token_usage import TokenUsageRecorder, normalize_token_usage

logger = logging.getLogger(__name__)


class _WrappedMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _WrappedChoice:
    def __init__(self, content: str) -> None:
        self.message = _WrappedMessage(content)


class _WrappedResponse:
    def __init__(
        self,
        content: str,
        *,
        usage: Any = None,
        citations: Optional[List[Any]] = None,
        search_results: Optional[List[Any]] = None,
    ) -> None:
        self.choices = [_WrappedChoice(content)]
        self.usage = usage
        self.citations = list(citations or [])
        self.search_results = list(search_results or [])


class LLMClient:
    """Thin wrapper around Gen AI Hub LLM invocation."""

    def __init__(self, settings: MetalCompositionSettings) -> None:
        self.settings = settings

    def load_runtime_env(self) -> None:
        load_dotenv(self.settings.api_env_path, override=False)

    def make_openai_chat(self, model_name: str):
        self.load_runtime_env()
        from gen_ai_hub.proxy.langchain.openai import ChatOpenAI

        return ChatOpenAI(proxy_model_name=model_name)

    def invoke_json_chat(
        self,
        model_name: str,
        prompt: str,
        *,
        phase: Optional[str] = None,
        task: Optional[str] = None,
        usage_recorder: Optional[TokenUsageRecorder] = None,
    ) -> Dict[str, Any]:
        llm = self.make_openai_chat(model_name)
        response = llm.invoke(prompt)
        self._record_usage(
            model_name=model_name,
            response=response,
            phase=phase,
            task=task,
            usage_recorder=usage_recorder,
        )
        return _extract_json_payload(_response_text(response))

    def invoke_native_chat_completion(
        self,
        *,
        model_name: str,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        thinking_type: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        phase: Optional[str] = None,
        task: Optional[str] = None,
        usage_recorder: Optional[TokenUsageRecorder] = None,
    ):
        self.load_runtime_env()

        normalized_model_name = model_name.strip().lower()
        is_gpt5 = normalized_model_name.startswith("gpt-5")
        is_anthropic = normalized_model_name.startswith("anthropic--")
        is_gemini = normalized_model_name.startswith("gemini-")

        if is_anthropic:
            return self.invoke_bedrock_converse(
                model_name=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                thinking_type=thinking_type,
                thinking_budget=thinking_budget,
                phase=phase,
                task=task,
                usage_recorder=usage_recorder,
            )

        if is_gemini:
            return self.invoke_google_generate_content(
                model_name=model_name,
                messages=messages,
                phase=phase,
                task=task,
                usage_recorder=usage_recorder,
            )

        from gen_ai_hub.proxy.native.openai import chat

        kwargs: Dict[str, Any] = {"model": model_name, "messages": messages}
        if is_gpt5:
            kwargs["reasoning_effort"] = reasoning_effort or "low"
        else:
            if temperature is not None:
                kwargs["temperature"] = temperature

        token_limit_parameter: Optional[str] = None
        if max_tokens is not None and not is_gpt5:
            token_limit_parameter = "max_tokens"
            kwargs[token_limit_parameter] = max_tokens

        try:
            response = chat.completions.create(**kwargs)
        except Exception as exc:
            if max_tokens is None or token_limit_parameter is None:
                raise

            error_text = str(exc)
            retry_parameter: Optional[str] = None
            if token_limit_parameter == "max_tokens" and "Unsupported parameter: 'max_tokens'" in error_text:
                retry_parameter = "max_completion_tokens"
            elif (
                token_limit_parameter == "max_completion_tokens"
                and "Unsupported parameter: 'max_completion_tokens'" in error_text
            ):
                retry_parameter = "max_tokens"

            if retry_parameter is None:
                raise

            logger.info(
                "Retrying native chat completion for model=%s with %s after unsupported parameter error",
                model_name,
                retry_parameter,
            )
            kwargs = dict(kwargs)
            kwargs.pop(token_limit_parameter, None)
            kwargs[retry_parameter] = max_tokens
            token_limit_parameter = retry_parameter
            response = chat.completions.create(**kwargs)

        self._record_usage(
            model_name=model_name,
            response=response,
            phase=phase,
            task=task,
            usage_recorder=usage_recorder,
        )
        return response

    def invoke_google_generate_content(
        self,
        *,
        model_name: str,
        messages: List[Dict[str, Any]],
        phase: Optional[str] = None,
        task: Optional[str] = None,
        usage_recorder: Optional[TokenUsageRecorder] = None,
    ):
        from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
        from gen_ai_hub.proxy.native.google_genai.clients import Client

        proxy_client = get_proxy_client("gen-ai-hub")
        client = Client(proxy_client=proxy_client)
        contents, system_instruction = self._convert_messages_for_google_genai(messages)

        kwargs: Dict[str, Any] = {
            "model": model_name,
            "contents": contents,
        }
        if system_instruction:
            kwargs["config"] = {"system_instruction": system_instruction}

        response = client.models.generate_content(**kwargs)
        wrapped_response = self._wrap_google_genai_response(response)
        self._record_usage(
            model_name=model_name,
            response=wrapped_response,
            phase=phase,
            task=task,
            usage_recorder=usage_recorder,
        )
        return wrapped_response

    def invoke_bedrock_converse(
        self,
        *,
        model_name: str,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        thinking_type: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        phase: Optional[str] = None,
        task: Optional[str] = None,
        usage_recorder: Optional[TokenUsageRecorder] = None,
    ):
        from gen_ai_hub.proxy.native.amazon.clients import Session

        bedrock = Session().client(model_name=model_name)
        bedrock_messages, system_prompt = self._convert_messages_for_bedrock(messages)

        inference_config: Dict[str, Any] = {}
        if max_tokens is not None:
            inference_config["maxTokens"] = max_tokens

        effective_type = (thinking_type or "disabled").strip().lower()
        additional_fields: Dict[str, Any] = {}
        if effective_type == "adaptive":
            additional_fields["thinking"] = {"type": "adaptive"}
        elif effective_type == "enabled":
            budget = thinking_budget or 4000
            additional_fields["thinking"] = {"type": "enabled", "budgetTokens": budget}
        else:
            additional_fields["thinking"] = {"type": "disabled"}

        if temperature is not None and effective_type == "disabled":
            inference_config["temperature"] = temperature

        converse_kwargs: Dict[str, Any] = {
            "messages": bedrock_messages,
        }
        if inference_config:
            converse_kwargs["inferenceConfig"] = inference_config
        if additional_fields:
            converse_kwargs["additionalModelRequestFields"] = additional_fields
        if system_prompt:
            converse_kwargs["system"] = [{"text": system_prompt}]

        bedrock_response = bedrock.converse(**converse_kwargs)
        wrapped_response = self._wrap_bedrock_response(bedrock_response)
        self._record_usage(
            model_name=model_name,
            response=wrapped_response,
            phase=phase,
            task=task,
            usage_recorder=usage_recorder,
        )
        return wrapped_response

    @staticmethod
    def _extract_usage(response: Any) -> Any:
        if response is None:
            return None
        direct_usage = getattr(response, "usage", None)
        if direct_usage is not None:
            return direct_usage

        usage_metadata = getattr(response, "usage_metadata", None)
        if usage_metadata is not None:
            return usage_metadata

        response_metadata = getattr(response, "response_metadata", None)
        if isinstance(response_metadata, dict):
            for key in ("token_usage", "usage", "usage_metadata"):
                if key in response_metadata:
                    return response_metadata.get(key)
        return None

    @classmethod
    def _record_usage(
        cls,
        *,
        model_name: str,
        response: Any,
        phase: Optional[str],
        task: Optional[str],
        usage_recorder: Optional[TokenUsageRecorder],
    ) -> None:
        if usage_recorder is None or not phase or not task:
            return
        usage = cls._extract_usage(response)
        normalized = normalize_token_usage(usage)
        if not any(value is not None for value in normalized.values()):
            usage = None
        usage_recorder.record(
            phase=phase,
            task=task,
            model=model_name,
            usage=usage,
        )

    @staticmethod
    def _convert_messages_for_bedrock(
        messages: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], str]:
        system_parts: List[str] = []
        bedrock_messages: List[Dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", "")).strip().lower()
            content = message.get("content", "")
            if role == "system":
                system_parts.append(str(content) if isinstance(content, str) else json.dumps(content))
                continue
            if isinstance(content, str):
                bedrock_content = [{"text": content}]
            elif isinstance(content, list):
                bedrock_content = []
                for block in content:
                    if isinstance(block, str):
                        bedrock_content.append({"text": block})
                    elif isinstance(block, dict):
                        block_type = block.get("type", "")
                        if block_type == "text":
                            bedrock_content.append({"text": block.get("text", "")})
                        elif block_type == "image_url":
                            image_url = block.get("image_url", {}).get("url", "")
                            if image_url.startswith("data:"):
                                media_type, _, b64_data = image_url.partition(";base64,")
                                media_type = media_type.replace("data:", "")
                                bedrock_content.append(
                                    {
                                        "image": {
                                            "format": media_type.split("/")[-1] if "/" in media_type else "png",
                                            "source": {"bytes": b64_data},
                                        }
                                    }
                                )
                            else:
                                bedrock_content.append({"text": f"[image: {image_url}]"})
                        else:
                            bedrock_content.append({"text": json.dumps(block)})
            else:
                bedrock_content = [{"text": str(content)}]
            bedrock_messages.append({"role": role, "content": bedrock_content})
        system_prompt = "\n\n".join(system_parts).strip()
        return bedrock_messages, system_prompt

    @staticmethod
    def _convert_messages_for_google_genai(
        messages: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], str]:
        system_parts: List[str] = []
        google_contents: List[Dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", "")).strip().lower()
            content = message.get("content", "")
            if role == "system":
                system_parts.append(str(content) if isinstance(content, str) else json.dumps(content))
                continue

            google_role = "model" if role == "assistant" else "user"
            google_parts: List[Dict[str, Any]] = []
            if isinstance(content, str):
                google_parts = [{"text": content}]
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, str):
                        google_parts.append({"text": block})
                    elif isinstance(block, dict):
                        block_type = block.get("type", "")
                        if block_type == "text":
                            google_parts.append({"text": block.get("text", "")})
                        elif block_type == "image_url":
                            image_url = block.get("image_url", {}).get("url", "")
                            if image_url.startswith("data:") and ";base64," in image_url:
                                media_type, _, b64_data = image_url.partition(";base64,")
                                google_parts.append(
                                    {
                                        "inline_data": {
                                            "mime_type": media_type.replace("data:", "") or "application/octet-stream",
                                            "data": b64_data,
                                        }
                                    }
                                )
                            else:
                                google_parts.append({"text": f"[image: {image_url}]"})
                        else:
                            google_parts.append({"text": json.dumps(block)})
            else:
                google_parts = [{"text": str(content)}]

            if not google_parts:
                google_parts = [{"text": ""}]
            google_contents.append({"role": google_role, "parts": google_parts})

        system_prompt = "\n\n".join(system_parts).strip()
        return google_contents, system_prompt

    @staticmethod
    def _wrap_bedrock_response(bedrock_response: Dict[str, Any]):
        output_message = bedrock_response.get("output", {}).get("message", {})
        text_parts: List[str] = []
        for block in output_message.get("content", []):
            if isinstance(block, dict):
                if "text" in block:
                    text_parts.append(block["text"])
        final_text = "\n".join(text_parts).strip()
        return _WrappedResponse(final_text, usage=bedrock_response.get("usage"))

    @staticmethod
    def _wrap_google_genai_response(response: Any):
        final_text = getattr(response, "text", None)
        if not isinstance(final_text, str):
            final_text = _response_text(response)
        usage = None
        usage_metadata = getattr(response, "usage_metadata", None)
        if usage_metadata is not None:
            usage = usage_metadata
        elif hasattr(response, "model_dump"):
            try:
                payload = response.model_dump()
            except Exception:  # noqa: BLE001
                payload = None
            if isinstance(payload, dict):
                usage = payload.get("usage_metadata") or payload.get("usageMetadata")
        return _WrappedResponse(final_text.strip(), usage=usage)
