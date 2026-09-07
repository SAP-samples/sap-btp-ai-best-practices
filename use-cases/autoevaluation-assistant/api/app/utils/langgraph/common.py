"""Native OpenAI-compatible utilities for LangGraph-backed workflows.

The assessment document review PoC uses OpenAI-compatible deployments through SAP Gen
AI Hub. This module intentionally avoids provider-specific LangChain wrappers so
the backend does not import Bedrock, Vertex/Gemini, or other provider packages at
application startup. It resolves the native Responses and Chat Completions
parser surfaces used by the review client.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from dotenv import load_dotenv

load_dotenv()

ReasoningEffort = Literal["none", "low", "medium", "high"]
SUPPORTED_REASONING_EFFORTS: set[str] = {"none", "low", "medium", "high"}
DEFAULT_REASONING_EFFORT: ReasoningEffort = "low"


def validate_reasoning_effort(value: str) -> ReasoningEffort:
    """Validate a configured OpenAI Responses reasoning effort value.

    Inputs:
        value: Reasoning effort string from configuration or constructor input.

    Outputs:
        ReasoningEffort: Normalized lower-case effort value accepted by
        OpenAI-compatible Responses requests.

    Raises:
        ValueError: Raised when the value is not one of ``none``, ``low``,
        ``medium``, or ``high``.
    """

    normalized = value.strip().lower()
    if normalized not in SUPPORTED_REASONING_EFFORTS:
        allowed_values = ", ".join(sorted(SUPPORTED_REASONING_EFFORTS))
        raise ValueError(
            "GENAI_REASONING_EFFORT must be one of "
            f"{allowed_values}; received {value!r}."
        )
    return normalized  # type: ignore[return-value]


def get_reasoning_effort(
    env_var: str = "GENAI_REASONING_EFFORT",
    default: ReasoningEffort = DEFAULT_REASONING_EFFORT,
) -> ReasoningEffort:
    """Read the configured reasoning effort for Responses API calls.

    Inputs:
        env_var: Environment variable name to read.
        default: Effort value used when ``env_var`` is absent or empty.

    Outputs:
        ReasoningEffort: Validated reasoning effort. Defaults to ``low`` so
        ``gpt-5.4`` requests stay fast for the PoC.
    """

    configured_value = os.getenv(env_var, default) or default
    return validate_reasoning_effort(configured_value)


def resolve_openai_responses_api(openai_module: Any) -> Any | None:
    """Resolve a Responses API object from supported Gen AI Hub SDK shapes.

    Inputs:
        openai_module: Imported ``gen_ai_hub.proxy.native.openai`` module or a
        test double with equivalent attributes.

    Outputs:
        Any | None: Object exposing callable ``parse`` when found, otherwise
        ``None``. SDK 6.10.0+ exposes module-level ``responses``; older SDKs may
        only expose ``OpenAI().responses``.
    """

    module_responses = getattr(openai_module, "responses", None)
    if callable(getattr(module_responses, "parse", None)):
        return module_responses

    openai_factory = getattr(openai_module, "OpenAI", None)
    if not callable(openai_factory):
        return None

    client = openai_factory()
    client_responses = getattr(client, "responses", None)
    if callable(getattr(client_responses, "parse", None)):
        return client_responses
    return None


def resolve_openai_chat_completions_api(openai_module: Any) -> Any | None:
    """Resolve a Chat Completions API object from supported SDK shapes.

    Inputs:
        openai_module: Imported ``gen_ai_hub.proxy.native.openai`` module or a
        test double with equivalent attributes.

    Outputs:
        Any | None: Object exposing callable ``parse`` when found, otherwise
        ``None``. The native SDK may expose chat completions either as
        module-level ``chat.completions`` or as
        ``OpenAI().chat.completions``.
    """

    module_chat = getattr(openai_module, "chat", None)
    module_chat_completions = getattr(module_chat, "completions", None)
    if callable(getattr(module_chat_completions, "parse", None)):
        return module_chat_completions

    openai_factory = getattr(openai_module, "OpenAI", None)
    if not callable(openai_factory):
        return None

    client = openai_factory()
    client_chat = getattr(client, "chat", None)
    client_chat_completions = getattr(client_chat, "completions", None)
    if callable(getattr(client_chat_completions, "parse", None)):
        return client_chat_completions
    return None


def make_openai_responses_api() -> Any:
    """Create the native SAP Gen AI Hub OpenAI Responses API object.

    Inputs:
        None. SAP AI Core credentials are loaded from ``api/.env`` or the
        process environment before importing the SDK module.

    Outputs:
        Any: Responses API object exposing ``create`` and ``parse``.

    Raises:
        RuntimeError: Raised when the SDK is missing or does not expose a usable
        Responses API object.
    """

    try:
        import gen_ai_hub.proxy.native.openai as openai_module
    except ImportError as exc:
        raise RuntimeError(
            "SAP Gen AI Hub OpenAI-compatible Responses API is unavailable. "
            "Install sap-ai-sdk-gen>=6.10.0 and configure SAP AI Core "
            "credentials before creating a native Responses client."
        ) from exc

    responses_api = resolve_openai_responses_api(openai_module)
    if responses_api is None:
        raise RuntimeError(
            "SAP Gen AI Hub Responses API parse unavailable. Use "
            "sap-ai-sdk-gen>=6.10.0 so module-level OpenAI Responses support is "
            "available."
        )
    return responses_api


def make_openai_chat_completions_api() -> Any:
    """Create the native SAP Gen AI Hub OpenAI Chat Completions API object.

    Inputs:
        None. SAP AI Core credentials are loaded from ``api/.env`` or the
        process environment before importing the SDK module.

    Outputs:
        Any: Chat Completions object exposing ``parse`` for structured output.

    Raises:
        RuntimeError: Raised when the SDK is missing or does not expose a usable
        Chat Completions parser.
    """

    try:
        import gen_ai_hub.proxy.native.openai as openai_module
    except ImportError as exc:
        raise RuntimeError(
            "SAP Gen AI Hub OpenAI-compatible Chat Completions API is "
            "unavailable. Install sap-ai-sdk-gen>=6.10.0 and configure SAP AI "
            "Core credentials before creating a native chat completions client."
        ) from exc

    chat_completions_api = resolve_openai_chat_completions_api(openai_module)
    if chat_completions_api is None:
        raise RuntimeError(
            "SAP Gen AI Hub Chat Completions API parse unavailable. Use "
            "sap-ai-sdk-gen>=6.10.0 so native OpenAI Chat Completions "
            "structured parsing is available."
        )
    return chat_completions_api


def save_graph_mermaid_png(app: Any, output_path: str) -> bool:
    """Render and save a Mermaid PNG for a compiled LangGraph application.

    Inputs:
        app: Compiled LangGraph application returned by ``builder.compile()``.
        output_path: Absolute path where the PNG should be written.

    Outputs:
        bool: ``True`` when rendering succeeds, otherwise ``False``. Rendering
        is best-effort and never required for the runtime review path.
    """

    try:
        from langchain_core.runnables.graph import MermaidDrawMethod  # type: ignore

        graph = app.get_graph()
        png_bytes = graph.draw_mermaid_png(draw_method=MermaidDrawMethod.API)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as output_file:
            output_file.write(png_bytes)
        return True
    except Exception:
        return False


__all__ = [
    "DEFAULT_REASONING_EFFORT",
    "ReasoningEffort",
    "get_reasoning_effort",
    "make_openai_chat_completions_api",
    "make_openai_responses_api",
    "resolve_openai_chat_completions_api",
    "resolve_openai_responses_api",
    "save_graph_mermaid_png",
    "validate_reasoning_effort",
]
