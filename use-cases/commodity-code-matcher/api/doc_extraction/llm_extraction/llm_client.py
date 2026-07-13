"""
Minimal wrapper around the SAP Gen AI Hub OpenAI client.

The goal is to keep the rest of the extraction pipeline unaware of the client
details (model name, temperature, token limits, etc.) while still providing
easy hooks to tweak behaviour through environment variables.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

# Load from the single .env file at api/.env
_env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_env_path)
from gen_ai_hub.proxy.native.openai import chat


@dataclass(frozen=True)
class LLMConfig:
    """
    Container for runtime configuration retrieved from the environment.

    The defaults are intentionally conservative to minimise token usage and
    stick to deterministic outputs.
    """

    model: str = os.environ.get("LLM_MODEL", "gpt-4.1")
    temperature: float = float(os.environ.get("LLM_TEMPERATURE", "0"))
    max_tokens: Optional[int] = (
        int(os.environ["LLM_MAX_TOKENS"]) if os.environ.get("LLM_MAX_TOKENS") else None
    )


class LLMClient:
    """
    Thin façade over ``chat.completions.create``.

    The ``complete`` method accepts the standard OpenAI chat message payload and
    returns a tuple containing the raw message content and the full response
    object for detailed usage metrics.
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig()

    def complete(
        self,
        messages: List[Dict[str, Any]],
        *,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Any]:
        """
        Execute a chat completion request and return ``(content, raw_response)``.

        The optional ``response_format`` parameter allows callers to leverage
        the JSON mode API by passing ``{\"type\": \"json_object\"}``, but by
        default we simply parse the assistant's text content.
        """

        kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens is not None:
            kwargs["max_tokens"] = self.config.max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        return content, response

    def complete_json(
        self,
        messages: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Any]:
        """
        Execute a completion request and parse the response as JSON.

        This helper enables the JSON mode API to improve reliability. If the
        response cannot be parsed, a ValueError is raised so the caller can
        decide whether to retry or abort.
        """

        content, response = self.complete(
            messages,
            response_format={"type": "json_object"},
        )
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM response is not valid JSON") from exc
        return parsed, response

