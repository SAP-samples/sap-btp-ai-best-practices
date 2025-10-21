"""
Summary API router for concise field summarization.

Provides a FastAPI endpoint that receives a dictionary of values from the
"Investigación Legal" extraction and returns concise summaries for a predefined
set of verbose fields. It uses the shared `make_llm` helper to call the
OpenAI GPT-4o model through SAP GenAI Hub LangChain proxy.

Target fields and required summarization formats:
- Historial_procesos_judiciales_empresa: Briefly list the judicial processes, nothing else
- Factor_riesgo_empresa: Single word among {"bajo", "medio", "alto"}
- Historial_procesos_judiciales_persona: Briefly list the judicial processes, nothing else
- Factor_riesgo_persona: Single word among {"bajo", "medio", "alto"}
- Confirmacion_Representacion_social_persona: Single word among {"si", "no"}

Input: JSON body with a dictionary of values (arbitrary keys), typically the
fields listed above. Only the provided fields will be summarized.
Output: JSON containing a `summaries` dictionary with the summarized values for
the provided target fields. Non-target fields are ignored.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from fastapi import APIRouter
from pydantic import BaseModel, Field

from extraction.common import make_llm
from langchain.schema import HumanMessage, SystemMessage


# FastAPI router instance
router = APIRouter()


# Fields we support summarizing and the expected output constraints
TARGET_FIELDS: List[str] = [
    "Historial_procesos_judiciales_empresa",
    "Factor_riesgo_empresa",
    "Historial_procesos_judiciales_persona",
    "Factor_riesgo_persona",
    "Confirmacion_Representacion_social_persona",
]


class SummarizeRequest(BaseModel):
    """Request body for the summarization endpoint.

    - values: Arbitrary dictionary with extracted fields as keys.
    Only keys present in TARGET_FIELDS will be summarized and returned.
    """

    values: Dict[str, Any] = Field(default_factory=dict)


def _build_messages(input_values: Dict[str, Any]) -> List[Any]:
    """Construct messages for the LLM call with strict JSON instructions.

    Instruct the model to return ONLY JSON without code fences or commentary,
    mapping only the provided TARGET_FIELDS to their concise summaries.
    """

    # Filter to only the fields we handle and that are actually present
    present_fields = {k: v for k, v in input_values.items() if k in TARGET_FIELDS}

    system_msg = SystemMessage(
        content=(
            "You are a precise summarization assistant. "
            "Summarize the provided Spanish fields into very concise Spanish outputs. "
            "Return ONLY a valid JSON object, with no code fences or explanations. "
            "Keys must exactly match the input keys. "
            "If a target field is not provided in the input, omit it from the output.\n\n"
            "Output constraints per field:\n"
            "- Historial_procesos_judiciales_empresa: Brief list of judicial processes, nothing else.\n"
            "- Factor_riesgo_empresa: Single word strictly in {'bajo','medio','alto'}.\n"
            "- Historial_procesos_judiciales_persona: Brief list of judicial processes, nothing else.\n"
            "- Factor_riesgo_persona: Single word strictly in {'bajo','medio','alto'}.\n"
            "- Confirmacion_Representacion_social_persona: Single word strictly in {'si','no'}.\n"
        )
    )

    human_msg = HumanMessage(
        content=(
            "Summarize ONLY these provided fields into the constrained formats and return ONLY JSON.\n"
            + json.dumps(present_fields, ensure_ascii=False, indent=2, default=str)
        )
    )

    return [system_msg, human_msg]


@router.post("/summarize")
def summarize_fields(req: SummarizeRequest) -> Dict[str, Any]:
    """Summarize verbose Investigación Legal fields using GPT-4o via OpenAI proxy.

    Returns a dictionary with the summarized values for any of the TARGET_FIELDS
    that are present in the input. Non-target fields are ignored.
    """

    # Create the LLM client using the shared helper
    llm = make_llm(
        provider="openai",
        model_name="gpt-4o",
        temperature=0.0,
    )

    # Prepare messages and invoke the model
    messages = _build_messages(req.values)

    try:
        response = llm.invoke(messages)
        content = getattr(response, "content", None) or str(response)

        # Best-effort to parse the returned JSON; if fences exist, strip them
        text = content.strip()
        if text.startswith("```"):
            # Remove potential triple-backtick fences
            text = text.strip("`\n ")
            # If there's a language hint like json, remove up to first newline
            if "\n" in text:
                text = text.split("\n", 1)[1]

        data = json.loads(text)

        # Ensure we return only a dict of summaries
        if not isinstance(data, dict):
            return {"success": False, "error": "Model did not return a JSON object", "raw": content}

        # Keep only target fields that were requested
        summaries = {k: v for k, v in data.items() if k in req.values and k in TARGET_FIELDS}
        return {"success": True, "summaries": summaries}

    except Exception as e:
        return {"success": False, "error": f"LLM invocation or parsing failed: {e}"}


