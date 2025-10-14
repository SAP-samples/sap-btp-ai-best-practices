import logging
from pathlib import Path
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends

# Gen AI Hub imports
from gen_ai_hub.proxy.native.amazon.clients import Session
from gen_ai_hub.proxy.native.openai import chat
from gen_ai_hub.proxy.native.google_vertexai.clients import GenerativeModel

from ..models.chat import (
    ChatRequest,
    ChatResponse,
    ChatHistoryRequest,
)
from ..security import get_api_key

# Setup logging
logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(get_api_key)])

# Constants
OPENAI_MODEL = "gpt-4.1"
GEMINI_MODEL = "gemini-2.5-flash"

# Load grounding data once at import time to avoid per-request I/O
GROUNDING_FILE = (
    Path(__file__).resolve().parent.parent / "data" / "business_emails.json"
)
try:
    GROUNDING_DATA = GROUNDING_FILE.read_text(encoding="utf-8")
except Exception:  # Keep server resilient if file is missing
    GROUNDING_DATA = ""


def build_grounded_system_prompt(user_request: str) -> str:
    """Compose a general-purpose grounded prompt over the email dataset.

    This prompt is not tuned for a single task. It flexibly handles queries such as
    invoice/payment status, W9 updates, reroutes, owners of SOPs, and other topics
    present in the dataset, while staying strictly within the provided context.
    """
    header = (
        "You are SAP Accounts Payable Assistant. Answer ONLY using facts in the provided "
        "email dataset. If the dataset does not contain the answer, say so explicitly. "
        "Do not use outside knowledge.\n\n"
        "Dataset: JSON array of emails with: from{name,email}, subject, body{text,html}, "
        "sentDate (ISO), messageId, tags, status.\n\n"
        "General principles:\n"
        "- Never fabricate information.\n"
        "- Extract entities from the request (invoice/PO IDs, vendor names, dates, topics).\n"
        "- Find relevant emails by matching IDs/subjects/tags/body text; prefer the latest by sentDate when multiple match.\n"
        "- Apply time constraints from the request (e.g., 'last 7 days') using sentDate.\n"
        "- If an explicit status exists in an email, report it; if you infer a status, mark it as (inferred).\n"
        "- If nothing relevant is found, reply: 'I could not find this in the provided emails.'\n\n"
        "Output style (Markdown):\n"
        "- Start with a single concise Answer sentence.\n"
        "- If the query requests a list or summary, follow with short bullets or a compact table.\n"
        '- Add an Evidence section with 1–5 bullets: From — Subject (YYYY-MM-DD) — "short quote".\n'
        "- End with IDs recognized: <comma-separated or 'none'>.\n\n"
    )

    dataset_block = f"Email dataset follows:\n```json\n{GROUNDING_DATA}\n```\n\n"
    user_block = f"User request: {user_request}"
    return f"{header}{dataset_block}{user_block}"


@router.post("/openai", response_model=ChatResponse)
async def chat_openai(request: ChatRequest) -> ChatResponse:
    """Chat with GPT-4 via OpenAI integration.

    Provides access to OpenAI's GPT-4.1 model through SAP Gen AI Hub's
    OpenAI integration.

    Args:
        request: Validated ChatRequest containing message and parameters.

    Returns:
        ChatResponse: Standardized response containing:
            - text: Generated response text
            - model: "gpt-4.1"
            - success: True if successful, False otherwise
            - usage: Token usage statistics (prompt_tokens, completion_tokens, total_tokens)
            - error: Error message if request failed

    Raises:
        All exceptions are caught and returned in the error field of ChatResponse.
    """
    try:
        logger.info(f"OpenAI request: {request.message[:50]}...")

        # system_prompt: str = (
        #     "You are a precise and reliable assistant. Using only the provided context, "
        #     "generate a concise and accurate answer relevant to the user's request. "
        #     "Do not infer or fabricate information beyond the given context. "
        #     "If the requested information is not available in the context, clearly state that.\n\n"
        #     f"User request: {request.message}\n\n"
        #     f"Context: {GROUNDING_DATA}"
        # )
        system_prompt: str = build_grounded_system_prompt(request.message)

        messages: list[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.message},
        ]

        response = chat.completions.create(
            messages=messages,
            model=OPENAI_MODEL,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        text: str = response.choices[0].message.content
        usage: Dict[str, int] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        return ChatResponse(text=text, model=OPENAI_MODEL, success=True, usage=usage)

    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        return ChatResponse(text="", model=OPENAI_MODEL, success=False, error=str(e))


@router.post("/openai/history", response_model=ChatResponse)
async def chat_openai_with_history(request: ChatHistoryRequest) -> ChatResponse:
    """Chat with GPT-4 using full conversation history.

    Prepends a system message with grounding context, then forwards the provided
    conversation messages. The last message should be the user's new input.
    """
    try:
        preview = request.messages[-1].content if request.messages else ""
        logger.info(f"OpenAI history request: {preview[:50]}...")

        # system_prompt: str = (
        #     "You are a precise and reliable assistant. Using only the provided context, "
        #     "generate a concise and accurate answer relevant to the user's request. "
        #     "Do not infer or fabricate information beyond the given context. "
        #     "If the requested information is not available in the context, clearly state that.\n\n"
        #     f"Context: {GROUNDING_DATA}"
        # )
        system_prompt: str = build_grounded_system_prompt(preview)

        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        # Append the provided conversation as-is
        messages.extend(
            {"role": m.role, "content": m.content} for m in request.messages
        )

        response = chat.completions.create(
            messages=messages,
            model=OPENAI_MODEL,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        text: str = response.choices[0].message.content
        usage: Dict[str, int] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        return ChatResponse(text=text, model=OPENAI_MODEL, success=True, usage=usage)

    except Exception as e:
        logger.error(f"OpenAI history error: {e}")
        return ChatResponse(text="", model=OPENAI_MODEL, success=False, error=str(e))


@router.post("/gemini", response_model=ChatResponse)
async def chat_gemini(request: ChatRequest) -> ChatResponse:
    """Chat with Google Gemini via Vertex AI integration.

    Provides access to Google's Gemini 2.5 Flash model through SAP Gen AI Hub's
    Vertex AI integration.

    Args:
        request: Validated ChatRequest containing message and parameters.

    Returns:
        ChatResponse: Standardized response containing:
            - text: Generated response text
            - model: "gemini-2.5-flash"
            - success: True if successful, False otherwise
            - usage: Token usage statistics (prompt_tokens, completion_tokens, total_tokens)
            - error: Error message if request failed

    Note:
        Generation config includes temperature, max_output_tokens, top_p=0.95, and top_k=40.

    Raises:
        All exceptions are caught and returned in the error field of ChatResponse.
    """
    try:
        logger.info(f"Gemini request: {request.message[:50]}...")

        model = GenerativeModel(GEMINI_MODEL)

        generation_config: Dict[str, Any] = {
            "temperature": request.temperature,
            "max_output_tokens": request.max_tokens,
            "top_p": 0.95,
            "top_k": 40,
        }

        # grounded_prompt: str = (
        #     "You are a precise and reliable assistant. Using only the provided context, "
        #     "generate a concise and accurate answer relevant to the user's request. "
        #     "Do not infer or fabricate information beyond the given context. "
        #     "If the requested information is not available in the context, clearly state that.\n\n"
        #     f"User request: {request.message}\n\n"
        #     f"Context: {GROUNDING_DATA}"
        # )
        grounded_prompt: str = build_grounded_system_prompt(request.message)

        response = model.generate_content(
            contents=grounded_prompt, generation_config=generation_config
        )

        text: str = response.text
        usage: Dict[str, int] = {
            "prompt_tokens": (
                response.usage_metadata.prompt_token_count
                if hasattr(response, "usage_metadata")
                else 0
            ),
            "completion_tokens": (
                response.usage_metadata.candidates_token_count
                if hasattr(response, "usage_metadata")
                else 0
            ),
            "total_tokens": (
                response.usage_metadata.total_token_count
                if hasattr(response, "usage_metadata")
                else 0
            ),
        }

        return ChatResponse(text=text, model=GEMINI_MODEL, success=True, usage=usage)

    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return ChatResponse(text="", model=GEMINI_MODEL, success=False, error=str(e))
