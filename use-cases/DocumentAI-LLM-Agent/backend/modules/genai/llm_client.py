"""
llm_client.py
-------------
Cliente LLM para SAP Gen AI Hub con soporte multimodal.

Configura el proxy client y expone funciones para invocar
el LLM con texto o con documentos PDF completos (vision/multimodal).

Credenciales desde .env
"""

import base64
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_ENV_PATH = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

logger = logging.getLogger(__name__)

LLM_MODEL_NAME: str = os.getenv("GENAI_MODEL_NAME", "gpt-4o")
LLM_MAX_RETRIES: int = int(os.getenv("GENAI_MAX_RETRIES", "3"))


class LLMClientError(Exception):
    """Error al invocar el LLM via SAP Gen AI Hub."""
    pass


@lru_cache(maxsize=1)
def _get_proxy_client():
    """Inicializa y cachea el proxy client de SAP Gen AI Hub (singleton)."""
    try:
        from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
        client = get_proxy_client("gen-ai-hub")
        logger.info("SAP Gen AI Hub proxy client initialized.")
        return client
    except ImportError:
        raise LLMClientError(
            "Paquete 'generative-ai-hub-sdk' no instalado.\n"
            "Ejecute: pip install generative-ai-hub-sdk"
        )
    except Exception as exc:
        raise LLMClientError(
            f"No se pudo inicializar SAP Gen AI Hub proxy client.\n"
            f"Verifique las variables en .env\nDetalle: {exc}"
        )


def get_llm(model_name: str | None = None):
    """Retorna instancia de ChatOpenAI configurada con SAP Gen AI Hub."""
    try:
        from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
    except ImportError:
        raise LLMClientError("Paquete 'generative-ai-hub-sdk' no instalado.")

    return ChatOpenAI(
        proxy_model_name=model_name or LLM_MODEL_NAME,
        proxy_client=_get_proxy_client(),
    )


def ask_llm(prompt: str, model_name: str | None = None) -> str:
    """
    Invoca el LLM con un prompt de texto simple.

    Returns:
        Respuesta del LLM como string.
    """
    llm = get_llm(model_name)
    last_exc = None

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            logger.debug("Invoking LLM text (attempt %d/%d)...", attempt, LLM_MAX_RETRIES)
            response = llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            last_exc = exc
            logger.warning("Attempt %d/%d failed: %s", attempt, LLM_MAX_RETRIES, exc)

    raise LLMClientError(
        f"LLM did not respond after {LLM_MAX_RETRIES} attempts. Last error: {last_exc}"
    )


# MIME types for image formats supported via image_url
_IMAGE_MIME: dict[str, str] = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".tif":  "image/tiff",
    ".tiff": "image/tiff",
}


def _read_file_base64(file_path: Path) -> str:
    """Read a file and return it as a base64 string."""
    if not file_path.exists():
        raise LLMClientError(f"File not found: {file_path}")
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    logger.info("File loaded: %s (%d KB)", file_path.name, len(data) // 1024)
    return b64


def _build_document_message(prompt: str, doc_path: Path) -> Any:
    """
    Build a HumanMessage for the given document.

    - PDF  → sent as 'file' (native PDF support, GPT-4o+)
    - Image (JPEG, PNG, TIF, TIFF) → sent as 'image_url' with base64

    Args:
        prompt: Instruction text for the LLM.
        doc_path: Path to the document (PDF or image).

    Returns:
        LangChain HumanMessage ready to send.
    """
    try:
        from langchain_core.messages import HumanMessage
    except ImportError:
        raise LLMClientError("Package 'langchain-core' is not installed.")

    ext = doc_path.suffix.lower()
    b64 = _read_file_base64(doc_path)

    if ext == ".pdf":
        # PDF: use the native file content type (GPT-4o and above)
        content: list[dict] = [
            {"type": "text", "text": prompt},
            {
                "type": "file",
                "file": {
                    "filename": doc_path.name,
                    "file_data": f"data:application/pdf;base64,{b64}",
                },
            },
        ]
    elif ext in _IMAGE_MIME:
        # Image: use image_url with base64 data URI
        mime = _IMAGE_MIME[ext]
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{b64}",
                    "detail": "high",
                },
            },
        ]
    else:
        supported = ".pdf, " + ", ".join(_IMAGE_MIME.keys())
        raise LLMClientError(
            f"Unsupported file format for LLM: '{ext}'\n"
            f"Supported: {supported}"
        )

    return HumanMessage(content=content)


# Keep backward-compatible alias
def _build_pdf_message(prompt: str, pdf_path: Path) -> Any:
    """Alias for _build_document_message() — kept for backward compatibility."""
    return _build_document_message(prompt, pdf_path)


def ask_llm_multimodal(
    prompt: str,
    pdf_path: Path,
    model_name: str | None = None,
) -> str:
    """
    Invoke the LLM sending the COMPLETE PDF directly as base64.

    The PDF is sent as a file to the model without prior conversion.
    The model processes it natively (layout, tables, text, images).

    Args:
        prompt: Instruction for the LLM.
        pdf_path: Path to the PDF to process.
        model_name: Model to use.

    Returns:
        LLM response as string.
    """
    llm = get_llm(model_name)
    message = _build_document_message(prompt, pdf_path)

    last_exc = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            logger.info(
                "Invoking LLM with complete PDF (attempt %d/%d): %s",
                attempt, LLM_MAX_RETRIES, pdf_path.name,
            )
            response = llm.invoke([message])
            result = response.content if hasattr(response, "content") else str(response)
            logger.info("LLM responded (%d chars).", len(result))
            return result
        except Exception as exc:
            last_exc = exc
            logger.warning("Attempt %d/%d failed: %s", attempt, LLM_MAX_RETRIES, exc)

    raise LLMClientError(
        f"LLM did not respond after {LLM_MAX_RETRIES} attempts.\n"
        f"Last error: {last_exc}"
    )
