"""Evidence token estimation and direct-versus-RAG route selection."""

from __future__ import annotations

from enum import StrEnum
from math import ceil

from pydantic import BaseModel

from app.services.document_extractors import ExtractedBlock, ExtractedDocument

DIRECT_REVIEW_TOKEN_LIMIT = 500_000
"""Conservative input-token ceiling for direct ``gpt-5.4`` review calls."""

DEFAULT_PROMPT_OVERHEAD_TOKENS = 20_000
"""Reserved tokens for instructions, schema, citations, and response overhead."""

TOKEN_CHARACTER_RATIO = 3.5
"""Conservative fallback ratio for English-heavy corporate evidence text."""


class EvidenceRoute(StrEnum):
    """Supported worker evidence processing routes.

    Inputs:
        Enum values are selected by ``choose_evidence_route``.

    Outputs:
        String enum values that can be persisted in HANA route metadata.
    """

    DIRECT = "direct"
    RAG = "rag"


class EvidenceRouteDecision(BaseModel):
    """Route decision for one question task evidence pack.

    Inputs:
        Field values describing selected route and token estimates.

    Outputs:
        Validated route decision used by the worker before model calls.
    """

    route: EvidenceRoute
    estimated_input_tokens: int
    evidence_tokens: int
    prompt_tokens: int
    overhead_tokens: int
    direct_token_limit: int
    reason: str


def estimate_text_tokens(text: str) -> int:
    """Estimate token count from text length using a conservative ratio.

    Inputs:
        text: Text to estimate.

    Outputs:
        int: Estimated token count, rounded up and never negative.
    """
    if not text:
        return 0
    return ceil(len(text) / TOKEN_CHARACTER_RATIO)


def _format_block_metadata(block: ExtractedBlock) -> list[str]:
    """Format available source metadata for an extracted block.

    Inputs:
        block: Extracted evidence block with optional location metadata.

    Outputs:
        list[str]: Human-readable metadata lines included in token estimates.
    """
    metadata = [
        f"Block ID: {block.block_id}",
        f"Block type: {block.block_type}",
    ]
    if block.page is not None:
        metadata.append(f"Page: {block.page}")
    if block.section_label:
        metadata.append(f"Section: {block.section_label}")
    if block.sheet_name:
        metadata.append(f"Sheet: {block.sheet_name}")
    if block.table_name:
        metadata.append(f"Table: {block.table_name}")
    if block.row_start is not None or block.row_end is not None:
        row_start = block.row_start if block.row_start is not None else ""
        row_end = block.row_end if block.row_end is not None else ""
        metadata.append(f"Rows: {row_start}-{row_end}")
    return metadata


def _format_evidence_document_for_estimation(document: ExtractedDocument) -> str:
    """Format extracted evidence with direct-review framing for estimation.

    Inputs:
        document: Extracted evidence document to format.

    Outputs:
        str: Text payload including document, block, location, and content fields.
    """
    parts = [
        f"Evidence file: {document.file_name}",
        f"Document type: {document.document_type}",
    ]
    for block in document.blocks:
        parts.extend(
            [
                "",
                *_format_block_metadata(block),
                "Text:",
                block.text,
                "---",
            ]
        )
    return "\n".join(parts)


def estimate_extracted_documents_tokens(
    extracted_documents: list[ExtractedDocument],
) -> int:
    """Estimate token count for extracted evidence documents.

    Inputs:
        extracted_documents: Ordered extracted evidence documents.

    Outputs:
        int: Sum of estimated tokens for formatted evidence document payloads.
    """
    return sum(
        estimate_text_tokens(_format_evidence_document_for_estimation(document))
        for document in extracted_documents
    )


def choose_evidence_route(
    question_prompt_text: str,
    extracted_documents: list[ExtractedDocument],
    direct_token_limit: int = DIRECT_REVIEW_TOKEN_LIMIT,
    prompt_overhead_tokens: int = DEFAULT_PROMPT_OVERHEAD_TOKENS,
) -> EvidenceRouteDecision:
    """Choose direct review or RAG for one question evidence pack.

    Inputs:
        question_prompt_text: Prompt text containing question and answer items.
        extracted_documents: Extracted evidence documents for the task.
        direct_token_limit: Maximum estimated input tokens allowed for direct mode.
        prompt_overhead_tokens: Reserved prompt/schema/citation overhead.

    Outputs:
        EvidenceRouteDecision: Route and token accounting for persistence.
    """
    if direct_token_limit <= 0:
        raise ValueError("direct_token_limit must be positive")
    if prompt_overhead_tokens < 0:
        raise ValueError("prompt_overhead_tokens must be non-negative")

    prompt_tokens = estimate_text_tokens(question_prompt_text)
    evidence_tokens = estimate_extracted_documents_tokens(extracted_documents)
    estimated_input_tokens = prompt_tokens + evidence_tokens + prompt_overhead_tokens
    if estimated_input_tokens <= direct_token_limit:
        return EvidenceRouteDecision(
            route=EvidenceRoute.DIRECT,
            estimated_input_tokens=estimated_input_tokens,
            evidence_tokens=evidence_tokens,
            prompt_tokens=prompt_tokens,
            overhead_tokens=prompt_overhead_tokens,
            direct_token_limit=direct_token_limit,
            reason="estimated_input_within_direct_limit",
        )
    return EvidenceRouteDecision(
        route=EvidenceRoute.RAG,
        estimated_input_tokens=estimated_input_tokens,
        evidence_tokens=evidence_tokens,
        prompt_tokens=prompt_tokens,
        overhead_tokens=prompt_overhead_tokens,
        direct_token_limit=direct_token_limit,
        reason="estimated_input_exceeds_direct_limit",
    )
