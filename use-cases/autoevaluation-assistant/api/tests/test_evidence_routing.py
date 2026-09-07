"""Tests for automatic evidence route selection."""

import pytest

from app.services.document_extractors import ExtractedBlock, ExtractedDocument
from app.services.evidence_routing import (
    DIRECT_REVIEW_TOKEN_LIMIT,
    EvidenceRoute,
    choose_evidence_route,
    estimate_extracted_documents_tokens,
    estimate_text_tokens,
)


def _document_with_text(text: str) -> ExtractedDocument:
    """Build a test extracted document with one text block.

    Inputs:
        text: Extracted block text to include.

    Outputs:
        ExtractedDocument: Minimal extracted document for routing tests.
    """
    return ExtractedDocument(
        file_name="evidence.pdf",
        document_type="pdf",
        blocks=[
            ExtractedBlock(
                block_id="pdf-page-0001",
                block_type="page",
                text=text,
                page=1,
            )
        ],
    )


def test_estimate_text_tokens_uses_conservative_character_ratio() -> None:
    """Verify text token estimation rounds up from characters.

    Inputs:
        None. The test sends short text through the estimator.

    Outputs:
        None. Assertions confirm ``ceil(characters / 3.5)`` behavior.
    """
    assert estimate_text_tokens("abcdefgh") == 3


def test_estimate_extracted_documents_tokens_includes_metadata_overhead() -> None:
    """Verify evidence token estimation includes document and block metadata.

    Inputs:
        None. The test creates a block with location metadata and short text.

    Outputs:
        None. Assertions confirm the evidence estimate exceeds raw text tokens.
    """
    document = ExtractedDocument(
        file_name="evidence.pdf",
        document_type="pdf",
        blocks=[
            ExtractedBlock(
                block_id="pdf-page-0001",
                block_type="page",
                text="abcd",
                page=1,
                section_label="Strategy",
                sheet_name="Sheet1",
                table_name="Table 1",
                row_start=2,
                row_end=3,
            )
        ],
    )

    assert estimate_extracted_documents_tokens([document]) > estimate_text_tokens("abcd")


def test_choose_evidence_route_keeps_small_evidence_direct() -> None:
    """Verify small extracted evidence stays on the direct review path.

    Inputs:
        None. The test creates a short extracted evidence document.

    Outputs:
        None. Assertions confirm direct route metadata and token accounting.
    """
    decision = choose_evidence_route(
        question_prompt_text="Question and answer items",
        extracted_documents=[_document_with_text("documented strategy roadmap")],
        prompt_overhead_tokens=100,
        direct_token_limit=1_000,
    )

    assert decision.route == EvidenceRoute.DIRECT
    assert decision.estimated_input_tokens <= 1_000
    assert decision.direct_token_limit == 1_000
    assert decision.reason == "estimated_input_within_direct_limit"


def test_choose_evidence_route_uses_direct_at_exact_limit_boundary() -> None:
    """Verify direct route is selected when estimate equals the limit.

    Inputs:
        None. The test creates prompt-only accounting at an exact limit.

    Outputs:
        None. Assertions confirm the inclusive direct-route boundary.
    """
    decision = choose_evidence_route(
        question_prompt_text="abcdefgh",
        extracted_documents=[],
        prompt_overhead_tokens=7,
        direct_token_limit=10,
    )

    assert decision.route == EvidenceRoute.DIRECT
    assert decision.estimated_input_tokens == 10
    assert decision.reason == "estimated_input_within_direct_limit"


def test_choose_evidence_route_reports_exact_token_accounting() -> None:
    """Verify route decisions expose exact prompt/evidence/overhead accounting.

    Inputs:
        None. The test creates a prompt-only direct routing decision.

    Outputs:
        None. Assertions confirm all token accounting fields exactly.
    """
    decision = choose_evidence_route(
        question_prompt_text="abcdefgh",
        extracted_documents=[],
        prompt_overhead_tokens=7,
        direct_token_limit=100,
    )

    assert decision.prompt_tokens == 3
    assert decision.evidence_tokens == 0
    assert decision.overhead_tokens == 7
    assert decision.estimated_input_tokens == 10


@pytest.mark.parametrize("direct_token_limit", [0, -1])
def test_choose_evidence_route_rejects_non_positive_direct_limit(
    direct_token_limit: int,
) -> None:
    """Verify direct token limit validation rejects zero and negative values.

    Inputs:
        direct_token_limit: Invalid direct route token ceiling.

    Outputs:
        None. Assertions confirm a clear ``ValueError`` is raised.
    """
    with pytest.raises(ValueError, match="direct_token_limit must be positive"):
        choose_evidence_route(
            question_prompt_text="Question and answer items",
            extracted_documents=[],
            direct_token_limit=direct_token_limit,
        )


def test_choose_evidence_route_rejects_negative_prompt_overhead() -> None:
    """Verify prompt overhead validation rejects negative values.

    Inputs:
        None. The test sends an invalid prompt overhead.

    Outputs:
        None. Assertions confirm a clear ``ValueError`` is raised.
    """
    with pytest.raises(ValueError, match="prompt_overhead_tokens must be non-negative"):
        choose_evidence_route(
            question_prompt_text="Question and answer items",
            extracted_documents=[],
            prompt_overhead_tokens=-1,
        )


def test_choose_evidence_route_sends_oversized_evidence_to_rag() -> None:
    """Verify oversized extracted evidence is routed to RAG before model calls.

    Inputs:
        None. The test creates a synthetic evidence block above the limit.

    Outputs:
        None. Assertions confirm RAG route metadata and token accounting.
    """
    oversized_text = "a" * int(DIRECT_REVIEW_TOKEN_LIMIT * 3.5)

    decision = choose_evidence_route(
        question_prompt_text="Question and answer items",
        extracted_documents=[_document_with_text(oversized_text)],
    )

    assert decision.route == EvidenceRoute.RAG
    assert decision.estimated_input_tokens > DIRECT_REVIEW_TOKEN_LIMIT
    assert decision.reason == "estimated_input_exceeds_direct_limit"
