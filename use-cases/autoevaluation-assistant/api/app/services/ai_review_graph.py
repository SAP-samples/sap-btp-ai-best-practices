"""LangGraph workflow for reviewing one assessment question.

The graph assembles framework context and evidence files into Responses API
input messages, calls the structured review client, and returns a validated
``QuestionReviewResult``. Tests inject a fake review client so this module does
not require live SAP Gen AI Hub access during unit verification.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.models.ai_review import QuestionReviewResult
from app.models.assessment import AssessmentQuestion
from app.models.language import DEFAULT_LANGUAGE
from app.services.document_extractors import (
    ExtractedDocument,
    extract_pdf,
    extract_docx,
    extract_eml,
    extract_xlsx,
)
from app.services.evidence_indexing import (
    EvidenceChunk,
    chunk_extracted_documents,
    embed_evidence_chunks,
)
from app.services.evidence_rag import run_retrieval_assessment_loop
from app.services.evidence_retrieval import (
    RetrievedEvidenceChunk,
    build_initial_retrieval_queries,
    retrieve_evidence_candidates,
)
from app.services.evidence_routing import (
    EvidenceRoute,
    choose_evidence_route,
    estimate_text_tokens,
)
from app.services.genai_responses import GenAiReviewClient, ReviewContextLimitError
from app.services.joule_knowledge_importer import (
    DEFAULT_EMBEDDING_MODEL,
    GenAiHubEmbeddingClient,
)

logger = logging.getLogger(__name__)


SELECTABLE_AI_DECISIONS = {"keep_selected", "select", "low_confidence"}
"""AI decisions that should appear in verified selected answer IDs."""

IMAGE_MIME_TYPES = {"image/png", "image/jpeg"}
"""Image MIME types accepted as direct Responses API visual evidence."""

IMAGE_SUFFIX_MIME_TYPES = {
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
}
"""Supported image filename suffixes and their Responses API MIME types."""


DEFAULT_EVIDENCE_EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL
"""Embedding deployment used when the caller does not inject one explicitly."""


class QuestionReviewInput(BaseModel):
    """Input required to review one assessment question.

    Inputs:
        Field values identifying the task, framework question, current selected
        answer item IDs, and evidence attachment paths.

    Outputs:
        A validated input object that the LangGraph workflow can transform into
        Responses API messages.

    Attributes:
        task_id: AI review task identifier being evaluated.
        language: Supported language code for generated AI review text.
        question: Framework question, explanation, and answer item catalog.
        current_selected_answer_item_ids: Current user-selected answer item IDs.
        attachment_paths: Local evidence files available to the review graph.
    """

    task_id: str
    language: str = DEFAULT_LANGUAGE
    question: AssessmentQuestion
    current_selected_answer_item_ids: list[str] = Field(default_factory=list)
    attachment_paths: list[Path] = Field(default_factory=list)


class QuestionReviewState(TypedDict, total=False):
    """LangGraph state for the per-question review workflow.

    Inputs:
        The initial graph state contains ``review_input``.

    Outputs:
        Later nodes add assembled ``input_messages`` and the final ``result``.
    """

    review_input: QuestionReviewInput
    input_messages: list[dict[str, Any]]
    result: QuestionReviewResult


def _mime_type(path: Path) -> str:
    """Infer a stable MIME type for an evidence file.

    Inputs:
        path: Local evidence file path.

    Outputs:
        str: Guessed MIME type, or ``application/octet-stream`` when the type
        cannot be inferred from the filename.
    """

    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _image_mime_type(path: Path, guessed_mime_type: str) -> str | None:
    """Return a supported image MIME type for an evidence file when possible.

    Inputs:
        path: Local evidence file path whose suffix may identify the image
            format.
        guessed_mime_type: MIME type already inferred from the file name.

    Outputs:
        str | None: ``image/png`` or ``image/jpeg`` when the file is supported,
        otherwise ``None``.
    """

    if guessed_mime_type in IMAGE_MIME_TYPES:
        return guessed_mime_type
    return IMAGE_SUFFIX_MIME_TYPES.get(path.suffix.lower())


def _format_extracted_document(extracted: ExtractedDocument) -> str:
    """Format extracted Office document blocks for model input.

    Inputs:
        extracted: Traceable document extraction result from a DOCX or XLSX
        source file.

    Outputs:
        str: Plain text containing file metadata, block IDs, locations, and
        extracted text for prompt inclusion.
    """

    lines = [
        f"Evidence file: {extracted.file_name}",
        f"Document type: {extracted.document_type}",
    ]
    for block in extracted.blocks:
        location_parts = [
            part
            for part in [
                f"section={block.section_label}" if block.section_label else None,
                f"sheet={block.sheet_name}" if block.sheet_name else None,
                f"table={block.table_name}" if block.table_name else None,
                (
                    f"rows={block.row_start}-{block.row_end}"
                    if block.row_start is not None and block.row_end is not None
                    else None
                ),
            ]
            if part is not None
        ]
        location = f" ({', '.join(location_parts)})" if location_parts else ""
        lines.append(f"[{block.block_id}] {block.block_type}{location}:")
        lines.append(block.text)

    if not extracted.blocks:
        lines.append("No extractable text blocks found.")

    return "\n".join(lines)


def _supports_text_extraction(path: Path) -> bool:
    """Return whether an attachment should be extracted for routed text review.

    Inputs:
        path: Local evidence file path.

    Outputs:
        bool: ``True`` for PDF, DOCX, XLSX, XLSM, and EML files.
    """

    suffix = path.suffix.lower()
    mime_type = _mime_type(path)
    return (
        suffix == ".pdf"
        or mime_type == "application/pdf"
        or suffix == ".docx"
        or suffix in {".xlsx", ".xlsm"}
        or suffix == ".eml"
    )


def extract_evidence_files(paths: list[Path]) -> list[ExtractedDocument]:
    """Extract text-bearing evidence files for routing and review.

    Inputs:
        paths: Temporary attachment paths for one question task.

    Outputs:
        list[ExtractedDocument]: Extracted document payloads in attachment
        order for attachments that support text extraction.

    Raises:
        ValueError: Raised when a non-image unsupported file type is present.
    """

    extracted_documents: list[ExtractedDocument] = []
    for path in paths:
        suffix = path.suffix.lower()
        mime_type = _mime_type(path)
        if suffix == ".pdf" or mime_type == "application/pdf":
            extracted_documents.append(extract_pdf(path))
        elif suffix == ".docx":
            extracted_documents.append(extract_docx(path))
        elif suffix in {".xlsx", ".xlsm"}:
            extracted_documents.append(extract_xlsx(path))
        elif suffix == ".eml":
            extracted_documents.append(extract_eml(path))
        elif _image_mime_type(path, mime_type) is not None:
            continue
        else:
            raise ValueError(
                f"Unsupported evidence file MIME type {mime_type}: {path.name}"
            )
    return extracted_documents


def _image_blocks(path: Path, mime_type: str) -> list[dict[str, Any]]:
    """Build Responses API content blocks for one image evidence file.

    Inputs:
        path: Local path to a PNG or JPEG evidence attachment.
        mime_type: Supported image MIME type for the data URL.

    Outputs:
        list[dict[str, Any]]: Metadata text block followed by an
        ``input_image`` block using a base64 data URL. The metadata block gives
        the model a filename it can cite because Responses image blocks do not
        carry a filename field.
    """

    encoded_image = base64.b64encode(path.read_bytes()).decode("ascii")
    return [
        {
            "type": "input_text",
            "text": "\n".join(
                [
                    f"Evidence image file: {path.name}",
                    "Document type: image",
                    f"MIME type: {mime_type}",
                    (
                        "The next content block contains the image pixels for "
                        "visual evidence review. Cite this filename when using "
                        "visible text, logos, forms, signs, tables, charts, or "
                        "other image content as evidence."
                    ),
                ]
            ),
        },
        {
            "type": "input_image",
            "image_url": f"data:{mime_type};base64,{encoded_image}",
        },
    ]


def _file_blocks(path: Path) -> list[dict[str, Any]]:
    """Build Responses API content blocks for an evidence file.

    Inputs:
        path: Local path to a PDF, DOCX, XLSX, EML, PNG, or JPEG evidence attachment.

    Outputs:
        list[dict[str, Any]]: ``input_file`` block for PDFs, ``input_text``
        block containing extracted Office/email content for DOCX/XLSX/EML files, or
        metadata plus ``input_image`` blocks for image evidence.

    Raises:
        ValueError: Raised when the file type is unsupported. The message
        includes the inferred MIME type to simplify debugging upload issues.
    """

    suffix = path.suffix.lower()
    mime_type = _mime_type(path)
    if suffix == ".pdf" or mime_type == "application/pdf":
        encoded_file = base64.b64encode(path.read_bytes()).decode("ascii")
        return [
            {
                "type": "input_file",
                "filename": path.name,
                "file_data": f"data:application/pdf;base64,{encoded_file}",
            }
        ]

    if suffix == ".docx":
        return [
            {
                "type": "input_text",
                "text": _format_extracted_document(extract_docx(path)),
            }
        ]

    if suffix in {".xlsx", ".xlsm"}:
        return [
            {
                "type": "input_text",
                "text": _format_extracted_document(extract_xlsx(path)),
            }
        ]

    if suffix == ".eml":
        return [
            {
                "type": "input_text",
                "text": _format_extracted_document(extract_eml(path)),
            }
        ]

    image_mime_type = _image_mime_type(path, mime_type)
    if image_mime_type is not None:
        return _image_blocks(path, image_mime_type)

    raise ValueError(f"Unsupported evidence file MIME type {mime_type}: {path.name}")


def _question_prompt_text(review_input: QuestionReviewInput) -> str:
    """Build the text portion of the per-question review prompt.

    Inputs:
        review_input: Validated question review input containing the framework
        question and current answer selections.

    Outputs:
        str: Prompt text with question metadata and answer item lines.
    """

    question = review_input.question
    answer_lines = [
        (
            f"- {item.answer_item_id} | level={item.level} | "
            f"item_index={item.item_index} | {item.text}"
        )
        for item in question.answer_items
    ]
    return "\n".join(
        [
            (
                "Review this assessment question using only the "
                "provided evidence."
            ),
            f"Task ID: {review_input.task_id}",
            f"Question ID: {question.question_id}",
            f"Dimension: {question.dimension}",
            f"Section: {question.section}",
            f"Question: {question.question}",
            f"Explanation: {question.explanation or 'None provided.'}",
            (
                "Current selected answer item IDs: "
                f"{review_input.current_selected_answer_item_ids}"
            ),
            "Answer items:",
            *answer_lines,
            "Evidence attachments follow in subsequent content blocks.",
        ]
    )


def _build_prompt(review_input: QuestionReviewInput) -> list[dict[str, Any]]:
    """Assemble Responses API input messages for one question review.

    Inputs:
        review_input: Validated task, question, current selections, and evidence
        file paths.

    Outputs:
        list[dict[str, Any]]: One user message containing text and file content
        blocks ready for ``GenAiReviewClient.review_question``.
    """

    content_blocks: list[dict[str, Any]] = [
        {"type": "input_text", "text": _question_prompt_text(review_input)}
    ]
    for path in review_input.attachment_paths:
        content_blocks.extend(_file_blocks(path))
    return [{"role": "user", "content": content_blocks}]


def _format_extracted_documents_for_review(
    extracted_documents: list[ExtractedDocument],
) -> list[dict[str, Any]]:
    """Build direct review text blocks from extracted documents.

    Inputs:
        extracted_documents: Extracted evidence documents for the review task.

    Outputs:
        list[dict[str, Any]]: Responses ``input_text`` blocks carrying the
        extracted evidence payloads.
    """

    return [
        {"type": "input_text", "text": _format_extracted_document(document)}
        for document in extracted_documents
    ]


def _build_direct_prompt_from_extracted(
    review_input: QuestionReviewInput,
    extracted_documents: list[ExtractedDocument],
) -> list[dict[str, Any]]:
    """Build a direct review prompt from extracted text and image files.

    Inputs:
        review_input: Validated question task input.
        extracted_documents: Extracted text-bearing evidence documents.

    Outputs:
        list[dict[str, Any]]: One user message suitable for direct review.
    """

    content_blocks: list[dict[str, Any]] = [
        {"type": "input_text", "text": _question_prompt_text(review_input)}
    ]
    content_blocks.extend(_format_extracted_documents_for_review(extracted_documents))
    content_blocks.extend(
        _raw_pdf_blocks_for_sparse_extractions(
            review_input=review_input,
            extracted_documents=extracted_documents,
        )
    )
    for path in review_input.attachment_paths:
        image_mime_type = _image_mime_type(path, _mime_type(path))
        if image_mime_type is not None:
            content_blocks.extend(_image_blocks(path, image_mime_type))
    return [{"role": "user", "content": content_blocks}]


def _pdf_extraction_needs_raw_direct_evidence(
    extracted_document: ExtractedDocument,
) -> bool:
    """Return whether direct review should also receive the original PDF.

    Inputs:
        extracted_document: Extracted evidence document with PDF extraction
        metadata.

    Outputs:
        bool: ``True`` when PDF text extraction is too sparse to be a reliable
        text-only substitute for the uploaded source.
    """

    if extracted_document.document_type != "pdf":
        return False
    metadata = extracted_document.metadata
    return not extracted_document.blocks or bool(metadata.get("low_text_density"))


def _raw_pdf_blocks_for_sparse_extractions(
    review_input: QuestionReviewInput,
    extracted_documents: list[ExtractedDocument],
) -> list[dict[str, Any]]:
    """Build raw PDF prompt blocks for sparse direct-route PDF extractions.

    Inputs:
        review_input: Validated question task input containing attachment paths.
        extracted_documents: Text-bearing extraction payloads in attachment
            order.

    Outputs:
        list[dict[str, Any]]: Prompt blocks that preserve original PDF evidence
        for direct review when plain text extraction is sparse. The function
        returns an empty list when extracted documents cannot be safely aligned
        to source paths.
    """

    extraction_paths = [
        path for path in review_input.attachment_paths if _supports_text_extraction(path)
    ]
    if len(extraction_paths) != len(extracted_documents):
        return []

    content_blocks: list[dict[str, Any]] = []
    for path, extracted_document in zip(
        extraction_paths,
        extracted_documents,
        strict=True,
    ):
        mime_type = _mime_type(path)
        is_pdf = path.suffix.lower() == ".pdf" or mime_type == "application/pdf"
        if not is_pdf or not _pdf_extraction_needs_raw_direct_evidence(
            extracted_document
        ):
            continue
        content_blocks.append(
            {
                "type": "input_text",
                "text": "\n".join(
                    [
                        f"Original PDF evidence file: {path.name}",
                        (
                            "Text extraction for this PDF was sparse, so the "
                            "next content block contains the original PDF file. "
                            "Use it as the evidence source when visible text, "
                            "tables, charts, or page images are readable."
                        ),
                    ]
                ),
            }
        )
        content_blocks.extend(_file_blocks(path))
    return content_blocks


def _format_retrieved_chunk_for_final_review(chunk: RetrievedEvidenceChunk) -> str:
    """Format one accepted retrieved chunk for the final RAG review prompt.

    Inputs:
        chunk: Accepted retrieved evidence chunk.

    Outputs:
        str: Compact, traceable chunk text block for prompt inclusion.
    """

    return "\n".join(
        [
            f"Chunk ID: {chunk.chunk_id}",
            *(
                [f"Attachment ID: {chunk.attachment_id}"]
                if chunk.attachment_id
                else []
            ),
            f"Evidence file: {chunk.file_name}",
            f"Document type: {chunk.document_type}",
            f"Location: {json.dumps(chunk.location_json, ensure_ascii=True)}",
            f"Similarity score: {chunk.similarity_score}",
            "Text:",
            chunk.chunk_text,
        ]
    )


def _client_model_name(client: Any) -> str:
    """Return a stable model name from a review client or test double.

    Inputs:
        client: Review client or fake review client.

    Outputs:
        str: Client model name, defaulting to ``gpt-5.4`` when unavailable.
    """

    return str(getattr(client, "model", None) or "gpt-5.4")


def _insufficient_evidence_result(
    review_input: QuestionReviewInput,
    model: str,
    overall_status: str,
    warning: str,
) -> QuestionReviewResult:
    """Build an explicit non-final review result for insufficient evidence.

    Inputs:
        review_input: Question task that could not be fully reviewed.
        model: Model or model-like client name to preserve in the result.
        overall_status: Machine-readable status explaining the limitation.
        warning: User-facing warning describing why no verified answers were
            selected.

    Outputs:
        QuestionReviewResult: Result that keeps current selections for context
        but makes no verified answer selections.
    """

    return QuestionReviewResult(
        question_id=review_input.question.question_id,
        model=model,
        overall_status=overall_status,
        highest_supported_level=None,
        current_selected_answer_item_ids=list(
            review_input.current_selected_answer_item_ids
        ),
        verified_selected_answer_item_ids=[],
        warnings=[warning],
    )


def _rag_insufficient_result(
    review_input: QuestionReviewInput,
    model: str,
    evidence_gaps: list[str],
    stop_reason: str,
) -> QuestionReviewResult:
    """Build a result when RAG retrieval is not sufficient for final review.

    Inputs:
        review_input: Question task being reviewed.
        model: Review model name.
        evidence_gaps: Evidence gaps reported by the retrieval assessment loop.
        stop_reason: Reason the bounded RAG loop stopped.

    Outputs:
        QuestionReviewResult: Explicit insufficient-evidence result without a
        final answer-selection call.
    """

    gap_text = "; ".join(evidence_gaps) if evidence_gaps else "No accepted evidence."
    return _insufficient_evidence_result(
        review_input=review_input,
        model=model,
        overall_status="insufficient_evidence",
        warning=(
            "RAG retrieval stopped before the evidence assessor marked the "
            f"context sufficient for final review (stop_reason={stop_reason}). "
            f"Remaining evidence gaps: {gap_text}"
        ),
    )


def _has_unindexed_visual_evidence(
    review_input: QuestionReviewInput,
    extracted_documents: list[ExtractedDocument],
) -> bool:
    """Return whether direct evidence includes visual content RAG cannot index.

    Inputs:
        review_input: Question task input with attachment paths.
        extracted_documents: Extracted text-bearing evidence in attachment
            order.

    Outputs:
        bool: ``True`` when attachments include PNG/JPEG images or sparse PDFs
        that the current text-only RAG fallback cannot faithfully review.
    """

    if any(
        _image_mime_type(path, _mime_type(path)) is not None
        for path in review_input.attachment_paths
    ):
        return True

    extraction_paths = [
        path for path in review_input.attachment_paths if _supports_text_extraction(path)
    ]
    if len(extraction_paths) != len(extracted_documents):
        return False

    for path, extracted_document in zip(
        extraction_paths,
        extracted_documents,
        strict=True,
    ):
        mime_type = _mime_type(path)
        is_pdf = path.suffix.lower() == ".pdf" or mime_type == "application/pdf"
        if is_pdf and _pdf_extraction_needs_raw_direct_evidence(extracted_document):
            return True
    return False


def _visual_context_limit_result(
    review_input: QuestionReviewInput,
    model: str,
) -> QuestionReviewResult:
    """Build a result for visual evidence that exceeded direct context limits.

    Inputs:
        review_input: Question task that could not be reviewed directly.
        model: Review model name.

    Outputs:
        QuestionReviewResult: Explicit limitation result explaining that the
        current text-only RAG fallback cannot review discarded visual evidence.
    """

    return _insufficient_evidence_result(
        review_input=review_input,
        model=model,
        overall_status="insufficient_extractable_text",
        warning=(
            "Direct review exceeded the model context limit, and the oversized "
            "evidence includes images or scan-heavy PDFs that the current "
            "text-only RAG fallback cannot index. No verified answer selections "
            "were generated from incomplete evidence."
        ),
    )


def _build_rag_final_prompt(
    review_input: QuestionReviewInput,
    accepted_chunks: list[RetrievedEvidenceChunk],
    evidence_gaps: list[str],
    stop_reason: str,
) -> list[dict[str, Any]]:
    """Build the compact final review prompt from accepted RAG evidence.

    Inputs:
        review_input: Validated review task input.
        accepted_chunks: Accepted chunks selected by the RAG loop.
        evidence_gaps: Remaining evidence gaps after the loop stops.
        stop_reason: Reason the retrieval loop stopped.

    Outputs:
        list[dict[str, Any]]: One user message for final structured review.
    """

    accepted_chunk_lines = [
        _format_retrieved_chunk_for_final_review(chunk) for chunk in accepted_chunks
    ]
    prompt_text = "\n\n".join(
        [
            _question_prompt_text(review_input),
            "Use only the accepted evidence chunks below for the final review.",
            (
                "RAG stop reason: "
                f"{stop_reason}"
            ),
            "Remaining evidence gaps:",
            *([f"- {gap}" for gap in evidence_gaps] or ["- None"]),
            "Accepted evidence chunks:",
            *(accepted_chunk_lines or ["No chunks were accepted."]),
        ]
    )
    return [{"role": "user", "content": [{"type": "input_text", "text": prompt_text}]}]


def _has_answer_item_decisions(result: QuestionReviewResult) -> bool:
    """Return whether a review result includes decision rows.

    Inputs:
        result: Structured review result returned by the model client.

    Outputs:
        bool: ``True`` when at least one level has answer item decisions.
    """

    return any(level.answer_item_decisions for level in result.level_results)


def _verified_answer_ids_from_decisions(
    question: AssessmentQuestion,
    result: QuestionReviewResult,
) -> list[str]:
    """Build verified selected answer IDs from selectable decisions.

    Inputs:
        question: Assessment question carrying answer items in framework order.
        result: Structured review result containing level-grouped answer item
            decisions.

    Outputs:
        list[str]: Answer item IDs whose decisions are selectable, ordered by
        the framework answer catalog.
    """

    selected_answer_ids = {
        decision.answer_item_id
        for level in result.level_results
        for decision in level.answer_item_decisions
        if decision.decision in SELECTABLE_AI_DECISIONS
    }
    return [
        answer_item.answer_item_id
        for answer_item in question.answer_items
        if answer_item.answer_item_id in selected_answer_ids
    ]


def _normalize_question_review_result(
    question: AssessmentQuestion,
    result: QuestionReviewResult,
) -> QuestionReviewResult:
    """Normalize model-selected answer IDs from detailed decision rows.

    Inputs:
        question: Assessment question whose answer catalog provides canonical
            ordering for verified selections.
        result: Structured model result to normalize.

    Outputs:
        QuestionReviewResult: Result with ``verified_selected_answer_item_ids``
        recomputed from keep-selected, select, and low-confidence decisions when
        decision rows are present. Results without decision rows are preserved
        for compatibility with older minimal test doubles.
    """

    has_decision_rows = _has_answer_item_decisions(result)
    max_supported_level = max(
        (item.level for item in question.answer_items),
        default=None,
    )
    scoped_result = result
    if max_supported_level is not None:
        highest_supported_level = result.highest_supported_level
        if highest_supported_level is not None:
            highest_supported_level = min(highest_supported_level, max_supported_level)
        scoped_result = result.model_copy(
            update={
                "highest_supported_level": highest_supported_level,
                "level_results": [
                    level
                    for level in result.level_results
                    if level.level <= max_supported_level
                ],
            }
        )
    scoped_answer_ids = {item.answer_item_id for item in question.answer_items}
    scoped_result = scoped_result.model_copy(
        update={
            "verified_selected_answer_item_ids": [
                answer_id
                for answer_id in scoped_result.verified_selected_answer_item_ids
                if answer_id in scoped_answer_ids
            ]
        }
    )

    if not has_decision_rows:
        return scoped_result

    return scoped_result.model_copy(
        update={
            "verified_selected_answer_item_ids": _verified_answer_ids_from_decisions(
                question,
                scoped_result,
            )
        }
    )


def build_question_review_graph(review_client: Any) -> Any:
    """Build and compile the per-question LangGraph workflow.

    Inputs:
        review_client: Object exposing ``review_question(input_messages,
        language=...)`` and returning ``QuestionReviewResult``.

    Outputs:
        Compiled LangGraph application with ``assemble`` and ``model_call``
        nodes wired from ``START`` to ``END``.
    """

    def assemble(state: QuestionReviewState) -> dict[str, Any]:
        """Create model input messages from the review input.

        Inputs:
            state: Graph state containing ``review_input``.

        Outputs:
            dict[str, Any]: Partial state update with ``input_messages``.
        """

        return {"input_messages": _build_prompt(state["review_input"])}

    def model_call(state: QuestionReviewState) -> dict[str, Any]:
        """Call the injected review client with assembled input messages.

        Inputs:
            state: Graph state containing ``input_messages``.

        Outputs:
            dict[str, Any]: Partial state update with the review ``result``.
        """

        review_input = state["review_input"]
        result = review_client.review_question(
            state["input_messages"],
            language=review_input.language,
        )
        return {
            "result": _normalize_question_review_result(
                review_input.question,
                result,
            )
        }

    graph = StateGraph(QuestionReviewState)
    graph.add_node("assemble", assemble)
    graph.add_node("model_call", model_call)
    graph.add_edge(START, "assemble")
    graph.add_edge("assemble", "model_call")
    graph.add_edge("model_call", END)
    return graph.compile()


def run_question_review(
    review_input: QuestionReviewInput,
    review_client: Any | None = None,
) -> QuestionReviewResult:
    """Run the per-question review graph and return its structured result.

    Inputs:
        review_input: Validated task, framework question, current selections,
        and evidence attachment paths.
        review_client: Optional fake or production review client. When omitted,
        ``GenAiReviewClient`` is created with the default non-mini model.

    Outputs:
        QuestionReviewResult: Structured review result produced by the graph.
    """

    client = review_client or GenAiReviewClient()
    graph = build_question_review_graph(client)
    final_state = graph.invoke({"review_input": review_input})
    return final_state["result"]


def _chunk_extracted_documents_for_attachments(
    task_id: str,
    extracted_documents: list[ExtractedDocument],
    attachment_id_by_file: dict[str, str],
    attachment_ids_by_document: list[str] | None = None,
) -> list[EvidenceChunk]:
    """Create chunks while preserving duplicate attachment filename identity.

    Inputs:
        task_id: Review task identifier.
        extracted_documents: Text-bearing documents in extraction order.
        attachment_id_by_file: Backward-compatible filename to attachment ID
            mapping used when each extracted file name is unique.
        attachment_ids_by_document: Optional attachment IDs aligned with
            ``extracted_documents`` order. This avoids collisions when two
            uploaded attachments share the same basename.

    Outputs:
        list[EvidenceChunk]: Ordered retrieval chunks with source attachment IDs
        suitable for embedding and HANA persistence.

    Raises:
        ValueError: Raised when ordered attachment IDs are provided but do not
        align one-to-one with the extracted documents.
    """

    if attachment_ids_by_document is None:
        return chunk_extracted_documents(
            task_id=task_id,
            attachment_id_by_file=attachment_id_by_file,
            extracted_documents=extracted_documents,
        )

    if len(attachment_ids_by_document) != len(extracted_documents):
        raise ValueError(
            "attachment_ids_by_document must align with extracted_documents"
        )

    chunks: list[EvidenceChunk] = []
    for document, attachment_id in zip(
        extracted_documents,
        attachment_ids_by_document,
        strict=True,
    ):
        chunks.extend(
            chunk_extracted_documents(
                task_id=task_id,
                attachment_id_by_file={document.file_name: attachment_id},
                extracted_documents=[document],
            )
        )
    return chunks


def run_rag_question_review(
    review_input: QuestionReviewInput,
    extracted_documents: list[ExtractedDocument],
    attachment_id_by_file: dict[str, str],
    repository: Any,
    review_client: Any | None = None,
    embedding_client: Any | None = None,
    attachment_ids_by_document: list[str] | None = None,
) -> QuestionReviewResult:
    """Run the HANA-backed RAG review flow for one question task.

    Inputs:
        review_input: Validated review task input.
        extracted_documents: Text-bearing extracted evidence documents.
        attachment_id_by_file: Mapping from extracted document filename to the
            source attachment identifier used for chunk persistence.
        repository: Repository exposing chunk persistence and retrieval methods.
        review_client: Optional injected review client for tests.
        embedding_client: Optional injected embedding client for tests.
        attachment_ids_by_document: Optional source attachment IDs aligned with
            ``extracted_documents`` order for duplicate filename safety.

    Outputs:
        QuestionReviewResult: Normalized structured review result.
    """

    client = review_client or GenAiReviewClient()
    embedder = embedding_client or GenAiHubEmbeddingClient(
        DEFAULT_EVIDENCE_EMBEDDING_MODEL
    )
    embedding_model = (
        getattr(embedder, "model_name", None) or DEFAULT_EVIDENCE_EMBEDDING_MODEL
    )

    chunks = _chunk_extracted_documents_for_attachments(
        task_id=review_input.task_id,
        extracted_documents=extracted_documents,
        attachment_id_by_file=attachment_id_by_file,
        attachment_ids_by_document=attachment_ids_by_document,
    )
    embedded_chunks = embed_evidence_chunks(
        chunks=chunks,
        embedding_client=embedder,
        embedding_model=embedding_model,
    )
    repository.save_evidence_chunks(
        task_id=review_input.task_id,
        chunks=[chunk.model_dump(mode="json") for chunk in embedded_chunks],
    )

    initial_queries = build_initial_retrieval_queries(review_input.question)
    rag_result = run_retrieval_assessment_loop(
        task_id=review_input.task_id,
        question=review_input.question,
        language=review_input.language,
        initial_queries=initial_queries,
        retrieve_candidates=lambda queries: retrieve_evidence_candidates(
            repository=repository,
            embedding_client=embedder,
            task_id=review_input.task_id,
            queries=queries,
        ),
        review_client=client,
        repository=repository,
    )

    if not rag_result.sufficient_for_final_review:
        return _rag_insufficient_result(
            review_input=review_input,
            model=_client_model_name(client),
            evidence_gaps=rag_result.evidence_gaps,
            stop_reason=rag_result.stop_reason,
        )

    result = client.review_question(
        _build_rag_final_prompt(
            review_input=review_input,
            accepted_chunks=rag_result.accepted_chunks,
            evidence_gaps=rag_result.evidence_gaps,
            stop_reason=rag_result.stop_reason,
        ),
        language=review_input.language,
    )
    return _normalize_question_review_result(review_input.question, result)


def run_question_review_with_routing(
    review_input: QuestionReviewInput,
    extracted_documents: list[ExtractedDocument],
    attachment_id_by_file: dict[str, str],
    repository: Any,
    review_client: Any | None = None,
    embedding_client: Any | None = None,
    attachment_ids_by_document: list[str] | None = None,
) -> QuestionReviewResult:
    """Route one worker question review through direct or RAG evidence review.

    Inputs:
        review_input: Validated review task input.
        extracted_documents: Text-bearing extracted evidence documents.
        attachment_id_by_file: Mapping from extracted filename to attachment ID.
        repository: Repository exposing question-attempt persistence.
        review_client: Optional injected review client.
        embedding_client: Optional injected embedding client for RAG.
        attachment_ids_by_document: Optional source attachment IDs aligned with
            ``extracted_documents`` order for duplicate filename safety.

    Outputs:
        QuestionReviewResult: Review result from the selected route.
    """

    client = review_client or GenAiReviewClient()
    route_decision = choose_evidence_route(
        question_prompt_text=_question_prompt_text(review_input),
        extracted_documents=extracted_documents,
    )
    repository.save_question_attempt(
        task_id=review_input.task_id,
        mode=route_decision.route.value,
        status="routed",
        model=getattr(client, "model", None),
        usage_json=route_decision.model_dump(mode="json"),
        error_code=None,
        error_message=None,
    )

    if route_decision.route == EvidenceRoute.RAG:
        return run_rag_question_review(
            review_input=review_input,
            extracted_documents=extracted_documents,
            attachment_id_by_file=attachment_id_by_file,
            repository=repository,
            attachment_ids_by_document=attachment_ids_by_document,
            review_client=client,
            embedding_client=embedding_client,
        )

    try:
        result = client.review_question(
            _build_direct_prompt_from_extracted(review_input, extracted_documents),
            language=review_input.language,
        )
    except ReviewContextLimitError as exc:
        repository.save_question_attempt(
            task_id=review_input.task_id,
            mode=EvidenceRoute.DIRECT.value,
            status="failed",
            model=getattr(client, "model", None),
            usage_json=None,
            error_code="context_limit",
            error_message=str(exc),
        )
        if _has_unindexed_visual_evidence(review_input, extracted_documents):
            return _visual_context_limit_result(
                review_input=review_input,
                model=_client_model_name(client),
            )
        return run_rag_question_review(
            review_input=review_input,
            extracted_documents=extracted_documents,
            attachment_id_by_file=attachment_id_by_file,
            repository=repository,
            attachment_ids_by_document=attachment_ids_by_document,
            review_client=client,
            embedding_client=embedding_client,
        )

    return _normalize_question_review_result(review_input.question, result)


def _selected_answer_ids(task: dict[str, Any]) -> list[str]:
    """Read current answer selections from a leased task row.

    Inputs:
        task: Task dictionary from a repository lease or direct HANA query.

    Outputs:
        list[str]: Current selected answer item IDs parsed from the task.
    """

    if "current_answers_json" in task and task["current_answers_json"] is not None:
        return list(json.loads(task["current_answers_json"]))
    return list(task.get("current_selected_answer_item_ids", []))


def _commit_if_supported(repository: Any) -> None:
    """Commit progress telemetry when the repository exposes a commit hook.

    Inputs:
        repository: Repository object used by the manual worker path.

    Outputs:
        None. Durable repositories commit active progress for polling clients;
        in-memory or stub repositories can ignore the call.
    """

    commit = getattr(repository, "commit", None)
    if callable(commit):
        commit()


def _format_timing_log_value(value: Any) -> str:
    """Format one timing-log value for default Python logging output.

    Inputs:
        value: Metric value to render in the log message.

    Outputs:
        str: Compact value text. Simple scalar values are unquoted; strings
        containing whitespace are JSON-quoted for readability.
    """

    if value is None:
        return "null"
    if isinstance(value, str):
        return value if value and not any(char.isspace() for char in value) else json.dumps(value)
    return str(value)


def _timing_log_message(stage: str, duration_ms: int, metrics: dict[str, Any]) -> str:
    """Build a readable timing message for default log formatters.

    Inputs:
        stage: Short machine-readable stage name.
        duration_ms: Stage duration in milliseconds.
        metrics: Additional key/value fields to append to the message.

    Outputs:
        str: Log message containing event name plus key/value timing fields.
    """

    fields = {"stage": stage, "duration_ms": duration_ms, **metrics}
    parts = [
        f"{key}={_format_timing_log_value(value)}"
        for key, value in fields.items()
        if value is not None
    ]
    return "manual_ai_review_stage_timing " + " ".join(parts)


def _update_manual_question_progress(
    repository: Any,
    task: dict[str, Any],
    status: str,
    progress_message: str,
    rag_call_count: int | None = None,
    query_count: int | None = None,
    retrieved_chunk_count: int | None = None,
) -> None:
    """Publish one manual question progress update when supported.

    Inputs:
        repository: Repository that may expose ``update_question_progress``.
        task: Leased task row containing ``task_id`` and ``lease_owner``.
        status: Active worker sub-status.
        progress_message: Human-readable progress text for polling clients.
        rag_call_count: Optional running RAG call count.
        query_count: Optional running retrieval query count.
        retrieved_chunk_count: Optional running retrieved chunk count.

    Outputs:
        None. Unsupported repositories or missing worker IDs are skipped.
    """

    progress_updater = getattr(repository, "update_question_progress", None)
    worker_id = str(task.get("lease_owner", ""))
    if not callable(progress_updater) or not worker_id:
        return
    progress_updater(
        task_id=task["task_id"],
        worker_id=worker_id,
        status=status,
        progress_message=progress_message,
        rag_call_count=rag_call_count,
        query_count=query_count,
        retrieved_chunk_count=retrieved_chunk_count,
    )
    _commit_if_supported(repository)


def _log_manual_worker_stage_timing(
    task: dict[str, Any],
    stage: str,
    started_at: float,
    **metrics: Any,
) -> int:
    """Log one manual worker stage duration with structured metadata.

    Inputs:
        task: Leased task row that identifies the job and question.
        stage: Short machine-readable stage name.
        started_at: ``time.perf_counter`` timestamp captured before the stage.
        **metrics: Stage-specific counters such as attachment or document
            counts.

    Outputs:
        int: Stage duration in milliseconds.
    """

    duration_ms = int((time.perf_counter() - started_at) * 1000)
    log_metrics = {
        "task_id": task.get("task_id"),
        "job_id": task.get("job_id"),
        "question_id": task.get("question_id"),
        "dimension": task.get("dimension"),
        **metrics,
    }
    logger.info(
        _timing_log_message(stage, duration_ms, log_metrics),
        extra={
            "task_id": task.get("task_id"),
            "job_id": task.get("job_id"),
            "question_id": task.get("question_id"),
            "dimension": task.get("dimension"),
            "stage": stage,
            "duration_ms": duration_ms,
            **metrics,
        },
    )
    return duration_ms


def _attachment_bytes(attachment: dict[str, Any]) -> bytes:
    """Normalize repository attachment content to bytes.

    Inputs:
        attachment: Attachment dictionary containing a ``content`` value.

    Outputs:
        bytes: Binary payload suitable for writing to a temporary evidence file.
    """

    content = attachment["content"]
    if isinstance(content, bytes):
        return content
    if isinstance(content, memoryview):
        return content.tobytes()
    return bytes(content)


def run_task_from_hana(
    repository: Any,
    task: dict[str, Any],
    attachments: list[dict[str, Any]],
) -> QuestionReviewResult:
    """Build and run a question review from HANA repository task payloads.

    Inputs:
        repository: Repository exposing ``list_questions(dimension)``.
        task: Leased or selected question task row containing task ID, dimension,
        question ID, and current answer selections.
        attachments: Attachment rows with ``file_name`` and binary ``content``.

    Outputs:
        QuestionReviewResult: Structured review result for the task. Persistence
        is intentionally left to the later worker task.

    Raises:
        ValueError: Raised when the task question ID cannot be found in the
        repository's framework question list.
    """

    questions = repository.list_questions(task["dimension"])
    question = next(
        (
            candidate
            for candidate in questions
            if candidate.question_id == task["question_id"]
        ),
        None,
    )
    if question is None:
        raise ValueError(
            "Cannot run AI review task because the framework question was not "
            f"found: {task['question_id']}"
        )

    task_started_at = time.perf_counter()
    with TemporaryDirectory(prefix="document-assessment-review-") as temporary_directory:
        write_started_at = time.perf_counter()
        attachment_paths: list[Path] = []
        attachment_id_by_path: dict[Path, str] = {}
        attachment_id_by_file: dict[str, str] = {}
        for index, attachment in enumerate(attachments, start=1):
            file_name = Path(attachment["file_name"]).name
            attachment_directory = Path(temporary_directory) / f"{index:03d}"
            attachment_directory.mkdir()
            temporary_path = attachment_directory / file_name
            temporary_path.write_bytes(_attachment_bytes(attachment))
            attachment_paths.append(temporary_path)
            attachment_id = str(
                attachment.get("attachment_id", f"attachment-{index:03d}")
            )
            attachment_id_by_path[temporary_path] = attachment_id
            attachment_id_by_file.setdefault(file_name, attachment_id)
        _log_manual_worker_stage_timing(
            task,
            "write_attachments",
            write_started_at,
            attachment_count=len(attachments),
        )

        review_input = QuestionReviewInput(
            task_id=task["task_id"],
            language=task.get("language", DEFAULT_LANGUAGE),
            question=question,
            current_selected_answer_item_ids=_selected_answer_ids(task),
            attachment_paths=attachment_paths,
        )
        upload_count = len(attachment_paths)
        upload_word = "document" if upload_count == 1 else "documents"
        _update_manual_question_progress(
            repository,
            task,
            "extracting_documents",
            f"Extracting text from {upload_count} uploaded {upload_word}.",
            0,
            0,
            0,
        )
        extraction_started_at = time.perf_counter()
        extracted_documents = extract_evidence_files(attachment_paths)
        _log_manual_worker_stage_timing(
            task,
            "extract_documents",
            extraction_started_at,
            attachment_count=len(attachment_paths),
            extracted_document_count=len(extracted_documents),
        )
        extracted_attachment_ids = [
            attachment_id_by_path[path]
            for path in attachment_paths
            if _supports_text_extraction(path)
        ]
        if len(extracted_documents) == len(extracted_attachment_ids):
            attachment_ids_by_document: list[str] | None = extracted_attachment_ids
            extraction_pairs = zip(
                extracted_documents,
                extracted_attachment_ids,
                strict=True,
            )
        else:
            attachment_ids_by_document = None
            extraction_pairs = (
                (
                    extracted_document,
                    attachment_id_by_file[extracted_document.file_name],
                )
                for extracted_document in extracted_documents
            )

        save_extractions_started_at = time.perf_counter()
        for extracted_document, attachment_id in extraction_pairs:
            repository.save_attachment_extraction(
                task_id=task["task_id"],
                attachment_id=attachment_id,
                extracted=extracted_document,
                estimated_tokens=sum(
                    estimate_text_tokens(block.text)
                    for block in extracted_document.blocks
                ),
                warnings=list(extracted_document.metadata.get("warnings", [])),
            )
        _log_manual_worker_stage_timing(
            task,
            "save_extractions",
            save_extractions_started_at,
            extracted_document_count=len(extracted_documents),
        )
        extracted_count = len(extracted_documents)
        extracted_word = "document" if extracted_count == 1 else "documents"
        _update_manual_question_progress(
            repository,
            task,
            "embedding_documents",
            (
                "Creating evidence chunks and embeddings for "
                f"{extracted_count} extracted {extracted_word}."
            ),
            0,
            0,
            0,
        )
        from app.services.batch_question_react_graph import (
            run_manual_question_react_review,
        )

        result = run_manual_question_react_review(
            review_input=review_input,
            extracted_documents=extracted_documents,
            attachment_id_by_file=attachment_id_by_file,
            repository=repository,
            worker_id=str(task.get("lease_owner", "")),
            attachment_ids_by_document=attachment_ids_by_document,
        )
        _log_manual_worker_stage_timing(
            task,
            "manual_task_total",
            task_started_at,
            attachment_count=len(attachments),
            extracted_document_count=len(extracted_documents),
            overall_status=result.overall_status,
        )
        return result
