"""Evidence chunking and embedding support for AI review RAG."""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field

from app.services.document_extractors import ExtractedBlock, ExtractedDocument
from app.services.evidence_routing import TOKEN_CHARACTER_RATIO, estimate_text_tokens

EMBEDDING_REQUEST_TOKEN_LIMIT = 250_000
"""Conservative token ceiling for one SAP Gen AI Hub embedding request."""


class EvidenceChunk(BaseModel):
    """Chunk of extracted evidence ready for embedding and retrieval.

    Inputs:
        Field values describing source metadata, chunk text, and embedding data.

    Outputs:
        Validated chunk object that can be persisted in HANA.
    """

    chunk_id: str
    task_id: str
    attachment_id: str
    file_name: str
    document_type: str
    source_block_ids: list[str]
    location_json: dict[str, object] = Field(default_factory=dict)
    chunk_text: str
    estimated_tokens: int
    embedding_model: str
    embedding: list[float] = Field(default_factory=list)
    content_hash: str


def content_hash(text: str) -> str:
    """Return a stable SHA-256 hash for chunk text.

    Inputs:
        text: Chunk text.

    Outputs:
        str: Hex-encoded SHA-256 hash.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_id_for(
    task_id: str,
    attachment_id: str,
    chunk_index: int,
    source_block_ids: list[str],
    chunk_text: str,
) -> str:
    """Build a deterministic chunk identifier for extracted evidence.

    Inputs:
        task_id: Review task identifier.
        attachment_id: Source attachment identifier.
        chunk_index: Zero-based chunk position within the source attachment.
        source_block_ids: Ordered extracted block identifiers in the chunk.
        chunk_text: Chunk text used to derive the content hash.

    Outputs:
        str: Stable concise chunk identifier for persistence and auditing.
    """
    stable_input = "\n".join(
        [
            task_id,
            attachment_id,
            str(chunk_index),
            "\x1f".join(source_block_ids),
            content_hash(chunk_text),
        ]
    )
    digest = hashlib.sha256(stable_input.encode("utf-8")).hexdigest()
    return f"chunk-{digest[:32]}"


def _block_location(blocks: list[ExtractedBlock]) -> dict[str, object]:
    """Build source location metadata for a set of extracted blocks.

    Inputs:
        blocks: Extracted source blocks included in one chunk.

    Outputs:
        dict[str, object]: Page, section, sheet, table, and row metadata. Row
        metadata is represented as the minimum ``row_start`` and maximum
        ``row_end`` across included blocks.
    """
    pages = [block.page for block in blocks if block.page is not None]
    sections = [block.section_label for block in blocks if block.section_label]
    sheets = [block.sheet_name for block in blocks if block.sheet_name]
    tables = [block.table_name for block in blocks if block.table_name]
    row_starts = [block.row_start for block in blocks if block.row_start is not None]
    row_ends = [block.row_end for block in blocks if block.row_end is not None]
    location: dict[str, object] = {}
    if pages:
        location["pages"] = pages
    if sections:
        location["sections"] = list(dict.fromkeys(sections))
    if sheets:
        location["sheets"] = list(dict.fromkeys(sheets))
    if tables:
        location["tables"] = list(dict.fromkeys(tables))
    if row_starts:
        location["row_start"] = min(row_starts)
    if row_ends:
        location["row_end"] = max(row_ends)
    return location


def _split_text_for_target_tokens(text: str, target_tokens: int) -> list[str]:
    """Split text into segments that fit the target token estimate.

    Inputs:
        text: Source block text to split without mutating the source block.
        target_tokens: Maximum estimated tokens per returned segment.

    Outputs:
        list[str]: Ordered text segments, each estimated within ``target_tokens``.
    """
    if not text:
        return []
    if estimate_text_tokens(text) <= target_tokens:
        return [text]

    max_chars = max(1, int(target_tokens * TOKEN_CHARACTER_RATIO))
    remaining = text
    segments: list[str] = []

    while estimate_text_tokens(remaining) > target_tokens:
        split_at = remaining.rfind(" ", 0, max_chars + 1)
        if split_at <= 0:
            split_at = max_chars

        segment = remaining[:split_at].strip()
        if not segment:
            segment = remaining[:max_chars]
            split_at = max_chars

        segments.append(segment)
        remaining = remaining[split_at:].lstrip()

    if remaining:
        segments.append(remaining)
    return segments


def _chunk_document_blocks(
    task_id: str,
    attachment_id: str,
    document: ExtractedDocument,
    target_tokens: int,
) -> list[EvidenceChunk]:
    """Chunk one extracted document by accumulated token estimate.

    Inputs:
        task_id: Review task identifier.
        attachment_id: Source attachment identifier.
        document: Extracted document to chunk.
        target_tokens: Target maximum token estimate per chunk.

    Outputs:
        list[EvidenceChunk]: Ordered chunks for this document.
    """
    chunks: list[EvidenceChunk] = []
    current_blocks: list[ExtractedBlock] = []
    current_tokens = 0
    chunk_index = 0

    def append_chunk(blocks: list[ExtractedBlock], chunk_text: str) -> None:
        """Append one chunk for the supplied blocks and text.

        Inputs:
            blocks: Source blocks represented by the chunk.
            chunk_text: Text payload for this chunk.

        Outputs:
            None. A chunk is appended to ``chunks``.
        """
        nonlocal chunk_index
        if not blocks or not chunk_text:
            return
        source_block_ids = [block.block_id for block in blocks]
        chunks.append(
            EvidenceChunk(
                chunk_id=chunk_id_for(
                    task_id=task_id,
                    attachment_id=attachment_id,
                    chunk_index=chunk_index,
                    source_block_ids=source_block_ids,
                    chunk_text=chunk_text,
                ),
                task_id=task_id,
                attachment_id=attachment_id,
                file_name=document.file_name,
                document_type=document.document_type,
                source_block_ids=source_block_ids,
                location_json=_block_location(blocks),
                chunk_text=chunk_text,
                estimated_tokens=estimate_text_tokens(chunk_text),
                embedding_model="",
                embedding=[],
                content_hash=content_hash(chunk_text),
            )
        )
        chunk_index += 1

    def flush_current() -> None:
        """Append the current accumulated chunk when it has text.

        Inputs:
            None. The closure reads accumulated blocks.

        Outputs:
            None. A chunk is appended to ``chunks``.
        """
        nonlocal current_blocks, current_tokens
        if not current_blocks:
            return
        append_chunk(
            blocks=current_blocks,
            chunk_text="\n\n".join(block.text for block in current_blocks),
        )
        current_blocks = []
        current_tokens = 0

    for block in document.blocks:
        block_tokens = estimate_text_tokens(block.text)
        if block_tokens > target_tokens:
            flush_current()
            for segment in _split_text_for_target_tokens(block.text, target_tokens):
                append_chunk(blocks=[block], chunk_text=segment)
            continue
        if current_blocks and current_tokens + block_tokens > target_tokens:
            flush_current()
        current_blocks.append(block)
        current_tokens += block_tokens
    flush_current()
    return chunks


def chunk_extracted_documents(
    task_id: str,
    attachment_id_by_file: dict[str, str],
    extracted_documents: list[ExtractedDocument],
    target_tokens: int = 1_200,
) -> list[EvidenceChunk]:
    """Create retrieval chunks from extracted evidence documents.

    Inputs:
        task_id: Review task identifier.
        attachment_id_by_file: Mapping from extracted file name to attachment ID.
        extracted_documents: Ordered extracted evidence documents.
        target_tokens: Target token estimate per chunk.

    Outputs:
        list[EvidenceChunk]: Ordered chunk list ready for embedding.
    """
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")

    chunks: list[EvidenceChunk] = []
    for document in extracted_documents:
        attachment_id = attachment_id_by_file[document.file_name]
        chunks.extend(
            _chunk_document_blocks(
                task_id=task_id,
                attachment_id=attachment_id,
                document=document,
                target_tokens=target_tokens,
            )
        )
    return chunks


def _embedding_batches(
    chunks: list[EvidenceChunk],
    max_request_tokens: int,
) -> list[list[EvidenceChunk]]:
    """Split chunks into embedding request batches by estimated token count.

    Inputs:
        chunks: Ordered chunks to embed.
        max_request_tokens: Maximum estimated tokens allowed in one embedding
        API request.

    Outputs:
        list[list[EvidenceChunk]]: Ordered chunk batches whose estimated token
        totals do not exceed ``max_request_tokens``.

    Raises:
        ValueError: Raised when the token limit is invalid or one chunk exceeds
        the per-request budget by itself.
    """
    if max_request_tokens <= 0:
        raise ValueError("max_request_tokens must be positive")

    batches: list[list[EvidenceChunk]] = []
    current_batch: list[EvidenceChunk] = []
    current_tokens = 0

    for chunk in chunks:
        chunk_tokens = max(1, chunk.estimated_tokens)
        if chunk_tokens > max_request_tokens:
            raise ValueError(
                f"Chunk {chunk.chunk_id} has {chunk_tokens} estimated tokens, "
                f"exceeding the embedding request limit of {max_request_tokens}"
            )
        if current_batch and current_tokens + chunk_tokens > max_request_tokens:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(chunk)
        current_tokens += chunk_tokens

    if current_batch:
        batches.append(current_batch)
    return batches


def embed_evidence_chunks(
    chunks: list[EvidenceChunk],
    embedding_client: object,
    embedding_model: str,
    max_request_tokens: int = EMBEDDING_REQUEST_TOKEN_LIMIT,
) -> list[EvidenceChunk]:
    """Generate embeddings for evidence chunks.

    Inputs:
        chunks: Ordered chunks to embed.
        embedding_client: Object exposing ``embed_batch(texts)``.
        embedding_model: Embedding deployment name to persist.
        max_request_tokens: Maximum estimated tokens to send in one embedding
        request.

    Outputs:
        list[EvidenceChunk]: Copies of chunks with embedding model and vectors.
    """
    if not chunks:
        return []

    embeddings: list[list[float]] = []
    for batch in _embedding_batches(
        chunks=chunks,
        max_request_tokens=max_request_tokens,
    ):
        batch_embeddings = embedding_client.embed_batch(
            [chunk.chunk_text for chunk in batch]
        )
        if len(batch_embeddings) != len(batch):
            raise ValueError(
                f"Embedding client returned {len(batch_embeddings)} vectors "
                f"for {len(batch)} chunks"
            )
        embeddings.extend(batch_embeddings)

    return [
        chunk.model_copy(
            update={
                "embedding_model": embedding_model,
                "embedding": list(embedding),
            }
        )
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]
