"""Tests for evidence chunking and embedding indexing."""

import pytest

from app.services.document_extractors import ExtractedBlock, ExtractedDocument
from app.services.evidence_indexing import (
    EvidenceChunk,
    chunk_extracted_documents,
    embed_evidence_chunks,
)


class FakeEmbeddingClient:
    """Fake embedding client returning deterministic vectors.

    Inputs:
        None. The fake records texts received by ``embed_batch``.

    Outputs:
        Test double compatible with ``GenAiHubEmbeddingClient``.
    """

    def __init__(self) -> None:
        """Initialize an empty call log.

        Inputs:
            None.

        Outputs:
            None. The fake is ready to record embedded text.
        """
        self.texts: list[str] = []
        self.batches: list[list[str]] = []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return one simple vector per input text.

        Inputs:
            texts: Texts to embed.

        Outputs:
            list[list[float]]: Deterministic vectors with text-length signals.
        """
        self.batches.append(list(texts))
        self.texts.extend(texts)
        return [[float(len(text)), 1.0] for text in texts]


def test_chunk_extracted_documents_preserves_source_metadata() -> None:
    """Verify chunks include source block IDs, file metadata, and token estimates.

    Inputs:
        None. The test builds a two-page extracted PDF.

    Outputs:
        None. Assertions confirm chunk metadata is traceable.
    """
    extracted = ExtractedDocument(
        file_name="report.pdf",
        document_type="pdf",
        blocks=[
            ExtractedBlock(
                block_id="pdf-page-0001",
                block_type="page",
                text="Board approves strategy.",
                page=1,
            ),
            ExtractedBlock(
                block_id="pdf-page-0002",
                block_type="page",
                text="Objectives are monitored quarterly.",
                page=2,
            ),
        ],
    )

    chunks = chunk_extracted_documents(
        task_id="task-1",
        attachment_id_by_file={"report.pdf": "attachment-1"},
        extracted_documents=[extracted],
        target_tokens=50,
    )

    assert len(chunks) == 1
    assert chunks[0].task_id == "task-1"
    assert chunks[0].attachment_id == "attachment-1"
    assert chunks[0].source_block_ids == ["pdf-page-0001", "pdf-page-0002"]
    assert chunks[0].location_json["pages"] == [1, 2]
    assert "Board approves strategy" in chunks[0].chunk_text


def test_chunk_extracted_documents_rejects_non_positive_target_tokens() -> None:
    """Verify invalid chunk token configuration fails clearly.

    Inputs:
        None. The test passes an invalid target token value.

    Outputs:
        None. Assertions confirm a ``ValueError`` is raised.
    """
    with pytest.raises(ValueError, match="target_tokens must be positive"):
        chunk_extracted_documents(
            task_id="task-1",
            attachment_id_by_file={},
            extracted_documents=[],
            target_tokens=0,
        )


def test_chunk_extracted_documents_generates_stable_chunk_ids() -> None:
    """Verify identical extracted evidence produces deterministic chunk IDs.

    Inputs:
        None. The test builds one extracted document and chunks it twice.

    Outputs:
        None. Assertions confirm repeated chunking yields matching IDs.
    """
    extracted = ExtractedDocument(
        file_name="report.pdf",
        document_type="pdf",
        blocks=[
            ExtractedBlock(
                block_id="pdf-page-0001",
                block_type="page",
                text="Board approves strategy.",
                page=1,
            ),
            ExtractedBlock(
                block_id="pdf-page-0002",
                block_type="page",
                text="Objectives are monitored quarterly.",
                page=2,
            ),
        ],
    )

    first_chunks = chunk_extracted_documents(
        task_id="task-1",
        attachment_id_by_file={"report.pdf": "attachment-1"},
        extracted_documents=[extracted],
        target_tokens=50,
    )
    second_chunks = chunk_extracted_documents(
        task_id="task-1",
        attachment_id_by_file={"report.pdf": "attachment-1"},
        extracted_documents=[extracted],
        target_tokens=50,
    )

    assert [chunk.chunk_id for chunk in first_chunks] == [
        chunk.chunk_id for chunk in second_chunks
    ]
    assert first_chunks[0].chunk_id.startswith("chunk-")
    assert len(first_chunks[0].chunk_id) == 38

    changed = extracted.model_copy(
        update={
            "blocks": [
                extracted.blocks[0].model_copy(
                    update={"text": "Board rejects strategy."}
                ),
                extracted.blocks[1],
            ]
        }
    )
    changed_chunks = chunk_extracted_documents(
        task_id="task-1",
        attachment_id_by_file={"report.pdf": "attachment-1"},
        extracted_documents=[changed],
        target_tokens=50,
    )

    assert changed_chunks[0].chunk_id != first_chunks[0].chunk_id


def test_chunk_extracted_documents_splits_oversized_single_blocks() -> None:
    """Verify one large source block is split into target-sized chunks.

    Inputs:
        None. The test builds one oversized extracted PDF page.

    Outputs:
        None. Assertions confirm every chunk respects the token target.
    """
    oversized_text = " ".join(f"strategy-{index:03d}" for index in range(80))
    extracted = ExtractedDocument(
        file_name="large-report.pdf",
        document_type="pdf",
        blocks=[
            ExtractedBlock(
                block_id="pdf-page-0001",
                block_type="page",
                text=oversized_text,
                page=1,
            )
        ],
    )

    chunks = chunk_extracted_documents(
        task_id="task-1",
        attachment_id_by_file={"large-report.pdf": "attachment-1"},
        extracted_documents=[extracted],
        target_tokens=20,
    )

    assert len(chunks) > 1
    assert all(chunk.estimated_tokens <= 20 for chunk in chunks)
    assert all(chunk.source_block_ids == ["pdf-page-0001"] for chunk in chunks)
    assert all(chunk.location_json["pages"] == [1] for chunk in chunks)


def test_chunk_extracted_documents_preserves_row_range_metadata() -> None:
    """Verify chunks include XLSX/table row range citation metadata.

    Inputs:
        None. The test builds two spreadsheet-style extracted blocks.

    Outputs:
        None. Assertions confirm row metadata is preserved in location JSON.
    """
    extracted = ExtractedDocument(
        file_name="evidence.xlsx",
        document_type="xlsx",
        blocks=[
            ExtractedBlock(
                block_id="xlsx-block-0001",
                block_type="sheet",
                text="Metric | Value\nRevenue | 10",
                sheet_name="Summary",
                table_name="UsedRange",
                row_start=2,
                row_end=3,
            ),
            ExtractedBlock(
                block_id="xlsx-block-0002",
                block_type="sheet",
                text="Metric | Value\nEmissions | 4",
                sheet_name="Summary",
                table_name="UsedRange",
                row_start=8,
                row_end=10,
            ),
        ],
    )

    chunks = chunk_extracted_documents(
        task_id="task-1",
        attachment_id_by_file={"evidence.xlsx": "attachment-1"},
        extracted_documents=[extracted],
        target_tokens=50,
    )

    assert chunks[0].location_json["sheets"] == ["Summary"]
    assert chunks[0].location_json["tables"] == ["UsedRange"]
    assert chunks[0].location_json["row_start"] == 2
    assert chunks[0].location_json["row_end"] == 10


def test_embed_evidence_chunks_adds_vectors() -> None:
    """Verify embedding indexing attaches vectors to chunk payloads.

    Inputs:
        None. The test uses one chunk and a fake embedding client.

    Outputs:
        None. Assertions confirm embedding vectors and model names are assigned.
    """
    chunk = EvidenceChunk(
        chunk_id="chunk-1",
        task_id="task-1",
        attachment_id="attachment-1",
        file_name="report.pdf",
        document_type="pdf",
        source_block_ids=["pdf-page-0001"],
        location_json={"pages": [1]},
        chunk_text="Board approves strategy.",
        estimated_tokens=7,
        embedding_model="",
        embedding=[],
        content_hash="hash-1",
    )
    fake = FakeEmbeddingClient()

    embedded = embed_evidence_chunks(
        chunks=[chunk],
        embedding_client=fake,
        embedding_model="text-embedding-3-large",
    )

    assert fake.texts == ["Board approves strategy."]
    assert embedded[0].embedding_model == "text-embedding-3-large"
    assert embedded[0].embedding == [24.0, 1.0]


def test_embed_evidence_chunks_splits_large_embedding_requests() -> None:
    """Verify embedding calls stay under the provider request token budget.

    Inputs:
        None. The test builds four chunks whose combined estimated tokens exceed
        the configured embedding request limit.

    Outputs:
        None. Assertions confirm chunks are embedded in ordered sub-batches and
        vectors are still assigned to every chunk.
    """
    chunks = [
        EvidenceChunk(
            chunk_id=f"chunk-{index}",
            task_id="task-1",
            attachment_id="attachment-1",
            file_name="report.pdf",
            document_type="pdf",
            source_block_ids=[f"pdf-page-{index:04d}"],
            location_json={"pages": [index]},
            chunk_text=f"Chunk {index}",
            estimated_tokens=100_000,
            embedding_model="",
            embedding=[],
            content_hash=f"hash-{index}",
        )
        for index in range(1, 5)
    ]
    fake = FakeEmbeddingClient()

    embedded = embed_evidence_chunks(
        chunks=chunks,
        embedding_client=fake,
        embedding_model="text-embedding-3-large",
    )

    assert fake.batches == [
        ["Chunk 1", "Chunk 2"],
        ["Chunk 3", "Chunk 4"],
    ]
    assert [chunk.chunk_id for chunk in embedded] == [
        "chunk-1",
        "chunk-2",
        "chunk-3",
        "chunk-4",
    ]
    assert all(chunk.embedding_model == "text-embedding-3-large" for chunk in embedded)
    assert [chunk.embedding for chunk in embedded] == [
        [7.0, 1.0],
        [7.0, 1.0],
        [7.0, 1.0],
        [7.0, 1.0],
    ]


def test_embed_evidence_chunks_rejects_embedding_count_mismatch() -> None:
    """Verify embedding clients must return one vector per chunk.

    Inputs:
        None. The test uses a client that returns no vectors for one chunk.

    Outputs:
        None. Assertions confirm a clear ``ValueError`` is raised.
    """

    class EmptyEmbeddingClient:
        """Fake client that returns the wrong vector count.

        Inputs:
            None.

        Outputs:
            Test double that simulates a broken embedding response.
        """

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            """Return no vectors regardless of the requested texts.

            Inputs:
                texts: Texts requested for embedding.

            Outputs:
                list[list[float]]: Empty vector list.
            """
            return []

    chunk = EvidenceChunk(
        chunk_id="chunk-1",
        task_id="task-1",
        attachment_id="attachment-1",
        file_name="report.pdf",
        document_type="pdf",
        source_block_ids=["pdf-page-0001"],
        location_json={"pages": [1]},
        chunk_text="Board approves strategy.",
        estimated_tokens=7,
        embedding_model="",
        embedding=[],
        content_hash="hash-1",
    )

    with pytest.raises(
        ValueError,
        match="Embedding client returned 0 vectors for 1 chunks",
    ):
        embed_evidence_chunks(
            chunks=[chunk],
            embedding_client=EmptyEmbeddingClient(),
            embedding_model="text-embedding-3-large",
        )
