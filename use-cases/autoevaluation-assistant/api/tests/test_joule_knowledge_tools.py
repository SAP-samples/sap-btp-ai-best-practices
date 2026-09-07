"""Tests for Joule knowledge agent retrieval tools."""

from app.services.joule_knowledge import (
    DimensionExplanation,
    GlossarySearchResult,
    JouleKnowledgeService,
    QuestionExplanationResult,
)


class _FakeKnowledgeRepository:
    """Fake repository for testing tool orchestration without HANA.

    Inputs:
        None. Calls are recorded on attributes for assertions.

    Outputs:
        Object exposing the subset of repository methods used by
        ``JouleKnowledgeService``.
    """

    def __init__(self) -> None:
        """Initialize fake rows and call logs.

        Inputs:
            None.

        Outputs:
            None. Attributes are prepared for test inspection.
        """

        self.vector_queries: list[tuple[list[float], int]] = []
        self.admin_vector_queries: list[tuple[list[float], int]] = []
        self.exact_terms: list[tuple[str, str]] = []
        self.fuzzy_terms: list[tuple[str, str, int]] = []

    def get_question_explanation_by_id(
        self,
        question_id: str,
    ) -> QuestionExplanationResult | None:
        """Return a direct question match for one known ID.

        Inputs:
            question_id: Canonical question identifier.

        Outputs:
            QuestionExplanationResult | None: Fake result when the ID matches.
        """

        if question_id != "Q.STR.01.01":
            return None
        return QuestionExplanationResult(
            question_id=question_id,
            dimension="Strategy",
            question="Does your organization have a process in place for strategic planning?",
            explanation="Strategic planning guides resources toward objectives.",
            similarity_score=1.0,
        )

    def search_question_explanations(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[QuestionExplanationResult]:
        """Return a fake vector search result and record call arguments.

        Inputs:
            query_embedding: Query vector supplied by the embedding client.
            top_k: Maximum number of results requested.

        Outputs:
            list[QuestionExplanationResult]: One semantic question match.
        """

        self.vector_queries.append((query_embedding, top_k))
        return [
            QuestionExplanationResult(
                question_id="Q.RCG.01.01",
                dimension="Risk & Control Governance",
                question="Has your organization structured a multi-level control system?",
                explanation="A multi-level control system mitigates operational risks.",
                similarity_score=0.87,
            )
        ]

    def search_admin_document_chunks(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict[str, object]]:
        """Return fake admin corpus chunks and record vector search inputs.

        Inputs:
            query_embedding: Query vector supplied by the embedding client.
            top_k: Maximum number of admin chunks requested.

        Outputs:
            list[dict[str, object]]: Fake retrieved admin document chunks.
        """

        self.admin_vector_queries.append((query_embedding, top_k))
        return [
            {
                "chunk_id": "admin-chunk-1",
                "document_id": "admin-document-1",
                "file_name": "policy.pdf",
                "document_type": "pdf",
                "chunk_text": "Admin policy requires documented ownership.",
                "location_json": {"pages": [1]},
                "similarity_score": 0.91,
            }
        ]

    def get_dimension_explanation(
        self,
        dimension_key: str,
    ) -> DimensionExplanation | None:
        """Return a fake dimension explanation for a canonical dimension key.

        Inputs:
            dimension_key: Canonical dimension name.

        Outputs:
            DimensionExplanation | None: Fake dimension explanation.
        """

        if dimension_key != "Information Systems & Digital":
            return None
        return DimensionExplanation(
            dimension_key=dimension_key,
            name="Information Systems & Digital",
            explanation="Covers IT, data management, and digital solutions.",
            aliases=["Information & Digital Systems"],
        )

    def search_glossary_exact(
        self,
        term: str,
        language: str,
    ) -> list[GlossarySearchResult]:
        """Return fake exact glossary matches for duplicate Procurement rows.

        Inputs:
            term: User term.
            language: Requested language.

        Outputs:
            list[GlossarySearchResult]: Duplicate exact matches when applicable.
        """

        self.exact_terms.append((term, language))
        if term.lower() != "procurement" or language != "en":
            return []
        return [
            GlossarySearchResult(
                term="Procurement",
                definition="A business process for acquiring goods.",
                language="en",
                score=1.0,
                source_row=11,
            ),
            GlossarySearchResult(
                term="Procurement",
                definition="The process of acquiring goods and services.",
                language="en",
                score=1.0,
                source_row=182,
            ),
        ]

    def search_glossary_fuzzy(
        self,
        term: str,
        language: str,
        top_k: int,
    ) -> list[GlossarySearchResult]:
        """Return fake HANA fuzzy search results.

        Inputs:
            term: User term.
            language: Requested language.
            top_k: Maximum number of results requested.

        Outputs:
            list[GlossarySearchResult]: One fuzzy match.
        """

        self.fuzzy_terms.append((term, language, top_k))
        return [
            GlossarySearchResult(
                term="AI",
                definition="Artificial Intelligence.",
                language=language,
                score=0.92,
                source_row=5,
            )
        ]


class _FakeEmbeddingClient:
    """Fake embedding client used to verify semantic search behavior."""

    def __init__(self) -> None:
        """Initialize the embedding request log.

        Inputs:
            None.

        Outputs:
            None.
        """

        self.calls: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        """Return a deterministic query vector and record the requested text.

        Inputs:
            text: Text to embed.

        Outputs:
            list[float]: Fake embedding vector.
        """

        self.calls.append(text)
        return [0.1, 0.2, 0.3]


class _FakeTranslator:
    """Fake translator for testing Italian runtime translation flow."""

    def __init__(self) -> None:
        """Initialize translation call logs.

        Inputs:
            None.

        Outputs:
            None.
        """

        self.query_calls: list[tuple[str, str]] = []
        self.answer_calls: list[tuple[str, str]] = []

    def translate_query_to_english(self, text: str, language: str) -> str:
        """Return a deterministic English retrieval query.

        Inputs:
            text: User query.
            language: Requested answer language.

        Outputs:
            str: English retrieval text.
        """

        self.query_calls.append((text, language))
        return "risk control governance"

    def translate_answer_from_english(self, text: str, language: str) -> str:
        """Return a deterministic translated answer.

        Inputs:
            text: English answer.
            language: Requested answer language.

        Outputs:
            str: Translated answer text.
        """

        self.answer_calls.append((text, language))
        return f"IT: {text}"


def test_question_explanation_uses_direct_question_id_without_embedding() -> None:
    """Verify direct question code lookups bypass semantic vector search.

    Inputs:
        None.

    Outputs:
        None. Assertions confirm no embedding is generated for known IDs.
    """

    repository = _FakeKnowledgeRepository()
    embeddings = _FakeEmbeddingClient()
    service = JouleKnowledgeService(repository, embeddings)

    result = service.question_explanation("Q.STR.01.01", language="en")

    assert result[0].question_id == "Q.STR.01.01"
    assert embeddings.calls == []
    assert repository.vector_queries == []


def test_question_explanation_embeds_italian_query_and_translates_result() -> None:
    """Verify Italian semantic lookup translates around English source data.

    Inputs:
        None.

    Outputs:
        None. Assertions confirm Italian queries are translated before HANA
        vector search and result text is translated after retrieval.
    """

    repository = _FakeKnowledgeRepository()
    embeddings = _FakeEmbeddingClient()
    translator = _FakeTranslator()
    service = JouleKnowledgeService(repository, embeddings, translator)

    result = service.question_explanation("Che cosa significa controllo?", language="it")

    assert embeddings.calls == ["risk control governance"]
    assert repository.vector_queries == [([0.1, 0.2, 0.3], 3)]
    assert result[0].explanation.startswith("IT: ")
    assert translator.query_calls == [("Che cosa significa controllo?", "it")]
    assert translator.answer_calls


def test_dimension_explanation_resolves_alias_without_embeddings() -> None:
    """Verify dimensions are selected from aliases instead of vector search.

    Inputs:
        None.

    Outputs:
        None. Assertions confirm the alias maps to the canonical dimension.
    """

    repository = _FakeKnowledgeRepository()
    embeddings = _FakeEmbeddingClient()
    service = JouleKnowledgeService(repository, embeddings)

    result = service.dimension_explanation("information digital systems", language="en")

    assert result is not None
    assert result.dimension_key == "Information Systems & Digital"
    assert embeddings.calls == []


def test_glossary_search_returns_all_exact_duplicates_before_fuzzy() -> None:
    """Verify exact duplicate glossary matches are preserved and returned.

    Inputs:
        None.

    Outputs:
        None. Assertions confirm fuzzy search is skipped after exact matches.
    """

    repository = _FakeKnowledgeRepository()
    service = JouleKnowledgeService(repository, _FakeEmbeddingClient())

    results = service.glossary_search("Procurement", language="en")

    assert len(results) == 2
    assert {result.source_row for result in results} == {11, 182}
    assert repository.fuzzy_terms == []


def test_glossary_search_falls_back_to_hana_fuzzy_search() -> None:
    """Verify glossary lookup falls back to HANA fuzzy search when needed.

    Inputs:
        None.

    Outputs:
        None. Assertions confirm fuzzy search receives normalized top-k input.
    """

    repository = _FakeKnowledgeRepository()
    service = JouleKnowledgeService(repository, _FakeEmbeddingClient())

    results = service.glossary_search("artifical inteligence", language="en", top_k=2)

    assert results[0].term == "AI"
    assert repository.fuzzy_terms == [("artifical inteligence", "en", 2)]


def test_admin_document_search_uses_embedding_and_admin_repository() -> None:
    """Verify general Joule RAG searches the isolated admin document corpus."""

    repository = _FakeKnowledgeRepository()
    embeddings = _FakeEmbeddingClient()
    service = JouleKnowledgeService(repository, embeddings)

    results = service.admin_document_search("How is ownership documented?", top_k=4)

    assert embeddings.calls == ["How is ownership documented?"]
    assert repository.admin_vector_queries == [([0.1, 0.2, 0.3], 4)]
    assert results == [
        {
            "chunk_id": "admin-chunk-1",
            "document_id": "admin-document-1",
            "file_name": "policy.pdf",
            "document_type": "pdf",
            "chunk_text": "Admin policy requires documented ownership.",
            "location_json": {"pages": [1]},
            "similarity_score": 0.91,
        }
    ]
