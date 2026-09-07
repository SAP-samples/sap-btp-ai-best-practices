"""Domain models and retrieval services for the Assessment knowledge agent."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Protocol

from app.models.language import normalize_language

KNOWN_QUESTION_ID_FIXES = {"Q,FDR.12.01": "Q.FDR.12.01"}
QUESTION_ID_PATTERN = re.compile(r"^Q\.[A-Z]{3}\.\d{2}\.\d{2}$")
QUESTION_ID_SEARCH_PATTERN = re.compile(r"Q[,.][A-Z]{3}\.\d{2}\.\d{2}", re.IGNORECASE)


DIMENSION_ALIASES: dict[str, tuple[str, ...]] = {
    "Strategy": ("Strategy", "Strategia"),
    "Risk & Control Governance": (
        "Risk & Control Governance",
        "Risk and Control Governance",
        "Risk Control Governance",
        "Governance del rischio e controllo",
    ),
    "Organization & Regulatory System": (
        "Organization & Regulatory System",
        "Organization & Regulatory Framework",
        "Organization & Internal Regulatory System",
        "Organization and Regulatory System",
        "Organizzazione e sistema normativo",
    ),
    "People & Culture": ("People & Culture", "People and Culture", "Persone e cultura"),
    "Combined Assurance & Management Oversight": (
        "Combined Assurance & Management Oversight",
        "Combined Assurance and Management Oversight",
        "Combined Assurance",
    ),
    "Information Systems & Digital": (
        "Information Systems & Digital",
        "Information & Digital Systems",
        "Information and Digital Systems",
        "Sistemi Informativi & Digital",
        "Sistemi Informativi e Digital",
    ),
    "Resilience Factors": ("Resilience Factors", "Fattori di resilienza"),
}
"""Canonical dimension names and accepted aliases for direct dimension lookup."""


def normalize_search_text(value: str) -> str:
    """Normalize user/search text for deterministic exact matching.

    Inputs:
        value: Raw text from a workbook or user query.

    Outputs:
        str: Lowercase text with surrounding whitespace removed and internal
        whitespace collapsed.
    """

    return " ".join(value.strip().lower().split())


def normalize_question_id(value: str) -> str:
    """Normalize and validate an assessment question identifier.

    Inputs:
        value: Raw question identifier from a workbook or user query.

    Outputs:
        str: Canonical ``Q.ABC.00.00`` question identifier.

    Raises:
        ValueError: Raised when the ID is malformed and is not the one known
        typo present in the PoC workbook.
    """

    question_id = value.strip().upper().replace(",", ".", 1)
    question_id = KNOWN_QUESTION_ID_FIXES.get(value.strip().upper(), question_id)
    if not QUESTION_ID_PATTERN.match(question_id):
        raise ValueError(f"Malformed question ID: {value!r}")
    return question_id


def canonical_dimension_key(value: str) -> str:
    """Resolve a dimension name or alias to the canonical dimension key.

    Inputs:
        value: Dimension name from a workbook, user query, or tool argument.

    Outputs:
        str: Canonical dimension key used in HANA.

    Raises:
        ValueError: Raised when the dimension is not one of the supported assessment
        dimensions or aliases.
    """

    normalized = _dimension_match_key(value)
    for dimension_key, aliases in DIMENSION_ALIASES.items():
        for alias in aliases:
            if normalized == _dimension_match_key(alias):
                return dimension_key
    raise ValueError(f"Unsupported assessment dimension: {value!r}")


def _dimension_match_key(value: str) -> str:
    """Return a permissive key for matching dimension names and aliases.

    Inputs:
        value: Raw dimension text.

    Outputs:
        str: Lowercase key that ignores punctuation and optional conjunctions
        such as ``and``.
    """

    normalized = normalize_search_text(value).replace("&", "and")
    tokens = re.findall(r"[a-z0-9]+", normalized)
    return " ".join(token for token in tokens if token != "and")


def extract_question_id(text: str) -> str | None:
    """Find a question identifier embedded in free-form user text.

    Inputs:
        text: User query.

    Outputs:
        str | None: Canonical question ID when present, otherwise ``None``.
    """

    match = QUESTION_ID_SEARCH_PATTERN.search(text)
    if match is None:
        return None
    return normalize_question_id(match.group(0))


@dataclass(frozen=True)
class GlossaryTermSeed:
    """One glossary row loaded from the source workbook."""

    language: str
    term: str
    normalized_term: str
    definition: str
    source_workbook: str
    source_sheet: str
    source_row: int


@dataclass(frozen=True)
class QuestionExplanationSeed:
    """One question explanation row loaded from the source workbook."""

    question_id: str
    dimension_key: str
    source_dimension: str
    question: str
    explanation: str
    source_workbook: str
    source_sheet: str
    source_row: int


@dataclass(frozen=True)
class DimensionSeed:
    """One dimension explanation loaded from the source workbook."""

    dimension_key: str
    name: str
    aliases: tuple[str, ...]
    explanation: str
    source_workbook: str
    source_sheet: str
    source_row: int


@dataclass(frozen=True)
class JouleKnowledgeSeed:
    """Normalized workbook data ready to persist into HANA."""

    glossary_terms: list[GlossaryTermSeed]
    question_explanations: list[QuestionExplanationSeed]
    dimensions: list[DimensionSeed]
    glossary_workbook: str
    explanations_workbook: str


@dataclass(frozen=True)
class QuestionExplanationResult:
    """Question explanation result returned by the retrieval service."""

    question_id: str
    dimension: str
    question: str
    explanation: str
    similarity_score: float | None = None


@dataclass(frozen=True)
class DimensionExplanation:
    """Dimension explanation result returned by the retrieval service."""

    dimension_key: str
    name: str
    explanation: str
    aliases: list[str]


@dataclass(frozen=True)
class GlossarySearchResult:
    """Glossary search result returned by exact or fuzzy lookup."""

    term: str
    definition: str
    language: str
    score: float | None
    source_row: int


class EmbeddingClient(Protocol):
    """Protocol for embedding clients used by semantic search."""

    def embed_query(self, text: str) -> list[float]:
        """Embed one query string.

        Inputs:
            text: Query text.

        Outputs:
            list[float]: Numeric embedding vector.
        """


class RuntimeTranslator(Protocol):
    """Protocol for runtime translation around English source content."""

    def translate_query_to_english(self, text: str, language: str) -> str:
        """Translate a non-English retrieval query into English.

        Inputs:
            text: User query.
            language: User-requested language code.

        Outputs:
            str: English query text for semantic retrieval.
        """

    def translate_answer_from_english(self, text: str, language: str) -> str:
        """Translate English retrieved content into the requested language.

        Inputs:
            text: Grounded English content.
            language: Target language code.

        Outputs:
            str: Translated content.
        """


class JouleKnowledgeRepository(Protocol):
    """Repository protocol consumed by the Joule knowledge service."""

    def get_question_explanation_by_id(
        self,
        question_id: str,
    ) -> QuestionExplanationResult | None:
        """Return one question explanation by canonical question ID."""

    def search_question_explanations(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[QuestionExplanationResult]:
        """Return semantic question explanation matches."""

    def search_admin_document_chunks(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict[str, object]]:
        """Return semantic matches from the global admin document corpus."""

    def get_dimension_explanation(
        self,
        dimension_key: str,
    ) -> DimensionExplanation | None:
        """Return one dimension explanation by canonical key."""

    def search_glossary_exact(
        self,
        term: str,
        language: str,
    ) -> list[GlossarySearchResult]:
        """Return exact glossary term matches."""

    def search_glossary_fuzzy(
        self,
        term: str,
        language: str,
        top_k: int,
    ) -> list[GlossarySearchResult]:
        """Return HANA fuzzy glossary term matches."""


class JouleKnowledgeService:
    """Expose retrieval tools for question, dimension, and glossary answers.

    Inputs:
        repository: HANA-backed or fake repository implementing lookup methods.
        embedding_client: Client used to embed semantic question queries.
        translator: Optional translator for Italian runtime compatibility.

    Outputs:
        Service object with methods designed to be wrapped as LangGraph tools.
    """

    def __init__(
        self,
        repository: JouleKnowledgeRepository,
        embedding_client: EmbeddingClient,
        translator: RuntimeTranslator | None = None,
    ) -> None:
        """Store retrieval dependencies.

        Inputs:
            repository: Repository used for all HANA reads.
            embedding_client: Embedding client for semantic search.
            translator: Optional runtime translator.

        Outputs:
            None.
        """

        self.repository = repository
        self.embedding_client = embedding_client
        self.translator = translator

    def question_explanation(
        self,
        query: str,
        language: str = "en",
        top_k: int = 3,
    ) -> list[QuestionExplanationResult]:
        """Find why an assessment question is important.

        Inputs:
            query: User question, natural language description, or question ID.
            language: Desired answer language, ``en`` or ``it``.
            top_k: Maximum semantic matches to return when no direct ID is
                present.

        Outputs:
            list[QuestionExplanationResult]: Direct or semantic question
            explanation matches.
        """

        normalized_language = normalize_language(language)
        question_id = extract_question_id(query)
        if question_id is not None:
            direct_result = self.repository.get_question_explanation_by_id(question_id)
            if direct_result is not None:
                return self._translate_question_results([direct_result], normalized_language)

        retrieval_query = query
        if normalized_language != "en" and self.translator is not None:
            retrieval_query = self.translator.translate_query_to_english(
                query,
                normalized_language,
            )

        embedding = self.embedding_client.embed_query(retrieval_query)
        results = self.repository.search_question_explanations(embedding, top_k=top_k)
        return self._translate_question_results(results, normalized_language)

    def dimension_explanation(
        self,
        dimension: str,
        language: str = "en",
    ) -> DimensionExplanation | None:
        """Return the explanation for one assessment dimension.

        Inputs:
            dimension: Canonical dimension name or accepted alias.
            language: Desired answer language, ``en`` or ``it``.

        Outputs:
            DimensionExplanation | None: Matching dimension explanation, or
            ``None`` if no persisted dimension row exists.
        """

        normalized_language = normalize_language(language)
        dimension_key = canonical_dimension_key(dimension)
        result = self.repository.get_dimension_explanation(dimension_key)
        if (
            result is None
            or normalized_language == "en"
            or self.translator is None
        ):
            return result
        return replace(
            result,
            explanation=self.translator.translate_answer_from_english(
                result.explanation,
                normalized_language,
            ),
        )

    def glossary_search(
        self,
        term: str,
        language: str = "en",
        top_k: int = 5,
    ) -> list[GlossarySearchResult]:
        """Search the assessment glossary using exact and HANA fuzzy matching.

        Inputs:
            term: User-provided glossary term.
            language: Desired glossary language, ``en`` or ``it``.
            top_k: Maximum fuzzy matches to return when exact search misses.

        Outputs:
            list[GlossarySearchResult]: Exact duplicate matches when present,
            otherwise fuzzy matches ordered by HANA score.
        """

        normalized_language = normalize_language(language)
        exact_matches = self.repository.search_glossary_exact(term, normalized_language)
        if exact_matches:
            return exact_matches

        fuzzy_matches = self.repository.search_glossary_fuzzy(
            term,
            normalized_language,
            top_k=max(1, min(top_k, 20)),
        )
        if fuzzy_matches:
            return fuzzy_matches

        fallback_language = "it" if normalized_language == "en" else "en"
        return self.repository.search_glossary_fuzzy(
            term,
            fallback_language,
            top_k=max(1, min(top_k, 20)),
        )

    def admin_document_search(
        self,
        query: str,
        language: str = "en",
        top_k: int = 5,
    ) -> list[dict[str, object]]:
        """Search the global admin document corpus for general Joule answers.

        Inputs:
            query: User question or retrieval query.
            language: User language code. Non-English queries are translated to
                English before embedding when a translator is configured.
            top_k: Maximum admin chunks to return.

        Outputs:
            list[dict[str, object]]: Retrieved chunk dictionaries containing
            chunk_id, document_id, file_name, document_type, chunk_text,
            location_json, and similarity_score.
        """

        normalized_language = normalize_language(language)
        retrieval_query = query
        if normalized_language != "en" and self.translator is not None:
            retrieval_query = self.translator.translate_query_to_english(
                query,
                normalized_language,
            )
        embedding = self.embedding_client.embed_query(retrieval_query)
        return self.repository.search_admin_document_chunks(
            embedding,
            top_k=max(1, min(top_k, 20)),
        )

    def _translate_question_results(
        self,
        results: list[QuestionExplanationResult],
        language: str,
    ) -> list[QuestionExplanationResult]:
        """Translate question result text when runtime Italian output is needed.

        Inputs:
            results: English grounded question explanation results.
            language: Requested output language.

        Outputs:
            list[QuestionExplanationResult]: Results in the requested language
            when a translator is configured, otherwise unchanged results.
        """

        if language == "en" or self.translator is None:
            return results
        translated_results: list[QuestionExplanationResult] = []
        for result in results:
            translated_results.append(
                replace(
                    result,
                    question=self.translator.translate_answer_from_english(
                        result.question,
                        language,
                    ),
                    explanation=self.translator.translate_answer_from_english(
                        result.explanation,
                        language,
                    ),
                )
            )
        return translated_results
