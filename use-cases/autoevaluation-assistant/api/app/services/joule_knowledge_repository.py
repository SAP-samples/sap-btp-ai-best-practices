"""HANA repository for Assessment knowledge resources and conversation state."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.services.joule_knowledge import (
    DimensionExplanation,
    GlossarySearchResult,
    JouleKnowledgeSeed,
    QuestionExplanationResult,
    normalize_search_text,
)
from app.services.joule_knowledge_importer import source_metadata, vector_to_json
from app.services.joule_knowledge_schema import (
    JOULE_KNOWLEDGE_FUZZY_INDEX_STATEMENTS,
    JOULE_KNOWLEDGE_SCHEMA_STATEMENTS,
    index_name_from_joule_index_statement,
    table_name_from_joule_create_statement,
)


class HanaJouleKnowledgeRepository:
    """Persist and retrieve Joule knowledge resources in SAP HANA.

    Inputs:
        session: Active SQLAlchemy session bound to a HANA database.

    Outputs:
        Repository object exposing schema initialization, import replacement,
        retrieval, and conversation message persistence methods.
    """

    def __init__(self, session: Session) -> None:
        """Store the HANA session used by the repository.

        Inputs:
            session: SQLAlchemy session used for all SQL statements.

        Outputs:
            None.
        """

        self.session = session

    def create_schema(self) -> None:
        """Create Joule knowledge tables and fuzzy indexes if absent.

        Inputs:
            None.

        Outputs:
            None. HANA DDL statements are executed on the active session.
        """

        for statement in JOULE_KNOWLEDGE_SCHEMA_STATEMENTS:
            table_name = table_name_from_joule_create_statement(statement)
            if not self._table_exists(table_name):
                self.session.execute(text(statement))
        for statement in JOULE_KNOWLEDGE_FUZZY_INDEX_STATEMENTS:
            index_name = index_name_from_joule_index_statement(statement)
            if not self._index_exists(index_name):
                self.session.execute(text(statement))

    def replace_knowledge_seed(
        self,
        seed: JouleKnowledgeSeed,
        question_embeddings: dict[str, tuple[list[float], list[float]]] | None = None,
        embedding_model: str | None = None,
    ) -> None:
        """Replace persisted Joule knowledge resources with an imported seed.

        Inputs:
            seed: Normalized workbook seed.
            question_embeddings: Optional question and explanation vectors keyed
                by question ID.
            embedding_model: Embedding model name used for supplied vectors.

        Outputs:
            None. Existing knowledge rows are deleted and reinserted.
        """

        self.create_schema()
        self.session.execute(text("delete from joule_dimensions"))
        self.session.execute(text("delete from joule_question_explanations"))
        self.session.execute(text("delete from joule_glossary_terms"))
        self.session.execute(text("delete from joule_knowledge_imports"))

        for source_row in source_metadata(seed):
            self.session.execute(
                text(
                    "insert into joule_knowledge_imports "
                    "(import_id, source_name, source_path, source_sha256, "
                    "source_sheet, row_count, status) "
                    "values (:import_id, :source_name, :source_path, "
                    ":source_sha256, :source_sheet, :row_count, :status)"
                ),
                {"import_id": uuid4().hex, **source_row},
            )

        for index, term in enumerate(seed.glossary_terms, start=1):
            term_id = f"{term.language}-{term.source_sheet}-{term.source_row}-{index}"
            self.session.execute(
                text(
                    "insert into joule_glossary_terms "
                    "(term_id, language, term, normalized_term, definition, "
                    "source_workbook, source_sheet, source_row) "
                    "values (:term_id, :language, :term, :normalized_term, "
                    ":definition, :source_workbook, :source_sheet, :source_row)"
                ),
                {
                    "term_id": term_id,
                    "language": term.language,
                    "term": term.term,
                    "normalized_term": term.normalized_term,
                    "definition": term.definition,
                    "source_workbook": term.source_workbook,
                    "source_sheet": term.source_sheet,
                    "source_row": term.source_row,
                },
            )

        for question in seed.question_explanations:
            question_vector, explanation_vector = (None, None)
            if question_embeddings and question.question_id in question_embeddings:
                question_vector, explanation_vector = question_embeddings[question.question_id]
            self.session.execute(
                text(
                    "insert into joule_question_explanations "
                    "(question_id, dimension_key, source_dimension, question_text, "
                    "explanation, source_workbook, source_sheet, source_row, "
                    "embedding_model, question_embedding, explanation_embedding) "
                    "values (:question_id, :dimension_key, :source_dimension, "
                    ":question_text, :explanation, :source_workbook, :source_sheet, "
                    ":source_row, :embedding_model, to_real_vector(:question_embedding), "
                    "to_real_vector(:explanation_embedding))"
                ),
                {
                    "question_id": question.question_id,
                    "dimension_key": question.dimension_key,
                    "source_dimension": question.source_dimension,
                    "question_text": question.question,
                    "explanation": question.explanation,
                    "source_workbook": question.source_workbook,
                    "source_sheet": question.source_sheet,
                    "source_row": question.source_row,
                    "embedding_model": embedding_model,
                    "question_embedding": vector_to_json(question_vector),
                    "explanation_embedding": vector_to_json(explanation_vector),
                },
            )

        for display_order, dimension in enumerate(seed.dimensions, start=1):
            self.session.execute(
                text(
                    "insert into joule_dimensions "
                    "(dimension_key, display_order, name, aliases_json, explanation, "
                    "source_workbook, source_sheet, source_row) "
                    "values (:dimension_key, :display_order, :name, :aliases_json, "
                    ":explanation, :source_workbook, :source_sheet, :source_row)"
                ),
                {
                    "dimension_key": dimension.dimension_key,
                    "display_order": display_order,
                    "name": dimension.name,
                    "aliases_json": json.dumps(list(dimension.aliases), ensure_ascii=False),
                    "explanation": dimension.explanation,
                    "source_workbook": dimension.source_workbook,
                    "source_sheet": dimension.source_sheet,
                    "source_row": dimension.source_row,
                },
            )

    def get_question_explanation_by_id(
        self,
        question_id: str,
    ) -> QuestionExplanationResult | None:
        """Return one persisted question explanation by question ID.

        Inputs:
            question_id: Canonical assessment question identifier.

        Outputs:
            QuestionExplanationResult | None: Matching row or ``None``.
        """

        row = self.session.execute(
            text(
                "select question_id, dimension_key, question_text, explanation "
                "from joule_question_explanations "
                "where question_id = :question_id"
            ),
            {"question_id": question_id},
        ).mappings().first()
        if row is None:
            return None
        return QuestionExplanationResult(
            question_id=str(row["question_id"]),
            dimension=str(row["dimension_key"]),
            question=str(row["question_text"]),
            explanation=str(row["explanation"]),
            similarity_score=1.0,
        )

    def search_question_explanations(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[QuestionExplanationResult]:
        """Run HANA vector search over question and explanation embeddings.

        Inputs:
            query_embedding: Query vector generated from the user request.
            top_k: Maximum result count.

        Outputs:
            list[QuestionExplanationResult]: Semantic matches ordered by cosine
            similarity.
        """

        limit = max(1, min(int(top_k), 10))
        query_vector = vector_to_json(query_embedding)
        rows = self.session.execute(
            text(
                f"select top {limit} question_id, dimension_key, question_text, "
                "explanation, "
                "case "
                "when cosine_similarity(question_embedding, to_real_vector(:query_vector)) "
                ">= cosine_similarity(explanation_embedding, to_real_vector(:query_vector)) "
                "then cosine_similarity(question_embedding, to_real_vector(:query_vector)) "
                "else cosine_similarity(explanation_embedding, to_real_vector(:query_vector)) "
                "end as similarity_score "
                "from joule_question_explanations "
                "where question_embedding is not null "
                "and explanation_embedding is not null "
                "order by similarity_score desc"
            ),
            {"query_vector": query_vector},
        ).mappings().all()
        return [
            QuestionExplanationResult(
                question_id=str(row["question_id"]),
                dimension=str(row["dimension_key"]),
                question=str(row["question_text"]),
                explanation=str(row["explanation"]),
                similarity_score=float(row["similarity_score"]),
            )
            for row in rows
        ]

    def search_admin_document_chunks(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Run HANA vector search over global admin document chunks.

        Inputs:
            query_embedding: Query vector generated from the user request.
            top_k: Maximum result count.

        Outputs:
            list[dict[str, Any]]: Admin chunk rows ordered by cosine similarity.
        """

        limit = max(1, min(int(top_k), 20))
        query_vector = vector_to_json(query_embedding)
        rows = self.session.execute(
            text(
                f"select top {limit} chunk_id, document_id, file_name, "
                "document_type, location_json, chunk_text, "
                "cosine_similarity(embedding, to_real_vector(:query_vector)) as similarity_score "
                "from joule_admin_document_chunks "
                "where embedding is not null "
                "order by similarity_score desc"
            ),
            {"query_vector": query_vector},
        ).mappings().all()
        return [
            {
                "chunk_id": str(row["chunk_id"]),
                "document_id": str(row["document_id"]),
                "file_name": str(row["file_name"]),
                "document_type": str(row["document_type"]),
                "chunk_text": str(row["chunk_text"]),
                "location_json": (
                    json.loads(row["location_json"])
                    if isinstance(row["location_json"], str)
                    else dict(row["location_json"])
                ),
                "similarity_score": float(row["similarity_score"]),
            }
            for row in rows
        ]

    def get_dimension_explanation(
        self,
        dimension_key: str,
    ) -> DimensionExplanation | None:
        """Return one dimension explanation by canonical dimension key.

        Inputs:
            dimension_key: Canonical assessment dimension key.

        Outputs:
            DimensionExplanation | None: Matching dimension row or ``None``.
        """

        row = self.session.execute(
            text(
                "select dimension_key, name, aliases_json, explanation "
                "from joule_dimensions "
                "where dimension_key = :dimension_key"
            ),
            {"dimension_key": dimension_key},
        ).mappings().first()
        if row is None:
            return None
        return DimensionExplanation(
            dimension_key=str(row["dimension_key"]),
            name=str(row["name"]),
            explanation=str(row["explanation"]),
            aliases=list(json.loads(str(row["aliases_json"]))),
        )

    def search_glossary_exact(
        self,
        term: str,
        language: str,
    ) -> list[GlossarySearchResult]:
        """Return exact normalized glossary matches.

        Inputs:
            term: User-provided term.
            language: Requested glossary language.

        Outputs:
            list[GlossarySearchResult]: Exact matches preserving duplicates.
        """

        rows = self.session.execute(
            text(
                "select term, definition, language, source_row "
                "from joule_glossary_terms "
                "where language = :language "
                "and normalized_term = :normalized_term "
                "order by source_row asc"
            ),
            {
                "language": language,
                "normalized_term": normalize_search_text(term),
            },
        ).mappings().all()
        return [
            GlossarySearchResult(
                term=str(row["term"]),
                definition=str(row["definition"]),
                language=str(row["language"]),
                score=1.0,
                source_row=int(row["source_row"]),
            )
            for row in rows
        ]

    def search_glossary_fuzzy(
        self,
        term: str,
        language: str,
        top_k: int,
    ) -> list[GlossarySearchResult]:
        """Return fuzzy glossary matches using HANA ``CONTAINS``.

        Inputs:
            term: User-provided term.
            language: Requested glossary language.
            top_k: Maximum number of results.

        Outputs:
            list[GlossarySearchResult]: Fuzzy matches ordered by HANA score.
        """

        limit = max(1, min(int(top_k), 20))
        rows = self.session.execute(
            text(
                f"select top {limit} term, definition, language, source_row, "
                "score() as score "
                "from joule_glossary_terms "
                "where language = :language "
                "and contains(term, :term, fuzzy(0.75, 'searchMode=text')) "
                "order by score desc, source_row asc"
            ),
            {"language": language, "term": term},
        ).mappings().all()
        return [
            GlossarySearchResult(
                term=str(row["term"]),
                definition=str(row["definition"]),
                language=str(row["language"]),
                score=float(row["score"]),
                source_row=int(row["source_row"]),
            )
            for row in rows
        ]

    def append_agent_message(
        self,
        context_id: str,
        role: str,
        content: str,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Persist one conversation message for the remote Joule agent.

        Inputs:
            context_id: A2A context ID mapped to the LangGraph thread ID.
            role: Message role, for example ``user`` or ``assistant``.
            content: Message content.
            metadata: Optional structured metadata serialized as JSON.

        Outputs:
            None. The message row is inserted into HANA.
        """

        turn_number = self._next_turn_number(context_id)
        self.session.execute(
            text(
                "insert into joule_agent_messages "
                "(message_id, context_id, turn_number, role, content, metadata_json) "
                "values (:message_id, :context_id, :turn_number, :role, :content, "
                ":metadata_json)"
            ),
            {
                "message_id": uuid4().hex,
                "context_id": context_id,
                "turn_number": turn_number,
                "role": role,
                "content": content,
                "metadata_json": (
                    json.dumps(metadata, ensure_ascii=False) if metadata is not None else None
                ),
            },
        )

    def commit(self) -> None:
        """Commit pending HANA changes on the repository session.

        Inputs:
            None.

        Outputs:
            None. The underlying SQLAlchemy session is committed.
        """

        self.session.commit()

    def list_agent_messages(self, context_id: str, limit: int = 20) -> list[dict[str, str]]:
        """Return recent persisted messages for one A2A/LangGraph context.

        Inputs:
            context_id: A2A context ID.
            limit: Maximum number of recent messages to return.

        Outputs:
            list[dict[str, str]]: Ordered role/content message dictionaries.
        """

        max_rows = max(1, min(int(limit), 50))
        rows = self.session.execute(
            text(
                f"select top {max_rows} role, content "
                "from joule_agent_messages "
                "where context_id = :context_id "
                "order by turn_number desc"
            ),
            {"context_id": context_id},
        ).mappings().all()
        return [
            {"role": str(row["role"]), "content": str(row["content"])}
            for row in reversed(rows)
        ]

    def _next_turn_number(self, context_id: str) -> int:
        """Return the next message turn number for a context.

        Inputs:
            context_id: A2A context ID.

        Outputs:
            int: Next monotonically increasing turn number for the context.
        """

        row = self.session.execute(
            text(
                "select coalesce(max(turn_number), 0) as max_turn "
                "from joule_agent_messages "
                "where context_id = :context_id"
            ),
            {"context_id": context_id},
        ).mappings().first()
        return int(row["max_turn"] if row else 0) + 1

    def _table_exists(self, table_name: str) -> bool:
        """Return whether a table exists in the current HANA schema.

        Inputs:
            table_name: Logical table name.

        Outputs:
            bool: ``True`` if HANA reports the table, otherwise ``False``.
        """

        row = self.session.execute(
            text(
                "select table_name "
                "from sys.tables "
                "where schema_name = current_schema "
                "and table_name = :table_name"
            ),
            {"table_name": table_name.upper()},
        ).first()
        return row is not None

    def _index_exists(self, index_name: str) -> bool:
        """Return whether an index exists in the current HANA schema.

        Inputs:
            index_name: Logical index name.

        Outputs:
            bool: ``True`` if HANA reports the index, otherwise ``False``.
        """

        row = self.session.execute(
            text(
                "select index_name "
                "from sys.indexes "
                "where schema_name = current_schema "
                "and index_name = :index_name"
            ),
            {"index_name": index_name.upper()},
        ).first()
        return row is not None
