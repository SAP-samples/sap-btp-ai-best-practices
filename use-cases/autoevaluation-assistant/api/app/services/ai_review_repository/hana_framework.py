"""Assessment framework reads for the HANA AI review repository."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from typing import Any
from uuid import uuid4

from sqlalchemy import text

from app.models.ai_review import (
    BatchReviewJobResponse,
    QuestionReviewResult,
    ReviewJobResponse,
    ReviewJobStatusResponse,
    ReviewResetResponse,
    ReviewTaskStatus,
)
from app.models.assessment import AnswerItem, AssessmentDimension, AssessmentQuestion
from app.models.documents import (
    DocumentDownload,
    DocumentIngestionJobResponse,
    DocumentListResponse,
    DocumentSummary,
)
from app.models.language import DEFAULT_LANGUAGE, normalize_language
from app.services.document_extractors import ExtractedDocument
from app.services.hana_schema import HANA_SCHEMA_STATEMENTS, table_name_from_create_statement
from app.services.joule_knowledge_importer import vector_to_json

from .common import (
    ACTIVE_REVIEW_TASK_STATUSES,
    ADMIN_DOCUMENT_CORPUS_ID,
    _derive_finished_batch_status,
    _derive_job_status,
    _document_payload_bytes,
    _document_phase_processed_count,
    _normalize_task_status_for_response,
    _pending_task_status_for_runtime,
    _row_batches,
    document_content_hash,
)

class HanaFrameworkMixin:
    """Read localized assessment dimensions and questions from HANA."""

    def list_dimensions(self, language: str = DEFAULT_LANGUAGE) -> list[AssessmentDimension]:
        """Return assessment framework dimensions ordered for display.

        Inputs:
            language: Supported response language for localized display names.

        Outputs:
            list[AssessmentDimension]: Dimension summaries with total question
            counts and zero answered counts for the current stateless lookup.
        """

        language = normalize_language(language)
        rows = self.session.execute(
            text(
                "select d.dimension, "
                "coalesce(t.display_name, d.dimension) as display_name, "
                "d.question_count "
                "from assessment_dimensions d "
                "left join assessment_dimension_translations t "
                "on t.dimension = d.dimension and t.language = :language "
                "order by d.display_order"
            ),
            {"language": language},
        ).mappings()
        return [
            AssessmentDimension(
                dimension=row["dimension"],
                display_name=row["display_name"],
                question_count=row["question_count"],
                answered_count=0,
            )
            for row in rows
        ]

    def list_questions(
        self,
        dimension: str,
        language: str = DEFAULT_LANGUAGE,
    ) -> list[AssessmentQuestion]:
        """Return framework questions, explanations, and answer items.

        Inputs:
            dimension: Framework dimension whose questions should be returned.
            language: Supported response language for localized question text.

        Outputs:
            list[AssessmentQuestion]: Questions ordered by display order with
            answer items ordered by level and item index.
        """

        language = normalize_language(language)
        question_rows = self.session.execute(
            text(
                "select q.question_id, q.dimension, q.section, "
                "coalesce(t.section, q.section) as localized_section, "
                "coalesce(tt.topic_title, et.topic_title, t.section, q.section) "
                "as localized_topic_title, "
                "coalesce(t.question_text, q.question_text) as localized_question, "
                "e.explanation "
                "from assessment_questions q "
                "left join assessment_question_translations t "
                "on t.question_id = q.question_id and t.language = :language "
                "left join assessment_question_topic_translations tt "
                "on tt.question_id = q.question_id and tt.language = :language "
                "left join assessment_question_topic_translations et "
                "on et.question_id = q.question_id and et.language = 'en' "
                "left join assessment_question_explanations e "
                "on e.question_id = q.question_id "
                "where q.dimension = :dimension "
                "order by q.display_order"
            ),
            {"dimension": dimension, "language": language},
        ).mappings().all()
        item_rows = self.session.execute(
            text(
                "select a.answer_item_id, a.question_id, a.level, a.item_index, "
                "coalesce(t.answer_text, a.answer_text) as localized_answer_text "
                "from assessment_answer_items a "
                "left join assessment_answer_item_translations t "
                "on t.answer_item_id = a.answer_item_id and t.language = :language "
                "where a.question_id in ("
                "select question_id from assessment_questions "
                "where dimension = :dimension"
                ") "
                "order by a.question_id, a.level, a.item_index"
            ),
            {"dimension": dimension, "language": language},
        ).mappings().all()

        items_by_question: dict[str, list[AnswerItem]] = {}
        for row in item_rows:
            items_by_question.setdefault(row["question_id"], []).append(
                AnswerItem(
                    answer_item_id=row["answer_item_id"],
                    question_id=row["question_id"],
                    level=row["level"],
                    item_index=row["item_index"],
                    text=row["localized_answer_text"],
                )
            )

        return [
            AssessmentQuestion(
                question_id=row["question_id"],
                dimension=row["dimension"],
                section=row["localized_section"],
                topic_title=row["localized_topic_title"],
                question=row["localized_question"],
                explanation=row["explanation"],
                answer_items=items_by_question.get(row["question_id"], []),
            )
            for row in question_rows
        ]

    def list_all_questions(
        self,
        language: str = DEFAULT_LANGUAGE,
    ) -> list[AssessmentQuestion]:
        """Return every framework question across dimensions in display order.

        Inputs:
            language: Supported response language for localized question text.

        Outputs:
            list[AssessmentQuestion]: All framework questions grouped by
            dimension order and then question order.
        """

        questions: list[AssessmentQuestion] = []
        for dimension in self.list_dimensions(language=language):
            questions.extend(
                self.list_questions(
                    dimension=dimension.dimension,
                    language=language,
                )
            )
        return questions
