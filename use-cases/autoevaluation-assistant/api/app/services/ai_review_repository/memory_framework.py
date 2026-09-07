"""Assessment framework reads for the in-memory AI review repository."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from typing import Any
from uuid import uuid4

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
from app.services.framework_topics import topic_title_for

from .common import (
    ACTIVE_REVIEW_TASK_STATUSES,
    ADMIN_DOCUMENT_CORPUS_ID,
    _derive_finished_batch_status,
    _derive_job_status,
    _document_payload_bytes,
    _document_phase_processed_count,
    _normalize_task_status_for_response,
    _pending_task_status_for_runtime,
    document_content_hash,
)

class MemoryFrameworkMixin:
    """Read framework fixtures from in-memory question lists."""

    def list_dimensions(self, language: str = DEFAULT_LANGUAGE) -> list[AssessmentDimension]:
        """Return dimensions derived from in-memory framework question fixtures.

        Inputs:
            language: Supported language code, accepted for interface parity.

        Outputs:
            list[AssessmentDimension]: Dimension summaries ordered by first
            occurrence in ``framework_questions``.
        """

        _ = normalize_language(language)
        dimensions: dict[str, int] = {}
        for question in self.framework_questions:
            dimensions[question.dimension] = dimensions.get(question.dimension, 0) + 1
        return [
            AssessmentDimension(
                dimension=dimension,
                display_name=dimension,
                question_count=count,
                answered_count=0,
            )
            for dimension, count in dimensions.items()
        ]

    def list_questions(
        self,
        dimension: str,
        language: str = DEFAULT_LANGUAGE,
    ) -> list[AssessmentQuestion]:
        """Return in-memory framework questions for one dimension.

        Inputs:
            dimension: Canonical framework dimension.
            language: Supported language code, accepted for interface parity.

        Outputs:
            list[AssessmentQuestion]: Deep-copied question fixtures.
        """

        language = normalize_language(language)
        return [
            self._localized_topic_question(question, language)
            for question in self.framework_questions
            if question.dimension == dimension
        ]

    def list_all_questions(
        self,
        language: str = DEFAULT_LANGUAGE,
    ) -> list[AssessmentQuestion]:
        """Return every in-memory framework question.

        Inputs:
            language: Supported language code, accepted for interface parity.

        Outputs:
            list[AssessmentQuestion]: Deep-copied all-question list.
        """

        language = normalize_language(language)
        return [
            self._localized_topic_question(question, language)
            for question in self.framework_questions
        ]

    @staticmethod
    def _localized_topic_question(
        question: AssessmentQuestion,
        language: str,
    ) -> AssessmentQuestion:
        """Copy one fixture with its approved localized topic when available.

        Inputs:
            question: In-memory framework question fixture.
            language: Normalized supported response language.

        Outputs:
            AssessmentQuestion: Independent copy with canonical title, falling
            back to an explicitly supplied fixture title or section for custom IDs.
        """

        copied = deepcopy(question)
        try:
            title = topic_title_for(copied.question_id, language)
        except ValueError:
            title = copied.topic_title or copied.section
        return copied.model_copy(update={"topic_title": title})
