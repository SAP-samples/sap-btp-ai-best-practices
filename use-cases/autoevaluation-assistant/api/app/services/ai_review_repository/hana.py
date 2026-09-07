"""Concrete HANA-backed AI review repository."""

from __future__ import annotations

from sqlalchemy.orm import Session

from .hana_admin_documents import HanaAdminDocumentsMixin
from .hana_batches import HanaBatchesMixin
from .hana_documents import HanaDocumentsMixin
from .hana_framework import HanaFrameworkMixin
from .hana_reviews import HanaReviewsMixin
from .hana_reports import HanaAssessmentReportsMixin
from .hana_scoring import HanaScoringMixin
from .hana_schema_ops import HanaSchemaOpsMixin


class HanaAiReviewRepository(
    HanaSchemaOpsMixin,
    HanaFrameworkMixin,
    HanaDocumentsMixin,
    HanaAdminDocumentsMixin,
    HanaScoringMixin,
    HanaAssessmentReportsMixin,
    HanaReviewsMixin,
    HanaBatchesMixin,
):
    """Store AI review framework data, jobs, tasks, attachments, and results in HANA.

    Inputs:
        session: Active SQLAlchemy session bound to a HANA engine.

    Outputs:
        Repository object exposing durable framework lookup and AI review task
        persistence methods.
    """

    def __init__(self, session: Session) -> None:
        """Initialize the HANA-backed repository with a database session.

        Inputs:
            session: SQLAlchemy session used for all repository queries.

        Outputs:
            None. The repository keeps the session for subsequent operations.
        """

        self.session = session

    def commit(self) -> None:
        """Commit the active repository transaction.

        Inputs:
            None. The method commits the SQLAlchemy session owned by this
            repository.

        Outputs:
            None. The session transaction is committed.
        """
        self.session.commit()
