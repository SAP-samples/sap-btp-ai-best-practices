"""Public contracts for one-file analyses and revisioned funding runs."""

from typing import Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field


class CandidateScope(BaseModel):
    """Identify all eligible rows or an exact subset within one saved analysis."""
    model_config = ConfigDict(extra="forbid")
    mode: Literal["all_eligible", "selected"] = "all_eligible"
    row_ids: list[str] = Field(default_factory=list)


class FieldIssue(BaseModel):
    """Explain a validation problem and the source rows affected by it."""
    path: str
    message: str
    row_ids: list[str] = Field(default_factory=list)


class CreateRun(BaseModel):
    """Create a draft from saved source rows without uploading another file."""
    model_config = ConfigDict(extra="forbid")
    analysis_id: str
    candidate_scope: CandidateScope = Field(default_factory=CandidateScope)


class DeleteAnalyses(BaseModel):
    """Bound one authenticated bulk deletion to explicit saved analysis IDs."""
    model_config = ConfigDict(extra="forbid")
    analysis_ids: list[UUID] = Field(min_length=1, max_length=100)


class InsightsRequest(BaseModel):
    """Freeze selected rows or a filtered current population and comparable history filters."""
    model_config = ConfigDict(extra="forbid")
    row_ids: list[str] | None = None
    filters: dict[str,str] = Field(default_factory=dict)
    lookback_days: int = Field(90, ge=1, le=3650)


class SettingsRequest(BaseModel):
    """Submit a reviewed settings draft against an explicit saved revision."""
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(ge=0)
    settings: dict


class RunStatus(BaseModel):
    """Expose durable state and the revision required by every mutable operation."""
    run_id: str
    analysis_id: str
    revision: int
    status: Literal["draft", "estimating_lifetimes", "awaiting_lifetime_acknowledgement",
                    "optimizing", "completed", "failed", "cancelled"]
    row_ids: list[str] = Field(default_factory=list)
    preparation_id: str | None = None
    readiness_issues: list[FieldIssue] = Field(default_factory=list)
    settings: dict = Field(default_factory=dict)
    preparation: dict | None = None
    result: dict | None = None


class WorkspaceValidationError(ValueError):
    """Carry structured field issues from domain validation to the HTTP boundary."""

    def __init__(self, message: str, path: str = "candidate_scope", row_ids=None):
        """Build one field issue from a human message and optional source IDs."""
        super().__init__(message)
        self.fields = [FieldIssue(path=path, message=message, row_ids=row_ids or []).model_dump()]


class RevisionConflict(ValueError):
    """Signal stale or immutable input without exposing database exceptions."""


class RevisionRequest(BaseModel):
    """Identify the saved revision being explicitly prepared, cancelled or recovered."""
    model_config = ConfigDict(extra='forbid')
    expected_revision: int = Field(ge=0)


class AcknowledgementRequest(RevisionRequest):
    """Accept the exact displayed fallback snapshot without client-supplied actor identity."""
    preparation_id: str
    accepted: bool


class ActivateLifecycleDataset(BaseModel):
    """Select which saved lifecycle dataset future lifetime preparations use as RPT-1 context."""

    dataset_id: str = Field(min_length=1, max_length=128)
