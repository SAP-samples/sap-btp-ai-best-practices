"""Pydantic v2 models for the Handover Workspace API.

The handover workspace captures AI use case requirements after ideation.
Users fill an 8-section form assisted by an agent, using context from
multiple sources (ideation conversations, transcripts, documents).

The payload is large and flexible: drafts are partial, and several sections
are dynamic (repeatable) tables serialized as arrays of row objects. Rather
than model every column, :class:`HandoverPayload` promotes the known
scalar/object fields (all optional) and allows arbitrary extra keys through
so the dynamic-table arrays round-trip untouched.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class HandoverPayload(BaseModel):
    """Full (possibly partial) handover form JSON.

    All known scalar fields are optional; dynamic-table arrays and any other
    keys the client sends are accepted via ``extra='allow'`` and preserved.
    """

    model_config = ConfigDict(extra="allow")

    # --- Section 1: workshop / customer context --------------------------
    customer: Optional[str] = None
    industry: Optional[str] = None
    workshop_date: Optional[str] = None
    workshop_format: Optional[str] = None
    advisor_name: Optional[str] = None
    primary_contact: Optional[str] = None
    tech_contact: Optional[str] = None

    # --- Section 2: use case ---------------------------------------------
    usecase_title: Optional[str] = None
    priority: Optional[str] = None
    problem_statement: Optional[str] = None

    # --- Section 3: current process (as-is) ------------------------------
    process_name: Optional[str] = None
    process_freq: Optional[str] = None
    process_volume: Optional[str] = None
    process_team: Optional[str] = None
    process_people: Optional[str] = None

    # --- Section 4: data -------------------------------------------------
    data_input: Optional[str] = None
    data_output: Optional[str] = None
    data_usage: Optional[str] = None

    # --- Section 5: to-be / scope ----------------------------------------
    tobe_narrative: Optional[str] = None
    in_scope: Optional[str] = None
    out_scope: Optional[str] = None
    done_definition: Optional[str] = None
    open_notes: Optional[str] = None

    # --- Fixed-row object: business impact (section 1.2) -----------------
    # Keys: volume, time_spent, error_rate, cost_per_error, annual_cost,
    # downstream — each mapping to {current, target}.
    business_impact: Optional[dict[str, Any]] = None

    # --- AI capabilities (array of selected labels) ----------------------
    ai_capabilities: Optional[list[Any]] = None

    # --- Dynamic (repeatable) tables — arrays of row objects -------------
    data_sources: Optional[list[dict[str, Any]]] = None
    business_rules: Optional[list[dict[str, Any]]] = None
    thresholds: Optional[list[dict[str, Any]]] = None
    approvals: Optional[list[dict[str, Any]]] = None
    edge_cases: Optional[list[dict[str, Any]]] = None
    kpis: Optional[list[dict[str, Any]]] = None
    tech_components: Optional[list[dict[str, Any]]] = None
    integrations: Optional[list[dict[str, Any]]] = None
    contacts: Optional[list[dict[str, Any]]] = None

    # --- Final checklist: array of {label, checked} ----------------------
    checklist: Optional[list[dict[str, Any]]] = None

    def to_storage_dict(self) -> dict[str, Any]:
        """Return the payload as a plain dict for the persistence layer.

        Includes any extra keys the client sent. ``None`` values are kept so
        the stored payload faithfully reflects what was submitted.
        """
        return self.model_dump(mode="json")


class HandoverCreateRequest(BaseModel):
    """Request body for creating a new handover workspace."""

    linked_conversation_id: Optional[str] = None
    payload: Optional[HandoverPayload] = None


class HandoverUpdateRequest(BaseModel):
    """Request body for updating handover fields (partial merge)."""

    payload: HandoverPayload


class HandoverResponse(BaseModel):
    """Response for a single handover workspace."""

    id: str
    status: str
    customer: Optional[str] = None
    usecase_title: Optional[str] = None
    priority: Optional[str] = None
    linked_conversation_id: Optional[str] = None
    completion_pct: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    submitted_at: Optional[str] = None


class HandoverListResponse(BaseModel):
    """Response for listing handover workspaces."""

    items: list[HandoverResponse]
    total: int


class HandoverFileMetadata(BaseModel):
    """Metadata for an uploaded file (never includes the BLOB data)."""

    file_id: str
    filename: str
    size: int
    zone: str
    content_type: Optional[str] = None
    created_at: Optional[str] = None
