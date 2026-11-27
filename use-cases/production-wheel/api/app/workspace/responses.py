"""Evidence-bearing analytical responses shared by HTTP and agent tools."""

from typing import Any
from pydantic import Field
from .models import StrictModel


class QueryScope(StrictModel):
    """Identify analytical grain, population and server-side point filtering."""

    dataset_id: str | None = None
    run_id: str | None = None
    point_index: int | None = None
    view: str
    source_rows: int = Field(ge=0)
    filtered_rows: int = Field(ge=0)
    storage_filters: dict[str, int] = Field(default_factory=dict)
    aggregate_before_pagination: bool


class AnalyticalResponse(StrictModel):
    """Return a bounded page with units, total population and calculation evidence."""

    rows: list[dict[str, Any]]
    total: int = Field(ge=0)
    columns: list[str]
    scope: QueryScope
    units: dict[str, str | None]
    truncated: bool
    offset: int = Field(ge=0)
    limit: int = Field(gt=0)
    evidence: dict[str, Any]
