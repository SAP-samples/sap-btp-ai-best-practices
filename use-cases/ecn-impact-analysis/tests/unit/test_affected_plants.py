"""Plant labels shown in the ECN list: derivation, storage, lazy fill, selection.

Run:
    PYTHONPATH=api .venv/bin/python -m pytest -q tests/unit/test_affected_plants.py
"""

import json
from types import SimpleNamespace
import pytest

from sqlalchemy import create_engine, text

from app.ecn.api_models import EcnListResponse
from app.ecn.services.snapshots import (
    PLANT_LABELS_MAX_LENGTH,
    SnapshotService,
    plant_labels,
    plant_labels_json,
)
from app.routers.ecns import _fill_affected_plants
from app.routers.workspace_actions import FilteredSummarySelection, _resolve_summary_ids
from tests.unit.test_ecn_repository import ecn_list_row
from app.persistence.codec import ecn_list_item_from_row

SNAPSHOT = {"plant_impact": {"plants": [
    {"plant_code": "PL10", "plant_name": "Plant A"},
    {"plant_code": "Not declared", "plant_name": None},
]}}


def test_plant_labels_match_overview_plant_column() -> None:
    """Labels are "<code> <name>", or the code alone; no plant_impact means none."""
    assert plant_labels(SNAPSHOT) == ["PL10 Plant A", "Not declared"]
    assert plant_labels({}) == []


def test_plant_labels_json_fits_the_column() -> None:
    """Oversized label lists drop trailing labels instead of failing the write."""
    labels = [f"P{i:04d} Plant name number {i}" for i in range(200)]
    stored = plant_labels_json(labels)
    assert len(stored) <= PLANT_LABELS_MAX_LENGTH
    assert json.loads(stored) == labels[: len(json.loads(stored))]
    assert plant_labels_json(["Évreux"]) == '["Évreux"]'


def test_store_plant_labels_writes_once() -> None:
    """The first snapshot build saves labels; later builds never overwrite them."""
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(text('CREATE TABLE "ECN_IMPACT_ECN" ("ECN_ROW_ID" TEXT, "AFFECTED_PLANTS_JSON" TEXT)'))
        connection.execute(text('INSERT INTO "ECN_IMPACT_ECN" VALUES (\'e1\', NULL)'))
    service = SnapshotService(None, None, None, engine, SimpleNamespace(ecn="ECN_IMPACT_ECN"))

    service._store_plant_labels("e1", SNAPSHOT)
    service._store_plant_labels("e1", {})

    with engine.connect() as connection:
        stored = connection.execute(text('SELECT "AFFECTED_PLANTS_JSON" FROM "ECN_IMPACT_ECN"')).scalar_one()
    assert json.loads(stored) == ["PL10 Plant A", "Not declared"]


class FakeSnapshots:
    """Return fixed labels, or raise for one broken ECN, recording each call."""

    def __init__(self, broken_id):
        """Remember which ECN row fails to build."""
        self.broken_id, self.calls = broken_id, []

    def affected_plants(self, ecn_row_id):
        """Record the call and return labels or raise."""
        self.calls.append(ecn_row_id)
        if ecn_row_id == self.broken_id:
            raise RuntimeError("snapshot failed")
        return ["PL10 Plant A"]


def test_list_fill_only_computes_missing_rows_and_survives_failures() -> None:
    """Stored rows are untouched; missing rows are filled; a failing row stays unassessed."""
    rows = [
        ecn_list_row(ordinal=0, AFFECTED_PLANTS_JSON='["DE01"]'),
        ecn_list_row(ordinal=1),
        ecn_list_row(ordinal=2),
    ]
    items = [ecn_list_item_from_row(row) for row in rows]
    response = EcnListResponse(items=items, page=1, page_size=25, total_items=3, total_pages=1)
    snapshots = FakeSnapshots(broken_id=str(items[2].ecn_row_id))

    _fill_affected_plants(SimpleNamespace(snapshot_service=snapshots), response)

    assert snapshots.calls == [str(items[1].ecn_row_id), str(items[2].ecn_row_id)]
    assert [item.affected_plants for item in response.items] == [["DE01"], ["PL10 Plant A"], None]


def test_filtered_selection_passes_plant_filter() -> None:
    """Select-all-matching (generate and delete) resolves with the same plant filter."""
    seen = []

    def list_matching_ecn_ids(query):
        """Capture the resolved list query."""
        seen.append(query)
        return ["00000000-0000-4000-8000-000000000001"]

    services = SimpleNamespace(ecn_repository=SimpleNamespace(list_matching_ecn_ids=list_matching_ecn_ids))
    _resolve_summary_ids(services, FilteredSummarySelection(mode="filtered", plant=" PL10 "))
    assert seen[0].plant == "PL10"


@pytest.mark.parametrize("group", ["SUMMARIES_AVAILABLE", "IN_PROGRESS"])
def test_filtered_selection_preserves_analysis_groups_and_exclusions(group) -> None:
    """Generate and delete resolve the same grouped filters as the inbox list."""
    seen = []
    ids = ["00000000-0000-4000-8000-000000000001", "00000000-0000-4000-8000-000000000002"]

    def list_matching_ecn_ids(query):
        """Capture filter membership passed to the shared repository read."""
        seen.append(query)
        return ids

    services = SimpleNamespace(ecn_repository=SimpleNamespace(list_matching_ecn_ids=list_matching_ecn_ids))
    resolved = _resolve_summary_ids(services, FilteredSummarySelection(
        mode="filtered", analysis_state=group, query=" shaft ", plant=" PL10 ",
        excluded_ecn_ids=[ids[1]],
    ))
    assert resolved == [ids[0]]
    assert seen[0].analysis_state.value == group
    assert seen[0].query == "shaft"
    assert seen[0].plant == "PL10"
