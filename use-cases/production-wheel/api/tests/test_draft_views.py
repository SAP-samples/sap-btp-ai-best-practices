"""Regression tests for compact, model-facing draft and validation views.

A full-portfolio plant profile produces a draft and a draft validation far larger
than the per-tool context limit. Before this view existed, the context budget
replaced the validation with an "output omitted" placeholder, and the assistant
could not read ``valid`` or ``pv_mode_scope`` and refused to launch. These tests
build a synthetic profile of the same size and check that:

* the summaries are exact (counts, statuses, volumes);
* every validation field the assistant reasons about is kept unchanged;
* the draft tools return results below the configured per-tool limit while the
  browser event still carries the full draft;
* a summary echoed back into a draft patch is rejected, never saved.

Run with:
    cd api && ../.venv/bin/python -m pytest tests/test_draft_views.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agent.config import load_config
from app.agent.tools.draft_views import (
    compact_config,
    compact_draft,
    compact_validation,
    summarize_group_size_overrides,
    summarize_matrix_pairs,
)
from app.agent.tools.workspace_tools import workspace_tools

# Same size as the real full-portfolio profile that triggered the regression.
VOLUME_COUNT = 51
BLOCK_COUNT = 762


def _bytes(value) -> int:
    """Measure a payload the way the context budget does (compact UTF-8 JSON)."""
    return len(json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":")).encode())


def _tool_limit() -> int:
    """Read the production per-tool byte limit from the agent configuration."""
    config = load_config(Path(__file__).parents[1] / "app/agent/config/agent.yaml")
    return config.context_budget.max_tool_result_bytes


def _matrix_rows() -> list[dict]:
    """Build every ordered volume pair; same-volume and neighbours are compatible."""
    volumes = [f"{index * 0.25:g}" for index in range(1, VOLUME_COUNT + 1)]
    rows = []
    for a_index, volume_a in enumerate(volumes):
        for b_index, volume_b in enumerate(volumes):
            gap = abs(a_index - b_index)
            status = "Y" if gap <= 1 else "AVOID" if gap <= 3 else "N"
            rows.append({"volume_a": volume_a, "volume_b": volume_b, "status": status})
    return rows


def _overrides() -> list[dict]:
    """Build one group-size cap per synthetic plant/SEFI block."""
    return [{"plant": "P1", "sefi": f"S{index:04d}", "maximum": 5} for index in range(BLOCK_COUNT)]


def _config() -> dict:
    """Return an optimizer config carrying both large profile-owned lists."""
    return {
        "pv_mode": "OPTIMIZED",
        "matrix_mode": "PREFER_COMPATIBLE",
        "matrix_pairs": _matrix_rows(),
        "group_size_overrides": _overrides(),
        "business_rules": [{"constraint_id": "rule-1", "description": "Keep lines apart"}],
    }


def _draft() -> dict:
    """Return a large draft shaped like WorkspaceService.get_draft output."""
    return {
        "draft_id": "draft-1",
        "dataset_id": "dataset-1",
        "revision": 2,
        "title": "Large plant",
        "request": {"scope": [{"plant": "P1"}], "constraints": [], "config": _config()},
        "plant_profile": {"profile_id": "profile-1", "revision": 5, "plant": "P1",
                          "settings": {"maximum_group_size": 5}, "rules": [],
                          "matrix_rows": _matrix_rows()},
        "budget": {"frontier_points": 5},
    }


def _validation() -> dict:
    """Return a large successful validation shaped like WorkspaceService.validate_draft."""
    return {
        "valid": True,
        "errors": [],
        "warnings": ["Business acceptance is not automatically assessed."],
        "pv_mode_scope": {"pv_mode": "OPTIMIZED", "other_mode": "FIXED", "modeled_count": 475,
                          "other_mode_modeled_count": 470, "only_in_other_mode": [],
                          "only_in_this_mode": ["F1", "F2"], "warnings": ["2 FINIs only in OPTIMIZED"]},
        "effective_request": {"scope": [{"plant": "P1"}], "constraints": [], "config": _config()},
        "counts": {"modeled_fini_count": 475, "block_count": 43},
        "applied_constraints": [{"effect": "scope", "kept_fini_count": 475}],
        "matrix_coverage": {"complete": True, "missing_volumes": []},
        "profile_status": {"outdated": False, "draft_profile_revision": 5},
        "revision": 2,
    }


def test_matrix_summary_is_exact():
    """Pair counts per status and the volume list match the full matrix."""
    rows = _matrix_rows()
    summary = summarize_matrix_pairs(rows)
    assert summary["pair_count"] == VOLUME_COUNT ** 2
    assert sum(summary["pairs_by_status"].values()) == len(rows)
    assert summary["pairs_by_status"]["Y"] == sum(row["status"] == "Y" for row in rows)
    assert summary["volume_count"] == VOLUME_COUNT
    assert summary["volumes"][0] == "0.25" and summary["volumes"][-1] == "12.75"


def test_group_size_summary_counts_blocks_per_cap():
    """Every block is counted under its cap and a short sample is kept."""
    summary = summarize_group_size_overrides(_overrides())
    assert summary["block_count"] == BLOCK_COUNT
    assert summary["blocks_by_maximum"] == {"5": BLOCK_COUNT}
    assert len(summary["sample"]) == 5


def test_compact_config_keeps_small_settings_and_does_not_mutate_input():
    """Scalar settings and short lists stay intact; the source config is untouched."""
    config = _config()
    compact = compact_config(config)
    assert compact["pv_mode"] == "OPTIMIZED"
    assert compact["business_rules"] == config["business_rules"]
    assert compact["matrix_pairs"]["summarized"] is True
    assert isinstance(config["matrix_pairs"], list) and len(config["matrix_pairs"]) == VOLUME_COUNT ** 2
    small = compact_config({"group_size_overrides": _overrides()[:2]})
    assert small["group_size_overrides"] == _overrides()[:2]


def test_compact_config_summarizes_any_other_oversized_list():
    """A future large config list is reduced to a count and a sample."""
    compact = compact_config({"future_list": list(range(500))})
    assert compact["future_list"] == {"summarized": True, "item_count": 500, "sample": [0, 1, 2, 3, 4]}


def test_compact_validation_keeps_every_check_field_and_fits_the_limit():
    """The assistant still reads valid, errors and pv_mode_scope after compaction."""
    validation = _validation()
    assert _bytes(validation) > _tool_limit()
    compact = compact_validation(validation)
    for key in ("valid", "errors", "warnings", "pv_mode_scope", "counts",
                "applied_constraints", "matrix_coverage", "profile_status", "revision"):
        assert compact[key] == validation[key]
    assert compact["effective_request"]["scope"] == validation["effective_request"]["scope"]
    assert _bytes(compact) < _tool_limit()


def test_compact_draft_keeps_profile_identity_and_fits_the_limit():
    """The frozen profile keeps its revision; its matrix becomes counts and volumes."""
    draft = _draft()
    assert _bytes(draft) > _tool_limit()
    compact = compact_draft(draft)
    assert compact["revision"] == 2 and compact["budget"] == draft["budget"]
    assert compact["plant_profile"]["revision"] == 5
    assert "matrix_rows" not in compact["plant_profile"]
    assert compact["plant_profile"]["matrix_pair_count"] == VOLUME_COUNT ** 2
    assert len(compact["plant_profile"]["matrix_volumes"]) == VOLUME_COUNT
    assert _bytes(compact) < _tool_limit()


class LargeDraftService:
    """Serve one large draft so the real tool wrappers can be measured."""

    def __init__(self):
        """Start with an empty conversation selection."""
        self.selection = {}

    def context(self, context_id, patch=None):
        """Read or update the selected IDs for this test conversation."""
        self.selection.update(patch or {})
        return dict(self.selection)

    def get_draft(self, draft_id):
        """Return the large draft."""
        return _draft()

    def update_draft(self, draft_id, revision, patch):
        """Return the large draft at the next revision."""
        return {**_draft(), "revision": revision + 1}

    def validate_draft(self, draft_id):
        """Return the large successful validation."""
        return _validation()

    def recreate_draft_with_current_profile(self, draft_id):
        """Return a large recreated draft with carry-over details."""
        return {"draft": {**_draft(), "draft_id": "draft-2"}, "carried_over": ["title"], "skipped": []}


def test_draft_tools_return_results_below_the_per_tool_limit():
    """Draft read, update, validation and recreation all fit; the UI event stays full."""
    events = []
    tools = {tool.name: tool for tool in workspace_tools(LargeDraftService(), "ctx", events.append)}
    limit = _tool_limit()

    validation = tools["validate_run_draft"].invoke({"draft_id": "draft-1"})
    assert validation["valid"] is True and validation["pv_mode_scope"]["only_in_this_mode"] == ["F1", "F2"]
    assert _bytes(validation) < limit

    assert _bytes(tools["update_run_draft"].invoke({"draft_id": "draft-1"})) < limit

    updated = tools["update_run_draft"].invoke(
        {"draft_id": "draft-1", "revision": 2, "patch": {"title": "Renamed"}})
    assert updated["revision"] == 3 and _bytes(updated) < limit
    assert isinstance(events[-1]["draft"]["request"]["config"]["matrix_pairs"], list)

    recreated = tools["recreate_draft_with_current_profile"].invoke({"draft_id": "draft-1"})
    assert recreated["carried_over"] == ["title"] and _bytes(recreated) < limit
    assert isinstance(events[-1]["draft"]["plant_profile"]["matrix_rows"], list)


def test_echoed_matrix_summary_is_rejected_by_the_service():
    """A summary copied into a draft patch cannot replace the profile-owned list."""
    from test_plant_profiles import fixture

    service, source, profile = fixture()
    draft = service.create_draft(source["dataset_id"], profile["profile_id"])
    summary = compact_draft(service.get_draft(draft["draft_id"]))["request"]["config"]["matrix_pairs"]
    with pytest.raises(ValueError):
        service.update_draft(draft["draft_id"], draft["revision"],
                             {"request": {"config": {"matrix_pairs": summary}}})


def test_validate_tool_on_the_real_service_keeps_validation_fields():
    """Through the in-memory workspace service, the tool returns the same checks."""
    from test_plant_profiles import fixture

    service, source, profile = fixture()
    draft = service.create_draft(source["dataset_id"], profile["profile_id"])
    full = service.validate_draft(draft["draft_id"])
    tools = {tool.name: tool for tool in workspace_tools(service, "ctx")}
    compact = tools["validate_run_draft"].invoke({"draft_id": draft["draft_id"]})
    for key in ("valid", "errors", "pv_mode_scope", "counts", "profile_status", "revision"):
        assert compact[key] == full[key]


def test_dataset_list_entries_are_compact_and_keep_identity():
    """The assistant's dataset list keeps identity and finding counts, not full records."""
    from app.agent.tools.dataset_views import dataset_list_entry

    record = {
        "dataset_id": "dataset-1", "name": "Plant import", "status": "published", "revision": 2,
        "created_at": "2026-11-17T10:00:00+00:00",
        "metadata": {"plant": "P1", "block_count": 43, "settings": {"demand_days": 250},
                     "admission_counts": {"modeled": 475},
                     "source_tables": [{"table": f"t{index}", "columns": list(range(40))} for index in range(60)],
                     "input_hashes": {"primary": "a" * 64}},
        "issues": [{"rule_id": "PRODUCTION_FREQUENCY_UNDETERMINED", "severity": "warning"},
                   {"rule_id": "FINI_DATA_QUALITY", "severity": "warning"},
                   {"rule_id": "FINI_DATA_QUALITY", "severity": "error"}],
    }
    entry = dataset_list_entry(record)
    assert entry["dataset_id"] == "dataset-1" and entry["plant"] == "P1" and entry["status"] == "published"
    assert entry["settings"] == {"demand_days": 250} and entry["admission_counts"] == {"modeled": 475}
    assert entry["issue_counts"] == {"error": 1, "warning": 2}
    assert entry["issue_rule_ids"] == ["FINI_DATA_QUALITY", "PRODUCTION_FREQUENCY_UNDETERMINED"]
    assert "source_tables" not in entry and "issues" not in entry
    assert _bytes(entry) < _bytes(record) / 10
