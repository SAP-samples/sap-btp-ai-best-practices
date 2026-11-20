"""Non-live ECN workflow integration tests.

Run from the repository root:
    PYTHONPATH=api python -m pytest -q tests/integration/test_ecn_workflow.py

The suite uses production parsers/evidence/services/API with in-memory repository
adapters. It never connects to HANA, SQLite, or an external model provider.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from app.ecn.models import BusinessEvidenceSection

from tests.fixtures.builders import (
    duplicate_id_json_bytes,
    minimal_bom_workbook_bytes,
    minimal_ecn_json_bytes,
    replacement_bom_workbook_bytes,
)
from tests.integration.support import build_workflow_harness, six_section_json


@pytest.fixture
def minimal_ecn_json() -> bytes:
    """Return the stable two-ECN byte fixture."""
    return minimal_ecn_json_bytes()


@pytest.fixture
def minimal_bom_workbook() -> bytes:
    """Return the stable three-role BOM byte fixture."""
    return minimal_bom_workbook_bytes()


@pytest.fixture
def replacement_bom_workbook() -> bytes:
    """Return a distinct valid BOM snapshot."""
    return replacement_bom_workbook_bytes()


@pytest.fixture
def duplicate_id_json() -> bytes:
    """Return repeated and missing business identifiers."""
    return duplicate_id_json_bytes()


def test_manifest_keeps_one_bounded_api_process() -> None:
    """Prevent multi-instance processing or unbounded provider calls."""
    text = Path("manifest.yaml").read_text(encoding="utf-8")
    _, api_block, ui_block = text.split("  - name:", 2)
    assert "    path: api\n" in api_block
    assert "    instances: 1\n" in api_block
    assert "    memory: 4G\n" in api_block
    assert (
        "    command: uvicorn app.main:app --host 0.0.0.0 --port $PORT\n"
        in api_block
    )
    assert '      LLM_MAX_CONCURRENCY: "4"\n' in api_block
    assert '      LLM_REQUEST_TIMEOUT_SECONDS: "180"\n' in api_block
    assert '      LLM_MODEL: "gpt-5.4"\n' in api_block
    assert "RATE" not in api_block
    assert "    path: ui/dist\n" in ui_block

    api_route = re.search(r"^      - route: (\S+)$", api_block, re.MULTILINE)
    ui_route = re.search(r"^      - route: (\S+)$", ui_block, re.MULTILINE)
    assert api_route is not None
    assert ui_route is not None
    assert f"      API_BASE_URL: https://{api_route.group(1)}\n" in api_block
    assert f"      ALLOWED_ORIGIN: https://{ui_route.group(1)}\n" in api_block


@pytest.mark.asyncio
async def test_description_links_resolve_after_import_without_reverse_links(
    minimal_ecn_json: bytes, minimal_bom_workbook: bytes,
) -> None:
    """Real intake descriptions resolve to row UUIDs as referenced ECNs arrive."""
    sources = json.loads(minimal_ecn_json)
    original = {**sources[0], "Number": "ECN12345", "DescriptionSummary": "Original update.", "Description": "No preceding change."}
    followup = {**sources[1], "Number": "ECN67890", "DescriptionSummary": "Follow-up update.",
                "Description": "Linked to ecn 12345, ECN-12345 and ECN888; current ECN67890."}
    async with build_workflow_harness() as workflow:
        await workflow.upload(uuid4(), json.dumps([followup]).encode(), minimal_bom_workbook)
        followup_id = (await workflow.list_ecns()).items[0].ecn_row_id
        before = await workflow.get_detail(followup_id)
        assert [(item.ecn_number, item.ecn_row_id) for item in before.linked_ecns] == [
            ("ECN12345", None), ("ECN888", None),
        ]
        await workflow.upload(uuid4(), json.dumps([original]).encode(), None)
        rows = (await workflow.list_ecns()).items
        original_id = next(row.ecn_row_id for row in rows if row.ecn_number == "ECN12345")
        after = await workflow.get_detail(followup_id)
        assert [(item.ecn_number, item.ecn_row_id) for item in after.linked_ecns] == [
            ("ECN12345", original_id), ("ECN888", None),
        ]
        assert (await workflow.get_detail(original_id)).linked_ecns == []
        assert workflow.provider.call_count == 0


@pytest.mark.asyncio
async def test_first_upload_exposes_evidence_then_stores_and_reuses_one_analysis(
    minimal_ecn_json: bytes,
    minimal_bom_workbook: bytes,
) -> None:
    """Exercise publish, immediate detail, concurrent generation, and read reuse."""
    async with build_workflow_harness(hold_provider=True) as workflow:
        response = await workflow.upload(
            request_id=UUID("11111111-1111-4111-8111-111111111111"),
            ecn_bytes=minimal_ecn_json,
            bom_bytes=minimal_bom_workbook,
        )
        assert response.created is True
        assert response.imported_ecn_count == 2
        assert response.ready_count == 2
        assert workflow.provider.call_count == 0

        await workflow.generate_selected()
        await workflow.provider.wait_until_active(2)
        first_row_id = (await workflow.list_ecns()).items[0].ecn_row_id
        pending = await workflow.get_detail(first_row_id)
        assert pending.job.state.value == "RUNNING"
        material_evidence = await workflow.list_evidence(
            first_row_id, BusinessEvidenceSection.MODIFIED_MATERIALS
        )
        assert material_evidence.total_items > 0
        assert pending.analysis is None

        workflow.provider.release_all()
        await workflow.generate_selected()
        await workflow.wait_until_idle()
        completed = await workflow.get_detail(first_row_id)
        assert completed.job.state.value == "COMPLETED"
        assert completed.analysis is not None
        assert 2 <= workflow.provider.maximum_active <= 4
        calls_after_completion = workflow.provider.call_count
        assert (await workflow.get_detail(first_row_id)).analysis == completed.analysis
        assert workflow.provider.call_count == calls_after_completion


@pytest.mark.asyncio
async def test_explicit_generation_replaces_a_completed_summary(
    minimal_ecn_json: bytes,
    minimal_bom_workbook: bytes,
) -> None:
    """A fresh batch request generates a new analysis for the same ECN occurrence."""
    async with build_workflow_harness() as workflow:
        await workflow.upload(uuid4(), minimal_ecn_json, minimal_bom_workbook)
        await workflow.generate_selected()
        await workflow.wait_until_idle()
        row_id = (await workflow.list_ecns()).items[0].ecn_row_id
        prior = await workflow.get_detail(row_id)
        calls_before = workflow.provider.call_count
        workflow.provider.hold_calls()

        response = await workflow.client.post(
            "/api/summary-batches",
            json={
                "request_id": str(uuid4()),
                "selection": {"mode": "explicit", "ecn_ids": [str(row_id)]},
            },
        )
        assert response.status_code == 202
        await workflow.provider.wait_until_active(1)
        pending = await workflow.get_detail(row_id)
        assert pending.analysis == prior.analysis
        assert pending.job.state.value == "RUNNING"
        workflow.provider.release_all()
        await workflow.wait_until_idle()
        current = await workflow.get_detail(row_id)
        assert current.analysis.analysis_id != prior.analysis.analysis_id
        assert current.job.attempt_count == prior.job.attempt_count + 1
        assert workflow.provider.call_count == calls_before + 5


@pytest.mark.asyncio
async def test_reuse_then_replace_bom_keeps_old_evidence_hashes(
    minimal_ecn_json: bytes,
    minimal_bom_workbook: bytes,
    replacement_bom_workbook: bytes,
) -> None:
    """Keep prior analyses immutable while reusing and replacing BOM snapshots."""
    async with build_workflow_harness() as workflow:
        first = await workflow.upload(
            UUID("10000000-0000-4000-8000-000000000001"),
            minimal_ecn_json,
            minimal_bom_workbook,
        )
        await workflow.generate_selected()
        await workflow.wait_until_idle()
        old_hashes = {
            str(item.ecn_row_id): (
                await workflow.get_detail(item.ecn_row_id)
            ).analysis.evidence_hash
            for item in (await workflow.list_ecns()).items
        }
        second = await workflow.upload(
            UUID("10000000-0000-4000-8000-000000000002"),
            minimal_ecn_json,
            None,
        )
        third = await workflow.upload(
            UUID("10000000-0000-4000-8000-000000000003"),
            minimal_ecn_json,
            replacement_bom_workbook,
        )
        await workflow.generate_selected()
        await workflow.wait_until_idle()
        assert second.bom_reused is True
        # ECN intake always references an already-published Settings snapshot.
        assert third.bom_reused is True
        assert second.bom_artifact_id == first.bom_artifact_id
        assert third.bom_artifact_id != first.bom_artifact_id
        for row_id, old_hash in old_hashes.items():
            assert (await workflow.get_detail(row_id)).analysis.evidence_hash == old_hash


@pytest.mark.asyncio
async def test_duplicate_and_missing_business_ids_have_distinct_internal_routes(
    duplicate_id_json: bytes,
    minimal_bom_workbook: bytes,
) -> None:
    """Address duplicate and missing business IDs only by internal row UUID."""
    async with build_workflow_harness() as workflow:
        await workflow.upload(
            UUID("20000000-0000-4000-8000-000000000001"),
            duplicate_id_json,
            minimal_bom_workbook,
        )
        rows = (await workflow.list_ecns()).items
        assert len({row.ecn_row_id for row in rows}) == len(rows)
        assert (
            sum(
                row.ecn_number is None and row.display_ecn_number == "Missing ECN ID"
                for row in rows
            )
            == 1
        )


def mixed_section_json(projection: Any) -> str:
    """Return strict output containing sufficient, gap, and not-applicable states."""
    payload = json.loads(six_section_json(projection))
    forced = {str(key) for key in projection.sections_requiring_gap}
    eligible = [
        key
        for key, section in payload["sections"].items()
        if key not in forced and section["evidence_ids"]
    ]
    if len(eligible) < 2:
        raise RuntimeError("mixed response requires two evidence-backed sections")
    not_applicable = payload["sections"][eligible[1]]
    not_applicable.update(
        {
            "status": "NOT_APPLICABLE",
            "assessment_markdown": "This section is not applicable.",
            "evidence_ids": not_applicable["evidence_ids"][:1],
            "missing_evidence": [],
        }
    )
    if not forced:
        gap = payload["sections"][eligible[-1]]
        gap.update(
            {
                "status": "INSUFFICIENT_EVIDENCE",
                "assessment_markdown": "Authoritative evidence is missing.",
                "evidence_ids": gap["evidence_ids"][:1],
                "missing_evidence": ["Authoritative source"],
            }
        )
    return json.dumps(payload)


@pytest.mark.asyncio
async def test_mixed_section_states_complete_with_gaps(
    minimal_ecn_json: bytes,
    minimal_bom_workbook: bytes,
) -> None:
    """Persist all three section states as a completed analysis with gaps."""
    async with build_workflow_harness(response_factory=mixed_section_json) as workflow:
        await workflow.upload(
            UUID("30000000-0000-4000-8000-000000000001"),
            minimal_ecn_json,
            minimal_bom_workbook,
        )
        await workflow.generate_selected()
        await workflow.wait_until_idle()
        row_id = (await workflow.list_ecns()).items[0].ecn_row_id
        detail = await workflow.get_detail(row_id)
        assert detail.analysis is not None
        assert detail.analysis.completeness.value == "COMPLETED_WITH_GAPS"
        assert {
            section["status"]
            for section in detail.analysis.sections.model_dump(mode="json").values()
        } == {"SUFFICIENT", "INSUFFICIENT_EVIDENCE", "NOT_APPLICABLE"}


@pytest.mark.asyncio
async def test_one_failure_does_not_cancel_siblings_and_only_failed_can_retry(
    minimal_ecn_json: bytes,
    minimal_bom_workbook: bytes,
) -> None:
    """Contain a provider failure and accept retry only for its failed job."""
    async with build_workflow_harness(
        hold_provider=True, failing_calls={2}
    ) as workflow:
        upload = await workflow.upload(
            UUID("40000000-0000-4000-8000-000000000001"),
            minimal_ecn_json,
            minimal_bom_workbook,
        )
        await workflow.generate_selected()
        await workflow.provider.wait_until_active(2)
        workflow.provider.release_all()
        await workflow.generate_selected()
        await workflow.wait_until_idle()
        progress = await workflow.progress(upload.upload_id)
        assert progress.failed_count == 1
        assert progress.completed_count + progress.completed_with_gaps_count == 1
        assert 2 <= workflow.provider.maximum_active <= 4
        rows = (await workflow.list_ecns()).items
        failed = next(item for item in rows if item.analysis_state.value == "FAILED")
        assert failed.has_partial_analysis is True
        partial = await workflow.client.get(f"/api/ecns/{failed.ecn_row_id}/analysis-stages")
        assert partial.status_code == 200
        assert 1 <= len(partial.json()) <= 3
        completed = next(
            item
            for item in rows
            if item.analysis_state.value in {"COMPLETED", "COMPLETED_WITH_GAPS"}
        )
        await workflow.retry(failed.ecn_row_id)
        await workflow.generate_selected()
        await workflow.wait_until_idle()
        with pytest.raises(RuntimeError, match="retry|retried|failed"):
            await workflow.retry(completed.ecn_row_id)


@pytest.mark.asyncio
async def test_active_reset_guard_and_preserved_rules(
    minimal_ecn_json: bytes,
    minimal_bom_workbook: bytes,
) -> None:
    """Block active reset and preserve reusable Settings sources afterward."""
    async with build_workflow_harness(hold_provider=True) as workflow:
        await workflow.upload(
            UUID("50000000-0000-4000-8000-000000000001"),
            minimal_ecn_json,
            minimal_bom_workbook,
        )
        await workflow.generate_selected()
        await workflow.provider.wait_until_active(2)
        with pytest.raises(RuntimeError, match="queued|running"):
            await workflow.reset("RESET HISTORY")
        workflow.provider.release_all()
        await workflow.generate_selected()
        await workflow.wait_until_idle()
        counts = await workflow.reset("RESET HISTORY")
        assert counts.deleted_ecn_count == 2
        assert (
            workflow.store.rule_artifact_ids
            == workflow.initial_rule_artifact_ids
        )
        second = await workflow.upload(
            UUID("50000000-0000-4000-8000-000000000002"),
            minimal_ecn_json,
            None,
        )
        assert second.imported_ecn_count == 2


@pytest.mark.asyncio
async def test_settings_delete_blocks_pinned_file_then_removes_it_after_history_reset(
    minimal_ecn_json: bytes,
    minimal_bom_workbook: bytes,
) -> None:
    """Exercise the authenticated delete route and historical pin guard."""
    async with build_workflow_harness() as workflow:
        await workflow.upload(
            UUID("50000000-0000-4000-8000-000000000003"),
            minimal_ecn_json,
            minimal_bom_workbook,
        )
        listing = await workflow.client.get("/api/reference-datasets")
        assert listing.status_code == 200
        bom = next(item for item in listing.json()["items"] if item["dataset_type"] == "BOM")
        path = f'/api/reference-datasets/{bom["dataset_id"]}'
        blocked = await workflow.client.delete(path)
        assert blocked.status_code == 409
        assert blocked.json()["code"] == "REFERENCE_DATASET_IN_USE"

        await workflow.reset("RESET HISTORY")
        deleted = await workflow.client.delete(path)
        assert deleted.status_code == 204
        assert bom["dataset_id"] not in {
            item["dataset_id"] for item in (await workflow.client.get("/api/reference-datasets")).json()["items"]
        }
        assert (await workflow.client.delete(path)).status_code == 404


@pytest.mark.asyncio
async def test_restart_recovery_marks_seeded_leftovers_failed() -> None:
    """Apply production recovery semantics to queued and running leftovers."""
    async with build_workflow_harness(seed_restart_leftovers=True) as workflow:
        finished_at = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
        changed = await workflow.recover_interrupted(finished_at)
        assert changed == 2
        states = await workflow.progress()
        assert states.failed_count == 2
        assert states.queued_count == states.running_count == 0


@pytest.mark.asyncio
async def test_cross_ecn_evidence_is_rejected_and_reads_never_call_provider(
    minimal_ecn_json: bytes,
    minimal_bom_workbook: bytes,
) -> None:
    """Scope evidence to one occurrence and keep every read side-effect free."""
    async with build_workflow_harness() as workflow:
        await workflow.upload(
            UUID("60000000-0000-4000-8000-000000000001"),
            minimal_ecn_json,
            minimal_bom_workbook,
        )
        await workflow.generate_selected()
        await workflow.wait_until_idle()
        rows = (await workflow.list_ecns()).items
        foreign_page = await workflow.list_evidence(
            rows[1].ecn_row_id,
            BusinessEvidenceSection.MODIFIED_MATERIALS,
            include_technical=True,
        )
        assert foreign_page.items[0].technical_details is not None
        foreign_id = foreign_page.items[0].technical_details.internal_evidence_ids[0]
        calls = workflow.provider.call_count
        with pytest.raises(RuntimeError, match="not found"):
            await workflow.get_evidence(rows[0].ecn_row_id, foreign_id)
        await workflow.get_detail(rows[0].ecn_row_id)
        await workflow.list_ecns()
        assert workflow.provider.call_count == calls


@pytest.mark.asyncio
async def test_upload_commits_before_its_post_commit_projection_schedule(
    minimal_ecn_json: bytes,
    minimal_bom_workbook: bytes,
) -> None:
    """The production upload route schedules only identifiers already committed in storage."""
    async with build_workflow_harness() as workflow:
        await workflow.upload(
            UUID("70000000-0000-4000-8000-000000000001"),
            minimal_ecn_json,
            minimal_bom_workbook,
        )

        assert len(workflow.rdf.project_many_calls) == 1
        scheduled = workflow.rdf.project_many_calls[0]
        assert scheduled
        assert all(ecn_row_id in workflow.store.ecns for ecn_row_id in scheduled)
        assert all(
            workflow.rdf.require_current(ecn_row_id).status.value == "CURRENT"
            for ecn_row_id in scheduled
        )


@pytest.mark.asyncio
async def test_failed_projection_keeps_relational_reads_available_until_retry_unblocks_chat_and_summary(
    minimal_ecn_json: bytes,
    minimal_bom_workbook: bytes,
) -> None:
    """A graph failure blocks graph-dependent actions without discarding committed evidence."""
    async with build_workflow_harness(fail_rdf_projections=True) as workflow:
        await workflow.upload(
            UUID("80000000-0000-4000-8000-000000000001"),
            minimal_ecn_json,
            minimal_bom_workbook,
        )
        row = (await workflow.list_ecns()).items[0]
        detail = await workflow.get_detail(row.ecn_row_id)
        evidence = await workflow.list_evidence(
            row.ecn_row_id, BusinessEvidenceSection.MODIFIED_MATERIALS
        )
        assert detail.analysis is None
        assert evidence.total_items > 0

        summary = await workflow.client.post(
            "/api/summary-batches",
            json={
                "request_id": str(UUID("80000000-0000-4000-8000-000000000002")),
                "selection": {"mode": "explicit", "ecn_ids": [str(row.ecn_row_id)]},
            },
        )
        chat = await workflow.client.post(
            f"/api/ecns/{row.ecn_row_id}/chat",
            json={
                "assessment_id": str(row.ecn_row_id),
                "message": "Who requested this ECN?",
                "history": [],
            },
        )
        assert summary.status_code == 409
        assert summary.json()["code"] == "RDF_PROJECTION_FAILED"
        assert chat.status_code == 409
        assert chat.json()["code"] == "RDF_PROJECTION_FAILED"
        assert workflow.provider.call_count == 0

        workflow.rdf.fail_projections = False
        for item in (await workflow.list_ecns()).items:
            retry = await workflow.client.post(
                f"/api/ecns/{item.ecn_row_id}/rdf-projection/retry"
            )
            assert retry.status_code == 202
            assert (
                workflow.rdf.require_current(str(item.ecn_row_id)).status.value
                == "CURRENT"
            )

        await workflow.generate_selected()
        await workflow.wait_until_idle()
        unblocked_chat = await workflow.client.post(
            f"/api/ecns/{row.ecn_row_id}/chat",
            json={
                "assessment_id": str(row.ecn_row_id),
                "message": "Who requested this ECN?",
                "history": [],
            },
        )
        assert (await workflow.get_detail(row.ecn_row_id)).analysis is not None
        assert unblocked_chat.status_code == 200
        assert unblocked_chat.json()["answer"] == "Projection-backed chat is available."
