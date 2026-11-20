"""Snapshot workflow acceptance without remote calls.

Run: PYTHONPATH=api .venv/bin/python -m pytest -q tests/unit/test_snapshot_workspace.py
"""

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from app.ecn.evidence.snapshot import build_snapshot, snapshot_page
from app.ecn.services.snapshots import SnapshotService, material_brief
from app.ecn.models import BomKind, BomRowClassification
from app.persistence.workspace_repository import WorkspaceRepository
from sqlalchemy import create_engine, text

from tests.fixtures.builders import (
    make_bom_row,
    make_normalized_ecn,
    make_source_object,
    synthetic_uuid,
)


def test_reference_roles_and_ancestor_context_do_not_invent_changed_descendants():
    """Count direct objects once, preserve multiple depths, and separate BOM-only evidence."""
    ecn = make_normalized_ecn(
        ecn_number="ECN1",
        source_ordinal=25,
        affected_objects=(make_source_object(number="PART"),),
        resulting_objects=(),
    )
    parent = make_bom_row(
        bom_row_id=synthetic_uuid("parent"), component_number="VC", dotted_level="3"
    )
    first = make_bom_row(
        bom_row_id=synthetic_uuid("first"),
        component_number="PART",
        dotted_level="4",
        parent_bom_row_id=parent.bom_row_id,
    )
    second = replace(first, bom_row_id=synthetic_uuid("second"), dotted_level="5")
    unrelated = replace(
        first, bom_row_id=synthetic_uuid("unrelated"), component_number="OTHER"
    )
    raw = replace(
        first,
        bom_row_id=synthetic_uuid("raw"),
        bom_kind=BomKind.RAW_EBOM_130,
        component_number="STOCKPART",
        change_number="ECN1",
        row_classification=BomRowClassification.MATERIAL_LIKE_OBSERVED,
    )
    stock = {
        "material_number": "STOCKPART",
        "dataset_type": "STOCK",
        "unrestricted_quantity": "0",
        "plant_code": "P1",
    }
    snapshot = build_snapshot(ecn, (parent, first, second, unrelated, raw), (stock,))
    page = snapshot_page(snapshot, ancestor="VC")
    assert page["total"] == 1
    assert page["items"][0]["observed_levels"] == ["4", "5"]
    assert snapshot["counts"]["technical_objects"] == 1
    assert snapshot["counts"]["bom_change_objects"] == 1
    assert snapshot_page(snapshot, origin="bom")["items"][0]["number"] == "STOCKPART"
    assert snapshot["objects"][0]["source_references"][0]["path"].startswith("$[25]")
    assert next(o for o in snapshot["objects"] if o["number"] == "STOCKPART")[
        "reference_records"
    ] == [stock]
    assert not any(o["number"] == "OTHER" for o in snapshot["objects"])


def test_active_price_overlay_rows_use_the_matching_current_scope_only():
    """A display-only source price follows the active overlay, not ECN price pins."""
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(text("""
            CREATE TABLE "ECN_IMPACT_PRICE_TEST" (
                "DATASET_ID" TEXT,
                "SOURCE_ROW_NUMBER" INTEGER,
                "MATERIAL_NUMBER" TEXT,
                "PLANT_CODE" TEXT,
                "NET_PRICE" TEXT,
                "CURRENCY" TEXT,
                "PRICE_PER" TEXT,
                "ORDER_PRICE_UNIT" TEXT
            )
        """))
        connection.execute(text("""
            INSERT INTO "ECN_IMPACT_PRICE_TEST" (
                "DATASET_ID", "SOURCE_ROW_NUMBER", "MATERIAL_NUMBER", "PLANT_CODE",
                "NET_PRICE", "CURRENCY", "PRICE_PER", "ORDER_PRICE_UNIT"
            ) VALUES ('price-current', 2, 'PIPE-1', 'PK01', '12.50', 'EUR', '1', 'PC')
        """))

    current = SimpleNamespace(
        active=True,
        dataset_type="PURCHASE_INFO_PRICES",
        dataset_id="price-current",
        context={"model_range": "Q900V", "data_group": "customer", "environment": "QA"},
    )
    other_scope = SimpleNamespace(
        active=True,
        dataset_type="PURCHASE_INFO_PRICES",
        dataset_id="price-other-scope",
        context={"model_range": "Q900V", "data_group": "customer", "environment": "PRODUCTION"},
    )
    service = object.__new__(SnapshotService)
    service.references = SimpleNamespace(list=lambda **kwargs: [other_scope, current])
    service.engine = engine
    service.tables = SimpleNamespace(purchase_info_price_row="ECN_IMPACT_PRICE_TEST")

    rows = service._active_price_overlay_rows({
        "datasets": [{
            "dataset_type": "PURCHASE_INFO_RECORDS",
            "comparison": False,
            "model_range": "Q900V",
            "data_group": "customer",
            "environment": "QA",
        }],
    }, "PIPE-1")

    assert rows == [{
        "dataset_id": "price-current",
        "source_row_number": 2,
        "material_number": "PIPE-1",
        "plant_code": "PK01",
        "net_price": "12.50",
        "currency": "EUR",
        "price_per": "1",
        "order_price_unit": "PC",
        "dataset_type": "PURCHASE_INFO_PRICES",
        "active_overlay": True,
    }]


def test_material_brief_deduplicates_current_bom_links_and_keeps_sourcing_separate():
    """The material dialog gets described links and plant-specific sourcing, not raw occurrences."""
    obj = {
        "number": "DECAL", "reference_records": [
            {"dataset_type": "MATERIAL_MASTER", "plant_code": "PK01", "procurement_type_label": "In-house and external procurement allowed"},
            {"dataset_type": "PURCHASE_INFO_PRICES", "material_number": "DECAL", "plant_code": "PK01", "net_price": "0.85", "currency": "EUR", "price_per": "1", "order_price_unit": "PC"},
            {"dataset_type": "MATERIAL_MASTER", "plant_code": "PK01", "comparison": True, "procurement_type_label": "External procurement"},
        ],
        "locations": [
            {
                "kind": "ENRICHED_EBOM", "parent": "ASSEMBLY", "model_range": "Q900V",
                "path": [{"number": "Q900V", "description": "Q900V tractor"}, {"number": "ASSEMBLY", "description": "Control-panel assembly"}, {"number": "DECAL", "description": "Warning decal"}],
                "plant_scope": {"plants": [{"plant_code": "PK01", "plant_name": "Plant B"}]},
                "usage_status": "EFFECTIVE", "valid_from_date": "2026-05-19", "quantity": "2", "unit": "PC",
            },
            {
                "kind": "ENRICHED_EBOM", "parent": "ASSEMBLY", "model_range": "Q900V",
                "path": [{"number": "Q900V", "description": "Q900V tractor"}, {"number": "ASSEMBLY", "description": "Control-panel assembly"}, {"number": "DECAL", "description": "Warning decal"}],
                "plant_scope": {"plants": [{"plant_code": "PK01", "plant_name": "Plant B"}]},
                "usage_status": "EFFECTIVE", "valid_from_date": "2026-05-19", "quantity": "2", "unit": "PC",
            },
            {"comparison": True, "kind": "ENRICHED_EBOM", "parent": "HISTORICAL", "path": []},
        ],
    }
    snapshot = {
        "plant_impact": {"plants": [{"plant_code": "PK01", "plant_name": "Plant B"}]},
        # MOCK_DATA_RETIREMENT_VALIDATION_DISABLED: "purchase_prices": [{"material_number": "DECAL", "plant_code": "PK01", "net_price": "0.85", "currency": "EUR", "price_per": "1", "unit_of_measure": "PC", "status": "ACTIVE"}],
        # MOCK_DATA_RETIREMENT_VALIDATION_DISABLED: "demo_assumptions": ["Cost values are illustrative demonstration data."],
    }

    brief = material_brief(snapshot, obj)

    assert brief["linked_use"] == [{
        "assembly": "ASSEMBLY", "assembly_description": "Control-panel assembly", "model_range": "Q900V",
        "plant": None, "kind": "ENRICHED_EBOM",
        "path": [
            {"number": "Q900V", "description": "Q900V tractor"},
            {"number": "ASSEMBLY", "description": "Control-panel assembly"},
            {"number": "DECAL", "description": "Warning decal"},
        ],
        "plants": [{"plant_code": "PK01", "plant_name": "Plant B"}], "bom": "EBOM", "usage_status": "EFFECTIVE",
        "valid_from_date": "2026-05-19", "valid_to_date": None, "quantity": "2", "unit": "PC",
    }]
    assert brief["bom_coverage"] == {
        "engineering_occurrences": 1,
        "manufacturing_occurrences": 0,
        "coverage": "ENGINEERING_ONLY",
        "plants": [],
        "effective_interval_count": 1,
        "occurrences": [{
            "source": "Enriched EBOM", "kind": "ENRICHED_EBOM",
            "path": [
                {"number": "Q900V", "description": "Q900V tractor"},
                {"number": "ASSEMBLY", "description": "Control-panel assembly"},
                {"number": "DECAL", "description": "Warning decal"},
            ],
            "item_or_position": None, "quantity": "2", "unit": "PC",
            "valid_from_date": "2026-05-19", "valid_from_quality": None, "valid_from": None,
            "valid_to_date": None, "valid_to_quality": None, "valid_to": None,
            "change_number": None, "change_number_to": None, "plant": None,
        }],
    }
    assert brief["sourcing"] == [{
        "plant_code": "PK01", "plant_name": "Plant B",
        "procurement_type_labels": ["In-house and external procurement allowed"],
        "source_lead_time": {"status": "NOT_SUPPLIED"},
        "source_purchase_price": {
            "status": "SOURCE_PRICE_AVAILABLE",
            "matching_source_row_count": 1,
            "net_price": "0.85",
            "currency": "EUR",
            "price_per": "1",
            "order_price_unit": "PC",
        },
    }]
    assert "make_buy_assessments" not in brief
    assert "cost_assumption" not in brief


def test_material_brief_keeps_ambiguous_and_zero_prices_as_review_states():
    """Do not select a source price when its meaning is not deterministic."""
    obj = {
        "number": "PIPE-1",
        "reference_records": [
            {"dataset_type": "MATERIAL_MASTER", "plant_code": "PK01", "procurement_type_label": "External procurement"},
            {"dataset_type": "MATERIAL_MASTER", "plant_code": "PL10", "procurement_type_label": "External procurement"},
            {"dataset_type": "PURCHASE_INFO_PRICES", "material_number": "PIPE-1", "plant_code": "PK01", "net_price": "4.00"},
            {"dataset_type": "PURCHASE_INFO_PRICES", "material_number": "PIPE-1", "plant_code": "PK01", "net_price": "4.25"},
            {"dataset_type": "PURCHASE_INFO_PRICES", "material_number": "PIPE-1", "plant_code": "PL10", "net_price": "0"},
        ],
        "locations": [],
    }

    sourcing = material_brief({}, obj)["sourcing"]

    assert sourcing == [
        {
            "plant_code": "PK01",
            "plant_name": None,
            "procurement_type_labels": ["External procurement"],
            "source_lead_time": {"status": "NOT_SUPPLIED"},
            "source_purchase_price": {
                "status": "MULTIPLE_SOURCE_PRICES_REVIEW",
                "matching_source_row_count": 2,
            },
        },
        {
            "plant_code": "PL10",
            "plant_name": None,
            "procurement_type_labels": ["External procurement"],
            "source_lead_time": {"status": "NOT_SUPPLIED"},
            "source_purchase_price": {
                "status": "ZERO_PRICE_REVIEW",
                "matching_source_row_count": 1,
            },
        },
    ]

    obj = {
        "number": "DECAL",
        "reference_records": [],
        "locations": [
            {
                "occurrence_id": "engineering-1", "kind": "ENRICHED_EBOM",
                "path": [{"number": "PRODUCT"}, {"number": "DECAL"}],
                "item_or_position": "0010", "quantity": "1", "unit": "PC",
                "valid_from_date": "2026-05-19", "valid_to_date": "9999-12-31",
                "change_number": "ECN345274",
            },
            {
                "occurrence_id": "manufacturing-1", "kind": "MBOM_130",
                "path": [{"number": "ASSEMBLY"}, {"number": "DECAL"}],
                "item_or_position": "10", "quantity": "1", "unit": None,
                "valid_from_date": "2026-05-19", "valid_to_date": "9999-12-31",
                "change_number": "ECN345274", "change_number_to": "ECN345275",
            },
        ],
    }

    coverage = material_brief({}, obj)["bom_coverage"]

    assert coverage["engineering_occurrences"] == 1
    assert coverage["manufacturing_occurrences"] == 1
    assert coverage["coverage"] == "OBSERVED_IN_BOTH"
    assert [row["source"] for row in coverage["occurrences"]] == ["Enriched EBOM", "130% MBOM"]
    assert coverage["occurrences"][1]["change_number_to"] == "ECN345275"
    assert "mapping" not in coverage
    assert all("occurrence_id" not in row for row in coverage["occurrences"])


# MOCK_DATA_RETIREMENT_VALIDATION_DISABLED_START: curated make-or-buy material-brief test.
# def test_material_brief_exposes_curated_make_buy_assessments_with_current_cost_and_scope():
#     obj = {"number": "DECAL", "reference_records": [], "locations": []}
#     snapshot = {
#         "plant_impact": {"plants": [{"plant_code": "PL10", "plant_name": "Plant A"}]},
#         "purchase_prices": [{"material_number": "DECAL", "plant_code": "PL10", "net_price": "0.85", "currency": "EUR", "price_per": "1", "unit_of_measure": "PC", "status": "ACTIVE"}],
#         "make_buy_recommendations": [{
#             "material_number": "DECAL", "plant_code": "PL10", "recommendation": "BUY",
#             "make_cost": "1.10", "currency": "EUR", "price_per": "1", "unit_of_measure": "PC",
#             "comparable_history": "Externally sourced.", "tooling": "No local print process.",
#             "capacity": "Available.", "source": "CURATED_DEMONSTRATION",
#         }],
#     }
#
#     assessment = material_brief(snapshot, obj)["make_buy_assessments"]
#
#     assert assessment[0]["plant_name"] == "Plant A"
#     assert assessment[0]["buy_cost"]["net_price"] == "0.85"
#     assert assessment[0]["impacted_plant_count"] == 1
#     assert assessment[0]["comparable_history"] == "Externally sourced."
#     assert assessment[0]["source"] == "CURATED_DEMONSTRATION"
# MOCK_DATA_RETIREMENT_VALIDATION_DISABLED_END: curated make-or-buy material-brief test.
def test_unnumbered_documents_have_stable_inspection_identity():
    """A document remains inspectable without falsely joining a technical number."""
    doc = make_source_object(
        number=None, technical_id="DOC:1", is_technical_wtpart=False
    )
    ecn = make_normalized_ecn(affected_objects=(doc,), resulting_objects=())
    snapshot = build_snapshot(ecn, ())
    assert snapshot_page(snapshot, technical=False)["items"][0][
        "object_key"
    ].startswith("object-")
    assert snapshot["counts"]["technical_objects"] == 0


def test_readiness_without_source_citations_remains_a_valid_unknown_assessment():
    """An empty snapshot cannot fail validation merely because it has no named follow-ups."""
    from app.ecn.llm.validation import _deterministic_readiness

    result = _deterministic_readiness(SimpleNamespace(records=[], header_citation_ids=[],
        snapshot_context={"version":"ecn-snapshot-v2", "counts":{"located_objects":0,
            "technical_objects":0, "unlocated_objects":0}, "questions":[]}))
    assert result.severity.value == "UNKNOWN"
    assert result.missing_evidence


@pytest.fixture
def workspace_repository():
    """Use real transactions for portable action SQL; HANA DDL is verified separately."""
    engine = create_engine("sqlite://")
    repository = WorkspaceRepository(engine)
    with engine.begin() as connection:
        connection.execute(
            text(
                f'CREATE TABLE "{repository.tables.analysis_job}" (JOB_ID TEXT PRIMARY KEY, ECN_ROW_ID TEXT, STATE TEXT, MODEL_NAME TEXT, PROMPT_VERSION TEXT, OUTPUT_SCHEMA_VERSION TEXT, ERROR_CODE TEXT, ERROR_MESSAGE TEXT, STARTED_AT TEXT, FINISHED_AT TEXT)'
            )
        )
        connection.execute(
            text(
                f'CREATE TABLE "{repository.tables.generation_batch}" (BATCH_ID TEXT PRIMARY KEY, SELECTION_JSON TEXT, CREATED_AT TEXT, CANCELLED INTEGER)'
            )
        )
        connection.execute(
            text(
                f'CREATE TABLE "{repository.tables.batch_job}" (JOB_ID TEXT PRIMARY KEY, BATCH_ID TEXT)'
            )
        )
        connection.execute(
            text(
                f'CREATE TABLE "{repository.tables.analysis}" (ANALYSIS_ID TEXT PRIMARY KEY, ECN_ROW_ID TEXT UNIQUE, JOB_ID TEXT UNIQUE)'
            )
        )
        connection.execute(
            text(
                f'CREATE TABLE "{repository.tables.review}" (REVIEW_ID TEXT PRIMARY KEY, ECN_ROW_ID TEXT, EVIDENCE_HASH TEXT, SNAPSHOT_VERSION TEXT, REVIEWER TEXT, ACTOR_ID TEXT, OUTCOME TEXT, COMMENTS TEXT, CREATED_AT TEXT)'
            )
        )
        connection.execute(
            text(
                f"INSERT INTO \"{repository.tables.analysis_job}\" (JOB_ID,ECN_ROW_ID,STATE) VALUES ('j1','e1','READY'),('j2','e2','COMPLETED'),('j3','e3','RUNNING')"
            )
        )
        connection.execute(
            text(
                f"INSERT INTO \"{repository.tables.analysis}\" (ANALYSIS_ID,ECN_ROW_ID,JOB_ID) VALUES ('a2','e2','j2')"
            )
        )
    yield repository
    engine.dispose()


def test_explicit_batch_reuse_and_old_cancellation_cannot_stop_resubmitted_work(
    workspace_repository,
):
    """An explicit batch refreshes completed jobs while cancellation retains ownership."""
    repo = workspace_repository
    _, scheduled = repo.create_batch("batch1", ["e1", "e2", "e3"])
    assert scheduled == ("j1", "j2")
    assert repo.jobs(["e2"])[0]["STATE"] == "QUEUED"
    with repo.engine.connect() as connection:
        assert connection.execute(
            text(f'SELECT COUNT(*) FROM "{repo.tables.analysis}" WHERE JOB_ID = :id'),
            {"id": "j2"},
        ).scalar_one() == 1
    assert repo.create_batch("batch1", ["e1", "e2", "e3"])[1] == ()
    repo.cancel_batch("batch1")
    assert repo.jobs(["e1"])[0]["STATE"] == "READY"
    repo.create_batch("batch2", ["e1"])
    repo.cancel_batch("batch1")
    assert repo.jobs(["e1"])[0]["STATE"] == "QUEUED"
    with pytest.raises(Exception, match="different selection"):
        repo.create_batch("batch1", ["e2"])


@pytest.mark.asyncio
async def test_summary_submission_supports_explicit_or_filter_wide_selection() -> None:
    """Resolve filters on submission, apply exclusions, and persist only resolved IDs."""
    from pydantic import ValidationError

    from app.routers.workspace_actions import GenerateSelection, generate_summaries

    first = synthetic_uuid("first-summary")
    second = synthetic_uuid("second-summary")
    third = synthetic_uuid("third-summary")
    request_id = synthetic_uuid("summary-request")
    created = []
    scheduled = []
    gated = []

    def create_batch(batch_id, ids):
        """Record resolved membership and return one newly scheduled job."""
        created.append((batch_id, ids))
        return {
            "batch_id": batch_id,
            "selection": {"ecn_ids": ids, "scheduled_job_ids": ["job-1"]},
            "jobs": [],
            "finished": False,
        }, ("job-1",)

    services = SimpleNamespace(
        ecn_repository=SimpleNamespace(
            list_matching_ecn_ids=lambda query: [first, second, third]
        ),
        workspace_repository=SimpleNamespace(create_batch=create_batch),
        runtime=SimpleNamespace(schedule_jobs=lambda ids: scheduled.extend(ids)),
        rdf_projection_service=SimpleNamespace(
            require_all_current=lambda ids: gated.append(tuple(ids))
        ),
    )
    explicit = GenerateSelection.model_validate({
        "request_id": request_id,
        "selection": {"mode": "explicit", "ecn_ids": [first]},
    })
    await generate_summaries(explicit, services)
    filtered = GenerateSelection.model_validate({
        "request_id": synthetic_uuid("filtered-request"),
        "selection": {
            "mode": "filtered",
            "query": "gear",
            "ecn_number": "",
            "analysis_state": "READY",
            "has_warnings": None,
            "excluded_ecn_ids": [second],
        },
    })
    await generate_summaries(filtered, services)

    assert created[0][1] == [first]
    assert created[1][1] == [first, third]
    assert gated == [(first,), (first, third)]
    assert scheduled == ["job-1", "job-1"]
    with pytest.raises(ValidationError):
        GenerateSelection.model_validate({
            "request_id": request_id,
            "selection": {"mode": "explicit", "ecn_ids": [first], "query": "extra"},
        })


def test_review_retry_is_idempotent_and_conflicting_payload_rejected(
    workspace_repository,
):
    """Comments are append-only and a reused request cannot silently change its meaning."""
    values = {
        "REVIEW_ID": "r1",
        "ECN_ROW_ID": "e1",
        "EVIDENCE_HASH": "a" * 64,
        "SNAPSHOT_VERSION": "v1",
        "REVIEWER": "Engineer",
        "ACTOR_ID": "actor",
        "OUTCOME": "REVIEWED",
        "COMMENTS": "Needs check",
        "CREATED_AT": "2026-09-16",
    }
    workspace_repository.save_review(values)
    workspace_repository.save_review(values | {"CREATED_AT": "later"})
    assert len(workspace_repository.reviews("e1")) == 1
    with pytest.raises(Exception, match="different review"):
        workspace_repository.save_review(values | {"COMMENTS": "changed"})


def test_review_decision_requires_change_description_only_for_change_requests(
    workspace_repository,
):
    """Accept may be silent, a change request must explain itself, retired outcomes are rejected."""
    from app.routers.workspace_actions import ReviewSubmission
    from pydantic import ValidationError

    base = {
        "request_id": str(uuid4()),
        "snapshot_version": "ecn-snapshot-v2",
        "evidence_hash": "a" * 64,
        "reviewer": "Engineer",
    }
    assert ReviewSubmission.model_validate(base | {"outcome": "ACCEPTED"}).comments == ""
    with pytest.raises(ValidationError, match="Describe the changes"):
        ReviewSubmission.model_validate(base | {"outcome": "CHANGES_REQUESTED", "comments": "  "})
    with pytest.raises(ValidationError):
        ReviewSubmission.model_validate(base | {"outcome": "REVIEWED", "comments": "ok"})

    # A NULL comment (HANA may store an empty NCLOB that way) reads and replays as "".
    values = {
        "REVIEW_ID": "r2", "ECN_ROW_ID": "e2", "EVIDENCE_HASH": "a" * 64,
        "SNAPSHOT_VERSION": "v1", "REVIEWER": "Engineer", "ACTOR_ID": "actor",
        "OUTCOME": "ACCEPTED", "COMMENTS": None, "CREATED_AT": "2026-09-29",
    }
    workspace_repository.save_review(values)
    workspace_repository.save_review(values | {"COMMENTS": ""})
    assert [r["comments"] for r in workspace_repository.reviews("e2")] == [""]


@pytest.mark.asyncio
async def test_sidebar_tools_are_scoped_traced_and_not_iteration_limited(monkeypatch):
    """The real graph may exceed ten calls while retaining scoped tool diagnostics."""
    from app.ecn.assistant.runtime import AgentRuntime
    from app.ecn.assistant.tools import create_tools
    from langchain_core.messages import AIMessage

    selected = synthetic_uuid("selected")
    seen = []
    emissions = []

    def overview(key):
        """Require server-selected identity for every fact lookup."""
        seen.append(key)
        return {"assessment_id": key, "counts": {"technical_objects": 2}}

    snapshots = SimpleNamespace(overview=overview)
    rdf = SimpleNamespace()
    tools = create_tools(snapshots, None, rdf, selected, set(), [], set())
    assert all(
        "ecn_row_id" not in t.args and "assessment_id" not in t.args for t in tools
    )

    class Model:
        """Request eleven tool calls before returning a final answer."""

        def __init__(self):
            """Initialize the provider-call counter."""
            self.calls = 0

        def bind_tools(self, tools, **kwargs):
            """Return the deterministic model while checking read-only tool scope."""
            return self

        async def ainvoke(self, messages):
            """Return eleven overview reads followed by final prose."""
            self.calls += 1
            if self.calls > 11:
                return AIMessage(content="Finished after eleven tool calls.")
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_ecn_overview", "args": {}, "id": str(uuid4())}
                ],
            )

    monkeypatch.setattr(
        "app.ecn.assistant.runtime.emit_llm_usage_event",
        lambda **fields: emissions.append(fields),
    )
    runtime = AgentRuntime(
        snapshots, None, rdf, "test", asyncio.Semaphore(2), model_factory=Model
    )
    result = await runtime.ainvoke("Explain", selected, [])
    assert result["model_calls"] == 12
    assert result["answer"] == "Finished after eleven tool calls."
    assert set(seen) == {selected} and len(emissions) == 12
    assert len(result["tool_trace"]) == 11
    assert result["tool_trace"][0] == {
        "sequence": 1,
        "tool": "get_ecn_overview",
        "arguments": {},
        "duration_ms": result["tool_trace"][0]["duration_ms"],
        "outcome": "success",
        "result": {"counts": {"technical_objects": 2}},
    }
    assert result["tool_trace"][0]["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_assistant_preloads_mandatory_source_field_discovery_guidance() -> None:
    """Give the model direct-field retrieval guidance without spending a tool call."""
    from app.ecn.assistant.runtime import AgentRuntime
    from langchain_core.messages import AIMessage, SystemMessage

    selected = synthetic_uuid("preloaded-inspect-skill")
    prompts = []

    class Model:
        """Capture the actual runtime prompt while returning one final response."""

        def bind_tools(self, tools, **kwargs):
            """Retain the same model for the runtime's read-only tool binding."""
            return self

        async def ainvoke(self, messages):
            """Record the system instruction supplied to the model boundary."""
            prompts.append(next(message.content for message in messages if isinstance(message, SystemMessage)))
            return AIMessage(content="No further inspection is required.")

    runtime = AgentRuntime(
        SimpleNamespace(overview=lambda _: {"assessment_id": selected}),
        None,
        SimpleNamespace(),
        "test",
        asyncio.Semaphore(1),
        model_factory=Model,
    )

    result = await runtime.ainvoke("Who created this ECN?", selected, [])

    assert result["answer"] == "No further inspection is required."
    assert len(prompts) == 1
    assert "===== SKILL: inspect-ecn =====" in prompts[0]
    assert "Discover the exact source key with `list_ecn_fields`" in prompts[0]
    assert "exact fact retrieval both return no matches" in prompts[0]


@pytest.mark.asyncio
async def test_snapshot_routes_and_assistant_forward_keyword_filters() -> None:
    """Exercise every reported keyword-argument caller through its async boundary."""
    from app.ecn.assistant.tools import create_tools
    from app.routers.snapshots import object_details, objects

    selected = synthetic_uuid("keyword-forwarding")
    calls = []

    def list_objects(key, **filters):
        """Record list filters received by the synchronous snapshot service."""
        calls.append(("objects", key, filters))
        return {"assessment_id": key, "total": 0, "items": []}

    def inspect_object(key, number, **filters):
        """Record detail filters received by the synchronous snapshot service."""
        calls.append(("object", key, number, filters))
        return {"assessment_id": key, "number": number, "items": []}

    snapshots = SimpleNamespace(objects=list_objects, object_details=inspect_object)
    services = SimpleNamespace(snapshot_service=snapshots)
    await objects(
        UUID(selected), services, page=2, page_size=5, query="part", technical=False
    )
    await object_details(
        UUID(selected), "A-1", services, technical=False, page=3, page_size=4
    )
    assistant_tool = next(
        tool
        for tool in create_tools(snapshots, None, SimpleNamespace(), selected, set(), [], set())
        if tool.name == "list_referenced_objects"
    )
    await assistant_tool.ainvoke(
        {"query": "bolt", "page": 4, "technical": True, "origin": "bom"}
    )

    assert calls[0][2]["page"] == 2
    assert calls[1][3] == {"technical": False, "page": 3, "page_size": 4}
    assert calls[2][2]["origin"] == "bom"


def test_known_severity_survives_an_unestablished_dimension():
    """Known impact and evidence sufficiency are independent, including all-not-applicable."""
    from app.ecn.analysis_models import (
        ImpactFactorAssessment,
        ImpactFactors,
        ImpactSeverity,
        derive_overall_impact,
    )

    known = ImpactFactorAssessment(
        severity=ImpactSeverity.HIGH,
        assessment="Observed manufacturing consequence",
        evidence_ids=["E-1"],
        missing_evidence=["Rollout date"],
    )
    unknown = ImpactFactorAssessment(
        severity=ImpactSeverity.UNKNOWN,
        assessment="Supplier consequences not established",
        evidence_ids=[],
        missing_evidence=["Supplier evidence"],
    )
    factors = ImpactFactors(
        manufacturing=known,
        procurement=unknown,
        configuration=unknown,
        data_readiness=unknown,
    )
    assert derive_overall_impact(factors) is ImpactSeverity.HIGH


def test_snapshot_hydrates_only_selected_source_when_detail_reader_returns_metadata():
    """A metadata-only detail record must not silently become an empty ECN snapshot."""
    from app.ecn.services.snapshots import SnapshotService
    from app.persistence.table_contracts import PersistenceTableNames

    from tests.fixtures.builders import minimal_ecn_json_bytes
    from tests.fixtures.persistence import RecordingEngine

    source = json.loads(minimal_ecn_json_bytes())[0]
    engine = RecordingEngine()
    engine.queue_scalar(json.dumps(source))
    key = synthetic_uuid("selected-source")
    record = SimpleNamespace(
        ecn=SimpleNamespace(source_json={}, source_ordinal=25, evidence_hash="a" * 64),
        ecn_artifact=SimpleNamespace(artifact_id=synthetic_uuid("source")),
        bom_artifact=SimpleNamespace(artifact_id=synthetic_uuid("bom")),
        upload=SimpleNamespace(upload_id=synthetic_uuid("upload")),
    )

    class TargetedArtifacts:
        """Expose only the targeted snapshot read; full iteration must stay unused."""

        def __init__(self):
            """Initialize captured targeted-read inputs."""
            self.calls = []

        def snapshot_bom_rows(self, artifact_id, component_numbers, change_number):
            """Capture the selected artifact, object numbers, and ECN number."""
            self.calls.append((artifact_id, tuple(component_numbers), change_number))
            return ()

        def change_touched_components(self, artifact_id, ecn_number):
            """Report no parts started or ended by the ECN code."""
            return ()

        def iter_bom_rows(self, artifact_id):
            """Fail if snapshot construction falls back to the complete BOM."""
            raise AssertionError(f"full BOM read used for {artifact_id}")

    artifacts = TargetedArtifacts()
    service = SnapshotService(
        SimpleNamespace(get_ecn_detail=lambda _: record, get_evidence_bundle=lambda _: None),
        artifacts,
        None,
        engine,
        PersistenceTableNames(),
    )
    service.pinned_references = lambda *_: ([], [])
    try:
        result = service.get(key)
        assert result["counts"]["technical_objects"] > 0
        assert result["source_position"] == 26
        assert '"SOURCE_JSON"' in engine.executions[0][0]
        assert "CONTENT_BYTES" not in engine.executions[0][0]
        assert engine.executions[0][1] == {"id": key}
        assert artifacts.calls
        assert artifacts.calls[0][0] == record.bom_artifact.artifact_id
        assert artifacts.calls[0][1]
        assert artifacts.calls[0][2] == result["ecn_number"]
    finally:
        service.clear()


def test_overview_exposes_compact_descriptions_without_the_full_object_graph():
    """The headline can name ECN parts without widening the initial evidence payload."""
    from app.ecn.services.snapshots import SnapshotService

    service = object.__new__(SnapshotService)
    service.get = lambda _: {
        "ecn_number": "ECN346863",
        "questions": [],
        "objects": [
            {"number": "PART-1", "technical": True, "description": "Flexible shim ring"},
            {"number": "DOC-1", "technical": False, "description": "Drawing"},
            {"number": "PART-2", "technical": True, "description": None},
        ],
        "changes": [],
    }

    overview = service.overview("assessment-1")

    assert "objects" not in overview and "changes" not in overview
    assert overview["part_descriptions"] == [
        {"number": "PART-1", "description": "Flexible shim ring"},
    ]
    assert not {
        "purchase_prices",
        "work_instructions",
        "tooling_inventory",
        "phase_schedule",
        "make_buy_recommendations",
        "demo_assumptions",
    }.intersection(overview)
    assert not {"MISSING_INFO_RECORD", "MISSING_WORK_INSTRUCTION"}.intersection(
        question.get("code") for question in overview["questions"]
    )


def test_large_database_bindings_are_lazy_and_chunk_bounded():
    """The driver never receives the full publication's wide NCLOB parameter array."""
    from app.persistence.bulk import insert_chunks

    converted = []
    calls = []

    def rows():
        """Track conversion to prove rows are materialized only as each chunk is needed."""
        for index in range(1001):
            converted.append(index)
            yield {"value": index}

    def execute(statement, batch):
        """Record bounded driver requests and conversion progress."""
        calls.append((len(batch), len(converted)))

    insert_chunks(SimpleNamespace(execute=execute), "INSERT", rows())
    assert calls == [(500, 500), (500, 1000), (1, 1001)]
