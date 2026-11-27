"""Test shared workspace tools and chat without HANA connections or paid models."""

from app.agent.tools.workspace_tools import workspace_tools
from production_wheel.schemas import SolveRequest


class WorkspaceStub:
    """Record typed service calls for deterministic tool contract tests."""

    def __init__(self):
        """Initialize captured operations and a selected context."""
        self.calls = []

    def capabilities(self):
        """Return a finite synthetic optimizer capability menu."""
        return {"modes": ["PARETO"]}

    def update_draft(self, draft_id, revision, patch):
        """Return the revised draft and capture the supplied revision guard."""
        self.calls.append((draft_id, revision, patch))
        return {"draft_id": draft_id, "revision": revision + 1}

    def submit(self, submission):
        """Capture a typed idempotent submission and return its independent job."""
        self.calls.append(submission)
        return {"run_id": "run-1", "status": "queued"}

    def query(self, spec):
        """Capture a validated semantic query with no SQL execution."""
        self.calls.append(spec)
        return {"rows": [], "total": 0}


def test_workspace_mutations_use_revision_guards_and_emit_events():
    """Bind tool mutations to the service and announce manual-view refresh events."""
    service = WorkspaceStub()
    events = []
    tools = {
        tool.name: tool for tool in workspace_tools(service, "context-1", events.append)
    }
    changed = tools["update_run_draft"].invoke(
        {
            "draft_id": "draft-1",
            "revision": 3,
            "patch": {"request": SolveRequest().model_dump(mode="json")},
        }
    )
    assert changed["revision"] == 4
    assert events[0]["type"] == "draft_changed"
    run = tools["launch_optimization"].invoke(
        {"draft_id": "draft-1", "revision": 4, "idempotency_key": "submission-1"}
    )
    assert run["run_id"] == "run-1"
    assert service.calls[-1].idempotency_key == "submission-1"
    assert events[-1]["type"] == "run_created"


def test_semantic_queries_reject_missing_scope_and_raw_sql():
    """Reject SQL and missing source IDs before calling the service."""
    import pytest

    service = WorkspaceStub()
    tools = {tool.name: tool for tool in workspace_tools(service, "context-1")}
    with pytest.raises(ValueError):
        tools["query_optimizer_data"].invoke({"spec": {"view": "groups"}})
    with pytest.raises(ValueError):
        tools["query_optimizer_data"].invoke(
            {
                "spec": {
                    "view": "groups",
                    "run_id": "run-1",
                    "sql": "SELECT * FROM SECRET",
                }
            }
        )
    assert service.calls == []


def test_run_tools_use_compact_and_scoped_run_read_models():
    """Chat tools may discover, diagnose and page runs without raw snapshots."""
    import asyncio

    class Service:
        """Expose distinct compact values to detect accidental raw-service reads."""

        def list_run_summaries(self, filters):
            """Return a compact run-list response."""
            return [{"run_id": "r", "status": "failed"}]

        def run_status(self, run_id):
            """Return a compact selected-run response."""
            return {"run_id": run_id, "status": "failed", "configuration": {}}

        def run_failure_diagnostics(self, run_id):
            """Return bounded checkpoint evidence."""
            return {"run_id": run_id, "failure_types": ["uncovered_precheck"]}

        def run_matrix_page(self, run_id, offset, limit, status):
            """Return the requested filtered frozen-matrix page."""
            return {"run_id": run_id, "offset": offset, "limit": limit, "status": status}

    tools = {item.name: item for item in workspace_tools(Service(), "context")}

    assert tools["list_runs"].invoke({"filters": {}}) == [
        {"run_id": "r", "status": "failed"}
    ]
    assert asyncio.run(tools["get_run_status"].ainvoke({"run_id": "r"})) == {
        "run_id": "r",
        "status": "failed",
        "configuration": {},
    }
    assert tools["get_run_configuration"].invoke({"run_id": "r"}) == {}
    assert tools["get_run_failure_diagnostics"].invoke({"run_id": "r"}) == {
        "run_id": "r",
        "failure_types": ["uncovered_precheck"],
    }
    assert tools["get_run_matrix_page"].invoke(
        {"run_id": "r", "offset": 20, "limit": 10, "status": "blocked"}
    ) == {"run_id": "r", "offset": 20, "limit": 10, "status": "blocked"}


def test_launch_tool_does_not_return_the_new_run_replay_snapshot():
    """Creating a run must not inject its frozen matrix/request into agent context."""

    class Service:
        """Return a deliberately oversized durable submission record."""

        def submit(self, submission):
            """Persist a raw fixture resembling the real worker-owned record."""
            return {
                "run_id": "r",
                "status": "queued",
                "plant_profile": {"matrix_rows": [{"volume_a": "A"}]},
                "request": {"config": {"matrix_pairs": [{"volume_a": "A"}]}},
            }

        def run_status(self, run_id):
            """Return the intended compact launch acknowledgement."""
            return {"run_id": run_id, "status": "queued", "configuration": {}}

    tools = {item.name: item for item in workspace_tools(Service(), "context")}
    result = tools["launch_optimization"].invoke(
        {"draft_id": "d", "revision": 1, "idempotency_key": "key"}
    )

    assert result == {"run_id": "r", "status": "queued", "configuration": {}}


def test_chat_stream_filters_reasoning_without_a_persisted_history_endpoint():
    """Stream public activity while exposing no durable conversation reader."""
    import asyncio
    import json
    from types import SimpleNamespace

    from app.routers import workspace_chat
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    class Service:
        """Keep selected context state for the route's service calls."""

        def __init__(self):
            """Initialize the selection dictionary."""
            self.selected = {}

        def context(self, context_id, patch=None):
            """Return or update one persisted selection."""
            self.selected.update(patch or {})
            return {"context_id": context_id, **self.selected}

    class Runtime:
        """Emit callbacks from a worker thread with a final model answer."""

        async def ainvoke(
            self, message, context_id, on_event, session_history=None
        ):
            """Simulate tool activity and a forbidden internal reasoning event."""
            await asyncio.to_thread(
                on_event,
                {
                    "type": "tool_call",
                    "name": "inspect_dataset",
                    "args": {"dataset_id": "ds-1"},
                },
            )
            on_event({"type": "reasoning", "text": "private chain"})
            on_event(
                {
                    "type": "tool_result",
                    "name": "inspect_dataset",
                    "preview": "one dataset",
                }
            )
            return SimpleNamespace(output_text="Ready")

        async def aclose(self):
            """Release no external resources in this test."""

    async def factory(service, context_id, on_event):
        """Construct a fake runtime without initializing a paid provider."""
        return Runtime()

    service = Service()
    app = FastAPI()
    app.include_router(workspace_chat.router)
    app.dependency_overrides[workspace_chat.get_service] = lambda: service
    app.dependency_overrides[workspace_chat.get_runtime_factory] = lambda: factory
    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={"message": "Inspect", "context_id": "c1", "dataset_id": "ds-1"},
        )
        assert response.status_code == 200
        events = [json.loads(line) for line in response.text.splitlines()]
        assert {event["type"] for event in events} == {
            "tool_call",
            "tool_result",
            "assistant",
        }
        assert "private chain" not in response.text
        assert client.get("/api/chat/c1").status_code == 404
        assert service.selected["dataset_id"] == "ds-1"


def test_chat_context_summary_is_forwarded_only_within_the_page_session():
    """The browser-owned summary is returned to the client, never written through context."""
    import asyncio
    import json
    from types import SimpleNamespace

    from app.routers import workspace_chat
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    class Service:
        """Record persisted selection patches without a conversation-history store."""

        def __init__(self):
            """Start with no persisted selection values."""
            self.patch = None

        def context(self, context_id, patch=None):
            """Capture only the normal selected workspace values."""
            self.patch = patch
            return {"context_id": context_id}

    class Runtime:
        """Return a newly compacted browser summary from a fake agent turn."""

        async def ainvoke(self, message, context_id, on_event, session_history, context_summary):
            """Verify page-only summary input and return a replacement summary."""
            assert context_summary == "older browser context"
            return SimpleNamespace(
                output_text="Ready",
                context_summary="new compact context",
                history_compacted=True,
            )

        async def aclose(self):
            """Release no external resources in this test."""

    async def factory(service, context_id, on_event):
        """Construct a no-provider runtime."""
        return Runtime()

    app = FastAPI()
    service = Service()
    app.include_router(workspace_chat.router)
    app.dependency_overrides[workspace_chat.get_service] = lambda: service
    app.dependency_overrides[workspace_chat.get_runtime_factory] = lambda: factory
    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={
                "message": "Explain this run",
                "context_id": "c-summary",
                "context_summary": "older browser context",
            },
        )

    event = [json.loads(line) for line in response.text.splitlines()][-1]
    assert event == {
        "type": "assistant",
        "text": "Ready",
        "context_summary": "new compact context",
        "history_compacted": True,
    }
    assert service.patch is None


def test_chat_turns_serialize_and_finish_after_disconnect():
    """A closed response does not cancel its turn or overlap another context turn."""
    import asyncio
    from types import SimpleNamespace

    from app.routers.workspace_chat import ChatMessage, chat_message

    class Service:
        """Supply synchronous context persistence without external I/O."""

        def context(self, context_id, patch=None):
            """Return the selected context identifier."""
            return {"context_id": context_id}

    async def exercise():
        """Run two overlapping HTTP turns and disconnect the first response."""
        started = asyncio.Event()
        release = asyncio.Event()
        counts = {"active": 0, "maximum": 0, "completed": 0}

        class Runtime:
            """Track active fake invocations and block until the test releases them."""

            async def ainvoke(
                self, message, context_id, on_event, session_history=None
            ):
                """Record concurrency and finish independently of response delivery."""
                counts["active"] += 1
                counts["maximum"] = max(counts["maximum"], counts["active"])
                started.set()
                await release.wait()
                counts["active"] -= 1
                counts["completed"] += 1
                return SimpleNamespace(output_text="Saved")

            async def aclose(self):
                """Close the fake runtime without external side effects."""

        async def factory(service, context_id, on_event):
            """Construct one fake runtime per turn."""
            return Runtime()

        body = ChatMessage(message="Run", context_id="serialized-test")
        first = await chat_message(body, Service(), factory)
        await started.wait()
        second = await chat_message(body, Service(), factory)
        await first.body_iterator.aclose()
        await asyncio.sleep(0.01)
        assert counts["maximum"] == 1
        release.set()
        async for _ in second.body_iterator:
            pass
        assert counts["completed"] == 2
        assert counts["maximum"] == 1

    asyncio.run(exercise())


def test_chat_uses_only_client_session_history_and_exposes_no_history_read():
    """Keep prior turns within one page request chain without durable history APIs."""
    import json
    from types import SimpleNamespace

    from app.routers import workspace_chat
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    observed = []

    class Service:
        """Provide the selected context needed by workspace tools."""

        def context(self, context_id, patch=None):
            """Return the request's ephemeral selection."""
            return {"context_id": context_id, **(patch or {})}

    class Runtime:
        """Capture the real route-to-runtime session-history boundary."""

        async def ainvoke(self, message, context_id, on_event, session_history=None):
            """Record prior turns supplied only by this page session."""
            observed.extend(session_history or [])
            return SimpleNamespace(output_text="Current answer")

        async def aclose(self):
            """Release no external resources in this test."""

    async def factory(service, context_id, on_event):
        """Return a provider-free runtime double."""
        return Runtime()

    app = FastAPI()
    app.include_router(workspace_chat.router)
    app.dependency_overrides[workspace_chat.get_service] = lambda: Service()
    app.dependency_overrides[workspace_chat.get_runtime_factory] = lambda: factory
    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={
                "message": "Current question",
                "context_id": "page-only",
                "history": [
                    {"role": "user", "content": "Earlier question"},
                    {"role": "assistant", "content": "Earlier answer"},
                ],
            },
        )
        assert response.status_code == 200
        assert json.loads(response.text.splitlines()[-1])["text"] == "Current answer"
        assert observed == [
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer"},
        ]
        assert client.get("/api/chat/page-only").status_code == 404


def test_solution_comparison_keeps_point_identity():
    """Pass complete solution selections to the service without discarding points."""

    class Service:
        """Echo point-aware comparison arguments."""

        def compare(self, left, right):
            """Return both point selections unchanged for contract verification."""
            return {"left": left, "right": right}

    tools = {item.name: item for item in workspace_tools(Service(), "context")}
    left = {"run_id": "run-1", "point_index": 2}
    right = {"run_id": "run-1", "point_index": 4}
    result = tools["compare_solutions"].invoke({"left": left, "right": right})
    assert result == {"left": left, "right": right}


def test_create_draft_from_selected_or_explicit_dataset_and_apply_patch():
    """Start a shared draft without preexisting IDs and retain its revised selection."""

    class Service(WorkspaceStub):
        """Add draft creation and selected-context persistence to the tool stub."""

        def __init__(self):
            """Start with a selected dataset and no draft."""
            super().__init__()
            self.selection = {"dataset_id": "selected-dataset"}

        def context(self, context_id, patch=None):
            """Read or update the selected IDs for this test context."""
            self.selection.update(patch or {})
            return dict(self.selection)

        def create_draft(self, dataset_id, plant_profile_id=None, title=""):
            """Create a first revision for the specified dataset."""
            self.calls.append(("create", dataset_id))
            return {"draft_id": "new-draft", "dataset_id": dataset_id, "revision": 1}

        def update_draft(self, draft_id, revision, patch):
            """Capture the first-revision patch and return the changed draft."""
            self.calls.append((draft_id, revision, patch))
            return {
                "draft_id": draft_id,
                "dataset_id": self.calls[0][1],
                "revision": revision + 1,
            }

    service = Service()
    events = []
    tools = {
        item.name: item for item in workspace_tools(service, "context", events.append)
    }
    result = tools["update_run_draft"].invoke(
        {"patch": {"dataset_id": "explicit-dataset", "budget": {"frontier_points": 5}}}
    )
    assert result["revision"] == 2
    assert service.calls == [
        ("create", "explicit-dataset"),
        ("new-draft", 1, {"budget": {"frontier_points": 5}}),
    ]
    assert service.selection == {
        "dataset_id": "explicit-dataset",
        "draft_id": "new-draft",
    }
    assert events[-1]["type"] == "draft_changed"
    selected = Service()
    selected_tool = {item.name: item for item in workspace_tools(selected, "context")}[
        "update_run_draft"
    ]
    assert selected_tool.invoke({})["revision"] == 1
    assert selected.calls == [("create", "selected-dataset")]


def test_existing_draft_mutation_requires_revision():
    """Reject mutations without a revision before invoking the workspace service."""
    import pytest

    service = WorkspaceStub()
    tools = {item.name: item for item in workspace_tools(service, "context")}
    with pytest.raises(ValueError, match="revision"):
        tools["update_run_draft"].invoke(
            {"draft_id": "existing", "patch": {"budget": {"frontier_points": 5}}}
        )
    assert service.calls == []


def test_profile_proposals_validate_without_saving_and_emit_cards():
    """Proposal tools validate against the stored profile, emit a card event and never write."""
    import pytest
    from test_plant_profiles import fixture, line_cap_rule
    service, _, profile = fixture()
    events = []
    tools = {tool.name: tool for tool in workspace_tools(service, "context-p", events.append)}
    with pytest.raises(ValueError, match="plant profile"):
        tools["get_plant_profile_details"].invoke({})
    service.context("context-p", {"plant_profile_id": profile["profile_id"]})
    details = tools["get_plant_profile_details"].invoke({})
    assert details["revision"] == profile["revision"] and details["matrix_volumes"]
    result = tools["propose_profile_constraint"].invoke({"action": "add_constraint",
        "summary": "Cap line L1 groups at two FINIs.", "constraint": line_cap_rule()})
    assert result["status"] == "awaiting_user_approval"
    card = events[-1]
    assert card["type"] == "profile_proposal" and card["profile_revision"] == profile["revision"]
    assert card["change"]["constraint"]["constraint_id"] == "line-six-cap"
    assert service.get_plant_profile(profile["profile_id"])["rules"] == profile["rules"]
    count = len(events)
    with pytest.raises(ValueError):
        tools["propose_profile_constraint"].invoke({"action": "add_constraint", "summary": "bad",
            "constraint": {**line_cap_rule(), "assertion": {"op": "field", "field": "group.unknown"}}})
    with pytest.raises(ValueError, match="description"):
        tools["propose_profile_constraint"].invoke({"action": "add_constraint", "summary": "x",
            "constraint": {**line_cap_rule(), "description": ""}})
    with pytest.raises(ValueError, match="does not exist"):
        tools["propose_profile_constraint"].invoke({"action": "remove_constraint", "summary": "x",
            "constraint_id": "missing"})
    assert len(events) == count
    tools["propose_profile_settings"].invoke({"summary": "Smaller groups and a 3-volume matrix.",
        "settings_patch": {"maximum_group_size": 2},
        "volume_compatibility": {"volumes": ["1", "2", "3"], "compatible_groups": [["1", "2"]], "default_status": "N"}})
    change = events[-1]["change"]
    assert change["settings"] == {"maximum_group_size": 2} and len(change["matrix_rows"]) == 9
    applied = service.apply_plant_profile_change(profile["profile_id"], profile["revision"], change)
    assert applied["settings"]["maximum_group_size"] == 2


def test_assistant_extends_profile_matrix_with_dataset_volumes():
    """Coverage lists dataset volumes; a rule-based proposal keeps current pairs and shows a summary."""
    from test_plant_profiles import fixture
    from app.workspace.matrix_description import expand_matrix_description
    service, source, profile = fixture()
    narrow = expand_matrix_description({"volumes": ["999"], "default_status": "N"})
    profile = service.apply_plant_profile_change(profile["profile_id"], profile["revision"],
        {"action": "update_settings", "settings": {"volume_matching": "exact"}, "matrix_rows": narrow})
    events = []
    tools = {tool.name: tool for tool in workspace_tools(service, "context-m", events.append)}
    service.context("context-m", {"plant_profile_id": profile["profile_id"], "dataset_id": source["dataset_id"]})
    coverage = tools["get_dataset_volume_coverage"].invoke({})
    missing = [item["volume"] for item in coverage["missing_volumes"]]
    assert missing and coverage["profile_volumes_unused"] == ["999"]
    result = tools["propose_profile_settings"].invoke({"summary": "Add the dataset volumes.",
        "volume_compatibility": {"dataset_id": source["dataset_id"], "keep_existing_pairs": True,
            "default_status": "N", "rules": [{"kind": "max_difference_percent", "percent": 50}]}})
    summary = result["matrix_summary"]
    assert summary["added_volumes"] == sorted(missing, key=float) and summary["changed_pair_count"] == 0
    assert events[-1]["matrix_summary"] == summary
    assert service.get_plant_profile(profile["profile_id"])["matrix_rows"] == profile["matrix_rows"]


def test_chat_forwards_profile_proposals():
    """The NDJSON whitelist forwards proposal cards and drops unknown keys."""
    from app.routers.workspace_chat import _public_event
    event = _public_event({"type": "profile_proposal", "proposal_id": "p", "profile_id": "x",
        "profile_name": "n", "plant": "P", "profile_revision": 1, "summary": "s",
        "change": {"action": "remove_constraint", "constraint_id": "c"}, "secret": "no"})
    assert event["change"]["constraint_id"] == "c" and "secret" not in event


def test_explicit_null_selection_clears_stale_session_context():
    """A Settings turn sending null draft/run/point removes values left by the optimizer page."""
    from types import SimpleNamespace

    from app.routers import workspace_chat
    from app.workspace.service import WorkspaceService
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    seen = []

    class Runtime:
        """Record the selection visible to tools when the turn runs."""

        def __init__(self, service, context_id):
            """Keep the real service and page-session key."""
            self.service, self.context_id = service, context_id

        async def ainvoke(self, *args, **kwargs):
            """Capture the merged session context instead of calling a model."""
            seen.append(self.service.context(self.context_id))
            return SimpleNamespace(output_text="ok")

        async def aclose(self):
            """Release no external resources in this test."""

    async def factory(service, context_id, on_event):
        """Construct the recording runtime without a provider."""
        return Runtime(service, context_id)

    service = WorkspaceService(repository=None)
    app = FastAPI()
    app.include_router(workspace_chat.router)
    app.dependency_overrides[workspace_chat.get_service] = lambda: service
    app.dependency_overrides[workspace_chat.get_runtime_factory] = lambda: factory
    optimizer = {"plant_profile_id": "p1", "dataset_id": "d1", "draft_id": "dr1", "run_id": "r1", "point_index": 7}
    settings = {"plant_profile_id": "p1", "dataset_id": None, "draft_id": None, "run_id": None, "point_index": None}
    with TestClient(app) as client:
        for selection in (optimizer, settings):
            response = client.post("/api/chat", json={"message": "hi", "context_id": "c-null", **selection})
            assert response.status_code == 200
    assert seen[0] == {"context_id": "c-null", **optimizer}
    assert seen[1] == {"context_id": "c-null", **settings}


def test_assistant_recreates_outdated_draft_and_selects_it():
    """The tool returns the new draft, selects it in the conversation and notifies the page."""
    from test_plant_profiles import fixture
    service, source, profile = fixture()
    draft = service.create_draft(source["dataset_id"], profile["profile_id"])
    service.update_plant_profile(profile["profile_id"], {"revision": 1, "name": "Renamed"})
    events = []
    tools = {tool.name: tool for tool in workspace_tools(service, "context-r", events.append)}
    result = tools["recreate_draft_with_current_profile"].invoke({"draft_id": draft["draft_id"]})
    new_id = result["draft"]["draft_id"]
    assert result["draft"]["plant_profile"]["revision"] == 2
    assert service.context("context-r")["draft_id"] == new_id
    assert events[-1]["type"] == "draft_changed" and events[-1]["draft_id"] == new_id


def test_assistant_adds_a_class_stop_and_previews_reclassified_products():
    """In a class profile the assistant previews stops, proposes a 2 L stop and shows who moves."""
    import pytest
    from test_volume_classes import class_service
    service, source, profile = class_service([1.1, 2.2, 4, 20])
    events = []
    tools = {tool.name: tool for tool in workspace_tools(service, "context-c", events.append)}
    service.context("context-c", {"plant_profile_id": profile["profile_id"], "dataset_id": source["dataset_id"]})
    details = tools["get_plant_profile_details"].invoke({})
    assert details["volume_matching"] == "nearest_class" and details["matrix_volumes"][0] == "<1"
    preview = tools["get_dataset_volume_coverage"].invoke(
        {"proposed_stops": ["<1", "1", "2", "2.5", "5", "10", "15", ">15"]})
    assert preview["reclassified_volumes"] == [{"volume": "2.2", "product_count": 1, "from": "2.5", "to": "2"}]
    result = tools["propose_profile_settings"].invoke({"summary": "Add a 2 L stop that behaves like 2.5.",
        "volume_compatibility": {"volumes": ["2"], "keep_existing_pairs": True, "default_status": "AVOID",
                                 "inherit_from": {"2": "2.5"}}})
    summary = result["matrix_summary"]
    assert summary["added_volumes"] == ["2"] and summary["changed_pair_count"] == 0
    assert summary["volume_matching"] == "nearest_class" and summary["unclassified_volumes"] == []
    assert summary["reclassified_volumes"] == preview["reclassified_volumes"]
    assert events[-1]["matrix_summary"] == summary
    assert service.get_plant_profile(profile["profile_id"])["revision"] == profile["revision"]
    with pytest.raises(ValueError, match="classified"):
        tools["propose_profile_settings"].invoke({"summary": "Add dataset volumes.",
            "volume_compatibility": {"dataset_id": source["dataset_id"], "keep_existing_pairs": True,
                                     "default_status": "N"}})
