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
