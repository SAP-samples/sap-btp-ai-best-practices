"""Application lifespan, public health, and production cleanup tests."""

from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from app.main import create_app, production_service_lifespan
from fastapi.testclient import TestClient


def test_openapi_and_source_tree_expose_only_the_product_surface() -> None:
    """Keep health/business routes and reject superseded template artifacts."""
    test_app = create_app()
    paths = set(test_app.openapi()["paths"])
    assert paths == {
        "/api/ecns/{ecn_row_id}/changes",
        "/api/ecns/{ecn_row_id}/drawing",
        "/api/ecns/{ecn_row_id}/snapshot", "/api/ecns/{ecn_row_id}/acceptance-plan", "/api/ecns/{ecn_row_id}/objects", "/api/ecns/{ecn_row_id}/object",
        "/api/chat", "/api/ecns/{ecn_row_id}/chat", "/api/ecns/{ecn_row_id}/reviews", "/api/ecns/{ecn_row_id}/reassess", "/api/ecns/{ecn_row_id}/versions", "/api/ecns/delete",
        "/api/summary-batches", "/api/summary-batches/{batch_id}", "/api/summary-batches/{batch_id}/cancel",
        "/api/health",
        "/api/uploads",
        "/api/drawing-feasibility/assessments",
        "/api/ecns",
        "/api/ecns/{ecn_row_id}",
        "/api/ecns/{ecn_row_id}/evidence",
        "/api/ecns/{ecn_row_id}/evidence/{evidence_id}",
        "/api/ecns/{ecn_row_id}/rdf-projection",
        "/api/ecns/{ecn_row_id}/rdf-projection/retry",
        "/api/analysis-progress",
        "/api/ecns/{ecn_row_id}/analysis/retry",
        "/api/ecns/{ecn_row_id}/analysis-stages",
        "/api/history",
        "/api/provider-status",
        "/api/provider-status/check",
        "/api/reference-datasets",
        "/api/reference-datasets/{dataset_id}",
        "/api/plant-settings",
    }
    retired = [
        "api/app/routers/llm_demo.py",
        "api/app/routers/agent_demo.py",
        "api/app/routers/hana_demo.py",
        "api/app/routers/joule.py",
        "api/app/utils/hana.py",
        "api/app/utils/langgraph/common.py",
        "api/app/utils/langgraph/tools.py",
        "api/app/models/llm.py",
        "ui/src/pages/home/home.html",
        "ui/src/pages/home/home.js",
        "ui/src/pages/home/home.css",
        "ui/public/images/avatars/man_avatar_3.png",
    ]
    assert [path for path in retired if Path(path).exists()] == []


def test_supplied_lifespan_avoids_all_production_factories(monkeypatch) -> None:
    """Run the complete test container on the app loop without HANA or provider I/O."""
    monkeypatch.setenv("API_KEY", "test-api-key")
    services = SimpleNamespace(marker="same-loop")

    @asynccontextmanager
    async def lifespan(_app):
        """Yield one test-owned container."""
        yield services

    app = create_app(
        engine_factory=lambda: (_ for _ in ()).throw(AssertionError("engine used")),
        provider_factory=lambda _settings: (_ for _ in ()).throw(
            AssertionError("provider used")
        ),
        service_lifespan_factory=lifespan,
    )
    with TestClient(app) as client:
        assert app.state.services is services
        assert client.get("/api/health").status_code == 200


@pytest.mark.asyncio
async def test_production_lifespan_disposes_engine_when_schema_startup_fails(
    monkeypatch,
) -> None:
    """Dispose an already-created engine even when schema validation fails."""
    events: list[str] = []

    class EngineFake:
        """Record disposal."""

        def dispose(self) -> None:
            """Record engine disposal."""
            events.append("engine_dispose")

    def fail_schema(_engine) -> None:
        """Simulate an incompatible persistence contract."""
        events.append("ensure_schema")
        raise RuntimeError("schema failed")

    class DrawingServiceFake:
        """Avoid creating an SDK client in this schema-failure unit test."""

        def __init__(self, _settings) -> None:
            """Accept the production constructor shape."""

        async def close(self) -> None:
            """Implement the application lifecycle contract."""

    monkeypatch.setattr("app.main.ensure_schema", fail_schema)
    monkeypatch.setattr("app.main.DrawingFeasibilityService", DrawingServiceFake)
    with pytest.raises(RuntimeError, match="schema failed"):
        async with production_service_lifespan(
            SimpleNamespace(), lambda: EngineFake(), lambda _settings: None
        ):
            pass
    assert events == ["ensure_schema", "engine_dispose"]


@pytest.mark.asyncio
async def test_production_lifespan_initializes_recovers_and_closes_in_order(
    monkeypatch,
) -> None:
    """Validate schema before recovery and close runtime before disposing HANA."""
    events: list[str] = []

    class EngineFake:
        """Record engine disposal."""

        def dispose(self) -> None:
            """Record disposal after asynchronous service shutdown."""
            events.append("engine_dispose")

    class AnalysisRepositoryFake:
        """Record restart recovery."""

        def __init__(self, _engine) -> None:
            """Accept the production constructor shape."""

        def fail_interrupted_jobs(self, _finished_at) -> int:
            """Record recovery and report no interrupted jobs."""
            events.append("recover_jobs")
            return 0

    class ProviderFake:
        """Record fallback close if no runtime owns the provider."""

        async def close(self) -> None:
            """Record provider close."""
            events.append("provider_close")

    class ProviderStatusFake:
        """Record startup preflight without contacting an external service."""

        def __init__(self, _config) -> None:
            """Accept the production configuration shape."""

        def check(self) -> None:
            """Record the startup provider preflight."""
            events.append("provider_preflight")

    class RuntimeFake:
        """Own and close the provider in the successful construction path."""

        def __init__(self, *args, **kwargs) -> None:
            """Record runtime construction."""
            events.append("build_runtime")
            self.provider_capacity = __import__("asyncio").Semaphore(2)

        async def close(self) -> None:
            """Record task/provider shutdown."""
            events.append("runtime_close")

    class AssistantRuntimeFake:
        """Require the application to pass the projection service into chat runtime."""

        def __init__(self, _snapshots, _ecns, _rdf, _model_name, _capacity, *, workspace) -> None:
            """Record construction after the trusted RDF boundary is ready."""
            assert workspace is not None, "review tools need the workspace repository"
            events.append("build_assistant")

        async def close(self) -> None:
            """Record assistant shutdown before the provider runtime closes."""
            events.append("assistant_close")

    class DrawingServiceFake:
        """Avoid creating a real SAP client while asserting lifecycle ownership."""

        def __init__(self, _settings) -> None:
            """Record construction before schema initialization."""
            events.append("build_drawing")

        async def close(self) -> None:
            """Record teardown before the shared provider runtime closes."""
            events.append("drawing_close")

    class GenerationServiceFake:

        def __init__(
            self,
            _repository,
            _provider,
            _telemetry,
                _clock,
                _uuid_factory,
                snapshot_service=None,
                discovery_service=None,
                rdf_projection_service=None,
            ) -> None:
                """Require the production discovery and RDF-currentness dependencies."""
                assert snapshot_service is not None
                assert discovery_service is not None
                assert rdf_projection_service is not None

    def ensure(_engine) -> None:
        """Record schema initialization."""
        events.append("ensure_schema")

    monkeypatch.setattr("app.main.ensure_schema", ensure)
    monkeypatch.setattr(
        "app.main.load_ecn_settings",
        lambda: SimpleNamespace(
            llm_max_concurrency=4,
            llm_model_name="gpt-5.4",
            llm_request_timeout_seconds=180,
        ),
    )
    monkeypatch.setattr("app.main.DrawingFeasibilityService", DrawingServiceFake)
    monkeypatch.setattr("app.main.UploadRepository", lambda _engine: SimpleNamespace())
    monkeypatch.setattr(
        "app.main.ArtifactRepository", lambda _engine: SimpleNamespace()
    )
    monkeypatch.setattr("app.main.EcnRepository", lambda _engine: SimpleNamespace())
    monkeypatch.setattr("app.main.AnalysisRepository", AnalysisRepositoryFake)
    monkeypatch.setattr("app.main.HistoryRepository", lambda _engine: SimpleNamespace())
    monkeypatch.setattr(
        "app.main.RdfRepository",
        lambda _engine: SimpleNamespace(
            ensure_capabilities=lambda: events.append("rdf_capability")
        ),
    )
    monkeypatch.setattr(
        "app.main.RdfProjectionService",
        lambda _repository, _clock: SimpleNamespace(
            reconcile=lambda: None,
            clear_all=lambda: None,
            delete_ecns=lambda _ids: None,
        ),
    )
    monkeypatch.setattr(
        "app.main.IngestionService", lambda **_kwargs: SimpleNamespace()
    )
    monkeypatch.setattr("app.main.HistoryService", lambda *_args: SimpleNamespace())
    monkeypatch.setattr("app.main.RetryService", lambda *_args: SimpleNamespace())
    monkeypatch.setattr("app.main.ProviderStatusService", ProviderStatusFake)
    monkeypatch.setattr("app.main.AnalysisGenerationService", GenerationServiceFake)
    monkeypatch.setattr("app.main.GenerationRuntime", RuntimeFake)
    monkeypatch.setattr("app.ecn.assistant.runtime.AgentRuntime", AssistantRuntimeFake)

    async with production_service_lifespan(
        SimpleNamespace(), lambda: EngineFake(), lambda _settings: ProviderFake()
    ) as services:
        assert services.runtime.__class__ is RuntimeFake

    assert events == [
        "build_drawing",
        "ensure_schema",
        "rdf_capability",
        "provider_preflight",
        "build_runtime",
        "build_assistant",
        "recover_jobs",
        "assistant_close",
        "drawing_close",
        "runtime_close",
        "engine_dispose",
    ]


@pytest.mark.asyncio
async def test_provider_and_engine_close_when_runtime_construction_fails(
    monkeypatch,
) -> None:
    """Close a constructed provider and engine after a later startup failure."""
    events: list[str] = []

    class EngineFake:
        """Record disposal."""

        def dispose(self) -> None:
            """Record disposal."""
            events.append("engine_dispose")

    class ProviderFake:
        """Record direct provider cleanup."""

        async def close(self) -> None:
            """Record close because no runtime was established."""
            events.append("provider_close")

    class ProviderStatusFake:
        """Avoid external network calls in the cleanup-path test."""

        def __init__(self, _config) -> None:
            """Accept the production configuration shape."""

        def check(self) -> None:
            """Complete one inert startup preflight."""

    class DrawingServiceFake:
        """Avoid constructing a real SAP client in the failure cleanup test."""

        def __init__(self, _settings) -> None:
            """Record construction before the later runtime failure."""
            events.append("build_drawing")

        async def close(self) -> None:
            """Record independent drawing-service cleanup."""
            events.append("drawing_close")

    monkeypatch.setattr("app.main.ensure_schema", lambda _engine: None)
    monkeypatch.setattr(
        "app.main.load_ecn_settings",
        lambda: SimpleNamespace(
            llm_max_concurrency=4,
            llm_model_name="gpt-5.4",
            llm_request_timeout_seconds=180,
        ),
    )
    monkeypatch.setattr("app.main.DrawingFeasibilityService", DrawingServiceFake)
    monkeypatch.setattr("app.main.UploadRepository", lambda _engine: SimpleNamespace())
    monkeypatch.setattr(
        "app.main.ArtifactRepository", lambda _engine: SimpleNamespace()
    )
    monkeypatch.setattr("app.main.EcnRepository", lambda _engine: SimpleNamespace())
    monkeypatch.setattr(
        "app.main.AnalysisRepository",
        lambda _engine: SimpleNamespace(fail_interrupted_jobs=lambda _time: 0),
    )
    monkeypatch.setattr("app.main.HistoryRepository", lambda _engine: SimpleNamespace())
    monkeypatch.setattr(
        "app.main.RdfRepository",
        lambda _engine: SimpleNamespace(ensure_capabilities=lambda: None),
    )
    monkeypatch.setattr(
        "app.main.RdfProjectionService",
        lambda _repository, _clock: SimpleNamespace(
            reconcile=lambda: None,
            clear_all=lambda: None,
            delete_ecns=lambda _ids: None,
        ),
    )
    monkeypatch.setattr(
        "app.main.IngestionService", lambda **_kwargs: SimpleNamespace()
    )
    monkeypatch.setattr("app.main.HistoryService", lambda *_args: SimpleNamespace())
    monkeypatch.setattr("app.main.RetryService", lambda *_args: SimpleNamespace())
    monkeypatch.setattr("app.main.ProviderStatusService", ProviderStatusFake)
    monkeypatch.setattr(
        "app.main.AnalysisGenerationService", lambda *_args, **_kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(
        "app.main.GenerationRuntime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("runtime failed")),
    )

    with pytest.raises(RuntimeError, match="runtime failed"):
        async with production_service_lifespan(
            SimpleNamespace(), lambda: EngineFake(), lambda _settings: ProviderFake()
        ):
            pass
    assert events == ["build_drawing", "drawing_close", "provider_close", "engine_dispose"]
