"""Tests for safe SAP AI Core provider preflight and status projection."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import asynccontextmanager
from types import SimpleNamespace

from app.main import create_app
from app.ecn.llm.preflight import (
    AiCorePreflightConfig,
    ProviderAvailability,
    ProviderStatusService,
)
from fastapi.testclient import TestClient


class FakeResponse:
    """Provide the response subset consumed by provider preflight."""

    def __init__(
        self,
        status_code: int,
        payload: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        """Store one fake HTTP status, JSON body, and response headers."""
        self.status_code = status_code
        self._payload = dict(payload or {})
        self.headers = dict(headers or {})

    def json(self) -> Mapping[str, object]:
        """Return the configured JSON mapping."""
        return self._payload


class FakeSession:
    """Record preflight calls and return queued fake responses."""

    def __init__(self, responses: list[FakeResponse]) -> None:
        """Initialize a FIFO response queue and a sanitized call log."""
        self.responses = responses
        self.calls: list[tuple[str, str]] = []
        self.oauth_form_keys: frozenset[str] = frozenset()
        self.used_basic_auth = False

    def post(self, url: str, **kwargs: object) -> FakeResponse:
        """Record an OAuth POST and return the next response."""
        data = kwargs.get("data")
        self.oauth_form_keys = (
            frozenset(str(key) for key in data) if isinstance(data, Mapping) else frozenset()
        )
        self.used_basic_auth = "auth" in kwargs
        self.calls.append(("POST", url))
        return self.responses.pop(0)

    def get(self, url: str, **kwargs: object) -> FakeResponse:
        """Record an AI Core GET and return the next response."""
        del kwargs
        self.calls.append(("GET", url))
        return self.responses.pop(0)


class RecordingBlockedRepository:
    """Return provider-blocked jobs and record recovery requests."""

    def __init__(self, job_ids: tuple[str, ...]) -> None:
        """Store the stable IDs returned by a successful recovery."""
        self.job_ids = job_ids
        self.calls = 0

    def queue_provider_blocked_jobs(self) -> tuple[str, ...]:
        """Record and return the configured provider-blocked job IDs."""
        self.calls += 1
        return self.job_ids


class RecordingRuntime:
    """Record batches scheduled after provider recovery."""

    def __init__(self) -> None:
        """Initialize an empty scheduled-batch log."""
        self.scheduled: list[tuple[str, ...]] = []

    def schedule_jobs(self, job_ids: tuple[str, ...]) -> None:
        """Record one recovered batch without starting background work."""
        self.scheduled.append(job_ids)


def _config(**overrides: str) -> AiCorePreflightConfig:
    """Build one valid non-production credential set for unit tests."""
    values = {
        "auth_url": "https://tenant.example.test/oauth/token",
        "client_id": "client-id-secret",
        "client_secret": "client-secret-value",
        "base_url": "https://tenant.example.test",
        "resource_group": "default",
        "llm_model": "gpt-5.4",
    }
    values.update(overrides)
    return AiCorePreflightConfig(**values)


def test_preflight_reports_missing_configuration_without_network_calls() -> None:
    """Return one safe configuration state when OAuth values are incomplete."""
    session = FakeSession([])
    service = ProviderStatusService(
        AiCorePreflightConfig.from_environ({"LLM_MODEL": "gpt-5.4"}),
        session=session,
    )
    result = service.check()
    assert result.oauth.status is ProviderAvailability.CONFIGURATION_MISSING
    assert all(item.status is ProviderAvailability.NOT_CHECKED for item in result.models)
    assert session.calls == []


def test_preflight_classifies_oauth_401_and_redacts_credentials() -> None:
    """Expose an authentication category without response content or URLs."""
    service = ProviderStatusService(
        _config(),
        session=FakeSession([FakeResponse(401, {"error": "client-secret-value"})]),
    )
    result = service.check()
    assert result.oauth.status is ProviderAvailability.AUTH_FAILED
    rendered = repr(result)
    assert "client-secret-value" not in rendered
    assert "tenant.example.test" not in rendered


def _discovery_session(llm_deployment_status: str) -> FakeSession:
    """Build an OAuth + discovery exchange for the text-generation model.

    Args:
        llm_deployment_status: AI Core status of the ``gpt-5.4`` deployment.

    Returns:
        A fake session that also lists a RUNNING deployment of a similarly
        named model, so tests prove that matching is exact.
    """
    return FakeSession(
        [
            FakeResponse(200, {"access_token": "oauth-token-secret"}),
            FakeResponse(
                200,
                {
                    "resources": [
                        {"id": "cfg-llm", "name": "gpt-5.4"},
                        {"id": "cfg-other", "name": "gpt-5.4-mini"},
                    ]
                },
            ),
            FakeResponse(
                200,
                {
                    "resources": [
                        {
                            "id": "dep-llm",
                            "configurationId": "cfg-llm",
                            "status": llm_deployment_status,
                        },
                        {"id": "dep-other", "configurationId": "cfg-other", "status": "RUNNING"},
                    ]
                },
            ),
        ]
    )


def test_preflight_resolves_running_text_generation_model() -> None:
    """Report the text-generation model available when its deployment RUNS."""
    session = _discovery_session("RUNNING")
    result = ProviderStatusService(_config(), session=session).check()
    assert result.oauth.status is ProviderAvailability.AVAILABLE
    assert [(item.capability, item.model) for item in result.models] == [
        ("text_generation", "gpt-5.4")
    ]
    assert result.models[0].status is ProviderAvailability.AVAILABLE
    assert all("oauth-token-secret" not in repr(call) for call in session.calls)


def test_preflight_ignores_stopped_deployment_and_similar_model_names() -> None:
    """Report NOT_AVAILABLE when only a similarly named model is RUNNING."""
    result = ProviderStatusService(
        _config(), session=_discovery_session("STOPPED")
    ).check()
    assert result.models[0].status is ProviderAvailability.NOT_AVAILABLE


def test_preflight_normalizes_service_key_base_urls() -> None:
    """Append OAuth and LM paths exactly once for standard service-key URLs."""
    session = FakeSession(
        [
            FakeResponse(200, {"access_token": "token"}),
            FakeResponse(200, {"resources": []}),
            FakeResponse(200, {"resources": []}),
        ]
    )
    config = _config(
        auth_url="https://auth.example.test",
        base_url="https://api.example.test/v2",
    )
    ProviderStatusService(config, session=session).check()
    assert session.calls == [
        ("POST", "https://auth.example.test/oauth/token"),
        ("GET", "https://api.example.test/v2/lm/configurations"),
        ("GET", "https://api.example.test/v2/lm/deployments"),
    ]
    assert session.oauth_form_keys == {
        "grant_type",
        "client_id",
        "client_secret",
    }
    assert session.used_basic_auth is False


def test_status_read_does_not_trigger_a_preflight() -> None:
    """Return cached NOT_CHECKED state until an explicit check is requested."""
    session = FakeSession([])
    service = ProviderStatusService(_config(), session=session)
    result = service.get_status()
    assert result.oauth.status is ProviderAvailability.NOT_CHECKED
    assert session.calls == []


def test_generation_blocker_uses_oauth_before_model_status() -> None:
    """Classify missing credentials before the uninspected model placeholder."""
    service = ProviderStatusService(
        AiCorePreflightConfig.from_environ({"LLM_MODEL": "gpt-5.4"}),
        session=FakeSession([]),
    )
    service.check()
    assert service.generation_blocker() == "PROVIDER_AUTH_FAILED"


def test_generation_blocker_clears_when_text_model_is_running() -> None:
    """Allow generation even when optional prototype capabilities are unavailable."""
    session = FakeSession(
        [
            FakeResponse(200, {"access_token": "token"}),
            FakeResponse(200, {"resources": [{"id": "cfg-llm", "name": "gpt-5.4"}]}),
            FakeResponse(
                200,
                {
                    "resources": [
                        {
                            "id": "dep-llm",
                            "configurationId": "cfg-llm",
                            "status": "RUNNING",
                        }
                    ]
                },
            ),
        ]
    )
    service = ProviderStatusService(_config(), session=session)
    service.check()
    assert service.generation_blocker() is None


def test_provider_status_endpoints_read_cache_and_trigger_check(monkeypatch) -> None:
    """Expose a read-only cached GET and an explicit checking POST."""
    monkeypatch.setenv("API_KEY", "test-api-key")
    service = ProviderStatusService(
        AiCorePreflightConfig.from_environ({"LLM_MODEL": "gpt-5.4"}),
        session=FakeSession([]),
    )

    @asynccontextmanager
    async def lifespan(_app):
        """Yield the provider-status service required by these routes."""
        yield SimpleNamespace(provider_status_service=service)

    headers = {"X-API-Key": "test-api-key"}
    with TestClient(create_app(service_lifespan_factory=lifespan)) as client:
        cached = client.get("/api/provider-status", headers=headers)
        checked = client.post("/api/provider-status/check", headers=headers)
    assert cached.status_code == 200
    assert cached.json()["oauth"]["status"] == "NOT_CHECKED"
    assert checked.status_code == 200
    assert checked.json()["oauth"]["status"] == "CONFIGURATION_MISSING"


def test_successful_status_check_requeues_and_schedules_provider_blocked_jobs(
    monkeypatch,
) -> None:
    """Resume held analysis jobs only after the configured text model is available."""
    monkeypatch.setenv("API_KEY", "test-api-key")
    service = ProviderStatusService(
        _config(),
        session=FakeSession(
            [
                FakeResponse(200, {"access_token": "token"}),
                FakeResponse(
                    200,
                    {"resources": [{"id": "cfg-llm", "name": "gpt-5.4"}]},
                ),
                FakeResponse(
                    200,
                    {
                        "resources": [
                            {
                                "id": "dep-llm",
                                "configurationId": "cfg-llm",
                                "status": "RUNNING",
                            }
                        ]
                    },
                ),
            ]
        ),
    )
    job_ids = ("00000000-0000-4000-8000-000000000001",)
    repository = RecordingBlockedRepository(job_ids)
    runtime = RecordingRuntime()

    @asynccontextmanager
    async def lifespan(_app):
        """Yield only the collaborators exercised by the status route."""
        yield SimpleNamespace(
            provider_status_service=service,
            analysis_repository=repository,
            runtime=runtime,
        )

    with TestClient(create_app(service_lifespan_factory=lifespan)) as client:
        checked = client.post(
            "/api/provider-status/check",
            headers={"X-API-Key": "test-api-key"},
        )
    assert checked.status_code == 200
    assert repository.calls == 1
    assert runtime.scheduled == [job_ids]
