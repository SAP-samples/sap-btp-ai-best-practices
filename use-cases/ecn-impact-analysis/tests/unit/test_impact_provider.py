"""Contract tests for the native asynchronous structured-output provider."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from app.ecn.llm.prompts import project_evidence_for_prompt
from app.ecn.llm.staged import FactorStageResult, ReadinessStageResult, StagePrompt, SummaryStageResult
from app.ecn.llm.provider import (
    NativeImpactProvider,
    ProviderCallStage,
    ProviderCallStatus,
    create_impact_provider,
)
from app.ecn.llm.two_pass import build_planner_projection
from app.ecn.settings import EcnSettings
from app.observability import TokenUsage
from app.persistence.records import PinnedGenerationConfig

from tests.fixtures.builders import make_evidence_bundle


class FakeResponses:
    """Record native Responses API calls and return one configured response."""

    def __init__(self, response: object | None = None, error: Exception | None = None):
        """Store the response or exception used by the fake async endpoint."""
        self.response = response or SimpleNamespace(
            status="completed",
            output_text='{"sections":{}}',
            output=[],
            usage={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            model="deployment-1",
            id="response-1",
        )
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        """Record one request before returning or raising its configured outcome."""
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeNativeClient:
    """Expose the subset of AsyncOpenAI used by the production adapter."""

    def __init__(self, response: object | None = None, error: Exception | None = None):
        """Create a fake client with a nested Responses endpoint."""
        self.responses = FakeResponses(response, error)
        self.closed = False

    @property
    def call_count(self) -> int:
        """Return the number of provider requests observed."""
        return len(self.responses.calls)

    @property
    def last_call(self) -> dict[str, object]:
        """Return the most recent request keyword arguments."""
        return self.responses.calls[-1]

    async def close(self) -> None:
        """Record that the provider awaited client shutdown."""
        self.closed = True


class FakeMonotonicClock:
    """Return deterministic start and finish monotonic timestamps."""

    def __init__(self) -> None:
        """Prepare two timestamps corresponding to 42 milliseconds."""
        self.values = iter((10.0, 10.042))

    def __call__(self) -> float:
        """Return the next deterministic monotonic timestamp."""
        return next(self.values)


class FakeHttpError(RuntimeError):
    """Expose only the HTTP metadata used by safe provider classification."""

    def __init__(
        self,
        status_code: int,
        *,
        provider_code: str | None = None,
        request_id: str | None = None,
        retry_after: str | None = None,
    ) -> None:
        """Create a provider-like exception without relying on SDK internals."""
        headers: dict[str, str] = {}
        if request_id is not None:
            headers["x-request-id"] = request_id
        if retry_after is not None:
            headers["retry-after"] = retry_after
        body = {"error": {"code": provider_code}} if provider_code else {}
        self.response = SimpleNamespace(
            status_code=status_code,
            headers=headers,
            json=lambda: body,
        )
        super().__init__("credential and tenant details must never be persisted")


class RecordingSleeper:
    """Record retry delays without making provider tests wait."""

    def __init__(self) -> None:
        """Initialize an empty delay log."""
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        """Record one requested asynchronous retry delay."""
        self.delays.append(delay)


@pytest.fixture
def fake_native_client() -> FakeNativeClient:
    """Return a successful fake native client."""
    return FakeNativeClient()


@pytest.fixture
def projection():
    """Return a minimal bounded projection for provider contract tests."""
    return project_evidence_for_prompt(make_evidence_bundle(), 10, 20_000)


@pytest.fixture
def pinned_config() -> PinnedGenerationConfig:
    """Return one strict pinned provider configuration."""
    return PinnedGenerationConfig(
        prompt_version="ecn-impact-v1",
        output_schema_version="ecn-impact-sections-v1",
        model_name="gpt-5.4",
        request_timeout_seconds=180,
        max_prompt_evidence_items=10,
        max_prompt_characters=20_000,
        strict_structured_output=True,
    )


@pytest.mark.asyncio
async def test_provider_uses_one_strict_async_request(
    fake_native_client: FakeNativeClient, projection, pinned_config
) -> None:
    """Send one request with the pinned model, timeout, messages, and JSON schema."""
    provider = NativeImpactProvider(
        client=fake_native_client,
        monotonic=FakeMonotonicClock(),
    )
    result = await provider.generate(projection, pinned_config)
    assert fake_native_client.call_count == 1
    call = fake_native_client.last_call
    assert call["model"] == pinned_config.model_name
    assert call["timeout"] == pinned_config.request_timeout_seconds
    assert call["store"] is False
    assert call["text"]["format"]["strict"] is True
    schema = call["text"]["format"]["schema"]
    assert set(schema["required"]) == set(schema["properties"])
    assert "default" not in schema["properties"]["overall_impact"]
    impact_factors = schema["$defs"]["ImpactFactors"]
    assert set(impact_factors["required"]) == set(impact_factors["properties"])
    assert result.status is ProviderCallStatus.RECEIVED
    assert result.raw_json is not None
    assert result.raw_json.startswith("{")
    assert result.latency_ms == 42
    assert result.usage.usage_available is True


@pytest.mark.asyncio
async def test_planner_schema_requires_every_declared_property(
    fake_native_client: FakeNativeClient, pinned_config
) -> None:
    """Generate an OpenAI-compatible strict schema for planner defaults."""
    provider = NativeImpactProvider(
        client=fake_native_client,
        monotonic=FakeMonotonicClock(),
    )

    await provider.generate(
        build_planner_projection(make_evidence_bundle()),
        pinned_config,
        stage=ProviderCallStage.PLANNER,
    )

    schema = fake_native_client.last_call["text"]["format"]["schema"]
    assert set(schema["required"]) == set(schema["properties"])
    assert "focus_material_numbers" in schema["required"]


@pytest.mark.asyncio
async def test_provider_builds_strict_schemas_for_every_staged_call(
    fake_native_client: FakeNativeClient, pinned_config,
) -> None:
    """Send each perspective through the native strict-schema adapter."""
    provider = NativeImpactProvider(fake_native_client, monotonic=lambda: 10.0)
    schemas = {
        ProviderCallStage.MANUFACTURING: FactorStageResult,
        ProviderCallStage.PROCUREMENT: FactorStageResult,
        ProviderCallStage.CONFIGURATION: FactorStageResult,
        ProviderCallStage.DATA_READINESS: ReadinessStageResult,
        ProviderCallStage.SUMMARY: SummaryStageResult,
    }
    for stage, schema_type in schemas.items():
        await provider.generate(StagePrompt("Grounded instructions", {}, schema_type), pinned_config, stage)
        request = fake_native_client.last_call
        schema = request["text"]["format"]["schema"]
        assert request["text"]["format"]["name"] == f"ecn_{stage.value.lower()}_v5"
        assert set(schema["required"]) == set(schema["properties"])
        if stage is ProviderCallStage.SUMMARY:
            assert "pattern" in schema["$defs"]["AnalysisSections"]["properties"]["executive_summary"]["properties"]["assessment_markdown"]
        elif stage is not ProviderCallStage.DATA_READINESS:
            assert "pattern" in schema["$defs"]["ImpactFactorAssessment"]["properties"]["assessment"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(
            status="incomplete",
            output_text='{"sections":{}}',
            output=[],
            usage={"input_tokens": 1},
            model=None,
            id=None,
        ),
        SimpleNamespace(
            status="completed",
            output_text=None,
            output=[SimpleNamespace(type="refusal", refusal="secret provider text")],
            usage=None,
            model=None,
            id=None,
        ),
    ],
)
async def test_received_noncompletion_preserves_metadata_for_validation(
    response: object, projection, pinned_config
) -> None:
    """Return incomplete, refusal, and missing output as received provider calls."""
    provider = NativeImpactProvider(
        FakeNativeClient(response=response), FakeMonotonicClock()
    )
    result = await provider.generate(projection, pinned_config)
    assert result.status is ProviderCallStatus.RECEIVED
    assert result.error_code is None
    assert "secret provider text" not in repr(result)


@pytest.mark.asyncio
async def test_provider_exception_is_safe_and_is_not_retried(
    projection, pinned_config
) -> None:
    """Convert a transport failure to a fixed safe result after exactly one request."""
    client = FakeNativeClient(
        error=RuntimeError("credential and source payload secret")
    )
    provider = NativeImpactProvider(client, FakeMonotonicClock())
    result = await provider.generate(projection, pinned_config)
    assert client.call_count == 1
    assert result.status is ProviderCallStatus.ERROR
    assert result.error_code == "PROVIDER_RESPONSE_INVALID"
    assert result.raw_json is None
    assert "secret" not in repr(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_code", "expected_retryable"),
    [
        (FakeHttpError(401), "PROVIDER_AUTH_FAILED", False),
        (FakeHttpError(403), "PROVIDER_FORBIDDEN", False),
        (FakeHttpError(404, provider_code="DeploymentNotFound"), "MODEL_NOT_AVAILABLE", False),
        (FakeHttpError(429), "PROVIDER_RATE_LIMITED", True),
        (asyncio.TimeoutError(), "PROVIDER_TIMEOUT", True),
        (FakeHttpError(503), "PROVIDER_SERVER_ERROR", True),
        (FakeHttpError(422), "PROVIDER_RESPONSE_INVALID", False),
    ],
)
async def test_provider_classifies_safe_failure_categories(
    error: Exception,
    expected_code: str,
    expected_retryable: bool,
    projection,
    pinned_config,
) -> None:
    """Map provider failures to bounded public categories without leaking text."""
    client = FakeNativeClient(error=error)
    provider = NativeImpactProvider(
        client,
        FakeMonotonicClock(),
        max_attempts=1,
    )
    result = await provider.generate(projection, pinned_config)
    assert result.status is ProviderCallStatus.ERROR
    assert result.error_code == expected_code
    assert result.retryable is expected_retryable
    assert "credential" not in repr(result)


@pytest.mark.asyncio
async def test_provider_preserves_bounded_safe_http_metadata(
    projection, pinned_config
) -> None:
    """Retain status, provider code, request ID, stage, and retry-after only."""
    error = FakeHttpError(
        429,
        provider_code="RateLimitExceeded",
        request_id="request-safe-123",
        retry_after="12",
    )
    result = await NativeImpactProvider(
        FakeNativeClient(error=error),
        FakeMonotonicClock(),
        max_attempts=1,
    ).generate(
        build_planner_projection(make_evidence_bundle()),
        pinned_config,
        stage=ProviderCallStage.PLANNER,
    )
    assert result.http_status == 429
    assert result.provider_code == "RateLimitExceeded"
    assert result.provider_request_id == "request-safe-123"
    assert result.retry_after_seconds == 12
    assert result.stage is ProviderCallStage.PLANNER


@pytest.mark.asyncio
async def test_provider_retries_only_transient_failures(
    projection, pinned_config
) -> None:
    """Retry a transient 5xx with bounded backoff and stop after the cap."""
    client = FakeNativeClient(error=FakeHttpError(503))
    sleeper = RecordingSleeper()
    provider = NativeImpactProvider(
        client,
        monotonic=lambda: 10.0,
        max_attempts=3,
        sleep=sleeper,
        jitter=lambda: 0.0,
        retry_base_seconds=0.25,
        retry_max_seconds=1.0,
    )
    result = await provider.generate(projection, pinned_config)
    assert result.error_code == "PROVIDER_SERVER_ERROR"
    assert client.call_count == 3
    assert sleeper.delays == [0.25, 0.5]


@pytest.mark.asyncio
async def test_provider_never_retries_authentication_failure(
    projection, pinned_config
) -> None:
    """Stop immediately for a 401 because new credentials are required."""
    client = FakeNativeClient(error=FakeHttpError(401))
    sleeper = RecordingSleeper()
    provider = NativeImpactProvider(
        client,
        monotonic=lambda: 10.0,
        max_attempts=3,
        sleep=sleeper,
    )
    result = await provider.generate(projection, pinned_config)
    assert result.error_code == "PROVIDER_AUTH_FAILED"
    assert client.call_count == 1
    assert sleeper.delays == []


@pytest.mark.asyncio
async def test_provider_does_not_swallow_cancellation(
    projection, pinned_config
) -> None:
    """Allow application timeout and shutdown cancellation to reach the service."""
    client = FakeNativeClient(error=asyncio.CancelledError())
    provider = NativeImpactProvider(client, FakeMonotonicClock())
    with pytest.raises(asyncio.CancelledError):
        await provider.generate(projection, pinned_config)


@pytest.mark.asyncio
async def test_provider_bounds_optional_identifiers(projection, pinned_config) -> None:
    """Hash optional provider labels that exceed persistence column widths."""
    response = SimpleNamespace(
        status="completed",
        output_text='{"sections":{}}',
        output=[],
        usage={},
        model="m" * 500,
        id="r" * 500,
    )
    result = await NativeImpactProvider(
        FakeNativeClient(response=response), FakeMonotonicClock()
    ).generate(projection, pinned_config)
    assert result.model_deployment.startswith("sha256:")
    assert result.provider_request_id.startswith("sha256:")
    assert len(result.model_deployment) <= 200


@pytest.mark.asyncio
async def test_provider_close_awaits_native_client(
    fake_native_client: FakeNativeClient,
) -> None:
    """Close the long-lived native client during application shutdown."""
    provider = NativeImpactProvider(fake_native_client)
    await provider.close()
    assert fake_native_client.closed is True


def test_factory_builds_native_client_without_sdk_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construct the SAP Gen AI Hub native AsyncOpenAI client with retries disabled."""
    recorded: dict[str, object] = {}

    class FakeAsyncOpenAI:
        """Capture constructor keywords for the provider factory."""

        def __init__(self, **kwargs: object) -> None:
            """Record construction and expose a Responses-like surface."""
            recorded.update(kwargs)
            self.responses = SimpleNamespace(create=lambda **_: None)

        async def close(self) -> None:
            """Provide the native asynchronous close contract."""

    monkeypatch.setattr("app.ecn.llm.provider.AsyncOpenAI", FakeAsyncOpenAI)
    provider = create_impact_provider(EcnSettings())
    assert isinstance(provider, NativeImpactProvider)
    assert recorded == {"max_retries": 0}


def test_requirements_pin_responses_capable_sdk_family() -> None:
    """Require the supported SAP AI SDK family used by the native Responses surface."""
    requirements = Path("api/requirements.txt").read_text(encoding="utf-8").splitlines()
    assert "sap-ai-sdk-gen[all]==7.2.0" in requirements


def test_missing_usage_is_not_reported_as_measured_zero(
    projection, pinned_config
) -> None:
    """Retain the semantic distinction between absent metadata and zero tokens."""
    response = SimpleNamespace(
        status="completed",
        output_text='{"sections":{}}',
        output=[],
        usage=None,
        model=None,
        id=None,
    )

    async def run() -> TokenUsage:
        """Run one provider call and return its normalized usage."""
        result = await NativeImpactProvider(
            FakeNativeClient(response=response), FakeMonotonicClock()
        ).generate(projection, pinned_config)
        return result.usage

    assert asyncio.run(run()) == TokenUsage()


def test_snapshot_generation_schema_enforces_bullet_bounds_without_changing_legacy():
    """Provider constraints accept concise qualified findings and reject extra bullets."""
    import re
    from app.ecn.analysis_models import AnalysisResponse
    from app.ecn.llm.provider import _strict_provider_schema
    schema = _strict_provider_schema(AnalysisResponse.model_json_schema(), concise=True)
    definitions = schema["$defs"]
    executive = definitions["AnalysisSections"]["properties"]["executive_summary"]
    for field, maximum in [(executive["properties"]["assessment_markdown"], 5),
                           (definitions["ImpactFactorAssessment"]["properties"]["assessment"], 3)]:
        passage = "\n".join(["- Still used elsewhere; configuration remains unresolved."] * maximum)
        assert re.fullmatch(field["pattern"], passage)
        assert not re.fullmatch(field["pattern"], passage + "\n- One extra finding.")
        assert not re.fullmatch(field["pattern"], "A paragraph without bullets.")
    legacy = _strict_provider_schema(AnalysisResponse.model_json_schema())
    assert "pattern" not in legacy["$defs"]["ImpactFactorAssessment"]["properties"]["assessment"]


@pytest.mark.asyncio
async def test_staged_citation_schema_restricts_output_to_supplied_ids(fake_native_client, pinned_config):
    """The provider must choose citations from the projection, never invent IDs."""
    from app.ecn.llm.staged import build_stage_prompt
    from app.ecn.llm.synthesis import build_synthesis_projection
    from dataclasses import replace
    from app.ecn.models import AnalysisSectionKey
    projection = replace(build_synthesis_projection(make_evidence_bundle()),
                         header_citation_ids=("E-0123456789abcdef",),
                         allowed_evidence_ids_by_section={AnalysisSectionKey.EXECUTIVE_SUMMARY: frozenset({"E-0123456789abcdef"}),
                                                          AnalysisSectionKey.CONFIGURATION_SCOPE: frozenset()})
    provider = NativeImpactProvider(fake_native_client, monotonic=lambda: 10.0)
    prompt = build_stage_prompt(projection, 'MANUFACTURING')
    await provider.generate(prompt, pinned_config, ProviderCallStage.MANUFACTURING)
    schema = fake_native_client.last_call['text']['format']['schema']
    citation_items = schema['$defs']['ImpactFactorAssessment']['properties']['evidence_ids']['items']
    assert schema['$defs']['ImpactFactorAssessment']['properties']['assessment']['maxLength'] <= 550
    expected = set(projection.header_citation_ids) | {cid for row in projection.records for cid in row.citation_ids}
    assert set(citation_items['enum']) == expected
    prompt = build_stage_prompt(projection, 'SUMMARY')
    await provider.generate(prompt, pinned_config, ProviderCallStage.SUMMARY)
    schema = fake_native_client.last_call['text']['format']['schema']
    for section, ids in projection.allowed_evidence_ids_by_section.items():
        field = schema['$defs']['AnalysisSections']['properties'][section.value]['properties']['evidence_ids']
        assert set(field['items'].get('enum', [])) == set(ids)
        if not ids:
            assert field['maxItems'] == 0


@pytest.mark.asyncio
async def test_summary_schema_requires_known_gaps(fake_native_client, pinned_config):
    """A deterministic required gap cannot be generated as a sufficient section."""
    from dataclasses import replace
    from app.ecn.llm.staged import build_stage_prompt
    from app.ecn.llm.synthesis import build_synthesis_projection
    from app.ecn.models import AnalysisSectionKey
    gap = AnalysisSectionKey.CONFIGURATION_SCOPE
    projection = replace(build_synthesis_projection(make_evidence_bundle()), sections_requiring_gap=frozenset({gap}))
    prompt = build_stage_prompt(projection, 'SUMMARY')
    await NativeImpactProvider(fake_native_client, monotonic=lambda: 10.0).generate(prompt, pinned_config, ProviderCallStage.SUMMARY)
    schema = fake_native_client.last_call['text']['format']['schema']
    properties = schema['$defs']['AnalysisSections']['properties'][gap.value]['properties']
    assert properties['status'] == {'type': 'string', 'enum': ['INSUFFICIENT_EVIDENCE']}
    assert properties['missing_evidence']['minItems'] == 1
