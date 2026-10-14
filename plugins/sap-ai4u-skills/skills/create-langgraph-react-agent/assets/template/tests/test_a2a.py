"""Exercise A2A 1.0 and Joule-compatible 0.3 requests against the ASGI app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from starlette.testclient import TestClient

from template_agent.a2a_server import create_a2a_app


@dataclass
class FakeAgentResult:
    """Provide the one runtime result field consumed by the A2A adapter."""

    output_text: str


class FakeRuntime:
    """Record A2A/runtime interactions without SAP services or credentials."""

    def __init__(self, *, failure: Exception | None = None) -> None:
        """Configure an optional exception for the next runtime invocation."""

        self.failure = failure
        self.calls: list[tuple[str, str]] = []
        self.closed = False

    async def ainvoke(
        self,
        text: str,
        context_id: str,
        attachments: tuple[Any, ...] = (),
        response_model: type[Any] | dict[str, Any] | None = None,
    ) -> FakeAgentResult:
        """Return deterministic text while retaining the mapped context ID."""

        assert attachments == ()
        assert response_model is None
        self.calls.append((text, context_id))
        if self.failure:
            raise self.failure
        return FakeAgentResult(output_text=f"answer: {text}")

    async def aclose(self) -> None:
        """Record that application shutdown released the runtime."""

        self.closed = True


def _write_config(tmp_path: Path) -> Path:
    """Create the smallest enabled A2A configuration for server tests."""

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (tmp_path / "agent_skills").mkdir()
    config = config_dir / "agent.yaml"
    config.write_text(
        """
base_prompt: Test
model:
  provider: openai
  name: gpt-test
skills:
  directory: ../agent_skills
a2a:
  enabled: true
  name: Test Agent
  description: Test A2A adapter
  version: 1.2.3
  public_url: https://agent.example.test/
  host: 127.0.0.1
  port: 9090
""".strip(),
        encoding="utf-8",
    )
    return config


def _app(tmp_path: Path, runtime: FakeRuntime) -> Any:
    """Build an application whose lifespan returns the supplied fake runtime."""

    async def runtime_factory(config_path: str | Path) -> FakeRuntime:
        """Verify the requested configuration and return the fake runtime."""

        assert Path(config_path) == _write_config_path
        return runtime

    _write_config_path = _write_config(tmp_path)
    return create_a2a_app(_write_config_path, runtime_factory=runtime_factory)


def _v1_request(text: str, context_id: str = "context-v1") -> dict[str, Any]:
    """Return one current A2A ProtoJSON `SendMessage` request."""

    return {
        "jsonrpc": "2.0",
        "id": "request-v1",
        "method": "SendMessage",
        "params": {
            "message": {
                "messageId": "message-v1",
                "contextId": context_id,
                "role": "ROLE_USER",
                "parts": [{"text": text}],
            }
        },
    }


def _joule_v03_request(
    text: str, context_id: str = "context-joule"
) -> dict[str, Any]:
    """Return the legacy JSON-RPC shape sent through Joule `agent-request`."""

    return {
        "jsonrpc": "2.0",
        "id": "request-joule",
        "method": "message/send",
        "params": {
            "message": {
                "messageId": "message-joule",
                "contextId": context_id,
                "role": "user",
                "parts": [{"kind": "text", "text": text}],
            }
        },
    }


def test_agent_cards_advertise_current_and_joule_interfaces(tmp_path: Path) -> None:
    """Serve equivalent cards on current and legacy discovery paths."""

    runtime = FakeRuntime()
    app = _app(tmp_path, runtime)

    with TestClient(app) as client:
        current = client.get("/.well-known/agent-card.json")
        legacy = client.get("/.well-known/agent.json")

        assert current.status_code == 200
        assert legacy.status_code == 200
        assert current.json() == legacy.json()
        card = current.json()
        assert card["name"] == "Test Agent"
        assert card["url"] == "https://agent.example.test"
        assert card["protocolVersion"] == "0.3"
        assert {
            item["protocolVersion"] for item in card["supportedInterfaces"]
        } == {"1.0", "0.3"}

    assert runtime.closed is True


def test_current_a2a_request_reuses_context_and_returns_artifact(tmp_path: Path) -> None:
    """Route A2A 1.0 text through the existing runtime and return a task."""

    runtime = FakeRuntime()
    app = _app(tmp_path, runtime)

    with TestClient(app) as client:
        response = client.post(
            "/",
            json=_v1_request("hello current"),
            headers={"A2A-Version": "1.0"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert "error" not in payload
    task = payload["result"]["task"]
    assert task["contextId"] == "context-v1"
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert task["artifacts"][0]["name"] == "agent_result"
    assert task["artifacts"][0]["parts"][0]["text"] == "answer: hello current"
    assert runtime.calls == [("hello current", "context-v1")]


def test_joule_v03_request_reuses_context_and_returns_expected_shape(
    tmp_path: Path,
) -> None:
    """Prove SDK 1.1.2 accepts Joule's 0.3 `message/send` contract."""

    runtime = FakeRuntime()
    app = _app(tmp_path, runtime)

    with TestClient(app) as client:
        response = client.post("/", json=_joule_v03_request("hello joule"))

    assert response.status_code == 200
    payload = response.json()
    assert "error" not in payload
    task = payload["result"]
    assert task["contextId"] == "context-joule"
    assert task["status"]["state"] == "completed"
    assert task["artifacts"][0]["name"] == "agent_result"
    assert task["artifacts"][0]["parts"][0] == {
        "kind": "text",
        "text": "answer: hello joule",
    }
    assert runtime.calls == [("hello joule", "context-joule")]


def test_server_generates_context_when_client_omits_it(tmp_path: Path) -> None:
    """Generate and consistently return a context identifier for a new thread."""

    runtime = FakeRuntime()
    app = _app(tmp_path, runtime)
    request = _joule_v03_request("new conversation")
    del request["params"]["message"]["contextId"]

    with TestClient(app) as client:
        response = client.post("/", json=request)

    task = response.json()["result"]
    assert task["contextId"]
    assert runtime.calls == [("new conversation", task["contextId"])]


def test_runtime_failure_becomes_safe_failed_task(tmp_path: Path) -> None:
    """Return a failed task without exposing the underlying exception text."""

    runtime = FakeRuntime(failure=RuntimeError("secret-token-value"))
    app = _app(tmp_path, runtime)

    with TestClient(app) as client:
        response = client.post("/", json=_joule_v03_request("fail safely"))

    serialized = response.text
    task = response.json()["result"]
    assert task["status"]["state"] == "failed"
    assert "secret-token-value" not in serialized
    assert "Agent request failed" in serialized
    assert runtime.closed is True


def test_a2a_boundary_rejects_non_text_parts(tmp_path: Path) -> None:
    """Reject remote file parts until a use case defines an explicit policy."""

    runtime = FakeRuntime()
    app = _app(tmp_path, runtime)
    request = _v1_request("text plus remote file")
    request["params"]["message"]["parts"].append(
        {"url": "https://files.example.test/report.pdf"}
    )

    with TestClient(app) as client:
        response = client.post(
            "/", json=request, headers={"A2A-Version": "1.0"}
        )

    payload = response.json()
    assert payload["error"]["code"] == -32602
    assert "text parts only" in payload["error"]["message"]
    assert runtime.calls == []


def test_joule_v03_validation_error_keeps_invalid_params_code(tmp_path: Path) -> None:
    """Translate executor validation failures to the legacy JSON-RPC code."""

    runtime = FakeRuntime()
    app = _app(tmp_path, runtime)

    with TestClient(app) as client:
        response = client.post("/", json=_joule_v03_request("   "))

    payload = response.json()
    assert payload["error"]["code"] == -32602
    assert "non-empty text" in payload["error"]["message"]
    assert runtime.calls == []


def test_joule_v03_missing_task_keeps_task_not_found_code(tmp_path: Path) -> None:
    """Translate missing-task failures to the legacy A2A task error code."""

    runtime = FakeRuntime()
    app = _app(tmp_path, runtime)
    request = {
        "jsonrpc": "2.0",
        "id": "request-missing-task",
        "method": "tasks/get",
        "params": {"id": "missing-task"},
    }

    with TestClient(app) as client:
        response = client.post("/", json=request)

    payload = response.json()
    assert payload["error"]["code"] == -32001
    assert "Task not found" in payload["error"]["message"]
