"""Verify global model settings persistence, HTTP validation and runtime mapping."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agent.config import load_config
from app.workspace.ai_model_settings import AI_MODELS, DEFAULT_AI_MODEL, model_configuration
from app.workspace.dependencies import get_service
from app.workspace.repository import MemoryRepository
from app.workspace.service import WorkspaceService


def test_model_choice_persists_with_revision_and_maps_provider():
    """The supported catalogue remains ordered and stale writes cannot replace a choice."""
    repository = MemoryRepository()
    service = WorkspaceService(repository)
    first = service.get_ai_model_settings()
    assert first["model"] == DEFAULT_AI_MODEL == "gpt-5.4"
    assert [item["name"] for item in first["models"]] == [item[0] for item in AI_MODELS]
    assert first["revision"] == 0
    saved = service.save_ai_model_settings("anthropic--claude-4.6-sonnet", 0)
    assert saved["revision"] == 1
    assert WorkspaceService(repository).get_ai_model_settings()["model"] == saved["model"]
    assert model_configuration(saved["model"]).provider == "claude"
    assert model_configuration("gpt-5.6-luna").provider == "openai"
    with pytest.raises(ValueError, match="stale revision"):
        service.save_ai_model_settings("gpt-5.6-terra", 0)
    with pytest.raises(ValueError, match="Unsupported"):
        service.save_ai_model_settings("unapproved-model", 1)


def test_model_choice_http_contract():
    """HTTP reads and writes expose supported choices and reject invalid names."""
    from app.routers.workspace import router

    service = WorkspaceService(MemoryRepository())
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_service] = lambda: service
    client = TestClient(app)
    assert client.get("/api/ai-model-settings").json()["model"] == "gpt-5.4"
    saved = client.put("/api/ai-model-settings", json={"model": "gpt-5.6-terra", "revision": 0})
    assert saved.status_code == 200
    assert saved.json()["revision"] == 1
    assert client.put("/api/ai-model-settings", json={"model": "gpt-5.4", "revision": 0}).status_code == 409
    assert client.put("/api/ai-model-settings", json={"model": "unknown", "revision": 1}).status_code == 422


def test_packaged_default_is_gpt_54():
    """The unsaved fallback used by direct agent configuration matches the UI."""
    from pathlib import Path

    config = load_config(Path(__file__).parents[1] / "app/agent/config/agent.yaml")
    assert config.model.name == DEFAULT_AI_MODEL
    assert config.model.provider == "openai"


def test_sidebar_runtime_reads_saved_model(monkeypatch):
    """A new chat runtime receives the latest HANA-backed deployment choice."""
    import asyncio
    from app.agent import runtime as runtime_module
    from app.routers.workspace_chat import create_runtime

    service = WorkspaceService(MemoryRepository())
    service.save_ai_model_settings("gpt-5.6-luna", 0)
    captured = {}

    class Runtime:
        """Capture runtime creation without connecting to an AI provider."""

        @classmethod
        async def create(cls, path, extra_tools, model_name):
            """Record the configured deployment and return a test runtime."""
            captured["model"] = model_name
            return cls()

    monkeypatch.setattr(runtime_module, "AgentRuntime", Runtime)
    asyncio.run(create_runtime(service, "test-session", lambda event: None))
    assert captured["model"] == "gpt-5.6-luna"
