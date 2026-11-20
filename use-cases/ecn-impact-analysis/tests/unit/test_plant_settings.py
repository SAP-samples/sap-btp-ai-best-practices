"""Plant Code Mapping validation, HANA repository and Settings endpoint tests.

Run with::

    PYTHONPATH=api .venv/bin/python -m pytest -q tests/unit/test_plant_settings.py
"""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from app.ecn.errors import SourceValidationError
from app.ecn.plant_settings import DEFAULT_EXCLUDED_RULE_STATES, PlantMapping, PlantSettings, normalize_plant_settings
from app.main import create_app
from app.persistence.plant_settings_repository import PlantSettingsRepository
from fastapi.testclient import TestClient

from tests.fixtures.persistence import RecordingEngine


def _row(suffix, code, name):
    """Build one raw mapping row as sent by the Settings page."""
    return {"suffix": suffix, "plant_code": code, "plant_name": name}


def test_normalize_upper_cases_sorts_and_allows_shared_plants() -> None:
    """Several suffixes may map to one plant; suffixes and codes are canonical upper-case."""
    settings = normalize_plant_settings(
        [_row(" jd ", "pk01", "Plant B"), _row("JB", "PL10", "Plant A"), _row("is", "PL10", "Plant A")],
        ["To be Obsolete", "To be Obsolete"],
    )
    assert [m.suffix for m in settings.mappings] == ["IS", "JB", "JD"]
    assert settings.mappings[2] == PlantMapping("JD", "PK01", "Plant B")
    assert settings.excluded_rule_states == ("To be Obsolete",)
    assert settings.frozen_context()["mappings"][0] == {"suffix": "IS", "plant_code": "PL10", "plant_name": "Plant A"}


@pytest.mark.parametrize(
    ("rows", "code"),
    [
        ([_row("JB", "PL10", "Plant A"), _row("jb", "PK01", "Plant B")], "PLANT_MAPPING_DUPLICATE"),
        ([_row("E_BRAND_JB", "PL10", "Plant A")], "PLANT_MAPPING_INVALID"),
        ([_row("JB", "", "Plant A")], "PLANT_MAPPING_INVALID"),
        ([_row("JB", "PL10", "Plant A"), _row("IS", "PL10", "Other")], "PLANT_NAME_CONFLICT"),
    ],
)
def test_normalize_rejects_ambiguous_rows(rows, code) -> None:
    """Duplicate suffixes, non-suffix values, blanks and conflicting names are rejected."""
    with pytest.raises(SourceValidationError) as error:
        normalize_plant_settings(rows, [])
    assert error.value.code == code


def test_repository_defaults_exclusion_and_lists_seen_states() -> None:
    """A missing setting row means the default exclusion; seen states include saved ones."""
    engine = RecordingEngine()
    engine.queue_mappings([{"brand_suffix": "JB", "plant_code": "PL10", "plant_name": "Plant A"}])
    engine.queue_mappings([])
    engine.queue_mappings([{"LIFECYCLE_STATE": "In Work"}, {"LIFECYCLE_STATE": "App For Series"}])
    settings = PlantSettingsRepository(engine).get()  # type: ignore[arg-type]
    assert settings.mappings == (PlantMapping("JB", "PL10", "Plant A"),)
    assert settings.excluded_rule_states == DEFAULT_EXCLUDED_RULE_STATES
    assert settings.available_rule_states == ("App For Series", "In Work", "To be Obsolete")
    assert "ECN_IMPACT_PLANT_MAPPING" in engine.executions[0][0]


def test_repository_replace_writes_everything_in_one_transaction() -> None:
    """Delete-and-insert of rows and the state setting share one begin() block."""
    engine = RecordingEngine()
    repository = PlantSettingsRepository(engine)  # type: ignore[arg-type]
    repository.replace(PlantSettings((PlantMapping("JB", "PL10", "Plant A"),), ("In Work",)))
    statements = [sql for sql, _ in engine.executions]
    assert engine.begin_count == 1
    assert statements[0].startswith('DELETE FROM "ECN_IMPACT_PLANT_MAPPING"')
    assert engine.executions[1][1]["suffix"] == "JB"
    assert engine.executions[3][1] == {"key": "excluded_rule_states", "value": '["In Work"]', "now": engine.executions[3][1]["now"]}


class _Repository:
    """In-memory stand-in for the HANA repository behind the endpoint."""

    def __init__(self) -> None:
        """Start empty with the default exclusion."""
        self.settings = PlantSettings((), DEFAULT_EXCLUDED_RULE_STATES, DEFAULT_EXCLUDED_RULE_STATES)

    def get(self):
        """Return the stored settings."""
        return self.settings

    def replace(self, settings):
        """Store validated settings and return them."""
        self.settings = settings
        return settings


def _client(monkeypatch):
    """Build a test app whose container only holds the plant settings repository."""
    monkeypatch.setenv("API_KEY", "test-api-key")
    services = SimpleNamespace(plant_settings_repository=_Repository())

    @asynccontextmanager
    async def lifespan(_app):
        """Yield test services."""
        yield services

    return TestClient(create_app(service_lifespan_factory=lifespan))


def test_endpoint_round_trips_and_rejects_invalid_rows(monkeypatch) -> None:
    """PUT validates and stores; GET returns the stored rows; invalid input is a 400."""
    headers = {"X-API-Key": "test-api-key"}
    with _client(monkeypatch) as client:
        saved = client.put("/api/plant-settings", headers=headers, json={
            "mappings": [_row("jb", "PL10", "Plant A")], "excluded_rule_states": ["To be Obsolete"]})
        read = client.get("/api/plant-settings", headers=headers)
        bad = client.put("/api/plant-settings", headers=headers, json={"mappings": [_row("JB", "", "x")]})
    assert saved.status_code == 200
    assert read.json()["mappings"] == [{"suffix": "JB", "plant_code": "PL10", "plant_name": "Plant A"}]
    assert bad.status_code == 400
