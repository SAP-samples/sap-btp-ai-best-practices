"""Free-text dataset extraction overrides are translated, then validated deterministically."""
import asyncio

import pytest

from app.agent import providers
from app.workspace.dataset_settings_interpretation import interpret_dataset_settings


class _Service:
    """Minimal service exposing the HANA-saved model choice."""

    def get_ai_model_settings(self):
        return {"model": "gpt-5.4"}


def _stub_model(monkeypatch, output):
    """Replace the chat model so with_structured_output(...).ainvoke returns ``output``."""

    class Model:
        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, _messages):
            return output

    monkeypatch.setattr(providers, "create_chat_model", lambda _: Model())


def test_only_changed_settings_become_overrides(monkeypatch):
    _stub_model(monkeypatch, {"demand_days": 240, "modeled_frequencies": ["1W", "02W", "04W"],
                              "interpretation": "240 demand days; also model 04W."})
    result = asyncio.run(interpret_dataset_settings(_Service(), "240 days, add 04W"))
    assert result["overrides"]["demand_days"] == 240
    assert set(result["overrides"]) == {"demand_days", "modeled_frequencies"}
    assert result["resolved"]["productive_weeks"] == 50


def test_clarification_keeps_defaults(monkeypatch):
    _stub_model(monkeypatch, {"demand_days": 1, "interpretation": "Unclear.", "clarification_required": True})
    result = asyncio.run(interpret_dataset_settings(_Service(), "make it better"))
    assert result["overrides"] == {}


def test_invalid_model_values_are_rejected(monkeypatch):
    _stub_model(monkeypatch, {"canonical_factor": 1.5, "interpretation": "Factor 1.5."})
    with pytest.raises(ValueError):
        asyncio.run(interpret_dataset_settings(_Service(), "factor 1.5"))
