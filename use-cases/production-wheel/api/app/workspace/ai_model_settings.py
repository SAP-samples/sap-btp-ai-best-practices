"""Validated application-wide AI Core model selection stored in HANA."""

from __future__ import annotations

from app.agent.config import ModelSettings


# These are the supported deployment identifiers in the configured AI Core group.
AI_MODELS = (
    ("gpt-5.4", "GPT-5.4", "openai"),
    ("gpt-5.6-luna", "GPT-5.6 Luna", "openai"),
    ("gpt-5.6-terra", "GPT-5.6 Terra", "openai"),
    ("anthropic--claude-4.6-sonnet", "Claude Sonnet 4.6", "claude"),
)
DEFAULT_AI_MODEL = AI_MODELS[0][0]
_MODEL_BY_NAME = {name: (label, provider) for name, label, provider in AI_MODELS}
_SETTING_ID = "global"


def model_configuration(name: str) -> ModelSettings:
    """Map one allowed deployment name to safe provider-specific request settings.

    Args:
        name: One of the supported AI Core deployment identifiers.
    Returns:
        Validated configuration for the agent, formatter and summarizer.
    """
    if name not in _MODEL_BY_NAME:
        raise ValueError("Unsupported AI Core model")
    provider = _MODEL_BY_NAME[name][1]
    return ModelSettings(provider=provider, name=name, temperature=None, reasoning_effort="high")


class AIModelSettingsService:
    """Read and revise the single HANA-backed model choice for all app LLM calls."""

    def get_ai_model_settings(self) -> dict:
        """Return the saved choice or the effective default before first save."""
        try:
            saved = self.repo.get("ai_settings", _SETTING_ID)
        except KeyError:
            saved = {"model": DEFAULT_AI_MODEL, "revision": 0}
        return {
            **saved,
            "models": [
                {"name": name, "label": label, "provider": provider}
                for name, label, provider in AI_MODELS
            ],
        }

    def save_ai_model_settings(self, name: str, revision: int) -> dict:
        """Persist a validated global choice using optimistic revision control.

        Args:
            name: Supported AI Core model deployment identifier.
            revision: Revision returned by the last read.
        Returns:
            Effective selection and updated revision.
        """
        model_configuration(name)
        if revision < 0:
            raise ValueError("revision must be nonnegative")
        with self.repo.transaction():
            try:
                current = self.repo.lock_entity("ai_settings", _SETTING_ID)
            except KeyError:
                if revision != 0:
                    raise ValueError("stale revision; reload current state")
                self.repo.insert("ai_settings", _SETTING_ID, {"model": name, "revision": 1})
            else:
                self.repo.cas("ai_settings", _SETTING_ID, revision, {"model": name})
        return self.get_ai_model_settings()
