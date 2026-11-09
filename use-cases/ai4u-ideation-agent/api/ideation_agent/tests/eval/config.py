"""Evaluation configuration: RunnerConfig dataclass."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class RunnerConfig:
    """All parameters that affect a conversation run."""

    model_provider: str = "openai"
    model_name: str = "gpt-4.1"
    temperature: float = 0.2
    cse_enabled: bool = True
    perplexity_enabled: bool = True
    perplexity_problem_research: bool = True
    linkedin_enabled: bool = True
    repair_enabled: bool = True
    max_questions: int = 9
    min_questions: int = 5           # MIN enforcement (matches production MIN_CLARIFICATION_QUESTIONS)
    two_tier_strategy: bool = True
    max_turns: int = 7
    json_mode: bool = True

    # Provocateur + Reimagine settings
    provocateur_enabled: bool = False
    reimagine_enabled: bool = False
    provocateur_temperature: float = 0.4
    reimagine_temperature: float = 0.7
    reimagine_max_reasoning_tokens: int = 10000
    provocateur_start_at: int = 3       # Provocateur activates after N primary questions
    reimagine_after: int = 5            # Reimagine fires after N total questions

    @property
    def label(self) -> str:
        return f"{self.model_name}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
