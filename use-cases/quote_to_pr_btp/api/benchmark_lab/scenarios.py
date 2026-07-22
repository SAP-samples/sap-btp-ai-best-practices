"""Scenario registry for DocAI and LLM benchmark dimensions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone


AVAILABLE_MODELS = [
    "gpt-5",
    "gpt-5.5",
    "gpt-5-mini",
    "gpt-4.1",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3.1-flash-lite",
    "gemini-2.0-flash",
    "anthropic--claude-4.7-opus",
    "anthropic--claude-4.6-opus",
    "anthropic--claude-4.5-opus",
    "sonar-pro",
    "sonar",
]


@dataclass(frozen=True)
class BenchmarkScenario:
    """A selectable benchmark scenario."""

    key: str
    label: str
    description: str
    family: str
    requires_judge: bool = False
    iterative: bool = False


LLM_SCENARIOS = [
    BenchmarkScenario(
        key="simple_prompt",
        label="Simple prompt",
        description="Ask the model to extract all relevant fields from the file.",
        family="llm",
    ),
    BenchmarkScenario(
        key="detailed_static_prompt",
        label="Detailed static prompt",
        description="Use a stable quote-to-PR schema with explicit field rules.",
        family="llm",
    ),
    BenchmarkScenario(
        key="dynamic_prompt",
        label="Dynamic prompt",
        description="Analyze the file first, then generate a document-specific extraction prompt.",
        family="llm",
        iterative=True,
    ),
    BenchmarkScenario(
        key="dynamic_prompt_judge_loop",
        label="Dynamic prompt + judge loop",
        description="Extract, judge, repair, and repeat until pass or max iterations.",
        family="llm",
        requires_judge=True,
        iterative=True,
    ),
]


DOCAI_SCENARIOS = [
    BenchmarkScenario(
        key="default",
        label="Default",
        description="Run SAP Document AI with the baseline/predefined configuration.",
        family="docai",
    ),
    BenchmarkScenario(
        key="wide_attributes",
        label="Wide attributes",
        description="Use a broad quote/PR schema to maximize field recall.",
        family="docai",
    ),
    BenchmarkScenario(
        key="dynamic_attributes",
        label="Dynamic attributes",
        description="Analyze misses, update schema fields, and re-run DocAI.",
        family="docai",
        iterative=True,
    ),
    BenchmarkScenario(
        key="dynamic_attributes_judge_loop",
        label="Dynamic attributes + judge loop",
        description="Use an LLM judge to propose schema improvements across iterations.",
        family="docai",
        requires_judge=True,
        iterative=True,
    ),
]


def scenario_by_key(key: str) -> BenchmarkScenario:
    """Return a scenario by key."""

    for scenario in [*LLM_SCENARIOS, *DOCAI_SCENARIOS]:
        if scenario.key == key:
            return scenario
    raise KeyError(f"Unknown scenario: {key}")


def slugify(value: str, max_length: int = 48) -> str:
    """Return a lowercase slug safe for run and schema names."""

    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    cleaned = re.sub(r"_+", "_", cleaned)
    return (cleaned or "item")[:max_length].strip("_") or "item"


class SchemaNameFactory:
    """Generate controlled schema names to avoid service clutter."""

    def __init__(self, prefix: str = "WRG_QTE") -> None:
        self.prefix = prefix

    def make(
        self,
        *,
        scenario_key: str,
        iteration: int,
        run_id: str,
        document_hint: str | None = None,
    ) -> str:
        """Build a deterministic, compact schema name."""

        if iteration < 0:
            raise ValueError("iteration must be >= 0")
        run_suffix = slugify(run_id)[-10:]
        scenario = slugify(scenario_key, max_length=24)
        hint = slugify(document_hint or "all", max_length=18)
        name = f"{self.prefix}_{scenario}_{hint}_i{iteration}_{run_suffix}"
        return name[:80].rstrip("_")


def utc_timestamp_for_run() -> str:
    """Return a compact UTC timestamp for run IDs."""

    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
