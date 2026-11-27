"""Gather everything SAP settings generation reads for one run, in one place.

Generation and the readiness report must see exactly the same inputs, so both
call ``SapInputsMixin.sap_inputs``:

- the run's published results and its dataset (plant, SAP lookup views);
- the SAP settings rules of the run's plant profile, read **live** from the
  profile's current revision (a missing calendar code can be fixed after the
  run, without re-running). When the profile was removed, the copy frozen with
  the run is used; a run without any profile uses the defaults;
- the runner thresholds and cycle-stock policy frozen with the run (they shaped
  the optimization, so they are never read live);
- where each value came from, so the summary, the UI and the assistant can say
  "default", "plant profile", "dataset" or "missing".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from production_wheel.sap_settings.profile_policy import SapSettingsPolicy

# Same pattern as the API accepts for POST /api/runs/{id}/sap-settings.
PERIOD_LABEL_PATTERN = r"^[A-Za-z0-9 ._-]{1,32}$"


def current_quarter_label(today: date | None = None) -> str:
    """Return the planning period label of the current quarter, e.g. ``26Q4``.

    Args:
        today: Date to use; ``None`` means today.

    Returns:
        Two-digit year, ``Q`` and the quarter number.
    """
    today = today or date.today()
    return f"{today.year % 100:02d}Q{(today.month - 1) // 3 + 1}"


@dataclass
class SapInputs:
    """Inputs of SAP settings generation for one run.

    Attributes:
        run: Run record (with ``decision_selection`` when one is stored).
        points: Published ``solutions`` rows of the run.
        dataset: Dataset record of the run.
        plant: Producing plant of the dataset.
        lookups: SAP lookup views by name; ``None`` marks a view the dataset lacks.
        policy_model: SAP settings rules used (profile section).
        policy: Same rules in the resolved shape the generators consume.
        policy_sources: ``{setting: source}`` with ``default``, ``profile``,
            ``frozen profile``, ``dataset:recipe_characters`` or ``missing``.
        profile: ``{profile_id, name, plant, revision, origin, run_revision}``;
            origin is ``live``, ``frozen`` or ``none``.
        pool: Recipe characters available, in assignment order.
        thresholds / thresholds_source: Runner cut-offs frozen with the run.
        cycle_policy / cycle_source: Cycle-stock policy frozen with the run.
        days_per_week: Demand days per productive week.
    """

    run: dict[str, Any]
    points: list[dict[str, Any]]
    dataset: dict[str, Any]
    plant: str
    lookups: dict[str, list | None]
    policy_model: SapSettingsPolicy
    policy: dict[str, Any]
    policy_sources: dict[str, str]
    profile: dict[str, Any]
    pool: list[str]
    thresholds: Any
    thresholds_source: str
    cycle_policy: Any
    cycle_source: str
    days_per_week: float
    notes: list[str] = field(default_factory=list)


class SapInputsMixin:
    """Service mixin: resolve SAP settings inputs of a run (needs ``repo`` and ``results``)."""

    def sap_profile_policy(self, run: dict[str, Any]) -> tuple[SapSettingsPolicy, dict[str, Any]]:
        """Return the SAP settings rules that apply to a run, and where they came from.

        Order: the run's plant profile at its current revision (``live``); the
        profile copy frozen with the run when the profile was removed
        (``frozen``); defaults when the run has no profile (``none``).

        Args:
            run: Run record.

        Returns:
            ``(policy, profile_ref)`` where ``profile_ref`` holds ``profile_id``,
            ``name``, ``plant``, ``revision`` (the one read), ``origin`` and
            ``run_revision`` (the revision frozen with the run).
        """
        from .plant_profiles import upgrade_settings

        profile_id = run.get("plant_profile_id")
        frozen = run.get("plant_profile") or {}
        ref = {"profile_id": profile_id, "name": frozen.get("name"), "plant": frozen.get("plant"),
               "revision": None, "origin": "none", "run_revision": run.get("plant_profile_revision")}
        if profile_id:
            try:
                current = self.repo.get("profiles", profile_id)
            except KeyError:
                current = None
            if current and current.get("status") == "active":
                settings = upgrade_settings(current.get("settings") or {})
                ref.update(name=current.get("name"), plant=current.get("plant"),
                           revision=current.get("revision"), origin="live")
                return SapSettingsPolicy.model_validate(settings["sap_settings"]), ref
        if frozen:
            ref.update(revision=frozen.get("revision"), origin="frozen")
            stored = (frozen.get("settings") or {}).get("sap_settings")
            return (SapSettingsPolicy.model_validate(stored) if stored else SapSettingsPolicy()), ref
        return SapSettingsPolicy(), ref

    def sap_inputs(self, run_id: str) -> SapInputs:
        """Collect every input SAP settings generation reads for a run.

        Args:
            run_id: Run with published results.

        Returns:
            ``SapInputs``.

        Raises:
            ValueError: Results are not published yet.
        """
        from production_wheel.extraction.sap_lookups import LOOKUP_VIEWS
        from production_wheel.runners import CycleStockPolicy, RunnerThresholds
        from production_wheel.sap_settings.recipes import character_pool

        result = self.results(run_id)
        run = result["run"]
        dataset = self.repo.get("datasets", run["dataset_id"])
        plant = dataset["metadata"]["plant"]
        counts = dataset["metadata"].get("counts", {})
        lookups = {
            view: self.repo.rows(run["dataset_id"], view) if view in counts else None
            for view in LOOKUP_VIEWS
        }
        config = (run.get("metadata") or {}).get("config") or {}
        days_per_week = float(config.get("demand_days", 250)) / float(config.get("productive_weeks", 50))
        # Runner classes and cycle-stock rules shaped the optimization, so they are the
        # copies frozen with the run, never the live profile.
        frozen = config.get("runner_thresholds")
        thresholds = RunnerThresholds.model_validate(frozen) if frozen else RunnerThresholds()
        thresholds_source = "run plant profile" if frozen else "defaults (run predates profile runner thresholds)"
        frozen_cycle = config.get("cycle_stock_policy")
        cycle_policy = CycleStockPolicy.model_validate(frozen_cycle) if frozen_cycle else CycleStockPolicy()
        cycle_source = "run plant profile" if frozen_cycle else "defaults (run predates profile cycle stock policy)"

        policy_model, profile = self.sap_profile_policy(run)
        policy = policy_model.to_resolved()
        pool = character_pool(lookups.get("recipe_characters"), policy)
        sources = policy_sources(policy_model, profile["origin"], bool(lookups.get("recipe_characters")))
        notes = []
        if profile["origin"] == "frozen":
            notes.append("The run's plant profile was removed; its SAP settings come from the copy "
                         "frozen with the run and cannot be edited.")
        elif profile["origin"] == "none":
            notes.append("The run has no plant profile; SAP settings use the defaults and the "
                         "plant-specific codes cannot be set for it.")
        return SapInputs(
            run=run, points=result["points"], dataset=dataset, plant=plant, lookups=lookups,
            policy_model=policy_model, policy=policy, policy_sources=sources, profile=profile,
            pool=pool, thresholds=thresholds, thresholds_source=thresholds_source,
            cycle_policy=cycle_policy, cycle_source=cycle_source, days_per_week=days_per_week,
            notes=notes,
        )


def policy_sources(policy: SapSettingsPolicy, origin: str, dataset_has_characters: bool) -> dict[str, str]:
    """Label where each SAP setting value comes from.

    Args:
        policy: Rules in use.
        origin: ``live``, ``frozen`` or ``none`` (see ``sap_profile_policy``).
        dataset_has_characters: The dataset provides a RecipeCharacters list,
            which takes precedence over the profile fallback.

    Returns:
        ``{setting: "default" | "profile" | "frozen profile" | "missing" |
        "dataset:recipe_characters"}``.
    """
    defaults = SapSettingsPolicy().model_dump()
    current = policy.model_dump()
    owner = {"live": "profile", "frozen": "frozen profile"}.get(origin, "default")
    sources = {}
    for name, value in current.items():
        if value is None:
            sources[name] = "missing"
        else:
            sources[name] = "default" if value == defaults[name] else owner
    if dataset_has_characters:
        sources["recipe_characters"] = "dataset:recipe_characters"
    elif current["recipe_characters"] is None:
        sources["recipe_characters"] = "missing"
    return sources
