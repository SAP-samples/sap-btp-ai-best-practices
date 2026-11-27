"""Input checks shared by SAP settings generation and the readiness report.

Every SAP tab needs some dataset lookups and some plant-profile settings. These
checks say, without generating anything, what is missing and how to fix it.
``generate.py``, ``calendar.py`` and ``recipes.py`` take their "blocked"
messages from here, so the readiness report and a real generation can never
disagree about why a tab is blocked.

Checks that need the selected point's members (an ambiguous SEFI forecast, more
subgroups than recipe characters) stay in their tab module: readiness gets them
from a dry generation instead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

INTEGRATION_DOC = "docs/sap-data-integration-points.md"
PROFILE_SAP_SECTION = "Settings > SAP settings"


@dataclass(frozen=True)
class Gap:
    """One missing input of one SAP tab.

    Attributes:
        tab: Summary tab key (``cycle_stock``, ``planning_calendar``, ...).
        status: ``blocked`` (tab not produced) or ``partial`` (produced with gaps).
        code: ``profile_setting`` (a plant-profile SAP setting is empty),
            ``dataset_view`` (the dataset lacks a lookup table), ``depends_on``
            (another tab must be ready first) or ``not_ported`` (the logic needs
            data that is not extracted today).
        message: What is missing, as shown in the tab status.
        setting: Profile setting to fill, for ``profile_setting`` gaps.
        view: Dataset view that is missing, for ``dataset_view`` gaps.
        fix: What the user (or the assistant) does to close the gap.
    """

    tab: str
    status: Literal["blocked", "partial"]
    code: str
    message: str
    setting: str | None = None
    view: str | None = None
    fix: str = ""

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form for the API and the assistant."""
        return asdict(self)


def cycle_stock_gaps(lookups: dict[str, list | None]) -> list[Gap]:
    """DC network and pallet quantities are required; the ABC classes only complete the tab."""
    gaps = []
    if lookups.get("dc_network") is None or lookups.get("dc_pallet_quantities") is None:
        gaps.append(Gap(
            "cycle_stock", "blocked", "dataset_view",
            f"The dataset has no DC network or pallet lookup; upload the inputs workbook or run backfill-sap-lookups ({INTEGRATION_DOC}).",
            view="dc_network / dc_pallet_quantities",
            fix="Recreate the dataset with the inputs (enrichment) workbook, which holds the DC network and pallet tables."))
    elif lookups.get("dc_abc_classes") is None:
        gaps.append(Gap(
            "cycle_stock", "partial", "dataset_view",
            "no ABC lookup: materials outside the wheel are missing",
            view="dc_abc_classes",
            fix="Use an inputs workbook with the ABC_calculated table."))
    return gaps


def calendar_policy_gap(policy: dict[str, Any]) -> Gap | None:
    """The odd- and even-week lot-size profiles are plant customizing codes with no default."""
    if policy.get("calendar_odd_profile") and policy.get("calendar_even_profile"):
        return None
    missing = [name for name in ("calendar_odd_profile", "calendar_even_profile") if not policy.get(name)]
    return Gap(
        "planning_calendar", "blocked", "profile_setting",
        f"Set the odd- and even-week lot-size profiles in the plant profile ({PROFILE_SAP_SECTION}).",
        setting=", ".join(missing),
        fix="Enter both codes (<profile>-<calendar>) in the plant profile, use the values detected from an "
            "earlier wheel of the plant, or ask the assistant to propose them.")


def recipe_pool_gap(pool: list[str]) -> Gap | None:
    """Recipe combinations need the list of characters allowed in BW."""
    if pool:
        return None
    return Gap(
        "recipe_combination", "blocked", "dataset_view",
        "No recipe characters: upload an inputs workbook with the RecipeCharacters table, "
        f"or set fallback recipe characters in the plant profile ({PROFILE_SAP_SECTION}).",
        setting="recipe_characters", view="recipe_characters",
        fix="Recreate the dataset with an inputs workbook that has the RecipeCharacters table "
            "('List of characters allowed in BW').")


def apo_input_gap(lookups: dict[str, list | None]) -> Gap | None:
    """APO lot sizes need the KG lot size of each production version."""
    if lookups.get("production_version_masses") is not None:
        return None
    return Gap(
        "apo_lot_size", "blocked", "dataset_view",
        "The dataset has no production-version KG lookup.",
        view="production_version_masses",
        fix="Use an inputs workbook with the Production_versions___List table.")


def apo_dependency_gap() -> Gap:
    """APO rows are keyed by recipe combination, so that tab must be ready first."""
    return Gap("apo_lot_size", "blocked", "depends_on", "Needs the recipe combination tab.",
               fix="Fix the recipe combination tab first.")


def central_dc_gap(policy: dict[str, Any]) -> Gap | None:
    """Intermediate-DC settings are not ported (they need extractions the POC does not have)."""
    if policy.get("multi_tier_dc") != "yes":
        return None
    return Gap(
        "central_dc_lot_size", "blocked", "not_ported",
        f"Intermediate-DC settings need sales scenario and producing-plant ABC extractions ({INTEGRATION_DOC}).",
        fix="Production-phase work; see the integration document.")


def input_gaps(lookups: dict[str, list | None], policy: dict[str, Any], pool: list[str]) -> list[Gap]:
    """Every gap that can be known without the selected point's members.

    Args:
        lookups: Dataset lookup views by name; ``None`` marks a missing view.
        policy: Resolved SAP settings (``SapSettingsPolicy.to_resolved()`` shape).
        pool: Recipe characters available (dataset list, else profile fallback).

    Returns:
        Gaps in template tab order.
    """
    gaps = list(cycle_stock_gaps(lookups))
    for gap in (calendar_policy_gap(policy), recipe_pool_gap(pool)):
        if gap:
            gaps.append(gap)
    apo = apo_dependency_gap() if not pool else apo_input_gap(lookups)
    if apo:
        gaps.append(apo)
    central = central_dc_gap(policy)
    if central:
        gaps.append(central)
    return gaps
