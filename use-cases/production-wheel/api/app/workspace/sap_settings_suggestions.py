"""Data-derived SAP settings values a plant profile can adopt (never applied silently).

Today one value can be learnt from the plant's own files: the odd- and even-week
lot-size profiles of the planning calendar, from an earlier wheel's calendar
table (see ``production_wheel.sap_settings.calendar_profiles``). Suggestions are
shown in the Settings form ("Detect from dataset"), in the SAP settings card and
by the assistant; saving them into the profile is always an explicit step (Save
profile, or Approve on the assistant's card).

Datasets searched, in order: the one given, the run's dataset, then the other
datasets of the same plant, newest first. Only datasets that stored the
``planning_calendar_inform`` view are read.
"""

from __future__ import annotations

from typing import Any

from production_wheel.sap_settings.calendar_profiles import suggest_calendar_profiles
from production_wheel.sap_settings.profile_policy import SapSettingsPolicy

CALENDAR_VIEW = "planning_calendar_inform"


class SapSuggestionsMixin:
    """Service mixin: SAP settings suggestions for a plant profile (needs ``repo``)."""

    def _calendar_candidates(self, plant: str, first: list[str | None]) -> list[dict[str, Any]]:
        """Datasets to search for an earlier planning calendar, best first.

        Args:
            plant: Profile plant; datasets of other plants are ignored.
            first: Dataset ids to try before the others (``None`` entries skipped).

        Returns:
            Dataset records of that plant that stored the calendar view.
        """
        records = [row for row in self.repo.list("datasets") if not row.get("removed_at")]
        by_id = {row["dataset_id"]: row for row in records}
        ordered = [by_id[i] for i in dict.fromkeys(i for i in first if i) if i in by_id]
        rest = sorted((row for row in records if row["dataset_id"] not in {r["dataset_id"] for r in ordered}),
                      key=lambda row: row.get("created_at") or "", reverse=True)
        return [row for row in ordered + rest
                if (row.get("metadata") or {}).get("plant") == plant
                and ((row.get("metadata") or {}).get("counts") or {}).get(CALENDAR_VIEW)]

    def sap_settings_suggestions(self, profile_id: str, dataset_id: str | None = None,
                                 run_dataset_id: str | None = None) -> dict[str, Any]:
        """Suggest SAP settings values for a plant profile from the plant's datasets.

        Args:
            profile_id: Active plant profile.
            dataset_id: Dataset to search first (e.g. the one shown on screen).
            run_dataset_id: Dataset of the run being checked, searched next.

        Returns:
            ``{"profile_id", "plant", "suggestions": [...], "searched": [...]}``.
            Each suggestion holds ``setting`` (``calendar_profiles``), ``status``
            (``suggested`` / ``ambiguous``), ``values`` and ``patch`` (a
            ``ProfileChange`` settings patch ready for Save or an approval card),
            ``current`` values, ``differs`` (adopting it changes the profile),
            ``source``, ``dataset_name``, ``message`` and ``evidence``.
            ``searched`` lists the datasets read with their outcome.
        """
        profile = self.get_plant_profile(profile_id)
        sap = SapSettingsPolicy.model_validate(profile["settings"]["sap_settings"])
        current = {"calendar_odd_profile": sap.calendar_odd_profile, "calendar_even_profile": sap.calendar_even_profile}
        searched, fallback = [], None
        for record in self._calendar_candidates(profile["plant"], [dataset_id, run_dataset_id]):
            found = suggest_calendar_profiles(
                self.repo.rows(record["dataset_id"], CALENDAR_VIEW),
                self.repo.rows(record["dataset_id"], "fini_master"),
                sap.calendar_lot_size,
            )
            searched.append({"dataset_id": record["dataset_id"], "name": record.get("name"), "status": found["status"]})
            if found["status"] == "absent":
                continue
            values = {"calendar_odd_profile": found["odd"], "calendar_even_profile": found["even"]}
            suggestion = {
                "setting": "calendar_profiles",
                "status": found["status"],
                "values": values if found["status"] == "suggested" else None,
                "patch": {"sap_settings": values} if found["status"] == "suggested" else None,
                "current": current,
                "differs": found["status"] == "suggested" and values != current,
                "source": f"dataset:{record['dataset_id']}:{CALENDAR_VIEW}",
                "dataset_id": record["dataset_id"],
                "dataset_name": record.get("name"),
                "message": found["message"],
                "evidence": found["evidence"],
            }
            if found["status"] == "suggested":
                return {"profile_id": profile_id, "plant": profile["plant"], "suggestions": [suggestion], "searched": searched}
            fallback = fallback or suggestion
        return {"profile_id": profile_id, "plant": profile["plant"],
                "suggestions": [fallback] if fallback else [], "searched": searched}

    def sap_settings_suggestions_for_run(self, inputs) -> list[dict[str, Any]]:
        """Suggestions for a run's live profile while its calendar codes are missing.

        Args:
            inputs: ``SapInputs`` of the run.

        Returns:
            The profile's suggestions, or ``[]`` when the profile is not live
            (removed or absent) or both calendar codes are already set.
        """
        if inputs.profile.get("origin") != "live":
            return []
        if inputs.policy_model.calendar_odd_profile and inputs.policy_model.calendar_even_profile:
            return []
        return self.sap_settings_suggestions(inputs.profile["profile_id"],
                                             run_dataset_id=inputs.run.get("dataset_id"))["suggestions"]
