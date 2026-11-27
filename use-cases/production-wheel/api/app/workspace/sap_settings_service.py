"""SAP settings service: generation, readiness and the former policy store.

The rules SAP settings generation needs (DC lot sizes, planning calendar
profiles, recipe characters, ...) live in each plant profile's
``settings.sap_settings`` section and are read live at generation time (see
``sap_settings_inputs``). The former global store of policy rows
(``sap_policy_rules`` under one fixed owner) was migrated into the profiles at
startup (``schema_migrations.migrate_sap_settings``); it is kept read-only, so
rows that could not be migrated (plants without a profile) stay visible.
"""

from __future__ import annotations

from typing import Any

from production_wheel.sap_settings.policy import SETTINGS

from .sap_settings_inputs import SapInputs, SapInputsMixin
from .sap_settings_suggestions import SapSuggestionsMixin

POLICY_OWNER = "sap-policy"
POLICY_MOVED = ("SAP settings rules now live in each plant profile (Settings > SAP settings); "
                "ask the assistant or edit the profile instead of policy rows.")


class SapSettingsService(SapSuggestionsMixin, SapInputsMixin):
    """Generate SAP settings for a run's selected scenario and report what is missing."""

    def sap_policy(self):
        """Return the rows left in the former policy store, read-only.

        Returns:
            ``{"rows": [...], "settings": {name: {default, kind, per_class,
            choices, description}}, "deprecated": True, "moved_to": str}``.
            ``rows`` only holds rows the migration could not place in a profile.
        """

        return {
            "rows": self.repo.rows(POLICY_OWNER, "sap_policy_rules"),
            "settings": {
                name: {
                    "default": spec.default,
                    "kind": spec.kind,
                    "per_class": spec.per_class,
                    "choices": sorted(spec.choices),
                    "description": spec.description,
                }
                for name, spec in SETTINGS.items()
            },
            "deprecated": True,
            "moved_to": "plant profile settings.sap_settings",
        }

    def save_sap_policy(self, rows):
        """Refuse writes to the former policy store; the rules belong to the plant profile.

        Raises:
            ValueError: Always, with a pointer to the plant profile.
        """

        raise ValueError(POLICY_MOVED)

    def _dry_generate(self, inputs: SapInputs, point: int, selection: dict[str, Any], period_label: str):
        """Run every SAP tab in memory for one point; nothing is written.

        Returns:
            ``(tables, summary)`` from ``production_wheel.sap_settings.generate``.
        """
        from production_wheel.extraction.common import as_number
        from production_wheel.sap_settings.generate import generate

        point_row = next((row for row in inputs.points if int(as_number(row.get("point_index")) or 0) == point), {})
        return generate(
            self.repo.rows(inputs.run["run_id"], "members", {"point_index": point}),
            inputs.lookups,
            inputs.policy,
            selection,
            inputs.plant,
            period_label,
            inputs.days_per_week,
            as_number(point_row.get("j_ch")),
            runner_thresholds=inputs.thresholds,
            runner_thresholds_source=inputs.thresholds_source,
            cycle_stock_policy=inputs.cycle_policy,
            cycle_stock_policy_source=inputs.cycle_source,
        )

    def generate_sap_settings(self, run_id, period_label):
        """Generate and persist every SAP settings tab for the run's selected scenario.

        Inputs (see ``SapInputsMixin.sap_inputs``):
        - the stored ``decision_selection`` and that point's ``members`` rows;
        - the dataset's SAP lookup views (absent views block their tab);
        - the SAP settings rules of the run's plant profile, current revision;
        - the run's demand days, productive weeks, runner thresholds and
          cycle-stock policy (frozen with the run).

        Outputs are written in one transaction: the ``sap_*`` result views
        (owner = run) and the ``sap_settings`` summary in the run payload. The
        summary records the profile and revision the rules came from
        (``sap_settings_source``, ``profile_revision``) and the source of each
        rule (``policy_sources``).

        Args:
            run_id: Completed run with a decision selection.
            period_label: Planning period written into the APO comment.

        Returns:
            The ``sap_settings`` summary (per-tab status and KPIs).

        Raises:
            ValueError: Results not published, or no scenario selected yet.
        """

        from .datasets_service import now

        inputs = self.sap_inputs(run_id)
        selection = inputs.run.get("decision_selection")
        if not selection:
            raise ValueError("select a decision scenario before generating SAP settings")
        tables, summary = self._dry_generate(inputs, int(selection["point_index"]), selection, period_label)
        summary["generated_at"] = now()
        summary["sap_settings_source"] = inputs.profile
        summary["profile_revision"] = inputs.profile.get("revision")
        summary["policy_sources"] = inputs.policy_sources
        with self.repo.transaction():
            self.repo.replace_tables(run_id, tables)
            self.patch_run(run_id, {"sap_settings": summary})
        return summary

    def sap_settings(self, run_id):
        """Return the stored SAP settings summary of a run, or ``None`` before generation."""

        return self.get_run(run_id).get("sap_settings")

    def sap_settings_readiness(self, run_id, point_index=None):
        """Report, without writing anything, what SAP settings generation would produce.

        Args:
            run_id: Run to check.
            point_index: Point to check; defaults to the stored selection. With
                neither, only the gaps known from the inputs are reported.

        Returns:
            ``{"available": False, "reason": ...}`` before results are published;
            otherwise the payload of ``readiness.readiness_report``.
        """
        from production_wheel.sap_settings.readiness import last_summary_view, readiness_report
        from production_wheel.sap_settings.requirements import input_gaps

        run = self.get_run(run_id)
        if not run.get("results_ready"):
            return {"run_id": run_id, "available": False,
                    "reason": f"results are not available yet (run status: {run.get('status')})"}
        inputs = self.sap_inputs(run_id)
        selection = inputs.run.get("decision_selection")
        if point_index is not None:
            point, point_source = int(point_index), "requested"
        elif selection:
            point, point_source = int(selection["point_index"]), "selection"
        else:
            point, point_source = None, "none"
        dry_tabs = None
        if point is not None:
            known = {int(row.get("point_index")) for row in inputs.points if row.get("point_index") is not None}
            if point not in known:
                raise ValueError(f"point {point} is not a published point of this run")
            dry_selection = selection if selection and int(selection["point_index"]) == point else {"point_index": point}
            _, dry_summary = self._dry_generate(inputs, point, dry_selection, "readiness")
            dry_tabs = dry_summary["tabs"]
        return readiness_report(
            run_id=run_id, point_index=point, point_source=point_source, selection=selection,
            profile=inputs.profile, static_gaps=input_gaps(inputs.lookups, inputs.policy, inputs.pool),
            dry_tabs=dry_tabs, policy_values=inputs.policy_model.model_dump(mode="json"),
            policy_sources=inputs.policy_sources, multi_tier_dc=inputs.policy_model.multi_tier_dc,
            suggestions=self.sap_settings_suggestions_for_run(inputs),
            last_summary=last_summary_view(inputs.run.get("sap_settings"), inputs.profile),
            notes=inputs.notes,
        )
