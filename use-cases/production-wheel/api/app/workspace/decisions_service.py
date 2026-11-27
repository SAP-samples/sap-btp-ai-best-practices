"""Decision-scenario read model and the persisted scenario selection of a run.

The frontier classification itself is pure (``production_wheel.decision_scenarios``).
This mixin adds per-point metrics read from HANA and stores the user's choice
in the run payload, where SAP settings generation picks it up.
"""

from __future__ import annotations

from collections import defaultdict

from production_wheel.decision_scenarios import DEFAULT_NEIGHBOURS, classify_frontier
from production_wheel.extraction.common import as_number

from .datasets_service import now


class DecisionService:
    """Classify a completed frontier and record which scenario people selected."""

    def decision_scenarios(self, run_id, neighbours=DEFAULT_NEIGHBOURS):
        """Return every usable frontier point with its decision label and metrics.

        Args:
            run_id: Completed run whose results are published.
            neighbours: Points labelled on each side of the knee.

        Returns:
            The ``classify_frontier`` payload with these additions:
            - per point: ``group_count``, ``sefi_batches_per_week`` and the
              cycle stock counts ``single_roc_group_count`` /
              ``runner_multiple_group_count`` (``None`` for older runs);
            - ``selection``: the stored selection, if any;
            - ``frontier_points_retained`` and ``frontier_shortfall``: frontier
              density evidence.
            ``sefi_batches_per_week`` is the sum of ``frequency_per_week`` over
            the point's groups (one SEFI batch per group cycle).
        """

        result = self.results(run_id)
        frontier = classify_frontier(result["points"], neighbours)
        solutions = {
            int(as_number(row.get("point_index")) or 0): row
            for row in result["points"]
        }
        batches = defaultdict(float)
        for row in self.repo.rows(run_id, "groups"):
            batches[int(as_number(row.get("point_index")) or 0)] += as_number(row.get("frequency_per_week")) or 0.0
        for point in frontier["points"]:
            point["group_count"] = as_number(solutions.get(point["point_index"], {}).get("group_count"))
            point["sefi_batches_per_week"] = batches.get(point["point_index"])
            for key in ("single_roc_group_count", "runner_multiple_group_count"):
                point[key] = as_number(solutions.get(point["point_index"], {}).get(key))
        metadata = result.get("metadata") or {}
        return {
            **frontier,
            "run_id": run_id,
            "neighbours": neighbours,
            "selection": result["run"].get("decision_selection"),
            "frontier_points_retained": metadata.get("frontier_points_retained"),
            "frontier_shortfall": metadata.get("frontier_shortfall"),
        }

    def select_decision(
        self,
        run_id,
        point_index,
        neighbours=DEFAULT_NEIGHBOURS,
    ):
        """Persist the chosen frontier point for SAP settings generation.

        Any usable point can be chosen, so people can still pick a point beyond
        the 7 key scenarios ("show all").

        Args:
            run_id: Completed run.
            point_index: Chosen frontier point. Its subgroups carry their cycle
                stock strategy (single RoC or runner multiples), which SAP
                settings generation reads.
            neighbours: Neighbour count used to label the scenarios.

        Returns:
            The stored ``decision_selection`` record.

        Raises:
            ValueError: Unusable point.
        """

        scenarios = self.decision_scenarios(run_id, neighbours)
        chosen = next(
            (point for point in scenarios["points"] if point["point_index"] == int(point_index)),
            None,
        )
        if chosen is None:
            raise ValueError(f"point {point_index} is not a usable frontier point")
        selection = {
            "point_index": chosen["point_index"],
            "decision_class": chosen["decision_class"],
            "decision_label": chosen["decision_label"],
            "neighbours": neighbours,
            "selected_at": now(),
        }
        from production_wheel.sap_settings.generate import SAP_RESULT_VIEWS

        with self.repo.transaction():
            # Settings generated for the previous choice must never look current.
            self.repo.replace_tables(run_id, {view: [] for view in SAP_RESULT_VIEWS})
            self.patch_run(run_id, {"decision_selection": selection, "sap_settings": None})
        return selection
