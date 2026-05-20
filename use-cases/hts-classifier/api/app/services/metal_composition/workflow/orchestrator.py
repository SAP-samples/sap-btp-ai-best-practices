"""Workflow orchestrator: MetalCompositionWorkflowRunner and LangGraph wiring."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..config import MetalCompositionSettings, get_settings
from ..hts_catalog import HanaHTSCatalogResolver
from ..timing import finish_timing, summarize_parallel_timings, utc_now_iso
from .composition import _augment_final_payload_for_ui
from .diagrams import analyze_diagrams, analyze_diagrams_for_hts_clues
from .hana_tree_search import omitted_hana_tree_search_output, run_hana_tree_search
from .hts_fact_profile import synthesize_hts_fact_profile
from .llm import LLMClient
from .trade_decision import run_trade_decision
from .token_usage import TokenUsageRecorder
from .types import DiagramPayload, MetalCompositionState

if TYPE_CHECKING:
    from langgraph.graph import StateGraph

logger = logging.getLogger(__name__)


class _FallbackCompiledGraph:
    """Minimal fallback used when the full LangGraph package is unavailable."""

    def __init__(self, runner: "MetalCompositionWorkflowRunner") -> None:
        self.runner = runner

    def invoke(self, state: MetalCompositionState) -> Dict[str, Any]:
        merged_state: Dict[str, Any] = dict(state)
        for node in (
            self.runner._parallel_inputs_node,
            self.runner._hts_fact_profile_node,
            self.runner._legal_evidence_node,
            self.runner._trade_decision_node,
        ):
            merged_state.update(node(merged_state))
        return merged_state


class MetalCompositionWorkflowRunner:
    """Compiled LangGraph runner for the metal composition endpoint."""

    def __init__(
        self,
        settings: Optional[MetalCompositionSettings] = None,
        *,
        hts_catalog_resolver: Optional[HanaHTSCatalogResolver] = None,
        section_232_source_store: Optional[object] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = LLMClient(self.settings)
        self.hts_catalog_resolver = hts_catalog_resolver
        self.section_232_source_store = section_232_source_store
        self.graph = self._build_graph()

    def _get_hts_catalog_resolver(self) -> HanaHTSCatalogResolver:
        if self.hts_catalog_resolver is None:
            self.hts_catalog_resolver = HanaHTSCatalogResolver(settings=self.settings)
        return self.hts_catalog_resolver

    @staticmethod
    def _usage_recorder(state: MetalCompositionState) -> Optional[TokenUsageRecorder]:
        recorder = state.get("token_usage_recorder")
        return recorder if isinstance(recorder, TokenUsageRecorder) else None

    def _diagram_output(self, state: MetalCompositionState) -> Dict[str, Any]:
        diagram_payloads = state.get("diagram_payloads", [])
        composition_mode = str(state.get("composition_mode") or "diagram_manual").strip().lower()
        if composition_mode == "gcc_tracker":
            return {
                "diagram_output": analyze_diagrams_for_hts_clues(
                    diagram_payloads,
                    self.settings,
                    self.llm,
                    gcc_material_profile=dict(state.get("gcc_tracker_composition") or {}),
                    product_code=state.get("product_code"),
                    source_summary=state.get("source_summary"),
                    source_row=state.get("source_row"),
                    usage_recorder=self._usage_recorder(state),
                )
            }
        if not diagram_payloads:
            return {"diagram_output": {"status": "omitted"}}
        return {
            "diagram_output": analyze_diagrams(
                diagram_payloads,
                self.settings,
                self.llm,
                product_code=state.get("product_code"),
                source_summary=state.get("source_summary"),
                source_row=state.get("source_row"),
                usage_recorder=self._usage_recorder(state),
            )
        }

    def _hana_tree_search_output(self, state: MetalCompositionState) -> Dict[str, Any]:
        try:
            return {
                "hana_tree_search_output": run_hana_tree_search(
                    state,
                    self.settings,
                    self.llm,
                    self._get_hts_catalog_resolver(),
                    usage_recorder=self._usage_recorder(state),
                )
            }
        except Exception as exc:  # noqa: BLE001 - HANA recall is additive, not blocking
            logger.warning("HANA tree search degraded: %s", exc)
            started_perf = perf_counter()
            started_at = utc_now_iso()
            payload = omitted_hana_tree_search_output(reason=str(exc))
            payload["status"] = "failed"
            payload["timing"] = finish_timing(
                started_perf,
                started_at,
                status="failed",
                details={"model": self.settings.hana_tree_router_model_name},
            )
            payload["errors"] = [str(exc)]
            return {"hana_tree_search_output": payload}

    def _run_timed_phase(self, phase_name: str, state: MetalCompositionState) -> Dict[str, Any]:
        phase_runner = {
            "diagram": self._diagram_output,
        }[phase_name]
        started_perf = perf_counter()
        started_at = utc_now_iso()
        output = phase_runner(state)
        status = "completed"
        if len(output) == 1:
            payload = next(iter(output.values()))
            if isinstance(payload, dict):
                status = str(payload.get("status", status))
        timing = finish_timing(started_perf, started_at, status=status)
        return {"output": output, "timing": timing}

    def _run_timed_legal_phase(self, phase_name: str, state: MetalCompositionState) -> Dict[str, Any]:
        phase_runner = {
            "hana_tree_search": self._hana_tree_search_output,
        }[phase_name]
        output = phase_runner(state)
        payload = output.get("hana_tree_search_output", {})
        timing = dict(payload.get("timing") or {})
        if not timing:
            timing = finish_timing(perf_counter(), utc_now_iso(), status=str(payload.get("status") or "completed"))
        return {"output": output, "timing": timing}

    def _parallel_inputs_node(self, state: MetalCompositionState) -> Dict[str, Any]:
        started_perf = perf_counter()
        started_at = utc_now_iso()
        phase_outputs: Dict[str, Any] = {}
        phase_timings: Dict[str, Dict[str, Any]] = {}
        phase_names = ("diagram",)

        with ThreadPoolExecutor(max_workers=len(phase_names)) as executor:
            futures = {
                executor.submit(self._run_timed_phase, phase_name, state): phase_name
                for phase_name in phase_names
            }
            for future in as_completed(futures):
                phase_name = futures[future]
                phase_result = future.result()
                phase_outputs.update(phase_result["output"])
                phase_timings[phase_name] = phase_result["timing"]

        parallel_timing = finish_timing(
            started_perf,
            started_at,
            substeps=phase_timings,
        )
        parallel_timing["details"] = summarize_parallel_timings(
            phase_timings,
            wall_clock_duration_ms=float(parallel_timing["duration_ms"]),
        )

        timing = dict(state.get("timing") or {})
        phases = dict(timing.get("phases") or {})
        phases.update(phase_timings)
        phases["parallel_agents"] = parallel_timing
        timing["phases"] = phases
        timing["summary"] = {
            "slowest_parallel_phase": parallel_timing["details"]["bottleneck"],
            "parallel_agent_wall_clock_duration_ms": parallel_timing["details"][
                "parallel_wall_clock_duration_ms"
            ],
            "parallel_agent_time_saved_ms": parallel_timing["details"][
                "estimated_parallel_time_saved_ms"
            ],
        }

        phase_outputs["timing"] = timing

        # Extract final_composition from diagram output (composition is now
        # produced directly by the diagram LLM instead of a separate combine_final step).
        diagram_output = phase_outputs.get("diagram_output", {}) or {}
        composition_mode = str(state.get("composition_mode") or "diagram_manual").strip().lower()
        if composition_mode == "gcc_tracker":
            composition_raw = dict(state.get("gcc_tracker_composition") or {})
        else:
            composition_raw = diagram_output.pop("composition", None) or {}
        if composition_raw.get("top_level_grams"):
            from .validation import validate_and_repair_final_composition
            from .normalize import _normalize_float

            total_weight = _normalize_float(
                (state.get("source_summary") or {}).get("total_weight_gram")
            )
            final = validate_and_repair_final_composition(
                composition_raw,
                total_weight_grams=total_weight,
            )
            final_payload = _augment_final_payload_for_ui(
                final_payload=final.model_dump(),
                state={**state, **phase_outputs},
            )
            phase_outputs["final_composition"] = final_payload
        else:
            phase_outputs["final_composition"] = composition_raw

        return phase_outputs

    def _hts_fact_profile_node(self, state: MetalCompositionState) -> Dict[str, Any]:
        started_perf = perf_counter()
        started_at = utc_now_iso()
        hts_fact_profile, details = synthesize_hts_fact_profile(
            state,
            self.settings,
            self.llm,
            usage_recorder=self._usage_recorder(state),
        )
        phase_timing = finish_timing(
            started_perf,
            started_at,
            details=details,
        )
        timing = dict(state.get("timing") or {})
        phases = dict(timing.get("phases") or {})
        phases["hts_fact_profile"] = phase_timing
        timing["phases"] = phases
        summary = dict(timing.get("summary") or {})
        summary["hts_fact_profile_duration_ms"] = float(phase_timing["duration_ms"])
        timing["summary"] = summary
        return {"hts_fact_profile": hts_fact_profile, "timing": timing}

    def _legal_evidence_node(self, state: MetalCompositionState) -> Dict[str, Any]:
        started_perf = perf_counter()
        started_at = utc_now_iso()
        phase_outputs: Dict[str, Any] = {}
        phase_timings: Dict[str, Dict[str, Any]] = {}
        phase_names = ("hana_tree_search",)

        with ThreadPoolExecutor(max_workers=len(phase_names)) as executor:
            futures = {
                executor.submit(self._run_timed_legal_phase, phase_name, state): phase_name
                for phase_name in phase_names
            }
            for future in as_completed(futures):
                phase_name = futures[future]
                phase_result = future.result()
                phase_outputs.update(phase_result["output"])
                phase_timings[phase_name] = phase_result["timing"]

        legal_timing = finish_timing(
            started_perf,
            started_at,
            substeps=phase_timings,
        )
        legal_timing["details"] = summarize_parallel_timings(
            phase_timings,
            wall_clock_duration_ms=float(legal_timing["duration_ms"]),
        )

        timing = dict(state.get("timing") or {})
        phases = dict(timing.get("phases") or {})
        phases.update(phase_timings)
        phases["legal_evidence"] = legal_timing
        timing["phases"] = phases
        summary = dict(timing.get("summary") or {})
        summary["hana_tree_search_duration_ms"] = float(
            phase_timings.get("hana_tree_search", {}).get("duration_ms", 0.0) or 0.0
        )
        summary["slowest_legal_phase"] = legal_timing["details"]["bottleneck"]
        summary["legal_evidence_wall_clock_duration_ms"] = legal_timing["details"][
            "parallel_wall_clock_duration_ms"
        ]
        timing["summary"] = summary
        phase_outputs["timing"] = timing
        return phase_outputs

    def _trade_decision_node(self, state: MetalCompositionState) -> Dict[str, Any]:
        started_perf = perf_counter()
        started_at = utc_now_iso()
        trade_result, trade_details = run_trade_decision(
            state,
            self.settings,
            self.llm,
            self._get_hts_catalog_resolver(),
            section_232_source_store=self.section_232_source_store,
            usage_recorder=self._usage_recorder(state),
        )
        phase_timing = finish_timing(
            started_perf,
            started_at,
            details=trade_details,
        )
        timing = dict(state.get("timing") or {})
        phases = dict(timing.get("phases") or {})
        phases["trade_decision"] = phase_timing
        timing["phases"] = phases
        summary = dict(timing.get("summary") or {})
        summary["trade_decision_duration_ms"] = float(phase_timing["duration_ms"])
        timing["summary"] = summary
        trade_result["timing"] = timing
        return trade_result

    def _build_graph(self):
        try:
            from langgraph.graph import END, START, StateGraph
        except ModuleNotFoundError:
            return _FallbackCompiledGraph(self)

        graph = StateGraph(MetalCompositionState)
        graph.add_node("parallel_inputs", self._parallel_inputs_node)
        graph.add_node("hts_fact_profile", self._hts_fact_profile_node)
        graph.add_node("legal_evidence", self._legal_evidence_node)
        graph.add_node("trade_decision", self._trade_decision_node)
        graph.add_edge(START, "parallel_inputs")
        graph.add_edge("parallel_inputs", "hts_fact_profile")
        graph.add_edge("hts_fact_profile", "legal_evidence")
        graph.add_edge("legal_evidence", "trade_decision")
        graph.add_edge("trade_decision", END)
        return graph.compile()

    def run(
        self,
        *,
        product_code: str,
        source_row_id: int,
        source_summary: Dict[str, Any],
        source_row: Dict[str, Any],
        composition_mode: str = "diagram_manual",
        document_mode: str = "text_only",
        diagram_payloads: Optional[List[DiagramPayload]] = None,
        gcc_tracker_composition: Optional[Dict[str, Any]] = None,
        include_token_usage: bool = False,
    ) -> Dict[str, Any]:
        started_perf = perf_counter()
        started_at = utc_now_iso()
        token_usage_recorder = TokenUsageRecorder() if include_token_usage else None
        state: MetalCompositionState = {
            "product_code": product_code,
            "source_row_id": int(source_row_id),
            "source_summary": source_summary,
            "source_row": source_row,
            "composition_mode": composition_mode,
            "document_mode": document_mode,
            "diagram_payloads": diagram_payloads or [],
            "gcc_tracker_composition": gcc_tracker_composition,
            "include_token_usage": include_token_usage,
            "token_usage_recorder": token_usage_recorder,
        }
        result = self.graph.invoke(state)
        workflow_timing = finish_timing(started_perf, started_at)
        timing = dict(result.get("timing") or {})
        phases = dict(timing.get("phases") or {})
        phases["workflow"] = workflow_timing
        timing["phases"] = phases

        parallel_details = phases.get("parallel_agents", {}).get("details", {})
        legal_details = phases.get("legal_evidence", {}).get("details", {})
        hts_fact_profile_duration_ms = float(
            phases.get("hts_fact_profile", {}).get("duration_ms", 0.0) or 0.0
        )
        trade_decision_duration_ms = float(
            phases.get("trade_decision", {}).get("duration_ms", 0.0) or 0.0
        )
        timing["summary"] = {
            **dict(timing.get("summary") or {}),
            "workflow_duration_ms": float(workflow_timing["duration_ms"]),
            "critical_path_duration_ms": round(
                float(parallel_details.get("parallel_wall_clock_duration_ms", 0.0) or 0.0)
                + hts_fact_profile_duration_ms
                + float(legal_details.get("parallel_wall_clock_duration_ms", 0.0) or 0.0)
                + trade_decision_duration_ms,
                2,
            ),
        }
        if token_usage_recorder is not None:
            result["token_usage"] = token_usage_recorder.build_summary()
        result["timing"] = timing
        return result
