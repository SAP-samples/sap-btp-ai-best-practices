"""
Pipeline Runner

Runs single-week or multi-week optimizer flows from the API process manager.
"""
from __future__ import annotations

import json
import logging
import re
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable

import pandas as pd

from app.optimizer.io.load_offer_file import load_offer_file
from app.optimizer.io.load_extraction import load_extraction
from app.optimizer.model.canonical import (
    SOURCE_PROFILE_EXTRACTION,
    SOURCE_PROFILE_HYBRID,
    SOURCE_PROFILE_OFFER,
    detect_source_profile,
)
from app.optimizer.model.exposure import (
    build_week_starts,
    reconstruct_base_weekly_exposure,
)
from app.optimizer.model.lifecycle import derive_lifecycle, profile_lifecycle
from app.optimizer.model.lifetime_estimation import (
    LifetimeEstimationConfig,
    estimate_candidate_lifetime_with_rpt1,
)
from app.optimizer.model.limits import (
    limits_to_money_dict,
    load_limits_config,
    resolve_limits,
)
from app.optimizer.opt.explain_multi_week import explain_multi_week_non_selection
from app.optimizer.opt.explain import explain_non_selection
from app.optimizer.opt.optimizer_multi_week import (
    MultiWeekOptimizerSettings,
    optimize_multi_week,
)
from app.optimizer.opt.optimizer_single_week import OptimizerSettings, optimize_single_week
from app.optimizer.report.summary_metrics import (
    compute_multi_week_run_metrics,
    compute_run_metrics,
    render_multi_week_run_summary_markdown,
    render_run_summary_markdown,
)
from app.optimizer.rules.rule_engine import apply_rules, load_rules_config

logger = logging.getLogger(__name__)


def _resolve_cohort_target(cohort_value: str | None) -> tuple[pd.Timestamp, str]:
    if not cohort_value:
        return pd.Timestamp(datetime.utcnow().date()), "date"
    value = cohort_value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return pd.to_datetime(value), "date"
    return pd.to_datetime(value), "exact"


def _infer_source_profile(input_path: Path, sheet_name: str | None) -> str:
    for read_sheet in (sheet_name or "SAPUI5 Export", 0):
        try:
            preview = pd.read_excel(input_path, sheet_name=read_sheet, nrows=0, engine="openpyxl")
            return detect_source_profile(preview.columns)
        except Exception:
            continue
    return SOURCE_PROFILE_EXTRACTION


def _positive_lifecycle_duration_rows(df_with_lifecycle: pd.DataFrame) -> int:
    if df_with_lifecycle.empty or "credit_duration_days" not in df_with_lifecycle.columns:
        return 0
    duration = pd.to_numeric(df_with_lifecycle["credit_duration_days"], errors="coerce")
    return int(((duration.notna()) & (duration > 0)).sum())


def _derive_lifecycle_with_release_fallback(
    source_df: pd.DataFrame,
    *,
    release_event: str,
    stage_label: str,
) -> tuple[pd.DataFrame, str]:
    """Derive lifecycle and auto-fallback release event when no usable history exists."""
    primary = derive_lifecycle(source_df, release_event=release_event)
    primary_count = _positive_lifecycle_duration_rows(primary)
    if primary_count > 0:
        return primary, release_event

    fallback_order = [
        "reconciliation_file_date",
        "reconciliation",
        "paid_on",
        "min_paid_or_repurchase",
    ]

    best_df = primary
    best_event = release_event
    best_count = primary_count
    for alt_event in fallback_order:
        if alt_event == release_event:
            continue
        try:
            alt_df = derive_lifecycle(source_df, release_event=alt_event)
        except Exception:
            continue
        alt_count = _positive_lifecycle_duration_rows(alt_df)
        if alt_count > best_count:
            best_df = alt_df
            best_event = alt_event
            best_count = alt_count

    if best_event != release_event and best_count > 0:
        logger.warning(
            "No positive lifecycle durations for release_event=%s at %s; "
            "falling back to release_event=%s (%d rows with credit_duration_days > 0).",
            release_event,
            stage_label,
            best_event,
            best_count,
        )
    return best_df, best_event


def detect_cohorts(input_path: Path, sheet_name: str = "SAPUI5 Export") -> list[Dict[str, Any]]:
    """Detect available cohort dates from an extraction Excel file.

    Returns a list of dicts with 'date' (string) and 'invoice_count'.
    """
    try:
        df, _ = load_extraction(input_path=input_path, sheet_name=sheet_name)
    except Exception as exc:
        logger.warning("Could not detect cohorts from %s: %s", input_path, exc)
        return []

    col = "Offer File Date (UTC)"
    if col not in df.columns:
        return []

    counts = df[col].dropna().dt.strftime("%Y-%m-%dT%H:%M:%S").value_counts().sort_index()
    cohorts = [{"date": date_str, "invoice_count": int(count)} for date_str, count in counts.items()]
    return cohorts


def _offer_prefilter(candidates_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[Dict[str, Any]]]:
    """Apply minimal deterministic prefilters for offer candidates."""
    if candidates_df.empty:
        return candidates_df.copy(), pd.DataFrame(), []

    working = candidates_df.copy().reset_index(drop=True)
    excluded_frames: list[pd.DataFrame] = []
    summaries: list[Dict[str, Any]] = []

    amount_col = "candidate_amount" if "candidate_amount" in working.columns else (
        "Purchase Price" if "Purchase Price" in working.columns else None
    )
    if amount_col:
        input_rows = len(working)
        amount_series = pd.to_numeric(working[amount_col], errors="coerce")
        keep_mask = (amount_series >= 0).fillna(False)

        excluded_negative = working[~keep_mask].copy().reset_index(drop=True)
        if not excluded_negative.empty:
            excluded_negative["excluded_reason"] = "exclude_negative_purchase_price"
            excluded_negative["excluded_rule_type"] = "numeric_min"
            excluded_negative["excluded_stage"] = "rule"
            excluded_frames.append(excluded_negative)

        working = working[keep_mask].copy().reset_index(drop=True)
        summaries.append(
            {
                "rule_name": "exclude_negative_purchase_price",
                "rule_type": "numeric_min",
                "input_rows": input_rows,
                "excluded_rows": int(len(excluded_negative)),
                "output_rows": int(len(working)),
            }
        )

    input_rows = len(working)
    keep_mask = ~working["invoice_reference"].astype(str).duplicated(keep="first")
    kept = working[keep_mask].copy().reset_index(drop=True)
    excluded_dedup = working[~keep_mask].copy().reset_index(drop=True)
    if not excluded_dedup.empty:
        excluded_dedup["excluded_reason"] = "deduplicate_invoice_reference_offer"
        excluded_dedup["excluded_rule_type"] = "deduplicate_by"
        excluded_dedup["excluded_stage"] = "rule"
        excluded_frames.append(excluded_dedup)

    summaries.append(
        {
            "rule_name": "deduplicate_invoice_reference_offer",
            "rule_type": "deduplicate_by",
            "input_rows": input_rows,
            "excluded_rows": int(len(excluded_dedup)),
            "output_rows": int(len(kept)),
        }
    )

    excluded = (
        pd.concat(excluded_frames, ignore_index=True)
        if excluded_frames
        else pd.DataFrame(columns=[*kept.columns, "excluded_reason", "excluded_rule_type", "excluded_stage"])
    )
    return kept, excluded, summaries


def _build_base_exposure_lookup(weekly_exposure_df: pd.DataFrame) -> Dict[pd.Timestamp, Dict[str, Dict[str, float]]]:
    if weekly_exposure_df.empty:
        return {}
    output: Dict[pd.Timestamp, Dict[str, Dict[str, float]]] = {}
    for _, row in weekly_exposure_df.iterrows():
        week = pd.to_datetime(row["week_start"]).to_period("W-MON").start_time
        entity_type = str(row["entity_type"])
        entity_id = str(row["entity_id"])
        exposure = float(row["exposure_amount"])
        output.setdefault(week, {}).setdefault(entity_type, {})[entity_id] = exposure
    return output


def _inject_static_base_exposure(
    base_exposure_lookup: Dict[pd.Timestamp, Dict[str, Dict[str, float]]],
    week_starts: Iterable[pd.Timestamp],
    limits,
) -> Dict[pd.Timestamp, Dict[str, Dict[str, float]]]:
    """Inject static base exposure from limits.yaml into each planning week."""
    static_maps = (
        ("facility", getattr(limits, "base_exposure_facility", {})),
        ("customer", getattr(limits, "base_exposure_customer", {})),
        ("group", getattr(limits, "base_exposure_group", {})),
    )

    output = dict(base_exposure_lookup)
    for week in week_starts:
        week_key = pd.to_datetime(week).to_period("W-MON").start_time
        week_data = output.setdefault(week_key, {})
        for entity_type, exposure_map_cents in static_maps:
            if not exposure_map_cents:
                continue
            entity_map = week_data.setdefault(entity_type, {})
            for entity_id, exposure_cents in exposure_map_cents.items():
                entity_map[str(entity_id)] = float(entity_map.get(str(entity_id), 0.0)) + (
                    float(exposure_cents) / 100.0
                )
    return output


def _usage_rows(usage: Dict[str, Dict[str, Dict[str, float]]], entity_type: str) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for week, entities in usage.items():
        for entity_id, metrics in entities.items():
            rows.append(
                {
                    "week_start": week,
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    **metrics,
                }
            )
    return rows


def _count_deferred_reasons(explained_df: pd.DataFrame) -> Dict[str, int]:
    if explained_df.empty or "excluded_reason" not in explained_df.columns:
        return {}
    counts = explained_df["excluded_reason"].value_counts(dropna=False)
    return {str(k): int(v) for k, v in counts.items()}


def _disable_single_week_cohort_rule_for_multi_week(
    rules_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Disable exact-cohort equals rule when running multi-week planning."""
    cfg = deepcopy(rules_cfg)
    rules = cfg.get("rules", [])
    if not isinstance(rules, list):
        return cfg

    for rule in rules:
        if not isinstance(rule, dict):
            continue
        by_name = str(rule.get("name", "")).strip().lower() == "cohort_target_offer_file_date"
        by_shape = (
            str(rule.get("type", "")).strip().lower() == "equals"
            and str(rule.get("column", "")).strip() == "Offer File Date (UTC)"
            and str(rule.get("value_from_context", "")).strip() == "cohort_ts"
        )
        if by_name or by_shape:
            rule["enabled"] = False
    return cfg


def _apply_multi_week_candidate_window(
    candidates_df: pd.DataFrame,
    planning_start_date: str | pd.Timestamp,
    horizon_weeks: int,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any] | None]:
    """Keep only Offer-file cohorts within the multi-week planning horizon."""
    if candidates_df.empty:
        return candidates_df.copy(), pd.DataFrame(), None
    if "Offer File Date (UTC)" not in candidates_df.columns:
        return candidates_df.copy(), pd.DataFrame(), None

    horizon = max(1, int(horizon_weeks))
    start_week = pd.to_datetime(planning_start_date).to_period("W-MON").start_time
    end_week = start_week + pd.Timedelta(weeks=horizon - 1)

    working = candidates_df.copy().reset_index(drop=True)
    offer_dt = pd.to_datetime(working["Offer File Date (UTC)"], errors="coerce")
    offer_week = offer_dt.dt.to_period("W-MON").dt.start_time
    keep_mask = offer_week.notna() & (offer_week >= start_week) & (offer_week <= end_week)

    kept = working[keep_mask].copy().reset_index(drop=True)
    excluded = working[~keep_mask].copy().reset_index(drop=True)
    if not excluded.empty:
        excluded["excluded_reason"] = "planning_window_offer_file_date"
        excluded["excluded_rule_type"] = "date_between"
        excluded["excluded_reason_detail"] = (
            f"Offer File Date week must be between {start_week.date().isoformat()} "
            f"and {end_week.date().isoformat()} (inclusive)."
        )

    summary = {
        "rule_name": "planning_window_offer_file_date",
        "rule_type": "date_between",
        "input_rows": int(len(working)),
        "excluded_rows": int(len(excluded)),
        "output_rows": int(len(kept)),
    }
    return kept, excluded, summary


def run_optimizer_pipeline(
    config: Dict[str, Any],
    *,
    progress_callback: Callable[[Dict[str, Any]], None] | None = None,
) -> Dict[str, Any]:
    """Run the full optimizer pipeline with the given config.

    Config keys:
        input_path: str          -- path to input Excel (offer or extraction)
        sheet_name: str          -- Excel sheet name
        cohort: str              -- cohort target value
        release_event: str       -- credit release event mode
        planning_mode: str       -- single_week | multi_week
        source_profile: str      -- offer_file | extraction_file | hybrid
        planning_start_date: str -- multi-week start date
        horizon_weeks: int       -- multi-week horizon
        attempt_cap: int         -- max submissions per invoice
        lifecycle_input_path: str -- extraction file for base exposure reconstruction
        limits_config_path: str  -- path to limits YAML
        rules_config_path: str   -- path to rules YAML
        output_dir: str          -- directory for output artifacts
        solver_max_time_seconds: int
        solver_random_seed: int
        solver_num_search_workers: int
        exclude_payment_block: bool

    Returns:
        Metadata dict (same structure as run_metadata.json).
    """
    pipeline_start = time.time()

    def _emit_progress(
        step_index: int,
        step_code: str,
        step_label: str,
        *,
        details: Dict[str, Any] | None = None,
        phase_status: str = "running",
    ) -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "step_index": int(step_index),
                "step_total": 10,
                "step_code": step_code,
                "step_label": step_label,
                "phase_status": phase_status,
                "details": details or {},
            }
        )

    input_path = Path(config["input_path"])
    planning_mode = str(config.get("planning_mode", "single_week"))
    source_profile = str(config.get("source_profile", "")).strip().lower()
    sheet_name = config.get("sheet_name")
    cohort_value = config.get("cohort")
    release_event = config.get("release_event", "reconciliation_file_date")
    limits_config_path = config["limits_config_path"]
    rules_config_path = config["rules_config_path"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_profile:
        source_profile = _infer_source_profile(input_path, sheet_name)

    if source_profile in (SOURCE_PROFILE_OFFER, SOURCE_PROFILE_HYBRID) and sheet_name == "SAPUI5 Export":
        sheet_name = "Sheet1"
    if source_profile == SOURCE_PROFILE_OFFER and not sheet_name:
        sheet_name = "Sheet1"
    if not sheet_name:
        sheet_name = "SAPUI5 Export"

    cohort_ts, cohort_match_granularity = _resolve_cohort_target(cohort_value)
    normalized_cohort = cohort_value or cohort_ts.date().isoformat()
    planning_start_date = config.get("planning_start_date") or normalized_cohort
    horizon_weeks = int(config.get("horizon_weeks", 8))

    logger.info("[1/10] Loading candidate data from %s (source_profile=%s)", input_path, source_profile)
    _emit_progress(1, "load_candidates", "Loading candidate data", details={"source_profile": source_profile})

    if source_profile in (SOURCE_PROFILE_OFFER, SOURCE_PROFILE_HYBRID):
        candidates_raw_df, load_report = load_offer_file(input_path=input_path, sheet_name=sheet_name)
        candidates_df, rule_excluded_df, rule_summaries = _offer_prefilter(candidates_raw_df)
        _emit_progress(
            2,
            "apply_rules",
            "Applying rule engine",
            details={"skipped": True, "source_profile": source_profile},
        )
    else:
        extraction_df, load_report = load_extraction(input_path=input_path, sheet_name=sheet_name)
        logger.info("[2/10] Applying rule engine")
        _emit_progress(2, "apply_rules", "Applying rule engine")
        rules_cfg = load_rules_config(rules_config_path)
        if planning_mode == "multi_week":
            rules_cfg = _disable_single_week_cohort_rule_for_multi_week(rules_cfg)
        rule_context = {
            "cohort_ts": cohort_ts,
            "cohort_match_granularity": cohort_match_granularity,
            "exclude_payment_block": bool(config.get("exclude_payment_block", False)),
        }
        rule_result = apply_rules(extraction_df, rules_config=rules_cfg, context=rule_context)
        candidates_df = rule_result.eligible_candidates_df
        rule_excluded_df = rule_result.excluded_df
        rule_summaries = [s.to_dict() for s in rule_result.summaries]

        if planning_mode == "multi_week":
            candidates_df, window_excluded_df, window_summary = _apply_multi_week_candidate_window(
                candidates_df,
                planning_start_date=planning_start_date,
                horizon_weeks=horizon_weeks,
            )
            if not window_excluded_df.empty:
                rule_excluded_df = pd.concat(
                    [rule_excluded_df, window_excluded_df],
                    ignore_index=True,
                )
            if window_summary:
                rule_summaries.append(window_summary)

    logger.info(
        "Candidates loaded: %d, rule excluded: %d",
        len(candidates_df),
        len(rule_excluded_df),
    )

    logger.info("[3/10] Deriving lifecycle profiles")
    _emit_progress(3, "derive_lifecycle", "Deriving lifecycle profiles")
    lifecycle_df, lifecycle_profile_release_event = _derive_lifecycle_with_release_fallback(
        candidates_df,
        release_event=release_event,
        stage_label="candidate_profiling",
    )
    lifecycle_profile = profile_lifecycle(lifecycle_df)

    logger.info("[4/10] Resolving credit limits")
    _emit_progress(4, "resolve_limits", "Resolving credit limits")
    limits_cfg = load_limits_config(limits_config_path)
    limits = resolve_limits(candidates_df, limits_cfg)

    logger.info("[5/10] Preparing optimizer run (%s)", planning_mode)
    _emit_progress(5, "prepare_optimizer_run", "Preparing optimizer run", details={"planning_mode": planning_mode})
    weekly_plan_rows: list[Dict[str, Any]] = []
    weekly_exposure_rows: list[Dict[str, Any]] = []
    deferred_reasons: Dict[str, int] = {}
    lifetime_estimation: Dict[str, Any] = {}
    weekly_plan_df = pd.DataFrame()

    if planning_mode == "multi_week":
        attempt_cap = int(config.get("attempt_cap", 1))

        rpt1_history_path = Path(
            config.get("lifecycle_input_path")
            or config.get("reference_extraction_path")
            or input_path
        )
        rpt1_history_sheet_name = config.get("lifecycle_sheet_name", "SAPUI5 Export")

        logger.info("[6/10] Loading RPT-1 history from %s and reconstructing base exposure from extraction", rpt1_history_path)
        _emit_progress(6, "reconstruct_base_exposure", "Reconstructing base exposure")
        rpt1_history_release_event = release_event
        if rpt1_history_path.exists():
            try:
                rpt1_history_df, _ = load_extraction(
                    input_path=rpt1_history_path,
                    sheet_name=rpt1_history_sheet_name,
                )
                rpt1_history_df, rpt1_history_release_event = _derive_lifecycle_with_release_fallback(
                    rpt1_history_df,
                    release_event=release_event,
                    stage_label="rpt1_history",
                )
            except Exception as exc:
                logger.warning("Could not load RPT-1 history %s: %s", rpt1_history_path, exc)
                rpt1_history_df = pd.DataFrame()
        else:
            rpt1_history_df = pd.DataFrame()

        logger.info("[7/10] Estimating invoice lifetime with RPT-1")
        _emit_progress(7, "estimate_lifetime_rpt1", "Estimating invoice lifetime with RPT-1")
        lifetime_cfg = LifetimeEstimationConfig(
            enabled=bool(config.get("enable_rpt1_lifetime_estimation", True)),
            context_min_rows=int(config.get("lifetime_context_min_rows", 500)),
            context_max_rows=int(config.get("lifetime_context_max_rows", 800)),
            query_batch_size=int(config.get("lifetime_query_batch_size", 25)),
            grouping_columns=tuple(
                config.get(
                    "lifetime_grouping_columns",
                    ("COMPANY_CODE", "CUSTOMER_ID", "PROGRAM_ID", "FUNDING_CURRENCY", "ORIGINAL_CURRENCY"),
                )
            ),
            default_lifetime_weeks=int(config.get("default_lifetime_weeks", 4)),
            timeout_seconds=int(config.get("rpt1_timeout_seconds", 90)),
            max_retries=int(config.get("rpt1_max_retries", 3)),
            retry_backoff_seconds=float(config.get("rpt1_retry_backoff_seconds", 1.0)),
            env_path=config.get("rpt1_env_path"),
            max_parallel_calls=int(config.get("lifetime_max_parallel_calls", 2)),
        )
        candidates_df, lifetime_estimation = estimate_candidate_lifetime_with_rpt1(
            candidates_df,
            rpt1_history_df,
            config=lifetime_cfg,
            progress_callback=lambda details: _emit_progress(
                7,
                "estimate_lifetime_rpt1",
                "Estimating invoice lifetime with RPT-1",
                details=details,
            ),
        )
        if lifetime_estimation:
            lifetime_estimation["configured_release_event"] = release_event
            lifetime_estimation["history_release_event"] = rpt1_history_release_event
        logger.info(
            "RPT-1 lifetime estimation status=%s, predicted=%s, fallback=%s, "
            "history_rows=%s, api_calls=%s, history_release_event=%s",
            lifetime_estimation.get("status"),
            lifetime_estimation.get("predicted_candidates"),
            lifetime_estimation.get("fallback_candidates"),
            lifetime_estimation.get("historical_rows_available"),
            lifetime_estimation.get("api_calls"),
            lifetime_estimation.get("history_release_event", rpt1_history_release_event),
        )

        week_starts = build_week_starts(planning_start_date, horizon_weeks)
        exposure_reconstruction = reconstruct_base_weekly_exposure(
            pd.DataFrame(),
            week_starts=week_starts,
            customer_to_group=limits.customer_to_group,
            amount_column="Purchase Price",
        )
        base_exposure_lookup = _build_base_exposure_lookup(exposure_reconstruction.weekly_exposure_df)
        base_exposure_lookup = _inject_static_base_exposure(
            base_exposure_lookup=base_exposure_lookup,
            week_starts=week_starts,
            limits=limits,
        )

        logger.info("[8/10] Running multi-week solver")
        _emit_progress(8, "run_multi_week_solver", "Running multi-week solver")
        multi_settings = MultiWeekOptimizerSettings(
            max_time_seconds=int(config.get("solver_max_time_seconds", 60)),
            random_seed=int(config.get("solver_random_seed", 0)),
            num_search_workers=int(config.get("solver_num_search_workers", 1)),
            horizon_weeks=horizon_weeks,
            attempt_cap=attempt_cap,
            default_lifetime_weeks=int(config.get("default_lifetime_weeks", 4)),
        )
        logger.info(
            "Entering multi-week solver with %d candidates, %d weeks, attempt_cap=%d, "
            "max_time_seconds=%d, num_search_workers=%d",
            len(candidates_df),
            len(week_starts),
            attempt_cap,
            multi_settings.max_time_seconds,
            multi_settings.num_search_workers,
        )
        optimization_result = optimize_multi_week(
            candidates_df,
            limits=limits,
            week_starts=week_starts,
            base_weekly_exposure=base_exposure_lookup,
            settings=multi_settings,
        )
        logger.info(
            "Multi-week solver status: %s, selected: %d, not-selected: %d",
            optimization_result.status,
            len(optimization_result.selected_df),
            len(optimization_result.not_selected_df),
        )

        explained_not_selected = explain_multi_week_non_selection(
            candidates_df=candidates_df,
            optimization_result=optimization_result,
            limits=limits,
        ).explained_not_selected_df

        deferred_reasons = _count_deferred_reasons(explained_not_selected)
        weekly_plan_df = optimization_result.weekly_plan_df.copy()
        if not weekly_plan_df.empty:
            weekly_plan_rows = weekly_plan_df.to_dict(orient="records")

        weekly_exposure_rows = (
            _usage_rows(optimization_result.facility_weekly_usage, "facility")
            + _usage_rows(optimization_result.customer_weekly_usage, "customer")
            + _usage_rows(optimization_result.group_weekly_usage, "group")
        )

    else:
        _emit_progress(
            6,
            "reconstruct_base_exposure",
            "Reconstructing base exposure",
            details={"skipped": True, "planning_mode": "single_week"},
        )
        _emit_progress(
            7,
            "estimate_lifetime_rpt1",
            "Estimating invoice lifetime with RPT-1",
            details={"skipped": True, "planning_mode": "single_week"},
        )
        _emit_progress(8, "run_single_week_solver", "Running single-week solver")
        settings = OptimizerSettings(
            max_time_seconds=int(config.get("solver_max_time_seconds", 60)),
            random_seed=int(config.get("solver_random_seed", 0)),
            num_search_workers=int(config.get("solver_num_search_workers", 1)),
        )
        logger.info(
            "Entering single-week solver with %d candidates, max_time_seconds=%d, num_search_workers=%d",
            len(candidates_df),
            settings.max_time_seconds,
            settings.num_search_workers,
        )
        optimization_result = optimize_single_week(
            candidates_df,
            limits=limits,
            settings=settings,
        )
        logger.info(
            "Single-week solver status: %s, selected: %d, not-selected: %d",
            optimization_result.status,
            len(optimization_result.selected_df),
            len(optimization_result.not_selected_df),
        )
        explained_not_selected = explain_non_selection(
            selected_df=optimization_result.selected_df,
            not_selected_df=optimization_result.not_selected_df,
            limits=limits,
        ).explained_not_selected_df

    logger.info("[9/10] Writing output artifacts")
    _emit_progress(9, "write_output_artifacts", "Writing output artifacts")
    selected_path = output_dir / "selected.xlsx"
    excluded_path = output_dir / "excluded.xlsx"
    summary_path = output_dir / "run_summary.md"
    metadata_path = output_dir / "run_metadata.json"
    weekly_plan_path = output_dir / "weekly_plan.xlsx"
    weekly_exposure_path = output_dir / "weekly_exposure.xlsx"

    optimization_excluded = explained_not_selected.copy()
    if not optimization_excluded.empty:
        optimization_excluded["excluded_stage"] = "optimizer"

    rule_excluded = rule_excluded_df.copy()
    if not rule_excluded.empty:
        rule_excluded["excluded_stage"] = "rule"

    excluded_combined = pd.concat(
        [rule_excluded, optimization_excluded],
        ignore_index=True,
    )

    selected_for_output = (
        weekly_plan_df
        if planning_mode == "multi_week" and not weekly_plan_df.empty
        else optimization_result.selected_df
    )

    with pd.ExcelWriter(selected_path, engine="openpyxl") as writer:
        selected_for_output.to_excel(writer, sheet_name="selected", index=False)

    with pd.ExcelWriter(excluded_path, engine="openpyxl") as writer:
        excluded_combined.to_excel(writer, sheet_name="excluded", index=False)

    if planning_mode == "multi_week":
        with pd.ExcelWriter(weekly_plan_path, engine="openpyxl") as writer:
            weekly_plan_df.to_excel(writer, sheet_name="weekly_plan", index=False)
        pd.DataFrame(weekly_exposure_rows).to_excel(weekly_exposure_path, index=False)

        metrics = compute_multi_week_run_metrics(
            candidates_df=candidates_df,
            optimization_result=optimization_result,
            rule_excluded_df=rule_excluded_df,
            explained_not_selected_df=optimization_excluded,
            lifecycle_profile=lifecycle_profile,
        )
        summary_md = render_multi_week_run_summary_markdown(normalized_cohort, metrics)
    else:
        metrics = compute_run_metrics(
            candidates_df=candidates_df,
            selected_df=optimization_result.selected_df,
            rule_excluded_df=rule_excluded_df,
            optimization_result=optimization_result,
            lifecycle_profile=lifecycle_profile,
        )
        summary_md = render_run_summary_markdown(normalized_cohort, metrics)

    summary_path.write_text(summary_md, encoding="utf-8")

    metadata = {
        "cohort": normalized_cohort,
        "cohort_match_granularity": cohort_match_granularity,
        "planning_mode": planning_mode,
        "source_profile": source_profile,
        "planning_start_date": str(config.get("planning_start_date") or normalized_cohort),
        "horizon_weeks": int(config.get("horizon_weeks", 8)),
        "attempt_cap": int(config.get("attempt_cap", 1)),
        "input_path": str(input_path),
        "sheet_name": sheet_name,
        "release_event": release_event,
        "lifecycle_profile_release_event": lifecycle_profile_release_event,
        "output_dir": str(output_dir),
        "load_report": load_report.to_dict(),
        "rule_summaries": rule_summaries,
        "limits": limits_to_money_dict(limits),
        "metrics": metrics,
        "weekly_plan": weekly_plan_rows,
        "weekly_exposure": weekly_exposure_rows,
        "deferred_reasons": deferred_reasons,
        "lifetime_estimation": lifetime_estimation,
        "selected_output": str(selected_path),
        "excluded_output": str(excluded_path),
        "summary_output": str(summary_path),
        "weekly_plan_output": str(weekly_plan_path) if planning_mode == "multi_week" else None,
        "weekly_exposure_output": str(weekly_exposure_path) if planning_mode == "multi_week" else None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    # Generate the full rich report (markdown + PDF + DOCX).  The basic
    # run_summary.md written above serves as an immediate fallback; the full
    # report overwrites it with a much richer version.
    try:
        from app.optimizer.report.report_builder import generate_full_report
        generate_full_report(metadata, output_dir)
    except Exception as exc:
        logger.warning("Full report generation failed (basic summary preserved): %s", exc)

    total_elapsed = time.time() - pipeline_start
    logger.info("[10/10] Pipeline complete in %.1fs", total_elapsed)
    _emit_progress(
        10,
        "pipeline_complete",
        "Pipeline complete",
        details={"total_elapsed_seconds": round(total_elapsed, 2)},
        phase_status="completed",
    )

    return metadata
