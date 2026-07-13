"""
Process Manager

Orchestrates the optimizer process lifecycle: file management, config
persistence, background execution, and result retrieval.
"""
from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
import zipfile
from io import BytesIO
from datetime import datetime
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from ...optimizer.model.canonical import (
    SOURCE_PROFILE_EXTRACTION,
    detect_source_profile,
)
from ...optimizer.report.report_context import (
    _build_exclusion_summary,
    _build_top_customers,
    _find_binding_constraints,
)
from .artifact_store import OptimizerArtifactStore
from .pipeline_runner import detect_cohorts, run_optimizer_pipeline
from .process_store import ProcessStore

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parents[2]  # api/app/
DATA_DIR = APP_DIR / "data" / "optimizer_runs"
DEFAULT_LIMITS_CONFIG = APP_DIR / "optimizer" / "config" / "limits_synthetic.yaml"
DEFAULT_RULES_CONFIG = APP_DIR / "optimizer" / "rules" / "rules_config.yaml"
DEFAULT_LIFECYCLE_HISTORY_INPUT = (
    APP_DIR / "optimizer" / "reference_data" / "reconciliation_history.xlsx"
)
SUMMARY_FALLBACK_SHEETS = (
    "Funded Invoices",
    "Funded invoices",
)
TEMP_MATERIALIZED_ROOT = Path(gettempdir()) / "optimizer-artifacts"
SUMMARY_ARTIFACT_VERSION = "v1"
SUMMARY_ARTIFACT_KIND = "optimizer/summary+json"
SUMMARY_ARTIFACT_KEYS = {
    "overview": f"optimizer_summary.overview.{SUMMARY_ARTIFACT_VERSION}.json",
    "exclusions": f"optimizer_summary.exclusions.{SUMMARY_ARTIFACT_VERSION}.json",
    "utilization": f"optimizer_summary.utilization.{SUMMARY_ARTIFACT_VERSION}.json",
    "schedule": f"optimizer_summary.schedule.{SUMMARY_ARTIFACT_VERSION}.json",
}
SUMMARY_ENTITY_TYPES = ("facility", "customer", "group")


class ProcessManager:
    """Orchestrates optimizer process lifecycle."""

    _CHAT_CONTEXT_CACHE: Dict[str, Dict[str, Any]] = {}
    _PROGRESS_LOG_STATE: Dict[str, Tuple[str, str, float]] = {}
    _PROGRESS_LOG_HEARTBEAT_SECONDS = 30.0

    @staticmethod
    def _default_lifecycle_history_path() -> Optional[Path]:
        if DEFAULT_LIFECYCLE_HISTORY_INPUT.exists():
            return DEFAULT_LIFECYCLE_HISTORY_INPUT
        return None

    def __init__(
        self,
        store: Optional[ProcessStore] = None,
        artifact_store: Optional[OptimizerArtifactStore] = None,
    ):
        self.store = store or ProcessStore()
        self.artifact_store = artifact_store or OptimizerArtifactStore(
            db_path=getattr(self.store, "db_path", None),
        )
        self._summary_section_cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    @staticmethod
    def _progress_payload(
        *,
        step_code: str,
        step_label: str,
        step_index: int,
        step_total: int = 10,
        phase_status: str = "running",
        started_at: str | None = None,
        details: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        started_dt = None
        if started_at:
            try:
                started_dt = datetime.fromisoformat(started_at)
            except Exception:
                started_dt = None
        elapsed_seconds = 0.0
        if started_dt is not None:
            elapsed_seconds = max(0.0, (datetime.now() - started_dt).total_seconds())
        now_iso = datetime.now().isoformat()
        return {
            "step_code": step_code,
            "step_label": step_label,
            "step_index": int(step_index),
            "step_total": int(step_total),
            "phase_status": phase_status,
            "elapsed_seconds": elapsed_seconds,
            "updated_at": now_iso,
            "details": details or {},
        }

    def _update_progress(
        self,
        process_id: str,
        *,
        step_code: str,
        step_label: str,
        step_index: int,
        step_total: int = 10,
        phase_status: str = "running",
        started_at: str | None = None,
        details: Dict[str, Any] | None = None,
    ) -> None:
        payload = self._progress_payload(
            step_code=step_code,
            step_label=step_label,
            step_index=step_index,
            step_total=step_total,
            phase_status=phase_status,
            started_at=started_at,
            details=details,
        )
        self.store.update_process(
            process_id,
            progress_json=json.dumps(payload, default=str),
            progress_updated_at=payload["updated_at"],
        )
        self._log_progress(process_id, payload)

    @classmethod
    def _log_progress(cls, process_id: str, payload: Dict[str, Any]) -> None:
        step_code = str(payload.get("step_code", ""))
        phase_status = str(payload.get("phase_status", ""))
        details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
        details_json = json.dumps(details, sort_keys=True, default=str)
        current_key = (step_code, phase_status, details_json)
        last_entry = cls._PROGRESS_LOG_STATE.get(process_id)
        now = time.monotonic()

        if last_entry is not None:
            last_step_code, last_details_json, last_logged_at = last_entry
            same_progress = (
                last_step_code == step_code
                and last_details_json == details_json
                and phase_status == "running"
            )
            if same_progress and (now - last_logged_at) < cls._PROGRESS_LOG_HEARTBEAT_SECONDS:
                return

        cls._PROGRESS_LOG_STATE[process_id] = (step_code, details_json, now)
        logger.info(
            "Process %s progress %s/%s [%s] %s status=%s elapsed=%.1fs details=%s",
            process_id,
            payload.get("step_index"),
            payload.get("step_total"),
            step_code,
            payload.get("step_label"),
            phase_status,
            float(payload.get("elapsed_seconds", 0.0) or 0.0),
            details_json,
        )

    @staticmethod
    def _safe_excel_read(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_excel(path, engine="openpyxl")
        except Exception as exc:
            logger.warning("Failed reading %s: %s", path, exc)
            return pd.DataFrame()

    @staticmethod
    def _stable_sort(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        available = [c for c in columns if c in df.columns]
        if not available or df.empty:
            return df
        return df.sort_values(available, kind="stable")

    @staticmethod
    def _pagination(rows: List[Dict[str, Any]], total: int, limit: int, offset: int) -> Dict[str, Any]:
        return {
            "rows": rows,
            "total": int(total),
            "limit": int(limit),
            "offset": int(offset),
        }

    @staticmethod
    def _extract_invoice_ref(row: Dict[str, Any]) -> str:
        for key in ("Invoice Reference", "invoice_reference", "invoice_ref"):
            value = row.get(key)
            if value is not None and not pd.isna(value):
                return str(value)
        return ""

    @staticmethod
    def _extract_company_code(row: Dict[str, Any]) -> str:
        for key in ("Company Code", "company_code", "seller_id_external"):
            value = row.get(key)
            if value is not None and not pd.isna(value):
                return str(value)
        return ""

    @staticmethod
    def _extract_customer(row: Dict[str, Any]) -> str:
        for key in ("Customer", "customer", "debtor_id"):
            value = row.get(key)
            if value is not None and not pd.isna(value):
                return str(value)
        return ""

    @staticmethod
    def _extract_purchase_price(row: Dict[str, Any]) -> Optional[float]:
        for key in ("Purchase Price", "candidate_amount", "purchase_price"):
            value = row.get(key)
            if value is None or pd.isna(value):
                continue
            try:
                return float(value)
            except Exception:
                return None
        return None

    @staticmethod
    def _extract_week_start(row: Dict[str, Any]) -> str:
        for key in ("planned_week_start_iso", "planned_week_start", "week_start"):
            value = row.get(key)
            if value is not None and not pd.isna(value):
                return str(value)
        return ""

    @staticmethod
    def _invoice_row_payload(row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "invoice_ref": ProcessManager._extract_invoice_ref(row),
            "company_code": ProcessManager._extract_company_code(row),
            "customer": ProcessManager._extract_customer(row),
            "purchase_price": ProcessManager._extract_purchase_price(row),
            "due_date": str(row.get("Due Date", "")) if row.get("Due Date", None) is not None and not pd.isna(row.get("Due Date", None)) else "",
            "status": str(row.get("Status", "")) if row.get("Status", None) is not None and not pd.isna(row.get("Status", None)) else "",
            "offer_file_date": str(row.get("Offer File Date (UTC)", row.get("offer_file_date", ""))) if row.get("Offer File Date (UTC)", row.get("offer_file_date", None)) is not None and not pd.isna(row.get("Offer File Date (UTC)", row.get("offer_file_date", None))) else "",
            "summary_file_date": str(row.get("Summary File Date (UTC)", row.get("summary_file_date", ""))) if row.get("Summary File Date (UTC)", row.get("summary_file_date", None)) is not None and not pd.isna(row.get("Summary File Date (UTC)", row.get("summary_file_date", None))) else "",
            "excluded_stage": str(row.get("excluded_stage", "")) if row.get("excluded_stage", None) is not None and not pd.isna(row.get("excluded_stage", None)) else "",
            "excluded_reason": str(row.get("excluded_reason", "")) if row.get("excluded_reason", None) is not None and not pd.isna(row.get("excluded_reason", None)) else "",
            "excluded_reason_detail": str(row.get("excluded_reason_detail", "")) if row.get("excluded_reason_detail", None) is not None and not pd.isna(row.get("excluded_reason_detail", None)) else "",
            "planned_week_index": int(row.get("planned_week_index")) if row.get("planned_week_index", None) is not None and not pd.isna(row.get("planned_week_index", None)) else None,
            "planned_week_start": ProcessManager._extract_week_start(row),
        }

    @staticmethod
    def _contains_case_insensitive(series: pd.Series, value: str) -> pd.Series:
        return series.fillna("").astype(str).str.contains(value, case=False, regex=False)

    @staticmethod
    def _signature_for_chat_context(process_dir: Path, record: Dict[str, Any]) -> Tuple[Any, ...]:
        def _mtime(path: Path) -> float:
            return path.stat().st_mtime if path.exists() else 0.0

        return (
            record.get("completed_at"),
            _mtime(process_dir / "run_metadata.json"),
            _mtime(process_dir / "selected.xlsx"),
            _mtime(process_dir / "excluded.xlsx"),
            _mtime(process_dir / "weekly_plan.xlsx"),
            _mtime(process_dir / "weekly_exposure.xlsx"),
        )

    @staticmethod
    def _weekly_schedule_from_df(weekly_plan_df: pd.DataFrame) -> List[Dict[str, Any]]:
        if weekly_plan_df.empty:
            return []

        week_col = "planned_week_start_iso" if "planned_week_start_iso" in weekly_plan_df.columns else (
            "planned_week_start" if "planned_week_start" in weekly_plan_df.columns else None
        )
        amount_col = "Purchase Price" if "Purchase Price" in weekly_plan_df.columns else (
            "purchase_price" if "purchase_price" in weekly_plan_df.columns else None
        )
        index_col = "planned_week_index" if "planned_week_index" in weekly_plan_df.columns else None
        if week_col is None:
            return []

        grouped = weekly_plan_df.groupby(week_col, dropna=False)
        rows: List[Dict[str, Any]] = []
        for week, grp in grouped:
            row: Dict[str, Any] = {
                "week_start": str(week) if week is not None and not pd.isna(week) else "",
                "invoice_count": int(len(grp)),
                "total_amount": float(grp[amount_col].sum()) if amount_col else 0.0,
            }
            if index_col and index_col in grp.columns:
                try:
                    row["week_index"] = int(grp[index_col].dropna().astype(int).min())
                except Exception:
                    row["week_index"] = None
            rows.append(row)
        rows = sorted(rows, key=lambda item: (item.get("week_start", ""), item.get("week_index", 0) or 0))
        return rows

    @staticmethod
    def _usage_rows_from_metrics(metrics: Dict[str, Any], entity_type: str) -> List[Dict[str, Any]]:
        key = f"{entity_type}_weekly_usage"
        weekly = metrics.get(key) or {}
        if not weekly:
            single = metrics.get(f"{entity_type}_usage") or {}
            if single:
                weekly = {"current": single}

        rows: List[Dict[str, Any]] = []
        for week_start, entities in weekly.items():
            if not isinstance(entities, dict):
                continue
            for entity_id, usage in entities.items():
                if not isinstance(usage, dict):
                    continue
                rows.append(
                    {
                        "week_start": str(week_start),
                        "entity_type": entity_type,
                        "entity_id": str(entity_id),
                        "used_new": float(usage.get("used_new", 0) or 0),
                        "used_base": float(usage.get("used_base", 0) or 0),
                        "used_total": float(usage.get("used_total", usage.get("used", 0)) or 0),
                        "limit": float(usage.get("limit", 0) or 0),
                        "utilization_pct": float(usage.get("utilization_pct", 0) or 0),
                    }
                )
        rows.sort(key=lambda item: (item["week_start"], item["entity_id"]))
        return rows

    @staticmethod
    def _summary_cache_token(record: Dict[str, Any]) -> str:
        return str(
            record.get("completed_at")
            or record.get("progress_updated_at")
            or record.get("status")
            or "current"
        )

    def _clear_summary_cache(self, process_id: str) -> None:
        keys_to_remove = [key for key in self._summary_section_cache if key[0] == process_id]
        for key in keys_to_remove:
            self._summary_section_cache.pop(key, None)

    @staticmethod
    def _summary_artifact_key(section: str) -> str:
        if section not in SUMMARY_ARTIFACT_KEYS:
            raise ValueError(f"Unknown summary section: {section}")
        return SUMMARY_ARTIFACT_KEYS[section]

    @staticmethod
    def _compute_weekly_exposure_row_count(
        utilization_rows: Dict[str, List[Dict[str, Any]]],
        metadata: Dict[str, Any],
    ) -> int:
        count = sum(len(rows) for rows in utilization_rows.values())
        if count > 0:
            return int(count)
        weekly_exposure = metadata.get("weekly_exposure")
        if isinstance(weekly_exposure, list):
            return int(len(weekly_exposure))
        return 0

    @staticmethod
    def _available_entity_types_from_utilization(
        utilization_rows: Dict[str, List[Dict[str, Any]]],
    ) -> List[str]:
        available = [entity_type for entity_type, rows in utilization_rows.items() if rows]
        return sorted(available) or list(SUMMARY_ENTITY_TYPES)

    @staticmethod
    def _summary_json_payload(payload: Dict[str, Any]) -> str:
        return json.dumps(payload, default=str, separators=(",", ":"))

    def _persist_summary_sections(
        self,
        process_id: str,
        sections: Dict[str, Dict[str, Any]],
        record: Optional[Dict[str, Any]] = None,
    ) -> None:
        cache_token = self._summary_cache_token(record or {})
        for section, payload in sections.items():
            artifact_key = self._summary_artifact_key(section)
            self.artifact_store.upsert_text_artifact(
                process_id,
                artifact_key,
                self._summary_json_payload(payload),
                metadata={
                    "summary_section": section,
                    "summary_version": SUMMARY_ARTIFACT_VERSION,
                },
                artifact_kind=SUMMARY_ARTIFACT_KIND,
            )
            self._summary_section_cache[(process_id, section, cache_token)] = payload

    def _load_summary_section_from_artifact(
        self,
        process_id: str,
        section: str,
    ) -> Optional[Dict[str, Any]]:
        text = self.artifact_store.get_text_artifact(process_id, self._summary_artifact_key(section))
        if not text:
            return None
        try:
            payload = json.loads(text)
        except Exception:
            logger.warning("Failed decoding optimizer summary artifact %s for %s", section, process_id)
            return None
        return payload if isinstance(payload, dict) else None

    def _build_summary_sections(
        self,
        *,
        metadata: Dict[str, Any],
        selected_df: pd.DataFrame,
        excluded_df: pd.DataFrame,
        weekly_plan_df: pd.DataFrame,
    ) -> Dict[str, Dict[str, Any]]:
        metrics = metadata.get("metrics") or {}
        planning_mode = str(metadata.get("planning_mode", metrics.get("planning_mode", "single_week")))
        deferred_reasons = metrics.get("deferred_reasons", metadata.get("deferred_reasons", {})) or {}
        exclusion_summary = _build_exclusion_summary(excluded_df)
        top_customers = _build_top_customers(selected_df)
        binding_constraints = _find_binding_constraints(metrics)
        weekly_schedule = self._weekly_schedule_from_df(weekly_plan_df)
        if not weekly_schedule and isinstance(metadata.get("weekly_plan"), list):
            weekly_schedule = self._weekly_schedule_from_df(pd.DataFrame(metadata.get("weekly_plan") or []))

        utilization_rows = {
            entity_type: self._usage_rows_from_metrics(metrics, entity_type)
            for entity_type in SUMMARY_ENTITY_TYPES
        }
        available_entity_types = self._available_entity_types_from_utilization(utilization_rows)
        selected_row_count = int(len(selected_df))
        if selected_row_count == 0 and metrics.get("optimized_submitted_count") is not None:
            selected_row_count = int(metrics.get("optimized_submitted_count") or 0)
        excluded_row_count = int(len(excluded_df))
        if excluded_row_count == 0:
            excluded_row_count = int(
                (metrics.get("rule_excluded_count", 0) or 0)
                + (metrics.get("not_selected_count", 0) or 0)
            )
        weekly_plan_row_count = int(len(weekly_plan_df))
        if weekly_plan_row_count == 0 and metrics.get("weekly_plan_count") is not None:
            weekly_plan_row_count = int(metrics.get("weekly_plan_count") or 0)
        row_counts = {
            "selected": selected_row_count,
            "excluded": excluded_row_count,
            "weekly_plan": weekly_plan_row_count,
            "weekly_exposure": self._compute_weekly_exposure_row_count(utilization_rows, metadata),
        }

        overview = {
            "cohort": metadata.get("cohort"),
            "planning_mode": planning_mode,
            "source_profile": metadata.get("source_profile"),
            "horizon_weeks": metadata.get("horizon_weeks", metrics.get("horizon_weeks")),
            "kpis": {
                "candidate_count": metrics.get("baseline_submitted_count"),
                "selected_count": metrics.get("optimized_submitted_count"),
                "rule_excluded_count": metrics.get("rule_excluded_count"),
                "optimizer_not_selected_count": metrics.get("not_selected_count"),
                "candidate_amount": metrics.get("candidate_total_amount"),
                "selected_amount": metrics.get("selected_total_amount"),
                "selected_amount_ratio_pct": metrics.get("selected_amount_ratio_pct"),
                "top3_customer_concentration_pct": metrics.get("top3_customer_concentration_pct"),
                "optimizer_status": metrics.get("optimizer_status"),
                "weekly_plan_count": metrics.get("weekly_plan_count", len(weekly_plan_df)),
            },
            "deferred_reasons": deferred_reasons,
            "binding_constraints": binding_constraints,
            "top_customers": top_customers[:10],
            "weekly_schedule_summary": weekly_schedule[:10],
            "row_counts": row_counts,
            "available_entity_types": available_entity_types,
        }

        exclusions = {"rows": exclusion_summary}
        utilization = {
            "available_entity_types": available_entity_types,
            "rows": utilization_rows,
        }
        schedule = {
            "planning_mode": planning_mode,
            "rows": weekly_schedule,
        }
        return {
            "overview": overview,
            "exclusions": exclusions,
            "utilization": utilization,
            "schedule": schedule,
        }

    def _build_summary_sections_for_process(
        self,
        process_id: str,
        *,
        record: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        record = record or self.store.get_process_head(process_id)
        if record is None:
            raise ValueError(f"Process {process_id} not found")

        metadata = self.get_results(process_id) or {}
        process_dir = Path(record["process_dir"]) if record.get("process_dir") else (TEMP_MATERIALIZED_ROOT / "__missing__")
        selected_df = self._load_bucket_df(process_id, process_dir, "selected")
        excluded_df = self._load_bucket_df(process_id, process_dir, "excluded")
        weekly_plan_df = self._load_bucket_df(process_id, process_dir, "weekly_plan")
        return self._build_summary_sections(
            metadata=metadata,
            selected_df=selected_df,
            excluded_df=excluded_df,
            weekly_plan_df=weekly_plan_df,
        )

    def _get_summary_section(
        self,
        process_id: str,
        section: str,
    ) -> Dict[str, Any]:
        record = self.store.get_process_head(process_id)
        if record is None:
            raise ValueError(f"Process {process_id} not found")

        cache_token = self._summary_cache_token(record)
        cache_key = (process_id, section, cache_token)
        cached = self._summary_section_cache.get(cache_key)
        if cached is not None:
            return cached

        payload = self._load_summary_section_from_artifact(process_id, section)
        if payload is not None:
            self._summary_section_cache[cache_key] = payload
            return payload

        sections = self._build_summary_sections_for_process(process_id, record=record)
        if str(record.get("status", "")).lower() == "completed":
            self._persist_summary_sections(process_id, sections, record=record)
        else:
            self._summary_section_cache[cache_key] = sections[section]
        return sections[section]

    def get_overview_summary(self, process_id: str) -> Dict[str, Any]:
        return self._get_summary_section(process_id, "overview")

    def get_exclusions_summary(self, process_id: str) -> Dict[str, Any]:
        return self._get_summary_section(process_id, "exclusions")

    def get_utilization_summary(self, process_id: str) -> Dict[str, Any]:
        return self._get_summary_section(process_id, "utilization")

    def get_schedule_summary(self, process_id: str) -> Dict[str, Any]:
        return self._get_summary_section(process_id, "schedule")

    def _read_bucket_df(self, process_dir: Path, bucket: str) -> pd.DataFrame:
        filename_map = {
            "selected": "selected.xlsx",
            "excluded": "excluded.xlsx",
            "weekly_plan": "weekly_plan.xlsx",
        }
        filename = filename_map.get(bucket)
        if filename is None:
            raise ValueError("bucket must be one of: selected, excluded, weekly_plan")
        return self._safe_excel_read(process_dir / filename)

    def _bucket_rows_from_artifacts(self, process_id: str, bucket: str) -> List[Dict[str, Any]]:
        return self.artifact_store.get_all_invoice_rows(process_id, bucket)

    def _bucket_df_from_artifacts(self, process_id: str, bucket: str) -> pd.DataFrame:
        rows = self._bucket_rows_from_artifacts(process_id, bucket)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def _load_bucket_df(self, process_id: str, process_dir: Path, bucket: str) -> pd.DataFrame:
        artifact_df = self._bucket_df_from_artifacts(process_id, bucket)
        if not artifact_df.empty:
            return artifact_df
        return self._read_bucket_df(process_dir, bucket)

    def _load_weekly_exposure_df(self, process_id: str, process_dir: Path) -> pd.DataFrame:
        rows = self.artifact_store.get_all_exposure_rows(process_id)
        if rows:
            return pd.DataFrame(rows)
        return self._safe_excel_read(process_dir / "weekly_exposure.xlsx")

    def _materialized_process_dir(self, process_id: str) -> Path:
        path = TEMP_MATERIALIZED_ROOT / process_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _store_text_artifact_from_path(
        self,
        process_id: str,
        artifact_key: str,
        path: Path,
        *,
        artifact_kind: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not path.exists():
            return
        self.artifact_store.upsert_text_artifact(
            process_id,
            artifact_key,
            path.read_text(encoding="utf-8"),
            metadata=metadata,
            artifact_kind=artifact_kind,
        )

    def _store_structured_artifacts(self, process_id: str, process_dir: Path, metadata: Optional[Dict[str, Any]]) -> None:
        metadata = metadata or self.get_results(process_id) or {}
        selected_df = pd.DataFrame()
        excluded_df = pd.DataFrame()
        weekly_plan_df = pd.DataFrame()

        self._store_text_artifact_from_path(
            process_id,
            "limits.yaml",
            process_dir / "limits.yaml",
            artifact_kind="config/yaml",
        )
        self._store_text_artifact_from_path(
            process_id,
            "rules.yaml",
            process_dir / "rules.yaml",
            artifact_kind="config/yaml",
        )
        self._store_text_artifact_from_path(
            process_id,
            "run_summary.md",
            process_dir / "run_summary.md",
            artifact_kind="report/markdown",
        )

        metadata_path = process_dir / "run_metadata.json"
        if metadata_path.exists():
            self.artifact_store.upsert_text_artifact(
                process_id,
                "run_metadata.json",
                metadata_path.read_text(encoding="utf-8"),
                metadata={"source": "file"},
                artifact_kind="application/json",
            )
        elif metadata:
            self.artifact_store.upsert_text_artifact(
                process_id,
                "run_metadata.json",
                json.dumps(metadata, indent=2, default=str),
                metadata={"source": "process_store"},
                artifact_kind="application/json",
            )

        for bucket in ("selected", "excluded", "weekly_plan"):
            df = self._read_bucket_df(process_dir, bucket)
            if bucket == "selected":
                selected_df = df
            elif bucket == "excluded":
                excluded_df = df
            else:
                weekly_plan_df = df
            if not df.empty:
                self.artifact_store.replace_invoice_rows(
                    process_id,
                    bucket,
                    df.fillna("").to_dict(orient="records"),
                )

        weekly_exposure_path = process_dir / "weekly_exposure.xlsx"
        if weekly_exposure_path.exists():
            exposure_df = self._safe_excel_read(weekly_exposure_path)
            rows = exposure_df.fillna("").to_dict(orient="records") if not exposure_df.empty else []
            if rows:
                self.artifact_store.replace_exposure_rows(process_id, rows)
        else:
            exposure_rows = metadata.get("weekly_exposure", []) if metadata else []
            if exposure_rows:
                self.artifact_store.replace_exposure_rows(process_id, exposure_rows)

        summary_sections = self._build_summary_sections(
            metadata=metadata,
            selected_df=selected_df,
            excluded_df=excluded_df,
            weekly_plan_df=weekly_plan_df,
        )
        self._clear_summary_cache(process_id)
        self._persist_summary_sections(
            process_id,
            summary_sections,
            record=self.store.get_process_head(process_id),
        )

    def sync_process_artifacts(self, process_id: str, force: bool = False) -> Dict[str, int]:
        record = self.store.get_process(process_id)
        if record is None:
            raise ValueError(f"Process {process_id} not found")

        process_dir = Path(record["process_dir"])
        metadata = self.get_results(process_id) or {}

        if force:
            self.artifact_store.delete_process_artifacts(process_id)
        self._store_structured_artifacts(process_id, process_dir, metadata)

        return {
            "selected_rows": len(self.artifact_store.get_all_invoice_rows(process_id, "selected")),
            "excluded_rows": len(self.artifact_store.get_all_invoice_rows(process_id, "excluded")),
            "weekly_plan_rows": len(self.artifact_store.get_all_invoice_rows(process_id, "weekly_plan")),
            "weekly_exposure_rows": len(self.artifact_store.get_all_exposure_rows(process_id)),
            "summary_sections": len(SUMMARY_ARTIFACT_KEYS),
        }

    def _ensure_text_file(self, process_id: str, process_dir: Path, filename: str) -> Optional[Path]:
        local_path = process_dir / filename
        if local_path.exists():
            return local_path
        return self.artifact_store.materialize_text_artifact(process_id, filename, filename=filename)

    def _ensure_runtime_config_file(self, process_id: str, process_dir: Path, filename: str) -> Optional[Path]:
        local_path = process_dir / filename
        if local_path.exists():
            return local_path
        materialized_dir = self._materialized_process_dir(process_id)
        content = self.artifact_store.get_text_artifact(process_id, filename)
        if content is None:
            return None
        path = materialized_dir / filename
        path.write_text(content, encoding="utf-8")
        return path

    def _materialize_bucket_workbook(self, process_id: str, bucket: str) -> Optional[Path]:
        path = self.artifact_store.materialize_workbook(process_id, bucket)
        if path is not None:
            return path
        return None

    def _materialize_report_files(self, process_id: str) -> Dict[str, Optional[Path]]:
        metadata = self.get_results(process_id)
        if not metadata:
            return {"summary": None, "pdf": None, "docx": None}

        process_dir = self._materialized_process_dir(process_id)
        selected_df = self._bucket_df_from_artifacts(process_id, "selected")
        excluded_df = self._bucket_df_from_artifacts(process_id, "excluded")
        if selected_df.empty and excluded_df.empty:
            return {"summary": self._ensure_text_file(process_id, process_dir, "run_summary.md"), "pdf": None, "docx": None}

        from app.optimizer.report.report_builder import generate_full_report

        md_path, pdf_path, docx_path = generate_full_report(
            metadata,
            process_dir,
            selected_df=selected_df if not selected_df.empty else None,
            excluded_df=excluded_df if not excluded_df.empty else None,
        )
        return {"summary": md_path, "pdf": pdf_path, "docx": docx_path}

    @staticmethod
    def _resolve_sheet_name_from_bytes(file_content: bytes, requested_sheet_name: str) -> str:
        """Resolve requested sheet via exact/case-insensitive/summary fallback."""
        try:
            workbook = pd.ExcelFile(BytesIO(file_content), engine="openpyxl")
        except Exception:
            return requested_sheet_name

        available = workbook.sheet_names
        if requested_sheet_name in available:
            return requested_sheet_name

        by_lower = {name.lower(): name for name in available}
        requested_lower = requested_sheet_name.lower()
        if requested_lower in by_lower:
            return by_lower[requested_lower]

        if requested_sheet_name == "SAPUI5 Export":
            for candidate in SUMMARY_FALLBACK_SHEETS:
                match = by_lower.get(candidate.lower())
                if match:
                    return match

        return requested_sheet_name

    @staticmethod
    def _infer_source_profile_from_bytes(file_content: bytes, sheet_name: str) -> str:
        for sheet in (sheet_name, 0):
            try:
                preview = pd.read_excel(
                    BytesIO(file_content),
                    sheet_name=sheet,
                    nrows=0,
                    engine="openpyxl",
                )
                return detect_source_profile(preview.columns)
            except Exception:
                continue
        return SOURCE_PROFILE_EXTRACTION

    def create_process(
        self,
        file_content: bytes,
        filename: str,
        cohort: Optional[str] = None,
        sheet_name: str = "SAPUI5 Export",
    ) -> Dict[str, Any]:
        """Create a new optimization process.

        Saves the uploaded file, copies default config files, detects
        available cohorts, and inserts a record in the store.
        """
        process_id = str(uuid.uuid4())
        process_dir = DATA_DIR / process_id
        process_dir.mkdir(parents=True, exist_ok=True)
        requested_sheet_name = sheet_name
        sheet_name = self._resolve_sheet_name_from_bytes(file_content, requested_sheet_name)
        source_profile = self._infer_source_profile_from_bytes(file_content, sheet_name)
        if source_profile in {"offer_file", "hybrid"} and requested_sheet_name == "SAPUI5 Export":
            sheet_name = self._resolve_sheet_name_from_bytes(file_content, "Sheet1")

        # Save uploaded extraction file
        extraction_path = process_dir / "extraction.xlsx"
        extraction_path.write_bytes(file_content)

        # Copy default config files into the process directory
        limits_dest = process_dir / "limits.yaml"
        rules_dest = process_dir / "rules.yaml"

        if DEFAULT_LIMITS_CONFIG.exists():
            shutil.copy2(DEFAULT_LIMITS_CONFIG, limits_dest)
        else:
            limits_dest.write_text(yaml.dump({
                "facility_limits_by_company_code": {},
                "customer_limits": {},
                "group_limits": {},
                "customer_to_group": {},
                "base_exposure": {
                    "facility": {},
                    "customer": {},
                    "group": {},
                },
                "defaults": {
                    "customer_limit_fraction_of_facility": 0.15,
                    "group_limit_fraction_of_facility": 0.30,
                },
                "synthetic_generation": {
                    "enabled": True,
                    "alpha": 0.85,
                    "beta": 0.15,
                    "gamma": 0.30,
                },
            }), encoding="utf-8")

        if DEFAULT_RULES_CONFIG.exists():
            shutil.copy2(DEFAULT_RULES_CONFIG, rules_dest)
        else:
            rules_dest.write_text(yaml.dump({"rules": []}), encoding="utf-8")

        self._store_text_artifact_from_path(
            process_id,
            "limits.yaml",
            limits_dest,
            artifact_kind="config/yaml",
        )
        self._store_text_artifact_from_path(
            process_id,
            "rules.yaml",
            rules_dest,
            artifact_kind="config/yaml",
        )

        # Detect available cohorts
        available_cohorts = detect_cohorts(extraction_path, sheet_name)

        # Create DB record
        record = self.store.create_process(
            process_id=process_id,
            process_dir=str(process_dir),
            extraction_filename=filename,
            cohort=cohort,
            sheet_name=sheet_name,
            source_profile=source_profile,
        )

        default_lifecycle_history_path = self._default_lifecycle_history_path()
        if default_lifecycle_history_path is not None:
            record = self.store.update_process(
                process_id,
                lifecycle_input_path=str(default_lifecycle_history_path),
            ) or record

        record["available_cohorts"] = available_cohorts
        return record

    def create_process_from_existing(
        self,
        extraction_filename: str,
        cohort: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a process from an already-uploaded extraction file.

        Searches existing process directories for the named extraction.
        """
        source_path = None
        if DATA_DIR.exists():
            for proc_dir in DATA_DIR.iterdir():
                candidate = proc_dir / "extraction.xlsx"
                if candidate.exists():
                    meta_check = proc_dir / "extraction_filename.txt"
                    stored_name = None
                    # Check store for filename match
                    for rec in self.store.list_processes(limit=500):
                        if rec.get("process_dir") == str(proc_dir) and rec.get("extraction_filename") == extraction_filename:
                            source_path = candidate
                            break
                    if source_path:
                        break

        if source_path is None:
            raise FileNotFoundError(f"No existing extraction found with filename: {extraction_filename}")

        content = source_path.read_bytes()
        return self.create_process(content, extraction_filename, cohort)

    def start_optimization(self, process_id: str) -> Dict[str, Any]:
        """Validate and mark process as running. Returns updated record.

        The actual execution should be dispatched via BackgroundTasks
        by calling _run_in_background(process_id).
        """
        record = self.store.get_process(process_id)
        if record is None:
            raise ValueError(f"Process {process_id} not found")

        status = record["status"]
        if status == "running":
            raise ValueError(f"Process {process_id} is already running")
        if status == "completed":
            raise ValueError(f"Process {process_id} has already completed")

        planning_mode = record.get("planning_mode", "single_week")
        if planning_mode == "single_week" and not record.get("cohort"):
            raise ValueError(f"Process {process_id} has no cohort set. Set a cohort before running.")

        logger.info(
            "Starting optimizer process %s with planning_mode=%s, cohort=%s, horizon_weeks=%s, attempt_cap=%s",
            process_id,
            planning_mode,
            record.get("cohort"),
            record.get("horizon_weeks", 8),
            record.get("attempt_cap", 1),
        )

        self.store.update_process(
            process_id,
            status="running",
            started_at=datetime.now().isoformat(),
            progress_json=None,
            progress_updated_at=None,
        )
        started = self.store.get_process(process_id)
        if started is not None:
            self._update_progress(
                process_id,
                step_code="start",
                step_label="Starting optimizer process",
                step_index=0,
                step_total=10,
                phase_status="running",
                started_at=started.get("started_at"),
            )
        return started

    def run_in_background(self, process_id: str) -> None:
        """Execute the optimizer pipeline. Called from BackgroundTasks."""
        record = self.store.get_process(process_id)
        if record is None:
            logger.error("Process %s not found for background run", process_id)
            return

        process_dir = Path(record["process_dir"])
        try:
            started_at = record.get("started_at")
            if not started_at:
                started_at = datetime.now().isoformat()
                self.store.update_process(process_id, started_at=started_at)

            self._update_progress(
                process_id,
                step_code="start",
                step_label="Preparing optimizer run",
                step_index=0,
                step_total=10,
                phase_status="running",
                started_at=started_at,
            )

            last_progress_write_monotonic = 0.0
            last_progress_key: tuple[Any, ...] | None = None

            def _progress_callback(payload: Dict[str, Any]) -> None:
                nonlocal last_progress_write_monotonic, last_progress_key
                step_index = int(payload.get("step_index", 0))
                step_total = int(payload.get("step_total", 10))
                step_code = str(payload.get("step_code", "running"))
                step_label = str(payload.get("step_label", "Running optimizer"))
                phase_status = str(payload.get("phase_status", "running"))
                details = payload.get("details")
                progress_key = (step_index, step_code, phase_status, json.dumps(details or {}, sort_keys=True, default=str))
                now = time.monotonic()
                if (
                    last_progress_key == progress_key
                    and (now - last_progress_write_monotonic) < 0.3
                ):
                    return
                last_progress_key = progress_key
                last_progress_write_monotonic = now
                self._update_progress(
                    process_id,
                    step_code=step_code,
                    step_label=step_label,
                    step_index=step_index,
                    step_total=step_total,
                    phase_status=phase_status,
                    started_at=started_at,
                    details=details if isinstance(details, dict) else {},
                )

            process_extraction_path = process_dir / "extraction.xlsx"
            materialized_limits_path = self._ensure_runtime_config_file(process_id, process_dir, "limits.yaml")
            materialized_rules_path = self._ensure_runtime_config_file(process_id, process_dir, "rules.yaml")
            default_lifecycle_history_path = self._default_lifecycle_history_path()
            resolved_lifecycle_input = (
                record.get("lifecycle_input_path")
                or (str(default_lifecycle_history_path) if default_lifecycle_history_path else None)
                or str(process_extraction_path)
            )
            if default_lifecycle_history_path and not record.get("lifecycle_input_path"):
                logger.info(
                    "Using default lifecycle history file for process %s: %s",
                    process_id,
                    default_lifecycle_history_path,
                )
            config = {
                "input_path": str(process_dir / "extraction.xlsx"),
                "sheet_name": record.get("sheet_name") or "SAPUI5 Export",
                "cohort": record["cohort"],
                "planning_mode": record.get("planning_mode") or "single_week",
                "source_profile": record.get("source_profile") or SOURCE_PROFILE_EXTRACTION,
                "planning_start_date": record.get("planning_start_date") or record.get("cohort"),
                "horizon_weeks": record.get("horizon_weeks", 8),
                "attempt_cap": record.get("attempt_cap", 1),
                "release_event": (
                    record.get("release_event_mode")
                    or record.get("release_event")
                    or "reconciliation_file_date"
                ),
                "lifecycle_input_path": resolved_lifecycle_input,
                "limits_config_path": str(materialized_limits_path or (process_dir / "limits.yaml")),
                "rules_config_path": str(materialized_rules_path or (process_dir / "rules.yaml")),
                "output_dir": str(process_dir),
                "solver_max_time_seconds": record.get("solver_max_time_seconds", 60),
                "solver_random_seed": record.get("solver_random_seed", 0),
                "solver_num_search_workers": record.get("solver_num_search_workers", 1),
                "exclude_payment_block": False,
            }

            metadata = run_optimizer_pipeline(config, progress_callback=_progress_callback)

            metrics = metadata.get("metrics", {})
            self.store.update_process(
                process_id,
                status="completed",
                completed_at=datetime.now().isoformat(),
                run_metadata_json=json.dumps(metadata, default=str),
                candidate_count=metrics.get("baseline_submitted_count"),
                selected_count=metrics.get("optimized_submitted_count"),
                excluded_count=(
                    metrics.get("rule_excluded_count", 0)
                    + metrics.get("not_selected_count", 0)
                ),
                candidate_amount=metrics.get("candidate_total_amount"),
                selected_amount=metrics.get("selected_total_amount"),
                optimizer_status=metrics.get("optimizer_status"),
                progress_json=json.dumps(
                    self._progress_payload(
                        step_code="pipeline_complete",
                        step_label="Pipeline complete",
                        step_index=10,
                        step_total=10,
                        phase_status="completed",
                        started_at=started_at,
                    ),
                    default=str,
                ),
                progress_updated_at=datetime.now().isoformat(),
                weekly_plan_output=metadata.get("weekly_plan_output"),
                weekly_exposure_output=metadata.get("weekly_exposure_output"),
            )
            self._store_structured_artifacts(process_id, process_dir, metadata)
            self._PROGRESS_LOG_STATE.pop(process_id, None)
            logger.info("Process %s completed successfully", process_id)

        except Exception as exc:
            logger.exception("Process %s failed: %s", process_id, exc)
            self.store.update_process(
                process_id,
                status="failed",
                completed_at=datetime.now().isoformat(),
                error_message=str(exc),
                progress_json=json.dumps(
                    self._progress_payload(
                        step_code="failed",
                        step_label="Optimizer failed",
                        step_index=10,
                        step_total=10,
                        phase_status="failed",
                        started_at=record.get("started_at"),
                        details={"error_message": str(exc)},
                    ),
                    default=str,
                ),
                progress_updated_at=datetime.now().isoformat(),
            )
            self._PROGRESS_LOG_STATE.pop(process_id, None)

    def get_process(self, process_id: str) -> Optional[Dict[str, Any]]:
        return self.store.get_process(process_id)

    def list_processes(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        return self.store.list_processes(status=status, limit=limit, offset=offset)

    def resolve_process_id(
        self,
        process_ref: str,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Resolve a full or truncated process id against existing processes."""
        normalized = (process_ref or "").strip()
        if normalized.endswith("..."):
            normalized = normalized[:-3]

        limit = max(1, min(int(limit), 100))
        records = self.store.list_processes(status=status, limit=limit, offset=0)

        exact = [r for r in records if str(r.get("id", "")) == normalized]
        if exact:
            match = exact[0]
            return {
                "match_type": "exact",
                "process_id": match["id"],
                "matches": [
                    {
                        "process_id": match["id"],
                        "status": match.get("status"),
                        "cohort": match.get("cohort"),
                        "created_at": match.get("created_at"),
                    }
                ],
                "scanned": len(records),
            }

        prefix = [r for r in records if str(r.get("id", "")).startswith(normalized)]
        matches = [
            {
                "process_id": r.get("id"),
                "status": r.get("status"),
                "cohort": r.get("cohort"),
                "created_at": r.get("created_at"),
            }
            for r in prefix
        ]
        if len(matches) == 1:
            return {
                "match_type": "prefix_unique",
                "process_id": matches[0]["process_id"],
                "matches": matches,
                "scanned": len(records),
            }
        if len(matches) > 1:
            return {
                "match_type": "prefix_ambiguous",
                "process_id": None,
                "matches": matches,
                "scanned": len(records),
            }

        return {
            "match_type": "not_found",
            "process_id": None,
            "matches": [],
            "scanned": len(records),
        }

    def delete_process(self, process_id: str) -> bool:
        record = self.store.get_process_head(process_id)
        if record is None:
            return False

        process_dir = Path(record["process_dir"])
        self._CHAT_CONTEXT_CACHE.pop(process_id, None)
        self._clear_summary_cache(process_id)
        self._PROGRESS_LOG_STATE.pop(process_id, None)
        self.artifact_store.delete_process_artifacts(process_id)
        self.artifact_store.clear_temp_artifacts(process_id)
        if process_dir.exists():
            shutil.rmtree(process_dir, ignore_errors=True)

        return self.store.delete_process(process_id)

    def get_results(self, process_id: str) -> Optional[Dict[str, Any]]:
        record = self.store.get_process(process_id)
        if record is None:
            return None

        metadata_json = record.get("run_metadata_json")
        if metadata_json:
            return json.loads(metadata_json)

        artifact_json = self.artifact_store.get_text_artifact(process_id, "run_metadata.json")
        if artifact_json:
            return json.loads(artifact_json)

        # Try reading from file
        process_dir = Path(record["process_dir"])
        metadata_path = process_dir / "run_metadata.json"
        if metadata_path.exists():
            return json.loads(metadata_path.read_text(encoding="utf-8"))

        return None

    def get_chat_context(self, process_id: str) -> Dict[str, Any]:
        """Build and cache compact chat-oriented optimizer context.

        This context is intentionally bounded and excludes full weekly_plan /
        weekly_exposure arrays from run metadata.
        """
        record = self.store.get_process(process_id)
        if record is None:
            raise ValueError(f"Process {process_id} not found")

        process_dir = Path(record["process_dir"])
        metadata = self.get_results(process_id) or {}
        metrics = metadata.get("metrics") or {}

        selected_df = self._load_bucket_df(process_id, process_dir, "selected")
        excluded_df = self._load_bucket_df(process_id, process_dir, "excluded")
        weekly_plan_df = self._load_bucket_df(process_id, process_dir, "weekly_plan")
        weekly_exposure_df = self._load_weekly_exposure_df(process_id, process_dir)

        signature = (
            record.get("completed_at"),
            int(len(selected_df)),
            int(len(excluded_df)),
            int(len(weekly_plan_df)),
            int(len(weekly_exposure_df)),
        )
        cached = self._CHAT_CONTEXT_CACHE.get(process_id)
        if cached and cached.get("signature") == signature:
            return cached["context"]

        planning_mode = str(metadata.get("planning_mode", metrics.get("planning_mode", "single_week")))
        deferred_reasons = metrics.get("deferred_reasons", metadata.get("deferred_reasons", {})) or {}
        exclusion_summary = _build_exclusion_summary(excluded_df)
        top_customers = _build_top_customers(selected_df)
        binding_constraints = _find_binding_constraints(metrics)
        weekly_schedule = self._weekly_schedule_from_df(weekly_plan_df)

        utilization = {
            "facility": self._usage_rows_from_metrics(metrics, "facility"),
            "customer": self._usage_rows_from_metrics(metrics, "customer"),
            "group": self._usage_rows_from_metrics(metrics, "group"),
        }

        context = {
            "process_id": process_id,
            "cohort": metadata.get("cohort"),
            "planning_mode": planning_mode,
            "source_profile": metadata.get("source_profile"),
            "horizon_weeks": metadata.get("horizon_weeks", metrics.get("horizon_weeks")),
            "kpis": {
                "candidate_count": metrics.get("baseline_submitted_count"),
                "selected_count": metrics.get("optimized_submitted_count"),
                "rule_excluded_count": metrics.get("rule_excluded_count"),
                "optimizer_not_selected_count": metrics.get("not_selected_count"),
                "candidate_amount": metrics.get("candidate_total_amount"),
                "selected_amount": metrics.get("selected_total_amount"),
                "selected_amount_ratio_pct": metrics.get("selected_amount_ratio_pct"),
                "top3_customer_concentration_pct": metrics.get("top3_customer_concentration_pct"),
                "optimizer_status": metrics.get("optimizer_status"),
                "weekly_plan_count": metrics.get("weekly_plan_count", len(weekly_plan_df)),
            },
            "rule_summaries": metadata.get("rule_summaries", []),
            "deferred_reasons": deferred_reasons,
            "exclusion_summary": exclusion_summary,
            "weekly_schedule_summary": weekly_schedule,
            "top_customers": top_customers,
            "binding_constraints": binding_constraints,
            "utilization": utilization,
            "row_counts": {
                "selected": int(len(selected_df)),
                "excluded": int(len(excluded_df)),
                "weekly_plan": int(len(weekly_plan_df)),
                "weekly_exposure": int(len(weekly_exposure_df)),
            },
            "available_entity_types": sorted(
                list(
                    {
                        str(value)
                        for value in (
                            weekly_exposure_df["entity_type"].tolist()
                            if "entity_type" in weekly_exposure_df.columns
                            else []
                        )
                    }
                )
            ) or ["facility", "customer", "group"],
        }

        self._CHAT_CONTEXT_CACHE[process_id] = {
            "signature": signature,
            "context": context,
        }
        return context

    def get_invoice_rows(
        self,
        process_id: str,
        bucket: str = "excluded",
        invoice_ref: Optional[str] = None,
        customer: Optional[str] = None,
        company_code: Optional[str] = None,
        excluded_stage: Optional[str] = None,
        excluded_reason: Optional[str] = None,
        week_start: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Load paged invoice rows for selected/excluded/weekly_plan with filters."""
        record = self.store.get_process_head(process_id)
        if record is None:
            return self._pagination([], total=0, limit=limit, offset=offset)

        limit = max(1, min(int(limit), 100))
        offset = max(0, int(offset))

        if self.artifact_store.has_invoice_rows(process_id, bucket):
            paged = self.artifact_store.load_invoice_rows(
                process_id,
                bucket,
                invoice_ref=invoice_ref,
                customer=customer,
                company_code=company_code,
                excluded_stage=excluded_stage,
                excluded_reason=excluded_reason,
                week_start=week_start,
                limit=limit,
                offset=offset,
            )
            rows = [self._invoice_row_payload(row) for row in paged["rows"]]
            return self._pagination(rows, total=paged["total"], limit=limit, offset=offset)

        process_dir = Path(record["process_dir"])
        df = self._read_bucket_df(process_dir, bucket)
        if df.empty:
            return self._pagination([], total=0, limit=limit, offset=offset)

        if invoice_ref:
            ref_col = "Invoice Reference" if "Invoice Reference" in df.columns else (
                "invoice_reference" if "invoice_reference" in df.columns else "invoice_ref"
            )
            if ref_col in df.columns:
                df = df[self._contains_case_insensitive(df[ref_col], invoice_ref)]

        if customer:
            cust_col = "Customer" if "Customer" in df.columns else (
                "customer" if "customer" in df.columns else "debtor_id"
            )
            if cust_col in df.columns:
                df = df[self._contains_case_insensitive(df[cust_col], customer)]

        if company_code:
            cc_col = "Company Code" if "Company Code" in df.columns else (
                "company_code" if "company_code" in df.columns else "seller_id_external"
            )
            if cc_col in df.columns:
                df = df[self._contains_case_insensitive(df[cc_col], company_code)]

        if excluded_stage and "excluded_stage" in df.columns:
            df = df[self._contains_case_insensitive(df["excluded_stage"], excluded_stage)]

        if excluded_reason and "excluded_reason" in df.columns:
            df = df[self._contains_case_insensitive(df["excluded_reason"], excluded_reason)]

        if week_start:
            if "planned_week_start_iso" in df.columns:
                df = df[self._contains_case_insensitive(df["planned_week_start_iso"], week_start)]
            elif "planned_week_start" in df.columns:
                df = df[self._contains_case_insensitive(df["planned_week_start"], week_start)]
            elif "week_start" in df.columns:
                df = df[self._contains_case_insensitive(df["week_start"], week_start)]

        sort_columns_map = {
            "selected": ["Due Date", "Invoice Reference", "invoice_reference", "invoice_ref"],
            "excluded": ["excluded_stage", "excluded_reason", "Invoice Reference", "invoice_reference", "invoice_ref"],
            "weekly_plan": ["planned_week_start_iso", "planned_week_start", "planned_week_index", "Invoice Reference", "invoice_reference", "invoice_ref"],
        }
        df = self._stable_sort(df, sort_columns_map.get(bucket, []))

        total = len(df)
        page = df.iloc[offset : offset + limit]
        rows = []
        for _, row in page.iterrows():
            rows.append(self._invoice_row_payload(dict(row)))
        return self._pagination(rows, total=total, limit=limit, offset=offset)

    def get_weekly_exposure_rows(
        self,
        process_id: str,
        limit: int = 50,
        offset: int = 0,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        week_start: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load paged weekly exposure rows with optional filters."""
        record = self.store.get_process_head(process_id)
        if record is None:
            return self._pagination([], total=0, limit=limit, offset=offset)

        limit = max(1, min(int(limit), 100))
        offset = max(0, int(offset))

        if self.artifact_store.has_exposure_rows(process_id):
            paged = self.artifact_store.load_exposure_rows(
                process_id,
                entity_type=entity_type,
                entity_id=entity_id,
                week_start=week_start,
                limit=limit,
                offset=offset,
            )
            normalized_rows = []
            for row in paged["rows"]:
                normalized_rows.append(
                    {
                        "week_start": str(row.get("week_start", "")),
                        "entity_type": str(row.get("entity_type", "")),
                        "entity_id": str(row.get("entity_id", "")),
                        "used_new": float(row.get("used_new", 0) or 0),
                        "used_base": float(row.get("used_base", 0) or 0),
                        "used_total": float(row.get("used_total", 0) or 0),
                        "limit": float(row.get("limit", 0) or 0),
                        "utilization_pct": float(row.get("utilization_pct", 0) or 0),
                    }
                )
            return self._pagination(normalized_rows, total=paged["total"], limit=limit, offset=offset)

        process_dir = Path(record["process_dir"])
        file_path = process_dir / "weekly_exposure.xlsx"

        if file_path.exists():
            df = self._safe_excel_read(file_path)
            if entity_type and "entity_type" in df.columns:
                df = df[self._contains_case_insensitive(df["entity_type"], entity_type)]
            if entity_id and "entity_id" in df.columns:
                df = df[self._contains_case_insensitive(df["entity_id"], entity_id)]
            if week_start and "week_start" in df.columns:
                df = df[self._contains_case_insensitive(df["week_start"], week_start)]

            df = self._stable_sort(df, ["week_start", "entity_type", "entity_id"])
            total = len(df)
            page = df.iloc[offset : offset + limit]
            rows = []
            for _, row in page.iterrows():
                payload = {
                    "week_start": str(row.get("week_start", "")) if pd.notna(row.get("week_start")) else "",
                    "entity_type": str(row.get("entity_type", "")) if pd.notna(row.get("entity_type")) else "",
                    "entity_id": str(row.get("entity_id", "")) if pd.notna(row.get("entity_id")) else "",
                    "used_new": float(row.get("used_new")) if pd.notna(row.get("used_new")) else 0.0,
                    "used_base": float(row.get("used_base")) if pd.notna(row.get("used_base")) else 0.0,
                    "used_total": float(row.get("used_total")) if pd.notna(row.get("used_total")) else 0.0,
                    "limit": float(row.get("limit")) if pd.notna(row.get("limit")) else 0.0,
                    "utilization_pct": float(row.get("utilization_pct")) if pd.notna(row.get("utilization_pct")) else 0.0,
                }
                rows.append(payload)
            return self._pagination(rows, total=total, limit=limit, offset=offset)

        # Fallback to metadata JSON when xlsx is missing.
        metadata = self.get_results(process_id) or {}
        rows = metadata.get("weekly_exposure", [])
        if entity_type:
            rows = [row for row in rows if entity_type.lower() in str(row.get("entity_type", "")).lower()]
        if entity_id:
            rows = [row for row in rows if entity_id.lower() in str(row.get("entity_id", "")).lower()]
        if week_start:
            rows = [row for row in rows if week_start.lower() in str(row.get("week_start", "")).lower()]
        rows = sorted(
            rows,
            key=lambda row: (
                str(row.get("week_start", "")),
                str(row.get("entity_type", "")),
                str(row.get("entity_id", "")),
            ),
        )
        total = len(rows)
        page = rows[offset : offset + limit]
        normalized_rows = []
        for row in page:
            normalized_rows.append(
                {
                    "week_start": str(row.get("week_start", "")),
                    "entity_type": str(row.get("entity_type", "")),
                    "entity_id": str(row.get("entity_id", "")),
                    "used_new": float(row.get("used_new", 0) or 0),
                    "used_base": float(row.get("used_base", 0) or 0),
                    "used_total": float(row.get("used_total", 0) or 0),
                    "limit": float(row.get("limit", 0) or 0),
                    "utilization_pct": float(row.get("utilization_pct", 0) or 0),
                }
            )
        return self._pagination(normalized_rows, total=total, limit=limit, offset=offset)

    def find_invoice_decisions(
        self,
        process_id: str,
        invoice_reference: str,
        match_mode: str = "contains",
        max_matches: int = 20,
    ) -> Dict[str, Any]:
        """Find selected/excluded invoice decisions by reference."""
        record = self.store.get_process_head(process_id)
        if record is None:
            return {"matches": [], "total": 0}

        process_dir = Path(record["process_dir"])
        query = (invoice_reference or "").strip().lower()
        mode = (match_mode or "contains").strip().lower()
        max_matches = max(1, min(int(max_matches), 100))

        if not query:
            return {"matches": [], "total": 0}

        def _matches(ref: str) -> bool:
            candidate = (ref or "").lower()
            if mode == "exact":
                return candidate == query
            if mode == "startswith":
                return candidate.startswith(query)
            return query in candidate

        matches: List[Dict[str, Any]] = []

        excluded_df = self._load_bucket_df(process_id, process_dir, "excluded")
        for _, row in excluded_df.iterrows():
            payload = self._invoice_row_payload(dict(row))
            if not _matches(payload["invoice_ref"]):
                continue
            payload["decision"] = "excluded"
            matches.append(payload)

        selected_df = self._load_bucket_df(process_id, process_dir, "selected")
        for _, row in selected_df.iterrows():
            payload = self._invoice_row_payload(dict(row))
            if not _matches(payload["invoice_ref"]):
                continue
            payload["decision"] = "selected"
            matches.append(payload)

        matches.sort(
            key=lambda item: (
                0 if item.get("decision") == "excluded" else 1,
                str(item.get("invoice_ref", "")),
            )
        )

        total = len(matches)
        return {"matches": matches[:max_matches], "total": total}

    def get_invoices(
        self,
        process_id: str,
        invoice_type: str = "selected",
        limit: int = 50,
        offset: int = 0,
        stage_filter: Optional[str] = None,
        reason_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load selected, excluded, or weekly-plan invoices with pagination.

        Args:
            invoice_type: "selected", "excluded", or "weekly_plan"
            limit: Page size
            offset: Page offset
            stage_filter: Filter excluded by "rule" or "optimizer"
            reason_filter: Filter excluded by reason substring
        """
        paged = self.get_invoice_rows(
            process_id=process_id,
            bucket=invoice_type,
            excluded_stage=stage_filter,
            excluded_reason=reason_filter,
            limit=limit,
            offset=offset,
        )
        return {
            "invoices": paged["rows"],
            "total": paged["total"],
            "limit": paged["limit"],
            "offset": paged["offset"],
        }

    def get_weekly_exposure(
        self,
        process_id: str,
        limit: int = 500,
        offset: int = 0,
        entity_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load weekly exposure rows with pagination."""
        paged = self.get_weekly_exposure_rows(
            process_id=process_id,
            limit=limit,
            offset=offset,
            entity_type=entity_type,
        )
        return {
            "rows": paged["rows"],
            "total": paged["total"],
            "limit": paged["limit"],
            "offset": paged["offset"],
        }

    def get_limits(self, process_id: str) -> Dict[str, Any]:
        record = self.store.get_process_head(process_id)
        if record is None:
            raise ValueError(f"Process {process_id} not found")

        limits_path = Path(record["process_dir"]) / "limits.yaml"
        if limits_path.exists():
            with limits_path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}

        text = self.artifact_store.get_text_artifact(process_id, "limits.yaml")
        if text is None:
            return {}
        return yaml.safe_load(text) or {}

    def update_limits(self, process_id: str, limits_dict: Dict[str, Any]) -> Dict[str, Any]:
        record = self.store.get_process(process_id)
        if record is None:
            raise ValueError(f"Process {process_id} not found")

        limits_path = Path(record["process_dir"]) / "limits.yaml"
        limits_path.parent.mkdir(parents=True, exist_ok=True)
        with limits_path.open("w", encoding="utf-8") as f:
            yaml.dump(limits_dict, f, default_flow_style=False)

        self._store_text_artifact_from_path(
            process_id,
            "limits.yaml",
            limits_path,
            artifact_kind="config/yaml",
        )
        self.store.update_process(process_id, status="configuring")
        return limits_dict

    def get_rules(self, process_id: str) -> Dict[str, Any]:
        record = self.store.get_process_head(process_id)
        if record is None:
            raise ValueError(f"Process {process_id} not found")

        rules_path = Path(record["process_dir"]) / "rules.yaml"
        if rules_path.exists():
            with rules_path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {"rules": []}

        text = self.artifact_store.get_text_artifact(process_id, "rules.yaml")
        if text is None:
            return {"rules": []}
        return yaml.safe_load(text) or {"rules": []}

    def update_rules(self, process_id: str, rules_dict: Dict[str, Any]) -> Dict[str, Any]:
        record = self.store.get_process(process_id)
        if record is None:
            raise ValueError(f"Process {process_id} not found")

        rules_path = Path(record["process_dir"]) / "rules.yaml"
        rules_path.parent.mkdir(parents=True, exist_ok=True)
        with rules_path.open("w", encoding="utf-8") as f:
            yaml.dump(rules_dict, f, default_flow_style=False)

        self._store_text_artifact_from_path(
            process_id,
            "rules.yaml",
            rules_path,
            artifact_kind="config/yaml",
        )
        self.store.update_process(process_id, status="configuring")
        return rules_dict

    def update_params(self, process_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        record = self.store.get_process(process_id)
        if record is None:
            raise ValueError(f"Process {process_id} not found")

        updates = {}
        if "cohort" in params:
            updates["cohort"] = params["cohort"]
        if "cohort_match_granularity" in params:
            updates["cohort_match_granularity"] = params["cohort_match_granularity"]
        if "sheet_name" in params:
            updates["sheet_name"] = params["sheet_name"]
        if "release_event" in params:
            updates["release_event"] = params["release_event"]
        if "release_event_mode" in params:
            updates["release_event_mode"] = params["release_event_mode"]
        if "planning_mode" in params:
            updates["planning_mode"] = params["planning_mode"]
        if "planning_start_date" in params:
            updates["planning_start_date"] = params["planning_start_date"]
        if "horizon_weeks" in params:
            updates["horizon_weeks"] = int(params["horizon_weeks"])
        if "attempt_cap" in params:
            updates["attempt_cap"] = int(params["attempt_cap"])
        if "source_profile" in params:
            updates["source_profile"] = params["source_profile"]
        if "lifecycle_input_path" in params:
            updates["lifecycle_input_path"] = params["lifecycle_input_path"]
        if "solver_max_time_seconds" in params:
            updates["solver_max_time_seconds"] = int(params["solver_max_time_seconds"])
        if "solver_random_seed" in params:
            updates["solver_random_seed"] = int(params["solver_random_seed"])
        if "solver_num_search_workers" in params:
            updates["solver_num_search_workers"] = int(params["solver_num_search_workers"])

        if updates:
            updates["status"] = "configuring"
            self.store.update_process(process_id, **updates)

        return self.store.get_process(process_id)

    def get_cohorts(self, process_id: str) -> List[Dict[str, Any]]:
        record = self.store.get_process(process_id)
        if record is None:
            raise ValueError(f"Process {process_id} not found")

        process_dir = Path(record["process_dir"])
        extraction_path = process_dir / "extraction.xlsx"
        sheet_name = record.get("sheet_name") or "SAPUI5 Export"
        return detect_cohorts(extraction_path, sheet_name)

    def get_file_path(self, process_id: str, file_type: str) -> Optional[Path]:
        """Get path to a downloadable file.

        file_type: "selected", "excluded", "summary", "pdf", "docx",
                   "weekly_plan", "weekly_exposure"
        """
        record = self.store.get_process_head(process_id)
        if record is None:
            return None

        process_dir = Path(record["process_dir"])
        mapping = {
            "selected": process_dir / "selected.xlsx",
            "excluded": process_dir / "excluded.xlsx",
            "summary": process_dir / "run_summary.md",
            "pdf": process_dir / "run_summary.pdf",
            "docx": process_dir / "run_summary.docx",
            "weekly_plan": process_dir / "weekly_plan.xlsx",
            "weekly_exposure": process_dir / "weekly_exposure.xlsx",
        }

        path = mapping.get(file_type)
        if path is None:
            return None

        if path.exists():
            return path

        if file_type == "summary":
            summary_path = self._ensure_text_file(process_id, process_dir, "run_summary.md")
            if summary_path is not None:
                return summary_path
            return self._materialize_report_files(process_id).get("summary")
        if file_type in ("selected", "excluded", "weekly_plan", "weekly_exposure"):
            return self._materialize_bucket_workbook(process_id, file_type)
        if file_type in ("pdf", "docx"):
            materialized = self._materialize_report_files(process_id)
            return materialized.get(file_type)
        return None

    def generate_report_zip(self, process_id: str) -> Optional[Path]:
        """Generate a ZIP bundle with all generated output files."""
        record = self.store.get_process_head(process_id)
        if record is None:
            return None

        process_dir = Path(record["process_dir"])
        zip_root = process_dir if process_dir.exists() else self._materialized_process_dir(process_id)
        zip_path = zip_root / "report.zip"

        files_to_include: List[Path] = []

        for workbook in sorted(process_dir.glob("*.xlsx")):
            if workbook.name == "extraction.xlsx" or workbook.name.startswith("~$"):
                continue
            files_to_include.append(workbook)

        for artifact_type in ("selected", "excluded", "weekly_plan", "weekly_exposure"):
            path = self.get_file_path(process_id, artifact_type)
            if path and path not in files_to_include:
                files_to_include.append(path)

        summary_path = self.get_file_path(process_id, "summary")
        pdf_path = self.get_file_path(process_id, "pdf")
        docx_path = self.get_file_path(process_id, "docx")
        for path in (summary_path, pdf_path, docx_path):
            if path and path not in files_to_include:
                files_to_include.append(path)

        if not files_to_include:
            return None

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files_to_include:
                zf.write(f, f.name)

        return zip_path

    def list_uploads(self) -> List[Dict[str, str]]:
        """List existing extraction files (for agent to reference)."""
        uploads = []
        for rec in self.store.list_processes(limit=500):
            fname = rec.get("extraction_filename")
            if fname and fname not in [u["filename"] for u in uploads]:
                uploads.append({
                    "filename": fname,
                    "process_id": rec["id"],
                })
        return uploads
