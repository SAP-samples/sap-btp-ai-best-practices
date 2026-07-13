"""Runtime configuration for the metal composition service."""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from .project_paths import API_ROOT, PROJECT_ROOT

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_path(raw_value: str, *, base_dir: Path) -> Path:
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path


@dataclass(frozen=True)
class MetalCompositionSettings:
    workbook_path: Path
    api_env_path: Path
    uploaded_document_root: Path = API_ROOT / ".cache" / "metal_composition" / "uploads"
    ui_state_db_path: Path = PROJECT_ROOT / "data" / "metal_composition_ui_state.sqlite3"
    ui_state_hana_schema: str = ""
    ui_state_document_assignments_table: str = "METAL_COMPOSITION_UI_DOCUMENT_ASSIGNMENTS"
    ui_state_app_settings_table: str = "METAL_COMPOSITION_UI_APP_SETTINGS"
    ui_state_classification_history_table: str = "METAL_COMPOSITION_UI_CLASSIFICATION_HISTORY"
    ui_state_classification_jobs_table: str = "METAL_COMPOSITION_UI_CLASSIFICATION_JOBS"
    ui_state_classification_job_items_table: str = "METAL_COMPOSITION_UI_CLASSIFICATION_JOB_ITEMS"
    ui_state_classification_ownership_table: str = "METAL_COMPOSITION_UI_CLASSIFICATION_OWNERSHIP"
    section_232_hana_schema: str = ""
    section_232_sources_table: str = "METAL_COMPOSITION_SECTION232_SOURCES"
    section_232_hts_codes_table: str = "METAL_COMPOSITION_SECTION232_HTS_CODES"
    section_232_draft_batches_table: str = "METAL_COMPOSITION_SECTION232_DRAFT_BATCHES"
    section_232_draft_rules_table: str = "METAL_COMPOSITION_SECTION232_DRAFT_RULES"
    section_232_rulesets_table: str = "METAL_COMPOSITION_SECTION232_RULESETS"
    section_232_ruleset_rules_table: str = "METAL_COMPOSITION_SECTION232_RULESET_RULES"
    section_232_delete_overrides_table: str = "METAL_COMPOSITION_SECTION232_DELETE_OVERRIDES"
    data_source: str = "hana"
    hana_schema: str = ""
    hana_table: str = "METAL_COMPOSITION_SERVING"
    cache_dir: Path = API_ROOT / ".cache" / "metal_composition"
    sheet_name: str = "Material Master"
    request_timeout_seconds: int = 30
    prewarm_on_startup: bool = True
    image_max_dimension: int = 2048
    image_max_bytes: int = 1048576
    pdf_image_max_dimension: int = 3072
    pdf_image_max_bytes: int = 4 * 1024 * 1024
    diagram_model_name: str = "gpt-5"  # alternatives: gemini-2.5-pro, anthropic--claude-4.6-sonnet
    diagram_page_routing_enabled: bool = True
    diagram_page_routing_fallback_model_name: str = "gpt-4o-mini"
    diagram_page_routing_skip_enabled: bool = True
    diagram_page_routing_preview_dpi: int = 48
    diagram_page_routing_fallback_render_dpi: int = 100
    hts_fact_profile_model_name: str = "gpt-5"  # alternative: anthropic--claude-4.6-sonnet
    hana_tree_router_model_name: str = "gpt-5"  # alternative: anthropic--claude-4.6-sonnet
    trade_decision_model_name: str = "gpt-5"  # HTS selector inside trade_decision
    section_232_model_name: str = "gpt-4.1"  # Section 232 reasoner inside trade_decision
    hts_enabled: bool = True
    hts_hana_schema: str = ""
    hts_catalog_dir: Path = PROJECT_ROOT / "data" / "hts_chapters"
    hts_catalog_hana_table: str = "HTS_2026_CATALOG"
    hts_code_map_path: Path = PROJECT_ROOT / "data" / "hts_chapters" / "hts_code_map.csv"
    hts_code_map_hana_table: str = "HTS_2026_CODE_MAP"
    hts_catalog_sources_table: str = "HTS_2026_SOURCE_FILES"
    hts_catalog_status_table: str = "HTS_2026_SOURCE_STATUS"
    hts_k_candidates: int = 5
    # Anthropic thinking configuration (used when model is anthropic--*)
    # reasoning_type: "adaptive" (Claude decides), "enabled" (fixed budget), "disabled" (no thinking)
    # reasoning_budget: only used when reasoning_type is "enabled"
    composition_reasoning_type: str = "adaptive"
    composition_reasoning_budget: int = 4000
    hts_fact_profile_reasoning_type: str = "adaptive"
    hts_fact_profile_reasoning_budget: int = 4000
    hana_tree_router_reasoning_type: str = "adaptive"
    hana_tree_router_reasoning_budget: int = 4000
    trade_decision_reasoning_type: str = "adaptive"
    trade_decision_reasoning_budget: int = 8000
    pdf_render_dpi: int = 300
    diagram_zoom_enabled: bool = True
    diagram_zoom_max_requests: int = 6
    diagram_zoom_render_dpi: int = 600
    diagram_zoom_padding_ratio: float = 0.03
    diagram_zoom_image_max_dimension: int = 4096
    diagram_zoom_image_max_bytes: int = 2 * 1024 * 1024
    max_diagram_images: int = 500
    max_diagram_payload_bytes: int = 50 * 1024 * 1024  # 50 MB
    max_diagram_text_chars_per_batch: int = 120000
    max_diagram_text_chars_per_page_chunk: int = 30000
    diagram_batch_max_concurrency: int = 2
    diagram_reasoning_type: str = "adaptive"
    diagram_reasoning_budget: int = 4000
    # Batch classification is memory- and network-heavy because each item can
    # render PDFs and invoke several downstream model/tool calls.
    batch_max_concurrency: int = 2
    batch_max_items: int = 25
    classification_job_worker_max_concurrency: int = 10
    classification_job_poll_interval_seconds: float = 2.0
    section_232_max_sources: int = 5
    section_232_max_pages_per_source: int = 3
    section_232_max_prompt_chars: int = 120000


@lru_cache(maxsize=1)
def get_settings() -> MetalCompositionSettings:
    """Load service settings from the API environment."""

    api_env_path = API_ROOT / ".env"
    load_dotenv(api_env_path, override=False)
    env_base_dir = api_env_path.parent

    workbook_path = _resolve_path(
        os.getenv(
            "METAL_COMPOSITION_WORKBOOK_PATH",
            str(PROJECT_ROOT / "data" / "Material Master.xlsb"),
        ),
        base_dir=env_base_dir,
    )
    hts_catalog_dir = _resolve_path(
        os.getenv(
            "METAL_COMPOSITION_HTS_CATALOG_DIR",
            str(PROJECT_ROOT / "data" / "hts_chapters"),
        ),
        base_dir=env_base_dir,
    )
    hts_code_map_path = _resolve_path(
        os.getenv(
            "METAL_COMPOSITION_HTS_CODE_MAP_PATH",
            str(PROJECT_ROOT / "data" / "hts_chapters" / "hts_code_map.csv"),
        ),
        base_dir=env_base_dir,
    )
    return MetalCompositionSettings(
        workbook_path=workbook_path,
        api_env_path=api_env_path.resolve(),
        uploaded_document_root=_resolve_path(
            os.getenv(
                "METAL_COMPOSITION_UPLOADED_DOCUMENT_ROOT",
                str(API_ROOT / ".cache" / "metal_composition" / "uploads"),
            ),
            base_dir=env_base_dir,
        ),
        ui_state_db_path=_resolve_path(
            os.getenv(
                "METAL_COMPOSITION_UI_STATE_DB",
                str(PROJECT_ROOT / "data" / "metal_composition_ui_state.sqlite3"),
            ),
            base_dir=env_base_dir,
        ),
        ui_state_hana_schema=os.getenv(
            "METAL_COMPOSITION_UI_STATE_HANA_SCHEMA",
            os.getenv("METAL_COMPOSITION_HANA_SCHEMA", ""),
        ).strip(),
        ui_state_document_assignments_table=os.getenv(
            "METAL_COMPOSITION_UI_STATE_DOCUMENT_ASSIGNMENTS_TABLE",
            "METAL_COMPOSITION_UI_DOCUMENT_ASSIGNMENTS",
        ).strip(),
        ui_state_app_settings_table=os.getenv(
            "METAL_COMPOSITION_UI_STATE_APP_SETTINGS_TABLE",
            "METAL_COMPOSITION_UI_APP_SETTINGS",
        ).strip(),
        ui_state_classification_history_table=os.getenv(
            "METAL_COMPOSITION_UI_STATE_CLASSIFICATION_HISTORY_TABLE",
            "METAL_COMPOSITION_UI_CLASSIFICATION_HISTORY",
        ).strip(),
        ui_state_classification_jobs_table=os.getenv(
            "METAL_COMPOSITION_UI_STATE_CLASSIFICATION_JOBS_TABLE",
            "METAL_COMPOSITION_UI_CLASSIFICATION_JOBS",
        ).strip(),
        ui_state_classification_job_items_table=os.getenv(
            "METAL_COMPOSITION_UI_STATE_CLASSIFICATION_JOB_ITEMS_TABLE",
            "METAL_COMPOSITION_UI_CLASSIFICATION_JOB_ITEMS",
        ).strip(),
        ui_state_classification_ownership_table=os.getenv(
            "METAL_COMPOSITION_UI_STATE_CLASSIFICATION_OWNERSHIP_TABLE",
            "METAL_COMPOSITION_UI_CLASSIFICATION_OWNERSHIP",
        ).strip(),
        section_232_hana_schema=os.getenv(
            "METAL_COMPOSITION_SECTION232_HANA_SCHEMA",
            os.getenv("METAL_COMPOSITION_UI_STATE_HANA_SCHEMA", os.getenv("METAL_COMPOSITION_HANA_SCHEMA", "")),
        ).strip(),
        section_232_sources_table=os.getenv(
            "METAL_COMPOSITION_SECTION232_SOURCES_TABLE",
            "METAL_COMPOSITION_SECTION232_SOURCES",
        ).strip(),
        section_232_hts_codes_table=os.getenv(
            "METAL_COMPOSITION_SECTION232_HTS_CODES_TABLE",
            "METAL_COMPOSITION_SECTION232_HTS_CODES",
        ).strip(),
        section_232_draft_batches_table=os.getenv(
            "METAL_COMPOSITION_SECTION232_DRAFT_BATCHES_TABLE",
            "METAL_COMPOSITION_SECTION232_DRAFT_BATCHES",
        ).strip(),
        section_232_draft_rules_table=os.getenv(
            "METAL_COMPOSITION_SECTION232_DRAFT_RULES_TABLE",
            "METAL_COMPOSITION_SECTION232_DRAFT_RULES",
        ).strip(),
        section_232_rulesets_table=os.getenv(
            "METAL_COMPOSITION_SECTION232_RULESETS_TABLE",
            "METAL_COMPOSITION_SECTION232_RULESETS",
        ).strip(),
        section_232_ruleset_rules_table=os.getenv(
            "METAL_COMPOSITION_SECTION232_RULESET_RULES_TABLE",
            "METAL_COMPOSITION_SECTION232_RULESET_RULES",
        ).strip(),
        section_232_delete_overrides_table=os.getenv(
            "METAL_COMPOSITION_SECTION232_DELETE_OVERRIDES_TABLE",
            "METAL_COMPOSITION_SECTION232_DELETE_OVERRIDES",
        ).strip(),
        data_source=os.getenv("METAL_COMPOSITION_DATA_SOURCE", "hana").strip().lower(),
        hana_schema=os.getenv("METAL_COMPOSITION_HANA_SCHEMA", "").strip(),
        hana_table=os.getenv("METAL_COMPOSITION_HANA_TABLE", "METAL_COMPOSITION_SERVING").strip(),
        cache_dir=_resolve_path(
            os.getenv(
                "METAL_COMPOSITION_CACHE_DIR",
                str(API_ROOT / ".cache" / "metal_composition"),
            ),
            base_dir=env_base_dir,
        ),
        sheet_name=os.getenv("METAL_COMPOSITION_SHEET_NAME", "Material Master"),
        request_timeout_seconds=int(
            os.getenv("METAL_COMPOSITION_REQUEST_TIMEOUT_SECONDS", "30")
        ),
        prewarm_on_startup=_env_bool("METAL_COMPOSITION_PREWARM_ON_STARTUP", True),
        image_max_dimension=int(os.getenv("METAL_COMPOSITION_IMAGE_MAX_DIMENSION", "2048")),
        image_max_bytes=int(os.getenv("METAL_COMPOSITION_IMAGE_MAX_BYTES", "1048576")),
        pdf_image_max_dimension=int(
            os.getenv("METAL_COMPOSITION_PDF_IMAGE_MAX_DIMENSION", "3072")
        ),
        pdf_image_max_bytes=int(
            os.getenv("METAL_COMPOSITION_PDF_IMAGE_MAX_BYTES", str(4 * 1024 * 1024))
        ),
        diagram_model_name=os.getenv(
            "METAL_COMPOSITION_DIAGRAM_MODEL_NAME",
            "gpt-5",  # alternatives: gemini-2.5-pro, anthropic--claude-4.6-sonnet
        ).strip(),
        diagram_page_routing_enabled=_env_bool(
            "METAL_COMPOSITION_DIAGRAM_PAGE_ROUTING_ENABLED", True
        ),
        diagram_page_routing_fallback_model_name=os.getenv(
            "METAL_COMPOSITION_DIAGRAM_PAGE_ROUTING_FALLBACK_MODEL_NAME",
            "gpt-4o-mini",
        ).strip(),
        diagram_page_routing_skip_enabled=_env_bool(
            "METAL_COMPOSITION_DIAGRAM_PAGE_ROUTING_SKIP_ENABLED", True
        ),
        diagram_page_routing_preview_dpi=int(
            os.getenv("METAL_COMPOSITION_DIAGRAM_PAGE_ROUTING_PREVIEW_DPI", "48")
        ),
        diagram_page_routing_fallback_render_dpi=int(
            os.getenv("METAL_COMPOSITION_DIAGRAM_PAGE_ROUTING_FALLBACK_RENDER_DPI", "100")
        ),
        hts_fact_profile_model_name=os.getenv(
            "METAL_COMPOSITION_HTS_FACT_PROFILE_MODEL_NAME",
            "gpt-5",  # alternative: anthropic--claude-4.6-sonnet
        ).strip(),
        hana_tree_router_model_name=os.getenv(
            "METAL_COMPOSITION_HANA_TREE_ROUTER_MODEL_NAME",
            "gpt-5",  # alternative: anthropic--claude-4.6-sonnet
        ).strip(),
        trade_decision_model_name=os.getenv(
            "METAL_COMPOSITION_HTS_SELECTOR_MODEL_NAME",
            "gpt-5",  # alternative: anthropic--claude-4.6-sonnet
        ).strip(),
        section_232_model_name=os.getenv(
            "METAL_COMPOSITION_SECTION_232_REASONER_MODEL_NAME",
            "gpt-4.1",
        ).strip(),
        hts_enabled=_env_bool("METAL_COMPOSITION_HTS_ENABLED", True),
        hts_hana_schema=os.getenv("METAL_COMPOSITION_HTS_HANA_SCHEMA", "").strip(),
        hts_catalog_dir=hts_catalog_dir,
        hts_catalog_hana_table=os.getenv("METAL_COMPOSITION_HTS_CATALOG_HANA_TABLE", "HTS_2026_CATALOG").strip(),
        hts_code_map_path=hts_code_map_path,
        hts_code_map_hana_table=os.getenv("METAL_COMPOSITION_HTS_CODE_MAP_HANA_TABLE", "HTS_2026_CODE_MAP").strip(),
        hts_catalog_sources_table=os.getenv(
            "METAL_COMPOSITION_HTS_CATALOG_SOURCES_TABLE",
            "HTS_2026_SOURCE_FILES",
        ).strip(),
        hts_catalog_status_table=os.getenv(
            "METAL_COMPOSITION_HTS_CATALOG_STATUS_TABLE",
            "HTS_2026_SOURCE_STATUS",
        ).strip(),
        hts_k_candidates=max(1, int(os.getenv("METAL_COMPOSITION_HTS_K_CANDIDATES", "5"))),
        composition_reasoning_type=os.getenv(
            "METAL_COMPOSITION_COMPOSITION_REASONING_TYPE", "adaptive"
        ).strip().lower(),
        composition_reasoning_budget=int(
            os.getenv("METAL_COMPOSITION_COMPOSITION_REASONING_BUDGET", "4000")
        ),
        hts_fact_profile_reasoning_type=os.getenv(
            "METAL_COMPOSITION_HTS_FACT_PROFILE_REASONING_TYPE", "adaptive"
        ).strip().lower(),
        hts_fact_profile_reasoning_budget=int(
            os.getenv("METAL_COMPOSITION_HTS_FACT_PROFILE_REASONING_BUDGET", "4000")
        ),
        hana_tree_router_reasoning_type=os.getenv(
            "METAL_COMPOSITION_HANA_TREE_ROUTER_REASONING_TYPE", "adaptive"
        ).strip().lower(),
        hana_tree_router_reasoning_budget=int(
            os.getenv("METAL_COMPOSITION_HANA_TREE_ROUTER_REASONING_BUDGET", "4000")
        ),
        trade_decision_reasoning_type=os.getenv(
            "METAL_COMPOSITION_TRADE_DECISION_REASONING_TYPE", "adaptive"
        ).strip().lower(),
        trade_decision_reasoning_budget=int(
            os.getenv("METAL_COMPOSITION_TRADE_DECISION_REASONING_BUDGET", "8000")
        ),
        diagram_reasoning_type=os.getenv(
            "METAL_COMPOSITION_DIAGRAM_REASONING_TYPE", "adaptive"
        ).strip().lower(),
        diagram_reasoning_budget=int(
            os.getenv("METAL_COMPOSITION_DIAGRAM_REASONING_BUDGET", "4000")
        ),
        pdf_render_dpi=int(
            os.getenv("METAL_COMPOSITION_PDF_RENDER_DPI", "300")
        ),
        diagram_zoom_enabled=_env_bool(
            "METAL_COMPOSITION_DIAGRAM_ZOOM_ENABLED", True
        ),
        diagram_zoom_max_requests=int(
            os.getenv("METAL_COMPOSITION_DIAGRAM_ZOOM_MAX_REQUESTS", "6")
        ),
        diagram_zoom_render_dpi=int(
            os.getenv("METAL_COMPOSITION_DIAGRAM_ZOOM_RENDER_DPI", "600")
        ),
        diagram_zoom_padding_ratio=float(
            os.getenv("METAL_COMPOSITION_DIAGRAM_ZOOM_PADDING_RATIO", "0.03")
        ),
        diagram_zoom_image_max_dimension=int(
            os.getenv("METAL_COMPOSITION_DIAGRAM_ZOOM_IMAGE_MAX_DIMENSION", "4096")
        ),
        diagram_zoom_image_max_bytes=int(
            os.getenv("METAL_COMPOSITION_DIAGRAM_ZOOM_IMAGE_MAX_BYTES", str(2 * 1024 * 1024))
        ),
        max_diagram_images=int(
            os.getenv("METAL_COMPOSITION_MAX_DIAGRAM_IMAGES", "500")
        ),
        max_diagram_payload_bytes=int(
            os.getenv("METAL_COMPOSITION_MAX_DIAGRAM_PAYLOAD_BYTES", str(50 * 1024 * 1024))
        ),
        max_diagram_text_chars_per_batch=int(
            os.getenv("METAL_COMPOSITION_MAX_DIAGRAM_TEXT_CHARS_PER_BATCH", "120000")
        ),
        max_diagram_text_chars_per_page_chunk=int(
            os.getenv("METAL_COMPOSITION_MAX_DIAGRAM_TEXT_CHARS_PER_PAGE_CHUNK", "30000")
        ),
        diagram_batch_max_concurrency=int(
            os.getenv("METAL_COMPOSITION_DIAGRAM_BATCH_MAX_CONCURRENCY", "2")
        ),
        batch_max_concurrency=int(
            os.getenv("METAL_COMPOSITION_BATCH_MAX_CONCURRENCY", "2")
        ),
        batch_max_items=int(
            os.getenv("METAL_COMPOSITION_BATCH_MAX_ITEMS", "25")
        ),
        classification_job_worker_max_concurrency=int(
            os.getenv("METAL_COMPOSITION_CLASSIFICATION_JOB_WORKER_MAX_CONCURRENCY", "10")
        ),
        classification_job_poll_interval_seconds=float(
            os.getenv("METAL_COMPOSITION_CLASSIFICATION_JOB_POLL_INTERVAL_SECONDS", "2")
        ),
        section_232_max_sources=int(
            os.getenv("METAL_COMPOSITION_SECTION232_MAX_SOURCES", "5")
        ),
        section_232_max_pages_per_source=int(
            os.getenv("METAL_COMPOSITION_SECTION232_MAX_PAGES_PER_SOURCE", "3")
        ),
        section_232_max_prompt_chars=int(
            os.getenv("METAL_COMPOSITION_SECTION232_MAX_PROMPT_CHARS", "120000")
        ),
    )
