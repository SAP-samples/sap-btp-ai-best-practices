"""Runtime settings of the order intake pipeline, read once from environment variables.

| Variable                       | Default | Meaning                                                        |
|--------------------------------|---------|----------------------------------------------------------------|
| ORDER_WORKERS                  | 4       | Orders processed by the agent concurrently (per app instance)  |
| ORDER_AGENT_TIMEOUT_SECONDS    | 900     | Max duration of one agent run                                  |
| ORDER_MAX_ATTEMPTS             | 2       | Agent attempts before an order is marked FAILED                |
| AUTO_POST_ENABLED              | true    | Post orders that pass the strict gate without a human          |
| CONFIDENCE_THRESHOLD           | 0.80    | Fields below it need review (requirements: 80 %)               |
| MAX_UPLOAD_MB                  | 25      | Max total size of one upload request                           |
| MAX_NATIVE_FILES               | 5       | Max PDFs/images sent natively to the model per order           |
| MAX_NATIVE_MB                  | 15      | Max total size of natively sent files per order                |
| INLINE_IMAGE_MIN_PX            | 150     | Inline e-mail images smaller (shorter side) are treated as logos |
| DEFAULT_ORDER_TYPE             | TA      | Sales document type (requirements: hard-coded TA)              |
| DEFAULT_DISTR_CHANNEL          | 10      | Distribution channel (requirements: hard-coded 10)             |
| DEFAULT_PO_TYPE                | WWW     | Customer purchase order type (requirements: hard-coded WWW)    |
| ORDER_DELETE_ENABLED           | true    | PoC only: allow deleting orders to re-test them; set false in production |
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


def _env(name: str, default: str) -> str:
    """Read an environment variable, treating empty strings as unset."""
    return os.getenv(name) or default


@dataclass(frozen=True)
class OrderSettings:
    """Immutable snapshot of the pipeline settings (see module docstring)."""

    workers: int
    agent_timeout_seconds: int
    max_attempts: int
    auto_post_enabled: bool
    confidence_threshold: float
    max_upload_mb: int
    max_native_files: int
    max_native_mb: int
    inline_image_min_px: int
    default_order_type: str
    default_distr_channel: str
    default_po_type: str
    order_delete_enabled: bool


@lru_cache(maxsize=1)
def get_settings() -> OrderSettings:
    """Build the settings from the environment (cached for the process lifetime).

    Returns:
        OrderSettings instance.
    """
    return OrderSettings(
        workers=int(_env("ORDER_WORKERS", "4")),
        agent_timeout_seconds=int(_env("ORDER_AGENT_TIMEOUT_SECONDS", "900")),
        max_attempts=int(_env("ORDER_MAX_ATTEMPTS", "2")),
        auto_post_enabled=_env("AUTO_POST_ENABLED", "true").lower() in {"1", "true", "yes"},
        confidence_threshold=float(_env("CONFIDENCE_THRESHOLD", "0.8")),
        max_upload_mb=int(_env("MAX_UPLOAD_MB", "25")),
        max_native_files=int(_env("MAX_NATIVE_FILES", "5")),
        max_native_mb=int(_env("MAX_NATIVE_MB", "15")),
        inline_image_min_px=int(_env("INLINE_IMAGE_MIN_PX", "150")),
        default_order_type=_env("DEFAULT_ORDER_TYPE", "TA"),
        default_distr_channel=_env("DEFAULT_DISTR_CHANNEL", "10"),
        default_po_type=_env("DEFAULT_PO_TYPE", "WWW"),
        order_delete_enabled=_env("ORDER_DELETE_ENABLED", "true").lower() in {"1", "true", "yes"},
    )
