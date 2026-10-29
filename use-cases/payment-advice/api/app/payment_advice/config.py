"""
Validated configuration for the Payment Advice Extractor (UC-01).

Unlike the reused DocumentAI-Agent (which loads its SAP Document AI service key
from an environment variable), this project loads the service key from a local
JSON file by default:

    api/app/payment_advice/schema/service_key.json   (git-ignored)

Environment variables still take precedence when present, so the eventual Cloud
Foundry deployment can inject the key without a file:

    DOCUMENT_AI_SERVICE_KEY_BASE64   base64-encoded service-key JSON (CF transport)
    DOCUMENT_AI_SERVICE_KEY_JSON     raw service-key JSON
    DOCUMENT_AI_SERVICE_KEY_PATH     override the default file path

Other settings:

    DOCUMENT_AI_CLIENT_ID    SAP Document AI client id (default: ai4u_payment_advice)
    PAYMENT_ADVICE_MAPPER_MODEL   canonical-mapping LLM (default: gpt-5.6-luna)
    PAYMENT_ADVICE_AGENT_MODEL    verifier/agent LLM     (default: gpt-5.6-luna)
    PAYMENT_ADVICE_OUT_DIR   output directory for canonical JSON (default: ./out)

SAP AI Core and SAP HANA credentials are read elsewhere (from ``api/.env`` via
``AICORE_*`` and ``HANA_*``); this module only owns Document AI + pipeline config.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from dox_client import ServiceKey

# Default SAP Document AI client id for the whole project (one id, schemas named per client).
DEFAULT_DOX_CLIENT_ID = "ai4u_payment_advice"

# Models allowed for the canonical mapper and the verifier agent.
# gpt-5.6-luna: cheap and reliable. gemini-3.1-flash-lite: cheaper and faster.
ALLOWED_MODELS = ("gpt-5.6-luna", "gemini-3.1-flash-lite")
DEFAULT_MODEL = "gpt-5.6-luna"

# SAP Document AI hard service limits that drive the splitter (UC-01).
MAX_PAGES = 100          # PDF/image pages per extraction job
MAX_LINE_ITEMS = 2000    # table rows per extraction job
MAX_COLUMNS = 49         # table columns per extraction job (not splittable -> refuse)

# Default location of the copied service-key file, relative to this module.
_DEFAULT_SERVICE_KEY_PATH = Path(__file__).resolve().parent / "schema" / "service_key.json"


class PaymentAdviceSettings(BaseModel):
    """
    Validated Document AI + pipeline settings for the extractor.

    Attributes:
        service_key_data: Parsed SAP Document AI service-key JSON (validated).
        dox_client_id: SAP Document AI client id used for all schema/job calls.
        mapper_model: LLM used to map raw extraction to the canonical schema.
        agent_model: LLM used by the verifier/ReAct agent.
        out_dir: Directory where canonical JSON + run logs are written.
        max_pages / max_line_items / max_columns: Document AI split thresholds.
    """

    service_key_data: dict[str, Any]
    dox_client_id: str = DEFAULT_DOX_CLIENT_ID
    mapper_model: str = DEFAULT_MODEL
    agent_model: str = DEFAULT_MODEL
    out_dir: str = "out"
    max_pages: int = MAX_PAGES
    max_line_items: int = MAX_LINE_ITEMS
    max_columns: int = MAX_COLUMNS

    @classmethod
    def from_env(cls, service_key_path: str | os.PathLike[str] | None = None) -> "PaymentAdviceSettings":
        """
        Build settings from environment variables and the service-key file.

        Args:
            service_key_path: Optional explicit path to the service-key JSON file.
                Falls back to ``DOCUMENT_AI_SERVICE_KEY_PATH`` then the packaged
                default. Ignored when a service key is supplied via environment.

        Returns:
            A validated ``PaymentAdviceSettings`` instance.

        Raises:
            ValueError: If no service key can be found, the JSON is invalid, or a
                configured model is not in ``ALLOWED_MODELS``.
        """
        service_key_data = cls._load_service_key(service_key_path)
        # Validate the key shape early so a bad key fails at startup, not mid-run.
        ServiceKey.from_json(service_key_data)

        dox_client_id = os.getenv("DOCUMENT_AI_CLIENT_ID", "").strip() or DEFAULT_DOX_CLIENT_ID
        mapper_model = cls._validated_model("PAYMENT_ADVICE_MAPPER_MODEL")
        agent_model = cls._validated_model("PAYMENT_ADVICE_AGENT_MODEL")
        out_dir = os.getenv("PAYMENT_ADVICE_OUT_DIR", "").strip() or "out"

        return cls(
            service_key_data=service_key_data,
            dox_client_id=dox_client_id,
            mapper_model=mapper_model,
            agent_model=agent_model,
            out_dir=out_dir,
        )

    @staticmethod
    def _load_service_key(service_key_path: str | os.PathLike[str] | None) -> dict[str, Any]:
        """
        Resolve the service-key JSON from env (preferred) or a local file.

        Precedence: DOCUMENT_AI_SERVICE_KEY_BASE64 -> DOCUMENT_AI_SERVICE_KEY_JSON
        -> file at (argument | DOCUMENT_AI_SERVICE_KEY_PATH | packaged default).
        """
        encoded = os.getenv("DOCUMENT_AI_SERVICE_KEY_BASE64", "").strip()
        if encoded:
            try:
                raw = base64.b64decode(encoded, validate=True).decode("utf-8")
            except (UnicodeDecodeError, ValueError) as exc:
                raise ValueError(
                    "DOCUMENT_AI_SERVICE_KEY_BASE64 must contain valid base64-encoded JSON"
                ) from exc
            return PaymentAdviceSettings._parse_service_key_json(raw)

        raw_env = os.getenv("DOCUMENT_AI_SERVICE_KEY_JSON", "").strip()
        if raw_env:
            return PaymentAdviceSettings._parse_service_key_json(raw_env)

        path = Path(
            service_key_path
            or os.getenv("DOCUMENT_AI_SERVICE_KEY_PATH", "").strip()
            or _DEFAULT_SERVICE_KEY_PATH
        )
        if not path.is_file():
            raise ValueError(
                "SAP Document AI service key not found. Provide DOCUMENT_AI_SERVICE_KEY_JSON/"
                f"BASE64, or place the key file at {path}."
            )
        return PaymentAdviceSettings._parse_service_key_json(path.read_text(encoding="utf-8"))

    @staticmethod
    def _parse_service_key_json(raw: str) -> dict[str, Any]:
        """Parse a service-key JSON string into a top-level object dict."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("SAP Document AI service key must contain valid JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("SAP Document AI service key must be a JSON object at the top level")
        return data

    @staticmethod
    def _validated_model(env_name: str) -> str:
        """Read a model name from env, defaulting and enforcing the allowlist."""
        model = os.getenv(env_name, "").strip() or DEFAULT_MODEL
        if model not in ALLOWED_MODELS:
            raise ValueError(f"{env_name} must be one of {ALLOWED_MODELS}, got {model!r}")
        return model
