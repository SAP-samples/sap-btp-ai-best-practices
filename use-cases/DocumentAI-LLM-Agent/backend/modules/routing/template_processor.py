"""
template_processor.py
---------------------
Reprocesses an invoice using a matched SAP Document AI template.
Bypasses the GenAI comparison flow completely.

CRITICAL: SAP Document AI requires BOTH schemaId/schemaName AND templateId
in the same POST request. Sending only templateId returns EP003.

Correct payload:
    {
        "schemaId": "cf8cc8a9-1eee-42d9-9a3e-507a61baac23",
        "templateId": "4be26082-fdde-4bd0-b739-abddb9284fb1",
        "clientId": "default",
        "documentType": "invoice",
        "receivedDate": "2026-05-20"
    }
"""

import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Any

import requests
from requests.exceptions import ConnectionError, HTTPError, Timeout

from modules.auth.get_token import AuthenticationError, TokenManager
from modules.invoice.process_invoice import (
    API_REQUEST_TIMEOUT,
    DEFAULT_CLIENT_ID,
    DEFAULT_DOCUMENT_TYPE,
    DEFAULT_SCHEMA_NAME,
    JOBS_ENDPOINT,
    MAX_POLLING_ATTEMPTS,
    POLLING_INTERVAL_SECONDS,
    SERVICE_BASE_URL,
    SUPPORTED_EXTENSIONS,
    TERMINAL_STATUSES,
    JobFailedError,
    PollingTimeoutError,
)
from utils.config_loader import load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default base schema ID (SAP_invoice_schema)
DEFAULT_SCHEMA_ID: str = "cf8cc8a9-1eee-42d9-9a3e-507a61baac23"

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
OUTPUT_DIR: Path = _PROJECT_ROOT / "output"
OUTPUT_ROUTING_DIR: Path = OUTPUT_DIR / "routing"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TemplateProcessingError(Exception):
    """Raised when template-based invoice processing fails."""

    pass


# ---------------------------------------------------------------------------
# TemplateInvoiceProcessor
# ---------------------------------------------------------------------------


class TemplateInvoiceProcessor:
    """
    Processes an invoice using a specific SAP Document AI template.

    IMPORTANT: SAP Document AI requires BOTH schemaId/schemaName AND templateId
    in the POST options payload. Omitting the schema causes EP003 error.

    Usage:
        processor = TemplateInvoiceProcessor()
        result, path = processor.process(
            pdf_path,
            template_id="TEMPLATE_ID",
            schema_name="SAP_invoice_schema",
        )
    """

    def __init__(self, token_manager: TokenManager | None = None) -> None:
        self._config = load_config()
        self._token_manager = token_manager or TokenManager(self._config)
        self._session = requests.Session()
        self._base_url = SERVICE_BASE_URL.rstrip("/")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_ROUTING_DIR.mkdir(parents=True, exist_ok=True)

        logger.debug(
            "TemplateInvoiceProcessor initialized. Base URL: %s", self._base_url
        )

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        token = self._token_manager.get_token()
        return {"Authorization": f"Bearer {token}"}

    # ------------------------------------------------------------------
    # Options builder — MUST include schema + template
    # ------------------------------------------------------------------

    def _build_options(
        self,
        template_id: str,
        schema_name: str = DEFAULT_SCHEMA_NAME,
        schema_id: str | None = None,
        client_id: str = DEFAULT_CLIENT_ID,
        document_type: str = DEFAULT_DOCUMENT_TYPE,
    ) -> dict[str, Any]:
        """
        Build the SAP Document AI options payload.

        CRITICAL: Must include BOTH a schema reference AND templateId.
        SAP returns EP003 if schema is missing.

        Priority:
          - If schema_id is provided → use schemaId
          - Otherwise → use schemaName
        """
        if not schema_id and not schema_name:
            raise TemplateProcessingError(
                "Schema ID or Schema Name is required for template processing. "
                "SAP Document AI returns EP003 without a schema reference."
            )

        options: dict[str, Any] = {
            "templateId": template_id,
            "clientId": client_id,
            "documentType": document_type,
            "receivedDate": date.today().isoformat(),
        }

        if schema_id:
            options["schemaId"] = schema_id
            logger.debug("Using schemaId: %s", schema_id)
        else:
            options["schemaName"] = schema_name
            logger.debug("Using schemaName: %s", schema_name)

        return options

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    def submit_with_template(
        self,
        pdf_path: Path,
        template_id: str,
        schema_name: str = DEFAULT_SCHEMA_NAME,
        schema_id: str | None = None,
        client_id: str = DEFAULT_CLIENT_ID,
        document_type: str = DEFAULT_DOCUMENT_TYPE,
    ) -> str:
        """
        Submit an invoice to SAP Document AI using schema + template.

        Args:
            pdf_path: Path to the invoice file.
            template_id: SAP Document AI template ID.
            schema_name: Base schema name (used if schema_id not provided).
            schema_id: Base schema ID (takes priority over schema_name).
            client_id: Document AI client ID.
            document_type: Document type.

        Returns:
            job_id (str) of the created job.

        Raises:
            TemplateProcessingError: On validation, file, or HTTP errors.
            AuthenticationError: If the token is rejected.
        """
        if not pdf_path.exists():
            raise TemplateProcessingError(
                f"Invoice file not found: {pdf_path.resolve()}"
            )

        ext = pdf_path.suffix.lower()
        mime_type = SUPPORTED_EXTENSIONS.get(ext)
        if not mime_type:
            supported = ", ".join(SUPPORTED_EXTENSIONS.keys())
            raise TemplateProcessingError(
                f"Unsupported file format: '{ext}'. Supported: {supported}"
            )

        url = f"{self._base_url}{JOBS_ENDPOINT}"
        options = self._build_options(
            template_id=template_id,
            schema_name=schema_name,
            schema_id=schema_id,
            client_id=client_id,
            document_type=document_type,
        )

        logger.info(
            "Reprocessing invoice using schema + template. File: %s",
            pdf_path.name,
        )
        logger.info(
            "  schemaId=%s | schemaName=%s | templateId=%s",
            options.get("schemaId", "—"),
            options.get("schemaName", "—"),
            template_id,
        )
        logger.info("  Full options: %s", json.dumps(options))

        try:
            with open(pdf_path, "rb") as pdf_file:
                response = self._session.post(
                    url,
                    headers=self._auth_headers(),
                    files={
                        "file": (pdf_path.name, pdf_file, mime_type),
                        "options": (None, json.dumps(options), "application/json"),
                    },
                    timeout=API_REQUEST_TIMEOUT,
                )
            response.raise_for_status()

        except Timeout:
            raise TemplateProcessingError(
                f"Timeout submitting document (limit: {API_REQUEST_TIMEOUT}s)"
            )
        except ConnectionError as exc:
            raise TemplateProcessingError(
                f"Could not connect to SAP Document AI: {exc}"
            )
        except HTTPError as exc:
            self._handle_http_error(exc, context="submit_with_template")

        try:
            data = response.json()
        except ValueError:
            raise TemplateProcessingError(
                f"Invalid server response: {response.text[:300]}"
            )

        job_id = data.get("id")
        if not job_id:
            raise TemplateProcessingError(
                f"Response does not contain job 'id'. Response: {data}"
            )

        logger.info(
            "Template job created. ID: %s | Status: %s",
            job_id,
            data.get("status", "UNKNOWN"),
        )
        return job_id

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def _get_job_status(self, job_id: str) -> dict[str, Any]:
        url = f"{self._base_url}{JOBS_ENDPOINT}/{job_id}"
        try:
            response = self._session.get(
                url,
                headers={**self._auth_headers(), "Accept": "application/json"},
                timeout=API_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except Timeout:
            raise TemplateProcessingError(f"Timeout querying job {job_id}")
        except ConnectionError as exc:
            raise TemplateProcessingError(
                f"Connection error querying job {job_id}: {exc}"
            )
        except HTTPError as exc:
            self._handle_http_error(exc, context=f"get_job_status({job_id})")

        try:
            return response.json()
        except ValueError:
            raise TemplateProcessingError(
                f"Invalid response querying job {job_id}: {response.text[:300]}"
            )

    def poll_until_done(self, job_id: str) -> dict[str, Any]:
        """
        Poll the template job until it reaches a terminal status.

        Returns:
            Final job result dict.

        Raises:
            JobFailedError: If the job ends in FAILED or ERROR.
            PollingTimeoutError: If MAX_POLLING_ATTEMPTS is exceeded.
        """
        logger.info("Polling template job: %s", job_id)
        last_status = ""

        for attempt in range(1, MAX_POLLING_ATTEMPTS + 1):
            job_data = self._get_job_status(job_id)
            status = job_data.get("status", "UNKNOWN").upper()

            if status != last_status:
                icons = {
                    "PENDING": "⏳",
                    "RUNNING": "🔄",
                    "PROCESSING": "⚙️ ",
                    "DONE": "✅",
                    "FAILED": "❌",
                    "ERROR": "❌",
                }
                icon = icons.get(status, "❓")
                logger.info(
                    "%s Template job status: %s (attempt #%d)", icon, status, attempt
                )
                last_status = status

            if status in TERMINAL_STATUSES:
                if status == "DONE":
                    logger.info(
                        "Template job completed successfully. ID: %s", job_id
                    )
                    return job_data
                else:
                    error_msg = (
                        job_data.get("message") or job_data.get("error") or status
                    )
                    raise JobFailedError(
                        f"Template job {job_id} ended with status '{status}'. "
                        f"Detail: {error_msg}"
                    )

            logger.debug(
                "Attempt %d/%d — Status: %s — Waiting %ds...",
                attempt,
                MAX_POLLING_ATTEMPTS,
                status,
                POLLING_INTERVAL_SECONDS,
            )
            time.sleep(POLLING_INTERVAL_SECONDS)

        raise PollingTimeoutError(
            f"Timeout: template job {job_id} did not complete within "
            f"{MAX_POLLING_ATTEMPTS * POLLING_INTERVAL_SECONDS}s."
        )

    # ------------------------------------------------------------------
    # Save result
    # ------------------------------------------------------------------

    def save_result(self, job_id: str, result: dict[str, Any]) -> Path:
        """Save the template job result to output/routing/{JOB_ID}_template.json."""
        output_path = OUTPUT_ROUTING_DIR / f"{job_id}_template.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info("Template result saved: %s", output_path.resolve())
        return output_path

    # ------------------------------------------------------------------
    # Error handler
    # ------------------------------------------------------------------

    def _handle_http_error(self, exc: HTTPError, context: str = "") -> None:
        """Convert HTTPError into descriptive exceptions."""
        status_code = exc.response.status_code if exc.response is not None else "N/A"
        body = exc.response.text if exc.response is not None else ""
        prefix = f"[{context}] " if context else ""

        if status_code == 401:
            self._token_manager.invalidate_token()
            raise AuthenticationError(f"{prefix}Token rejected (401). Try again.")
        elif status_code == 400:
            raise TemplateProcessingError(
                f"{prefix}Bad request (400). Check schema/template payload.\n"
                f"Response: {body[:500]}"
            )
        elif status_code == 403:
            raise TemplateProcessingError(
                f"{prefix}Access denied (403). Check service key permissions.\n"
                f"Response: {body[:300]}"
            )
        elif status_code == 404:
            raise TemplateProcessingError(
                f"{prefix}Resource not found (404).\nResponse: {body[:300]}"
            )
        elif status_code == 500:
            raise TemplateProcessingError(
                f"{prefix}SAP internal server error (500).\nResponse: {body[:300]}"
            )
        else:
            raise TemplateProcessingError(
                f"{prefix}HTTP {status_code} error.\nResponse: {body[:300]}"
            )

    # ------------------------------------------------------------------
    # Main flow
    # ------------------------------------------------------------------

    def process(
        self,
        pdf_path: Path,
        template_id: str,
        schema_name: str = DEFAULT_SCHEMA_NAME,
        schema_id: str | None = None,
        client_id: str = DEFAULT_CLIENT_ID,
        document_type: str = DEFAULT_DOCUMENT_TYPE,
    ) -> tuple[dict[str, Any], Path]:
        """
        Execute the complete template-based processing flow.

        Sends BOTH schemaId/schemaName AND templateId to SAP Document AI.

        Args:
            pdf_path: Path to the invoice file.
            template_id: Matched SAP Document AI template ID.
            schema_name: Base schema name (default: SAP_invoice_schema).
            schema_id: Base schema ID (takes priority over schema_name).
            client_id: Document AI client ID.
            document_type: Document type.

        Returns:
            Tuple (result_json, saved_file_path).
        """
        schema_ref = schema_id or schema_name
        logger.info("Reprocessing invoice using schema + template...")
        logger.info("  Schema  : %s", schema_ref)
        logger.info("  Template: %s", template_id)
        logger.info("  File    : %s", pdf_path.name)

        job_id = self.submit_with_template(
            pdf_path=pdf_path,
            template_id=template_id,
            schema_name=schema_name,
            schema_id=schema_id,
            client_id=client_id,
            document_type=document_type,
        )

        print(f"\n  Template Job ID: {job_id}")
        print("  Waiting for SAP template result...\n")

        result = self.poll_until_done(job_id)
        output_path = self.save_result(job_id, result)

        print(f"\n{'='*52}")
        print("  PROCESSING COMPLETED")
        print(f"{'='*52}")
        print(f"\n  Job ID  : {job_id}")
        print(f"  Status  : {result.get('status', 'N/A')}")
        print(f"  Saved   : {output_path}\n")

        return result, output_path


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def process_with_template(
    pdf_path: Path,
    template_id: str,
    schema_name: str = DEFAULT_SCHEMA_NAME,
    schema_id: str | None = None,
    client_id: str = DEFAULT_CLIENT_ID,
    document_type: str = DEFAULT_DOCUMENT_TYPE,
) -> tuple[dict[str, Any], Path]:
    """
    Convenience function to process an invoice with a specific template.

    Args:
        pdf_path: Path to the invoice file.
        template_id: Matched SAP Document AI template ID.
        schema_name: Base schema name (default: SAP_invoice_schema).
        schema_id: Base schema ID (takes priority over schema_name).
        client_id: Document AI client ID.
        document_type: Document type.

    Returns:
        Tuple (result_json, saved_file_path).
    """
    processor = TemplateInvoiceProcessor()
    return processor.process(
        pdf_path=pdf_path,
        template_id=template_id,
        schema_name=schema_name,
        schema_id=schema_id,
        client_id=client_id,
        document_type=document_type,
    )