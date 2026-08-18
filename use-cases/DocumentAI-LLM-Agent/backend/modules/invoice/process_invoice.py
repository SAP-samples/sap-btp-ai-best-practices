"""
process_invoice.py
------------------
Processes invoices with SAP Document AI.

Flow:
  1. Scans the /invoice folder and lists available PDFs
  2. User selects the file by number
  3. Sends the PDF via POST multipart/form-data to the jobs endpoint
  4. Polls automatically until the job is DONE
  5. Saves the result to /output/{JOB_ID}.json
  6. Returns the final formatted JSON

Endpoints:
  POST  /document-information-extraction/v1/document/jobs
  GET   /document-information-extraction/v1/document/jobs/{JOB_ID}
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
from utils.config_loader import load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Service base URL — loaded from docai.json (tenantuiurl / endpoints.backend.url / url)
SERVICE_BASE_URL: str = load_config()["service_url"].rstrip("/")

# Endpoints
JOBS_ENDPOINT: str = "/document-information-extraction/v1/document/jobs"

# Directories
INVOICE_DIR: Path = Path(__file__).parent.parent.parent / "invoice"
OUTPUT_DIR: Path  = Path(__file__).parent.parent.parent / "output"

# Supported document formats and their MIME types
SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf":  "application/pdf",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".tif":  "image/tiff",
    ".tiff": "image/tiff",
}

# Polling configuration
POLLING_INTERVAL_SECONDS: int = 5
MAX_POLLING_ATTEMPTS: int = 60  # 5 minutes maximum

# HTTP timeout (seconds)
API_REQUEST_TIMEOUT: int = 60

# Terminal job statuses
TERMINAL_STATUSES: frozenset[str] = frozenset({"DONE", "FAILED", "ERROR"})

# Default processing options
DEFAULT_SCHEMA_NAME: str  = "SAP_invoice_schema"
DEFAULT_CLIENT_ID: str    = "default"
DEFAULT_DOCUMENT_TYPE: str = "invoice"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class InvoiceProcessingError(Exception):
    """Raised during invoice processing."""
    pass


class JobFailedError(InvoiceProcessingError):
    """SAP Document AI job ended in FAILED or ERROR status."""
    pass


class PollingTimeoutError(InvoiceProcessingError):
    """Maximum wait time for the job exceeded."""
    pass


# ---------------------------------------------------------------------------
# InvoiceProcessor
# ---------------------------------------------------------------------------

class InvoiceProcessor:
    """
    Processes PDF invoices using SAP Document AI.

    Manages the complete cycle:
      upload → polling → result → save to disk.

    Usage:
        processor = InvoiceProcessor()
        result, path = processor.run()
    """

    def __init__(self, token_manager: TokenManager | None = None) -> None:
        self._config = load_config()
        self._token_manager = token_manager or TokenManager(self._config)
        self._session = requests.Session()
        self._base_url = SERVICE_BASE_URL.rstrip("/")

        # Create output directory if it does not exist
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        logger.debug("InvoiceProcessor initialized. Base URL: %s", self._base_url)

    # ------------------------------------------------------------------
    # Authentication helpers
    # ------------------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        """Return headers with the current Bearer token."""
        token = self._token_manager.get_token()
        return {"Authorization": f"Bearer {token}"}

    # ------------------------------------------------------------------
    # File selection
    # ------------------------------------------------------------------

    def _list_documents(self) -> list[Path]:
        """
        Scan INVOICE_DIR and return a sorted list of supported document files.

        Supported formats: PDF, JPEG, JPG, PNG, TIF, TIFF.

        Raises:
            InvoiceProcessingError: If the directory does not exist or is empty.
        """
        if not INVOICE_DIR.exists():
            raise InvoiceProcessingError(
                f"Invoice directory not found: {INVOICE_DIR.resolve()}\n"
                "Create the 'invoice/' folder in the project root and add document files."
            )

        docs = sorted(
            p for p in INVOICE_DIR.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not docs:
            exts = ", ".join(SUPPORTED_EXTENSIONS.keys())
            raise InvoiceProcessingError(
                f"No supported document files found in: {INVOICE_DIR.resolve()}\n"
                f"Supported formats: {exts}"
            )

        return docs

    def select_document(self) -> Path:
        """
        Display the list of available documents and prompt the user to select one.

        Supported formats: PDF, JPEG, JPG, PNG, TIF, TIFF.

        Returns:
            Path to the selected document.
        """
        docs = self._list_documents()

        print(f"\n{'='*60}")
        print("  SAP Document AI — Select Document")
        print(f"{'='*60}\n")
        print(f"  Available files in '{INVOICE_DIR.name}/':\n")

        for idx, doc in enumerate(docs, start=1):
            size_kb = doc.stat().st_size / 1024
            print(f"  [{idx}] {doc.name}  ({size_kb:.1f} KB)")

        print()

        while True:
            try:
                raw = input("  Enter the file number to process: ").strip()
                choice = int(raw)
                if 1 <= choice <= len(docs):
                    selected = docs[choice - 1]
                    logger.info("File selected: %s", selected.name)
                    return selected
                else:
                    print(f"  Invalid number. Enter a value between 1 and {len(docs)}.")
            except ValueError:
                print("  Invalid input. Enter an integer.")
            except (KeyboardInterrupt, EOFError):
                print("\n  Operation cancelled by user.")
                raise InvoiceProcessingError("Operation cancelled.")

    # Keep backward-compatible alias
    def select_pdf(self) -> Path:
        """Alias for select_document() — kept for backward compatibility."""
        return self.select_document()

    # ------------------------------------------------------------------
    # Document upload
    # ------------------------------------------------------------------

    def _build_options(
        self,
        schema_name: str = DEFAULT_SCHEMA_NAME,
        client_id: str = DEFAULT_CLIENT_ID,
        document_type: str = DEFAULT_DOCUMENT_TYPE,
    ) -> dict[str, Any]:
        """Build the 'options' payload with today's date."""
        return {
            "schemaName": schema_name,
            "clientId": client_id,
            "documentType": document_type,
            "receivedDate": date.today().isoformat(),  # YYYY-MM-DD automatic
        }

    def submit_document(
        self,
        pdf_path: Path,
        schema_name: str = DEFAULT_SCHEMA_NAME,
        client_id: str = DEFAULT_CLIENT_ID,
        document_type: str = DEFAULT_DOCUMENT_TYPE,
    ) -> str:
        """
        Send the PDF to SAP Document AI via POST multipart/form-data.

        Args:
            pdf_path: Path to the PDF file.
            schema_name: Schema name to use.
            client_id: Document AI client ID.
            document_type: Document type.

        Returns:
            job_id (str) of the created job.

        Raises:
            InvoiceProcessingError: If the file does not exist or upload fails.
            AuthenticationError: If the token is rejected.
        """
        if not pdf_path.exists():
            raise InvoiceProcessingError(
                f"Document file not found: {pdf_path.resolve()}"
            )

        ext = pdf_path.suffix.lower()
        mime_type = SUPPORTED_EXTENSIONS.get(ext)
        if not mime_type:
            supported = ", ".join(SUPPORTED_EXTENSIONS.keys())
            raise InvoiceProcessingError(
                f"Unsupported file format: '{ext}'\n"
                f"Supported formats: {supported}"
            )

        url = f"{self._base_url}{JOBS_ENDPOINT}"
        options = self._build_options(schema_name, client_id, document_type)

        logger.info("Submitting document: %s (MIME: %s)", pdf_path.name, mime_type)
        logger.info("Options: %s", json.dumps(options))
        logger.info("Endpoint: %s", url)

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
            raise InvoiceProcessingError(
                f"Timeout submitting document (limit: {API_REQUEST_TIMEOUT}s)"
            )
        except ConnectionError as exc:
            raise InvoiceProcessingError(
                f"Could not connect to SAP Document AI: {exc}"
            )
        except HTTPError as exc:
            self._handle_http_error(exc, context="submit_document")

        try:
            data = response.json()
        except ValueError:
            raise InvoiceProcessingError(
                f"Invalid server response: {response.text[:300]}"
            )

        job_id = data.get("id")
        if not job_id:
            raise InvoiceProcessingError(
                f"Response does not contain job 'id'.\nResponse: {data}"
            )

        initial_status = data.get("status", "UNKNOWN")
        logger.info("Job created. ID: %s | Initial status: %s", job_id, initial_status)

        return job_id

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def _get_job_status(self, job_id: str) -> dict[str, Any]:
        """
        Query the current status of the job.

        Returns:
            Dictionary with the complete job response.
        """
        url = f"{self._base_url}{JOBS_ENDPOINT}/{job_id}"

        try:
            response = self._session.get(
                url,
                headers={**self._auth_headers(), "Accept": "application/json"},
                timeout=API_REQUEST_TIMEOUT,
            )
            response.raise_for_status()

        except Timeout:
            raise InvoiceProcessingError(
                f"Timeout querying job {job_id}"
            )
        except ConnectionError as exc:
            raise InvoiceProcessingError(
                f"Connection error querying job {job_id}: {exc}"
            )
        except HTTPError as exc:
            self._handle_http_error(exc, context=f"get_job_status({job_id})")

        try:
            return response.json()
        except ValueError:
            raise InvoiceProcessingError(
                f"Invalid response querying job {job_id}: {response.text[:300]}"
            )

    def poll_until_done(self, job_id: str) -> dict[str, Any]:
        """
        Poll the job until the status is terminal (DONE/FAILED/ERROR).

        Args:
            job_id: Job ID to monitor.

        Returns:
            Dictionary with the final job result.

        Raises:
            JobFailedError: If the job ends in FAILED or ERROR.
            PollingTimeoutError: If MAX_POLLING_ATTEMPTS is exceeded.
        """
        logger.info("Starting polling for job: %s", job_id)
        logger.info(
            "Interval: %ds | Max attempts: %d (%d min)",
            POLLING_INTERVAL_SECONDS,
            MAX_POLLING_ATTEMPTS,
            (POLLING_INTERVAL_SECONDS * MAX_POLLING_ATTEMPTS) // 60,
        )

        last_status = ""

        for attempt in range(1, MAX_POLLING_ATTEMPTS + 1):
            job_data = self._get_job_status(job_id)
            status = job_data.get("status", "UNKNOWN").upper()

            if status != last_status:
                self._log_status(status, attempt)
                last_status = status

            if status in TERMINAL_STATUSES:
                if status == "DONE":
                    logger.info("Job completed successfully. ID: %s", job_id)
                    return job_data
                else:
                    error_msg = job_data.get("message") or job_data.get("error") or status
                    raise JobFailedError(
                        f"Job {job_id} ended with status '{status}'.\n"
                        f"Detail: {error_msg}"
                    )

            logger.debug(
                "Attempt %d/%d — Status: %s — Waiting %ds...",
                attempt, MAX_POLLING_ATTEMPTS, status, POLLING_INTERVAL_SECONDS,
            )
            time.sleep(POLLING_INTERVAL_SECONDS)

        raise PollingTimeoutError(
            f"Timeout: job {job_id} did not complete within "
            f"{MAX_POLLING_ATTEMPTS * POLLING_INTERVAL_SECONDS}s."
        )

    @staticmethod
    def _log_status(status: str, attempt: int) -> None:
        """Log a status change with visual formatting."""
        icons = {
            "PENDING":    "⏳",
            "RUNNING":    "🔄",
            "PROCESSING": "⚙️ ",
            "DONE":       "✅",
            "FAILED":     "❌",
            "ERROR":      "❌",
        }
        icon = icons.get(status, "❓")
        logger.info("%s Status: %s (attempt #%d)", icon, status, attempt)

    # ------------------------------------------------------------------
    # Save result
    # ------------------------------------------------------------------

    def save_result(self, job_id: str, result: dict[str, Any]) -> Path:
        """
        Save the job result to /output/{JOB_ID}.json.

        Returns:
            Path to the saved file.
        """
        output_path = OUTPUT_DIR / f"{job_id}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info("Result saved to: %s", output_path.resolve())
        return output_path

    # ------------------------------------------------------------------
    # Centralized error handler
    # ------------------------------------------------------------------

    def _handle_http_error(self, exc: HTTPError, context: str = "") -> None:
        """Convert HTTPError into descriptive exceptions."""
        status_code = exc.response.status_code if exc.response is not None else "N/A"
        body = exc.response.text if exc.response is not None else ""

        prefix = f"[{context}] " if context else ""

        if status_code == 401:
            self._token_manager.invalidate_token()
            raise AuthenticationError(
                f"{prefix}Token rejected (401). Try again."
            )
        elif status_code == 403:
            raise InvoiceProcessingError(
                f"{prefix}Access denied (403). Check service key permissions.\n"
                f"Response: {body[:300]}"
            )
        elif status_code == 404:
            raise InvoiceProcessingError(
                f"{prefix}Resource not found (404).\nResponse: {body[:300]}"
            )
        elif status_code == 500:
            raise InvoiceProcessingError(
                f"{prefix}SAP internal server error (500).\nResponse: {body[:300]}"
            )
        else:
            raise InvoiceProcessingError(
                f"{prefix}HTTP {status_code} error.\nResponse: {body[:300]}"
            )

    # ------------------------------------------------------------------
    # Main flow
    # ------------------------------------------------------------------

    def run(
        self,
        schema_name: str = DEFAULT_SCHEMA_NAME,
        client_id: str = DEFAULT_CLIENT_ID,
        document_type: str = DEFAULT_DOCUMENT_TYPE,
    ) -> tuple[dict[str, Any], Path]:
        """
        Execute the complete invoice processing flow.

        1. Interactive PDF selection
        2. Upload to SAP Document AI
        3. Poll until DONE
        4. Save result
        5. Return final JSON

        Returns:
            Tuple (result_json, saved_file_path).
        """
        # 1. Select document
        pdf_path = self.select_document()

        print(f"\n  Processing: {pdf_path.name}")
        print(f"  Format: {pdf_path.suffix.upper().lstrip('.')}")
        print(f"  Received date: {date.today().isoformat()}")
        print(f"  Schema: {schema_name}\n")

        # 2. Submit document
        job_id = self.submit_document(pdf_path, schema_name, client_id, document_type)

        print(f"\n  Job ID: {job_id}")
        print("  Waiting for result...\n")

        # 3. Poll
        result = self.poll_until_done(job_id)

        # 4. Save result
        output_path = self.save_result(job_id, result)

        return result, output_path


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def process_invoice(
    schema_name: str = DEFAULT_SCHEMA_NAME,
    client_id: str = DEFAULT_CLIENT_ID,
    document_type: str = DEFAULT_DOCUMENT_TYPE,
) -> tuple[dict[str, Any], Path]:
    """
    Convenience function to process an invoice.

    Returns:
        Tuple (result_json, saved_file_path).
    """
    processor = InvoiceProcessor()
    return processor.run(schema_name, client_id, document_type)