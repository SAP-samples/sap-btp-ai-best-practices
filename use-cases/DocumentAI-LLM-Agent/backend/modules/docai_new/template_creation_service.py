"""
template_creation_service.py
-----------------------------
DOC AI NEW — Template Creation Service.

Creates SAP Document AI templates automatically.
Schema: SAP_invoice_schema — schemaId: cf8cc8a9-1eee-42d9-9a3e-507a61baac23
Template Name: customer_name (from LLM extraction)

FIX: SAP API requires "schemaId" (UUID), NOT "schemaName".
Error ET018 is raised when schemaId is missing or wrong.

CORRECT workflow for add_document_to_template():
  Step 1: Upload PDF via POST /document/jobs (without templateId)
          → SAP returns document_id (= job id)
  Step 2: Poll GET /document/jobs/{document_id} until DONE
  Step 3: Associate via POST /templates/{template_id}/documents/{document_id}

SAP DOX endpoints that actually exist:
  GET  /document-information-extraction/v1/templates                          → list templates
  POST /document-information-extraction/v1/templates                          → create template
  POST /document-information-extraction/v1/templates/{id}/train               → train template
  POST /document-information-extraction/v1/document/jobs                      → upload document
  GET  /document-information-extraction/v1/document/jobs/{id}                 → poll job
  POST /document-information-extraction/v1/templates/{id}/documents/{doc_id}  → associate document
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
    JOBS_ENDPOINT,
    MAX_POLLING_ATTEMPTS,
    POLLING_INTERVAL_SECONDS,
    SERVICE_BASE_URL,
    TERMINAL_STATUSES,
)
from utils.config_loader import load_config

logger = logging.getLogger(__name__)

TEMPLATES_ENDPOINT: str = "/document-information-extraction/v1/templates"
SCHEMAS_ENDPOINT: str = "/document-information-extraction/v1/schemas"

# SAP_invoice_schema — hardcoded UUID (primary)
# Fallback: resolved dynamically via getSchemas() if this is wrong
SCHEMA_ID: str = "cf8cc8a9-1eee-42d9-9a3e-507a61baac23"
SCHEMA_NAME: str = "SAP_invoice_schema"

DEFAULT_CLIENT_ID: str = "default"
DEFAULT_DOCUMENT_TYPE: str = "invoice"


class TemplateCreationError(Exception):
    """Raised when template creation fails."""
    pass


class TemplateCreationService:
    """Creates SAP Document AI templates automatically."""

    def __init__(self, token_manager: TokenManager | None = None) -> None:
        self._config = load_config()
        self._token_manager = token_manager or TokenManager(self._config)
        self._session = requests.Session()
        self._base_url = SERVICE_BASE_URL.rstrip("/")
        self._resolved_schema_id: str | None = None

    def _auth_headers(self) -> dict[str, str]:
        token = self._token_manager.get_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _auth_headers_no_ct(self) -> dict[str, str]:
        token = self._token_manager.get_token()
        return {"Authorization": f"Bearer {token}"}

    def _resolve_schema_id(self) -> str:
        """Return schemaId for SAP_invoice_schema (hardcoded, with API fallback)."""
        if self._resolved_schema_id:
            return self._resolved_schema_id

        if SCHEMA_ID:
            logger.info("Using hardcoded schemaId: %s", SCHEMA_ID)
            self._resolved_schema_id = SCHEMA_ID
            return self._resolved_schema_id

        logger.info("schemaId not configured — resolving via getSchemas() API...")
        url = f"{self._base_url}{SCHEMAS_ENDPOINT}"
        try:
            response = self._session.get(url, headers=self._auth_headers(), timeout=API_REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise TemplateCreationError(f"Failed to resolve schemaId via API: {exc}")

        schemas = (
            data.get("value") or data.get("schemas") or data.get("results")
            or (data if isinstance(data, list) else [])
        )
        for schema in schemas:
            name = schema.get("name", "") or schema.get("schemaName", "")
            if name == SCHEMA_NAME:
                schema_id = schema.get("id") or schema.get("schemaId")
                if schema_id:
                    logger.info("Resolved schemaId for '%s': %s", SCHEMA_NAME, schema_id)
                    self._resolved_schema_id = schema_id
                    return self._resolved_schema_id

        raise TemplateCreationError(
            f"Could not find schemaId for '{SCHEMA_NAME}'. "
            f"Available: {[s.get('name') for s in schemas]}"
        )

    def create_template(
        self,
        customer_name: str,
        client_id: str = DEFAULT_CLIENT_ID,
    ) -> dict[str, Any]:
        """
        Create a new SAP Document AI template for the given customer.

        SAP API requires "schemaId" (UUID) — NOT "schemaName".
        Error ET018 is raised when schemaId is missing.
        """
        schema_id = self._resolve_schema_id()
        if not schema_id:
            raise TemplateCreationError("schemaId missing — cannot create template.")

        url = f"{self._base_url}{TEMPLATES_ENDPOINT}"
        payload = {
            "schemaId": schema_id,
            "name": customer_name,
            "description": f"Auto-generated template for customer {customer_name}",
            "clientId": client_id,
            "documentType": DEFAULT_DOCUMENT_TYPE,
        }

        logger.info("CREATE TEMPLATE PAYLOAD: %s", json.dumps(payload, indent=2))

        try:
            response = self._session.post(
                url, headers=self._auth_headers(), json=payload, timeout=API_REQUEST_TIMEOUT
            )
            response.raise_for_status()
        except Timeout:
            raise TemplateCreationError("Timeout creating template")
        except ConnectionError as exc:
            raise TemplateCreationError(f"Connection error: {exc}")
        except HTTPError as exc:
            self._handle_http_error(exc, "create_template")

        try:
            data = response.json()
        except ValueError:
            raise TemplateCreationError(f"Invalid response: {response.text[:300]}")

        template_id = data.get("id")
        if not template_id:
            raise TemplateCreationError(f"No template id in response: {data}")

        logger.info("Template created. ID: %s | Name: %s", template_id, data.get("name"))
        return data

    def activate_template(
        self,
        template_id: str,
        client_id: str = DEFAULT_CLIENT_ID,
    ) -> dict[str, Any]:
        """
        Activate a template via POST /templates/{template_id}/activate.

        Templates are created in DRAFT status and must be activated
        before documents can be associated or processed.

        Args:
            template_id: SAP Document AI template ID.
            client_id: SAP Document AI client ID.

        Returns:
            Activation result dict.
        """
        url = f"{self._base_url}{TEMPLATES_ENDPOINT}/{template_id}/activate"
        payload = {"clientId": client_id}

        logger.info("ACTIVATE TEMPLATE")
        logger.info("Template ID: %s", template_id)
        logger.info("HTTP METHOD: POST")
        logger.info("REQUEST URL: %s", url)

        try:
            response = self._session.post(
                url, headers=self._auth_headers(), json=payload, timeout=API_REQUEST_TIMEOUT
            )
            logger.info("ACTIVATE STATUS: %s", response.status_code)
            response.raise_for_status()
        except Timeout:
            raise TemplateCreationError("Timeout activating template")
        except ConnectionError as exc:
            raise TemplateCreationError(f"Connection error: {exc}")
        except HTTPError as exc:
            logger.error("ACTIVATE ERROR STATUS: %s", exc.response.status_code if exc.response else "N/A")
            logger.error("ACTIVATE ERROR BODY: %s", exc.response.text if exc.response else "")
            self._handle_http_error(exc, "activate_template")

        try:
            data = response.json()
        except ValueError:
            data = {"status": "activated", "template_id": template_id}

        logger.info("Template activated. Status: %s", data.get("status", "activated"))
        return data

    def upload_document(
        self,
        pdf_path: Path,
        client_id: str = DEFAULT_CLIENT_ID,
    ) -> str:
        """
        Upload a PDF to SAP Document AI and return the document_id.

        Calls POST /document/jobs WITHOUT templateId.
        SAP returns {"id": "<document_id>", "status": "PENDING"}.
        The returned id is the document_id used for template association.

        Args:
            pdf_path: Path to the PDF file.
            client_id: SAP Document AI client ID.

        Returns:
            document_id (str) — the SAP job/document ID.
        """
        schema_id = self._resolve_schema_id()
        url = f"{self._base_url}{JOBS_ENDPOINT}"

        options = {
            "schemaId": schema_id,
            "clientId": client_id,
            "documentType": DEFAULT_DOCUMENT_TYPE,
            "receivedDate": date.today().isoformat(),
        }

        logger.info("UPLOAD DOCUMENT")
        logger.info("PDF: %s", pdf_path)
        logger.info("HTTP METHOD: POST")
        logger.info("REQUEST URL: %s", url)
        logger.info("PAYLOAD: %s", json.dumps(options, indent=2))

        try:
            with open(pdf_path, "rb") as pdf_file:
                response = self._session.post(
                    url,
                    headers=self._auth_headers_no_ct(),
                    files={
                        "file": (pdf_path.name, pdf_file, "application/pdf"),
                        "options": (None, json.dumps(options), "application/json"),
                    },
                    timeout=API_REQUEST_TIMEOUT,
                )
            logger.info("UPLOAD STATUS: %s", response.status_code)
            response.raise_for_status()
        except Timeout:
            raise TemplateCreationError("Timeout uploading document")
        except ConnectionError as exc:
            raise TemplateCreationError(f"Connection error: {exc}")
        except HTTPError as exc:
            logger.error("UPLOAD ERROR STATUS: %s", exc.response.status_code if exc.response else "N/A")
            logger.error("UPLOAD ERROR BODY: %s", exc.response.text if exc.response else "")
            self._handle_http_error(exc, "upload_document")

        try:
            data = response.json()
        except ValueError:
            raise TemplateCreationError(f"Invalid upload response: {response.text[:300]}")

        document_id = data.get("id")
        if not document_id:
            raise TemplateCreationError(
                f"No document_id in upload response. Response: {data}"
            )

        logger.info("DOCUMENT ID: %s", document_id)
        return document_id

    def poll_document_job(
        self,
        document_id: str,
        client_id: str = DEFAULT_CLIENT_ID,
    ) -> dict[str, Any]:
        """
        Poll GET /document/jobs/{document_id} until terminal status.

        Args:
            document_id: SAP document/job ID.
            client_id: SAP Document AI client ID.

        Returns:
            Final job result dict.
        """
        url = f"{self._base_url}{JOBS_ENDPOINT}/{document_id}"
        params = {"clientId": client_id}

        logger.info("Polling document job '%s'...", document_id)

        for attempt in range(1, MAX_POLLING_ATTEMPTS + 1):
            try:
                response = self._session.get(
                    url, headers=self._auth_headers(), params=params, timeout=API_REQUEST_TIMEOUT
                )
                response.raise_for_status()
                data = response.json()
            except Exception as exc:
                logger.warning("Poll attempt %d failed: %s", attempt, exc)
                time.sleep(POLLING_INTERVAL_SECONDS)
                continue

            status = data.get("status", "")
            logger.info("Document %s — status: %s (attempt %d)", document_id, status, attempt)

            if status in TERMINAL_STATUSES:
                return data

            time.sleep(POLLING_INTERVAL_SECONDS)

        raise TemplateCreationError(
            f"Document '{document_id}' did not complete after {MAX_POLLING_ATTEMPTS} attempts."
        )

    def associate_document_to_template(
        self,
        template_id: str,
        document_id: str,
        client_id: str = DEFAULT_CLIENT_ID,
    ) -> dict[str, Any]:
        """
        Associate an uploaded document with a template.

        Calls: POST /templates/{template_id}/documents/{document_id}

        Args:
            template_id: SAP Document AI template ID.
            document_id: SAP document ID (from upload_document()).
            client_id: SAP Document AI client ID.

        Returns:
            Association result dict.
        """
        if not document_id:
            raise TemplateCreationError(
                "document_id is missing — cannot associate document to template. "
                "Call upload_document() first to obtain a document_id."
            )

        url = f"{self._base_url}{TEMPLATES_ENDPOINT}/{template_id}/documents/{document_id}"

        logger.info("DOCUMENT ID: %s", document_id)
        logger.info("TEMPLATE ID: %s", template_id)
        logger.info(
            "ASSOCIATE URL: /templates/%s/documents/%s",
            template_id,
            document_id,
        )
        logger.info("HTTP METHOD: POST")
        logger.info("REQUEST URL: %s", url)

        payload = {"clientId": client_id}
        logger.info("PAYLOAD: %s", json.dumps(payload, indent=2))

        try:
            response = self._session.post(
                url, headers=self._auth_headers(), json=payload, timeout=API_REQUEST_TIMEOUT
            )
            logger.info("STATUS: %s", response.status_code)
            response.raise_for_status()
        except Timeout:
            raise TemplateCreationError("Timeout associating document to template")
        except ConnectionError as exc:
            raise TemplateCreationError(f"Connection error: {exc}")
        except HTTPError as exc:
            logger.error("STATUS: %s", exc.response.status_code if exc.response else "N/A")
            logger.error("BODY: %s", exc.response.text if exc.response else "")
            self._handle_http_error(exc, "associate_document_to_template")

        try:
            return response.json()
        except ValueError:
            return {"status": "associated", "template_id": template_id, "document_id": document_id}

    def add_document_to_template(
        self,
        template_id: str,
        pdf_path: Path,
        client_id: str = DEFAULT_CLIENT_ID,
    ) -> dict[str, Any]:
        """
        Full workflow to attach a PDF to a template.

        Steps:
          1. Upload PDF via POST /document/jobs → get document_id
          2. Poll until DONE
          3. Associate via POST /templates/{template_id}/documents/{document_id}

        Args:
            template_id: SAP Document AI template ID.
            pdf_path: Path to the PDF file.
            client_id: SAP Document AI client ID.

        Returns:
            Dict with document_id, association result, and job result.
        """
        logger.info("ATTACH DOCUMENT")
        logger.info("Template ID: %s", template_id)
        logger.info("Client ID: %s", client_id)
        logger.info("PDF: %s", pdf_path)

        # Step 1: Upload PDF → get document_id
        document_id = self.upload_document(pdf_path, client_id)

        if not document_id:
            raise TemplateCreationError(
                "document_id is missing after upload — cannot associate to template."
            )

        logger.info("DOCUMENT ID: %s", document_id)

        # Step 2: Poll until DONE
        job_result = self.poll_document_job(document_id, client_id)
        logger.info("Document job DONE. Status: %s", job_result.get("status"))

        # Step 3: Associate document to template
        association = self.associate_document_to_template(template_id, document_id, client_id)
        logger.info("Document associated to template. Result: %s", association)

        return {
            "document_id": document_id,
            "job_result": job_result,
            "association": association,
        }

    def configure_metadata(
        self,
        template_id: str,
        extracted_fields: dict[str, Any] | None = None,
        client_id: str = DEFAULT_CLIENT_ID,
    ) -> dict[str, Any]:
        """
        Configure metadata for a template via POST /templates/{template_id}/metadata.

        For each extracted field, sends:
          { "name": "<field_name>", "extraction": "template" }

        For line item fields, adds:
          { "name": "<field_name>", "extraction": "template", "isLineItemField": true }

        Args:
            template_id: SAP Document AI template ID.
            extracted_fields: Dict of extracted field names → values (from Free Prompt).
            client_id: SAP Document AI client ID.

        Returns:
            Metadata configuration result dict.
        """
        url = f"{self._base_url}{TEMPLATES_ENDPOINT}/{template_id}/metadata"

        # Standard header fields
        header_fields = [
            "customer_name",
            "customer_address",
            "customer_tax_id",
            "invoice_number",
            "invoice_date",
            "due_date",
            "subtotal",
            "tax_amount",
            "total_amount",
        ]

        # Line item fields
        line_item_fields = [
            "description",
            "quantity",
            "unit_price",
            "line_total",
        ]

        metadata_fields = []

        # Add header fields
        for field_name in header_fields:
            metadata_fields.append({
                "name": field_name,
                "extraction": "template",
            })

        # Add line item fields
        for field_name in line_item_fields:
            metadata_fields.append({
                "name": field_name,
                "extraction": "template",
                "isLineItemField": True,
            })

        # Add any extra fields from extraction not already covered
        if extracted_fields:
            known = set(header_fields + line_item_fields)
            for field_name in extracted_fields:
                if field_name not in known and field_name != "line_items":
                    metadata_fields.append({
                        "name": field_name,
                        "extraction": "template",
                    })

        payload = {
            "clientId": client_id,
            "fields": metadata_fields,
        }

        logger.info("CONFIGURE METADATA")
        logger.info("Template ID: %s", template_id)
        logger.info("HTTP METHOD: POST")
        logger.info("REQUEST URL: %s", url)
        logger.info("Fields count: %d", len(metadata_fields))

        try:
            response = self._session.post(
                url, headers=self._auth_headers(), json=payload, timeout=API_REQUEST_TIMEOUT
            )
            logger.info("METADATA STATUS: %s", response.status_code)
            response.raise_for_status()
        except Timeout:
            raise TemplateCreationError("Timeout configuring template metadata")
        except ConnectionError as exc:
            raise TemplateCreationError(f"Connection error: {exc}")
        except HTTPError as exc:
            logger.error("METADATA ERROR STATUS: %s", exc.response.status_code if exc.response else "N/A")
            logger.error("METADATA ERROR BODY: %s", exc.response.text if exc.response else "")
            self._handle_http_error(exc, "configure_metadata")

        try:
            data = response.json()
        except ValueError:
            data = {"status": "metadata_configured", "template_id": template_id, "fields_count": len(metadata_fields)}

        logger.info("Metadata configured. Fields: %d", len(metadata_fields))
        return data

    def train_template(
        self,
        template_id: str,
        client_id: str = DEFAULT_CLIENT_ID,
    ) -> dict[str, Any]:
        """
        NOTE: SAP DOX API does NOT expose a /train endpoint.
        This method is kept for reference but should NOT be called.
        The correct workflow is:
          Create → Activate → Associate Document → Configure Metadata → Complete

        Raises:
            TemplateCreationError: Always — endpoint does not exist.
        """
        raise TemplateCreationError(
            "train_template() is not supported. "
            "SAP DOX API does not expose POST /templates/{id}/train. "
            "Use configure_metadata() instead."
        )

    def _handle_http_error(self, exc: HTTPError, context: str = "") -> None:
        status_code = exc.response.status_code if exc.response is not None else "N/A"
        body = exc.response.text if exc.response is not None else ""
        prefix = f"[{context}] " if context else ""

        if status_code == 401:
            self._token_manager.invalidate_token()
            raise AuthenticationError(f"{prefix}Token rejected (401).")
        elif status_code == 400:
            raise TemplateCreationError(f"{prefix}Bad request (400).\nResponse: {body[:500]}")
        elif status_code == 403:
            raise TemplateCreationError(f"{prefix}Access denied (403).\nResponse: {body[:300]}")
        elif status_code == 404:
            raise TemplateCreationError(f"{prefix}Not found (404).\nResponse: {body[:300]}")
        elif status_code == 409:
            raise TemplateCreationError(
                f"{prefix}Conflict (409) — template may already exist.\nResponse: {body[:300]}"
            )
        else:
            raise TemplateCreationError(f"{prefix}HTTP {status_code}.\nResponse: {body[:300]}")
