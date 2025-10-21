"""
Reusable SAP Document Information Extraction (Document AI / DOX) client.

This client reads OAuth and API endpoint information from a service key JSON
file (e.g., `service_key.txt` in this workspace) and exposes helpers to:

- Authenticate and retrieve a Bearer token via client_credentials
- Upload a document for extraction
  - Using an existing schema (schemaId/schemaName)
  - Or ad-hoc extraction using headerFields/lineItemFields (custom)
- Get job status/results
- Optionally poll until completion (success/failure)
- Delete a job

Example usage:

    from sap_dox_client import SapDoxClient

    # Initialize the client from a service key JSON file
    client = SapDoxClient.from_service_key("service_key.txt")

    # Start a job using a schema
    job = client.upload_document(
        file_path="uploads/2025-05-21_Conoce_a_tu_cliente.pdf",
        client_id="default",
        schema_name="KYC_sesajal"
    )
    job_id = job.get("id")

    # Or start a job with ad-hoc fields (custom)
    # job = client.upload_document(
    #     file_path="path/to/file.pdf",
    #     client_id="default",
    #     header_fields=[{"name": "Fecha", "formattingType": "date"}],
    #     line_item_fields=[]
    # )

    # Poll for results
    result = client.wait_for_result(job_id, timeout_seconds=120, poll_interval_seconds=3)

    # Access job details directly
    job_details = client.get_job(job_id)

    # Delete job when no longer needed
    client.delete_job(job_id)

Notes:
- The service key JSON must contain an `uaa` block with `clientid`, `clientsecret`, and `url` (the XSUAA token URL base).
- The service key must also contain the DOX service `url` and `swagger` path to build the REST base, like:
  - url: "https://aiservices-dox.cfapps.eu10.hana.ondemand.com"
  - swagger: "/document-information-extraction/v1/"

"""

from __future__ import annotations

import json
import mimetypes
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


def _join_url(base: str, path: str) -> str:
    """Join two URL parts ensuring exactly one slash in between.

    This avoids common pitfalls with double slashes or missing separators.
    """
    if not base:
        return path
    if not path:
        return base
    if base.endswith("/") and path.startswith("/"):
        return base[:-1] + path
    if not base.endswith("/") and not path.startswith("/"):
        return base + "/" + path
    return base + path


@dataclass
class ServiceKey:
    """Structured representation of the minimal fields used from the service key."""

    token_base_url: str  # e.g., https://<xsuaa-domain>
    client_id: str
    client_secret: str
    dox_base_url: str    # e.g., https://aiservices-dox.cfapps.eu10.hana.ondemand.com
    swagger_path: str    # e.g., /document-information-extraction/v1/

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "ServiceKey":
        """Create a ServiceKey from dict parsed from JSON file.

        Expected structure (subset):
        {
          "uaa": {
            "url": "https://<xsuaa-domain>",
            "clientid": "...",
            "clientsecret": "..."
          },
          "url": "https://aiservices-dox.cfapps...",
          "swagger": "/document-information-extraction/v1/"
        }
        """
        uaa = data.get("uaa", {})
        token_base_url = uaa.get("url") or uaa.get("sburl")
        if not token_base_url:
            raise ValueError("Service key missing 'uaa.url' (token base URL)")
        client_id = uaa.get("clientid")
        client_secret = uaa.get("clientsecret")
        if not client_id or not client_secret:
            raise ValueError("Service key missing 'uaa.clientid' or 'uaa.clientsecret'")

        dox_base_url = data.get("url")
        swagger_path = data.get("swagger", "/document-information-extraction/v1/")
        if not dox_base_url:
            raise ValueError("Service key missing DOX 'url'")

        return cls(
            token_base_url=token_base_url,
            client_id=client_id,
            client_secret=client_secret,
            dox_base_url=dox_base_url,
            swagger_path=swagger_path,
        )


class SapDoxClient:
    """Client for SAP Document Information Extraction REST API.

    This client encapsulates OAuth token retrieval and provides methods to
    upload documents, check status/results, and delete jobs.
    """

    def __init__(self, service_key: ServiceKey, session: Optional[requests.Session] = None) -> None:
        # Persist configuration
        self._service_key = service_key
        # Build REST base URL, e.g., https://.../document-information-extraction/v1
        self._rest_base = _join_url(service_key.dox_base_url, service_key.swagger_path).rstrip("/")
        # Lazy-created session
        self._http = session or requests.Session()
        # Cached bearer token and expiry
        self._access_token: Optional[str] = None
        self._token_expires_at_epoch: float = 0.0

    @classmethod
    def from_service_key(cls, service_key_path: str) -> "SapDoxClient":
        """Instantiate client by loading a service key JSON file.

        - service_key_path: Path to the service key JSON file (e.g., service_key.txt).
        """
        if not os.path.isfile(service_key_path):
            raise FileNotFoundError(f"Service key file not found: {service_key_path}")
        with open(service_key_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        sk = ServiceKey.from_json(raw)
        return cls(sk)

    # -------------------------
    # Authentication Utilities
    # -------------------------
    def _token_url(self) -> str:
        """Return full token endpoint URL for XSUAA."""
        return _join_url(self._service_key.token_base_url, "/oauth/token")

    def _now(self) -> float:
        """Return current epoch time in seconds. Separated for testability."""
        return time.time()

    def get_token(self, force_refresh: bool = False) -> str:
        """Retrieve a Bearer access token using client_credentials.

        Caches the token until its expiry window is reached. Set force_refresh=True
        to fetch a new token regardless of cache state.
        """
        # Refresh token a bit before expiry (safety window 30 seconds)
        if not force_refresh and self._access_token and self._now() < (self._token_expires_at_epoch - 30):
            return self._access_token  # Reuse cached token

        data = {"grant_type": "client_credentials"}
        response = self._http.post(
            self._token_url(),
            data=data,
            auth=(self._service_key.client_id, self._service_key.client_secret),
            headers={"Accept": "application/json"},
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Failed to obtain token: {response.status_code} - {response.text}")

        payload = response.json()
        self._access_token = payload.get("access_token")
        if not self._access_token:
            raise RuntimeError("Token response missing 'access_token'")

        expires_in = payload.get("expires_in", 3600)
        self._token_expires_at_epoch = self._now() + float(expires_in)
        return self._access_token

    def _auth_header(self) -> Dict[str, str]:
        """Return Authorization header with a valid token."""
        token = self.get_token()
        return {"Authorization": f"Bearer {token}"}

    # -------------------------
    # Schema Operations
    # -------------------------
    def create_schema(
        self,
        client_id: str,
        schema_name: str,
        schema_desc: Optional[str] = None,
        document_type: Optional[str] = None,
        document_type_desc: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new schema.

        Calls POST /schemas with clientId in the request body.

        - client_id: DOX tenant client ID
        - schema_name: Schema name to create (must be unique within client)
        - schema_desc: Optional description
        - document_type: Optional document type identifier (e.g., "custom")
        - document_type_desc: Optional document type description
        """
        if not schema_name:
            raise ValueError("schema_name is required")
        url = _join_url(self._rest_base, "/schemas")
        headers = {**self._auth_header(), "Content-Type": "application/json"}
        # API requires clientId in JSON body for create
        payload: Dict[str, Any] = {"clientId": client_id, "name": schema_name}
        if schema_desc is not None:
            payload["schemaDescription"] = schema_desc
        if document_type is not None:
            payload["documentType"] = document_type
        if document_type_desc is not None:
            payload["documentTypeDescription"] = document_type_desc

        response = self._http.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code not in (200, 201):
            raise RuntimeError(f"Failed to create schema: {response.status_code} - {response.text}")
        return response.json()


    def add_fields_to_schema_version(
        self,
        schema_id: str,
        version: str | int,
        client_id: str = "default",
        header_fields: Optional[List[Dict[str, Any]]] = None,
        line_item_fields: Optional[List[Dict[str, Any]]] = None,
        replace: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Add fields to a specific schema version.

        Calls POST /schemas/{schema_id}/versions/{version}/fields?clientId=...

        The API expects field names. This method accepts either a list of
        strings or a list of objects containing at least a `name` property; it
        will send only names to the API.

        - header_fields: List[str] or List[Dict] describing header field names
        - line_item_fields: List[str] or List[Dict] describing line item names
        - replace: If True, replace existing fields; if False, append; if None, server default
        """
        if not schema_id:
            raise ValueError("schema_id is required")
        if version is None:
            raise ValueError("version is required")
        url = _join_url(self._rest_base, f"/schemas/{schema_id}/versions/{version}/fields")
        headers = {**self._auth_header(), "Content-Type": "application/json"}
        params: Dict[str, Any] = {"clientId": client_id}
        payload: Dict[str, Any] = {}

        def _to_names(items: Optional[List[Any]]) -> Optional[List[str]]:
            if items is None:
                return None
            names: List[str] = []
            for item in items:
                if isinstance(item, str):
                    names.append(item)
                elif isinstance(item, dict):
                    name_value = item.get("name")
                    if not name_value or not isinstance(name_value, str):
                        raise ValueError("Field dict items must contain a non-empty 'name' string")
                    names.append(name_value)
                else:
                    raise ValueError("Fields must be a list of strings or dicts with a 'name' key")
            return names

        header_names = _to_names(header_fields)
        line_names = _to_names(line_item_fields)
        if header_names is not None:
            payload["headerFields"] = header_names
        if line_names is not None:
            payload["lineItemFields"] = line_names
        # Some API variants support a replace flag. Only include if specified.
        if replace is not None:
            payload["replace"] = bool(replace)

        response = self._http.post(url, headers=headers, params=params, json=payload, timeout=60)
        if response.status_code not in (200, 201):
            raise RuntimeError(
                f"Failed to add fields to schema version: {response.status_code} - {response.text}"
            )
        return response.json()

    def activate_schema_version(self, schema_id: str, version: str | int, client_id: str = "default") -> Dict[str, Any]:
        """Activate a specific schema version.

        Calls POST /schemas/{schema_id}/versions/{version}/activate?clientId=...
        """
        if not schema_id:
            raise ValueError("schema_id is required")
        if version is None:
            raise ValueError("version is required")
        url = _join_url(self._rest_base, f"/schemas/{schema_id}/versions/{version}/activate")
        headers = {**self._auth_header(), "Content-Type": "application/json"}
        params = {"clientId": client_id}
        response = self._http.post(url, headers=headers, params=params, json={}, timeout=60)
        if response.status_code not in (200, 201):
            raise RuntimeError(
                f"Failed to activate schema version: {response.status_code} - {response.text}"
            )
        return response.json()

    def deactivate_schema_version(self, schema_id: str, version: str | int, client_id: str = "default") -> Dict[str, Any]:
        """Deactivate a specific schema version.

        Calls POST /schemas/{schema_id}/versions/{version}/deactivate?clientId=...
        """
        if not schema_id:
            raise ValueError("schema_id is required")
        if version is None:
            raise ValueError("version is required")
        url = _join_url(self._rest_base, f"/schemas/{schema_id}/versions/{version}/deactivate")
        headers = {**self._auth_header(), "Content-Type": "application/json"}
        params = {"clientId": client_id}
        response = self._http.post(url, headers=headers, params=params, json={}, timeout=60)
        if response.status_code not in (200, 201):
            raise RuntimeError(
                f"Failed to deactivate schema version: {response.status_code} - {response.text}"
            )
        return response.json()

    def list_schemas(
        self,
        client_id: str = "default",
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return available schemas for the given client.

        This calls GET /schemas with optional pagination parameters.

        - client_id: DOX tenant client ID
        - limit: Optional max number of items to return
        - offset: Optional starting offset for pagination
        """
        url = _join_url(self._rest_base, "/schemas")
        headers = self._auth_header()
        params: Dict[str, Any] = {"clientId": client_id}
        if limit is not None:
            params["limit"] = int(limit)
        if offset is not None:
            params["offset"] = int(offset)

        response = self._http.get(url, headers=headers, params=params, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to list schemas: {response.status_code} - {response.text}")
        data = response.json()
        # Some APIs return {"items": [...], "count": N}; handle both list or wrapped
        if isinstance(data, dict) and "items" in data:
            return data["items"]  # type: ignore[return-value]
        if isinstance(data, list):
            return data
        # Fallback: unknown shape
        return [data]

    def get_schema_details(self, schema_id: str, client_id: str = "default") -> Dict[str, Any]:
        """Return details for a specific schema by ID.

        Calls GET /schemas/{schema_id}?clientId=... and returns the JSON body.
        """
        if not schema_id:
            raise ValueError("schema_id is required")
        url = _join_url(self._rest_base, f"/schemas/{schema_id}")
        headers = self._auth_header()
        params = {"clientId": client_id}
        response = self._http.get(url, headers=headers, params=params, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to get schema details: {response.status_code} - {response.text}"
            )
        return response.json()

    def list_schema_versions(self, schema_id: str, client_id: str = "default") -> List[Dict[str, Any]]:
        """Return all versions for a schema.

        Calls GET /schemas/{schema_id}/versions?clientId=...
        """
        if not schema_id:
            raise ValueError("schema_id is required")
        url = _join_url(self._rest_base, f"/schemas/{schema_id}/versions")
        headers = self._auth_header()
        params = {"clientId": client_id}
        response = self._http.get(url, headers=headers, params=params, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to list schema versions: {response.status_code} - {response.text}"
            )
        data = response.json()
        if isinstance(data, dict) and "items" in data:
            return data["items"]  # type: ignore[return-value]
        if isinstance(data, list):
            return data
        return [data]

    def get_schema_version_details(self, schema_id: str, version: str | int, client_id: str = "default") -> Dict[str, Any]:
        """Return details for a specific schema version.

        Calls GET /schemas/{schema_id}/versions/{version}?clientId=...
        """
        if not schema_id:
            raise ValueError("schema_id is required")
        if version is None:
            raise ValueError("version is required")
        url = _join_url(self._rest_base, f"/schemas/{schema_id}/versions/{version}")
        headers = self._auth_header()
        params = {"clientId": client_id}
        response = self._http.get(url, headers=headers, params=params, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to get schema version details: {response.status_code} - {response.text}"
            )
        return response.json()

    def is_schema_version_editable(self, schema_id: str, version: str | int, client_id: str = "default") -> bool:
        """Return True if the schema version appears editable per API conventions.

        Heuristics based on documentation:
        - Predefined schemas cannot be edited.
        - Only inactive versions are editable; active versions must be deactivated first.
        """
        details = self.get_schema_version_details(schema_id=schema_id, version=version, client_id=client_id)
        # Some fields may vary; check common flags
        predefined = bool(details.get("predefined", False))
        state = (details.get("state") or "").lower()
        if predefined:
            return False
        if state != "inactive":
            return False
        return True

    def delete_schema(self, schema_id: str, client_id: str = "default") -> None:
        """Delete a schema by ID.

        Calls DELETE /schemas/{schema_id}?clientId=...
        """
        if not schema_id:
            raise ValueError("schema_id is required")
        url = _join_url(self._rest_base, f"/schemas/{schema_id}")
        headers = self._auth_header()
        params = {"clientId": client_id}
        response = self._http.delete(url, headers=headers, params=params, timeout=60)
        if response.status_code not in (200, 202, 204):
            raise RuntimeError(
                f"Failed to delete schema: {response.status_code} - {response.text}"
            )

    # -------------------------
    # Job Operations
    # -------------------------
    def upload_document(
        self,
        file_path: str,
        client_id: str = "default",
        schema_id: Optional[str] = None,
        schema_name: Optional[str] = None,
        schema_version: Optional[str] = None,
        template_id: Optional[str] = None,
        header_fields: Optional[List[Dict[str, Any]]] = None,
        line_item_fields: Optional[List[Dict[str, Any]]] = None,
        mime_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new extraction job by uploading a document.

        Provide either a schema reference (schema_id or schema_name) OR
        ad-hoc field definitions via header_fields/line_item_fields.

        - file_path: Path to the document to upload.
        - client_id: DOX tenant client ID (often "default").
        - schema_id: Optional schema ID to use.
        - schema_name: Optional schema name to use.
        - schema_version: Optional schema version to use.
        - template_id: Optional template ID to use (sends as templateId).
        - header_fields: Optional array of field definitions for header-level extraction.
        - line_item_fields: Optional array of field definitions for line items.
        - mime_type: Optional MIME type; guessed from the filename if not provided.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if not mime_type:
            mime_type = mimetypes.guess_type(file_path)[0] or "application/pdf"

        # Build options depending on inputs
        options: Dict[str, Any] = {"clientId": client_id}

        if schema_id or schema_name:
            if schema_id:
                options["schemaId"] = schema_id
            if schema_name:
                options["schemaName"] = schema_name
            if schema_version:
                options["schemaVersion"] = schema_version
        elif header_fields or line_item_fields:
            # Ad-hoc custom extraction
            options["documentType"] = "custom"
            if header_fields:
                options["headerFields"] = header_fields
            if line_item_fields:
                options["lineItemFields"] = line_item_fields
        else:
            raise ValueError(
                "Must provide either schema_id/schema_name or header_fields/line_item_fields"
            )

        # When provided, include the templateId as per API documentation
        if template_id:
            options["templateId"] = template_id

        url = _join_url(self._rest_base, "/document/jobs")
        headers = self._auth_header()

        # Use a context manager to avoid leaking file handles
        with open(file_path, "rb") as f:
            files = {
                "file": (os.path.basename(file_path), f, mime_type),
                "options": (None, json.dumps(options), "application/json"),
            }
            response = self._http.post(url, headers=headers, files=files, timeout=120)

        if response.status_code not in (200, 201, 202):
            raise RuntimeError(
                f"Failed to upload document: {response.status_code} - {response.text}"
            )
        return response.json()

        
    def get_job(self, job_id: str) -> Dict[str, Any]:
        """Fetch details (status/results) for a specific job."""
        if not job_id:
            raise ValueError("job_id is required")
        url = _join_url(self._rest_base, f"/document/jobs/{job_id}")
        headers = self._auth_header()
        response = self._http.get(url, headers=headers, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to get job: {response.status_code} - {response.text}")
        return response.json()

    def delete_job(self, job_id: str) -> None:
        """Delete a job and its stored artifacts (if any)."""
        if not job_id:
            raise ValueError("job_id is required")
        url = _join_url(self._rest_base, f"/document/jobs/{job_id}")
        headers = self._auth_header()
        response = self._http.delete(url, headers=headers, timeout=60)
        if response.status_code not in (200, 202, 204):
            raise RuntimeError(f"Failed to delete job: {response.status_code} - {response.text}")

    # -------------------------
    # Convenience Helpers
    # -------------------------
    def wait_for_result(
        self,
        job_id: str,
        timeout_seconds: int = 180,
        poll_interval_seconds: int = 2,
        terminal_statuses: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Poll until the job reaches a terminal state or times out.

        Returns the final job payload.

        - job_id: The ID of the job to poll
        - timeout_seconds: Max total time to wait
        - poll_interval_seconds: Delay between polls
        - terminal_statuses: Custom terminal states; defaults to
          ["SUCCEEDED", "DONE", "FAILED", "ERROR", "CANCELED", "CANCELLED"]
        """
        if terminal_statuses is None:
            # SAP DOX frequently returns "DONE" instead of "SUCCEEDED"; also handle both spellings of canceled
            terminal_statuses = ["SUCCEEDED", "DONE", "FAILED", "ERROR", "CANCELED", "CANCELLED"]

        deadline = self._now() + float(timeout_seconds)
        last_payload: Dict[str, Any] = {}
        while self._now() < deadline:
            last_payload = self.get_job(job_id)
            status = (last_payload.get("status") or last_payload.get("state") or "").upper()
            if status in terminal_statuses:
                return last_payload
            time.sleep(poll_interval_seconds)

        # If we time out, return the last payload we have for easier debugging
        raise TimeoutError(
            f"Timed out waiting for job {job_id} to finish. Last payload: {json.dumps(last_payload)})"
        )




