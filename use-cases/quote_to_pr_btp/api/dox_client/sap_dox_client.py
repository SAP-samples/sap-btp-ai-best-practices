"""
Reusable SAP Document Information Extraction (Document AI / DOX) client.

The client reads OAuth and API endpoint information from a SAP service key and
wraps the core REST API flow for clients, schemas, document upload, polling, and
document catalog operations.
"""

from __future__ import annotations

import json
import mimetypes
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests

from .models import FieldDefinition


def _join_url(base: str, path: str) -> str:
    """Join two URL parts ensuring exactly one slash in between."""
    if not base:
        return path
    if not path:
        return base
    if base.endswith("/") and path.startswith("/"):
        return base[:-1] + path
    if not base.endswith("/") and not path.startswith("/"):
        return base + "/" + path
    return base + path


def _flatten_one_level(items: Any) -> List[Any]:
    """Flatten SAP list wrappers like [[...]] into a simple list."""
    if not isinstance(items, list):
        return []
    flattened: List[Any] = []
    for item in items:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


@dataclass
class ServiceKey:
    """Structured representation of the minimal fields used from the service key."""

    token_base_url: str
    client_id: str
    client_secret: str
    dox_base_url: str
    swagger_path: str

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "ServiceKey":
        """Create a ServiceKey from a dict parsed from the service key JSON."""
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


class DoxApiError(RuntimeError):
    """Structured error raised for unsuccessful SAP Document AI responses."""

    def __init__(
        self,
        *,
        method: str,
        path: str,
        status_code: int,
        sap_code: Optional[str] = None,
        sap_message: Optional[str] = None,
        details: Any = None,
        response_text: str = "",
    ) -> None:
        self.method = method.upper()
        self.path = path
        self.status_code = status_code
        self.sap_code = sap_code
        self.sap_message = sap_message
        self.details = details
        self.response_text = response_text

        detail_text = sap_message or response_text or "No response body"
        if sap_code:
            detail_text = f"{sap_code}: {detail_text}"
        super().__init__(f"{self.method} {path} failed with {status_code}: {detail_text}")

    @classmethod
    def from_response(cls, method: str, path: str, response: requests.Response) -> "DoxApiError":
        sap_code: Optional[str] = None
        sap_message: Optional[str] = None
        details: Any = None
        response_text = getattr(response, "text", "") or ""

        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            error_payload = payload.get("error")
            if isinstance(error_payload, dict):
                sap_code = error_payload.get("code")
                sap_message = error_payload.get("message")
                details = error_payload.get("details")
            else:
                sap_code = payload.get("code")
                sap_message = payload.get("message")
                details = payload.get("details")

        return cls(
            method=method,
            path=path,
            status_code=response.status_code,
            sap_code=sap_code,
            sap_message=sap_message,
            details=details,
            response_text=response_text,
        )


class SapDoxClient:
    """Client for SAP Document Information Extraction REST API."""

    def __init__(self, service_key: ServiceKey, session: Optional[requests.Session] = None) -> None:
        self._service_key = service_key
        self._rest_base = _join_url(service_key.dox_base_url, service_key.swagger_path).rstrip("/")
        self._http = session or requests.Session()
        self._access_token: Optional[str] = None
        self._token_expires_at_epoch: float = 0.0

    @classmethod
    def from_service_key(cls, service_key_path: str) -> "SapDoxClient":
        """Instantiate client by loading a SAP Document AI service key JSON file."""
        if not os.path.isfile(service_key_path):
            raise FileNotFoundError(f"Service key file not found: {service_key_path}")
        with open(service_key_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls(ServiceKey.from_json(raw))

    # -------------------------
    # Authentication Utilities
    # -------------------------
    def _token_url(self) -> str:
        return _join_url(self._service_key.token_base_url, "/oauth/token")

    def _now(self) -> float:
        return time.time()

    def get_token(self, force_refresh: bool = False) -> str:
        """Retrieve and cache a bearer token using the client_credentials grant."""
        if not force_refresh and self._access_token and self._now() < (self._token_expires_at_epoch - 30):
            return self._access_token

        response = self._http.post(
            self._token_url(),
            data={"grant_type": "client_credentials"},
            auth=(self._service_key.client_id, self._service_key.client_secret),
            headers={"Accept": "application/json"},
            timeout=30,
        )
        if response.status_code != 200:
            raise DoxApiError.from_response("POST", "/oauth/token", response)

        payload = response.json()
        self._access_token = payload.get("access_token")
        if not self._access_token:
            raise RuntimeError("Token response missing 'access_token'")

        self._token_expires_at_epoch = self._now() + float(payload.get("expires_in", 3600))
        return self._access_token

    def _auth_header(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.get_token()}"}

    # -------------------------
    # Request Utilities
    # -------------------------
    def _request(
        self,
        method: str,
        path: str,
        *,
        expected_statuses: Sequence[int],
        params: Optional[Dict[str, Any]] = None,
        json_payload: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        timeout: int = 60,
    ) -> requests.Response:
        url = _join_url(self._rest_base, path)
        headers = self._auth_header()
        if json_payload is not None and files is None:
            headers["Content-Type"] = "application/json"

        response = self._http.request(
            method.upper(),
            url,
            headers=headers,
            params=params,
            json=json_payload,
            files=files,
            timeout=timeout,
        )
        if response.status_code not in expected_statuses:
            raise DoxApiError.from_response(method, path, response)
        return response

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        expected_statuses: Sequence[int] = (200,),
        params: Optional[Dict[str, Any]] = None,
        json_payload: Optional[Dict[str, Any]] = None,
        timeout: int = 60,
    ) -> Any:
        response = self._request(
            method,
            path,
            expected_statuses=expected_statuses,
            params=params,
            json_payload=json_payload,
            timeout=timeout,
        )
        if response.status_code == 204 or not response.content:
            return {}
        return response.json()

    def _request_multipart_json(
        self,
        method: str,
        path: str,
        *,
        files: Dict[str, Any],
        expected_statuses: Sequence[int] = (200,),
        params: Optional[Dict[str, Any]] = None,
        timeout: int = 60,
    ) -> Any:
        response = self._request(
            method,
            path,
            expected_statuses=expected_statuses,
            params=params,
            files=files,
            timeout=timeout,
        )
        if response.status_code == 204 or not response.content:
            return {}
        return response.json()

    @staticmethod
    def _normalize_wrapped_list(data: Any, primary_key: str) -> List[Dict[str, Any]]:
        if isinstance(data, dict):
            if primary_key in data:
                return _flatten_one_level(data[primary_key])
            for key in ("items", "value", "payload", "results"):
                if key in data:
                    return _flatten_one_level(data[key])
        if isinstance(data, list):
            if data and isinstance(data[0], dict) and primary_key in data[0]:
                return _flatten_one_level(data[0][primary_key])
            if data and isinstance(data[0], dict) and "payload" in data[0]:
                output: List[Dict[str, Any]] = []
                for item in data:
                    output.extend(_flatten_one_level(item.get("payload", [])))
                return output
            return _flatten_one_level(data)
        return []

    @staticmethod
    def _bool_param(value: Optional[bool]) -> Optional[str]:
        if value is None:
            return None
        return "true" if value else "false"

    @staticmethod
    def _field_name(field: Any) -> str:
        if isinstance(field, FieldDefinition):
            return field.name
        if isinstance(field, str):
            if not field:
                raise ValueError("Field names cannot be empty")
            return field
        if isinstance(field, dict):
            name = field.get("name")
            if not name or not isinstance(name, str):
                raise ValueError("Field dict items must contain a non-empty 'name' string")
            return name
        raise ValueError("Fields must be strings, FieldDefinition objects, or dicts with a 'name' key")

    @classmethod
    def _field_definition(cls, field: Any) -> Dict[str, Any]:
        if isinstance(field, FieldDefinition):
            return field.to_dict()
        if isinstance(field, str):
            return FieldDefinition(name=field).to_dict()
        if isinstance(field, dict):
            name = cls._field_name(field)
            if set(field.keys()) <= {"name"}:
                return FieldDefinition(name=name).to_dict()
            return dict(field)
        raise ValueError("Fields must be strings, FieldDefinition objects, or dicts with a 'name' key")

    @classmethod
    def _normalize_fields(cls, fields: Optional[Iterable[Any]], *, full_definitions: bool) -> Optional[List[Any]]:
        if fields is None:
            return None
        normalized = [
            cls._field_definition(field) if full_definitions else cls._field_name(field)
            for field in fields
        ]
        return normalized

    # -------------------------
    # Discovery Operations
    # -------------------------
    def get_capabilities(self) -> Dict[str, Any]:
        """Return SAP Document AI extraction capabilities."""
        return self._request_json("GET", "/capabilities")

    def get_schema_capabilities(self) -> Dict[str, Any]:
        """Return schema document types, states, setup types, and formatting capabilities."""
        return self._request_json("GET", "/schemas/capabilities")

    # -------------------------
    # Client Operations
    # -------------------------
    def create_clients(self, clients: List[Dict[str, str]]) -> Dict[str, Any]:
        """Create or update one or more SAP Document AI clients."""
        if not clients:
            raise ValueError("clients list cannot be empty")
        return self._request_json(
            "POST",
            "/clients",
            expected_statuses=(200, 201),
            json_payload={"value": clients},
        )

    def list_clients(
        self,
        limit: int = 100,
        offset: int = 0,
        client_id_starts_with: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return clients as a simple list of dictionaries."""
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if client_id_starts_with:
            params["clientIdStartsWith"] = client_id_starts_with
        data = self._request_json("GET", "/clients", params=params)
        return self._normalize_wrapped_list(data, "payload")

    def delete_clients(self, client_ids: List[str] | str) -> Dict[str, Any]:
        """Delete one or more clients."""
        normalized_ids = [client_ids] if isinstance(client_ids, str) else list(client_ids)
        if not normalized_ids:
            raise ValueError("client_ids must contain at least one client identifier")
        return self._request_json(
            "DELETE",
            "/clients",
            expected_statuses=(200, 202, 204),
            json_payload={"value": normalized_ids},
        )

    # -------------------------
    # Schema Operations
    # -------------------------
    def create_schema(
        self,
        client_id: str,
        schema_name: str,
        schema_description: Optional[str] = None,
        document_type: Optional[str] = None,
        document_type_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new schema for a client."""
        if not schema_name:
            raise ValueError("schema_name is required")

        payload: Dict[str, Any] = {"clientId": client_id, "name": schema_name}
        if schema_description is not None:
            payload["schemaDescription"] = schema_description
        if document_type is not None:
            payload["documentType"] = document_type
        if document_type_description is not None:
            payload["documentTypeDescription"] = document_type_description

        return self._request_json(
            "POST",
            "/schemas",
            expected_statuses=(200, 201),
            json_payload=payload,
        )

    def update_schema(
        self,
        schema_id: str,
        *,
        client_id: str = "default",
        name: Optional[str] = None,
        schema_description: Optional[str] = None,
        document_type_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update mutable schema metadata."""
        if not schema_id:
            raise ValueError("schema_id is required")
        payload: Dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if schema_description is not None:
            payload["schemaDescription"] = schema_description
        if document_type_description is not None:
            payload["documentTypeDescription"] = document_type_description
        if not payload:
            raise ValueError("At least one schema metadata field is required")

        return self._request_json(
            "PUT",
            f"/schemas/{schema_id}",
            expected_statuses=(200, 201),
            params={"clientId": client_id},
            json_payload=payload,
        )

    def list_schemas(
        self,
        client_id: str = "default",
        limit: int = 100,
        offset: int = 0,
        document_type: Optional[str] = None,
        order: Optional[str] = None,
        predefined: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Return schemas for a client as a simple list of dictionaries."""
        params: Dict[str, Any] = {"clientId": client_id, "limit": limit, "offset": offset}
        if document_type is not None:
            params["documentType"] = document_type
        if order is not None:
            params["order"] = order
        if predefined is not None:
            params["predefined"] = self._bool_param(predefined)

        data = self._request_json("GET", "/schemas", params=params)
        return self._normalize_wrapped_list(data, "schemas")

    def get_schema_by_name(
        self,
        schema_name: str,
        *,
        client_id: str = "default",
        limit: int = 1000,
        document_type: Optional[str] = None,
        predefined: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        """Find a schema by exact name within a client."""
        for schema in self.list_schemas(
            client_id=client_id,
            limit=limit,
            document_type=document_type,
            predefined=predefined,
        ):
            if schema.get("name") == schema_name:
                return schema
        return None

    def get_schema_details(self, schema_id: str, client_id: str = "default") -> Dict[str, Any]:
        """Return details for a specific schema by ID."""
        if not schema_id:
            raise ValueError("schema_id is required")
        return self._request_json("GET", f"/schemas/{schema_id}", params={"clientId": client_id})

    def delete_schema(self, schema_ids: List[str] | str, client_id: str = "default") -> Dict[str, Any]:
        """Delete one or more schemas using the official bulk schema endpoint."""
        normalized_ids = [schema_ids] if isinstance(schema_ids, str) else list(schema_ids)
        if not normalized_ids:
            raise ValueError("schema_ids must contain at least one schema identifier")
        return self._request_json(
            "DELETE",
            "/schemas",
            expected_statuses=(200, 202, 204),
            params={"clientId": client_id},
            json_payload={"value": normalized_ids},
        )

    def create_schema_version(self, schema_id: str, client_id: str = "default") -> Dict[str, Any]:
        """Create a new version for a schema."""
        if not schema_id:
            raise ValueError("schema_id is required")
        return self._request_json(
            "POST",
            f"/schemas/{schema_id}",
            expected_statuses=(200, 201),
            params={"clientId": client_id},
        )

    def update_schema_version(
        self,
        schema_id: str,
        version: str | int,
        *,
        client_id: str = "default",
        schema_description: str,
    ) -> Dict[str, Any]:
        """Update mutable metadata for a schema version."""
        if not schema_id:
            raise ValueError("schema_id is required")
        if version is None:
            raise ValueError("version is required")
        if not schema_description:
            raise ValueError("schema_description is required")
        return self._request_json(
            "PUT",
            f"/schemas/{schema_id}/versions/{version}",
            expected_statuses=(200, 201),
            params={"clientId": client_id},
            json_payload={"schemaDescription": schema_description},
        )

    def list_schema_versions(self, schema_id: str, client_id: str = "default") -> List[Dict[str, Any]]:
        """Return all versions for a schema as a simple list."""
        if not schema_id:
            raise ValueError("schema_id is required")
        data = self._request_json(
            "GET",
            f"/schemas/{schema_id}/versions",
            params={"clientId": client_id},
        )
        return self._normalize_wrapped_list(data, "schemas")

    def get_schema_version_details(self, schema_id: str, version: str | int, client_id: str = "default") -> Dict[str, Any]:
        """Return details for a specific schema version."""
        if not schema_id:
            raise ValueError("schema_id is required")
        if version is None:
            raise ValueError("version is required")
        return self._request_json(
            "GET",
            f"/schemas/{schema_id}/versions/{version}",
            params={"clientId": client_id},
        )

    def is_schema_version_editable(self, schema_id: str, version: str | int, client_id: str = "default") -> bool:
        """Return True when a schema version is inactive and not predefined."""
        details = self.get_schema_version_details(schema_id=schema_id, version=version, client_id=client_id)
        if _truthy(details.get("predefined", False)):
            return False
        return (details.get("state") or "").lower() == "inactive"

    def add_fields_to_schema_version(
        self,
        schema_id: str,
        version: str | int,
        client_id: str = "default",
        header_fields: Optional[List[Any]] = None,
        line_item_fields: Optional[List[Any]] = None,
        replace: Optional[bool] = None,
        full_definitions: bool = False,
    ) -> Dict[str, Any]:
        """Add fields to a schema version."""
        if not schema_id:
            raise ValueError("schema_id is required")
        if version is None:
            raise ValueError("version is required")
        if header_fields is None and line_item_fields is None:
            raise ValueError("header_fields or line_item_fields is required")

        payload: Dict[str, Any] = {}
        normalized_headers = self._normalize_fields(header_fields, full_definitions=full_definitions)
        normalized_lines = self._normalize_fields(line_item_fields, full_definitions=full_definitions)
        if normalized_headers is not None:
            payload["headerFields"] = normalized_headers
        if normalized_lines is not None:
            payload["lineItemFields"] = normalized_lines
        if replace is not None:
            payload["replace"] = bool(replace)

        return self._request_json(
            "POST",
            f"/schemas/{schema_id}/versions/{version}/fields",
            expected_statuses=(200, 201),
            params={"clientId": client_id},
            json_payload=payload,
        )

    def activate_schema_version(self, schema_id: str, version: str | int, client_id: str = "default") -> Dict[str, Any]:
        """Activate a schema version."""
        if not schema_id:
            raise ValueError("schema_id is required")
        if version is None:
            raise ValueError("version is required")
        return self._request_json(
            "POST",
            f"/schemas/{schema_id}/versions/{version}/activate",
            expected_statuses=(200, 201),
            params={"clientId": client_id},
            json_payload={},
        )

    def deactivate_schema_version(self, schema_id: str, version: str | int, client_id: str = "default") -> Dict[str, Any]:
        """Deactivate a schema version so it can be edited."""
        if not schema_id:
            raise ValueError("schema_id is required")
        if version is None:
            raise ValueError("version is required")
        return self._request_json(
            "POST",
            f"/schemas/{schema_id}/versions/{version}/deactivate",
            expected_statuses=(200, 201),
            params={"clientId": client_id},
            json_payload={},
        )

    def configure_schema_version(
        self,
        schema_id: str,
        version: str | int,
        *,
        client_id: str = "default",
        header_fields: Optional[List[Any]] = None,
        line_item_fields: Optional[List[Any]] = None,
        replace: Optional[bool] = None,
        activate: bool = True,
    ) -> Dict[str, Any]:
        """Deactivate if needed, add full field definitions, and optionally activate."""
        details = self.get_schema_version_details(schema_id, version, client_id=client_id)
        if _truthy(details.get("predefined", False)):
            raise ValueError("Predefined schema versions cannot be configured")

        state = (details.get("state") or "").lower()
        if state not in {"inactive", "draft"}:
            self.deactivate_schema_version(schema_id, version, client_id=client_id)

        field_result = self.add_fields_to_schema_version(
            schema_id,
            version,
            client_id=client_id,
            header_fields=header_fields,
            line_item_fields=line_item_fields,
            replace=replace,
            full_definitions=True,
        )
        if not activate:
            return field_result
        return self.activate_schema_version(schema_id, version, client_id=client_id)

    # -------------------------
    # Document Operations
    # -------------------------
    def upload_document(
        self,
        file_path: str,
        client_id: str = "default",
        schema_id: Optional[str] = None,
        schema_name: Optional[str] = None,
        schema_version: Optional[str | int] = None,
        template_id: Optional[str] = None,
        header_fields: Optional[List[Any]] = None,
        line_item_fields: Optional[List[Any]] = None,
        mime_type: Optional[str] = None,
        document_type: Optional[str] = None,
        received_date: Optional[str] = None,
        custom_label: Optional[str] = None,
        enrichment: Optional[Dict[str, Any]] = None,
        candidate_template_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Upload a document and create an extraction job."""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if schema_id and schema_name:
            raise ValueError("Provide either schema_id or schema_name, not both")

        has_schema = bool(schema_id or schema_name)
        has_ad_hoc_fields = bool(header_fields or line_item_fields)
        if has_schema and has_ad_hoc_fields:
            raise ValueError("schema_id/schema_name cannot be combined with header_fields/line_item_fields")
        if template_id and not has_schema:
            raise ValueError("template_id requires schema_id or schema_name")
        if candidate_template_ids and not template_id:
            raise ValueError("candidate_template_ids requires template_id='detect' or a template_id")
        if not has_schema and not has_ad_hoc_fields:
            raise ValueError("Must provide either schema_id/schema_name or header_fields/line_item_fields")

        if not mime_type:
            mime_type = mimetypes.guess_type(file_path)[0] or "application/pdf"

        options: Dict[str, Any] = {"clientId": client_id}
        if document_type:
            options["documentType"] = document_type
        elif has_ad_hoc_fields:
            options["documentType"] = "custom"

        if received_date:
            options["receivedDate"] = received_date
        if custom_label:
            options["customLabel"] = custom_label
        if enrichment is not None:
            options["enrichment"] = enrichment

        if has_schema:
            if schema_id:
                options["schemaId"] = schema_id
            if schema_name:
                options["schemaName"] = schema_name
            if schema_version is not None:
                options["schemaVersion"] = str(schema_version)
        else:
            extraction: Dict[str, Any] = {}
            header_names = self._normalize_fields(header_fields, full_definitions=False)
            line_names = self._normalize_fields(line_item_fields, full_definitions=False)
            if header_names:
                extraction["headerFields"] = header_names
            if line_names:
                extraction["lineItemFields"] = line_names
            options["extraction"] = extraction

        if template_id:
            options["templateId"] = template_id
        if candidate_template_ids:
            options["candidateTemplateIds"] = candidate_template_ids

        with open(file_path, "rb") as f:
            files = {
                "file": (os.path.basename(file_path), f, mime_type),
                "options": (None, json.dumps(options), "application/json"),
            }
            return self._request_multipart_json(
                "POST",
                "/document/jobs",
                expected_statuses=(200, 201, 202),
                files=files,
                timeout=120,
            )

    def get_job(
        self,
        job_id: str,
        *,
        extracted_values: Optional[bool] = None,
        return_null_values: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Fetch details and extraction results for a document job."""
        if not job_id:
            raise ValueError("job_id is required")
        params: Dict[str, Any] = {}
        extracted = self._bool_param(extracted_values)
        nulls = self._bool_param(return_null_values)
        if extracted is not None:
            params["extractedValues"] = extracted
        if nulls is not None:
            params["returnNullValues"] = nulls
        return self._request_json(
            "GET",
            f"/document/jobs/{job_id}",
            params=params or None,
        )

    def list_documents(self, client_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return up to 200 processed documents, optionally filtered by client."""
        params = {"clientId": client_id} if client_id else None
        data = self._request_json("GET", "/document/jobs", params=params)
        return self._normalize_wrapped_list(data, "results")

    def search_document_catalog(
        self,
        *,
        client_id: Optional[str] = None,
        filter_query: Optional[str] = None,
        like_filter: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search and page through the document catalog."""
        options: Dict[str, Any] = {}
        if client_id is not None:
            options["clientId"] = client_id
        if filter_query is not None:
            options["filter"] = filter_query
        if like_filter is not None:
            options["likeFilter"] = like_filter
        if limit is not None:
            options["limit"] = limit
        if offset is not None:
            options["offset"] = offset
        if order is not None:
            options["order"] = order

        files = {"options": (None, json.dumps(options), "application/json")}
        return self._request_multipart_json(
            "POST",
            "/document/catalog",
            expected_statuses=(200,),
            files=files,
        )

    def delete_jobs(self, job_ids: List[str] | str) -> Dict[str, Any]:
        """Delete one or more document jobs."""
        normalized_ids = [job_ids] if isinstance(job_ids, str) else list(job_ids)
        if not normalized_ids:
            raise ValueError("job_ids must contain at least one document job identifier")
        return self._request_json(
            "DELETE",
            "/document/jobs",
            expected_statuses=(200, 202, 204),
            json_payload={"value": normalized_ids},
        )

    def delete_job(self, job_id: str) -> Dict[str, Any]:
        """Delete a single document job."""
        if not job_id:
            raise ValueError("job_id is required")
        return self.delete_jobs(job_id)

    # -------------------------
    # Convenience Helpers
    # -------------------------
    def wait_for_result(
        self,
        job_id: str,
        timeout_seconds: int = 180,
        poll_interval_seconds: int = 2,
        terminal_statuses: Optional[List[str]] = None,
        extracted_values: Optional[bool] = None,
        return_null_values: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Poll until a document job reaches a terminal state or times out."""
        if terminal_statuses is None:
            terminal_statuses = ["SUCCEEDED", "DONE", "FAILED", "ERROR", "CANCELED", "CANCELLED", "CONFIRMED"]

        deadline = self._now() + float(timeout_seconds)
        last_payload: Dict[str, Any] = {}
        while self._now() < deadline:
            last_payload = self.get_job(
                job_id,
                extracted_values=extracted_values,
                return_null_values=return_null_values,
            )
            status = (last_payload.get("status") or last_payload.get("state") or "").upper()
            if status in terminal_statuses:
                return last_payload
            time.sleep(poll_interval_seconds)

        raise TimeoutError(
            f"Timed out waiting for job {job_id} to finish. Last payload: {json.dumps(last_payload)}"
        )
