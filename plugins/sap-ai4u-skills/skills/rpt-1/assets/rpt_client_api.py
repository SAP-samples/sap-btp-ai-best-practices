"""Reusable direct REST client for SAP RPT-1.5 and RPT-1.6 on AI Core.

Copy this file into a project that already provides ``requests``,
``requests-toolbelt``, and ``python-dotenv``. The client deliberately accepts and
returns dictionaries so newer RPT request and response fields are not discarded
by an older SDK model.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import time
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import requests
from dotenv import find_dotenv, load_dotenv
from requests_toolbelt import MultipartEncoder

_REQUIRED_ENVIRONMENT = (
    "AICORE_AUTH_URL",
    "AICORE_CLIENT_ID",
    "AICORE_CLIENT_SECRET",
    "AICORE_BASE_URL",
    "AICORE_RESOURCE_GROUP",
)
_APPLICATION_ERROR_CODES = {2, 3}


class RPTAPIError(RuntimeError):
    """Describe an RPT authentication, discovery, transport, or API failure.

    Args:
        message: Safe, user-facing failure description.
        response_payload: Parsed response body when the server returned JSON.
        http_status: HTTP status code when a response was received.

    Attributes:
        response_payload: Parsed response body, if available.
        http_status: HTTP status code, if available.
    """

    def __init__(
        self,
        message: str,
        *,
        response_payload: dict[str, Any] | None = None,
        http_status: int | None = None,
    ) -> None:
        """Initialize an error with safe text and optional response diagnostics."""
        super().__init__(message)
        self.response_payload = response_payload
        self.http_status = http_status


def _normalize_api_v2_url(base_url: str) -> str:
    """Return an AI Core API base URL ending exactly once in ``/v2``.

    Args:
        base_url: Value of ``AICORE_BASE_URL`` with or without a ``/v2`` suffix.

    Returns:
        Normalized URL without a trailing slash.
    """
    normalized = base_url.rstrip("/")
    return normalized if normalized.endswith("/v2") else normalized + "/v2"


def _safe_excerpt(value: object, limit: int = 2_000) -> str:
    """Return a bounded server-response excerpt suitable for an exception.

    Args:
        value: Response text or another printable error detail.
        limit: Maximum number of characters retained.

    Returns:
        Bounded string with a truncation marker when necessary.
    """
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "... (truncated)"


def _status_code(payload: Mapping[str, Any]) -> int:
    """Read the required RPT application status code from an inference response.

    Args:
        payload: Parsed RPT inference response.

    Returns:
        Integer application status code.

    Raises:
        RPTAPIError: If the response does not contain a numeric ``status.code``.
    """
    status = payload.get("status")
    if not isinstance(status, Mapping) or "code" not in status:
        raise RPTAPIError(
            "RPT response did not contain the required status.code.",
            response_payload=dict(payload),
        )
    try:
        return int(status["code"])
    except (TypeError, ValueError) as exc:
        raise RPTAPIError(
            "RPT response contained a non-numeric status.code.",
            response_payload=dict(payload),
        ) from exc


def _source_rows(
    source_data: Sequence[Mapping[str, Any]] | Mapping[str, Sequence[Any]],
) -> list[dict[str, Any]]:
    """Convert row- or column-oriented input into addressable row dictionaries.

    Args:
        source_data: Original JSON ``rows`` list or ``columns`` mapping sent to RPT.

    Returns:
        Rows in their exact original order.

    Raises:
        RPTAPIError: If column arrays have different lengths or input is malformed.
    """
    if isinstance(source_data, Mapping):
        columns = list(source_data.items())
        if not columns:
            return []
        if any(
            isinstance(values, (str, bytes)) or not isinstance(values, Sequence)
            for _, values in columns
        ):
            raise RPTAPIError("Column-oriented source data must contain arrays.")
        lengths = {len(values) for _, values in columns}
        if len(lengths) != 1:
            raise RPTAPIError("Column-oriented source arrays must have equal lengths.")
        row_count = lengths.pop()
        return [
            {name: values[position] for name, values in columns}
            for position in range(row_count)
        ]

    if isinstance(source_data, (str, bytes)) or not isinstance(source_data, Sequence):
        raise RPTAPIError("Source data must be a rows list or columns mapping.")
    if any(not isinstance(row, Mapping) for row in source_data):
        raise RPTAPIError("Every row in source data must be a mapping.")
    return [dict(row) for row in source_data]


def flatten_predictions(
    response: Mapping[str, Any], *, index_column: str | None = None
) -> list[dict[str, Any]]:
    """Flatten all query, target, and top-k candidates into long-form records.

    Args:
        response: Raw successful RPT response.
        index_column: Optional key copied from each prediction record as ``index``.

    Returns:
        Records containing query position, target, rank, prediction, confidence,
        confidence interval, and optional source index.

    Raises:
        RPTAPIError: If the predictions member has an unexpected structure.
    """
    predictions = response.get("predictions")
    if not isinstance(predictions, list):
        raise RPTAPIError(
            "RPT response predictions must be a list.", response_payload=dict(response)
        )

    flattened: list[dict[str, Any]] = []
    for query_position, prediction_record in enumerate(predictions):
        if not isinstance(prediction_record, Mapping):
            raise RPTAPIError(
                "Each RPT prediction must be a mapping.",
                response_payload=dict(response),
            )
        for target, candidates in prediction_record.items():
            if target == index_column:
                continue
            if not isinstance(candidates, list) or not all(
                isinstance(candidate, Mapping) for candidate in candidates
            ):
                continue
            for rank, candidate in enumerate(candidates, start=1):
                row: dict[str, Any] = {
                    "query_position": query_position,
                    "target": target,
                    "rank": rank,
                    "prediction": candidate.get("prediction"),
                    "confidence": candidate.get("confidence"),
                    "confidence_interval": candidate.get("confidence_interval"),
                }
                if index_column:
                    row["index"] = prediction_record.get(index_column)
                    # Keep the stable column order promised by the template docs.
                    row = {
                        "query_position": row["query_position"],
                        "index": row["index"],
                        "target": row["target"],
                        "rank": row["rank"],
                        "prediction": row["prediction"],
                        "confidence": row["confidence"],
                        "confidence_interval": row["confidence_interval"],
                    }
                flattened.append(row)
    return flattened


def map_explanation_rows(
    response: Mapping[str, Any],
    source_data: Sequence[Mapping[str, Any]] | Mapping[str, Sequence[Any]],
) -> list[dict[str, Any]]:
    """Resolve explanation row positions against the complete original input.

    Args:
        response: Raw successful RPT response with optional explanations.
        source_data: Original row- or column-oriented data in request order.

    Returns:
        One record per explained query with column scores and resolved context rows.

    Raises:
        RPTAPIError: If explanation arrays or referenced positions are malformed.
    """
    explanations = response.get("explanations") or {}
    if not isinstance(explanations, Mapping):
        raise RPTAPIError("RPT response explanations must be a mapping.")
    column_scores = explanations.get("top_column_scores") or []
    relevant_positions = explanations.get("top_relevant_context_rows") or []
    if not isinstance(column_scores, list) or not isinstance(relevant_positions, list):
        raise RPTAPIError("RPT explanation values must be arrays ordered by query row.")

    rows = _source_rows(source_data)
    mapped: list[dict[str, Any]] = []
    for query_position in range(max(len(column_scores), len(relevant_positions))):
        scores = (
            column_scores[query_position] if query_position < len(column_scores) else {}
        )
        positions = (
            relevant_positions[query_position]
            if query_position < len(relevant_positions)
            else []
        )
        if not isinstance(scores, Mapping) or not isinstance(positions, list):
            raise RPTAPIError(
                "Each RPT explanation entry must match the documented structure."
            )
        resolved = []
        for position in positions:
            if not isinstance(position, int) or position < 0 or position >= len(rows):
                raise RPTAPIError(
                    f"Explanation row position {position!r} is outside the source data."
                )
            resolved.append({"position": position, "row": rows[position]})
        mapped.append(
            {
                "query_position": query_position,
                "column_scores": dict(scores),
                "relevant_context_rows": resolved,
            }
        )
    return mapped


class RPTClientAPI:
    """Call one automatically resolved SAP RPT deployment through AI Core REST.

    Args:
        model_name: RPT model configuration name. Defaults to ``sap-rpt-1.6``.
        model_version: Exact configured model version. Defaults to ``1``.
        timeout_s: Timeout applied to authentication, discovery, and inference.

    Raises:
        RPTAPIError: If environment configuration or deployment discovery fails.
    """

    def __init__(
        self,
        model_name: str = "sap-rpt-1.6",
        model_version: str = "1",
        timeout_s: int = 120,
    ) -> None:
        """Load environment configuration and resolve one running deployment."""
        # Resolve from the consuming process, not from this copied template's path.
        load_dotenv(find_dotenv(usecwd=True), override=False)
        missing = [name for name in _REQUIRED_ENVIRONMENT if not os.getenv(name)]
        if missing:
            raise RPTAPIError(
                "Missing required environment variables: " + ", ".join(missing)
            )

        self.model_name = model_name
        self.model_version = str(model_version)
        self.timeout_s = timeout_s
        self.resource_group = os.environ["AICORE_RESOURCE_GROUP"]
        self._auth_url = os.environ["AICORE_AUTH_URL"].rstrip("/") + "/oauth/token"
        self._client_id = os.environ["AICORE_CLIENT_ID"]
        self._client_secret = os.environ["AICORE_CLIENT_SECRET"]
        self._api_v2_url = _normalize_api_v2_url(os.environ["AICORE_BASE_URL"])
        self._session = requests.Session()
        self._token: str | None = None
        self._token_expires_at = 0.0
        self.deployment_url = self._resolve_deployment_url()

    def _access_token(self) -> str:
        """Return a cached OAuth token, refreshing it before expiry."""
        if self._token and time.monotonic() < self._token_expires_at:
            return self._token
        try:
            response = self._session.post(
                self._auth_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            raise RPTAPIError(
                "AI Core OAuth token request failed.", http_status=status
            ) from exc

        token = payload.get("access_token") if isinstance(payload, Mapping) else None
        if not isinstance(token, str) or not token:
            raise RPTAPIError("AI Core OAuth response did not contain access_token.")
        try:
            expires_in = int(payload.get("expires_in", 300))
        except (TypeError, ValueError):
            expires_in = 300
        self._token = token
        self._token_expires_at = time.monotonic() + max(0, expires_in - 60)
        return token

    def _headers(self, *, content_type: str | None = None) -> dict[str, str]:
        """Build authenticated AI Core headers without logging credentials."""
        headers = {
            "Authorization": f"Bearer {self._access_token()}",
            "AI-Resource-Group": self.resource_group,
        }
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    def _get_resources(
        self, path: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Fetch one AI Core management resource list used for deployment discovery."""
        url = self._api_v2_url + path
        try:
            response = self._session.get(
                url,
                headers=self._headers(),
                params=params,
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            raise RPTAPIError(
                f"AI Core deployment discovery failed for {path}.", http_status=status
            ) from exc
        resources = payload.get("resources") if isinstance(payload, Mapping) else None
        if not isinstance(resources, list):
            raise RPTAPIError(
                f"AI Core {path} response did not contain a resources list."
            )
        return [resource for resource in resources if isinstance(resource, dict)]

    @staticmethod
    def _parameter_bindings(configuration: Mapping[str, Any]) -> dict[str, str]:
        """Convert an AI Core configuration's parameter bindings into a mapping."""
        bindings = configuration.get("parameterBindings") or []
        if isinstance(bindings, Mapping):
            return {str(key): str(value) for key, value in bindings.items()}
        if not isinstance(bindings, list):
            return {}
        return {
            str(binding.get("key")): str(binding.get("value"))
            for binding in bindings
            if isinstance(binding, Mapping) and binding.get("key") is not None
        }

    def _resolve_deployment_url(self) -> str:
        """Find the unique running deployment matching model name and version."""
        configurations = self._get_resources(
            "/lm/configurations",
            {
                "scenarioId": "foundation-models",
                "executableIds": "aicore-sap",
                "$top": 1_000,
            },
        )
        configuration_ids = {
            str(configuration["id"])
            for configuration in configurations
            if configuration.get("id") is not None
            and self._parameter_bindings(configuration).get("modelName")
            == self.model_name
            and self._parameter_bindings(configuration).get("modelVersion")
            == self.model_version
        }
        if not configuration_ids:
            raise RPTAPIError(
                f"No AI Core configuration matched {self.model_name!r} version {self.model_version!r}."
            )

        deployments = self._get_resources("/lm/deployments", {"$top": 1_000})
        matches = []
        for deployment in deployments:
            status = deployment.get("status")
            if isinstance(status, Mapping):
                status = status.get("status") or status.get("value")
            if (
                str(deployment.get("configurationId")) in configuration_ids
                and str(status).upper() == "RUNNING"
            ):
                matches.append(deployment)
        if not matches:
            raise RPTAPIError(
                f"No running AI Core deployment matched {self.model_name!r} version {self.model_version!r}."
            )
        if len(matches) > 1:
            ids = ", ".join(str(match.get("id", "unknown")) for match in matches)
            raise RPTAPIError(
                f"Found multiple running deployments for {self.model_name!r} version "
                f"{self.model_version!r}: {ids}."
            )
        deployment = matches[0]
        deployment_url = deployment.get("deploymentUrl")
        if isinstance(deployment_url, str) and deployment_url:
            return deployment_url.rstrip("/")
        deployment_id = deployment.get("id")
        if not deployment_id:
            raise RPTAPIError(
                "Matched AI Core deployment did not contain an id or deploymentUrl."
            )
        return f"{self._api_v2_url}/inference/deployments/{deployment_id}"

    def _validate_prediction_config(self, prediction_config: Mapping[str, Any]) -> None:
        """Restrict RPT-1.6 context modes to compatible model deployments."""
        context_mode = prediction_config.get("context_mode")
        if context_mode is not None and self.model_name not in {
            "sap-rpt-1.6",
            "sap-rpt-1.6-large",
        }:
            raise RPTAPIError("context_mode is supported only by RPT-1.6 models.")
        if context_mode == "deep" and self.model_name != "sap-rpt-1.6-large":
            raise RPTAPIError("context_mode='deep' requires model sap-rpt-1.6-large.")

    def _redacted_excerpt(self, value: object) -> str:
        """Bound server text and redact credentials that an upstream might echo."""
        excerpt = str(value)
        for secret in (self._client_id, self._client_secret, self._token):
            if secret:
                excerpt = excerpt.replace(secret, "[REDACTED]")
        return _safe_excerpt(excerpt)

    def _redacted_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return an error payload with configured credentials recursively redacted."""

        def redact(value: Any) -> Any:
            """Redact strings while preserving nested JSON-compatible structure."""
            if isinstance(value, dict):
                return {key: redact(item) for key, item in value.items()}
            if isinstance(value, list):
                return [redact(item) for item in value]
            if isinstance(value, str):
                for secret in (self._client_id, self._client_secret, self._token):
                    if secret:
                        value = value.replace(secret, "[REDACTED]")
            return value

        redacted = redact(payload)
        return payload if redacted == payload else redacted

    def _parse_inference_response(
        self, response: Any, operation: str
    ) -> dict[str, Any]:
        """Parse one inference response and enforce HTTP/application failures."""
        try:
            parsed = response.json()
        except ValueError:
            parsed = None
        try:
            response.raise_for_status()
        except requests.RequestException as exc:
            error_payload = (
                self._redacted_payload(parsed) if isinstance(parsed, dict) else None
            )
            excerpt = self._redacted_excerpt(
                parsed if parsed is not None else getattr(response, "text", "")
            )
            raise RPTAPIError(
                f"RPT {operation} failed with HTTP {response.status_code}: {excerpt}",
                response_payload=error_payload,
                http_status=response.status_code,
            ) from exc
        if parsed is None:
            raise RPTAPIError(
                f"RPT {operation} returned malformed JSON.",
                http_status=response.status_code,
            )
        if not isinstance(parsed, dict):
            raise RPTAPIError(
                f"RPT {operation} returned a non-object JSON response.",
                http_status=response.status_code,
            )
        payload = parsed
        diagnostic_payload = self._redacted_payload(payload)
        code = _status_code(diagnostic_payload)
        if code in _APPLICATION_ERROR_CODES:
            error_payload = diagnostic_payload
            message = error_payload.get("status", {}).get(
                "message", "RPT application error"
            )
            detail = error_payload.get("detail")
            suffix = f": {self._redacted_excerpt(detail)}" if detail else ""
            raise RPTAPIError(
                f"RPT {operation} failed with status {code} ({self._redacted_excerpt(message)}){suffix}",
                response_payload=error_payload,
                http_status=response.status_code,
            )
        return payload

    def predict_json(
        self, payload: dict[str, Any], *, compress: bool = False
    ) -> dict[str, Any]:
        """Submit row- or column-oriented JSON to ``/predict``.

        Args:
            payload: Complete RPT request dictionary.
            compress: Gzip the serialized JSON body when true.

        Returns:
            Unmodified successful RPT response dictionary.

        Raises:
            RPTAPIError: If deep mode is incompatible or the request fails.
        """
        prediction_config = payload.get("prediction_config")
        if not isinstance(prediction_config, Mapping):
            raise RPTAPIError("JSON payload must contain a prediction_config mapping.")
        self._validate_prediction_config(prediction_config)
        url = self.deployment_url + "/predict"
        try:
            if compress:
                body = gzip.compress(
                    json.dumps(payload).encode("utf-8"), compresslevel=1
                )
                response = self._session.post(
                    url,
                    headers={
                        **self._headers(content_type="application/json"),
                        "Content-Encoding": "gzip",
                    },
                    data=body,
                    timeout=self.timeout_s,
                )
            else:
                response = self._session.post(
                    url,
                    headers=self._headers(content_type="application/json"),
                    json=payload,
                    timeout=self.timeout_s,
                )
        except requests.RequestException as exc:
            raise RPTAPIError("RPT /predict transport failed.") from exc
        return self._parse_inference_response(response, "/predict")

    async def apredict_json(
        self, payload: dict[str, Any], *, compress: bool = False
    ) -> dict[str, Any]:
        """Run ``predict_json`` without blocking the caller's event loop.

        Args:
            payload: Complete RPT request dictionary.
            compress: Gzip the serialized JSON body when true.

        Returns:
            Unmodified successful RPT response dictionary.
        """
        return await asyncio.to_thread(
            partial(self.predict_json, payload, compress=compress)
        )

    def predict_parquet(
        self,
        parquet_path: str | Path,
        prediction_config: dict[str, Any],
        *,
        index_column: str | None = None,
        parse_data_types: bool = True,
    ) -> dict[str, Any]:
        """Upload a Parquet table to ``/predict_parquet`` as multipart form data.

        Args:
            parquet_path: Existing Parquet file containing context and query rows.
            prediction_config: Target, explanation, and context-mode configuration.
            index_column: Optional column copied into prediction records.
            parse_data_types: Ask RPT to infer source data types when true.

        Returns:
            Unmodified successful RPT response dictionary.

        Raises:
            RPTAPIError: If the file is absent, deep mode is incompatible, or the
            request fails.
        """
        self._validate_prediction_config(prediction_config)
        path = Path(parquet_path)
        if not path.is_file():
            raise RPTAPIError(f"Parquet file does not exist: {path}")
        url = self.deployment_url + "/predict_parquet"
        try:
            with path.open("rb") as parquet_file:
                fields: dict[str, Any] = {
                    "file": (path.name, parquet_file, "application/octet-stream"),
                    "prediction_config": json.dumps(prediction_config),
                    "parse_data_types": "true" if parse_data_types else "false",
                }
                if index_column:
                    fields["index_column"] = index_column
                encoder = MultipartEncoder(fields=fields)
                response = self._session.post(
                    url,
                    headers=self._headers(content_type=encoder.content_type),
                    data=encoder,
                    timeout=self.timeout_s,
                )
        except requests.RequestException as exc:
            raise RPTAPIError("RPT /predict_parquet transport failed.") from exc
        return self._parse_inference_response(response, "/predict_parquet")

    async def apredict_parquet(
        self,
        parquet_path: str | Path,
        prediction_config: dict[str, Any],
        *,
        index_column: str | None = None,
        parse_data_types: bool = True,
    ) -> dict[str, Any]:
        """Run ``predict_parquet`` without blocking the caller's event loop.

        Args:
            parquet_path: Existing Parquet file containing context and query rows.
            prediction_config: Target, explanation, and context-mode configuration.
            index_column: Optional column copied into prediction records.
            parse_data_types: Ask RPT to infer source data types when true.

        Returns:
            Unmodified successful RPT response dictionary.
        """
        call = partial(
            self.predict_parquet,
            parquet_path,
            prediction_config,
            index_column=index_column,
            parse_data_types=parse_data_types,
        )
        return await asyncio.to_thread(call)
