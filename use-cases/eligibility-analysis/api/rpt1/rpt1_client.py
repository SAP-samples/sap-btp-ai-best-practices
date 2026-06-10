from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd
import requests
from dotenv import load_dotenv
from pandas.api import types as ptypes


class RPT1Error(Exception):
    """Base class for RPT-1 client errors."""


class RPT1ValidationError(RPT1Error):
    """Raised when fit/predict inputs are invalid."""


class RPT1AuthError(RPT1Error):
    """Raised when authentication fails."""


class RPT1RequestError(RPT1Error):
    """Raised when an HTTP/API request fails."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


@dataclass
class PredictionResult:
    predictions_df: pd.DataFrame
    metadata: dict[str, Any]
    status_code: int | None
    status_message: str
    request_id: str | None
    raw_response: dict[str, Any]


class RPT1Client:
    """DataFrame-first client for SAP RPT-1 inferencing through SAP AI Core."""

    _ALLOWED_TASK_TYPES = {"classification", "regression"}
    _ALLOWED_SCHEMA_TYPES = {"string", "numeric", "date"}
    _RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(
        self,
        *,
        aicore_base_url: str,
        aicore_auth_url: str,
        client_id: str,
        client_secret: str,
        resource_group: str = "default",
        deployment_url: str | None = None,
        deployment_id: str | None = None,
        timeout_seconds: int = 60,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
        session: requests.Session | None = None,
    ) -> None:
        if not aicore_base_url:
            raise RPT1ValidationError("aicore_base_url is required.")
        if not aicore_auth_url:
            raise RPT1ValidationError("aicore_auth_url is required.")
        if not client_id:
            raise RPT1ValidationError("client_id is required.")
        if not client_secret:
            raise RPT1ValidationError("client_secret is required.")
        if timeout_seconds <= 0:
            raise RPT1ValidationError("timeout_seconds must be > 0.")
        if max_retries < 0:
            raise RPT1ValidationError("max_retries must be >= 0.")
        if retry_backoff_seconds < 0:
            raise RPT1ValidationError("retry_backoff_seconds must be >= 0.")

        self.aicore_base_url = self._normalize_api_base_url(aicore_base_url)
        self.aicore_auth_url = aicore_auth_url.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.resource_group = resource_group
        self.deployment_url = deployment_url.rstrip("/") if deployment_url else None
        self.deployment_id = deployment_id
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.session = session or requests.Session()

        self._access_token: str | None = None
        self._access_token_expiry: float = 0.0

        self._context_df: pd.DataFrame | None = None
        self._target_columns: list[str] = []
        self._index_column: str | None = None
        self._task_types: dict[str, str] = {}
        self._data_schema: dict[str, dict[str, str]] | None = None
        self._prediction_placeholder: str = "[PREDICT]"

    @classmethod
    def from_env(
        cls,
        *,
        env_path: str | None = ".env",
        override: bool = False,
        timeout_seconds: int = 60,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
        deployment_url: str | None = None,
        deployment_id: str | None = None,
        resource_group: str | None = None,
        session: requests.Session | None = None,
    ) -> "RPT1Client":
        if env_path is not None:
            load_dotenv(dotenv_path=env_path, override=override)

        aicore_base_url = os.getenv("AICORE_BASE_URL", "")
        aicore_auth_url = os.getenv("AICORE_AUTH_URL", "")
        client_id = os.getenv("AICORE_CLIENT_ID", "")
        client_secret = os.getenv("AICORE_CLIENT_SECRET", "")

        resolved_resource_group = (
            resource_group
            if resource_group is not None
            else os.getenv("AICORE_RESOURCE_GROUP", "default")
        )
        resolved_deployment_url = (
            deployment_url
            if deployment_url is not None
            else os.getenv("RPT1_DEPLOYMENT_URL", "") or None
        )
        resolved_deployment_id = (
            deployment_id
            if deployment_id is not None
            else os.getenv("RPT1_DEPLOYMENT_ID", "") or None
        )

        return cls(
            aicore_base_url=aicore_base_url,
            aicore_auth_url=aicore_auth_url,
            client_id=client_id,
            client_secret=client_secret,
            resource_group=resolved_resource_group,
            deployment_url=resolved_deployment_url,
            deployment_id=resolved_deployment_id,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            session=session,
        )

    def fit(
        self,
        context_df: pd.DataFrame,
        target_columns: Sequence[str],
        *,
        index_column: str | None = None,
        task_types: Mapping[str, str] | None = None,
        data_schema: Mapping[str, Mapping[str, str]] | None = None,
        prediction_placeholder: str = "[PREDICT]",
    ) -> "RPT1Client":
        if not isinstance(context_df, pd.DataFrame):
            raise RPT1ValidationError("context_df must be a pandas DataFrame.")
        if context_df.empty:
            raise RPT1ValidationError("context_df must contain at least one row.")
        # target_columns can be a single string or a sequence of strings. Normalize to list.
        targets = self._normalize_target_columns(target_columns)
        if index_column is not None and index_column not in context_df.columns:
            raise RPT1ValidationError(
                f"index_column '{index_column}' is missing from context_df."
            )
        missing_targets = [col for col in targets if col not in context_df.columns]
        if missing_targets:
            raise RPT1ValidationError(
                f"target_columns missing from context_df: {missing_targets}"
            )

        if not isinstance(prediction_placeholder, str) or not prediction_placeholder:
            raise RPT1ValidationError("prediction_placeholder must be a non-empty string.")

        normalized_task_types = self._normalize_task_types(targets, task_types)
        normalized_data_schema = self._normalize_data_schema(data_schema)

        self._context_df = context_df.copy(deep=True)
        self._target_columns = targets
        self._index_column = index_column
        self._task_types = normalized_task_types
        self._data_schema = normalized_data_schema
        self._prediction_placeholder = prediction_placeholder

        return self

    def predict(
        self,
        query_df: pd.DataFrame,
        *,
        parse_data_types: bool = True,
    ) -> PredictionResult:
        self._ensure_fitted()
        payload, _ = self._build_prediction_payload(
            query_df, parse_data_types=parse_data_types
        )

        deployment_url = self._resolve_deployment_url()
        response = self._request(
            "POST",
            f"{deployment_url}/predict",
            payload=payload,
        )

        status = response.get("status", {})
        status_code = status.get("code")
        status_message = str(status.get("message", ""))
        if status_code in {2, 3}:
            raise RPT1RequestError(
                f"RPT-1 prediction failed with status code {status_code}: {status_message}",
                status_code=status_code,
                response_body=str(response),
            )

        predictions_df = self._parse_predictions(response, query_df)
        return PredictionResult(
            predictions_df=predictions_df,
            metadata=dict(response.get("metadata", {})),
            status_code=status_code,
            status_message=status_message,
            request_id=response.get("id"),
            raw_response=response,
        )

    def _build_prediction_payload(
        self,
        query_df: pd.DataFrame,
        *,
        parse_data_types: bool = True,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        self._ensure_fitted()
        self._validate_query_df(query_df)

        assert self._context_df is not None
        context_columns = list(self._context_df.columns)

        query_aligned = query_df.copy(deep=True)
        for target in self._target_columns:
            query_aligned[target] = self._prediction_placeholder

        for column in context_columns:
            if column not in query_aligned.columns:
                query_aligned[column] = None
        query_aligned = query_aligned[context_columns]

        combined_df = pd.concat(
            [self._context_df[context_columns], query_aligned], ignore_index=True
        )

        payload: dict[str, Any] = {
            "prediction_config": {
                "target_columns": self._build_target_column_config(),
            },
            "rows": self._df_to_json_rows(combined_df),
            "parse_data_types": bool(parse_data_types),
        }
        if self._index_column:
            payload["index_column"] = self._index_column
        payload["data_schema"] = (
            self._data_schema
            if self._data_schema is not None
            else self._infer_data_schema(self._context_df)
        )

        return payload, query_aligned

    def _parse_predictions(
        self,
        response_json: Mapping[str, Any],
        query_df: pd.DataFrame,
    ) -> pd.DataFrame:
        predictions = response_json.get("predictions", [])
        if not isinstance(predictions, list):
            raise RPT1RequestError("Invalid response format: 'predictions' must be a list.")

        rows: list[dict[str, Any]] = []
        for idx, item in enumerate(predictions):
            if not isinstance(item, Mapping):
                continue
            row: dict[str, Any] = {}

            if self._index_column:
                fallback_value = (
                    query_df.iloc[idx][self._index_column]
                    if idx < len(query_df) and self._index_column in query_df.columns
                    else None
                )
                row[self._index_column] = item.get(self._index_column, fallback_value)

            for target in self._target_columns:
                prediction_value = None
                confidence_value = None
                target_payload = item.get(target)

                if isinstance(target_payload, list) and target_payload:
                    first_value = target_payload[0]
                    if isinstance(first_value, Mapping):
                        prediction_value = first_value.get("prediction")
                        confidence_value = first_value.get("confidence")
                    else:
                        prediction_value = first_value
                elif isinstance(target_payload, Mapping):
                    prediction_value = target_payload.get("prediction")
                    confidence_value = target_payload.get("confidence")

                row[target] = prediction_value
                row[f"{target}__confidence"] = confidence_value

            rows.append(row)

        ordered_columns: list[str] = []
        if self._index_column:
            ordered_columns.append(self._index_column)
        for target in self._target_columns:
            ordered_columns.append(target)
            ordered_columns.append(f"{target}__confidence")

        if not rows:
            return pd.DataFrame(columns=ordered_columns)
        return pd.DataFrame(rows, columns=ordered_columns)

    def _resolve_deployment_url(self) -> str:
        if self.deployment_url:
            return self.deployment_url.rstrip("/")

        if self.deployment_id:
            payload = self._request(
                "GET",
                f"{self.aicore_base_url}/v2/lm/deployments/{self.deployment_id}",
            )
            deployment_url = payload.get("deploymentUrl")
            if not deployment_url:
                raise RPT1RequestError(
                    f"Deployment '{self.deployment_id}' does not contain 'deploymentUrl'.",
                    response_body=str(payload),
                )
            self.deployment_url = str(deployment_url).rstrip("/")
            return self.deployment_url

        raise RPT1ValidationError(
            "No deployment configured. Set deployment_url, RPT1_DEPLOYMENT_URL, or RPT1_DEPLOYMENT_ID."
        )

    def _request(
        self,
        method: str,
        url: str,
        *,
        payload: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    headers=self._build_headers(),
                    json=payload,
                    params=params,
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise RPT1RequestError(
                        f"Request failed after retries: {exc}"
                    ) from exc
                self._sleep_before_retry(attempt)
                continue

            if response.status_code == 401:
                self._access_token = None
                self._access_token_expiry = 0.0
                if attempt >= self.max_retries:
                    raise RPT1AuthError("Unauthorized (401). Please check credentials.")
                self._sleep_before_retry(attempt)
                continue

            if response.status_code == 403:
                raise RPT1AuthError("Forbidden (403). Check permissions and resource group.")

            if response.status_code in self._RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                self._sleep_before_retry(attempt)
                continue

            if not response.ok:
                raise RPT1RequestError(
                    f"HTTP {response.status_code} for {method.upper()} {url}",
                    status_code=response.status_code,
                    response_body=response.text[:2000],
                )

            if not response.text.strip():
                return {}
            try:
                return response.json()
            except ValueError as exc:
                raise RPT1RequestError(
                    f"Response from {url} is not valid JSON."
                ) from exc

        if last_error is not None:
            raise RPT1RequestError(f"Request failed after retries: {last_error}") from last_error
        raise RPT1RequestError("Request failed after retries.")

    def _build_headers(self) -> dict[str, str]:
        token = self._get_access_token()
        return {
            "Authorization": f"Bearer {token}",
            "AI-Resource-Group": self.resource_group,
            "Content-Type": "application/json",
        }

    def _get_access_token(self) -> str:
        if self._access_token and time.time() < self._access_token_expiry:
            return self._access_token

        response = self.session.post(
            f"{self.aicore_auth_url}/oauth/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=self.timeout_seconds,
        )

        if not response.ok:
            raise RPT1AuthError(
                f"Token request failed with HTTP {response.status_code}: {response.text[:1000]}"
            )

        payload = response.json()
        access_token = payload.get("access_token")
        if not access_token:
            raise RPT1AuthError("Authentication response missing 'access_token'.")

        expires_in = int(payload.get("expires_in", 3600))
        self._access_token = access_token
        # Subtract a safety buffer to refresh token before it expires.
        self._access_token_expiry = time.time() + max(expires_in - 30, 1)
        return access_token

    def _build_target_column_config(self) -> list[dict[str, str]]:
        target_config: list[dict[str, str]] = []
        for name in self._target_columns:
            item: dict[str, str] = {
                "name": name,
                "prediction_placeholder": self._prediction_placeholder,
            }
            task_type = self._task_types.get(name)
            if task_type:
                item["task_type"] = task_type
            target_config.append(item)
        return target_config

    def _infer_data_schema(self, df: pd.DataFrame) -> dict[str, dict[str, str]]:
        schema: dict[str, dict[str, str]] = {}
        for column in df.columns:
            series = df[column]
            if ptypes.is_datetime64_any_dtype(series):
                dtype = "date"
            elif ptypes.is_numeric_dtype(series):
                dtype = "numeric"
            else:
                dtype = "string"
            schema[column] = {"dtype": dtype}
        return schema

    def _df_to_json_rows(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        serializable = df.copy(deep=True)
        for column in serializable.columns:
            if ptypes.is_datetime64_any_dtype(serializable[column]):
                serializable[column] = serializable[column].dt.strftime("%Y-%m-%d")
        serializable = serializable.astype(object).where(pd.notna(serializable), None)
        return serializable.to_dict(orient="records")

    def _validate_query_df(self, query_df: pd.DataFrame) -> None:
        if not isinstance(query_df, pd.DataFrame):
            raise RPT1ValidationError("query_df must be a pandas DataFrame.")
        if query_df.empty:
            raise RPT1ValidationError("query_df must contain at least one row.")

        assert self._context_df is not None
        expected_features = [
            col for col in self._context_df.columns if col not in self._target_columns
        ]
        missing_features = [col for col in expected_features if col not in query_df.columns]
        if missing_features:
            raise RPT1ValidationError(
                f"query_df is missing required feature columns: {missing_features}"
            )

        unknown_columns = [
            col for col in query_df.columns if col not in self._context_df.columns
        ]
        if unknown_columns:
            raise RPT1ValidationError(
                f"query_df contains unknown columns not present in context_df: {unknown_columns}"
            )

    def _ensure_fitted(self) -> None:
        if self._context_df is None or not self._target_columns:
            raise RPT1ValidationError("Client is not fitted. Call fit(...) before predict(...).")

    @staticmethod
    def _normalize_api_base_url(url: str) -> str:
        normalized = url.strip().rstrip("/")
        if normalized.endswith("/v2"):
            return normalized[:-3]
        return normalized

    @staticmethod
    def _normalize_target_columns(target_columns: Sequence[str]) -> list[str]:
        if isinstance(target_columns, str):
            target_list = [target_columns]
        else:
            target_list = list(target_columns)
        if not target_list:
            raise RPT1ValidationError("target_columns must contain at least one column.")
        if len(set(target_list)) != len(target_list):
            raise RPT1ValidationError("target_columns contains duplicates.")
        return target_list

    def _normalize_task_types(
        self,
        target_columns: Sequence[str],
        task_types: Mapping[str, str] | None,
    ) -> dict[str, str]:
        """Normalize task types for target columns. Validates that task types are only provided for known target columns and that they are valid."""
        if task_types is None:
            return {}

        normalized: dict[str, str] = {}
        for key, value in task_types.items():
            if key not in target_columns:
                raise RPT1ValidationError(
                    f"task_types contains unknown target column '{key}'."
                )
            value_normalized = str(value).strip().lower()
            if value_normalized not in self._ALLOWED_TASK_TYPES:
                raise RPT1ValidationError(
                    f"Invalid task_type '{value}' for column '{key}'. "
                    f"Allowed: {sorted(self._ALLOWED_TASK_TYPES)}"
                )
            normalized[key] = value_normalized
        return normalized

    def _normalize_data_schema(
        self,
        data_schema: Mapping[str, Mapping[str, str]] | None,
    ) -> dict[str, dict[str, str]] | None:
        if data_schema is None:
            return None

        normalized: dict[str, dict[str, str]] = {}
        for column, definition in data_schema.items():
            dtype = definition.get("dtype") if isinstance(definition, Mapping) else None
            if dtype not in self._ALLOWED_SCHEMA_TYPES:
                raise RPT1ValidationError(
                    f"Invalid data_schema dtype for column '{column}': {dtype}. "
                    f"Allowed: {sorted(self._ALLOWED_SCHEMA_TYPES)}"
                )
            normalized[column] = {"dtype": str(dtype)}
        return normalized

    def _sleep_before_retry(self, attempt: int) -> None:
        # Exponential backoff: 1x, 2x, 4x, ... based on retry attempt.
        delay = self.retry_backoff_seconds * (2 ** attempt)
        if delay > 0:
            time.sleep(delay)
