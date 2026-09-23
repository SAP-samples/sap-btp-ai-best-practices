"""Self-contained direct-REST client for SAP TabPFN (tabpfn-3.5-plus) on AI Core.

Copy this single file into a project that provides ``requests`` and
``python-dotenv`` (and ``pandas`` if you use the DataFrame convenience method).
The SAP AI SDK has no TabPFN wrapper, so this client talks to the deployment's
``/predict`` endpoint directly, reusing AI Core's standard OAuth +
``AI-Resource-Group`` transport.

Contract (validated live against the deployment):

    POST {deploymentUrl}/predict
    {
      "X_train": [[f1, f2, ...], ...],   # labelled context rows (list of lists)
      "y_train": [label, ...],            # target aligned to X_train
      "X_test":  [[f1, f2, ...], ...],    # rows to predict
      "task_config": {                    # strict: unknown top-level keys -> 400
        "task": "classification" | "regression",
        "tabpfn_config": { ... }          # optional hyper-parameters (nested here)
      }
    }

TabPFN is an in-context learner: training and test data go in one request; there
is no separate fit step. See the skill's references/ for the full contract,
settings, and limits.

Required environment (loaded from a .env in the current working directory):

    AICORE_AUTH_URL, AICORE_CLIENT_ID, AICORE_CLIENT_SECRET,
    AICORE_BASE_URL, AICORE_RESOURCE_GROUP

Optional escape hatch (skip discovery, target a deployment directly):

    AICORE_TABPFN_DEPLOYMENT_URL   full inference URL, or
    AICORE_TABPFN_DEPLOYMENT_ID    deployment id (URL built from AICORE_BASE_URL)
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping
from typing import Any

import requests
from dotenv import find_dotenv, load_dotenv

_REQUIRED_ENVIRONMENT = (
    "AICORE_AUTH_URL",
    "AICORE_CLIENT_ID",
    "AICORE_CLIENT_SECRET",
    "AICORE_BASE_URL",
    "AICORE_RESOURCE_GROUP",
)


class AICoreError(RuntimeError):
    """An AI Core / TabPFN authentication, discovery, or transport failure.

    Args:
        message: Safe, user-facing description (never contains credentials).
        http_status: HTTP status code when a response was received.
    """

    def __init__(self, message: str, *, http_status: int | None = None) -> None:
        super().__init__(message)
        self.http_status = http_status


def _normalize_api_v2_url(base_url: str) -> str:
    """Return an AI Core API base URL ending exactly once in ``/v2``."""
    normalized = base_url.rstrip("/")
    return normalized if normalized.endswith("/v2") else normalized + "/v2"


def _safe_excerpt(value: object, limit: int = 2_000) -> str:
    """Bound an arbitrary server response to a printable, length-capped string."""
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "... (truncated)"


class AICoreTransport:
    """Authenticated HTTP access to one AI Core resource group.

    Construction only loads and validates configuration; the OAuth token is
    fetched lazily and cached (refreshed 60s before expiry).

    Args:
        timeout_s: Timeout applied to every auth, management, and inference call.
    """

    def __init__(self, timeout_s: int = 120) -> None:
        load_dotenv(find_dotenv(usecwd=True), override=False)
        missing = [name for name in _REQUIRED_ENVIRONMENT if not os.getenv(name)]
        if missing:
            raise AICoreError(
                "Missing required environment variables: " + ", ".join(missing)
            )
        self.timeout_s = timeout_s
        self.resource_group = os.environ["AICORE_RESOURCE_GROUP"]
        self._auth_url = os.environ["AICORE_AUTH_URL"].rstrip("/") + "/oauth/token"
        self._client_id = os.environ["AICORE_CLIENT_ID"]
        self._client_secret = os.environ["AICORE_CLIENT_SECRET"]
        self.api_v2_url = _normalize_api_v2_url(os.environ["AICORE_BASE_URL"])
        self._session = requests.Session()
        self._token: str | None = None
        self._token_expires_at = 0.0

    def access_token(self) -> str:
        """Return a cached OAuth bearer token, refreshing before expiry."""
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
            raise AICoreError("AI Core OAuth token request failed.", http_status=status) from exc
        token = payload.get("access_token") if isinstance(payload, Mapping) else None
        if not isinstance(token, str) or not token:
            raise AICoreError("AI Core OAuth response did not contain access_token.")
        try:
            expires_in = int(payload.get("expires_in", 300))
        except (TypeError, ValueError):
            expires_in = 300
        self._token = token
        self._token_expires_at = time.monotonic() + max(0, expires_in - 60)
        return token

    def headers(self, *, content_type: str | None = None) -> dict[str, str]:
        """Build the standard authenticated AI Core header pair."""
        built = {
            "Authorization": f"Bearer {self.access_token()}",
            "AI-Resource-Group": self.resource_group,
        }
        if content_type:
            built["Content-Type"] = content_type
        return built

    def redact(self, value: object) -> str:
        """Bound server text and strip any echoed credentials before display."""
        excerpt = str(value)
        for secret in (self._client_id, self._client_secret, self._token):
            if secret:
                excerpt = excerpt.replace(secret, "[REDACTED]")
        return _safe_excerpt(excerpt)

    def _get_resources(self, path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        """GET one ``/v2/lm/...`` list endpoint and return its ``resources`` array."""
        url = self.api_v2_url + path
        try:
            response = self._session.get(
                url, headers=self.headers(), params=params, timeout=self.timeout_s
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            raise AICoreError(f"AI Core resource fetch failed for {path}.", http_status=status) from exc
        resources = payload.get("resources") if isinstance(payload, Mapping) else None
        if not isinstance(resources, list):
            raise AICoreError(f"AI Core {path} response had no resources list.")
        return [item for item in resources if isinstance(item, dict)]

    @staticmethod
    def _parameter_bindings(configuration: Mapping[str, Any]) -> dict[str, str]:
        """Normalise a configuration's ``parameterBindings`` into a flat mapping."""
        bindings = configuration.get("parameterBindings") or []
        if isinstance(bindings, Mapping):
            return {str(k): str(v) for k, v in bindings.items()}
        if not isinstance(bindings, list):
            return {}
        return {
            str(b.get("key")): str(b.get("value"))
            for b in bindings
            if isinstance(b, Mapping) and b.get("key") is not None
        }

    @staticmethod
    def _status(deployment: Mapping[str, Any]) -> str:
        """Return a deployment's status as an upper-case string (e.g. RUNNING)."""
        status = deployment.get("status")
        if isinstance(status, Mapping):
            status = status.get("status") or status.get("value")
        return str(status).upper()

    def resolve_deployment_url(self, model_name: str, model_version: str | None = None) -> str:
        """Resolve the inference URL of a running deployment for ``model_name``.

        Order: env escape hatch -> a RUNNING deployment whose configuration binds
        ``modelName`` (optionally ``modelVersion``) -> a RUNNING deployment whose
        scenario / config name / model binding mentions ``model_name`` (loose,
        covers custom deployments without a modelName binding).

        Raises:
            AICoreError: On zero or multiple matches.
        """
        explicit_url = os.getenv("AICORE_TABPFN_DEPLOYMENT_URL")
        if explicit_url:
            return explicit_url.rstrip("/")
        explicit_id = os.getenv("AICORE_TABPFN_DEPLOYMENT_ID")
        if explicit_id:
            return f"{self.api_v2_url}/inference/deployments/{explicit_id}"

        configs = {
            str(c["id"]): c
            for c in self._get_resources("/lm/configurations", {"$top": 1_000})
            if c.get("id") is not None
        }
        needle = model_name.lower()
        matches: list[dict[str, Any]] = []
        for dep in self._get_resources("/lm/deployments", {"$top": 1_000}):
            if self._status(dep) != "RUNNING":
                continue
            cfg = configs.get(str(dep.get("configurationId")), {})
            binding = self._parameter_bindings(cfg)
            exact = binding.get("modelName") == model_name and (
                model_version is None or binding.get("modelVersion") in (None, str(model_version))
            )
            haystack = " ".join(
                str(x) for x in (cfg.get("scenarioId"), cfg.get("name"), binding.get("modelName"))
            ).lower()
            if exact or needle in haystack:
                matches.append(dep)
        if not matches:
            raise AICoreError(f"No running deployment matched {model_name!r}.")
        if len(matches) > 1:
            ids = ", ".join(str(m.get("id", "?")) for m in matches)
            raise AICoreError(f"Multiple running deployments matched {model_name!r}: {ids}.")
        url = matches[0].get("deploymentUrl")
        if isinstance(url, str) and url:
            return url.rstrip("/")
        dep_id = matches[0].get("id")
        if not dep_id:
            raise AICoreError("Matched deployment had neither deploymentUrl nor id.")
        return f"{self.api_v2_url}/inference/deployments/{dep_id}"

    def post_json(self, url: str, body: Any) -> requests.Response:
        """Authenticated JSON POST to an absolute URL; returns the raw response."""
        return self._session.post(
            url,
            headers=self.headers(content_type="application/json"),
            json=body,
            timeout=self.timeout_s,
        )


class TabPFNClient:
    """Call the ``tabpfn-3.5-plus`` AI Core deployment through direct REST.

    Args:
        model_name: Deployment/model name to resolve. Defaults to ``tabpfn-3.5-plus``.
        model_version: Optional exact configured version to disambiguate.
        timeout_s: Timeout for auth, discovery, and inference calls.

    Raises:
        AICoreError: If configuration, auth, or deployment resolution fails.
    """

    def __init__(
        self,
        model_name: str = "tabpfn-3.5-plus",
        model_version: str | None = None,
        timeout_s: int = 120,
    ) -> None:
        self.transport = AICoreTransport(timeout_s=timeout_s)
        self.deployment_url = self.transport.resolve_deployment_url(model_name, model_version)
        self._predict_url = self.deployment_url + "/predict"

    def predict_raw(
        self,
        x_train: list[list[Any]],
        y_train: list[Any],
        x_test: list[list[Any]],
        task: str,
        *,
        tabpfn_config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """POST one /predict request and return the parsed response dict.

        Args:
            x_train: Labelled context rows (list of feature lists).
            y_train: Targets aligned to ``x_train``.
            x_test: Rows to predict (same column order as ``x_train``).
            task: ``"classification"`` or ``"regression"``.
            tabpfn_config: Optional TabPFN hyper-parameters nested under
                ``task_config.tabpfn_config`` (e.g. ``n_estimators`` (<=8),
                ``random_state``, ``categorical_features_indices``).

        Returns:
            The raw response dict (``prediction``, ``metadata``, ``usage``).

        Raises:
            AICoreError: On HTTP error or a non-JSON/non-object response.
        """
        task_config: dict[str, Any] = {"task": task}
        if tabpfn_config:
            task_config["tabpfn_config"] = dict(tabpfn_config)
        body = {"X_train": x_train, "y_train": y_train, "X_test": x_test, "task_config": task_config}
        response = self.transport.post_json(self._predict_url, body)
        try:
            parsed = response.json()
        except ValueError:
            parsed = None
        if response.status_code >= 400 or not isinstance(parsed, dict):
            excerpt = self.transport.redact(parsed if parsed is not None else response.text)
            raise AICoreError(
                f"TabPFN /predict failed with HTTP {response.status_code}: {excerpt}",
                http_status=response.status_code,
            )
        return parsed

    def predict(
        self,
        train_df: "Any",
        query_df: "Any",
        target: str,
        task: str,
        *,
        feature_columns: list[str] | None = None,
        tabpfn_config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Predict ``target`` for ``query_df`` using ``train_df`` as context (pandas).

        Args:
            train_df: DataFrame with feature columns and a known ``target``.
            query_df: DataFrame to predict (feature columns; any ``target`` ignored).
            target: Target column name in ``train_df``.
            task: ``"classification"`` or ``"regression"``.
            feature_columns: Explicit feature order; defaults to every column of
                ``train_df`` except ``target``.
            tabpfn_config: Optional TabPFN hyper-parameters (see ``predict_raw``).

        Returns:
            Classification: ``{"labels", "probabilities", "classes", "index", "raw"}``
            (labels = argmax class per query row). Regression:
            ``{"predictions", "index", "raw"}``.

        Raises:
            AICoreError: On request failure.
            ValueError: If ``task`` is unsupported.
        """
        if task not in ("classification", "regression"):
            raise ValueError(f"Unsupported task: {task!r}")
        features = feature_columns or [c for c in train_df.columns if c != target]
        x_train = _rows_to_json(train_df[features])
        y_train = _values_to_json(train_df[target])
        x_test = _rows_to_json(query_df[features])
        raw = self.predict_raw(x_train, y_train, x_test, task, tabpfn_config=tabpfn_config)
        index = list(query_df.index)
        if task == "classification":
            classes = raw["metadata"]["classes"]
            probabilities = raw["prediction"]
            labels = [classes[_argmax(row)] for row in probabilities]
            return {
                "labels": labels,
                "probabilities": probabilities,
                "classes": classes,
                "index": index,
                "raw": raw,
            }
        return {"predictions": raw["prediction"], "index": index, "raw": raw}


def _argmax(values: list[float]) -> int:
    """Return the index of the largest value (ties resolve to the first)."""
    return max(range(len(values)), key=values.__getitem__)


def _clean(value: Any) -> Any:
    """Convert a cell to a JSON-safe value (NaN/NaT -> None, numpy -> native).

    ``requests`` serialises ``float('nan')`` to the literal ``NaN`` token, which
    is invalid JSON and rejected by strict servers, so missing values become
    ``None`` (JSON ``null``). TabPFN accepts nulls natively.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    item = getattr(value, "item", None)
    if callable(item):
        try:
            native = value.item()
        except (ValueError, AttributeError):
            native = value
        if isinstance(native, float) and math.isnan(native):
            return None
        return native
    return value


def _rows_to_json(frame: "Any") -> list[list[Any]]:
    """Convert a feature DataFrame into a JSON-safe list of row lists."""
    return [[_clean(v) for v in row] for row in frame.to_numpy(dtype=object).tolist()]


def _values_to_json(series: "Any") -> list[Any]:
    """Convert a target Series into a JSON-safe list."""
    return [_clean(v) for v in series.to_numpy(dtype=object).tolist()]
