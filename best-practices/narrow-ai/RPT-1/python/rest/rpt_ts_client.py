"""Direct REST client for the SAP-RPT-TS time-series model on SAP AI Core.

SAP-RPT-TS (``sap-rpt-ts``) is deployed like the tabular RPT models (scenario
``foundation-models``, executable ``aicore-sap``), so authentication and
deployment discovery are inherited from ``RPTClientAPI``. Only the inference
contract differs:

* Request: ``{"horizon": int, "data": [series, ...], "options": {...}}``.
* Response: forecast arrays (``result``, ``confidence_interval_output``, ...)
  plus ``usage``; there is no ``status.code`` envelope like the tabular models.
* Errors: FastAPI style ``{"detail": ...}`` with HTTP 400 (business rules) or
  422 (schema validation).

Example:
    from rpt_ts_client import RPTTSClientAPI, forecast_to_frame

    client = RPTTSClientAPI()
    response = client.forecast({"horizon": 3, "data": [{"target": [1, 2, 3, 4]}]})
    frame = forecast_to_frame(response)
"""

from __future__ import annotations

import gzip
import json
import time
from collections.abc import Sequence
from typing import Any

import pandas as pd
import requests

from rpt_client_api import RPTAPIError, RPTClientAPI

MAX_RATE_LIMIT_RETRIES = 5  # retries after HTTP 429 Too Many Requests


class RPTTSClientAPI(RPTClientAPI):
    """Call one running ``sap-rpt-ts`` deployment through AI Core REST.

    Args:
        model_name: Model configuration name. Defaults to ``sap-rpt-ts``.
        model_version: Exact configured model version. Defaults to ``1``.
        timeout_s: Timeout applied to authentication, discovery, and inference.

    Raises:
        RPTAPIError: If environment configuration or deployment discovery fails.
    """

    def __init__(
        self,
        model_name: str = "sap-rpt-ts",
        model_version: str = "1",
        timeout_s: int = 120,
    ) -> None:
        """Resolve the time-series deployment with the inherited discovery logic."""
        super().__init__(
            model_name=model_name, model_version=model_version, timeout_s=timeout_s
        )

    def forecast(
        self, payload: dict[str, Any], *, compress: bool = False
    ) -> dict[str, Any]:
        """Submit a forecast request to the deployment's ``/predict`` route.

        Args:
            payload: Forecast request with ``horizon``, ``data`` and optional
                ``options``. Unknown fields are rejected by the service.
            compress: Gzip the serialized JSON body when true (useful for many
                or long series).

        Returns:
            The unmodified forecast response dictionary.

        Raises:
            RPTAPIError: On transport failures or HTTP errors. The server's
                ``detail`` message is included (with credentials redacted).
                HTTP 429 (rate limit) is retried up to ``MAX_RATE_LIMIT_RETRIES``
                times, waiting ``Retry-After`` seconds when the server sends it.
        """
        url = self.deployment_url + "/predict"
        body = gzip.compress(json.dumps(payload).encode("utf-8"), 1) if compress else None
        # ponytail: fixed retry budget for HTTP 429 only; add jitter/backoff
        # policy if many clients share one deployment.
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            headers = self._headers(content_type="application/json")
            try:
                if compress:
                    response = self._session.post(
                        url,
                        headers={**headers, "Content-Encoding": "gzip"},
                        data=body,
                        timeout=self.timeout_s,
                    )
                else:
                    response = self._session.post(
                        url, headers=headers, json=payload, timeout=self.timeout_s
                    )
            except requests.RequestException as exc:
                raise RPTAPIError("RPT-TS /predict transport failed.") from exc
            if response.status_code != 429 or attempt == MAX_RATE_LIMIT_RETRIES:
                break
            # Rate limited: wait as long as the server asks (default 10 s, 20 s, ...).
            retry_after = getattr(response, "headers", {}).get("Retry-After")
            time.sleep(float(retry_after) if retry_after else 10.0 * (attempt + 1))

        # The parent parser requires the tabular ``status.code`` envelope, which
        # RPT-TS does not return, so HTTP status alone decides success here.
        try:
            parsed = response.json()
        except ValueError:
            parsed = None
        if not response.ok:
            detail = parsed.get("detail") if isinstance(parsed, dict) else response.text
            raise RPTAPIError(
                f"RPT-TS /predict failed with HTTP {response.status_code}: "
                f"{self._redacted_excerpt(detail)}",
                response_payload=self._redacted_payload(parsed)
                if isinstance(parsed, dict)
                else None,
                http_status=response.status_code,
            )
        if not isinstance(parsed, dict):
            raise RPTAPIError(
                "RPT-TS /predict returned a non-object JSON response.",
                http_status=response.status_code,
            )
        return parsed


def forecast_to_frame(
    response: dict[str, Any], series_ids: Sequence[Any] | None = None
) -> pd.DataFrame:
    """Flatten an RPT-TS response into one row per series, variate and step.

    The service returns nested arrays: ``result[series][variate][step]`` and
    ``confidence_interval_output[series][variate][0=lower|1=upper][step]``.

    Args:
        response: Successful ``RPTTSClientAPI.forecast`` response.
        series_ids: Optional labels for the series, in request order. Defaults
            to the zero-based series position.

    Returns:
        DataFrame with columns ``series``, ``variate``, ``step``, ``timestamp``,
        ``point``, ``lower`` and ``upper``. ``timestamp`` is ``None`` when the
        request had no timestamps; ``lower``/``upper`` are ``None`` when no
        confidence interval was returned. Empty for hierarchical forecasts,
        whose values live in ``hierarchical_result`` instead.
    """
    results = response.get("result") or []
    intervals = response.get("confidence_interval_output") or []
    timestamps = response.get("result_timestamp") or []
    records = []
    for s, variates in enumerate(results):
        label = series_ids[s] if series_ids is not None else s
        series_ts = timestamps[s] if s < len(timestamps) and timestamps[s] else None
        for v, points in enumerate(variates):
            bounds = intervals[s][v] if s < len(intervals) else None
            for step, point in enumerate(points):
                records.append(
                    {
                        "series": label,
                        "variate": v,
                        "step": step + 1,
                        "timestamp": series_ts[step] if series_ts else None,
                        "point": point,
                        "lower": bounds[0][step] if bounds else None,
                        "upper": bounds[1][step] if bounds else None,
                    }
                )
    return pd.DataFrame(
        records,
        columns=["series", "variate", "step", "timestamp", "point", "lower", "upper"],
    )


def hierarchy_to_frame(response: dict[str, Any]) -> pd.DataFrame:
    """Flatten the reconciled tree of a ``hierarchical_forecast`` response.

    Each tree node is ``{label: [[values per step] per variate], "__children__":
    [...]}``; leaf nodes have no ``__children__``. When ``hierarchical_levels``
    starts with ``null`` the root label is ``""`` (the grand total).

    Args:
        response: Forecast response requested with
            ``options.point_forecast = "hierarchical_forecast"``.

    Returns:
        DataFrame with columns ``level`` (0 = grand total), ``path`` (labels
        joined by `` / ``, ``"total"`` for the root), ``variate``, ``step`` and
        ``point``.
    """
    records = []

    def walk(nodes: Sequence[dict[str, Any]], path: list[str]) -> None:
        """Visit nodes depth-first and append one record per variate and step."""
        for node in nodes:
            label = next(key for key in node if key != "__children__")
            node_path = path + [label] if label else path
            for variate, values in enumerate(node[label]):
                for step, value in enumerate(values):
                    records.append(
                        {
                            "level": len(node_path),
                            "path": " / ".join(node_path) or "total",
                            "variate": variate,
                            "step": step + 1,
                            "point": value,
                        }
                    )
            walk(node.get("__children__") or [], node_path)

    walk(response.get("hierarchical_result") or [], [])
    return pd.DataFrame(records, columns=["level", "path", "variate", "step", "point"])
