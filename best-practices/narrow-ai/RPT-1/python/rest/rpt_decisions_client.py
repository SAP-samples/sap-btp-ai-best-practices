"""Direct REST client for the SAP-RPT decisions model (``sap-rpt-decisions``) on SAP AI Core.

``sap-rpt-decisions`` is deployed like the tabular RPT models (scenario
``foundation-models``, executable ``aicore-sap``), so authentication and
deployment discovery are inherited from ``RPTClientAPI``. Only the inference
contract differs:

* Route: ``POST {deployment_url}/systemone``. The AI Core gateway rejects every
  other subpath for this model (``/predict``, ``/forecast``, ...) with HTTP 400.
* Request: ``{"state": ..., "questions": {...}, "examples": [...]}``, the same
  body as the decision API of SAP's RPT Playground. One ``state`` per request;
  unknown fields (for example ``states``) are rejected with HTTP 422.
* Response: ``{"answers": {question_key: {...}}, "metadata": {...}, "model":
  ..., "usage": {...}}``. There is no ``status.code`` on success.
* Errors: HTTP 400 / 422 with ``{"detail": [...], "status": {"code": 2}}``.
  HTTP 429 means AI Core rate-limited the request; it is retried here.

Example:
    from rpt_decisions_client import RPTDecisionsClientAPI, answers_to_frame

    client = RPTDecisionsClientAPI()
    result = client.decide(request)                      # one state
    results = client.decide_many(request, states=texts)  # one call per state
    frame = answers_to_frame(results)
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
import requests
from tqdm.auto import tqdm

from rpt_client_api import RPTAPIError, RPTClientAPI

DECISIONS_ROUTE = "/systemone"  # the only subpath AI Core forwards for this model
MAX_RATE_LIMIT_RETRIES = 5  # retries after HTTP 429 Too Many Requests
DEFAULT_CONCURRENCY = 4  # parallel requests in decide_many; higher values hit 429 sooner


class RPTDecisionsClientAPI(RPTClientAPI):
    """Call one running ``sap-rpt-decisions`` deployment through AI Core REST.

    Args:
        model_name: Model configuration name. Defaults to ``sap-rpt-decisions``.
        model_version: Exact configured model version. Defaults to ``1``.
        timeout_s: Timeout applied to authentication, discovery, and inference.

    Raises:
        RPTAPIError: If environment configuration or deployment discovery fails.
    """

    def __init__(
        self,
        model_name: str = "sap-rpt-decisions",
        model_version: str = "1",
        timeout_s: int = 120,
    ) -> None:
        """Resolve the decisions deployment with the inherited discovery logic."""
        super().__init__(
            model_name=model_name, model_version=model_version, timeout_s=timeout_s
        )

    def decide(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Answer every question of a decision request for its single ``state``.

        Args:
            request: Decision request with ``state``, ``questions`` and
                ``examples``. The service validates it (1-10 questions, 2-2,048
                examples, ``instructions`` on every question, every option shown
                in at least one example) and rejects unknown fields.

        Returns:
            The unmodified response: ``answers`` (one entry per question key),
            ``metadata``, ``model`` and ``usage``.

        Raises:
            RPTAPIError: On transport failures or HTTP errors. The server's
                ``detail`` message is included (with credentials redacted).
                HTTP 429 is retried up to ``MAX_RATE_LIMIT_RETRIES`` times,
                waiting ``Retry-After`` seconds when the server sends it.
        """
        url = self.deployment_url + DECISIONS_ROUTE
        # ponytail: fixed retry budget for HTTP 429 only; add jitter if many
        # clients share one deployment.
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            try:
                response = self._session.post(
                    url,
                    headers=self._headers(content_type="application/json"),
                    json=dict(request),
                    timeout=self.timeout_s,
                )
            except requests.RequestException as exc:
                raise RPTAPIError(f"RPT {DECISIONS_ROUTE} transport failed.") from exc
            if response.status_code != 429 or attempt == MAX_RATE_LIMIT_RETRIES:
                break
            # Rate limited: wait as long as the server asks (default 10 s, 20 s, ...).
            retry_after = getattr(response, "headers", {}).get("Retry-After")
            time.sleep(float(retry_after) if retry_after else 10.0 * (attempt + 1))

        # The parent parser requires a ``status.code`` envelope, which successful
        # decision responses do not carry, so HTTP status alone decides success.
        try:
            parsed = response.json()
        except ValueError:
            parsed = None
        if not response.ok:
            detail = parsed.get("detail") if isinstance(parsed, dict) else None
            if detail is None:
                detail = parsed.get("message") if isinstance(parsed, dict) else response.text
            raise RPTAPIError(
                f"RPT {DECISIONS_ROUTE} failed with HTTP {response.status_code}: "
                f"{self._redacted_excerpt(detail)}",
                response_payload=self._redacted_payload(parsed)
                if isinstance(parsed, dict)
                else None,
                http_status=response.status_code,
            )
        if not isinstance(parsed, dict) or not isinstance(parsed.get("answers"), dict):
            raise RPTAPIError(
                f"RPT {DECISIONS_ROUTE} returned no answers object.",
                http_status=response.status_code,
            )
        return parsed

    def decide_many(
        self,
        request: Mapping[str, Any],
        states: Sequence[Any],
        *,
        max_concurrency: int = DEFAULT_CONCURRENCY,
        progress: bool = True,
    ) -> list[dict[str, Any]]:
        """Answer the same questions for many states, one request per state.

        The service accepts one ``state`` per request, so this sends one call
        per state with the same ``questions`` and ``examples``, a few at a time.
        Threads are used instead of ``asyncio`` so it also works inside Jupyter.

        Args:
            request: Decision request; its own ``state`` (if any) is replaced.
            states: Inputs to decide about, in the order the results should have.
            max_concurrency: Requests in flight at the same time.
            progress: Show a ``tqdm`` progress bar.

        Returns:
            One ``decide`` response per state, in the order of ``states``.

        Raises:
            RPTAPIError: From the first request that fails after retries.
        """
        base = {key: value for key, value in request.items() if key != "state"}
        self._access_token()  # fetch the token once before the threads start

        def one(state: Any) -> dict[str, Any]:
            """Decide for one state with the shared questions and examples."""
            return self.decide({**base, "state": state})

        with ThreadPoolExecutor(max_workers=max(1, max_concurrency)) as pool:
            # pool.map keeps the input order and re-raises the first error.
            results = pool.map(one, states)
            return list(
                tqdm(results, total=len(states), desc="decisions", disable=not progress)
            )


def most_likely(answer: Mapping[str, Any]) -> str | bool:
    """Return the single most likely answer of one question.

    Args:
        answer: One entry of a response's ``answers``.

    Returns:
        ``True``/``False`` for ``noul`` (probability at least 0.5), the option
        key for ``choice`` and the level key (``"0"``, ``"1"``, ...) with the
        highest probability for ``score``.
    """
    if answer["type"] == "noul":
        return answer["noul"] >= 0.5
    if answer["type"] == "choice":
        return answer["choice"]
    probabilities = answer["probabilities"]
    return max(probabilities, key=probabilities.get)


def answers_to_frame(
    results: Sequence[Mapping[str, Any]], index: Sequence[Any] | None = None
) -> pd.DataFrame:
    """Flatten decision responses into one row per state.

    Columns per question key ``q``:

    * ``noul``: ``q`` (probability of yes).
    * ``choice``: ``q`` (winning option) and ``q_confidence``.
    * ``score``: ``q`` (expected level), ``q_level`` (most likely level) and
      ``q_confidence``.

    Args:
        results: ``decide`` or ``decide_many`` responses.
        index: Optional row labels, one per result.

    Returns:
        DataFrame with one row per result. The full probability distributions
        stay in the raw responses. Confidence is copied from the endpoint,
        never derived from probabilities; missing confidence stays missing.
        Its semantics can differ from the Playground API documentation.
    """
    rows = []
    for result in results:
        row: dict[str, Any] = {}
        for key, answer in result["answers"].items():
            if answer["type"] == "noul":
                row[key] = answer["noul"]
            elif answer["type"] == "choice":
                row[key] = answer["choice"]
                row[f"{key}_confidence"] = answer.get("confidence")
            else:
                row[key] = answer["score"]
                row[f"{key}_level"] = most_likely(answer)
                row[f"{key}_confidence"] = answer.get("confidence")
        rows.append(row)
    return pd.DataFrame(rows, index=list(index) if index is not None else None)
