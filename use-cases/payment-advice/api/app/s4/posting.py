"""Safe posting of one S/4 Payment Advice when the outcome of the create can be unknown.

The create is a single OData POST (deep insert). A timeout or a gateway error
does not tell whether S/4 created the advice, so posting keeps a *claim* on
`advice["s4"]` from the moment it starts until the advice is posted or the
analyst releases it:

- `posting_started_at`: ISO UTC time the create was started,
- `claimed_payload`: the payload that was sent,
- `post_outcome`: `"unknown"` once a create ended without a definite answer.

The steps that `service.post` and `service.release` take with this module:

1. `claim_of` returns the claim keys present on an advice, so a new check
   result can be stored as `{**fresh, **claim}` without losing the claim.
2. `reconcile` decides what a post does before anything is created:
   - no claim: an S/4 advice with the same company code, account type, account
     and header text refuses the post as a duplicate; none means "create";
   - a claim (an earlier post was interrupted): several S/4 advices refuse
     the post; exactly one is read back and adopted only when it equals the
     claimed payload; none means "create" once the claim is older than the
     S/4 request timeout (a younger claim may still be running in S/4).
3. `failed_create` classifies a create failure (`is_definite_rejection`): a
   definite 4xx rejection clears the claim, anything else keeps it with
   `post_outcome: "unknown"` so the next post reconciles it. The POST itself
   is never retried automatically.
4. `ensure_releasable` guards `service.release`: an advice whose claim may
   still exist in S/4 is not handed back to the analyst for editing.

This module never imports `service`; `service` imports this module.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from .client import S4Client, S4HTTPError, odata_str, timeout_seconds
from .odata import read_all
from .payment_advice import KEY_FIELDS, SERVICE, compare_read_back, read_back

# Keys on `advice["s4"]` that belong to an unfinished post.
CLAIM_KEYS = ("posting_started_at", "claimed_payload", "post_outcome")
# Fields that identify "the same payment" in S/4: the advice account plus the payment reference.
SEARCH_FIELDS = ("CompanyCode", "PaymentAdviceAccountType", "PaymentAdviceAccount", "PaymentAdviceHeaderText")
# Differences quoted in a refusal message; the rest is left out to keep the message readable.
SHOWN_DIFFERENCES = 3


class S4PostError(RuntimeError):
    """S/4 rejected the payment advice, or did not confirm its creation.

    Args:
        detail: Message for the user (the S/4 message, or the unknown-outcome explanation).
        unknown: True when S/4 gave no definite answer and the advice may exist in S/4.
    """

    def __init__(self, detail: str, unknown: bool = False) -> None:
        super().__init__(detail)
        self.detail, self.unknown = detail, unknown


def claim_of(advice: dict[str, Any]) -> dict[str, Any]:
    """Return the claim keys (`CLAIM_KEYS`) present on ``advice["s4"]``.

    Args:
        advice: Advice record; ``s4`` may be missing or empty.

    Returns:
        A dict with only the claim keys that are present; empty when no post is unfinished.
    """
    s4 = advice.get("s4") or {}
    return {key: s4[key] for key in CLAIM_KEYS if key in s4}


def is_definite_rejection(exc: S4HTTPError) -> bool:
    """Tell whether a failed create was definitely refused by S/4.

    Args:
        exc: The error raised by the create call.

    Returns:
        True for a 4xx answer other than 408 (request timeout) and 429 (too many requests);
        False for no answer (status 0), 408, 429 and 5xx, where S/4 may have created the advice.
    """
    return 400 <= exc.status_code < 500 and exc.status_code not in (408, 429)


def failed_create(fresh: dict[str, Any], claim: dict[str, Any], exc: S4HTTPError,
                  now: str) -> tuple[dict[str, Any], S4PostError]:
    """Build the `advice["s4"]` state after a failed create and the error to raise.

    Args:
        fresh: The fresh check result the post was based on.
        claim: The claim written just before the create (`posting_started_at`, `claimed_payload`).
        exc: The error raised by the create call.
        now: ISO UTC time of the failure.

    Returns:
        ``(s4_state, error)``. A definite rejection drops the claim; an unknown outcome keeps it,
        adds `post_outcome: "unknown"` and returns an error with ``unknown=True``.
    """
    failure = {"post_error": exc.detail, "post_failed_at": now}
    if is_definite_rejection(exc):
        return {**fresh, **failure}, S4PostError(exc.detail)
    state = {**fresh, "posting_started_at": claim.get("posting_started_at"),
             "claimed_payload": claim.get("claimed_payload"), "post_outcome": "unknown", **failure}
    message = (f"S/4 did not confirm the creation of the payment advice ({exc.detail}); the outcome is unknown. "
               "Post again to reconcile it: an advice S/4 created meanwhile is adopted, otherwise it is created.")
    return state, S4PostError(message, unknown=True)


def existing_in_s4(client: S4Client, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Read every S/4 advice for the same account whose header text equals this payment reference.

    Args:
        client: S/4 client.
        payload: Payment advice payload; its `SEARCH_FIELDS` form the filter.

    Returns:
        Key fields (`KEY_FIELDS`) of each matching S/4 advice; empty when the header text is empty.

    Raises:
        S4HTTPError: The search failed or was incomplete (`odata.S4IncompleteRead`).
    """
    if not payload.get("PaymentAdviceHeaderText"):
        return []
    filt = " and ".join(f"{field} eq {odata_str(payload[field])}" for field in SEARCH_FIELDS)
    order = ",".join(KEY_FIELDS)
    return read_all(client, SERVICE, "/A_PaymentAdvice", {"$filter": filt, "$select": order}, order)


def claim_wait_seconds(claim: dict[str, Any], now: str) -> int:
    """Return how many seconds an earlier create may still be running in S/4 (0 when none).

    Args:
        claim: Claim keys of the advice (`posting_started_at` is read).
        now: Current ISO UTC time.

    Returns:
        Seconds until the claim is older than the S/4 request timeout, rounded up. An unreadable
        start time counts as old, because it cannot belong to a request still in flight.
    """
    try:
        started = datetime.fromisoformat(str(claim.get("posting_started_at")))
        current = datetime.fromisoformat(now)
    except ValueError:
        return 0
    # Older records or hand-written values may be naive; claims are always written in UTC.
    started = started if started.tzinfo else started.replace(tzinfo=timezone.utc)
    current = current if current.tzinfo else current.replace(tzinfo=timezone.utc)
    remaining = timeout_seconds() - (current - started).total_seconds()
    return max(0, math.ceil(remaining))


def _numbers(rows: list[dict[str, Any]]) -> str:
    """Join the S/4 advice numbers of ``rows`` for a message."""
    return ", ".join(str(row.get("PaymentAdvice")) for row in rows)


def reconcile(client: S4Client, claim: dict[str, Any], fresh_payload: dict[str, Any],
              now: str) -> dict[str, Any] | None:
    """Decide whether a post creates the advice, adopts one S/4 already holds, or is refused.

    Args:
        client: S/4 client.
        claim: Claim keys of the advice (`claim_of`); empty when no earlier post was interrupted.
        fresh_payload: Payload built by the fresh check.
        now: Current ISO UTC time.

    Returns:
        None when the advice must be created; otherwise the adoption
        ``{"key", "payload", "read_back", "verification"}`` where ``payload`` is the claimed payload.

    Raises:
        ValueError: A duplicate without a claim, several S/4 advices, one that differs from the
            claimed payload, or a claim that may still be running in S/4.
        S4HTTPError: The search or the read-back failed.
    """
    if not claim.get("posting_started_at"):
        existing = existing_in_s4(client, fresh_payload)
        if existing:
            raise ValueError(f"S/4 already holds payment advice {existing[0]['PaymentAdvice']} for this payment reference")
        return None
    # Old records may hold a claim without the payload that was sent; the fresh one is then the best evidence.
    payload = claim.get("claimed_payload") or fresh_payload
    existing = existing_in_s4(client, payload)
    if len(existing) > 1:
        raise ValueError(f"S/4 holds several payment advices for this payment reference ({_numbers(existing)}); "
                         "none was adopted. Clear them up in S/4 before posting again.")
    if existing:
        key = {field: existing[0][field] for field in KEY_FIELDS}
        stored = read_back(client, key)
        verification = compare_read_back(payload, stored)
        if verification["differences"]:
            shown = "; ".join(verification["differences"][:SHOWN_DIFFERENCES])
            raise ValueError(f"S/4 holds payment advice {key['PaymentAdvice']} with this payment reference, "
                             f"but it differs from this advice: {shown}")
        return {"key": key, "payload": payload, "read_back": stored, "verification": verification}
    wait = claim_wait_seconds(claim, now)
    if wait:
        raise ValueError(f"An earlier post of this advice may still be running in S/4; try again in {wait} seconds")
    return None


def ensure_releasable(client: S4Client, claim: dict[str, Any], payload: dict[str, Any], now: str) -> None:
    """Refuse to release a claim whose advice exists, or may still appear, in S/4.

    Args:
        client: S/4 client.
        claim: Claim keys of the advice (`claim_of`).
        payload: Payload to search with (the claimed payload, else the last check's payload).
        now: Current ISO UTC time.

    Raises:
        ValueError: No claim, S/4 holds an advice for the claimed reference, or the claim is
            younger than the S/4 request timeout.
        S4HTTPError: The search failed.
    """
    if not claim.get("posting_started_at"):
        raise ValueError("This advice has no unfinished S/4 post to release")
    existing = existing_in_s4(client, payload)
    if existing:
        raise ValueError(f"S/4 holds payment advice {_numbers(existing)} for this payment reference; "
                         "post the advice again so it is adopted")
    wait = claim_wait_seconds(claim, now)
    if wait:
        raise ValueError(f"An earlier post of this advice may still be running in S/4; try again in {wait} seconds")
