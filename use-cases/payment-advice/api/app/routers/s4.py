"""API-key protected S/4HANA endpoints for one payment advice: check (draft), post (final) and release.

Mounted under `/api/payment-advice/advices`:

- `POST /{advice_id}/s4/check` `{revision}`: read-only against S/4; stores the
  lookup results, validation issues and payload preview on the advice.
- `POST /{advice_id}/s4/post` `{revision}`: creates the S/4 Payment Advice for a
  reviewed advice after a fresh check; the advice becomes `posted` (immutable).
  After an interrupted post it adopts the advice S/4 already created, if equal.
- `POST /{advice_id}/s4/release` `{revision}`: removes the claim of an
  interrupted post that never reached S/4, so the advice can be edited again.
"""
from fastapi import APIRouter, Depends, HTTPException, Request

from ..s4 import service
from ..s4.client import S4ConfigError, S4HTTPError
from ..security import get_api_key
from .email_ingestion import Revision, checked, storage

router = APIRouter(dependencies=[Depends(get_api_key)])


def s4_call(function, request: Request, advice_id: str, revision: int):
    """Run a service call and map S/4 failures to safe HTTP errors (domain errors via `checked`).

    Args:
        function: Service entry point taking ``(store, client, advice_id, revision)``.
        request: Incoming request (gives the workspace store).
        advice_id: Advice to act on.
        revision: Expected advice revision.

    Returns:
        The saved advice returned by ``function``.

    Raises:
        HTTPException: 400 domain error, 404 unknown advice, 409 stale revision (via `checked`),
            503 S/4 not configured, 502 S/4 rejection, unknown post outcome or failed S/4 call.
    """
    try:
        return checked(function, storage(request), service.s4_client(), advice_id, revision)
    except S4ConfigError as exc:
        raise HTTPException(503, f"S/4 connection is not configured: {exc}") from None
    except service.S4PostError as exc:
        # An unknown outcome is not a rejection: its detail already explains what happened.
        raise HTTPException(502, exc.detail if exc.unknown else f"S/4 rejected the payment advice: {exc.detail}") from None
    except S4HTTPError as exc:
        raise HTTPException(502, f"S/4 call failed (HTTP {exc.status_code}): {exc.detail[:300]}") from None


@router.post("/{advice_id}/s4/check")
def check(advice_id: str, body: Revision, request: Request):
    """Look up the advice references in S/4, validate, and store the payment advice preview."""
    return s4_call(service.check, request, advice_id, body.revision)


@router.post("/{advice_id}/s4/post")
def post(advice_id: str, body: Revision, request: Request):
    """Create the S/4 Payment Advice for a reviewed advice (re-checks against fresh S/4 data first)."""
    return s4_call(service.post, request, advice_id, body.revision)


@router.post("/{advice_id}/s4/release")
def release(advice_id: str, body: Revision, request: Request):
    """Hand an advice with an interrupted post back for editing once S/4 holds no advice for it."""
    return s4_call(service.release, request, advice_id, body.revision)
