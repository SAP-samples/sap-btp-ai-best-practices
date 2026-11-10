"""API-key protected S/4HANA endpoints for one payment advice: check (draft) and post (final).

Mounted under `/api/payment-advice/advices`:

- `POST /{advice_id}/s4/check` `{revision}`: read-only against S/4; stores the
  lookup results, validation issues and payload preview on the advice.
- `POST /{advice_id}/s4/post` `{revision}`: creates the S/4 Payment Advice for a
  reviewed advice after a fresh check; the advice becomes `posted` (immutable).
"""
from fastapi import APIRouter, Depends, HTTPException, Request

from ..s4 import service
from ..s4.client import S4ConfigError, S4HTTPError
from ..security import get_api_key
from .email_ingestion import Revision, checked, storage

router = APIRouter(dependencies=[Depends(get_api_key)])


def s4_call(function, request: Request, advice_id: str, revision: int):
    """Run a service call and map S/4 failures to safe HTTP errors (domain errors via `checked`)."""
    try:
        return checked(function, storage(request), service.s4_client(), advice_id, revision)
    except S4ConfigError as exc:
        raise HTTPException(503, f"S/4 connection is not configured: {exc}") from None
    except service.S4PostError as exc:
        raise HTTPException(502, f"S/4 rejected the payment advice: {exc.detail}") from None
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
