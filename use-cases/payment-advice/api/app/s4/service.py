"""Check an advice against S/4 (draft) and post it as an S/4 Payment Advice (final).

Two explicit steps, mirroring Cash Application's draft/final split:

1. `check`: read-only. Looks up every line reference in S/4, validates, builds
   the payload preview and stores it on the advice as `advice["s4"]`:
   `{derived, lines, issues, ready, payload, trace, result_hash, checked_at, mode}`.
   `result_hash` fingerprints the advice result, so any later correction makes
   the stored check visibly stale.
2. `post`: only for `reviewed` advices. Re-runs the check against fresh S/4
   data, then:
   - claims the post with a revision-checked write (a concurrent second click
     fails instead of creating a second advice in S/4),
   - searches S/4 for an advice with the same account and header text
     (payment reference); one left by an interrupted post of this advice is
     adopted, anything else is refused as a duplicate,
   - creates the advice, reads it back, sets status `posted` and inserts an
     immutable `s4_posting` workspace record (payload, key, read-back).

Posted advices are immutable (see `corrections.advice_record(mutable=True)`).
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from ..email_ingestion.corrections import advice_record
from ..email_ingestion.intake import refresh_email
from .client import S4Client, S4Config, S4HTTPError, load_s4_config, odata_rows, odata_str
from .lookup import existing_customers, find_receivables
from .payment_advice import SERVICE, build_payload, create, read_back
from .validation import check_advice


class S4PostError(RuntimeError):
    """S/4 rejected the payment advice; `detail` is the S/4 message for the user."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


@lru_cache(maxsize=1)
def s4_config() -> S4Config:
    """Load S/4 settings once per process (BTP tokens are cached inside the config)."""
    return load_s4_config()


def s4_client() -> S4Client:
    """Return a client with its own HTTP session (CSRF cookies never shared across requests)."""
    return S4Client(s4_config())


def _aliases(name: str, value: str | None) -> dict[str, str]:
    """Parse a "LEFT=RIGHT,LEFT=RIGHT" setting (env ``name`` unless ``value`` is given) into {LEFT: RIGHT}."""
    raw = os.getenv(name, "") if value is None else value
    pairs = (item.split("=", 1) for item in raw.split(",") if "=" in item)
    return {left.strip().upper(): right.strip().upper() for left, right in pairs if left.strip() and right.strip()}


def company_code_aliases(value: str | None = None) -> dict[str, str]:
    """Parse ``S4_COMPANY_CODE_ALIASES`` ("CA01=Z291,US01=1710") into {rule code: S/4 code}.

    Only for demo/test systems whose company codes differ from the customer's
    documented ones; empty in production.
    """
    return _aliases("S4_COMPANY_CODE_ALIASES", value)


def reason_code_aliases(value: str | None = None) -> dict[str, str]:
    """Parse ``S4_REASON_CODE_ALIASES`` ("316=060,319=060") into {documented reason code: S/4 reason code}.

    Only for demo/test systems that lack the customer's reason codes (for example
    a demo company code with only SAP's standard 010-130); the advice keeps the documented code, only
    the S/4 payload uses the alias. Empty in production.
    """
    return _aliases("S4_REASON_CODE_ALIASES", value)


def result_hash(result: dict[str, Any]) -> str:
    """Fingerprint the advice result the check was computed from."""
    return hashlib.sha256(json.dumps(result, sort_keys=True, default=str).encode()).hexdigest()


def _now() -> str:
    """UTC timestamp for audit fields."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_check(client: S4Client, advice: dict[str, Any]) -> dict[str, Any]:
    """Look up, validate and build the payload preview for one advice (no persistence)."""
    result = advice.get("result")
    if not result or advice["status"] in {"queued", "processing", "failed"}:
        raise ValueError("The advice must finish processing before it can be checked against S/4")
    trace: list[dict[str, Any]] = []
    found = find_receivables(client, [row.get("invoice_reference") for row in result.get("line_items") or []], trace=trace)
    check = check_advice(result, found, advice.get("verification"),
                         customer_lookup=lambda accounts, code: existing_customers(client, accounts, code, trace),
                         aliases=company_code_aliases())
    payload = (build_payload(result, check, os.getenv("S4_PAYMENT_ADVICE_TYPE", "04") or "04", reason_code_aliases())
               if check["ready"] else None)
    return {**check, "payload": payload, "trace": trace, "result_hash": result_hash(result),
            "checked_at": _now(), "mode": client.config.mode}


def check(store, client: S4Client, advice_id: str, revision: int | None = None) -> dict[str, Any]:
    """Run the read-only S/4 check and store it on the advice; returns the saved advice."""
    advice = advice_record(store, advice_id, revision, mutable=True)
    advice["s4"] = run_check(client, advice)
    return store.update(advice)


def _existing_in_s4(client: S4Client, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Find S/4 advices for the same account whose header text equals this payment reference."""
    if not payload.get("PaymentAdviceHeaderText"):
        return []
    filt = " and ".join(f"{field} eq {odata_str(payload[field])}" for field in
                        ("CompanyCode", "PaymentAdviceAccountType", "PaymentAdviceAccount", "PaymentAdviceHeaderText"))
    return odata_rows(client.get_json(SERVICE, "/A_PaymentAdvice", {"$filter": filt, "$top": "5",
        "$select": "CompanyCode,PaymentAdviceAccountType,PaymentAdviceAccount,PaymentAdvice,CreationDate"}))


def post(store, client: S4Client, advice_id: str, revision: int) -> dict[str, Any]:
    """Create the S/4 payment advice for a reviewed advice; returns the saved (posted) advice.

    Raises:
        ValueError: Not reviewed, or blocking issues in the fresh check (saved on the advice).
        ValueError: Also when S/4 already holds an advice with this payment reference.
        Conflict: Concurrent change (another post or edit won the revision check).
        S4PostError: S/4 rejected the creation (the error is saved on the advice).
    """
    advice = advice_record(store, advice_id, revision, mutable=True)
    if advice["status"] != "reviewed":
        raise ValueError("Mark the advice reviewed before posting it to S/4")
    interrupted = bool((advice.get("s4") or {}).get("posting_started_at"))
    fresh = run_check(client, advice)
    if not fresh["ready"]:
        advice["s4"] = fresh
        store.update(advice)
        raise ValueError("S/4 check found blocking issues; nothing was posted")
    payload = fresh["payload"]
    existing = _existing_in_s4(client, payload)
    if existing and not interrupted:
        raise ValueError(f"S/4 already holds payment advice {existing[0]['PaymentAdvice']} for this payment reference")
    advice["s4"] = {**fresh, "posting_started_at": _now()}
    advice = store.update(advice)  # claim: a concurrent post now fails its revision check
    if existing:
        key = {field: existing[0][field] for field in ("CompanyCode", "PaymentAdviceAccountType", "PaymentAdviceAccount", "PaymentAdvice")}
    else:
        try:
            key = create(client, payload)
        except S4HTTPError as exc:
            advice["s4"] = {**fresh, "post_error": exc.detail, "post_failed_at": _now()}
            store.update(advice)
            raise S4PostError(exc.detail) from exc
    try:
        stored = read_back(client, key)
    except S4HTTPError as exc:
        stored = {"read_back_error": exc.detail}
    posting = {"key": key, "posted_at": _now(), "adopted": bool(existing), "read_back": stored}
    advice["s4"] = {**fresh, "posted": posting}
    advice["status"] = "posted"
    with store.engine.begin() as conn:
        saved = store.update(advice, conn=conn)
        store.insert("s4_posting", {**posting, "advice_id": advice_id, "payload": payload}, parent=advice_id,
                     status="posted", conn=conn)
    refresh_email(store, advice["parent_id"])
    return saved
