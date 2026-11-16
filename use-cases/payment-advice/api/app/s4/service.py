"""Check an advice against S/4 (draft), post it as an S/4 Payment Advice (final), or release an unfinished post.

Three explicit steps, mirroring Cash Application's draft/final split:

1. `check`: read-only. Collects the S/4 candidates of every line
   (`matching.collect_candidates`: references field by field, other spellings
   of unknown references, partial payments and residual items), resolves and
   validates them (`validation.check_advice`, with the customer master and the
   reason codes of `master_data`), builds the payload preview and stores it on
   the advice as `advice["s4"]`:
   `{derived, lines, issues, ready, summary, payload, trace, result_hash, checked_at, mode}`.
   `result_hash` fingerprints the advice result, so any later correction makes
   the stored check visibly stale. The claim of an unfinished post
   (`posting.CLAIM_KEYS`) is kept on the new result.
2. `post`: only for `reviewed` advices.
   a. Re-runs the check against fresh S/4 data; blocking issues are stored
      (claim kept) and nothing is posted.
   b. `posting.reconcile` searches S/4 for advices with the same account and
      header text (payment reference). Without a claim any match is refused as
      a duplicate. With a claim (an earlier post ended without an answer) a
      single match equal to the claimed payload is adopted, a different one or
      several are refused, and none means create once the claim is older than
      the S/4 request timeout.
   c. To create: writes the claim (`posting_started_at`, `claimed_payload`)
      with a revision-checked update, so a concurrent second click fails
      instead of creating a second advice, then POSTs once. A definite 4xx
      rejection clears the claim; any other failure keeps it with
      `post_outcome: "unknown"` (`posting.failed_create`). Never retried.
   d. Reads the advice back, sets status `posted` and inserts an immutable
      `s4_posting` workspace record (payload, key, read-back, `adopted`).
3. `release`: hands an advice with an unfinished post back to the analyst once
   S/4 holds no advice for the claimed reference and the claim is older than
   the S/4 request timeout; the claim keys are removed.

Posted advices are immutable, and advices with a claim cannot be edited
(see `corrections.advice_record(mutable=True)`).
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
from . import posting
from .client import S4Client, S4Config, S4HTTPError, load_s4_config
from .master_data import customer_status, reason_codes
from .matching import collect_candidates
from .payment_advice import build_payload, compare_read_back, create, read_back
# Re-exported: the router, the CLI and the tests use `service.S4PostError`.
from .posting import S4PostError  # noqa: F401
from .validation import check_advice


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
    aliases = company_code_aliases()
    found = collect_candidates(client, result, aliases, trace)
    check = check_advice(result, found, advice.get("verification"),
                         customer_lookup=lambda accounts, code: customer_status(client, accounts, code, trace),
                         aliases=aliases, reason_codes=lambda code: reason_codes(client, code, trace),
                         reason_aliases=reason_code_aliases())
    payload = (build_payload(result, check, os.getenv("S4_PAYMENT_ADVICE_TYPE", "04") or "04", reason_code_aliases())
               if check["ready"] else None)
    return {**check, "payload": payload, "trace": trace, "result_hash": result_hash(result),
            "checked_at": _now(), "mode": client.config.mode}


def check(store, client: S4Client, advice_id: str, revision: int | None = None) -> dict[str, Any]:
    """Run the read-only S/4 check and store it on the advice.

    Args:
        store: Workspace repository.
        client: S/4 client.
        advice_id: Advice to check.
        revision: Expected advice revision (None skips the revision check).

    Returns:
        The saved advice; the claim of an unfinished post is kept on the new check result.

    Raises:
        ValueError: The advice is posted, or not processed yet.
        Conflict: The advice changed since ``revision``.
        S4HTTPError: An S/4 lookup failed.
    """
    advice = advice_record(store, advice_id, revision, mutable=True, claimed_ok=True)
    advice["s4"] = {**run_check(client, advice), **posting.claim_of(advice)}
    return store.update(advice)


def _create(store, client: S4Client, advice: dict[str, Any], fresh: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Claim the post with a revision-checked write, then create the advice in S/4 (one POST).

    Args:
        store: Workspace repository.
        client: S/4 client.
        advice: The advice being posted (revision as read).
        fresh: The fresh, ready check result; its payload is sent.

    Returns:
        ``(advice, key)``: the advice as saved with the claim, and the key S/4 assigned.

    Raises:
        Conflict: A concurrent post or edit won the revision check (nothing was sent).
        S4PostError: The create failed; the claim is kept unless S/4 definitely rejected it.
    """
    advice["s4"] = {**fresh, "posting_started_at": _now(), "claimed_payload": fresh["payload"]}
    advice = store.update(advice)  # claim: a concurrent post now fails its revision check
    try:
        return advice, create(client, fresh["payload"])
    except S4HTTPError as exc:
        advice["s4"], error = posting.failed_create(fresh, posting.claim_of(advice), exc, _now())
        store.update(advice)
        raise error from exc


def post(store, client: S4Client, advice_id: str, revision: int) -> dict[str, Any]:
    """Create (or adopt after an interrupted post) the S/4 payment advice of a reviewed advice.

    Args:
        store: Workspace repository.
        client: S/4 client.
        advice_id: Advice to post.
        revision: Expected advice revision.

    Returns:
        The saved advice with status `posted` and `s4.posted` = `{key, posted_at, adopted, read_back, verification}`.

    Raises:
        ValueError: Not reviewed, or blocking issues in the fresh check (saved on the advice, claim kept).
        ValueError: S/4 already holds a different or several advices with this payment reference,
            or an earlier post may still be running in S/4 (see `posting.reconcile`).
        Conflict: Concurrent change (another post or edit won the revision check).
        S4PostError: S/4 rejected the creation, or did not confirm it (`unknown`); saved on the advice.
        S4HTTPError: A lookup, search or adoption read-back failed.
    """
    advice = advice_record(store, advice_id, revision, mutable=True, claimed_ok=True)
    if advice["status"] != "reviewed":
        raise ValueError("Mark the advice reviewed before posting it to S/4")
    claim = posting.claim_of(advice)
    fresh = run_check(client, advice)
    # An interrupted post is reconciled BEFORE the fresh check may block. If S/4 created the advice and has
    # since used it (the invoice is cleared now), the fresh check is no longer ready, yet the advice S/4 holds
    # is exactly the one that was sent and must be adopted, not left in a dead end.
    interrupted = bool(claim.get("posting_started_at") and claim.get("claimed_payload"))
    adopted = posting.reconcile(client, claim, fresh["payload"], _now()) if fresh["ready"] or interrupted else None
    if not adopted and not fresh["ready"]:
        advice["s4"] = {**fresh, **claim}
        store.update(advice)
        raise ValueError("S/4 check found blocking issues; nothing was posted")
    if adopted:
        payload, key = adopted["payload"], adopted["key"]
        stored, verification = adopted["read_back"], adopted["verification"]
    else:
        payload = fresh["payload"]
        advice, key = _create(store, client, advice, fresh)
        try:
            stored = read_back(client, key)
            verification = compare_read_back(payload, stored)
        except S4HTTPError as exc:
            # The advice exists in S/4; a failed read-back is recorded, not treated as a failed post.
            stored, verification = {"read_back_error": exc.detail}, None
    record = {"key": key, "posted_at": _now(), "adopted": bool(adopted), "read_back": stored, "verification": verification}
    advice["s4"] = {**fresh, "posted": record}
    advice["status"] = "posted"
    with store.engine.begin() as conn:
        saved = store.update(advice, conn=conn)
        store.insert("s4_posting", {**record, "advice_id": advice_id, "payload": payload}, parent=advice_id,
                     status="posted", conn=conn)
    refresh_email(store, advice["parent_id"])
    return saved


def release(store, client: S4Client, advice_id: str, revision: int) -> dict[str, Any]:
    """Give an advice with an unfinished post back to the analyst for editing.

    Args:
        store: Workspace repository.
        client: S/4 client.
        advice_id: Advice to release.
        revision: Expected advice revision.

    Returns:
        The saved advice without the claim keys; every other `s4` key is kept.

    Raises:
        ValueError: The advice is posted, has no claim, S/4 holds an advice for the claimed
            reference, or the claim is younger than the S/4 request timeout.
        Conflict: The advice changed since ``revision``.
        S4HTTPError: The S/4 search failed.
    """
    advice = advice_record(store, advice_id, revision, mutable=True, claimed_ok=True)
    claim = posting.claim_of(advice)
    s4 = advice.get("s4") or {}
    # Old claims carry no payload; the last check's payload holds the same reference.
    posting.ensure_releasable(client, claim, claim.get("claimed_payload") or s4.get("payload") or {}, _now())
    advice["s4"] = {key: value for key, value in s4.items() if key not in posting.CLAIM_KEYS}
    return store.update(advice)
