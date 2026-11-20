"""Single write path for engineering review decisions (Accept / Reject).

Both the review API route and the chat assistant's ``submit_review_decision``
tool call ``record_review`` so every row in ``ECN_IMPACT_REVIEW`` is checked
against the snapshot the reviewer saw and stored the same way.
"""

import hashlib
from datetime import datetime, timezone

from app.ecn.errors import PersistenceStateError
from app.ecn.evidence.acceptance_plan import build_acceptance_plan
from app.persistence.hana import to_hana_timestamp


def actor_id(api_key: str) -> str:
    """Return the stored actor identity: a SHA-256 of the caller's API key.

    Args:
        api_key: Raw ``X-API-Key`` header value ("" when absent).

    Returns:
        64-character hex digest; the key itself is never stored.
    """
    return hashlib.sha256(api_key.encode()).hexdigest()


def record_review(
    snapshots,
    workspace,
    ecn_row_id: str,
    *,
    request_id: str,
    reviewer: str,
    outcome: str,
    comments: str,
    actor: str,
    evidence_hash: str,
    snapshot_version: str,
) -> list[dict]:
    """Verify the reviewed snapshot is current, append one review row, return history.

    Blocking; call through ``run_blocking`` from async code.

    Args:
        snapshots: ``SnapshotService``; its overview identifies current evidence.
        workspace: ``WorkspaceRepository`` that appends to ``ECN_IMPACT_REVIEW``.
        ecn_row_id: Assessment being reviewed.
        request_id: Idempotency key; replaying identical content saves once.
        reviewer: Reviewer name (stripped; must not be blank).
        outcome: ``ACCEPTED`` or ``CHANGES_REQUESTED`` (validated by callers).
        comments: Review comment text (stripped).
        actor: Hashed caller identity from ``actor_id``.
        evidence_hash / snapshot_version: Identity of the snapshot the reviewer saw.

    Returns:
        All reviews for the assessment, oldest first.

    Raises:
        PersistenceStateError: ``REVIEW_SNAPSHOT_MISMATCH`` (409) when the
            snapshot changed, ``REVIEW_TEXT_REQUIRED`` (400) for a blank reviewer,
            ``ECN_ACCEPT_BLOCKED`` (409) when accepting while a new part has no
            price, ``REVIEW_REQUEST_CONFLICT`` from the repository on a reused ID.
    """
    snapshot = snapshots.overview(ecn_row_id)
    if snapshot["evidence_hash"] != evidence_hash or snapshot["version"] != snapshot_version:
        raise PersistenceStateError(
            "REVIEW_SNAPSHOT_MISMATCH",
            "Reload this snapshot before submitting the review.",
            status_code=409,
        )
    if not reviewer.strip():
        raise PersistenceStateError("REVIEW_TEXT_REQUIRED", "Enter the reviewer name.", status_code=400)
    # The card hides Accept for a blocked ECN; this stops the API and the assistant too.
    plan = build_acceptance_plan(snapshot) if outcome == "ACCEPTED" else None
    if plan and plan["blocked"]:
        raise PersistenceStateError("ECN_ACCEPT_BLOCKED", plan["block_reason"], status_code=409)
    workspace.save_review({
        "REVIEW_ID": request_id,
        "ECN_ROW_ID": ecn_row_id,
        "EVIDENCE_HASH": evidence_hash,
        "SNAPSHOT_VERSION": snapshot["version"],
        "REVIEWER": reviewer.strip(),
        "ACTOR_ID": actor,
        "OUTCOME": outcome,
        "COMMENTS": comments.strip(),
        "CREATED_AT": to_hana_timestamp(datetime.now(timezone.utc)),
    })
    return workspace.reviews(ecn_row_id)
