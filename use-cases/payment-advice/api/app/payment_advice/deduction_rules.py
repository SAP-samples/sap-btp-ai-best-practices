"""
Access to the per-client deduction playbook table (UC-02 skill catalog).

This module provides CRUD operations for PAYMENT_ADVICE_EXTRACTOR_DEDUCTION_RULES,
the table that stores a natural-language playbook and optional anchor-code mappings
for each client.  All mutating operations are Human-in-the-Loop (HITL) gated:

  - save_playbook(user_confirmed=False) previews the change and returns
    ``{"staged": True, ...}`` without writing anything.  The caller must present
    the preview to the user and re-invoke with ``user_confirmed=True`` to commit.

  - delete_playbook(user_confirmed=False) likewise previews without writing.

This makes the access layer safe to expose as LangGraph agent tools (Task 7) where
the agent must obtain human approval before mutating state.

Tables
------
    PAYMENT_ADVICE_EXTRACTOR_DEDUCTION_RULES   (defined in hana_schema.py)

Revision strategy
-----------------
Each UPSERT increments REVISION by 1.  The current revision is fetched at the start
of save_playbook so the preview shows the caller what the committed revision will be.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from .hana_schema import DEDUCTION_RULES


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Playbook:
    """
    One row of PAYMENT_ADVICE_EXTRACTOR_DEDUCTION_RULES.

    Attributes:
        client_key:    Stable client identifier (normalized, max 60 chars).
        playbook_text: Free-form natural language rules for the LLM agent.
        anchors:       Optional dict mapping deduction codes to descriptions.
        revision:      Monotonically increasing version counter.
        updated_by:    Identity of the last user who saved the playbook, or None.
    """

    client_key: str
    playbook_text: str
    anchors: dict[str, Any]
    revision: int
    updated_by: str | None


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _current_revision(engine: Engine, client_key: str) -> int | None:
    """
    Return the current REVISION for ``client_key``, or None if no row exists.

    Uses a read-only connection (engine.connect) so it does not start a write
    transaction.

    Args:
        engine:     SQLAlchemy Engine connected to SAP HANA.
        client_key: The client identifier to look up.

    Returns:
        The integer revision stored in the row, or None if the row is absent.
    """
    with engine.connect() as conn:
        return conn.execute(
            text(f'SELECT "REVISION" FROM "{DEDUCTION_RULES}" WHERE "CLIENT_KEY" = :ck'),
            {"ck": client_key},
        ).scalar()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_playbook(engine: Engine, client_key: str) -> Playbook | None:
    """
    Return the playbook row for a client, or None if none exists yet.

    Args:
        engine:     SQLAlchemy Engine connected to SAP HANA.
        client_key: The client identifier to retrieve.

    Returns:
        A Playbook dataclass populated from the stored row, or None.
    """
    with engine.connect() as conn:
        row = conn.execute(
            text(
                f'''SELECT "CLIENT_KEY","PLAYBOOK_TEXT","ANCHORS_JSON","REVISION","UPDATED_BY"
                     FROM "{DEDUCTION_RULES}" WHERE "CLIENT_KEY" = :ck'''
            ),
            {"ck": client_key},
        ).mappings().first()
    if row is None:
        return None
    # Normalize keys to uppercase in case the hdbcli driver returns lowercase.
    row = {str(k).upper(): v for k, v in row.items()}
    return Playbook(
        client_key=row["CLIENT_KEY"],
        playbook_text=row["PLAYBOOK_TEXT"] or "",
        anchors=json.loads(row["ANCHORS_JSON"]) if row["ANCHORS_JSON"] else {},
        revision=int(row["REVISION"]),
        updated_by=row["UPDATED_BY"],
    )


def save_playbook(
    engine: Engine,
    client_key: str,
    playbook_text: str,
    anchors: dict[str, Any],
    user_confirmed: bool,
    updated_by: str | None = None,
) -> dict[str, Any]:
    """
    Two-turn HITL gate for creating or updating a client's playbook.

    First call (user_confirmed=False):
      Returns a preview dict without writing anything to the database.

    Second call (user_confirmed=True):
      Persists the playbook with REVISION incremented by 1.  Inserts a new row
      when no row exists yet; issues an UPDATE for an existing row.

    Args:
        engine:        SQLAlchemy Engine connected to SAP HANA.
        client_key:    The client identifier to create/update.
        playbook_text: Natural language rules for the deduction agent.
        anchors:       Optional dict mapping deduction codes to descriptions.
        user_confirmed: False → stage preview only; True → persist.
        updated_by:    Optional identity of the requester (stored in UPDATED_BY).

    Returns:
        When user_confirmed=False:
            {"staged": True, "next_revision": int, "preview": {client_key, playbook_text, anchors}}
        When user_confirmed=True:
            {"staged": False, "revision": int}
    """
    # Read current revision to compute next_revision and choose INSERT vs UPDATE.
    current = _current_revision(engine, client_key)
    next_revision = 1 if current is None else int(current) + 1

    if not user_confirmed:
        # Stage: return preview without touching the database.
        return {
            "staged": True,
            "next_revision": next_revision,
            "preview": {
                "client_key": client_key,
                "playbook_text": playbook_text,
                "anchors": anchors,
            },
        }

    # Persist: serialize anchors to JSON then INSERT or UPDATE.
    anchors_json = json.dumps(anchors, ensure_ascii=False)
    with engine.begin() as conn:
        if current is None:
            # No existing row — insert.
            conn.execute(
                text(
                    f'''INSERT INTO "{DEDUCTION_RULES}"
                         ("CLIENT_KEY","PLAYBOOK_TEXT","ANCHORS_JSON","REVISION","UPDATED_BY")
                         VALUES (:ck,:pt,:aj,:rev,:by)'''
                ),
                {"ck": client_key, "pt": playbook_text, "aj": anchors_json,
                 "rev": next_revision, "by": updated_by},
            )
        else:
            # Row exists — update in place (a corrected playbook is just the next revision).
            conn.execute(
                text(
                    f'''UPDATE "{DEDUCTION_RULES}"
                         SET "PLAYBOOK_TEXT"=:pt, "ANCHORS_JSON"=:aj, "REVISION"=:rev,
                             "UPDATED_BY"=:by, "UPDATED_AT"=CURRENT_UTCTIMESTAMP
                         WHERE "CLIENT_KEY"=:ck'''
                ),
                {"ck": client_key, "pt": playbook_text, "aj": anchors_json,
                 "rev": next_revision, "by": updated_by},
            )
    return {"staged": False, "revision": next_revision}


def delete_playbook(
    engine: Engine,
    client_key: str,
    user_confirmed: bool,
) -> dict[str, Any]:
    """
    HITL-gated destructive operation to remove a client's playbook.

    First call (user_confirmed=False):
      Returns a preview dict showing whether a row would be deleted, without
      writing anything.

    Second call (user_confirmed=True):
      Deletes the row if it exists and returns whether the deletion took effect.

    Args:
        engine:         SQLAlchemy Engine connected to SAP HANA.
        client_key:     The client identifier whose playbook to delete.
        user_confirmed: False → preview only; True → delete.

    Returns:
        When user_confirmed=False:
            {"staged": True, "would_delete": bool}
        When user_confirmed=True:
            {"staged": False, "deleted": bool}  (deleted=False if no row existed)
    """
    # Check existence without starting a write transaction.
    exists = _current_revision(engine, client_key) is not None

    if not user_confirmed:
        # Stage: report whether we would delete, without touching the database.
        return {"staged": True, "would_delete": exists}

    # Persist: issue the DELETE and inspect rowcount to determine success.
    with engine.begin() as conn:
        result = conn.execute(
            text(f'DELETE FROM "{DEDUCTION_RULES}" WHERE "CLIENT_KEY" = :ck'),
            {"ck": client_key},
        )
    return {"staged": False, "deleted": bool(getattr(result, "rowcount", 0))}
