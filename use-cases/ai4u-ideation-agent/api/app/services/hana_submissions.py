"""HANA Cloud persistence layer for ideation submissions and leaderboard.

Stores evaluated solutions in SAP HANA Cloud for leaderboard ranking.
Follows the same hdbcli + asyncio.to_thread pattern as hana_store.py.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _compute_content_hash(industry: str, title: str, problem: str, solution: str) -> str:
    """Generate SHA-256 hash for deduplication."""
    normalized = f"{industry}|{title}|{problem}|{solution}".lower().strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class HanaSubmissionStore:
    """HANA-backed submission store for ideation leaderboard."""

    async def init_schema(self) -> None:
        """Create submissions table and indexes if they don't exist (idempotent)."""
        from ideation_agent.hana_store import _execute

        table_ddl = """CREATE COLUMN TABLE IDEATION_SUBMISSIONS (
            SUBMISSION_ID    NVARCHAR(64)  PRIMARY KEY,
            UID              NVARCHAR(256) NOT NULL,
            EMAIL            NVARCHAR(320) NOT NULL,
            COMPANY_DOMAIN   NVARCHAR(255),
            INDUSTRY         NVARCHAR(255),
            TITLE            NVARCHAR(512),
            PROBLEM          NCLOB,
            SOLUTION         NCLOB,
            APP_TYPE         NVARCHAR(64),
            COMPANY_CONTEXT  NCLOB,
            SOURCES_JSON     NCLOB,
            EVALUATOR_FEEDBACK           NVARCHAR(512),
            EVALUATOR_DETAILED_ANALYSIS  NCLOB,
            EVALUATOR_SUGGESTIONS_JSON   NCLOB,
            APPROPRIATENESS_LEVEL        NVARCHAR(32),
            SCORE            INTEGER NOT NULL,
            RATING           INTEGER DEFAULT 0,
            SUBMITTED_AT     TIMESTAMP NOT NULL,
            SOURCE           NVARCHAR(64) NOT NULL,
            CONTEXT_ID       NVARCHAR(64),
            CONTENT_HASH     NVARCHAR(64),
            MESSAGE_INDEX    INTEGER DEFAULT NULL,
            IS_ON_LEADERBOARD BOOLEAN DEFAULT TRUE
        )"""

        try:
            await _execute(table_ddl)
        except Exception as exc:
            err_str = str(exc).lower()
            if "already exists" not in err_str and "duplicate" not in err_str and "288" not in err_str:
                raise

        # Migration: add MESSAGE_INDEX to existing tables (no-op on fresh installs)
        try:
            await _execute(
                "ALTER TABLE IDEATION_SUBMISSIONS ADD (MESSAGE_INDEX INTEGER DEFAULT NULL)"
            )
        except Exception:
            pass  # Column already exists

        # Migration: add IS_ON_LEADERBOARD flag (one active entry per context on leaderboard)
        try:
            await _execute(
                "ALTER TABLE IDEATION_SUBMISSIONS ADD (IS_ON_LEADERBOARD BOOLEAN DEFAULT TRUE)"
            )
        except Exception:
            pass  # Column already exists

        # Migration: add EVENT_ID for event-scoped submissions
        try:
            await _execute(
                "ALTER TABLE IDEATION_SUBMISSIONS ADD (EVENT_ID NVARCHAR(36) DEFAULT NULL)"
            )
        except Exception:
            pass  # Column already exists

        for sql in [
            "CREATE INDEX IX_SUB_DOMAIN_SCORE ON IDEATION_SUBMISSIONS(COMPANY_DOMAIN, SCORE DESC, SUBMITTED_AT DESC)",
            "CREATE INDEX IX_SUB_UID ON IDEATION_SUBMISSIONS(UID, SUBMITTED_AT DESC)",
            "CREATE INDEX IX_SUB_HASH ON IDEATION_SUBMISSIONS(CONTENT_HASH)",
            "CREATE INDEX IX_SUB_CONTEXT ON IDEATION_SUBMISSIONS(CONTEXT_ID, UID)",
            "CREATE INDEX IX_SUB_EVENT ON IDEATION_SUBMISSIONS(EVENT_ID, SCORE DESC, SUBMITTED_AT DESC)",
        ]:
            try:
                await _execute(sql)
            except Exception:
                pass

    async def upsert_submission(self, submission: Dict[str, Any]) -> str:
        """Insert or update a submission. Returns the submission_id.

        Dedup strategy: one entry per (context_id, uid, message_index).
        Each final message in a conversation gets its own row.
        """
        from ideation_agent.hana_store import _execute, _fetch_one

        content_hash = _compute_content_hash(
            submission.get("industry", ""),
            submission.get("title", ""),
            submission.get("problem", ""),
            submission.get("solution", ""),
        )

        context_id = submission.get("context_id", "")
        message_index = submission.get("message_index")

        # Primary dedup: one entry per (context_id, uid, message_index)
        existing = None
        if context_id and message_index is not None:
            existing = await _fetch_one(
                "SELECT SUBMISSION_ID FROM IDEATION_SUBMISSIONS WHERE CONTEXT_ID = ? AND UID = ? AND MESSAGE_INDEX = ?",
                (context_id, submission["uid"], message_index),
            )
        elif context_id:
            # Legacy fallback (no message_index provided)
            existing = await _fetch_one(
                "SELECT SUBMISSION_ID FROM IDEATION_SUBMISSIONS WHERE CONTEXT_ID = ? AND UID = ?",
                (context_id, submission["uid"]),
            )

        # Fallback: check by content hash (covers cases with no context_id)
        if not existing:
            existing = await _fetch_one(
                "SELECT SUBMISSION_ID FROM IDEATION_SUBMISSIONS WHERE CONTENT_HASH = ? AND UID = ?",
                (content_hash, submission["uid"]),
            )

        submission_id = existing["submission_id"] if existing else content_hash[:64]
        now = datetime.now(timezone.utc)

        if existing:
            # Update existing submission (new evaluation replaces previous for same message)
            await _execute(
                """UPDATE IDEATION_SUBMISSIONS SET
                    COMPANY_DOMAIN = ?,
                    INDUSTRY = ?,
                    TITLE = ?,
                    PROBLEM = ?,
                    SOLUTION = ?,
                    APP_TYPE = ?,
                    COMPANY_CONTEXT = ?,
                    SOURCES_JSON = ?,
                    EVALUATOR_FEEDBACK = ?,
                    EVALUATOR_DETAILED_ANALYSIS = ?,
                    EVALUATOR_SUGGESTIONS_JSON = ?,
                    APPROPRIATENESS_LEVEL = ?,
                    SCORE = ?,
                    RATING = ?,
                    SUBMITTED_AT = ?,
                    SOURCE = ?,
                    CONTEXT_ID = ?,
                    CONTENT_HASH = ?,
                    MESSAGE_INDEX = ?,
                    EVENT_ID = ?
                WHERE SUBMISSION_ID = ?""",
                (
                    submission.get("company_domain", ""),
                    submission.get("industry", ""),
                    submission.get("title", ""),
                    submission.get("problem", ""),
                    submission.get("solution", ""),
                    submission.get("app_type", ""),
                    submission.get("company_context", ""),
                    json.dumps(submission.get("sources", [])),
                    submission.get("evaluator_feedback", ""),
                    submission.get("evaluator_detailed_analysis", ""),
                    json.dumps(submission.get("evaluator_suggestions", [])),
                    submission.get("appropriateness_level", ""),
                    int(submission.get("score", 0)),
                    int(submission.get("rating", 0)),
                    now,
                    submission.get("source", "ideation-agent"),
                    context_id,
                    content_hash,
                    message_index,
                    submission.get("event_id") or None,
                    submission_id,
                ),
            )
        else:
            # Insert new submission
            await _execute(
                """INSERT INTO IDEATION_SUBMISSIONS (
                    SUBMISSION_ID, UID, EMAIL, COMPANY_DOMAIN,
                    INDUSTRY, TITLE, PROBLEM, SOLUTION, APP_TYPE,
                    COMPANY_CONTEXT, SOURCES_JSON,
                    EVALUATOR_FEEDBACK, EVALUATOR_DETAILED_ANALYSIS,
                    EVALUATOR_SUGGESTIONS_JSON, APPROPRIATENESS_LEVEL,
                    SCORE, RATING, SUBMITTED_AT, SOURCE, CONTEXT_ID, CONTENT_HASH,
                    MESSAGE_INDEX, IS_ON_LEADERBOARD, EVENT_ID
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, TRUE, ?)""",
                (
                    submission_id,
                    submission["uid"],
                    submission["email"],
                    submission.get("company_domain", ""),
                    submission.get("industry", ""),
                    submission.get("title", ""),
                    submission.get("problem", ""),
                    submission.get("solution", ""),
                    submission.get("app_type", ""),
                    submission.get("company_context", ""),
                    json.dumps(submission.get("sources", [])),
                    submission.get("evaluator_feedback", ""),
                    submission.get("evaluator_detailed_analysis", ""),
                    json.dumps(submission.get("evaluator_suggestions", [])),
                    submission.get("appropriateness_level", ""),
                    int(submission.get("score", 0)),
                    int(submission.get("rating", 0)),
                    now,
                    submission.get("source", "ideation-agent"),
                    submission.get("context_id", ""),
                    content_hash,
                    message_index,
                    submission.get("event_id") or None,
                ),
            )

        # Ensure only this submission is on the leaderboard for this conversation.
        # Deactivate all other rows for the same (context_id, uid), then activate this one.
        if context_id:
            await _execute(
                """UPDATE IDEATION_SUBMISSIONS
                   SET IS_ON_LEADERBOARD = FALSE
                   WHERE CONTEXT_ID = ? AND UID = ? AND SUBMISSION_ID != ?""",
                (context_id, submission["uid"], submission_id),
            )
            await _execute(
                """UPDATE IDEATION_SUBMISSIONS
                   SET IS_ON_LEADERBOARD = TRUE
                   WHERE SUBMISSION_ID = ?""",
                (submission_id,),
            )

        return submission_id

    async def get_user_history(
        self, uid: str, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get a user's submission history, ordered by most recent first."""
        from ideation_agent.hana_store import _fetch_all

        rows = await _fetch_all(
            f"""SELECT SUBMISSION_ID, INDUSTRY, TITLE, PROBLEM, SOLUTION, APP_TYPE,
                       EVALUATOR_FEEDBACK, EVALUATOR_DETAILED_ANALYSIS,
                       EVALUATOR_SUGGESTIONS_JSON, APPROPRIATENESS_LEVEL,
                       SCORE, RATING, SUBMITTED_AT, CONTEXT_ID
                FROM IDEATION_SUBMISSIONS
                WHERE UID = ?
                ORDER BY SUBMITTED_AT DESC
                LIMIT {int(limit)} OFFSET {int(offset)}""",
            (uid,),
        )
        for row in rows:
            if isinstance(row.get("submitted_at"), datetime):
                row["submitted_at"] = row["submitted_at"].isoformat()
            if row.get("evaluator_suggestions_json"):
                try:
                    row["evaluator_suggestions"] = json.loads(row["evaluator_suggestions_json"])
                except (json.JSONDecodeError, TypeError):
                    row["evaluator_suggestions"] = []
            else:
                row["evaluator_suggestions"] = []
        return rows

    async def count_user_history(self, uid: str) -> int:
        """Count total submissions for a user."""
        from ideation_agent.hana_store import _fetch_one

        row = await _fetch_one(
            "SELECT COUNT(*) AS CNT FROM IDEATION_SUBMISSIONS WHERE UID = ?",
            (uid,),
        )
        return int(row["cnt"]) if row else 0

    async def get_company_leaderboard(
        self, company_domain: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get leaderboard entries for a company domain, ranked by score.

        Deduplicates by CONTEXT_ID: only the highest-scoring submission per
        conversation is kept, attributed to the conversation owner.
        """
        from ideation_agent.hana_store import _fetch_all

        # Fetch more rows than requested to account for duplicates that will collapse
        fetch_limit = limit * 3

        rows = await _fetch_all(
            f"""SELECT SUBMISSION_ID, UID, EMAIL, INDUSTRY, TITLE, PROBLEM,
                       SOLUTION, APP_TYPE, EVALUATOR_FEEDBACK,
                       EVALUATOR_DETAILED_ANALYSIS, EVALUATOR_SUGGESTIONS_JSON,
                       APPROPRIATENESS_LEVEL, SCORE, RATING, SUBMITTED_AT, CONTEXT_ID
                FROM IDEATION_SUBMISSIONS
                WHERE COMPANY_DOMAIN = ? AND IS_ON_LEADERBOARD = TRUE
                ORDER BY SCORE DESC, SUBMITTED_AT DESC
                LIMIT {int(fetch_limit)}""",
            (company_domain,),
        )
        for row in rows:
            if isinstance(row.get("submitted_at"), datetime):
                row["submitted_at"] = row["submitted_at"].isoformat()
            if row.get("evaluator_suggestions_json"):
                try:
                    row["evaluator_suggestions"] = json.loads(row["evaluator_suggestions_json"])
                except (json.JSONDecodeError, TypeError):
                    row["evaluator_suggestions"] = []
            else:
                row["evaluator_suggestions"] = []

        # Deduplicate by CONTEXT_ID — keep highest score per conversation
        best_by_context: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            ctx = row.get("context_id", "")
            if not ctx:
                # Rows without context_id are standalone; keep as-is
                best_by_context[row.get("submission_id", "")] = row
                continue
            existing = best_by_context.get(ctx)
            if existing is None or int(row.get("score", 0)) > int(existing.get("score", 0)):
                best_by_context[ctx] = row

        # Batch-fetch conversation owners to attribute entries to the owner
        context_ids = [
            ctx for ctx in best_by_context.keys()
            if ctx and ctx != best_by_context[ctx].get("submission_id", "")
        ]
        owner_map: Dict[str, str] = {}  # context_id -> owner UID
        if context_ids:
            placeholders = ", ".join(["?"] * len(context_ids))
            owner_rows = await _fetch_all(
                f"SELECT ID, USER_ID FROM CONVERSATIONS WHERE ID IN ({placeholders})",
                tuple(context_ids),
            )
            owner_map = {r["id"]: r["user_id"] for r in owner_rows}

        # Build a UID → email lookup from the fetched rows
        uid_to_email: Dict[str, str] = {}
        for row in rows:
            uid = row.get("uid", "")
            email = row.get("email", "")
            if uid and email:
                uid_to_email[uid] = email

        # Fetch emails for owners not already in uid_to_email
        # (owner may not have submitted anything themselves)
        missing_owner_uids = [
            uid for uid in set(owner_map.values())
            if uid and uid not in uid_to_email
        ]
        if missing_owner_uids:
            placeholders = ", ".join(["?"] * len(missing_owner_uids))
            email_rows = await _fetch_all(
                f"SELECT UID, EMAIL FROM IDEATION_SUBMISSIONS WHERE UID IN ({placeholders}) LIMIT 200",
                tuple(missing_owner_uids),
            )
            for er in email_rows:
                if er.get("uid") and er.get("email"):
                    uid_to_email[er["uid"]] = er["email"]

        # Override attribution: each entry gets the owner's email
        deduped = list(best_by_context.values())
        for row in deduped:
            ctx = row.get("context_id", "")
            owner_uid = owner_map.get(ctx)
            if owner_uid and owner_uid in uid_to_email:
                row["email"] = uid_to_email[owner_uid]
                row["uid"] = owner_uid

        # Re-sort and limit
        deduped.sort(key=lambda r: (-int(r.get("score", 0)), r.get("submitted_at", "")))
        return deduped[:limit]

    async def count_company_entries(self, company_domain: str) -> int:
        """Count unique conversations on the leaderboard for a company domain."""
        from ideation_agent.hana_store import _fetch_one

        row = await _fetch_one(
            "SELECT COUNT(DISTINCT CONTEXT_ID) AS CNT FROM IDEATION_SUBMISSIONS WHERE COMPANY_DOMAIN = ? AND IS_ON_LEADERBOARD = TRUE",
            (company_domain,),
        )
        return int(row["cnt"]) if row else 0

    async def get_email_by_uid(self, uid: str) -> Optional[str]:
        """Look up a user's email from their most recent submission."""
        from ideation_agent.hana_store import _fetch_one

        if not uid:
            return None

        row = await _fetch_one(
            "SELECT EMAIL FROM IDEATION_SUBMISSIONS WHERE UID = ? ORDER BY SUBMITTED_AT DESC LIMIT 1",
            (uid,),
        )
        return row["email"] if row else None

    async def get_by_context(self, uid: str, context_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a submission by user + context_id. Returns None if not found."""
        from ideation_agent.hana_store import _fetch_one

        if not context_id:
            return None

        row = await _fetch_one(
            """SELECT SUBMISSION_ID, INDUSTRY, TITLE, PROBLEM, SOLUTION, APP_TYPE,
                      EVALUATOR_FEEDBACK, EVALUATOR_DETAILED_ANALYSIS,
                      EVALUATOR_SUGGESTIONS_JSON, APPROPRIATENESS_LEVEL,
                      SCORE, RATING, SUBMITTED_AT, CONTEXT_ID
               FROM IDEATION_SUBMISSIONS
               WHERE UID = ? AND CONTEXT_ID = ?""",
            (uid, context_id),
        )
        if not row:
            return None

        if isinstance(row.get("submitted_at"), datetime):
            row["submitted_at"] = row["submitted_at"].isoformat()
        if row.get("evaluator_suggestions_json"):
            try:
                row["evaluator_suggestions"] = json.loads(row["evaluator_suggestions_json"])
            except (json.JSONDecodeError, TypeError):
                row["evaluator_suggestions"] = []
        else:
            row["evaluator_suggestions"] = []
        return row

    async def get_all_by_context(self, uid: str, context_id: str) -> List[Dict[str, Any]]:
        """Fetch ALL submissions for a user + context_id, ordered by message_index."""
        from ideation_agent.hana_store import _fetch_all

        if not context_id:
            return []

        rows = await _fetch_all(
            """SELECT SUBMISSION_ID, INDUSTRY, TITLE, PROBLEM, SOLUTION, APP_TYPE,
                      EVALUATOR_FEEDBACK, EVALUATOR_DETAILED_ANALYSIS,
                      EVALUATOR_SUGGESTIONS_JSON, APPROPRIATENESS_LEVEL,
                      SCORE, RATING, SUBMITTED_AT, CONTEXT_ID, MESSAGE_INDEX,
                      IS_ON_LEADERBOARD
               FROM IDEATION_SUBMISSIONS
               WHERE UID = ? AND CONTEXT_ID = ?
               ORDER BY MESSAGE_INDEX ASC""",
            (uid, context_id),
        )
        for row in rows:
            if isinstance(row.get("submitted_at"), datetime):
                row["submitted_at"] = row["submitted_at"].isoformat()
            if row.get("evaluator_suggestions_json"):
                try:
                    row["evaluator_suggestions"] = json.loads(row["evaluator_suggestions_json"])
                except (json.JSONDecodeError, TypeError):
                    row["evaluator_suggestions"] = []
            else:
                row["evaluator_suggestions"] = []
        return rows

    async def get_all_by_context_any_user(self, context_id: str) -> List[Dict[str, Any]]:
        """Fetch ALL submissions for a context_id from any user, ordered by message_index.

        Used for shared conversations where any collaborator's evaluation
        should be visible to all participants.
        """
        from ideation_agent.hana_store import _fetch_all

        if not context_id:
            return []

        rows = await _fetch_all(
            """SELECT SUBMISSION_ID, UID, INDUSTRY, TITLE, PROBLEM, SOLUTION, APP_TYPE,
                      EVALUATOR_FEEDBACK, EVALUATOR_DETAILED_ANALYSIS,
                      EVALUATOR_SUGGESTIONS_JSON, APPROPRIATENESS_LEVEL,
                      SCORE, RATING, SUBMITTED_AT, CONTEXT_ID, MESSAGE_INDEX,
                      IS_ON_LEADERBOARD
               FROM IDEATION_SUBMISSIONS
               WHERE CONTEXT_ID = ?
               ORDER BY MESSAGE_INDEX ASC, SCORE DESC""",
            (context_id,),
        )
        for row in rows:
            if isinstance(row.get("submitted_at"), datetime):
                row["submitted_at"] = row["submitted_at"].isoformat()
            if row.get("evaluator_suggestions_json"):
                try:
                    row["evaluator_suggestions"] = json.loads(row["evaluator_suggestions_json"])
                except (json.JSONDecodeError, TypeError):
                    row["evaluator_suggestions"] = []
            else:
                row["evaluator_suggestions"] = []
        return rows

    async def delete_by_context(self, context_id: str) -> None:
        """Delete all submissions for a conversation (used on conversation delete)."""
        from ideation_agent.hana_store import _execute

        if not context_id:
            return
        await _execute(
            "DELETE FROM IDEATION_SUBMISSIONS WHERE CONTEXT_ID = ?",
            (context_id,),
        )


# Module-level singleton
_store: Optional[HanaSubmissionStore] = None
_store_lock = asyncio.Lock()


async def get_submission_store() -> HanaSubmissionStore:
    """Get or create the singleton submission store (initializes schema on first call)."""
    global _store
    if _store is not None:
        return _store
    async with _store_lock:
        if _store is not None:
            return _store
        store = HanaSubmissionStore()
        try:
            await store.init_schema()
            logger.info("HANA submission store initialized successfully")
        except Exception as exc:
            logger.error("Failed to initialize HANA submissions schema: %s", exc)
            raise
        _store = store
        return _store
