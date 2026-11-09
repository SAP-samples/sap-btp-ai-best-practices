"""HANA Cloud persistence layer for telemetry data.

Stores installation pings from customer-deployed instances and daily snapshots.
Two-table schema: TELEMETRY_INSTALLATIONS (current state) + TELEMETRY_DAILY (history).

Only actively used on the SAP-hosted instance (INSTALLATION_ID=sap-hosted-primary)
where it receives and stores pings. Customer instances only write to TELEMETRY_CONFIG
for installation ID persistence.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class HanaTelemetryStore:
    """HANA-backed store for telemetry data (receiver side)."""

    async def init_schema(self) -> None:
        """Create tables (idempotent)."""
        from ideation_agent.hana_store import _execute

        # --- TELEMETRY_INSTALLATIONS table (current state, upserted on every ping) ---
        try:
            await _execute("""CREATE COLUMN TABLE TELEMETRY_INSTALLATIONS (
                INSTALLATION_ID   VARCHAR(64) PRIMARY KEY,
                CURRENT_VERSION   VARCHAR(32),
                CONVERSATIONS     INTEGER DEFAULT 0,
                USERS             INTEGER DEFAULT 0,
                SUBMISSIONS       INTEGER DEFAULT 0,
                FIRST_SEEN        TIMESTAMP NOT NULL,
                LAST_SEEN         TIMESTAMP NOT NULL
            )""")
        except Exception as exc:
            err_str = str(exc).lower()
            if "already exists" not in err_str and "duplicate" not in err_str and "288" not in err_str:
                raise

        # --- TELEMETRY_DAILY table (daily snapshots for trend analysis) ---
        try:
            await _execute("""CREATE COLUMN TABLE TELEMETRY_DAILY (
                SNAPSHOT_DATE     DATE,
                INSTALLATION_ID   VARCHAR(64),
                VERSION           VARCHAR(32),
                CONVERSATIONS     INTEGER DEFAULT 0,
                USERS             INTEGER DEFAULT 0,
                SUBMISSIONS       INTEGER DEFAULT 0,
                PRIMARY KEY (SNAPSHOT_DATE, INSTALLATION_ID)
            )""")
        except Exception as exc:
            err_str = str(exc).lower()
            if "already exists" not in err_str and "duplicate" not in err_str and "288" not in err_str:
                raise

    async def upsert_installation(
        self,
        installation_id: str,
        version: str,
        conversations: int,
        users: int,
        submissions: int,
    ) -> None:
        """Upsert an installation's current state. Preserves FIRST_SEEN on update."""
        from ideation_agent.hana_store import _execute

        now = datetime.now(timezone.utc)
        # Try INSERT first (sets FIRST_SEEN on new installations)
        try:
            await _execute(
                """INSERT INTO TELEMETRY_INSTALLATIONS
                   (INSTALLATION_ID, CURRENT_VERSION, CONVERSATIONS, USERS, SUBMISSIONS, FIRST_SEEN, LAST_SEEN)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (installation_id, version, conversations, users, submissions, now, now),
            )
        except Exception as exc:
            # Duplicate key — update existing row (preserve FIRST_SEEN)
            err_str = str(exc).lower()
            if "unique" in err_str or "duplicate" in err_str or "301" in err_str:
                await _execute(
                    """UPDATE TELEMETRY_INSTALLATIONS
                       SET CURRENT_VERSION = ?, CONVERSATIONS = ?, USERS = ?,
                           SUBMISSIONS = ?, LAST_SEEN = ?
                       WHERE INSTALLATION_ID = ?""",
                    (version, conversations, users, submissions, now, installation_id),
                )
            else:
                raise

    async def record_daily_snapshot(
        self,
        installation_id: str,
        version: str,
        conversations: int,
        users: int,
        submissions: int,
    ) -> None:
        """Record a daily snapshot. Idempotent — one entry per (date, installation)."""
        from ideation_agent.hana_store import _execute

        today = date.today()
        try:
            await _execute(
                """INSERT INTO TELEMETRY_DAILY
                   (SNAPSHOT_DATE, INSTALLATION_ID, VERSION, CONVERSATIONS, USERS, SUBMISSIONS)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (today, installation_id, version, conversations, users, submissions),
            )
        except Exception as exc:
            # Duplicate key = already recorded today — expected and fine
            err_str = str(exc).lower()
            if "unique" in err_str or "duplicate" in err_str or "301" in err_str:
                # Update today's snapshot with latest values
                await _execute(
                    """UPDATE TELEMETRY_DAILY
                       SET VERSION = ?, CONVERSATIONS = ?, USERS = ?, SUBMISSIONS = ?
                       WHERE SNAPSHOT_DATE = ? AND INSTALLATION_ID = ?""",
                    (version, conversations, users, submissions, today, installation_id),
                )
            else:
                logger.warning("Failed to record daily snapshot: %s", exc)

    async def get_all_installations(self) -> List[Dict[str, Any]]:
        """Get all tracked installations, most recently seen first."""
        from ideation_agent.hana_store import _fetch_all

        rows = await _fetch_all(
            """SELECT INSTALLATION_ID, CURRENT_VERSION, CONVERSATIONS, USERS, SUBMISSIONS,
                      FIRST_SEEN, LAST_SEEN
               FROM TELEMETRY_INSTALLATIONS
               ORDER BY LAST_SEEN DESC"""
        )
        result = []
        for row in rows:
            result.append({
                "installationId": row["installation_id"],
                "currentVersion": row["current_version"],
                "conversations": row.get("conversations", 0),
                "users": row.get("users", 0),
                "submissions": row.get("submissions", 0),
                "firstSeen": row["first_seen"].isoformat() if isinstance(row.get("first_seen"), datetime) else row.get("first_seen"),
                "lastSeen": row["last_seen"].isoformat() if isinstance(row.get("last_seen"), datetime) else row.get("last_seen"),
            })
        return result

    async def get_daily_stats(self, from_date: date, to_date: date) -> List[Dict[str, Any]]:
        """Get daily snapshots within a date range."""
        from ideation_agent.hana_store import _fetch_all

        rows = await _fetch_all(
            """SELECT SNAPSHOT_DATE, INSTALLATION_ID, VERSION, CONVERSATIONS, USERS, SUBMISSIONS
               FROM TELEMETRY_DAILY
               WHERE SNAPSHOT_DATE >= ? AND SNAPSHOT_DATE <= ?
               ORDER BY SNAPSHOT_DATE ASC""",
            (from_date, to_date),
        )
        result = []
        for row in rows:
            result.append({
                "snapshotDate": row["snapshot_date"].isoformat() if hasattr(row.get("snapshot_date"), "isoformat") else str(row.get("snapshot_date", "")),
                "installationId": row["installation_id"],
                "version": row.get("version", ""),
                "conversations": row.get("conversations", 0),
                "users": row.get("users", 0),
                "submissions": row.get("submissions", 0),
            })
        return result

    async def get_telemetry_summary(self) -> Dict[str, Any]:
        """Get aggregated telemetry summary for the admin dashboard."""
        from ideation_agent.hana_store import _fetch_one, _fetch_all

        # Total installations
        total_row = await _fetch_one("SELECT COUNT(*) AS cnt FROM TELEMETRY_INSTALLATIONS")
        total_installations = int(total_row["cnt"]) if total_row else 0

        # Active in last 7 days
        seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
        active_row = await _fetch_one(
            "SELECT COUNT(*) AS cnt FROM TELEMETRY_INSTALLATIONS WHERE LAST_SEEN >= ?",
            (seven_days_ago,),
        )
        active_7d = int(active_row["cnt"]) if active_row else 0

        # Total conversations/users/submissions across all installations
        totals_row = await _fetch_one(
            """SELECT COALESCE(SUM(CONVERSATIONS), 0) AS total_c,
                      COALESCE(SUM(USERS), 0) AS total_u,
                      COALESCE(SUM(SUBMISSIONS), 0) AS total_s
               FROM TELEMETRY_INSTALLATIONS"""
        )
        total_conversations = int(totals_row["total_c"]) if totals_row else 0
        total_users = int(totals_row["total_u"]) if totals_row else 0
        total_submissions = int(totals_row["total_s"]) if totals_row else 0

        # Version distribution
        version_rows = await _fetch_all(
            """SELECT CURRENT_VERSION AS version, COUNT(*) AS cnt
               FROM TELEMETRY_INSTALLATIONS
               GROUP BY CURRENT_VERSION
               ORDER BY cnt DESC"""
        )
        version_distribution = [
            {"version": row["version"], "count": int(row["cnt"])}
            for row in version_rows
        ]

        return {
            "totalInstallations": total_installations,
            "activeInstallations7d": active_7d,
            "totalConversationsAll": total_conversations,
            "totalUsersAll": total_users,
            "totalSubmissionsAll": total_submissions,
            "versionDistribution": version_distribution,
        }


# --- Singleton access ---

_store: Optional[HanaTelemetryStore] = None
_store_lock = asyncio.Lock()


async def get_telemetry_store() -> HanaTelemetryStore:
    """Get or create the telemetry store singleton."""
    global _store
    if _store is not None:
        return _store
    async with _store_lock:
        if _store is not None:
            return _store
        store = HanaTelemetryStore()
        try:
            await store.init_schema()
            logger.info("HANA telemetry store initialized successfully")
        except Exception as exc:
            logger.error("Failed to initialize HANA telemetry schema: %s", exc)
            raise
        _store = store
        return _store
