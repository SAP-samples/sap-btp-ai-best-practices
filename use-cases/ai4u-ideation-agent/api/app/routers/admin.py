"""Admin API endpoints for viewing user conversations.

Read-only access to all users' conversations for understanding usage
and improving the agent. All endpoints (except /check) require admin access.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ..admin import is_admin, require_admin
from ..auth import UserIdentity, get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Helpers ──────────────────────────────────────────────────────────────────


async def _hana_fetch_all(sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    """Execute a HANA query via the ideation_agent store's connection."""
    from ideation_agent.hana_store import _fetch_all

    return await _fetch_all(sql, params)


async def _hana_fetch_one(sql: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
    """Execute a HANA query returning a single row."""
    from ideation_agent.hana_store import _fetch_one

    return await _fetch_one(sql, params)


def _serialize_timestamps(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert datetime objects to ISO strings for JSON serialization."""
    for key in ("created_at", "updated_at"):
        if isinstance(row.get(key), datetime):
            row[key] = row[key].isoformat()
    return row


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.get("/check")
async def check_admin_status(
    admin: bool = Depends(is_admin),
) -> Dict[str, bool]:
    """Check if the current user has admin access.

    Does NOT return 403 for non-admins — just returns {isAdmin: false}.
    Used by the frontend to conditionally show the admin navigation.
    """
    return {"isAdmin": admin}


@router.get("/users")
async def list_users(
    _: UserIdentity = Depends(require_admin),
) -> Dict[str, Any]:
    """List all distinct users who have conversations, with stats."""
    try:
        rows = await _hana_fetch_all(
            """
            SELECT USER_ID,
                   COUNT(*) AS conv_count,
                   MAX(UPDATED_AT) AS last_active,
                   MAX(COMPANY_NAME) AS last_company
            FROM CONVERSATIONS
            GROUP BY USER_ID
            ORDER BY MAX(UPDATED_AT) DESC
            """
        )
    except Exception as exc:
        logger.error("Failed to query HANA for admin users list: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="HANA database unavailable. Check connection configuration.",
        )

    users = []
    for row in rows:
        if isinstance(row.get("last_active"), datetime):
            row["last_active"] = row["last_active"].isoformat()
        users.append(
            {
                "userId": row.get("user_id", ""),
                "conversationCount": row.get("conv_count", 0),
                "lastActive": row.get("last_active"),
                "lastCompany": row.get("last_company", ""),
            }
        )

    return {"users": users, "count": len(users)}


@router.get("/conversations")
async def list_conversations(
    _: UserIdentity = Depends(require_admin),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    """List conversations (optionally filtered by user), ordered by most recent."""
    try:
        if user_id:
            rows = await _hana_fetch_all(
                f"""
                SELECT ID, USER_ID, COMPANY_NAME, TITLE, STATUS,
                       MESSAGE_COUNT, CREATED_AT, UPDATED_AT
                FROM CONVERSATIONS
                WHERE USER_ID = ?
                ORDER BY UPDATED_AT DESC
                LIMIT {int(limit)} OFFSET {int(offset)}
                """,
                (user_id,),
            )
        else:
            rows = await _hana_fetch_all(
                f"""
                SELECT ID, USER_ID, COMPANY_NAME, TITLE, STATUS,
                       MESSAGE_COUNT, CREATED_AT, UPDATED_AT
                FROM CONVERSATIONS
                ORDER BY UPDATED_AT DESC
                LIMIT {int(limit)} OFFSET {int(offset)}
                """
            )
    except Exception as exc:
        logger.error("Failed to query HANA for conversations: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="HANA database unavailable. Check connection configuration.",
        )

    conversations = []
    for row in rows:
        _serialize_timestamps(row)
        conversations.append(
            {
                "id": row.get("id", ""),
                "userId": row.get("user_id", ""),
                "companyName": row.get("company_name", ""),
                "title": row.get("title", ""),
                "status": row.get("status", ""),
                "messageCount": row.get("message_count", 0),
                "createdAt": row.get("created_at"),
                "updatedAt": row.get("updated_at"),
            }
        )

    return {"conversations": conversations, "count": len(conversations)}


@router.get("/conversations/{conv_id}/messages")
async def get_conversation_messages(
    conv_id: str,
    _: UserIdentity = Depends(require_admin),
    limit: Optional[int] = Query(None, ge=1, le=500),
) -> Dict[str, Any]:
    """Get full message history for a specific conversation (admin view)."""
    try:
        # Get conversation metadata
        conv = await _hana_fetch_one(
            """
            SELECT ID, USER_ID, COMPANY_NAME, TITLE, STATUS,
                   MESSAGE_COUNT, CREATED_AT, UPDATED_AT
            FROM CONVERSATIONS
            WHERE ID = ?
            """,
            (conv_id,),
        )
    except Exception as exc:
        logger.error("Failed to query HANA for conversation %s: %s", conv_id, exc)
        raise HTTPException(status_code=503, detail="HANA database unavailable.")

    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    _serialize_timestamps(conv)

    # Get messages
    try:
        sql = """
            SELECT ID, CONVERSATION_ID, SEQUENCE_NUM, ROLE, CONTENT,
                   METADATA_JSON, CREATED_AT
            FROM CONVERSATION_MESSAGES
            WHERE CONVERSATION_ID = ?
            ORDER BY SEQUENCE_NUM ASC
        """
        if limit:
            sql += f" LIMIT {int(limit)}"
        messages_raw = await _hana_fetch_all(sql, (conv_id,))
    except Exception as exc:
        logger.error("Failed to query messages for %s: %s", conv_id, exc)
        raise HTTPException(status_code=503, detail="HANA database unavailable.")

    import json

    messages = []
    for row in messages_raw:
        if isinstance(row.get("created_at"), datetime):
            row["created_at"] = row["created_at"].isoformat()
        metadata = None
        if row.get("metadata_json"):
            try:
                metadata = json.loads(row["metadata_json"])
            except (json.JSONDecodeError, TypeError):
                pass
        messages.append(
            {
                "id": row.get("id", ""),
                "sequenceNum": row.get("sequence_num", 0),
                "role": row.get("role", ""),
                "content": row.get("content", ""),
                "metadata": metadata,
                "createdAt": row.get("created_at"),
            }
        )

    return {
        "conversation": {
            "id": conv.get("id", ""),
            "userId": conv.get("user_id", ""),
            "companyName": conv.get("company_name", ""),
            "title": conv.get("title", ""),
            "status": conv.get("status", ""),
            "messageCount": conv.get("message_count", 0),
            "createdAt": conv.get("created_at"),
            "updatedAt": conv.get("updated_at"),
        },
        "messages": messages,
    }


@router.get("/stats")
async def get_stats(
    _: UserIdentity = Depends(require_admin),
) -> Dict[str, Any]:
    """Return aggregated usage statistics for the admin dashboard."""
    from datetime import timezone

    try:
        # 1. Total sessions
        row = await _hana_fetch_one("SELECT COUNT(*) AS total FROM CONVERSATIONS")
        total_sessions = row["total"] if row else 0

        # 2. Unique users
        row = await _hana_fetch_one(
            "SELECT COUNT(DISTINCT USER_ID) AS total FROM CONVERSATIONS"
        )
        unique_users = row["total"] if row else 0

        # 3. Average messages per session
        row = await _hana_fetch_one(
            "SELECT AVG(MESSAGE_COUNT) AS avg_msg FROM CONVERSATIONS WHERE MESSAGE_COUNT > 0"
        )
        avg_messages = round(float(row["avg_msg"]), 1) if row and row["avg_msg"] else 0.0

        # 4. Completion rate (conversations that have at least one submission/evaluation)
        row = await _hana_fetch_one(
            """
            SELECT
                (SELECT COUNT(*) FROM CONVERSATIONS) AS total,
                (SELECT COUNT(DISTINCT CONTEXT_ID) FROM IDEATION_SUBMISSIONS) AS completed
            FROM DUMMY
            """
        )
        if row and row["total"] and row["total"] > 0:
            completion_rate = round(float(row["completed"] or 0) / float(row["total"]), 2)
        else:
            completion_rate = 0.0

        # 5. Sessions per day (last 30 days)
        daily_rows = await _hana_fetch_all(
            """
            SELECT TO_VARCHAR(CREATED_AT, 'YYYY-MM-DD') AS date_key, COUNT(*) AS cnt
            FROM CONVERSATIONS
            WHERE CREATED_AT >= ADD_DAYS(CURRENT_TIMESTAMP, -30)
            GROUP BY TO_VARCHAR(CREATED_AT, 'YYYY-MM-DD')
            ORDER BY date_key ASC
            """
        )
        sessions_per_day = [
            {"date": r["date_key"], "count": r["cnt"]} for r in daily_rows
        ]

        # 6. Sessions per week (last 12 weeks)
        weekly_rows = await _hana_fetch_all(
            """
            SELECT TO_VARCHAR(CREATED_AT, 'YYYY') || '-W' || LPAD(ISOWEEK(CREATED_AT), 2, '0') AS week_key,
                   COUNT(*) AS cnt
            FROM CONVERSATIONS
            WHERE CREATED_AT >= ADD_DAYS(CURRENT_TIMESTAMP, -84)
            GROUP BY TO_VARCHAR(CREATED_AT, 'YYYY') || '-W' || LPAD(ISOWEEK(CREATED_AT), 2, '0')
            ORDER BY week_key ASC
            """
        )
        sessions_per_week = [
            {"week": r["week_key"], "count": r["cnt"]} for r in weekly_rows
        ]

        # 7. Top 10 companies
        company_rows = await _hana_fetch_all(
            """
            SELECT COMPANY_NAME, COUNT(*) AS cnt
            FROM CONVERSATIONS
            WHERE COMPANY_NAME IS NOT NULL AND COMPANY_NAME != ''
            GROUP BY COMPANY_NAME
            ORDER BY cnt DESC
            LIMIT 10
            """
        )
        top_companies = [
            {"company": r["company_name"], "count": r["cnt"]} for r in company_rows
        ]

    except Exception as exc:
        logger.error("Failed to query HANA for admin stats: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="HANA database unavailable. Check connection configuration.",
        )

    # ── Users ───────────────────────────────────────────────────────────────
    users = {
        "engagementBands": [],
        "topUserCompanies": [],
        "totalCompanies": 0,
        "newUsers": 0,
        "returningUsers": 0,
    }
    try:
        row = await _hana_fetch_one(
            """
            SELECT
                SUM(CASE WHEN conv_count = 1 THEN 1 ELSE 0 END) AS band_1,
                SUM(CASE WHEN conv_count BETWEEN 2 AND 5 THEN 1 ELSE 0 END) AS band_2_5,
                SUM(CASE WHEN conv_count BETWEEN 6 AND 10 THEN 1 ELSE 0 END) AS band_6_10,
                SUM(CASE WHEN conv_count > 10 THEN 1 ELSE 0 END) AS band_10_plus
            FROM (SELECT USER_ID, COUNT(*) AS conv_count FROM CONVERSATIONS GROUP BY USER_ID)
            """
        )
        if row:
            b1 = int(row["band_1"] or 0)
            b2 = int(row["band_2_5"] or 0)
            b3 = int(row["band_6_10"] or 0)
            b4 = int(row["band_10_plus"] or 0)
            users["engagementBands"] = [
                {"band": "1 session", "count": b1},
                {"band": "2-5 sessions", "count": b2},
                {"band": "6-10 sessions", "count": b3},
                {"band": "10+ sessions", "count": b4},
            ]
            users["newUsers"] = b1
            users["returningUsers"] = b2 + b3 + b4

        row = await _hana_fetch_one(
            """
            SELECT COUNT(DISTINCT COMPANY_DOMAIN) AS total
            FROM IDEATION_SUBMISSIONS
            WHERE COMPANY_DOMAIN IS NOT NULL AND COMPANY_DOMAIN != ''
            """
        )
        users["totalCompanies"] = row["total"] if row else 0

        company_rows = await _hana_fetch_all(
            """
            SELECT COMPANY_DOMAIN, COUNT(DISTINCT UID) AS user_count
            FROM IDEATION_SUBMISSIONS
            WHERE COMPANY_DOMAIN IS NOT NULL AND COMPANY_DOMAIN != ''
            GROUP BY COMPANY_DOMAIN
            ORDER BY user_count DESC
            LIMIT 10
            """
        )
        users["topUserCompanies"] = [
            {"company": r["company_domain"], "userCount": r["user_count"]} for r in company_rows
        ]

    except Exception as exc:
        logger.warning("Failed to query user stats: %s", exc)

    # ── Outcomes & Quality (from IDEATION_SUBMISSIONS) ──────────────────
    outcomes = {
        "totalSubmissions": 0,
        "avgScore": 0,
        "minScore": 0,
        "maxScore": 0,
        "scoreDistribution": [],
        "topIndustries": [],
        "appTypeSplit": [],
        "submissionsPerDay": [],
    }
    try:
        row = await _hana_fetch_one("SELECT COUNT(*) AS total FROM IDEATION_SUBMISSIONS")
        outcomes["totalSubmissions"] = row["total"] if row else 0

        row = await _hana_fetch_one(
            """
            SELECT ROUND(AVG(SCORE), 0) AS avg_score, MIN(SCORE) AS min_score, MAX(SCORE) AS max_score
            FROM IDEATION_SUBMISSIONS WHERE SCORE > 0
            """
        )
        outcomes["avgScore"] = int(row["avg_score"]) if row and row["avg_score"] else 0
        outcomes["minScore"] = int(row["min_score"]) if row and row["min_score"] else 0
        outcomes["maxScore"] = int(row["max_score"]) if row and row["max_score"] else 0

        row = await _hana_fetch_one(
            """
            SELECT
                SUM(CASE WHEN SCORE BETWEEN 900 AND 1000 THEN 1 ELSE 0 END) AS r1,
                SUM(CASE WHEN SCORE BETWEEN 800 AND 899 THEN 1 ELSE 0 END) AS r2,
                SUM(CASE WHEN SCORE BETWEEN 650 AND 799 THEN 1 ELSE 0 END) AS r3,
                SUM(CASE WHEN SCORE BETWEEN 1 AND 649 THEN 1 ELSE 0 END) AS r4
            FROM IDEATION_SUBMISSIONS
            WHERE SCORE > 0
            """
        )
        if row:
            outcomes["scoreDistribution"] = [
                {"band": "900–1000", "count": int(row["r1"] or 0)},
                {"band": "800–899", "count": int(row["r2"] or 0)},
                {"band": "650–799", "count": int(row["r3"] or 0)},
                {"band": "1–649", "count": int(row["r4"] or 0)},
            ]

        industry_rows = await _hana_fetch_all(
            """
            SELECT INDUSTRY, COUNT(*) AS cnt
            FROM IDEATION_SUBMISSIONS
            WHERE INDUSTRY IS NOT NULL AND INDUSTRY != ''
            GROUP BY INDUSTRY
            ORDER BY cnt DESC
            LIMIT 10
            """
        )
        outcomes["topIndustries"] = [
            {"industry": r["industry"], "count": r["cnt"]} for r in industry_rows
        ]

        app_type_rows = await _hana_fetch_all(
            """
            SELECT APP_TYPE AS app_type, COUNT(*) AS cnt
            FROM IDEATION_SUBMISSIONS
            WHERE APP_TYPE IS NOT NULL AND APP_TYPE != ''
            GROUP BY APP_TYPE
            ORDER BY cnt DESC
            """
        )
        outcomes["appTypeSplit"] = [
            {"type": r["app_type"], "count": r["cnt"]} for r in app_type_rows
        ]

        sub_daily_rows = await _hana_fetch_all(
            """
            SELECT TO_VARCHAR(SUBMITTED_AT, 'YYYY-MM-DD') AS date_key, COUNT(*) AS cnt
            FROM IDEATION_SUBMISSIONS
            WHERE SUBMITTED_AT >= ADD_DAYS(CURRENT_TIMESTAMP, -30)
            GROUP BY TO_VARCHAR(SUBMITTED_AT, 'YYYY-MM-DD')
            ORDER BY date_key ASC
            """
        )
        outcomes["submissionsPerDay"] = [
            {"date": r["date_key"], "count": r["cnt"]} for r in sub_daily_rows
        ]

    except Exception as exc:
        logger.warning("Failed to query outcomes stats (table may not exist): %s", exc)

    # ── Collaboration & Events ──────────────────────────────────────────
    collaboration = {
        "totalSharedConversations": 0,
        "totalShareMembers": 0,
        "totalEvents": 0,
        "activeEvents": 0,
        "endedEvents": 0,
        "totalEventMembers": 0,
        "topEvents": [],
    }
    try:
        row = await _hana_fetch_one(
            "SELECT COUNT(DISTINCT CONVERSATION_ID) AS total FROM CONVERSATION_SHARES WHERE REVOKED_AT IS NULL"
        )
        collaboration["totalSharedConversations"] = row["total"] if row else 0

        row = await _hana_fetch_one(
            "SELECT COUNT(*) AS total FROM CONVERSATION_SHARE_MEMBERS"
        )
        collaboration["totalShareMembers"] = row["total"] if row else 0

        row = await _hana_fetch_one(
            """
            SELECT
                COUNT(*) AS total_events,
                SUM(CASE WHEN STATUS = 'active' THEN 1 ELSE 0 END) AS active_events,
                SUM(CASE WHEN STATUS != 'active' THEN 1 ELSE 0 END) AS ended_events
            FROM EVENTS
            """
        )
        if row:
            collaboration["totalEvents"] = int(row["total_events"] or 0)
            collaboration["activeEvents"] = int(row["active_events"] or 0)
            collaboration["endedEvents"] = int(row["ended_events"] or 0)

        row = await _hana_fetch_one("SELECT COUNT(*) AS total FROM EVENT_MEMBERS")
        collaboration["totalEventMembers"] = row["total"] if row else 0

        top_event_rows = await _hana_fetch_all(
            """
            SELECT E.NAME, COUNT(M.UID) AS member_count
            FROM EVENTS E
            INNER JOIN EVENT_MEMBERS M ON E.EVENT_ID = M.EVENT_ID
            GROUP BY E.EVENT_ID, E.NAME
            ORDER BY member_count DESC
            LIMIT 5
            """
        )
        collaboration["topEvents"] = [
            {"name": r["name"], "memberCount": r["member_count"]} for r in top_event_rows
        ]

    except Exception as exc:
        logger.warning("Failed to query collaboration stats (tables may not exist): %s", exc)

    return {
        "totalSessions": total_sessions,
        "uniqueUsers": unique_users,
        "avgMessagesPerSession": avg_messages,
        "completionRate": completion_rate,
        "sessionsPerDay": sessions_per_day,
        "sessionsPerWeek": sessions_per_week,
        "topCompanies": top_companies,
        "users": users,
        "outcomes": outcomes,
        "collaboration": collaboration,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/conversations/{conv_id}/research")
async def get_conversation_research(
    conv_id: str,
    _: UserIdentity = Depends(require_admin),
) -> Dict[str, Any]:
    """Get all research data for a specific conversation (admin view)."""
    import json

    try:
        rows = await _hana_fetch_all(
            """
            SELECT RESEARCH_TYPE, STATUS, PAYLOAD_JSON, UPDATED_AT
            FROM CONVERSATION_RESEARCH
            WHERE CONVERSATION_ID = ?
            """,
            (conv_id,),
        )
    except Exception as exc:
        logger.error("Failed to query research for %s: %s", conv_id, exc)
        raise HTTPException(status_code=503, detail="HANA database unavailable.")

    research: Dict[str, Any] = {}
    for row in rows:
        payload = None
        if row.get("payload_json"):
            try:
                payload = json.loads(row["payload_json"])
            except (json.JSONDecodeError, TypeError):
                pass
        updated_at = row.get("updated_at")
        if isinstance(updated_at, datetime):
            updated_at = updated_at.isoformat()
        research[row.get("research_type", "")] = {
            "status": row.get("status", ""),
            "payload": payload,
            "updatedAt": updated_at,
        }

    return {"research": research}


# ── Telemetry (SAP-hosted instance only) ─────────────────────────────────────


@router.get("/telemetry")
async def get_telemetry_data(
    _: UserIdentity = Depends(require_admin),
) -> Dict[str, Any]:
    """Return telemetry data for the admin dashboard.

    Only available on the SAP-hosted instance (INSTALLATION_ID=sap-hosted-primary).
    Returns 404 on customer instances so the frontend hides the section gracefully.
    """
    import os

    if os.getenv("INSTALLATION_ID", "").strip() != "sap-hosted-primary":
        raise HTTPException(status_code=404, detail="Not available")

    try:
        from ..services.hana_telemetry import get_telemetry_store
        from datetime import timedelta, date

        store = await get_telemetry_store()
        summary = await store.get_telemetry_summary()
        installations = await store.get_all_installations()
        daily = await store.get_daily_stats(
            from_date=date.today() - timedelta(days=30),
            to_date=date.today(),
        )
        return {
            "summary": summary,
            "installations": installations,
            "daily": daily,
        }
    except Exception as exc:
        logger.error("Failed to query telemetry data: %s", exc)
        raise HTTPException(status_code=503, detail="Telemetry data unavailable.")

