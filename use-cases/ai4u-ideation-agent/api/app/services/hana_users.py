"""HANA Cloud user profile storage.

Provides user profile persistence (login tracking, admin role) when CDC is
not configured. Uses the same HANA connection as other app services.

Table auto-creates on first access.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_TABLE_ENSURED = False
_TABLE_LOCK = asyncio.Lock()

_TABLE_NAME = "IDEATION_USERS"

_CREATE_TABLE_SQL = f"""
CREATE TABLE "{_TABLE_NAME}" (
    EMAIL VARCHAR(256) PRIMARY KEY,
    DISPLAY_NAME VARCHAR(256),
    ROLE VARCHAR(32) DEFAULT 'user',
    COMPANY_DOMAIN VARCHAR(256) DEFAULT '',
    FIRST_LOGIN TIMESTAMP,
    LAST_LOGIN TIMESTAMP,
    CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""


def _get_connection():
    """Get a synchronous HANA connection using shared credentials."""
    from hdbcli import dbapi

    host = os.getenv("HANA_HOST", "")
    if not host:
        raise RuntimeError("HANA_HOST not configured")

    conn = dbapi.connect(
        address=host,
        port=int(os.getenv("HANA_PORT", "443")),
        user=os.getenv("HANA_USER", ""),
        password=os.getenv("HANA_PASSWORD", ""),
        encrypt=os.getenv("HANA_ENCRYPT", "true").lower() in ("true", "1", "yes"),
        sslValidateCertificate=False,
        autocommit=True,
    )
    schema = os.getenv("HANA_SCHEMA", "")
    if schema:
        cursor = conn.cursor()
        cursor.execute(f'SET SCHEMA "{schema}"')
        cursor.close()
    return conn


async def _ensure_table():
    """Create the users table if it doesn't exist (once per process)."""
    global _TABLE_ENSURED
    if _TABLE_ENSURED:
        return

    async with _TABLE_LOCK:
        if _TABLE_ENSURED:
            return

        def _create():
            conn = _get_connection()
            cursor = conn.cursor()
            try:
                cursor.execute(
                    f"SELECT COUNT(*) FROM SYS.TABLES WHERE TABLE_NAME = '{_TABLE_NAME}'"
                )
                row = cursor.fetchone()
                if row and row[0] == 0:
                    cursor.execute(_CREATE_TABLE_SQL)
                    logger.info("Created %s table", _TABLE_NAME)
            finally:
                cursor.close()
                conn.close()

        try:
            await asyncio.to_thread(_create)
            _TABLE_ENSURED = True
        except Exception as exc:
            logger.error("Failed to ensure %s table: %s", _TABLE_NAME, exc)
            raise


async def track_login(email: str, display_name: str = "") -> None:
    """Record a user login. Creates the user row if it doesn't exist.

    This is the HANA equivalent of CDC's store_user_login_data().
    """
    await _ensure_table()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # Determine company domain
    domain = ""
    if email and "@" in email:
        email_domain = email.split("@", 1)[1].lower()
        public_domains = {"gmail.com", "hotmail.com", "outlook.com", "yahoo.com", "icloud.com"}
        if email_domain not in public_domains:
            domain = email_domain

    def _upsert():
        conn = _get_connection()
        cursor = conn.cursor()
        try:
            # Check if user exists
            cursor.execute(
                f'SELECT EMAIL, FIRST_LOGIN FROM "{_TABLE_NAME}" WHERE EMAIL = ?',
                (email.lower(),),
            )
            row = cursor.fetchone()

            if row:
                # Update last login
                cursor.execute(
                    f'UPDATE "{_TABLE_NAME}" SET LAST_LOGIN = ?, DISPLAY_NAME = COALESCE(NULLIF(?, \'\'), DISPLAY_NAME), COMPANY_DOMAIN = ? WHERE EMAIL = ?',
                    (now, display_name, domain, email.lower()),
                )
            else:
                # Insert new user
                cursor.execute(
                    f'INSERT INTO "{_TABLE_NAME}" (EMAIL, DISPLAY_NAME, ROLE, COMPANY_DOMAIN, FIRST_LOGIN, LAST_LOGIN) VALUES (?, ?, ?, ?, ?, ?)',
                    (email.lower(), display_name, "user", domain, now, now),
                )
        finally:
            cursor.close()
            conn.close()

    await asyncio.to_thread(_upsert)
    logger.debug("HANA login tracked for %s", email)


async def get_user(email: str) -> Optional[Dict[str, Any]]:
    """Retrieve a user profile by email. Returns None if not found."""
    await _ensure_table()

    def _fetch():
        conn = _get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                f'SELECT EMAIL, DISPLAY_NAME, ROLE, COMPANY_DOMAIN, FIRST_LOGIN, LAST_LOGIN FROM "{_TABLE_NAME}" WHERE EMAIL = ?',
                (email.lower(),),
            )
            row = cursor.fetchone()
            if not row:
                return None
            cols = ["email", "display_name", "role", "company_domain", "first_login", "last_login"]
            return dict(zip(cols, row))
        finally:
            cursor.close()
            conn.close()

    return await asyncio.to_thread(_fetch)


async def get_user_role(email: str) -> Optional[str]:
    """Get a user's role from HANA. Returns None if user not found."""
    user = await get_user(email)
    if user:
        return user.get("role")
    return None


async def set_user_role(email: str, role: str) -> None:
    """Set a user's role in HANA (e.g., 'admin', 'user')."""
    await _ensure_table()

    def _update():
        conn = _get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                f'UPDATE "{_TABLE_NAME}" SET ROLE = ? WHERE EMAIL = ?',
                (role, email.lower()),
            )
        finally:
            cursor.close()
            conn.close()

    await asyncio.to_thread(_update)
