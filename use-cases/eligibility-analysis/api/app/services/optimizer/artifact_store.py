"""
Workspace artifact persistence.

Stores the files generated for a completed recommendation run (workbooks, PDF,
Markdown, JSON snapshot, ZIP and the manifest) in the configured database
backend, so downloads survive process restarts. Rows live in the
optimizer_process_artifacts table keyed by (process_id = run_id, artifact_key).
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional

from ..database import get_backend

# Local SQLite location, used only when HANA is not configured (tests / local runs).
DEFAULT_DB_DIR = Path(__file__).resolve().parents[2] / "data"


def _get_default_db_path() -> Path:
    """Return the local SQLite file path, creating its directory when needed."""
    DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_DB_DIR / "optimizer_processes.db"


_ARTIFACTS_SQLITE_DDL = """
CREATE TABLE optimizer_process_artifacts (
    process_id TEXT NOT NULL,
    artifact_key TEXT NOT NULL,
    artifact_kind TEXT NOT NULL,
    storage_mode TEXT NOT NULL,
    text_content TEXT,
    binary_content BLOB,
    metadata_json TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    PRIMARY KEY (process_id, artifact_key)
)
"""

_ARTIFACTS_HANA_DDL = """
CREATE TABLE optimizer_process_artifacts (
    process_id NVARCHAR(5000) NOT NULL,
    artifact_key NVARCHAR(5000) NOT NULL,
    artifact_kind NVARCHAR(5000) NOT NULL,
    storage_mode NVARCHAR(5000) NOT NULL,
    text_content NCLOB,
    binary_content BLOB,
    metadata_json NCLOB,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    PRIMARY KEY (process_id, artifact_key)
)
"""


class OptimizerArtifactStore:
    """Persist run-scoped text and binary artifacts in SQLite or HANA using the shared backend."""

    def __init__(self, db_path: Optional[Path] = None, backend=None):
        """Initialize persisted artifacts using an optional explicitly injected test backend."""
        self.db_path = db_path or _get_default_db_path()
        self._db = backend or get_backend()
        self._ensure_tables()

    @contextmanager
    def _get_connection(self) -> Generator:
        """Yield a backend connection for the configured database."""
        with self._db.get_connection(self.db_path) as conn:
            yield conn

    def _ensure_tables(self) -> None:
        """Create the artifact table on first use."""
        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            if not self._db.table_exists(conn, "optimizer_process_artifacts"):
                cursor.execute(_ARTIFACTS_HANA_DDL if self._db.is_hana else _ARTIFACTS_SQLITE_DDL)
            self._db.commit(conn)

    def upsert_text_artifact(
        self,
        process_id: str,
        artifact_key: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        artifact_kind: str = "text",
        storage_mode: str = "text",
    ) -> None:
        """Insert or replace one text artifact (e.g. the run manifest) for a run."""
        metadata_json = json.dumps(metadata or {}, default=str)
        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            cursor.execute(
                "DELETE FROM optimizer_process_artifacts WHERE process_id = ? AND artifact_key = ?",
                (process_id, artifact_key),
            )
            cursor.execute(
                """
                INSERT INTO optimizer_process_artifacts (
                    process_id,
                    artifact_key,
                    artifact_kind,
                    storage_mode,
                    text_content,
                    binary_content,
                    metadata_json,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (
                    process_id,
                    artifact_key,
                    artifact_kind,
                    storage_mode,
                    content,
                    None,
                    metadata_json,
                ),
            )
            self._db.commit(conn)

    def get_text_artifact(self, process_id: str, artifact_key: str) -> Optional[str]:
        """Return one saved text artifact, or None when it does not exist."""
        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            cursor.execute(
                "SELECT text_content FROM optimizer_process_artifacts WHERE process_id = ? AND artifact_key = ?",
                (process_id, artifact_key),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return str(row[0]) if row[0] is not None else None

    def put_binary(self, process_id: str, artifact_key: str, content: bytes) -> None:
        """Save generated download bytes atomically under a run-scoped artifact key."""
        from ..workspace.schema import transaction
        with transaction(self._db, self.db_path) as cursor:
            cursor.execute('DELETE FROM optimizer_process_artifacts WHERE process_id = ? AND artifact_key = ?', (process_id, artifact_key))
            cursor.execute('INSERT INTO optimizer_process_artifacts '
                '(process_id,artifact_key,artifact_kind,storage_mode,binary_content,metadata_json,created_at,updated_at) '
                "VALUES (?, ?, 'download', 'binary', ?, '{}', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
                (process_id, artifact_key, content))

    def get_binary(self, process_id: str, artifact_key: str) -> Optional[bytes]:
        """Read persisted bytes for an exact run/artifact pair without accepting paths."""
        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            cursor.execute('SELECT binary_content FROM optimizer_process_artifacts WHERE process_id = ? AND artifact_key = ?', (process_id, artifact_key))
            row = cursor.fetchone()
            if row is None or row[0] is None: return None
            value = row[0].read() if hasattr(row[0], 'read') else row[0]
            return bytes(value)
