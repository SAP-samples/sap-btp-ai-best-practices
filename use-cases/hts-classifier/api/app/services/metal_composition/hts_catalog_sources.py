"""Managed HTS catalog source persistence and validation."""

from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import pandas as pd

from app.utils.hana import HANAConnection

from .config import MetalCompositionSettings
from .hts_catalog import CODE_MAP_COLUMNS
from .persistence_common import qualified_table as _qualified_table
from .timing import utc_now_iso


CHAPTER_FILENAME_RE = re.compile(r"^chapter(\d{2})\.csv$")
CODE_MAP_FILENAME = "hts_code_map.csv"
REQUIRED_CHAPTER_COLUMNS = [
    "HTS Number",
    "Indent",
    "Description",
    "Unit of Quantity",
    "General Rate of Duty",
    "Special Rate of Duty",
    "Column 2 Rate of Duty",
    "Quota Quantity",
    "Additional Duties",
]
REQUIRED_CODE_MAP_COLUMNS = list(CODE_MAP_COLUMNS)
INVALID_FILENAME_MESSAGE = (
    "HTS catalog uploads must be named chapterNN.csv or hts_code_map.csv, "
    "or be an official HTS chapter CSV whose first HTS code identifies the chapter."
)


@dataclass(frozen=True)
class PersistedHTSCatalogSourceFile:
    filename: str
    size_bytes: int
    content_text: str
    content_sha256: str
    uploaded_at: str

    @property
    def chapter_number(self) -> Optional[int]:
        match = CHAPTER_FILENAME_RE.fullmatch(self.filename)
        return int(match.group(1)) if match else None

    @property
    def source_kind(self) -> str:
        return "code_map" if self.filename == CODE_MAP_FILENAME else "chapter"


@dataclass(frozen=True)
class HTSCatalogRefreshState:
    last_refresh_status: str = "unknown"
    last_refresh_at: Optional[str] = None
    last_refresh_error: Optional[str] = None
    catalog_row_count: int = 0
    code_map_row_count: int = 0


def _infer_hts_chapter_filename(content_text: str) -> Optional[str]:
    reader = csv.reader(io.StringIO(content_text))
    rows = [row for row in reader if any(str(cell or "").strip() for cell in row)]
    if len(rows) < 2:
        return None
    for row in rows[1:4]:
        if not row:
            continue
        candidate_code = str(row[0] or "").strip()
        match = re.match(r"^(\d{2})", candidate_code)
        if match is not None:
            return f"chapter{match.group(1)}.csv"
    return None


def normalize_hts_catalog_filename(filename: str, *, content_text: Optional[str] = None) -> str:
    normalized = str(filename or "").strip()
    lowered = normalized.lower()
    if lowered == CODE_MAP_FILENAME:
        return CODE_MAP_FILENAME
    if CHAPTER_FILENAME_RE.fullmatch(lowered):
        return lowered
    if content_text is not None:
        inferred = _infer_hts_chapter_filename(content_text)
        if inferred is not None:
            return inferred
    raise ValueError(INVALID_FILENAME_MESSAGE)


def decode_catalog_upload(filename: str, content: bytes) -> str:
    display_filename = str(filename or "").strip() or "uploaded CSV"
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{display_filename} must be UTF-8 encoded.") from exc
    if not text.strip():
        raise ValueError(f"{display_filename} is empty.")
    return text


def validate_catalog_source_text(filename: str, content_text: str) -> None:
    normalized_filename = normalize_hts_catalog_filename(filename, content_text=content_text)
    frame = pd.read_csv(io.StringIO(content_text))
    required_columns = (
        REQUIRED_CODE_MAP_COLUMNS
        if normalized_filename == CODE_MAP_FILENAME
        else REQUIRED_CHAPTER_COLUMNS
    )
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            f"{normalized_filename} is missing required columns: {', '.join(missing_columns)}."
        )
    if normalized_filename != CODE_MAP_FILENAME and frame.empty:
        raise ValueError(f"{normalized_filename} must include at least one HTS row.")


def load_seed_hts_catalog_sources(settings: MetalCompositionSettings) -> List[tuple[str, str]]:
    uploads: List[tuple[str, str]] = []
    for csv_path in sorted(settings.hts_catalog_dir.glob("chapter*.csv")):
        uploads.append((csv_path.name.lower(), csv_path.read_text(encoding="utf-8-sig")))
    if settings.hts_code_map_path.exists():
        uploads.append((CODE_MAP_FILENAME, settings.hts_code_map_path.read_text(encoding="utf-8-sig")))
    return uploads


class InMemoryHTSCatalogSourceStore:
    """In-memory managed HTS source set used by tests."""

    def __init__(self, settings: Optional[MetalCompositionSettings] = None) -> None:
        self.settings = settings
        self._sources: Dict[str, PersistedHTSCatalogSourceFile] = {}
        self._refresh_state = HTSCatalogRefreshState()
        if self.settings is not None:
            self._seed_from_settings_if_empty()

    def _seed_from_settings_if_empty(self) -> None:
        if self._sources or self.settings is None:
            return
        self.upsert_sources(load_seed_hts_catalog_sources(self.settings))
        self._refresh_state = HTSCatalogRefreshState(
            last_refresh_status="seeded",
            last_refresh_at=utc_now_iso(),
        )

    def list_sources(self) -> List[PersistedHTSCatalogSourceFile]:
        self._seed_from_settings_if_empty()
        return sorted(
            self._sources.values(),
            key=lambda item: (item.chapter_number is None, item.chapter_number or 0, item.filename),
        )

    def get_source_texts(self) -> Dict[str, str]:
        return {item.filename: item.content_text for item in self.list_sources()}

    def upsert_sources(
        self,
        uploads: Sequence[tuple[str, str]],
    ) -> tuple[List[PersistedHTSCatalogSourceFile], int]:
        persisted_items: List[PersistedHTSCatalogSourceFile] = []
        overwritten_count = 0
        for raw_filename, content_text in uploads:
            filename = normalize_hts_catalog_filename(raw_filename)
            if filename in self._sources:
                overwritten_count += 1
            persisted = PersistedHTSCatalogSourceFile(
                filename=filename,
                size_bytes=len(content_text.encode("utf-8")),
                content_text=content_text,
                content_sha256=hashlib.sha256(content_text.encode("utf-8")).hexdigest(),
                uploaded_at=utc_now_iso(),
            )
            self._sources[filename] = persisted
            persisted_items.append(persisted)
        return persisted_items, overwritten_count

    def delete_source(self, filename: str) -> Optional[PersistedHTSCatalogSourceFile]:
        self._seed_from_settings_if_empty()
        normalized_filename = normalize_hts_catalog_filename(filename)
        return self._sources.pop(normalized_filename, None)

    def get_refresh_state(self) -> HTSCatalogRefreshState:
        return self._refresh_state

    def set_refresh_state(
        self,
        *,
        status: str,
        last_refresh_at: Optional[str],
        last_refresh_error: Optional[str],
        catalog_row_count: Optional[int] = None,
        code_map_row_count: Optional[int] = None,
    ) -> HTSCatalogRefreshState:
        self._refresh_state = HTSCatalogRefreshState(
            last_refresh_status=status,
            last_refresh_at=last_refresh_at,
            last_refresh_error=last_refresh_error,
            catalog_row_count=(
                self._refresh_state.catalog_row_count
                if catalog_row_count is None
                else int(catalog_row_count)
            ),
            code_map_row_count=(
                self._refresh_state.code_map_row_count
                if code_map_row_count is None
                else int(code_map_row_count)
            ),
        )
        return self._refresh_state


class HTSCatalogSourceStore:
    """Persist the managed HTS catalog source set in SAP HANA."""

    def __init__(
        self,
        settings: MetalCompositionSettings,
        *,
        connection: Optional[HANAConnection] = None,
    ) -> None:
        self.settings = settings
        self.connection = connection or HANAConnection()
        self.schema = settings.hts_hana_schema or settings.hana_schema or None
        self.sources_table = settings.hts_catalog_sources_table
        self.status_table = settings.hts_catalog_status_table
        self._initialize()
        self._seed_from_settings_if_empty()

    def _initialize(self) -> None:
        if not self.connection.table_exists(self.sources_table, schema=self.schema):
            self.connection.execute(
                f"""
                CREATE COLUMN TABLE {_qualified_table(self.sources_table, self.schema)} (
                    "FILENAME" NVARCHAR(255) PRIMARY KEY,
                    "SIZE_BYTES" BIGINT NOT NULL,
                    "CONTENT_TEXT" NCLOB NOT NULL,
                    "CONTENT_SHA256" NVARCHAR(64) NOT NULL,
                    "UPLOADED_AT" NVARCHAR(64) NOT NULL
                )
                """
            )
        if not self.connection.table_exists(self.status_table, schema=self.schema):
            self.connection.execute(
                f"""
                CREATE COLUMN TABLE {_qualified_table(self.status_table, self.schema)} (
                    "STATUS_ID" NVARCHAR(32) PRIMARY KEY,
                    "LAST_REFRESH_STATUS" NVARCHAR(32) NOT NULL,
                    "LAST_REFRESH_AT" NVARCHAR(64),
                    "LAST_REFRESH_ERROR" NCLOB,
                    "CATALOG_ROW_COUNT" BIGINT NOT NULL,
                    "CODE_MAP_ROW_COUNT" BIGINT NOT NULL
                )
                """
            )

    def _seed_from_settings_if_empty(self) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                f'SELECT 1 FROM {_qualified_table(self.sources_table, self.schema)} LIMIT 1'
            )
            has_rows = cursor.fetchone() is not None
        if has_rows:
            return
        seed_uploads = load_seed_hts_catalog_sources(self.settings)
        if not seed_uploads:
            return
        self.upsert_sources(seed_uploads)
        self.set_refresh_state(
            status="seeded",
            last_refresh_at=utc_now_iso(),
            last_refresh_error=None,
        )

    def list_sources(self) -> List[PersistedHTSCatalogSourceFile]:
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    "FILENAME",
                    "SIZE_BYTES",
                    "CONTENT_TEXT",
                    "CONTENT_SHA256",
                    "UPLOADED_AT"
                FROM {_qualified_table(self.sources_table, self.schema)}
                ORDER BY "FILENAME" ASC
                """
            )
            rows = cursor.fetchall()
        items = [
            PersistedHTSCatalogSourceFile(
                filename=str(row[0]),
                size_bytes=int(row[1] or 0),
                content_text=str(row[2] or ""),
                content_sha256=str(row[3] or ""),
                uploaded_at=str(row[4] or ""),
            )
            for row in rows
        ]
        return sorted(
            items,
            key=lambda item: (item.chapter_number is None, item.chapter_number or 0, item.filename),
        )

    def get_source_texts(self) -> Dict[str, str]:
        return {item.filename: item.content_text for item in self.list_sources()}

    def upsert_sources(
        self,
        uploads: Sequence[tuple[str, str]],
    ) -> tuple[List[PersistedHTSCatalogSourceFile], int]:
        persisted_items: List[PersistedHTSCatalogSourceFile] = []
        overwritten_count = 0
        for raw_filename, content_text in uploads:
            filename = normalize_hts_catalog_filename(raw_filename)
            with self.connection.cursor() as cursor:
                cursor.execute(
                    f'SELECT 1 FROM {_qualified_table(self.sources_table, self.schema)} WHERE "FILENAME" = ?',
                    [filename],
                )
                if cursor.fetchone() is not None:
                    overwritten_count += 1
            persisted = PersistedHTSCatalogSourceFile(
                filename=filename,
                size_bytes=len(content_text.encode("utf-8")),
                content_text=content_text,
                content_sha256=hashlib.sha256(content_text.encode("utf-8")).hexdigest(),
                uploaded_at=utc_now_iso(),
            )
            self.connection.execute(
                f'DELETE FROM {_qualified_table(self.sources_table, self.schema)} WHERE "FILENAME" = ?',
                [filename],
            )
            self.connection.execute(
                f"""
                INSERT INTO {_qualified_table(self.sources_table, self.schema)} (
                    "FILENAME",
                    "SIZE_BYTES",
                    "CONTENT_TEXT",
                    "CONTENT_SHA256",
                    "UPLOADED_AT"
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    persisted.filename,
                    persisted.size_bytes,
                    persisted.content_text,
                    persisted.content_sha256,
                    persisted.uploaded_at,
                ],
            )
            persisted_items.append(persisted)
        return persisted_items, overwritten_count

    def delete_source(self, filename: str) -> Optional[PersistedHTSCatalogSourceFile]:
        normalized_filename = normalize_hts_catalog_filename(filename)
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    "FILENAME",
                    "SIZE_BYTES",
                    "CONTENT_TEXT",
                    "CONTENT_SHA256",
                    "UPLOADED_AT"
                FROM {_qualified_table(self.sources_table, self.schema)}
                WHERE "FILENAME" = ?
                """,
                [normalized_filename],
            )
            row = cursor.fetchone()
        if row is None:
            return None
        persisted = PersistedHTSCatalogSourceFile(
            filename=str(row[0]),
            size_bytes=int(row[1] or 0),
            content_text=str(row[2] or ""),
            content_sha256=str(row[3] or ""),
            uploaded_at=str(row[4] or ""),
        )
        self.connection.execute(
            f'DELETE FROM {_qualified_table(self.sources_table, self.schema)} WHERE "FILENAME" = ?',
            [normalized_filename],
        )
        return persisted

    def get_refresh_state(self) -> HTSCatalogRefreshState:
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    "LAST_REFRESH_STATUS",
                    "LAST_REFRESH_AT",
                    "LAST_REFRESH_ERROR",
                    "CATALOG_ROW_COUNT",
                    "CODE_MAP_ROW_COUNT"
                FROM {_qualified_table(self.status_table, self.schema)}
                WHERE "STATUS_ID" = 'current'
                """
            )
            row = cursor.fetchone()
        if row is None:
            return HTSCatalogRefreshState()
        return HTSCatalogRefreshState(
            last_refresh_status=str(row[0] or "unknown"),
            last_refresh_at=str(row[1]) if row[1] else None,
            last_refresh_error=str(row[2]) if row[2] else None,
            catalog_row_count=int(row[3] or 0),
            code_map_row_count=int(row[4] or 0),
        )

    def set_refresh_state(
        self,
        *,
        status: str,
        last_refresh_at: Optional[str],
        last_refresh_error: Optional[str],
        catalog_row_count: Optional[int] = None,
        code_map_row_count: Optional[int] = None,
    ) -> HTSCatalogRefreshState:
        existing = self.get_refresh_state()
        next_state = HTSCatalogRefreshState(
            last_refresh_status=status,
            last_refresh_at=last_refresh_at,
            last_refresh_error=last_refresh_error,
            catalog_row_count=existing.catalog_row_count if catalog_row_count is None else int(catalog_row_count),
            code_map_row_count=existing.code_map_row_count if code_map_row_count is None else int(code_map_row_count),
        )
        self.connection.execute(
            f'DELETE FROM {_qualified_table(self.status_table, self.schema)} WHERE "STATUS_ID" = ?',
            ["current"],
        )
        self.connection.execute(
            f"""
            INSERT INTO {_qualified_table(self.status_table, self.schema)} (
                "STATUS_ID",
                "LAST_REFRESH_STATUS",
                "LAST_REFRESH_AT",
                "LAST_REFRESH_ERROR",
                "CATALOG_ROW_COUNT",
                "CODE_MAP_ROW_COUNT"
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                "current",
                next_state.last_refresh_status,
                next_state.last_refresh_at,
                next_state.last_refresh_error,
                next_state.catalog_row_count,
                next_state.code_map_row_count,
            ],
        )
        return next_state
