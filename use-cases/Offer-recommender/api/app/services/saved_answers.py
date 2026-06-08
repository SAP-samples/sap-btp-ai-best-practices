"""Persistence service for reusable customer follow-up answers."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Protocol

from app.nbo.config import TABLE_SAVED_CUSTOMER_ANSWERS
from app.nbo.fact_registry import (
    get_fact_definition,
    is_customer_answerable_fact,
    normalize_answer_value,
)
from app.nbo.hana import HanaSettings, connect, quote_identifier


SAVED_ANSWERS_TABLE = TABLE_SAVED_CUSTOMER_ANSWERS


@dataclass(frozen=True)
class SavedCustomerAnswerRecord:
    """One persisted answer overlay for a billing account and fact."""

    billing_account: str
    fact_id: str
    value: Any
    answer_label: str | None = None
    question_prompt: str | None = None
    source_surface: str | None = None
    updated_at: datetime | None = None


class SavedCustomerAnswerRepository(Protocol):
    """Storage contract used by the saved-answer service."""

    def get_answers(self, billing_account: str) -> dict[str, SavedCustomerAnswerRecord]:
        """Return saved records for one billing account."""

    def get_all_answers(self) -> dict[str, dict[str, SavedCustomerAnswerRecord]]:
        """Return saved records grouped by normalized billing account."""

    def upsert_answer(
        self,
        billing_account: str,
        fact_id: str,
        value: Any,
        *,
        answer_label: str | None = None,
        question_prompt: str | None = None,
        source_surface: str | None = None,
    ) -> None:
        """Insert or replace one saved answer."""

    def reset_account(self, billing_account: str) -> int:
        """Delete all saved answers for one billing account."""

    def reset_all(self) -> int:
        """Delete all saved answers."""


def normalize_billing_account(account: str | None) -> str:
    """Return the canonical billing account key used by saved answers."""
    if account is None:
        return ""
    normalized = str(account).strip()
    if normalized.isdigit() and len(normalized) > 1:
        normalized = str(int(normalized))
    return normalized


def create_saved_answers_table_sql(table_name: str = SAVED_ANSWERS_TABLE) -> str:
    """Build the HANA DDL for the saved customer answer table."""
    return (
        f"CREATE COLUMN TABLE {quote_identifier(table_name)} ("
        '"BILLING_ACCOUNT" NVARCHAR(100) NOT NULL, '
        '"FACT_ID" NVARCHAR(200) NOT NULL, '
        '"ANSWER_JSON" NCLOB NOT NULL, '
        '"ANSWER_LABEL" NVARCHAR(500), '
        '"QUESTION_PROMPT" NVARCHAR(5000), '
        '"SOURCE_SURFACE" NVARCHAR(100), '
        '"UPDATED_AT" TIMESTAMP NOT NULL, '
        'PRIMARY KEY ("BILLING_ACCOUNT", "FACT_ID")'
        ")"
    )


def _json_dumps_value(value: Any) -> str:
    """Serialize an answer value to compact JSON for storage."""
    return json.dumps(value, separators=(",", ":"), default=str)


def _json_loads_value(payload: str) -> Any:
    """Deserialize an answer value from storage JSON."""
    return json.loads(payload)


def _utcnow_naive() -> datetime:
    """Return the current UTC time without timezone info for HANA TIMESTAMP."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _record_from_row(row: tuple[Any, ...]) -> SavedCustomerAnswerRecord:
    """Convert one HANA result row into a saved answer record."""
    return SavedCustomerAnswerRecord(
        billing_account=normalize_billing_account(str(row[0])),
        fact_id=str(row[1]),
        value=_json_loads_value(str(row[2])),
        answer_label=str(row[3]) if row[3] is not None else None,
        question_prompt=str(row[4]) if row[4] is not None else None,
        source_surface=str(row[5]) if row[5] is not None else None,
        updated_at=row[6] if isinstance(row[6], datetime) else None,
    )


class InMemorySavedCustomerAnswerRepository:
    """Process-local saved-answer repository for tests and local development."""

    def __init__(self) -> None:
        """Create an empty in-memory repository."""
        self._records: dict[str, dict[str, SavedCustomerAnswerRecord]] = {}
        self._lock = RLock()

    def get_answers(self, billing_account: str) -> dict[str, SavedCustomerAnswerRecord]:
        """Return saved records for one normalized billing account."""
        account = normalize_billing_account(billing_account)
        with self._lock:
            return dict(self._records.get(account, {}))

    def get_all_answers(self) -> dict[str, dict[str, SavedCustomerAnswerRecord]]:
        """Return all saved records grouped by billing account."""
        with self._lock:
            return {
                account: dict(records)
                for account, records in self._records.items()
            }

    def upsert_answer(
        self,
        billing_account: str,
        fact_id: str,
        value: Any,
        *,
        answer_label: str | None = None,
        question_prompt: str | None = None,
        source_surface: str | None = None,
    ) -> None:
        """Insert or replace one in-memory saved answer."""
        account = normalize_billing_account(billing_account)
        with self._lock:
            self._records.setdefault(account, {})[fact_id] = SavedCustomerAnswerRecord(
                billing_account=account,
                fact_id=fact_id,
                value=value,
                answer_label=answer_label,
                question_prompt=question_prompt,
                source_surface=source_surface,
                updated_at=datetime.now(timezone.utc),
            )

    def reset_account(self, billing_account: str) -> int:
        """Delete all saved answers for one account and return the deleted count."""
        account = normalize_billing_account(billing_account)
        with self._lock:
            records = self._records.pop(account, {})
            return len(records)

    def reset_all(self) -> int:
        """Delete every saved answer and return the deleted count."""
        with self._lock:
            deleted = sum(len(records) for records in self._records.values())
            self._records.clear()
            return deleted


class HanaSavedCustomerAnswerRepository:
    """HANA-backed repository for saved customer answer overlays."""

    def __init__(
        self,
        connection_factory: Callable[[], Any] | None = None,
        table_name: str = SAVED_ANSWERS_TABLE,
    ) -> None:
        """Create a HANA repository using the provided connection factory."""
        self.connection_factory = connection_factory or self._connect_from_env
        self.table_name = table_name
        self._table_ready = False
        self._lock = RLock()

    @staticmethod
    def _connect_from_env() -> Any:
        """Open a HANA connection from configured environment variables."""
        return connect(HanaSettings.from_env())

    def _ensure_table(self, connection: Any) -> None:
        """Create the saved-answer table once, tolerating an existing table."""
        if self._table_ready:
            return

        with self._lock:
            if self._table_ready:
                return
            with closing(connection.cursor()) as cursor:
                try:
                    cursor.execute(create_saved_answers_table_sql(self.table_name))
                except Exception as create_error:
                    try:
                        cursor.execute(
                            f"SELECT COUNT(*) FROM {quote_identifier(self.table_name)}"
                        )
                    except Exception as select_error:
                        raise create_error from select_error
                connection.commit()
                self._table_ready = True

    def get_answers(self, billing_account: str) -> dict[str, SavedCustomerAnswerRecord]:
        """Fetch saved records for one billing account from HANA."""
        account = normalize_billing_account(billing_account)
        with closing(self.connection_factory()) as connection:
            self._ensure_table(connection)
            with closing(connection.cursor()) as cursor:
                cursor.execute(
                    (
                        "SELECT "
                        '"BILLING_ACCOUNT", "FACT_ID", "ANSWER_JSON", "ANSWER_LABEL", '
                        '"QUESTION_PROMPT", "SOURCE_SURFACE", "UPDATED_AT" '
                        f"FROM {quote_identifier(self.table_name)} "
                        'WHERE "BILLING_ACCOUNT" = ? '
                        'ORDER BY "FACT_ID"'
                    ),
                    (account,),
                )
                rows = cursor.fetchall()
        records = [_record_from_row(row) for row in rows]
        return {record.fact_id: record for record in records}

    def get_all_answers(self) -> dict[str, dict[str, SavedCustomerAnswerRecord]]:
        """Fetch every saved answer from HANA grouped by account."""
        with closing(self.connection_factory()) as connection:
            self._ensure_table(connection)
            with closing(connection.cursor()) as cursor:
                cursor.execute(
                    (
                        "SELECT "
                        '"BILLING_ACCOUNT", "FACT_ID", "ANSWER_JSON", "ANSWER_LABEL", '
                        '"QUESTION_PROMPT", "SOURCE_SURFACE", "UPDATED_AT" '
                        f"FROM {quote_identifier(self.table_name)} "
                        'ORDER BY "BILLING_ACCOUNT", "FACT_ID"'
                    )
                )
                rows = cursor.fetchall()

        grouped: dict[str, dict[str, SavedCustomerAnswerRecord]] = {}
        for record in (_record_from_row(row) for row in rows):
            grouped.setdefault(record.billing_account, {})[record.fact_id] = record
        return grouped

    def upsert_answer(
        self,
        billing_account: str,
        fact_id: str,
        value: Any,
        *,
        answer_label: str | None = None,
        question_prompt: str | None = None,
        source_surface: str | None = None,
    ) -> None:
        """Insert or replace one HANA saved answer using primary-key upsert."""
        account = normalize_billing_account(billing_account)
        with closing(self.connection_factory()) as connection:
            self._ensure_table(connection)
            with closing(connection.cursor()) as cursor:
                cursor.execute(
                    (
                        f"UPSERT {quote_identifier(self.table_name)} "
                        '("BILLING_ACCOUNT", "FACT_ID", "ANSWER_JSON", "ANSWER_LABEL", '
                        '"QUESTION_PROMPT", "SOURCE_SURFACE", "UPDATED_AT") '
                        "VALUES (?, ?, ?, ?, ?, ?, ?) WITH PRIMARY KEY"
                    ),
                    (
                        account,
                        fact_id,
                        _json_dumps_value(value),
                        answer_label,
                        question_prompt,
                        source_surface,
                        _utcnow_naive(),
                    ),
                )
                connection.commit()

    def reset_account(self, billing_account: str) -> int:
        """Delete all HANA saved answers for one account."""
        account = normalize_billing_account(billing_account)
        deleted = len(self.get_answers(account))
        with closing(self.connection_factory()) as connection:
            self._ensure_table(connection)
            with closing(connection.cursor()) as cursor:
                cursor.execute(
                    (
                        f"DELETE FROM {quote_identifier(self.table_name)} "
                        'WHERE "BILLING_ACCOUNT" = ?'
                    ),
                    (account,),
                )
                connection.commit()
        return deleted

    def reset_all(self) -> int:
        """Delete all HANA saved answers."""
        deleted = sum(len(records) for records in self.get_all_answers().values())
        with closing(self.connection_factory()) as connection:
            self._ensure_table(connection)
            with closing(connection.cursor()) as cursor:
                cursor.execute(f"DELETE FROM {quote_identifier(self.table_name)}")
                connection.commit()
        return deleted


def _has_hana_environment() -> bool:
    """Return whether the process has enough HANA variables for persistence."""
    required = (
        "hana_address",
        "hana_port",
        "hana_user",
        "hana_password",
        "hana_encrypt",
    )
    return all(os.getenv(name, "").strip() for name in required)


def build_default_saved_answer_repository() -> SavedCustomerAnswerRepository:
    """Create the default saved-answer repository for the current environment."""
    if _has_hana_environment():
        return HanaSavedCustomerAnswerRepository()
    return InMemorySavedCustomerAnswerRepository()


class SavedCustomerAnswerService:
    """Validate, persist, and read customer answer overlays."""

    def __init__(self, repository: SavedCustomerAnswerRepository | None = None) -> None:
        """Create a saved-answer service with the provided repository."""
        self.repository = repository or build_default_saved_answer_repository()

    def _normalize_valid_answer(self, fact_id: str, value: Any) -> tuple[bool, Any]:
        """Normalize a value if it is valid for a customer-answerable fact."""
        try:
            definition = get_fact_definition(fact_id)
        except KeyError:
            return False, None
        if not is_customer_answerable_fact(fact_id):
            return False, None

        options = definition.answer_options or ()
        if value is None:
            return any(option.value is None for option in options), None

        normalized = normalize_answer_value(fact_id, value)
        if normalized is not None:
            return True, normalized

        if isinstance(value, str):
            normalized_text = value.strip().casefold()
            if any(
                option.value is None and option.label.casefold() == normalized_text
                for option in options
            ):
                return True, None

        return False, None

    def _label_for_value(
        self,
        fact_id: str,
        value: Any,
        fallback: str | None = None,
    ) -> str | None:
        """Return the catalog option label for a normalized answer value."""
        if fallback:
            return fallback
        definition = get_fact_definition(fact_id)
        for option in definition.answer_options or ():
            if option.value == value:
                return option.label
        return None

    def valid_answer_values(self, answers: dict[str, Any] | None) -> dict[str, Any]:
        """Return normalized valid customer-answerable values from a raw mapping."""
        valid: dict[str, Any] = {}
        for fact_id, value in (answers or {}).items():
            is_valid, normalized = self._normalize_valid_answer(fact_id, value)
            if is_valid:
                valid[fact_id] = normalized
        return valid

    def save_answer(
        self,
        billing_account: str,
        fact_id: str,
        value: Any,
        *,
        answer_label: str | None = None,
        question_prompt: str | None = None,
        source_surface: str | None = None,
    ) -> bool:
        """Persist one valid answer and return whether it was saved."""
        is_valid, normalized = self._normalize_valid_answer(fact_id, value)
        if not is_valid:
            return False

        self.repository.upsert_answer(
            normalize_billing_account(billing_account),
            fact_id,
            normalized,
            answer_label=self._label_for_value(fact_id, normalized, answer_label),
            question_prompt=question_prompt,
            source_surface=source_surface,
        )
        return True

    def save_answers(
        self,
        billing_account: str,
        answers: dict[str, Any] | None,
        *,
        source_surface: str | None = None,
    ) -> int:
        """Persist all valid answers from a mapping and return the saved count."""
        saved = 0
        for fact_id, value in (answers or {}).items():
            if self.save_answer(
                billing_account,
                fact_id,
                value,
                source_surface=source_surface,
            ):
                saved += 1
        return saved

    def get_records(self, billing_account: str) -> dict[str, SavedCustomerAnswerRecord]:
        """Return saved answer records for one account."""
        return self.repository.get_answers(normalize_billing_account(billing_account))

    def get_answer_values(self, billing_account: str) -> dict[str, Any]:
        """Return saved answer values for one account."""
        return {
            fact_id: record.value
            for fact_id, record in self.get_records(billing_account).items()
        }

    def get_all_answer_values(self) -> dict[str, dict[str, Any]]:
        """Return all saved answer values grouped by billing account."""
        return {
            account: {
                fact_id: record.value
                for fact_id, record in records.items()
            }
            for account, records in self.repository.get_all_answers().items()
        }

    def reset_account(self, billing_account: str) -> int:
        """Reset saved answers for one account and return the deleted count."""
        return self.repository.reset_account(normalize_billing_account(billing_account))

    def reset_all(self) -> int:
        """Reset all saved answers and return the deleted count."""
        return self.repository.reset_all()


saved_answer_service = SavedCustomerAnswerService()
