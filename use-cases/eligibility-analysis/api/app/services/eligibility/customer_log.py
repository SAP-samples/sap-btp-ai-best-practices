"""
Customer Log Service

Database-backed persistence for tracking historical invoice eligibility results.
Enables analysis of rejection patterns per seller over time.
Supports both SQLite (local) and SAP HANA Cloud (production) backends.
"""

import json
import logging
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Generator, List, Optional

from ...config.eligibility_config import get_database_path
from ...models.eligibility import (
    CustomerLogEntry,
    CustomerLogSummary,
    EligibilityResult,
    OfferInvoice,
    RuleCode,
    RuleRejectionStats,
    RULE_DESCRIPTIONS,
)
from ..database import (
    get_backend,
    map_type,
    parse_date,
    parse_datetime,
    parse_decimal,
    sql_add_column,
    sql_group_concat_distinct,
    sql_reset_identity,
)

logger = logging.getLogger(__name__)


def _decimal_to_str(value: Optional[Decimal]) -> Optional[str]:
    return str(value) if value is not None else None



_SQLITE_DDL = """
CREATE TABLE invoice_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    programa TEXT,
    seller_id TEXT NOT NULL,
    seller_name TEXT,
    debtor_id TEXT,
    debtor_name TEXT,
    insurer_id TEXT,
    invoice_ref TEXT NOT NULL,
    doc_number TEXT,
    fiscal_year TEXT,
    reference_number TEXT,
    goods_services TEXT,
    item TEXT,
    total_invoice_amount_original TEXT,
    total_net_value_original TEXT,
    discount_percentage TEXT,
    original_currency TEXT,
    funding_currency TEXT,
    exchange_rate TEXT,
    despatch_date TEXT,
    issuance_date TEXT,
    due_date TEXT,
    margin TEXT,
    purchase_date TEXT,
    processed_date TIMESTAMP NOT NULL,
    is_eligible BOOLEAN NOT NULL,
    rejection_rules TEXT,
    amount TEXT,
    currency TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

_HANA_DDL = """
CREATE TABLE invoice_logs (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    programa NVARCHAR(5000),
    seller_id NVARCHAR(5000) NOT NULL,
    seller_name NVARCHAR(5000),
    debtor_id NVARCHAR(5000),
    debtor_name NVARCHAR(5000),
    insurer_id NVARCHAR(5000),
    invoice_ref NVARCHAR(5000) NOT NULL,
    doc_number NVARCHAR(5000),
    fiscal_year NVARCHAR(5000),
    reference_number NVARCHAR(5000),
    goods_services NVARCHAR(5000),
    item NVARCHAR(5000),
    total_invoice_amount_original NVARCHAR(5000),
    total_net_value_original NVARCHAR(5000),
    discount_percentage NVARCHAR(5000),
    original_currency NVARCHAR(5000),
    funding_currency NVARCHAR(5000),
    exchange_rate NVARCHAR(5000),
    despatch_date NVARCHAR(5000),
    issuance_date NVARCHAR(5000),
    due_date NVARCHAR(5000),
    margin NVARCHAR(5000),
    purchase_date NVARCHAR(5000),
    processed_date TIMESTAMP NOT NULL,
    is_eligible BOOLEAN NOT NULL,
    rejection_rules NCLOB,
    amount NVARCHAR(5000),
    currency NVARCHAR(5000),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""


class CustomerLogService:
    """
    Service for persisting and querying invoice eligibility history.

    This service maintains a database of all processed invoices,
    enabling historical analysis of rejection patterns per seller.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the customer log service.

        Args:
            db_path: Path to the database file (defaults to config path; ignored for HANA)
        """
        self.db_path = db_path or get_database_path()
        self._db = get_backend()
        self._ensure_tables()

    @contextmanager
    def _get_connection(self) -> Generator:
        with self._db.get_connection(self.db_path) as conn:
            yield conn

    def _ensure_tables(self) -> None:
        with self._get_connection() as conn:
            if not self._db.table_exists(conn, "invoice_logs"):
                cursor = self._db.cursor(conn)
                cursor.execute(_HANA_DDL if self._db.is_hana else _SQLITE_DDL)
                self._db.commit(conn)

            existing = self._db.get_table_columns(conn, "invoice_logs")
            columns_to_add = {
                "programa": "TEXT",
                "debtor_id": "TEXT",
                "debtor_name": "TEXT",
                "insurer_id": "TEXT",
                "doc_number": "TEXT",
                "fiscal_year": "TEXT",
                "reference_number": "TEXT",
                "goods_services": "TEXT",
                "item": "TEXT",
                "total_invoice_amount_original": "TEXT",
                "total_net_value_original": "TEXT",
                "discount_percentage": "TEXT",
                "original_currency": "TEXT",
                "funding_currency": "TEXT",
                "exchange_rate": "TEXT",
                "despatch_date": "TEXT",
                "issuance_date": "TEXT",
                "due_date": "TEXT",
                "margin": "TEXT",
                "purchase_date": "TEXT",
                "amount": "TEXT",
                "currency": "TEXT",
            }
            cursor = self._db.cursor(conn)
            for column, col_type in columns_to_add.items():
                if column in existing:
                    continue
                stmt = sql_add_column("invoice_logs", column, map_type(col_type))
                cursor.execute(stmt)

            _indexes = {
                "idx_seller_id": "invoice_logs(seller_id)",
                "idx_debtor_id": "invoice_logs(debtor_id)",
                "idx_processed_date": "invoice_logs(processed_date)",
                "idx_duplicate_invoice_key": "invoice_logs(doc_number, fiscal_year, reference_number)",
                "idx_seller_invoice": "invoice_logs(seller_id, invoice_ref)",
            }
            for idx_name, idx_def in _indexes.items():
                if not self._db.index_exists(conn, idx_name):
                    cursor.execute(f"CREATE INDEX {idx_name} ON {idx_def}")

            self._db.commit(conn)
            logger.info(f"Database tables ensured at {self.db_path}")

    def _row_to_entry(self, row) -> CustomerLogEntry:
        rejection_rules = None
        if row["rejection_rules"]:
            rejection_rules = json.loads(row["rejection_rules"])

        amount = parse_decimal(row["amount"])
        total_invoice_amount_original = parse_decimal(row["total_invoice_amount_original"])
        total_net_value_original = parse_decimal(row["total_net_value_original"])
        discount_percentage = parse_decimal(row["discount_percentage"])
        exchange_rate = parse_decimal(row["exchange_rate"])
        margin = parse_decimal(row["margin"])
        despatch_date = parse_date(row["despatch_date"])
        issuance_date = parse_date(row["issuance_date"])
        due_date = parse_date(row["due_date"])
        purchase_date = parse_date(row["purchase_date"])

        processed_date = parse_datetime(row["processed_date"])

        return CustomerLogEntry(
            id=row["id"],
            programa=row["programa"],
            seller_id=row["seller_id"],
            seller_name=row["seller_name"],
            debtor_id=row["debtor_id"],
            debtor_name=row["debtor_name"],
            insurer_id=row["insurer_id"],
            doc_number=row["doc_number"],
            fiscal_year=row["fiscal_year"],
            reference_number=row["reference_number"],
            invoice_ref=row["invoice_ref"],
            goods_services=row["goods_services"],
            item=row["item"],
            total_invoice_amount_original=total_invoice_amount_original,
            total_net_value_original=total_net_value_original,
            discount_percentage=discount_percentage,
            original_currency=row["original_currency"],
            funding_currency=row["funding_currency"],
            exchange_rate=exchange_rate,
            despatch_date=despatch_date,
            issuance_date=issuance_date,
            due_date=due_date,
            margin=margin,
            purchase_date=purchase_date,
            processed_date=processed_date,
            is_eligible=bool(row["is_eligible"]),
            rejection_rules=rejection_rules,
            amount=amount,
            currency=row["currency"],
        )

    def log_result(
        self,
        invoice: OfferInvoice,
        result: EligibilityResult,
        purchase_date: Optional[date] = None,
        processed_date: Optional[datetime] = None,
    ) -> CustomerLogEntry:
        """
        Log a single invoice eligibility result.

        Args:
            invoice: The original invoice
            result: The eligibility result
            processed_date: When the invoice was processed (defaults to now)

        Returns:
            The created log entry
        """
        processed_date = processed_date or datetime.now()
        purchase_date = purchase_date or date.today()
        rejection_rules = (
            [r.rule_code.value for r in result.failed_rules]
            if result.failed_rules
            else None
        )
        amount_str = _decimal_to_str(invoice.amount_original)
        total_invoice_amount_original_str = _decimal_to_str(
            invoice.total_invoice_amount_original
        )
        total_net_value_original_str = _decimal_to_str(invoice.total_net_value_original)
        discount_percentage_str = _decimal_to_str(invoice.discount_percentage)
        exchange_rate_str = _decimal_to_str(invoice.exchange_rate)
        margin_str = _decimal_to_str(invoice.margin)
        despatch_date_str = (
            invoice.despatch_date.isoformat() if invoice.despatch_date else None
        )
        issuance_date_str = (
            invoice.issuance_date.isoformat() if invoice.issuance_date else None
        )
        due_date_str = invoice.due_date.isoformat() if invoice.due_date else None
        purchase_date_str = purchase_date.isoformat() if purchase_date else None

        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            cursor.execute(
                """
                INSERT INTO invoice_logs
                (programa, seller_id, seller_name, debtor_id, debtor_name, insurer_id,
                 invoice_ref, doc_number, fiscal_year, reference_number, goods_services, item,
                 total_invoice_amount_original,
                 total_net_value_original, discount_percentage, original_currency,
                 funding_currency, exchange_rate, despatch_date, issuance_date, due_date,
                margin, purchase_date, processed_date, is_eligible, rejection_rules, amount, currency)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    invoice.programa,
                    invoice.seller_id,
                    invoice.seller_name,
                    invoice.debtor_id,
                    invoice.debtor_name,
                    invoice.insurer_id,
                    invoice.invoice_ref,
                    invoice.doc_number,
                    invoice.fiscal_year,
                    invoice.reference_number,
                    invoice.goods_services,
                    invoice.item,
                    total_invoice_amount_original_str,
                    total_net_value_original_str,
                    discount_percentage_str,
                    invoice.original_currency,
                    invoice.funding_currency,
                    exchange_rate_str,
                    despatch_date_str,
                    issuance_date_str,
                    due_date_str,
                    margin_str,
                    purchase_date_str,
                    processed_date.isoformat(),
                    result.is_eligible,
                    json.dumps(rejection_rules) if rejection_rules else None,
                    amount_str,
                    invoice.original_currency,
                ),
            )
            self._db.commit(conn)
            entry_id = cursor.lastrowid

        return CustomerLogEntry(
            id=entry_id,
            programa=invoice.programa,
            seller_id=invoice.seller_id,
            seller_name=invoice.seller_name,
            debtor_id=invoice.debtor_id,
            debtor_name=invoice.debtor_name,
            insurer_id=invoice.insurer_id,
            doc_number=invoice.doc_number,
            fiscal_year=invoice.fiscal_year,
            reference_number=invoice.reference_number,
            invoice_ref=invoice.invoice_ref,
            goods_services=invoice.goods_services,
            item=invoice.item,
            total_invoice_amount_original=invoice.total_invoice_amount_original,
            total_net_value_original=invoice.total_net_value_original,
            discount_percentage=invoice.discount_percentage,
            original_currency=invoice.original_currency,
            funding_currency=invoice.funding_currency,
            exchange_rate=invoice.exchange_rate,
            despatch_date=invoice.despatch_date,
            issuance_date=invoice.issuance_date,
            due_date=invoice.due_date,
            margin=invoice.margin,
            purchase_date=purchase_date,
            processed_date=processed_date,
            is_eligible=result.is_eligible,
            rejection_rules=rejection_rules,
            amount=invoice.amount_original,
            currency=invoice.original_currency,
        )

    def log_batch(
        self,
        invoices: List[OfferInvoice],
        results: List[EligibilityResult],
        purchase_date: Optional[date] = None,
        processed_date: Optional[datetime] = None,
    ) -> int:
        """
        Log a batch of invoice eligibility results.

        Args:
            invoices: List of original invoices
            results: List of eligibility results (same order as invoices)
            processed_date: When the batch was processed (defaults to now)

        Returns:
            Number of entries logged
        """
        if len(invoices) != len(results):
            raise ValueError("Invoices and results lists must have same length")

        processed_date = processed_date or datetime.now()
        purchase_date = purchase_date or date.today()

        entries = []
        for invoice, result in zip(invoices, results):
            rejection_rules = (
                [r.rule_code.value for r in result.failed_rules]
                if result.failed_rules
                else None
            )
            amount_str = _decimal_to_str(invoice.amount_original)
            total_invoice_amount_original_str = _decimal_to_str(
                invoice.total_invoice_amount_original
            )
            total_net_value_original_str = _decimal_to_str(
                invoice.total_net_value_original
            )
            discount_percentage_str = _decimal_to_str(invoice.discount_percentage)
            exchange_rate_str = _decimal_to_str(invoice.exchange_rate)
            margin_str = _decimal_to_str(invoice.margin)
            despatch_date_str = (
                invoice.despatch_date.isoformat() if invoice.despatch_date else None
            )
            issuance_date_str = (
                invoice.issuance_date.isoformat() if invoice.issuance_date else None
            )
            due_date_str = invoice.due_date.isoformat() if invoice.due_date else None
            purchase_date_str = purchase_date.isoformat() if purchase_date else None
            entries.append(
                (
                    invoice.programa,
                    invoice.seller_id,
                    invoice.seller_name,
                    invoice.debtor_id,
                    invoice.debtor_name,
                    invoice.insurer_id,
                    invoice.invoice_ref,
                    invoice.doc_number,
                    invoice.fiscal_year,
                    invoice.reference_number,
                    invoice.goods_services,
                    invoice.item,
                    total_invoice_amount_original_str,
                    total_net_value_original_str,
                    discount_percentage_str,
                    invoice.original_currency,
                    invoice.funding_currency,
                    exchange_rate_str,
                    despatch_date_str,
                    issuance_date_str,
                    due_date_str,
                    margin_str,
                    purchase_date_str,
                    processed_date.isoformat(),
                    result.is_eligible,
                    json.dumps(rejection_rules) if rejection_rules else None,
                    amount_str,
                    invoice.original_currency,
                )
            )

        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            cursor.executemany(
                """
                INSERT INTO invoice_logs
                (programa, seller_id, seller_name, debtor_id, debtor_name, insurer_id,
                 invoice_ref, doc_number, fiscal_year, reference_number, goods_services, item,
                 total_invoice_amount_original,
                 total_net_value_original, discount_percentage, original_currency,
                 funding_currency, exchange_rate, despatch_date, issuance_date, due_date,
                 margin, purchase_date, processed_date, is_eligible, rejection_rules, amount, currency)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                entries,
            )
            self._db.commit(conn)

        logger.info(f"Logged {len(entries)} invoice results to database")
        return len(entries)

    def reset_logs(self) -> int:
        """Delete all customer log entries and reset autoincrement."""
        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            cursor.execute("SELECT COUNT(*) FROM invoice_logs")
            deleted = cursor.fetchone()[0]
            if self._db.is_hana:
                cursor.execute("TRUNCATE TABLE invoice_logs")
            else:
                cursor.execute("DELETE FROM invoice_logs")
                cursor.execute(sql_reset_identity("invoice_logs", "id"))
            self._db.commit(conn)
        logger.info("Customer logs reset; deleted %s entries", deleted)
        return deleted

    def get_seller_summary(self, seller_id: str) -> CustomerLogSummary:
        """
        Get a summary of rejection patterns for a seller.

        This enables analysis like "70% of rejections for this seller are R1".

        Args:
            seller_id: The seller ID to analyze

        Returns:
            CustomerLogSummary with statistics and rejection breakdown
        """
        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)

            cursor.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN is_eligible = TRUE THEN 1 ELSE 0 END) as eligible,
                    SUM(CASE WHEN is_eligible = FALSE THEN 1 ELSE 0 END) as rejected,
                    MIN(processed_date) as first_processed,
                    MAX(processed_date) as last_processed,
                    MAX(seller_name) as seller_name
                FROM invoice_logs
                WHERE seller_id = ?
                """,
                (seller_id,),
            )
            row = cursor.fetchone()

            if row is None or row["total"] == 0:
                return CustomerLogSummary(
                    seller_id=seller_id,
                    total_invoices_processed=0,
                    total_eligible=0,
                    total_rejected=0,
                    eligibility_rate=0.0,
                )

            total = row["total"]
            eligible = row["eligible"]
            rejected = row["rejected"]

            cursor.execute(
                """
                SELECT rejection_rules
                FROM invoice_logs
                WHERE seller_id = ? AND rejection_rules IS NOT NULL
                """,
                (seller_id,),
            )

            rule_counts: dict = {}
            for rule_row in cursor.fetchall():
                rules = json.loads(rule_row["rejection_rules"])
                for rule in rules:
                    rule_counts[rule] = rule_counts.get(rule, 0) + 1

            rejection_stats = []
            for rule_code, count in sorted(
                rule_counts.items(), key=lambda x: x[1], reverse=True
            ):
                percentage = round(count / rejected * 100, 2) if rejected > 0 else 0
                try:
                    rule_enum = RuleCode(rule_code)
                    description = RULE_DESCRIPTIONS.get(rule_enum, "Unknown rule")
                except ValueError:
                    description = "Unknown rule"

                rejection_stats.append(
                    RuleRejectionStats(
                        rule_code=rule_code,
                        rule_description=description,
                        count=count,
                        percentage=percentage,
                    )
                )

            first_processed = parse_datetime(row["first_processed"])
            last_processed = parse_datetime(row["last_processed"])

            return CustomerLogSummary(
                seller_id=seller_id,
                seller_name=row["seller_name"],
                total_invoices_processed=total,
                total_eligible=eligible,
                total_rejected=rejected,
                eligibility_rate=round(eligible / total, 4) if total > 0 else 0.0,
                rejection_by_rule=rejection_stats,
                first_processed=first_processed,
                last_processed=last_processed,
            )

    def get_seller_history(
        self,
        seller_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[CustomerLogEntry]:
        """
        Get paginated history of invoices for a seller.

        Args:
            seller_id: The seller ID to query
            limit: Maximum number of entries to return
            offset: Number of entries to skip

        Returns:
            List of CustomerLogEntry objects
        """
        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            cursor.execute(
                """
                SELECT id, programa, seller_id, seller_name, debtor_id, debtor_name,
                       insurer_id, invoice_ref, doc_number, fiscal_year, reference_number,
                       goods_services, item,
                       total_invoice_amount_original, total_net_value_original,
                       discount_percentage, original_currency, funding_currency,
                       exchange_rate, despatch_date, issuance_date, due_date, margin,
                       purchase_date,
                       processed_date, is_eligible, rejection_rules, amount, currency
                FROM invoice_logs
                WHERE seller_id = ?
                ORDER BY processed_date DESC
                LIMIT ? OFFSET ?
                """,
                (seller_id, limit, offset),
            )

            entries = []
            for row in cursor.fetchall():
                entries.append(self._row_to_entry(row))

            return entries

    def get_invoice_history(
        self,
        invoice_ref: str,
        seller_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[CustomerLogEntry]:
        """
        Get recent log entries for a specific invoice reference.

        Args:
            invoice_ref: The invoice reference to query
            seller_id: Optional seller ID filter
            limit: Maximum number of entries to return

        Returns:
            List of CustomerLogEntry objects
        """
        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            if seller_id:
                cursor.execute(
                    """
                    SELECT id, programa, seller_id, seller_name, debtor_id, debtor_name,
                           insurer_id, invoice_ref, doc_number, fiscal_year, reference_number,
                           goods_services, item,
                           total_invoice_amount_original, total_net_value_original,
                           discount_percentage, original_currency, funding_currency,
                           exchange_rate, despatch_date, issuance_date, due_date, margin,
                           purchase_date,
                           processed_date, is_eligible, rejection_rules, amount, currency
                    FROM invoice_logs
                    WHERE invoice_ref = ? AND seller_id = ?
                    ORDER BY processed_date DESC
                    LIMIT ?
                    """,
                    (invoice_ref, seller_id, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, programa, seller_id, seller_name, debtor_id, debtor_name,
                           insurer_id, invoice_ref, doc_number, fiscal_year, reference_number,
                           goods_services, item,
                           total_invoice_amount_original, total_net_value_original,
                           discount_percentage, original_currency, funding_currency,
                           exchange_rate, despatch_date, issuance_date, due_date, margin,
                           purchase_date,
                           processed_date, is_eligible, rejection_rules, amount, currency
                    FROM invoice_logs
                    WHERE invoice_ref = ?
                    ORDER BY processed_date DESC
                    LIMIT ?
                    """,
                    (invoice_ref, limit),
                )

            return [self._row_to_entry(row) for row in cursor.fetchall()]

    def search_invoices(
        self,
        seller_id: Optional[str] = None,
        debtor_id: Optional[str] = None,
        invoice_ref: Optional[str] = None,
        rule_codes: Optional[List[str]] = None,
        match_all_rules: bool = False,
        is_eligible: Optional[bool] = None,
        processed_from: Optional[str] = None,
        processed_to: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[List[CustomerLogEntry], int]:
        """
        Search invoice logs with flexible filters.

        Args:
            seller_id: Optional seller ID filter
            debtor_id: Optional debtor ID filter
            invoice_ref: Optional invoice reference filter
            rule_codes: Optional list of rejection rule codes to match
            match_all_rules: If True, require all rule codes; otherwise match any
            is_eligible: Optional eligibility filter
            processed_from: Optional ISO timestamp (inclusive) lower bound
            processed_to: Optional ISO timestamp (inclusive) upper bound
            limit: Maximum number of entries to return
            offset: Number of entries to skip

        Returns:
            Tuple of (entries, total_count)
        """
        conditions: List[str] = []
        params: List[object] = []

        if seller_id:
            conditions.append("seller_id = ?")
            params.append(seller_id)
        if debtor_id:
            conditions.append("debtor_id = ?")
            params.append(debtor_id)
        if invoice_ref:
            conditions.append("invoice_ref = ?")
            params.append(invoice_ref)
        if is_eligible is not None:
            conditions.append("is_eligible = ?")
            params.append(1 if is_eligible else 0)
        if processed_from:
            conditions.append("processed_date >= ?")
            params.append(processed_from)
        if processed_to:
            conditions.append("processed_date <= ?")
            params.append(processed_to)

        if rule_codes:
            rule_clauses: List[str] = []
            for rule_code in rule_codes:
                rule_clauses.append("rejection_rules LIKE ?")
                params.append(f'%"{rule_code}"%')
            if match_all_rules:
                conditions.extend(rule_clauses)
            else:
                conditions.append("(" + " OR ".join(rule_clauses) + ")")

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            cursor.execute(
                f"SELECT COUNT(*) as total FROM invoice_logs {where_clause}",
                tuple(params),
            )
            row = cursor.fetchone()
            total = row["total"] if row else 0

            cursor.execute(
                f"""
                SELECT id, programa, seller_id, seller_name, debtor_id, debtor_name,
                       insurer_id, invoice_ref, doc_number, fiscal_year, reference_number,
                       goods_services, item,
                       total_invoice_amount_original, total_net_value_original,
                       discount_percentage, original_currency, funding_currency,
                       exchange_rate, despatch_date, issuance_date, due_date, margin,
                       purchase_date,
                       processed_date, is_eligible, rejection_rules, amount, currency
                FROM invoice_logs
                {where_clause}
                ORDER BY processed_date DESC
                LIMIT ? OFFSET ?
                """,
                tuple(params) + (limit, offset),
            )

            entries = [self._row_to_entry(row) for row in cursor.fetchall()]

        return entries, int(total)

    def get_top_noneligible_debtors(
        self, limit: int = 10, seller_id: Optional[str] = None
    ) -> List[dict]:
        """
        Get debtors with the highest non-eligibility counts.

        Args:
            limit: Maximum number of debtors to return
            seller_id: Optional seller ID filter

        Returns:
            List of dicts with debtor stats
        """
        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            if seller_id:
                cursor.execute(
                    """
                    SELECT debtor_id,
                           MAX(debtor_name) as debtor_name,
                           COUNT(*) as total_count,
                           SUM(CASE WHEN is_eligible = FALSE THEN 1 ELSE 0 END) as rejected_count
                    FROM invoice_logs
                    WHERE debtor_id IS NOT NULL AND debtor_id != '' AND seller_id = ?
                    GROUP BY debtor_id
                    ORDER BY rejected_count DESC
                    LIMIT ?
                    """,
                    (seller_id, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT debtor_id,
                           MAX(debtor_name) as debtor_name,
                           COUNT(*) as total_count,
                           SUM(CASE WHEN is_eligible = FALSE THEN 1 ELSE 0 END) as rejected_count
                    FROM invoice_logs
                    WHERE debtor_id IS NOT NULL AND debtor_id != ''
                    GROUP BY debtor_id
                    ORDER BY rejected_count DESC
                    LIMIT ?
                    """,
                    (limit,),
                )

            rows = []
            for row in cursor.fetchall():
                total_count = row["total_count"] or 0
                rejected_count = row["rejected_count"] or 0
                rejection_rate = round(rejected_count / total_count, 4) if total_count else 0
                rows.append(
                    {
                        "debtor_id": row["debtor_id"],
                        "debtor_name": row["debtor_name"],
                        "total_count": total_count,
                        "rejected_count": rejected_count,
                        "rejection_rate": rejection_rate,
                    }
                )
            return rows

    def get_top_noneligible_sellers(self, limit: int = 10) -> List[dict]:
        """
        Get sellers with the highest non-eligibility counts.

        Args:
            limit: Maximum number of sellers to return

        Returns:
            List of dicts with seller stats
        """
        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            cursor.execute(
                """
                SELECT seller_id,
                       MAX(seller_name) as seller_name,
                       COUNT(*) as total_count,
                       SUM(CASE WHEN is_eligible = FALSE THEN 1 ELSE 0 END) as rejected_count
                FROM invoice_logs
                GROUP BY seller_id
                ORDER BY rejected_count DESC
                LIMIT ?
                """,
                (limit,),
            )

            rows = []
            for row in cursor.fetchall():
                total_count = row["total_count"] or 0
                rejected_count = row["rejected_count"] or 0
                rejection_rate = round(rejected_count / total_count, 4) if total_count else 0
                rows.append(
                    {
                        "seller_id": row["seller_id"],
                        "seller_name": row["seller_name"],
                        "total_count": total_count,
                        "rejected_count": rejected_count,
                        "rejection_rate": rejection_rate,
                    }
                )
            return rows

    def get_duplicate_invoice_groups(
        self,
        seller_id: Optional[str] = None,
        min_count: int = 2,
        limit: int = 50,
        offset: int = 0,
    ) -> List[dict]:
        """
        Get invoice keys that appear multiple times in history.

        Args:
            seller_id: Optional seller ID filter
            min_count: Minimum number of occurrences to consider a duplicate
            limit: Maximum number of groups to return
            offset: Number of groups to skip

        Returns:
            List of dicts with duplicate group stats
        """
        concat_seller_ids = sql_group_concat_distinct("seller_id")
        concat_seller_names = sql_group_concat_distinct("seller_name")

        conditions = [
            "doc_number IS NOT NULL",
            "doc_number != ''",
            "fiscal_year IS NOT NULL",
            "fiscal_year != ''",
            "reference_number IS NOT NULL",
            "reference_number != ''",
        ]
        params: List[object] = []
        if seller_id:
            conditions.append("seller_id = ?")
            params.append(seller_id)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            cursor.execute(
                f"""
                SELECT {concat_seller_ids} as seller_ids,
                       {concat_seller_names} as seller_names,
                       doc_number,
                       fiscal_year,
                       reference_number,
                       COUNT(*) as duplicate_count,
                       MIN(processed_date) as first_processed,
                       MAX(processed_date) as last_processed
                FROM invoice_logs
                {where_clause}
                GROUP BY doc_number, fiscal_year, reference_number
                HAVING COUNT(*) >= ?
                ORDER BY duplicate_count DESC, last_processed DESC
                LIMIT ? OFFSET ?
                """,
                tuple(params) + (min_count, limit, offset),
            )

            rows = []
            for row in cursor.fetchall():
                rows.append(
                    {
                        "seller_ids": row["seller_ids"],
                        "seller_names": row["seller_names"],
                        "doc_number": row["doc_number"],
                        "fiscal_year": row["fiscal_year"],
                        "reference_number": row["reference_number"],
                        "duplicate_count": row["duplicate_count"],
                        "first_processed": row["first_processed"],
                        "last_processed": row["last_processed"],
                    }
                )

        return rows

    def check_duplicate_in_history(
        self,
        doc_number: str,
        fiscal_year: str,
        reference_number: str,
    ) -> bool:
        """
        Check if an invoice has been processed before (for historical duplicate detection).

        Args:
            doc_number: The document number
            fiscal_year: The fiscal year
            reference_number: The reference number

        Returns:
            True if the invoice exists in history, False otherwise
        """
        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            cursor.execute(
                """
                SELECT COUNT(*) as count
                FROM invoice_logs
                WHERE doc_number = ? AND fiscal_year = ? AND reference_number = ?
                """,
                (doc_number, fiscal_year, reference_number),
            )
            row = cursor.fetchone()
            return row["count"] > 0 if row else False
