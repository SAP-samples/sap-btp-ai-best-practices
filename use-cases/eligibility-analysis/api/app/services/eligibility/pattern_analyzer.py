"""
Pattern Analyzer Service

Mines the invoice_logs database for recurring rejection patterns.
Detects five pattern types:
  1. Chronic Rule Failures  - debtor consistently fails a specific rule
  2. Trending Increases     - rejection rate worsening over time
  3. Repeat Offenders       - debtor rejected in most batches they appear in
  4. Rule Concentration     - single rule dominates all rejections globally
  5. Amount at Risk         - high financial impact from a specific pattern

All detection is purely algorithmic (SQL + Python). No LLM calls.
Supports both SQLite (local) and SAP HANA Cloud (production) backends.
"""

import json
import logging
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

from ...config.eligibility_config import get_database_path
from ...models.eligibility import RULE_DESCRIPTIONS, RuleCode
from ...models.patterns import (
    RULE_RECOMMENDATIONS,
    DebtorRuleProfile,
    PatternAlert,
    PatternSummary,
    PatternType,
    Severity,
    TrendDataPoint,
)
from ..database import get_backend, parse_date, parse_datetime, parse_decimal, sql_date

logger = logging.getLogger(__name__)


def _rule_description(rule_code: str) -> str:
    """Get human-readable description for a rule code."""
    try:
        return RULE_DESCRIPTIONS.get(RuleCode(rule_code), rule_code)
    except ValueError:
        return rule_code


def _rule_recommendation(rule_code: str) -> str:
    """Get actionable recommendation for a rule code."""
    return RULE_RECOMMENDATIONS.get(
        rule_code,
        "Review the eligibility configuration for this rule and the client's invoice submission patterns.",
    )


class PatternAnalyzer:
    """
    Service for detecting recurring rejection patterns in historical invoice data.

    Uses the same database as CustomerLogService (invoice_logs table).
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or get_database_path()
        self._db = get_backend()

    @contextmanager
    def _get_connection(self) -> Generator:
        with self._db.get_connection(self.db_path) as conn:
            yield conn

    @staticmethod
    def _build_filter_conditions(
        filters: "PatternFilters",
        cutoff: datetime,
        *,
        require_debtor: bool = False,
        require_noneligible: bool = False,
        require_rejection_rules: bool = False,
        require_amount: bool = False,
    ) -> tuple[list[str], list]:
        """Build WHERE conditions and params from a PatternFilters bundle."""
        from ...models.patterns import PatternFilters as _PF  # noqa: F811

        conditions = ["purchase_date >= ?"]
        params: list = [cutoff.date().isoformat()]

        if filters.seller_id:
            conditions.append("seller_id = ?")
            params.append(filters.seller_id)
        if filters.debtor_id:
            conditions.append("debtor_id = ?")
            params.append(filters.debtor_id)
        if filters.programa:
            conditions.append("programa = ?")
            params.append(filters.programa)
        if filters.insurer_id:
            conditions.append("insurer_id = ?")
            params.append(filters.insurer_id)

        if require_debtor:
            conditions.append("debtor_id IS NOT NULL")
            conditions.append("debtor_id != ''")
        if require_noneligible:
            conditions.append("is_eligible = FALSE")
        if require_rejection_rules:
            conditions.append("rejection_rules IS NOT NULL")
        if require_amount:
            conditions.append("amount IS NOT NULL")
            conditions.append("amount != ''")

        return conditions, params

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_all(
        self,
        seller_id: Optional[str] = None,
        lookback_days: int = 90,
        min_invoices: int = 3,
        *,
        filters: Optional["PatternFilters"] = None,
    ) -> PatternSummary:
        """
        Run all pattern detection algorithms and return a combined summary.

        Args:
            seller_id: If provided, restrict analysis to this seller.
            lookback_days: Number of days of history to analyze.
            min_invoices: Minimum invoice count for a pattern to be flagged.
            filters: If provided, overrides seller_id/lookback_days/min_invoices.

        Returns:
            PatternSummary with all detected patterns sorted by severity.
        """
        if filters is None:
            from ...models.patterns import PatternFilters
            filters = PatternFilters(
                seller_id=seller_id,
                lookback_days=lookback_days,
                min_invoices=min_invoices,
            )
        cutoff = datetime.now() - timedelta(days=filters.lookback_days)
        window_start, window_end, total, eligible = self._window_stats(
            filters, cutoff
        )

        patterns: List[PatternAlert] = []
        patterns.extend(
            self._detect_chronic_failures(filters, cutoff)
        )
        patterns.extend(
            self._detect_trending_increases(filters)
        )
        patterns.extend(
            self._detect_repeat_offenders(filters, cutoff)
        )
        patterns.extend(
            self._detect_rule_concentration(filters, cutoff)
        )
        patterns.extend(
            self._detect_amount_at_risk(filters, cutoff)
        )

        # Sort: HIGH first, then MEDIUM, then LOW
        severity_order = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2}
        patterns.sort(key=lambda p: severity_order.get(p.severity, 3))

        high = sum(1 for p in patterns if p.severity == Severity.HIGH)
        medium = sum(1 for p in patterns if p.severity == Severity.MEDIUM)
        low = sum(1 for p in patterns if p.severity == Severity.LOW)

        eligibility_rate = round(eligible / total, 4) if total > 0 else 0.0

        return PatternSummary(
            total_patterns=len(patterns),
            high_severity=high,
            medium_severity=medium,
            low_severity=low,
            patterns=patterns,
            analysis_window_start=window_start,
            analysis_window_end=window_end,
            total_invoices_analyzed=total,
            overall_eligibility_rate=eligibility_rate,
        )

    def get_debtor_profiles(
        self,
        seller_id: Optional[str] = None,
        lookback_days: int = 90,
        *,
        filters: Optional["PatternFilters"] = None,
    ) -> List[DebtorRuleProfile]:
        """
        Get per-debtor rejection profiles with trend data.

        Returns one DebtorRuleProfile per (seller_id, debtor_id) pair,
        sorted by rejection rate descending.
        """
        if filters is None:
            from ...models.patterns import PatternFilters
            filters = PatternFilters(
                seller_id=seller_id,
                lookback_days=lookback_days,
            )
        cutoff = datetime.now() - timedelta(days=filters.lookback_days)
        profiles: List[DebtorRuleProfile] = []

        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)

            conditions, params = self._build_filter_conditions(
                filters, cutoff, require_debtor=True
            )

            where = "WHERE " + " AND ".join(conditions)
            date_expr = sql_date("purchase_date")

            # Fetch per-row data for rule counting and amounts.
            cursor.execute(
                f"""
                SELECT seller_id, seller_name, debtor_id, debtor_name,
                       is_eligible, rejection_rules, amount,
                       {date_expr} as batch_date
                FROM invoice_logs
                {where}
                ORDER BY seller_id, debtor_id, purchase_date
                """,
                tuple(params),
            )

            # Accumulate per (seller_id, debtor_id)
            accum: Dict[Tuple[str, str], dict] = {}
            for row in cursor.fetchall():
                key = (row["seller_id"], row["debtor_id"])
                if key not in accum:
                    accum[key] = {
                        "seller_id": row["seller_id"],
                        "seller_name": row["seller_name"],
                        "debtor_id": row["debtor_id"],
                        "debtor_name": row["debtor_name"],
                        "total": 0,
                        "rejected": 0,
                        "rule_counts": {},
                        "amount_rejected": Decimal("0"),
                        "batch_dates": set(),
                        "batch_dates_rejected": set(),
                    }
                acc = accum[key]
                acc["total"] += 1
                batch_date = row["batch_date"]
                acc["batch_dates"].add(batch_date)

                if not row["is_eligible"]:
                    acc["rejected"] += 1
                    acc["batch_dates_rejected"].add(batch_date)
                    # Parse rejection rules
                    if row["rejection_rules"]:
                        rules = json.loads(row["rejection_rules"])
                        for rule in rules:
                            acc["rule_counts"][rule] = (
                                acc["rule_counts"].get(rule, 0) + 1
                            )
                    # Accumulate rejected amount
                    amt = parse_decimal(row["amount"])
                    if amt is not None and amt > 0:
                        acc["amount_rejected"] += amt

        # Build profiles
        for key, acc in sorted(
            accum.items(), key=lambda x: x[1]["rejected"], reverse=True
        ):
            total = acc["total"]
            rejected = acc["rejected"]
            rejection_rate = round(rejected / total, 4) if total > 0 else 0.0

            # Dominant rule
            dominant_rule = None
            dominant_rule_rate = 0.0
            if acc["rule_counts"] and rejected > 0:
                dominant_rule = max(
                    acc["rule_counts"], key=acc["rule_counts"].get
                )
                dominant_rule_rate = round(
                    acc["rule_counts"][dominant_rule] / rejected, 4
                )

            profiles.append(
                DebtorRuleProfile(
                    debtor_id=acc["debtor_id"],
                    debtor_name=acc["debtor_name"],
                    seller_id=acc["seller_id"],
                    seller_name=acc["seller_name"],
                    total_invoices=total,
                    rejected_invoices=rejected,
                    rejection_rate=rejection_rate,
                    dominant_rule=dominant_rule,
                    dominant_rule_description=(
                        _rule_description(dominant_rule) if dominant_rule else None
                    ),
                    dominant_rule_rate=dominant_rule_rate,
                    rejection_by_rule=acc["rule_counts"],
                    total_amount_rejected=(
                        acc["amount_rejected"]
                        if acc["amount_rejected"] > 0
                        else None
                    ),
                    batch_count=len(acc["batch_dates"]),
                    batches_with_rejection=len(acc["batch_dates_rejected"]),
                    trend=[],  # Populated on-demand via get_rejection_trend
                )
            )

        return profiles

    def get_rejection_trend(
        self,
        seller_id: Optional[str] = None,
        debtor_id: Optional[str] = None,
        granularity: str = "week",
        lookback_days: int = 90,
        *,
        filters: Optional["PatternFilters"] = None,
    ) -> List[TrendDataPoint]:
        """
        Get rejection rate time-series for charting.

        Args:
            seller_id: Filter to a specific seller.
            debtor_id: Filter to a specific debtor.
            granularity: "day", "week", "month", or "quarter".
            lookback_days: How far back to look.
            filters: If provided, overrides seller_id/debtor_id/lookback_days.

        Returns:
            List of TrendDataPoint sorted chronologically.
        """
        if filters is None:
            from ...models.patterns import PatternFilters
            filters = PatternFilters(
                seller_id=seller_id,
                debtor_id=debtor_id,
                lookback_days=lookback_days,
            )
        cutoff = datetime.now() - timedelta(days=filters.lookback_days)

        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)

            conditions, params = self._build_filter_conditions(filters, cutoff)

            where = "WHERE " + " AND ".join(conditions)
            date_expr = sql_date("purchase_date")

            cursor.execute(
                f"""
                SELECT {date_expr} as proc_date,
                       is_eligible, rejection_rules
                FROM invoice_logs
                {where}
                ORDER BY purchase_date
                """,
                tuple(params),
            )

            # Bucket rows by granularity
            buckets: Dict[date, dict] = {}
            for row in cursor.fetchall():
                proc_date = parse_date(row["proc_date"])
                bucket_start = self._bucket_date(proc_date, granularity)
                if bucket_start not in buckets:
                    buckets[bucket_start] = {
                        "total": 0,
                        "rejected": 0,
                        "rule_counts": {},
                    }
                b = buckets[bucket_start]
                b["total"] += 1
                if not row["is_eligible"]:
                    b["rejected"] += 1
                    if row["rejection_rules"]:
                        rules = json.loads(row["rejection_rules"])
                        for rule in rules:
                            b["rule_counts"][rule] = (
                                b["rule_counts"].get(rule, 0) + 1
                            )

        # Convert to TrendDataPoint list
        points: List[TrendDataPoint] = []
        for bucket_start in sorted(buckets.keys()):
            b = buckets[bucket_start]
            bucket_end = self._bucket_end(bucket_start, granularity)
            rate = round(b["rejected"] / b["total"], 4) if b["total"] > 0 else 0.0
            points.append(
                TrendDataPoint(
                    period_start=bucket_start,
                    period_end=bucket_end,
                    total_invoices=b["total"],
                    rejected_invoices=b["rejected"],
                    rejection_rate=rate,
                    rejection_by_rule=b["rule_counts"],
                )
            )
        return points

    def get_filter_options(self, lookback_days: int = 90) -> dict:
        """Return distinct filter values for the UI dropdowns."""
        cutoff = (datetime.now() - timedelta(days=lookback_days)).date().isoformat()

        # Mapping from column name to the display-name column (if one exists)
        name_columns = {
            "seller_id": "seller_name",
            "debtor_id": "debtor_name",
        }

        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            result = {}
            for col, label in [
                ("seller_id", "sellers"),
                ("debtor_id", "debtors"),
                ("programa", "programas"),
                ("insurer_id", "insurers"),
            ]:
                name_col = name_columns.get(col, col)
                cursor.execute(
                    f"""
                    SELECT DISTINCT {col}, MAX({name_col}) as display_name
                    FROM invoice_logs
                    WHERE purchase_date >= ? AND {col} IS NOT NULL AND {col} != ''
                    GROUP BY {col}
                    ORDER BY {col}
                    """,
                    (cutoff,),
                )
                result[label] = [
                    {"id": row[col], "name": row["display_name"] or row[col]}
                    for row in cursor.fetchall()
                ]
            return result

    # ------------------------------------------------------------------
    # Pattern Detection: 2a - Chronic Rule Failures
    # ------------------------------------------------------------------

    def _detect_chronic_failures(
        self,
        filters: "PatternFilters",
        cutoff: datetime,
    ) -> List[PatternAlert]:
        """
        Detect debtor-seller-rule triples with chronic rejection rates.

        Flags when a debtor fails a specific rule on >60% of their invoices
        (minimum min_invoices total).
        """
        alerts: List[PatternAlert] = []

        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)

            conditions, params = self._build_filter_conditions(
                filters, cutoff, require_debtor=True
            )

            where = "WHERE " + " AND ".join(conditions)

            # Get all non-eligible rows with their rules
            cursor.execute(
                f"""
                SELECT seller_id, MAX(seller_name) as seller_name,
                       debtor_id, MAX(debtor_name) as debtor_name,
                       COUNT(*) as total,
                       MIN(purchase_date) as first_seen,
                       MAX(purchase_date) as last_seen
                FROM invoice_logs
                {where}
                GROUP BY seller_id, debtor_id
                HAVING COUNT(*) >= ?
                """,
                tuple(params) + (filters.min_invoices,),
            )
            debtor_totals: Dict[Tuple[str, str], dict] = {}
            for row in cursor.fetchall():
                key = (row["seller_id"], row["debtor_id"])
                debtor_totals[key] = {
                    "seller_name": row["seller_name"],
                    "debtor_name": row["debtor_name"],
                    "total": row["total"],
                    "first_seen": row["first_seen"],
                    "last_seen": row["last_seen"],
                }

            # Now count per-rule failures for each debtor
            cursor.execute(
                f"""
                SELECT seller_id, debtor_id, rejection_rules
                FROM invoice_logs
                {where}
                AND is_eligible = FALSE
                AND rejection_rules IS NOT NULL
                """,
                tuple(params),
            )

            # Accumulate: (seller_id, debtor_id, rule) -> count
            rule_failures: Dict[Tuple[str, str, str], int] = {}
            for row in cursor.fetchall():
                key_base = (row["seller_id"], row["debtor_id"])
                if key_base not in debtor_totals:
                    continue
                rules = json.loads(row["rejection_rules"])
                for rule in rules:
                    triple = (row["seller_id"], row["debtor_id"], rule)
                    rule_failures[triple] = rule_failures.get(triple, 0) + 1

        # Evaluate thresholds
        for (sid, did, rule), fail_count in rule_failures.items():
            key = (sid, did)
            if key not in debtor_totals:
                continue
            info = debtor_totals[key]
            total = info["total"]
            rate = fail_count / total

            if rate < 0.6 or total < filters.min_invoices:
                continue

            severity = Severity.HIGH if rate > 0.8 else Severity.MEDIUM
            desc = _rule_description(rule)
            debtor_label = info["debtor_name"] or did
            seller_label = info["seller_name"] or sid

            alerts.append(
                PatternAlert(
                    pattern_type=PatternType.CHRONIC_RULE_FAILURE,
                    severity=severity,
                    seller_id=sid,
                    seller_name=info["seller_name"],
                    debtor_id=did,
                    debtor_name=info["debtor_name"],
                    rule_code=rule,
                    rule_description=desc,
                    title=f"{debtor_label} chronically fails {rule}",
                    description=(
                        f"Debtor '{debtor_label}' (via Seller '{seller_label}') fails "
                        f"{rule} ({desc}) on {rate:.0%} of invoices "
                        f"({fail_count}/{total})."
                    ),
                    recommendation=_rule_recommendation(rule),
                    metrics={
                        "failure_count": fail_count,
                        "total_invoices": total,
                        "failure_rate": round(rate, 4),
                    },
                    first_seen=parse_datetime(info["first_seen"]),
                    last_seen=parse_datetime(info["last_seen"]),
                )
            )
        return alerts

    # ------------------------------------------------------------------
    # Pattern Detection: 2b - Trending Increases
    # ------------------------------------------------------------------

    def _detect_trending_increases(
        self,
        filters: "PatternFilters",
    ) -> List[PatternAlert]:
        """
        Detect rules whose rejection rate is increasing over time.

        Compares the recent half of the lookback window to the prior half.
        When lookback >= 180 days, uses quarter-over-quarter comparison.
        Flags if the rejection rate increased by >15 percentage points.
        """
        alerts: List[PatternAlert] = []
        lookback_days = filters.lookback_days
        half = lookback_days // 2
        now = datetime.now()
        recent_start = now - timedelta(days=half)
        prior_start = now - timedelta(days=lookback_days)

        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)

            for label, window_start, window_end in [
                ("prior", prior_start, recent_start),
                ("recent", recent_start, now),
            ]:
                conditions = [
                    "purchase_date >= ?",
                    "purchase_date < ?",
                ]
                params: list = [window_start.date().isoformat(), window_end.date().isoformat()]
                if filters.seller_id:
                    conditions.append("seller_id = ?")
                    params.append(filters.seller_id)
                if filters.debtor_id:
                    conditions.append("debtor_id = ?")
                    params.append(filters.debtor_id)
                if filters.programa:
                    conditions.append("programa = ?")
                    params.append(filters.programa)
                if filters.insurer_id:
                    conditions.append("insurer_id = ?")
                    params.append(filters.insurer_id)

                where = "WHERE " + " AND ".join(conditions)

                cursor.execute(
                    f"""
                    SELECT seller_id, MAX(seller_name) as seller_name,
                           COUNT(*) as total,
                           SUM(CASE WHEN is_eligible = FALSE THEN 1 ELSE 0 END) as rejected
                    FROM invoice_logs
                    {where}
                    GROUP BY seller_id
                    """,
                    tuple(params),
                )

                if label == "prior":
                    prior_data: Dict[str, dict] = {}
                    for row in cursor.fetchall():
                        prior_data[row["seller_id"]] = {
                            "total": row["total"],
                            "rejected": row["rejected"],
                            "seller_name": row["seller_name"],
                        }
                else:
                    recent_data: Dict[str, dict] = {}
                    for row in cursor.fetchall():
                        recent_data[row["seller_id"]] = {
                            "total": row["total"],
                            "rejected": row["rejected"],
                            "seller_name": row["seller_name"],
                        }

            # Also get per-rule breakdown for each window
            prior_rules: Dict[Tuple[str, str], int] = {}
            recent_rules: Dict[Tuple[str, str], int] = {}

            for label, window_start, window_end, target in [
                ("prior", prior_start, recent_start, prior_rules),
                ("recent", recent_start, now, recent_rules),
            ]:
                conditions = [
                    "purchase_date >= ?",
                    "purchase_date < ?",
                    "is_eligible = FALSE",
                    "rejection_rules IS NOT NULL",
                ]
                params = [window_start.date().isoformat(), window_end.date().isoformat()]
                if filters.seller_id:
                    conditions.append("seller_id = ?")
                    params.append(filters.seller_id)
                if filters.debtor_id:
                    conditions.append("debtor_id = ?")
                    params.append(filters.debtor_id)
                if filters.programa:
                    conditions.append("programa = ?")
                    params.append(filters.programa)
                if filters.insurer_id:
                    conditions.append("insurer_id = ?")
                    params.append(filters.insurer_id)

                where = "WHERE " + " AND ".join(conditions)
                cursor.execute(
                    f"""
                    SELECT seller_id, rejection_rules
                    FROM invoice_logs
                    {where}
                    """,
                    tuple(params),
                )
                for row in cursor.fetchall():
                    rules = json.loads(row["rejection_rules"])
                    for rule in rules:
                        key = (row["seller_id"], rule)
                        target[key] = target.get(key, 0) + 1

        # Compare per (seller_id, rule_code) rejection rates between windows.
        # The rate for a rule in a window = rule_count / total_invoices.
        all_seller_rule_pairs: set = set()
        for key in list(prior_rules.keys()) + list(recent_rules.keys()):
            all_seller_rule_pairs.add(key)

        for sid, rule in all_seller_rule_pairs:
            p = prior_data.get(sid, {"total": 0, "rejected": 0, "seller_name": None})
            r = recent_data.get(sid, {"total": 0, "rejected": 0, "seller_name": None})

            if p["total"] < 3 or r["total"] < 3:
                continue

            pr_count = prior_rules.get((sid, rule), 0)
            rc_count = recent_rules.get((sid, rule), 0)
            prior_rate = pr_count / p["total"]
            recent_rate = rc_count / r["total"]
            delta = recent_rate - prior_rate

            if delta < 0.15:
                continue

            severity = Severity.HIGH if delta > 0.30 else Severity.MEDIUM
            seller_label = r.get("seller_name") or p.get("seller_name") or sid
            desc = _rule_description(rule)

            alerts.append(
                PatternAlert(
                    pattern_type=PatternType.TRENDING_INCREASE,
                    severity=severity,
                    seller_id=sid,
                    seller_name=r.get("seller_name") or p.get("seller_name"),
                    debtor_id=None,
                    debtor_name=None,
                    rule_code=rule,
                    rule_description=desc,
                    title=(
                        f"Rule {rule} non-eligibility rate increasing for "
                        f"{seller_label}"
                    ),
                    description=(
                        f"Rule {rule} ({desc}) non-eligibility rate for seller "
                        f"'{seller_label}' increased from {prior_rate:.0%} to "
                        f"{recent_rate:.0%} (+{delta:.0%} over "
                        f"{lookback_days} days)."
                    ),
                    recommendation=_rule_recommendation(rule),
                    metrics={
                        "prior_rate": round(prior_rate, 4),
                        "recent_rate": round(recent_rate, 4),
                        "delta_pp": round(delta * 100, 1),
                        "prior_total": p["total"],
                        "recent_total": r["total"],
                        "prior_rule_count": pr_count,
                        "recent_rule_count": rc_count,
                    },
                    time_window_days=lookback_days,
                )
            )
        return alerts

    # ------------------------------------------------------------------
    # Pattern Detection: 2c - Repeat Offenders
    # ------------------------------------------------------------------

    def _detect_repeat_offenders(
        self,
        filters: "PatternFilters",
        cutoff: datetime,
    ) -> List[PatternAlert]:
        """
        Detect debtors rejected in most batches they appear in.

        A "batch" is a distinct purchase_date (truncated to day).
        Flags if debtor rejected in >70% of batches (minimum min_invoices).
        """
        alerts: List[PatternAlert] = []

        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)

            conditions, params = self._build_filter_conditions(
                filters, cutoff, require_debtor=True
            )

            where = "WHERE " + " AND ".join(conditions)
            date_expr = sql_date("purchase_date")

            cursor.execute(
                f"""
                SELECT seller_id, MAX(seller_name) as seller_name,
                       debtor_id, MAX(debtor_name) as debtor_name,
                       {date_expr} as batch_date,
                       COUNT(*) as total_in_batch,
                       SUM(CASE WHEN is_eligible = FALSE THEN 1 ELSE 0 END) as rejected_in_batch
                FROM invoice_logs
                {where}
                GROUP BY seller_id, debtor_id, {date_expr}
                """,
                tuple(params),
            )

            # Accumulate per (seller_id, debtor_id)
            accum: Dict[Tuple[str, str], dict] = {}
            for row in cursor.fetchall():
                key = (row["seller_id"], row["debtor_id"])
                if key not in accum:
                    accum[key] = {
                        "seller_name": row["seller_name"],
                        "debtor_name": row["debtor_name"],
                        "batch_count": 0,
                        "batches_rejected": 0,
                    }
                acc = accum[key]
                acc["batch_count"] += 1
                if row["rejected_in_batch"] > 0:
                    acc["batches_rejected"] += 1
                # Keep latest names
                if row["seller_name"]:
                    acc["seller_name"] = row["seller_name"]
                if row["debtor_name"]:
                    acc["debtor_name"] = row["debtor_name"]

        for (sid, did), acc in accum.items():
            batch_count = acc["batch_count"]
            batches_rejected = acc["batches_rejected"]

            if batch_count < max(filters.min_invoices, 3):
                continue

            rejection_batch_rate = batches_rejected / batch_count
            if rejection_batch_rate < 0.7:
                continue

            severity = (
                Severity.HIGH
                if rejection_batch_rate >= 1.0 and batch_count >= 5
                else Severity.MEDIUM
            )

            debtor_label = acc["debtor_name"] or did
            seller_label = acc["seller_name"] or sid

            alerts.append(
                PatternAlert(
                    pattern_type=PatternType.REPEAT_OFFENDER,
                    severity=severity,
                    seller_id=sid,
                    seller_name=acc["seller_name"],
                    debtor_id=did,
                    debtor_name=acc["debtor_name"],
                    rule_code=None,
                    rule_description=None,
                    title=f"{debtor_label} non-eligible in {batches_rejected}/{batch_count} batches",
                    description=(
                        f"Debtor '{debtor_label}' (via Seller '{seller_label}') "
                        f"has had non-eligible invoices in {batches_rejected} out of "
                        f"{batch_count} processing batches ({rejection_batch_rate:.0%}). "
                        f"This debtor is a persistent source of non-eligible invoices."
                    ),
                    recommendation=(
                        "Review this debtor's invoice submission process end-to-end. "
                        "Consider whether their commercial terms need renegotiation, "
                        "or if there is a systematic data quality issue."
                    ),
                    metrics={
                        "batch_count": batch_count,
                        "batches_rejected": batches_rejected,
                        "batch_rejection_rate": round(rejection_batch_rate, 4),
                    },
                )
            )
        return alerts

    # ------------------------------------------------------------------
    # Pattern Detection: 2d - Rule Concentration
    # ------------------------------------------------------------------

    def _detect_rule_concentration(
        self,
        filters: "PatternFilters",
        cutoff: datetime,
    ) -> List[PatternAlert]:
        """
        Detect if a single rule dominates all rejections globally.

        Flags if one rule accounts for >50% of all rejections.
        This can indicate a systemic configuration issue.
        """
        alerts: List[PatternAlert] = []

        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)

            conditions, params = self._build_filter_conditions(
                filters, cutoff,
                require_noneligible=True,
                require_rejection_rules=True,
            )

            where = "WHERE " + " AND ".join(conditions)

            # Count rejections per rule globally.
            # total_rejected_invoices = number of rejected invoices (for
            # minimum sample-size gating).
            # total_rule_triggers = sum of per-rule counts (an invoice that
            # fails two rules contributes 2 triggers). This is the correct
            # denominator when computing each rule's share.
            rule_counts: Dict[str, int] = {}
            total_rejected_invoices = 0
            cursor.execute(
                f"""
                SELECT rejection_rules
                FROM invoice_logs
                {where}
                """,
                tuple(params),
            )
            for row in cursor.fetchall():
                rules = json.loads(row["rejection_rules"])
                total_rejected_invoices += 1
                for rule in rules:
                    rule_counts[rule] = rule_counts.get(rule, 0) + 1

        if total_rejected_invoices < 5:
            return alerts

        total_rule_triggers = sum(rule_counts.values())
        for rule, count in rule_counts.items():
            share = count / total_rule_triggers
            if share < 0.50:
                continue

            severity = Severity.HIGH if share > 0.70 else Severity.MEDIUM
            desc = _rule_description(rule)
            scope_parts = []
            if filters.seller_id:
                scope_parts.append(f"Seller {filters.seller_id}")
            if filters.debtor_id:
                scope_parts.append(f"Debtor {filters.debtor_id}")
            if filters.programa:
                scope_parts.append(f"Program {filters.programa}")
            if filters.insurer_id:
                scope_parts.append(f"Insurer {filters.insurer_id}")
            scope_label = ", ".join(scope_parts) if scope_parts else "all sellers"

            alerts.append(
                PatternAlert(
                    pattern_type=PatternType.RULE_CONCENTRATION,
                    severity=severity,
                    seller_id=filters.seller_id or "*",
                    seller_name=None,
                    debtor_id=None,
                    debtor_name=None,
                    rule_code=rule,
                    rule_description=desc,
                    title=f"Rule {rule} dominates non-eligibility ({share:.0%})",
                    description=(
                        f"Rule {rule} ({desc}) accounts for {share:.0%} of all "
                        f"rule triggers across {scope_label} "
                        f"({count}/{total_rule_triggers} triggers from "
                        f"{total_rejected_invoices} non-eligible invoices). "
                        f"This may indicate a systemic configuration issue or "
                        f"a widespread pattern in invoice submissions."
                    ),
                    recommendation=(
                        f"Review whether the {rule} threshold is correctly configured. "
                        + _rule_recommendation(rule)
                    ),
                    metrics={
                        "rule_count": count,
                        "total_rule_triggers": total_rule_triggers,
                        "total_rejected_invoices": total_rejected_invoices,
                        "share": round(share, 4),
                        "all_rule_counts": rule_counts,
                    },
                )
            )
        return alerts

    # ------------------------------------------------------------------
    # Pattern Detection: 2e - Amount at Risk
    # ------------------------------------------------------------------

    def _detect_amount_at_risk(
        self,
        filters: "PatternFilters",
        cutoff: datetime,
    ) -> List[PatternAlert]:
        """
        Detect patterns with high financial impact.

        Sums rejected invoice amounts per (seller_id, debtor_id) and flags
        those exceeding a threshold. Only flags patterns not already covered
        by chronic_rule_failure (to avoid duplication).
        """
        alerts: List[PatternAlert] = []

        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)

            conditions, params = self._build_filter_conditions(
                filters, cutoff,
                require_noneligible=True,
                require_debtor=True,
                require_amount=True,
            )

            where = "WHERE " + " AND ".join(conditions)

            cursor.execute(
                f"""
                SELECT seller_id, MAX(seller_name) as seller_name,
                       debtor_id, MAX(debtor_name) as debtor_name,
                       COUNT(*) as rejected_count,
                       MIN(purchase_date) as first_seen,
                       MAX(purchase_date) as last_seen
                FROM invoice_logs
                {where}
                GROUP BY seller_id, debtor_id
                HAVING COUNT(*) >= ?
                """,
                tuple(params) + (filters.min_invoices,),
            )

            candidates: List[dict] = []
            for row in cursor.fetchall():
                candidates.append(
                    {
                        "seller_id": row["seller_id"],
                        "seller_name": row["seller_name"],
                        "debtor_id": row["debtor_id"],
                        "debtor_name": row["debtor_name"],
                        "rejected_count": row["rejected_count"],
                        "first_seen": row["first_seen"],
                        "last_seen": row["last_seen"],
                    }
                )

            # Now sum amounts per (seller, debtor)
            cursor.execute(
                f"""
                SELECT seller_id, debtor_id, amount
                FROM invoice_logs
                {where}
                """,
                tuple(params),
            )

            amount_sums: Dict[Tuple[str, str], Decimal] = {}
            for row in cursor.fetchall():
                key = (row["seller_id"], row["debtor_id"])
                amt = parse_decimal(row["amount"])
                if amt is not None and amt > 0:
                    amount_sums[key] = amount_sums.get(key, Decimal("0")) + amt

        if not amount_sums:
            return alerts

        # Sort by amount descending and flag top entries
        sorted_pairs = sorted(amount_sums.items(), key=lambda x: x[1], reverse=True)

        # Use percentile-based threshold: flag top 20% or amounts > median * 2
        amounts = [v for _, v in sorted_pairs]
        if len(amounts) >= 3:
            median_idx = len(amounts) // 2
            threshold = amounts[median_idx] * 2
        else:
            threshold = Decimal("0")  # Flag all if very few entries

        candidate_map = {(c["seller_id"], c["debtor_id"]): c for c in candidates}

        for (sid, did), total_amount in sorted_pairs:
            if total_amount < threshold and len(amounts) >= 3:
                continue

            info = candidate_map.get((sid, did))
            if not info:
                continue

            debtor_label = info["debtor_name"] or did
            seller_label = info["seller_name"] or sid

            severity = Severity.HIGH if total_amount >= threshold * Decimal("1.5") else Severity.MEDIUM

            alerts.append(
                PatternAlert(
                    pattern_type=PatternType.AMOUNT_AT_RISK,
                    severity=severity,
                    seller_id=sid,
                    seller_name=info["seller_name"],
                    debtor_id=did,
                    debtor_name=info["debtor_name"],
                    rule_code=None,
                    rule_description=None,
                    title=f"{debtor_label}: {total_amount:,.2f} at risk",
                    description=(
                        f"Debtor '{debtor_label}' (via Seller '{seller_label}') has "
                        f"{info['rejected_count']} non-eligible invoices totaling "
                        f"{total_amount:,.2f} in the analysis window. "
                        f"This represents significant financial exposure from "
                        f"ineligible invoices."
                    ),
                    recommendation=(
                        "Prioritize resolving this debtor's eligibility issues. "
                        "The financial impact justifies a dedicated review of "
                        "their invoice submission process and commercial terms."
                    ),
                    metrics={
                        "total_amount_rejected": str(total_amount),
                        "rejected_invoice_count": info["rejected_count"],
                    },
                    first_seen=parse_datetime(info["first_seen"]),
                    last_seen=parse_datetime(info["last_seen"]),
                )
            )
        return alerts

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _window_stats(
        self, filters: "PatternFilters", cutoff: datetime
    ) -> Tuple[Optional[datetime], Optional[datetime], int, int]:
        """Get basic stats for the analysis window."""
        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)

            conditions, params = self._build_filter_conditions(filters, cutoff)

            where = "WHERE " + " AND ".join(conditions)

            cursor.execute(
                f"""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN is_eligible = TRUE THEN 1 ELSE 0 END) as eligible,
                       MIN(purchase_date) as first_date,
                       MAX(purchase_date) as last_date
                FROM invoice_logs
                {where}
                """,
                tuple(params),
            )
            row = cursor.fetchone()
            if row is None or row["total"] == 0:
                return None, None, 0, 0

            first = parse_datetime(row["first_date"])
            last = parse_datetime(row["last_date"])
            return first, last, row["total"], row["eligible"] or 0

    @staticmethod
    def _bucket_date(d: date, granularity: str) -> date:
        """Round a date down to the start of its bucket."""
        if granularity == "day":
            return d
        elif granularity == "week":
            return d - timedelta(days=d.weekday())  # Monday
        elif granularity == "month":
            return d.replace(day=1)
        elif granularity == "quarter":
            quarter_month = ((d.month - 1) // 3) * 3 + 1
            return d.replace(month=quarter_month, day=1)
        return d

    @staticmethod
    def _bucket_end(bucket_start: date, granularity: str) -> date:
        """Get the end date of a bucket (exclusive, but we store inclusive)."""
        if granularity == "day":
            return bucket_start
        elif granularity == "week":
            return bucket_start + timedelta(days=6)
        elif granularity == "month":
            if bucket_start.month == 12:
                return bucket_start.replace(year=bucket_start.year + 1, month=1, day=1) - timedelta(days=1)
            return bucket_start.replace(month=bucket_start.month + 1, day=1) - timedelta(days=1)
        elif granularity == "quarter":
            quarter_month = ((bucket_start.month - 1) // 3) * 3 + 1
            next_quarter_month = quarter_month + 3
            if next_quarter_month > 12:
                return date(bucket_start.year + 1, 1, 1) - timedelta(days=1)
            return date(bucket_start.year, next_quarter_month, 1) - timedelta(days=1)
        return bucket_start
