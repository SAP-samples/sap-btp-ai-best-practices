"""Pure active-import cohort selection and peer metric aggregation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from app.models.benchmarking import (
    BenchmarkMetric,
    BenchmarkPeerSubmission,
)

MINIMUM_PEER_COMPANIES = 3
"""Minimum distinct exact-cohort companies required for peer disclosure."""

RELEASED_SUBMISSION_STATUS = "REL"
"""Normalized source status that denotes a released questionnaire."""


@dataclass(frozen=True)
class PeerScopeAggregate:
    """Arithmetic peer average, independent maximum, and sample size for a scope.

    Inputs:
        Aggregated score values for one overall, dimension, or topic scope.

    Outputs:
        Immutable statistics consumed by score response construction.
    """

    peer_average: float | None = None
    best_peer: float | None = None
    sample_size: int = 0


@dataclass(frozen=True)
class PeerBenchmarkAggregate:
    """Peer statistics at overall, dimension, and canonical question scopes.

    Inputs:
        Scope aggregates calculated from selected latest submissions.

    Outputs:
        Immutable lookup structure used to attach nested benchmark metrics.
    """

    overall: PeerScopeAggregate
    dimensions: dict[str, PeerScopeAggregate]
    topics: dict[str, PeerScopeAggregate]


def _latest_submission_key(submission: BenchmarkPeerSubmission) -> tuple[date, date, str]:
    """Return a deterministic ordering key for one released submission.

    Inputs:
        submission: Candidate source questionnaire metadata.

    Outputs:
        tuple[date, date, str]: Submission date, extraction date, and stable ID,
        with missing dates ordered before known dates.
    """

    return (
        submission.submission_date or date.min,
        submission.extraction_date or date.min,
        submission.questionnaire_id,
    )


def select_latest_released_submissions(
    submissions: list[BenchmarkPeerSubmission],
    *,
    customer_class: str,
    nace1: str,
    excluded_source_company_id: str | None,
) -> list[BenchmarkPeerSubmission]:
    """Select one latest released submission per exact-cohort peer company.

    Inputs:
        submissions: Active-import source submissions with cohort attributes.
        customer_class: Exact normalized profile class required by the cohort.
        nace1: Exact profile level-one NACE label required by the cohort.
        excluded_source_company_id: Optional assessed company identity to remove.

    Outputs:
        list[BenchmarkPeerSubmission]: Deterministically company-ordered latest
        released submissions after exact class/NACE filtering and exclusion.
    """

    latest_by_company: dict[str, BenchmarkPeerSubmission] = {}
    for submission in submissions:
        if submission.customer_class != customer_class or submission.nace1 != nace1:
            continue
        if (
            excluded_source_company_id is not None
            and submission.source_company_id == excluded_source_company_id
        ):
            continue
        if (submission.release_status or "").strip().upper() != RELEASED_SUBMISSION_STATUS:
            continue
        current = latest_by_company.get(submission.source_company_id)
        if current is None or _latest_submission_key(submission) > _latest_submission_key(
            current
        ):
            latest_by_company[submission.source_company_id] = submission
    return [latest_by_company[key] for key in sorted(latest_by_company)]


def _aggregate_values(values: list[float]) -> PeerScopeAggregate:
    """Return rounded arithmetic average, independent maximum, and count.

    Inputs:
        values: Calculated peer scores for one logical scope.

    Outputs:
        PeerScopeAggregate: Empty aggregate for no values, otherwise four-decimal
        average/maximum and the contributing submission count.
    """

    if not values:
        return PeerScopeAggregate()
    return PeerScopeAggregate(
        peer_average=round(sum(values) / len(values), 4),
        best_peer=round(max(values), 4),
        sample_size=len(values),
    )


def aggregate_peer_scope_scores(
    submissions: list[BenchmarkPeerSubmission],
) -> PeerBenchmarkAggregate:
    """Aggregate selected peer submissions independently at every score scope.

    Inputs:
        submissions: One latest released submission per distinct peer company.

    Outputs:
        PeerBenchmarkAggregate: Overall, per-dimension, and per-question averages,
        maxima, and sample sizes. Topic keys are canonical question IDs, never
        workbook section labels.
    """

    overall_values: list[float] = []
    dimension_values: dict[str, list[float]] = defaultdict(list)
    topic_values: dict[str, list[float]] = defaultdict(list)
    for submission in submissions:
        seen_scopes: set[tuple[str, str | None, str | None]] = set()
        for score in submission.scores:
            scope_key = (score.scope_type, score.dimension, score.question_id)
            if scope_key in seen_scopes:
                continue
            seen_scopes.add(scope_key)
            if score.scope_type == "overall":
                overall_values.append(score.calculated_score)
            elif score.scope_type == "dimension" and score.dimension:
                dimension_values[score.dimension].append(score.calculated_score)
            elif score.scope_type == "topic" and score.question_id:
                topic_values[score.question_id].append(score.calculated_score)
    return PeerBenchmarkAggregate(
        overall=_aggregate_values(overall_values),
        dimensions={
            key: _aggregate_values(values)
            for key, values in sorted(dimension_values.items())
        },
        topics={
            key: _aggregate_values(values)
            for key, values in sorted(topic_values.items())
        },
    )


def position_against_peers(company_score: float, peer_average: float | None) -> str:
    """Classify company score using inclusive 90%-110% peer-average bounds.

    Inputs:
        company_score: Current company score for one scope.
        peer_average: Optional arithmetic peer average for the same scope.

    Outputs:
        str: ``below_peers``, ``in_line_with_peers``, ``above_peers``, or
        ``unavailable``. For zero average, zero is in-line and positive is above.
    """

    if peer_average is None:
        return "unavailable"
    if peer_average == 0:
        return "in_line_with_peers" if company_score == 0 else "above_peers"
    if company_score < 0.9 * peer_average:
        return "below_peers"
    if company_score <= 1.1 * peer_average:
        return "in_line_with_peers"
    return "above_peers"


def rounded_relative_difference(
    company_score: float,
    peer_average: float,
) -> tuple[str, str] | None:
    """Return a rounded relative difference and direction from the peer average.

    Inputs:
        company_score: Current company score for one scope.
        peer_average: Non-null arithmetic peer average for the same scope.

    Outputs:
        tuple[str, str] | None: Display magnitude and ``above``/``below`` direction,
        or ``None`` when values match or the peer baseline is zero.
    """

    if peer_average == 0 or company_score == peer_average:
        return None
    difference = abs(company_score - peer_average) / peer_average * 100
    magnitude = "less_than_one" if difference < 1 else str(int(difference + 0.5))
    direction = "above" if company_score > peer_average else "below"
    return magnitude, direction


def _localized_commentary(
    positioning: str,
    language: str,
    company_score: float,
    peer_average: float | None,
) -> str:
    """Return deterministic English or Italian commentary with relative context.

    Inputs:
        positioning: Stable benchmark positioning status.
        language: Normalized ``en`` or ``it`` response language.
        company_score: Current company score for one scope.
        peer_average: Optional arithmetic peer average for the same scope.

    Outputs:
        str: Localized one-sentence commentary without raw score disclosure.
    """

    italian = language == "it"
    if positioning == "unavailable" or peer_average is None:
        return (
            "Confronto con i peer non disponibile."
            if italian
            else "Peer comparison unavailable."
        )
    if peer_average == 0:
        positions = (
            {
                "in_line_with_peers": "In linea con la media dei peer",
                "above_peers": "Sopra la media dei peer",
            }
            if italian
            else {
                "in_line_with_peers": "In line with the peer average",
                "above_peers": "Above the peer average",
            }
        )
        position = positions[positioning]
        suffix = (
            "la differenza percentuale relativa non è applicabile."
            if italian
            else "relative percentage is not applicable."
        )
        return f"{position}; {suffix}"
    difference = rounded_relative_difference(company_score, peer_average)
    if difference is None:
        return (
            "In linea con la media dei peer (nessuna differenza)."
            if italian
            else "In line with the peer average (no difference)."
        )
    magnitude, direction = difference
    if italian:
        percentage = "meno dell'1%" if magnitude == "less_than_one" else f"{magnitude}%"
        if positioning == "in_line_with_peers":
            direction_text = "al di sopra" if direction == "above" else "al di sotto"
            return (
                "In linea con la media dei peer "
                f"({percentage} {direction_text})."
            )
        position_text = "Sopra" if direction == "above" else "Sotto"
        return f"{position_text} la media dei peer del {percentage}."
    percentage = "less than 1%" if magnitude == "less_than_one" else f"{magnitude}%"
    if positioning == "in_line_with_peers":
        return f"In line with the peer average ({percentage} {direction})."
    position_text = "Above" if direction == "above" else "Below"
    return f"{position_text} the peer average by {percentage}."


def build_benchmark_metric(
    company_score: float,
    peer_scope: PeerScopeAggregate | None,
    *,
    available: bool,
    language: str,
) -> BenchmarkMetric:
    """Build one nested comparison metric with deterministic status/commentary.

    Inputs:
        company_score: Current score at the target scope.
        peer_scope: Optional aggregate for the matching imported peer scope.
        available: Whether the exact cohort satisfies the global disclosure rule.
        language: Normalized response language for commentary.

    Outputs:
        BenchmarkMetric: Available peer values or explicit unavailable nulls.
    """

    peer_average = peer_scope.peer_average if available and peer_scope else None
    best_peer = peer_scope.best_peer if available and peer_scope else None
    sample_size = peer_scope.sample_size if available and peer_scope else 0
    positioning = position_against_peers(company_score, peer_average)
    return BenchmarkMetric(
        company_score=round(float(company_score), 4),
        peer_average=peer_average,
        best_peer=best_peer,
        sample_size=sample_size,
        positioning=positioning,
        commentary_key=f"benchmark.{positioning}",
        commentary=_localized_commentary(
            positioning,
            language,
            company_score,
            peer_average,
        ),
    )
