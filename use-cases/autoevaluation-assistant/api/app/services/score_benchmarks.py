"""Generate deterministic mock score benchmarks for peer comparisons."""

from hashlib import sha1

from app.models.assessment import AssessmentDimension, AssessmentQuestion
from app.models.scoring import ScoreBenchmarkRow
from app.services.customer_class_scope import load_customer_class_scope

SECTOR_BENCHMARK_CLASS = "__sector__"

_SECTOR_BASELINES = {
    "energy": 57.0,
    "financial services": 61.0,
    "healthcare": 55.5,
    "manufacturing": 54.5,
    "public sector": 50.0,
    "retail": 53.0,
    "technology": 62.0,
    "utilities": 56.5,
}


def _stable_fraction(*parts: object) -> float:
    """Return a deterministic fraction from arbitrary benchmark scope parts.

    Inputs:
        parts: Scope identifiers such as class, sector, dimension, or question.

    Outputs:
        float: Number in the inclusive lower, exclusive upper range ``[0, 1)``.
    """
    scope = "|".join(str(part or "") for part in parts)
    digest = sha1(scope.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _stable_offset(*parts: object, spread: float) -> float:
    """Return a deterministic positive or negative score offset.

    Inputs:
        parts: Scope identifiers used as the stable hash seed.
        spread: Maximum distance between the lowest and highest offset.

    Outputs:
        float: Offset centered around zero.
    """
    return (_stable_fraction(*parts) - 0.5) * spread


def _bounded_score(value: float) -> float:
    """Clamp and round a mock benchmark score to the supported score range.

    Inputs:
        value: Candidate mock benchmark score.

    Outputs:
        float: One-decimal score between 20 and 92.
    """
    return round(min(92.0, max(20.0, value)), 1)


def _customer_class_rank(customer_class: str) -> int:
    """Return the configured company class rank.

    Inputs:
        customer_class: Normalized class identifier such as ``class_4``.

    Outputs:
        int: Configured rank, defaulting to the largest known class.
    """
    classes = load_customer_class_scope()["classes"]
    default_rank = max(int(details["rank"]) for details in classes.values())
    return int(classes.get(customer_class, {}).get("rank", default_rank))


def _size_base_score(customer_class: str) -> float:
    """Return the overall mock score baseline for same-size peers.

    Inputs:
        customer_class: Normalized company class.

    Outputs:
        float: Baseline score for peer companies in the same class.
    """
    return 42.0 + (_customer_class_rank(customer_class) * 4.4)


def _sector_base_score(sector: str) -> float:
    """Return the overall mock score baseline for same-sector peers.

    Inputs:
        sector: Operating sector requested by the score route.

    Outputs:
        float: Baseline score for companies in the same sector.
    """
    normalized_sector = sector.strip().lower()
    if normalized_sector in _SECTOR_BASELINES:
        return _SECTOR_BASELINES[normalized_sector]
    return 54.0 + _stable_offset(normalized_sector, spread=10.0)


def _size_sample_size(customer_class: str) -> int:
    """Return a deterministic mock sample size for same-size peers.

    Inputs:
        customer_class: Normalized company class.

    Outputs:
        int: Mock observation count used by report positioning.
    """
    return 72 + (_customer_class_rank(customer_class) * 8)


def _sector_sample_size(sector: str) -> int:
    """Return a deterministic mock sample size for same-sector peers.

    Inputs:
        sector: Operating sector requested by the score route.

    Outputs:
        int: Mock observation count used by report positioning.
    """
    return 95 + int(_stable_fraction(sector, "sample-size") * 45)


def _scoped_benchmark_score(base_score: float, peer_group: str, *scope_parts: object) -> float:
    """Return a deterministic benchmark score for a peer group and scope.

    Inputs:
        base_score: Overall benchmark score for the peer group.
        peer_group: Human-readable peer group label used as hash salt.
        scope_parts: Optional dimension or question scope values.

    Outputs:
        float: One-decimal mock benchmark score for the requested scope.
    """
    if not scope_parts:
        return _bounded_score(base_score)
    return _bounded_score(
        base_score + _stable_offset(peer_group, *scope_parts, spread=9.0)
    )


def generate_mock_score_benchmarks(
    customer_class: str,
    sector: str | None,
    dimensions: list[AssessmentDimension],
    questions: list[AssessmentQuestion],
) -> list[ScoreBenchmarkRow]:
    """Return deterministic same-size and same-sector benchmark rows.

    Inputs:
        customer_class: Normalized company class used for same-size peers.
        sector: Optional operating sector used for same-sector peers.
        dimensions: Framework dimensions available for scoring.
        questions: Framework questions available for scoring.

    Outputs:
        list[ScoreBenchmarkRow]: Benchmark rows for overall, dimension, and
        question-level comparisons.
    """
    normalized_sector = sector.strip() if sector else None
    rows: list[ScoreBenchmarkRow] = []

    size_base = _size_base_score(customer_class)
    size_sample_size = _size_sample_size(customer_class)
    rows.append(
        ScoreBenchmarkRow(
            customer_class=customer_class,
            sector=None,
            benchmark_score=_scoped_benchmark_score(size_base, "same-size"),
            sample_size=size_sample_size,
        )
    )
    for dimension in dimensions:
        rows.append(
            ScoreBenchmarkRow(
                customer_class=customer_class,
                sector=None,
                dimension=dimension.dimension,
                benchmark_score=_scoped_benchmark_score(
                    size_base,
                    "same-size",
                    "dimension",
                    dimension.dimension,
                ),
                sample_size=size_sample_size,
            )
        )
    for question in questions:
        rows.append(
            ScoreBenchmarkRow(
                customer_class=customer_class,
                sector=None,
                question_id=question.question_id,
                benchmark_score=_scoped_benchmark_score(
                    size_base,
                    "same-size",
                    "question",
                    question.dimension,
                    question.question_id,
                ),
                sample_size=size_sample_size,
            )
        )

    if not normalized_sector:
        return rows

    sector_base = _sector_base_score(normalized_sector)
    sector_sample_size = _sector_sample_size(normalized_sector)
    rows.append(
        ScoreBenchmarkRow(
            customer_class=SECTOR_BENCHMARK_CLASS,
            sector=normalized_sector,
            benchmark_score=_scoped_benchmark_score(sector_base, "same-sector"),
            sample_size=sector_sample_size,
        )
    )
    for dimension in dimensions:
        rows.append(
            ScoreBenchmarkRow(
                customer_class=SECTOR_BENCHMARK_CLASS,
                sector=normalized_sector,
                dimension=dimension.dimension,
                benchmark_score=_scoped_benchmark_score(
                    sector_base,
                    "same-sector",
                    "dimension",
                    dimension.dimension,
                ),
                sample_size=sector_sample_size,
            )
        )
    for question in questions:
        rows.append(
            ScoreBenchmarkRow(
                customer_class=SECTOR_BENCHMARK_CLASS,
                sector=normalized_sector,
                question_id=question.question_id,
                benchmark_score=_scoped_benchmark_score(
                    sector_base,
                    "same-sector",
                    "question",
                    question.dimension,
                    question.question_id,
                ),
                sample_size=sector_sample_size,
            )
        )
    return rows
