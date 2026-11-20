"""Opt-in acceptance checks for ignored development source files.

Run:
    DEVELOPMENT_FIXTURE_DIR=<repo-root>/data \
      PYTHONPATH=api python -m pytest -q -m development_fixture

These tests read files under repository ``data/`` only when they exist. Application
logic must never open these paths; browser uploads and HANA artifacts are the runtime
sources.
"""

from __future__ import annotations

import os
import multiprocessing
import resource
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from app.ecn.errors import SourceValidationError
from app.ecn.evidence.builder import build_ecn_evidence
from app.ecn.evidence.matching import (
    BomIndexes,
    EnrichedMatchStrength,
    build_bom_indexes,
    match_ecn_to_bom,
)
from app.ecn.intake.json_parser import parse_ecn_json
from app.ecn.intake.workbook_parser import parse_bom_workbook
from app.ecn.models import (
    BomRowClassification,
    EvidenceContext,
    EvidenceKind,
    NormalizedEcn,
    ParsedBomWorkbook,
)
from app.ecn.rules.importer import normalize_rule_workbook
from app.ecn.rules.models import (
    AndExpression,
    NotExpression,
    OrExpression,
    RuleArtifactMetadata,
    RuleExpression,
    RuleParseState,
)

pytestmark = pytest.mark.development_fixture
DATA = Path(
    os.environ.get(
        "DEVELOPMENT_FIXTURE_DIR",
        str(Path(__file__).resolve().parents[2] / "data"),
    )
).resolve()
JSON_PATH = DATA / "ECN.json"
MALFORMED_JSON_PATH = DATA / "ECN copy.json"
BOM_PATH = DATA / "EBOM MBOM_Q900V_04.09.2026.xlsx"
RULE_CANDIDATES = sorted(DATA.glob("R*glesconditionnelleslocales*.xlsx"))
RULE_PATH = RULE_CANDIDATES[0] if RULE_CANDIDATES else None
ECN_ARTIFACT_ID = "00000000-0000-0000-0000-000000000101"
BOM_ARTIFACT_ID = "00000000-0000-0000-0000-000000000102"
MALFORMED_ARTIFACT_ID = "00000000-0000-0000-0000-000000000103"
WORKBOOK_ONLY_ECN_ROW_ID = "00000000-0000-0000-0000-000000000104"
FIXTURE_UPLOAD_ID = "00000000-0000-0000-0000-000000000105"
RULE_ARTIFACT_ID = "00000000-0000-0000-0000-000000000106"
NormalizedSources = tuple[tuple[NormalizedEcn, ...], ParsedBomWorkbook, BomIndexes]


def _required(path: Path | None) -> Path:
    """Return an available fixture path or skip the opt-in assertion cleanly."""
    if path is None or not path.exists():
        pytest.skip("ignored development fixture is not available")
    return path


@pytest.fixture(scope="module")
def normalized_sources() -> NormalizedSources:
    """Parse the corrected JSON and three BOM layers once for source assertions."""
    parsed_ecn = parse_ecn_json(_required(JSON_PATH).read_bytes(), ECN_ARTIFACT_ID)
    workbook = parse_bom_workbook(
        _required(BOM_PATH).read_bytes(),
        BOM_ARTIFACT_ID,
        maximum_rows=100_000,
        maximum_cells=3_000_000,
    )
    return parsed_ecn.ecns, workbook, build_bom_indexes(workbook.rows)


def _contains_not(expression: RuleExpression | None) -> bool:
    """Return whether a rule AST contains any negation node."""
    if expression is None:
        return False
    if isinstance(expression, NotExpression):
        return True
    if isinstance(expression, (AndExpression, OrExpression)):
        return any(_contains_not(item) for item in expression.operands)
    return False


def test_corrected_and_original_malformed_json(
    normalized_sources: NormalizedSources,
) -> None:
    """Accept corrected JSON and retain the original syntax-location negative case."""
    ecns, _, _ = normalized_sources
    assert len(ecns) == 35
    assert all(ecn.ecn_number != "ECN345274" for ecn in ecns)
    with pytest.raises(SourceValidationError) as caught:
        parse_ecn_json(
            _required(MALFORMED_JSON_PATH).read_bytes(), MALFORMED_ARTIFACT_ID
        )
    assert getattr(caught.value, "code", None) == "ECN_JSON_SYNTAX_INVALID"
    details = getattr(caught.value, "details", {}) or {}
    assert details.get("line") == 318_816
    assert details.get("column") == 1


def test_workbook_only_ecn_345274_has_exact_document_and_material_crosswalk(
    normalized_sources: NormalizedSources,
) -> None:
    """Preserve a workbook crosswalk without claiming the ECN exists in JSON."""
    ecns, _, indexes = normalized_sources
    workbook_only = replace(
        ecns[0],
        ecn_row_id=WORKBOOK_ONLY_ECN_ROW_ID,
        ecn_number="ECN345274",
        affected_objects=(),
        resulting_objects=(),
    )
    result = match_ecn_to_bom(workbook_only, indexes)
    documents = [
        row
        for row in result.raw_direct
        if row.row_classification is BomRowClassification.CONFIRMED_DOCUMENT
    ]
    materials = [
        row
        for row in result.raw_direct
        if row.row_classification
        is BomRowClassification.MATERIAL_LIKE_OBSERVED
    ]
    confirmed = [
        item
        for item in result.enriched_matches
        if item.enriched_strength
        is EnrichedMatchStrength.COMPONENT_AND_ECN_CONFIRMED
    ]
    assert len(result.raw_direct) == 34
    assert len(documents) == 30
    assert len(materials) == 4
    assert {row.item_or_position for row in materials} == {"150", "160"}
    assert {row.component_number for row in materials} == {
        "QKX463881A",
        "QKX891416A",
    }
    assert {row.parent_material_number for row in materials} == {
        "Q923VV2526401",
        "Q923VV2526404",
    }
    assert {item.occurrence.source_row_number for item in confirmed} == {
        9_311,
        9_312,
        9_484,
        9_485,
    }
    document_ids = {row.bom_row_id for row in documents}
    assert document_ids.isdisjoint(
        {seed.source.bom_row_id for seed in result.eligible_structural_seeds}
    )


def test_current_json_ecn_126772_is_the_golden_end_to_end_join(
    normalized_sources: NormalizedSources,
) -> None:
    """Assert the golden joins and distinguish target from ECN-wide MBOM scope."""
    ecns, workbook, indexes = normalized_sources
    ecn = ecns[3]
    assert ecn.source_ordinal == 3
    assert ecn.ecn_number == "ECN126772"
    resulting = [
        item for item in ecn.resulting_objects if item.number == "495452W1"
    ]
    assert len(resulting) == 1
    assert resulting[0].source_path == (
        "$.ImplementationPlan[0].ResultingLinks[381].ResultingObjects[0]"
    )
    assert resulting[0].technical_id == "OR:wt.part.WTPart:15865146236"
    assert resulting[0].revision == "6"
    assert resulting[0].version == "6.3 (Design)"
    result = match_ecn_to_bom(ecn, indexes)
    assert {row.source_row_number for row in result.raw_direct} == {2_605}
    raw_leaf = result.raw_direct[0]
    by_id = {row.bom_row_id: row for row in workbook.rows}
    chain = [raw_leaf]
    while chain[-1].parent_bom_row_id is not None:
        chain.append(by_id[chain[-1].parent_bom_row_id])
    assert [row.source_row_number for row in reversed(chain)] == [
        2_517,
        2_518,
        2_604,
        2_605,
    ]
    confirmed = [
        item
        for item in result.enriched_matches
        if item.enriched_strength
        is EnrichedMatchStrength.COMPONENT_AND_ECN_CONFIRMED
    ]
    assert {item.occurrence.source_row_number for item in confirmed} == {1_392}
    assert not [
        item for item in result.mbom_presence if item.seed.number == "495452W1"
    ]
    assert len(result.mbom_presence) == 213
    assert len({item.occurrence.bom_row_id for item in result.mbom_presence}) == 213
    evidence = build_ecn_evidence(
        ecn,
        result,
        (),
        EvidenceContext(
            upload_id=FIXTURE_UPLOAD_ID,
            ecn_artifact_id=ECN_ARTIFACT_ID,
            bom_artifact_id=BOM_ARTIFACT_ID,
            rule_artifact_id=None,
            parser_version="ecn-source-v1",
            deterministic_logic_version="ecn-evidence-v1",
            rule_evaluator_version="ecn-rules-v1",
            analysis_as_of_date=None,
        ),
    )
    assert any(
        item.kind is EvidenceKind.MISSING_INPUT
        and item.values.get("code") == "MBOM_COMPONENT_NO_MATCH"
        and item.values.get("component_number") == "495452W1"
        for item in evidence.sections.uncertainties_and_missing_evidence
    )


def test_year_1900_valid_from_is_preserved_as_migration_evidence(
    normalized_sources: NormalizedSources,
) -> None:
    """Keep legacy migration placeholders without substituting another date."""
    _, workbook, _ = normalized_sources
    rows = [
        row
        for row in workbook.rows
        if row.valid_from_date and row.valid_from_date.year == 1900
    ]
    assert rows
    assert all(row.valid_from_raw for row in rows)
    assert all(row.is_migration_date for row in rows)


def test_rule_workbook_matches_observed_counts_without_claiming_executability(
) -> None:
    """Lock Rosetta-stone source facts while activation policy stays absent."""
    path = _required(RULE_PATH)
    normalization = normalize_rule_workbook(
        path.read_bytes(),
        RuleArtifactMetadata(
            artifact_id=RULE_ARTIFACT_ID,
            file_name=path.name,
            media_type=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            created_at=datetime(2026, 9, 14, tzinfo=timezone.utc),
        ),
    )
    rules = normalization.rules
    assert len(rules) == 4_575
    assert normalization.parse_counts == {
        RuleParseState.PARSED: 4_575,
        RuleParseState.UNSUPPORTED: 0,
        RuleParseState.AMBIGUOUS: 0,
    }
    assert sum(len(rule.action.assignments) for rule in rules if rule.action) == 6_081
    assert sum(_contains_not(rule.condition) for rule in rules) == 117
    assert sum(
        "SUSPICIOUS_OR_OF_NEGATIONS" in rule.warnings for rule in rules
    ) == 23
    assert sum("LEADING_EXPORT_APOSTROPHE" in rule.warnings for rule in rules) == 2
    large = {
        rule.rule_number: len(rule.action.assignments)
        for rule in rules
        if rule.action is not None and len(rule.action.assignments) > 100
    }
    assert large == {
        "RL0000105591": 486,
        "RL0000105552": 524,
        "RL0000105532": 499,
    }


def _parse_bom_and_report_rss(path: str, queue: object) -> None:
    """Parse the workbook in a child and return role count plus peak RSS."""
    parsed = parse_bom_workbook(
        Path(path).read_bytes(),
        "00000000-0000-4000-8000-000000000107",
        100_000,
        3_000_000,
    )
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if sys.platform == "darwin" else rss * 1024
    queue.put((len(parsed.sheet_roles), rss_bytes))  # type: ignore[attr-defined]


def test_three_workbook_roles_stay_below_peak_memory_cap() -> None:
    """Keep the accepted three-role workbook below the 850 MiB peak cap."""
    path = _required(BOM_PATH)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_parse_bom_and_report_rss,
        args=(str(path), queue),
    )
    process.start()
    process.join(timeout=60)
    assert process.exitcode == 0
    role_count, rss_bytes = queue.get(timeout=5)
    assert role_count == 3
    assert rss_bytes < 850 * 1024 * 1024
