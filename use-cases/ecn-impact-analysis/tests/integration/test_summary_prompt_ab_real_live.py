"""Opt-in read-only HANA A/B evaluation of ECN346863 executive-summary wording.

Run only after accepting six real-data SAP Generative AI Hub calls:
    RUN_REAL_SUMMARY_AB_TEST=1 PYTHONPATH=api python -m pytest -q -s -m llm_live \
        tests/integration/test_summary_prompt_ab_real_live.py

The test reads one persisted ECN occurrence, its immutable evidence bundle, and
its pinned BOM/rule/reference datasets from HANA. It never writes HANA, creates
an analysis attempt, changes the active prompt, or modifies a stored summary.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from dotenv import load_dotenv
from sqlalchemy import text

from app.ecn.llm.provider import ProviderCallStage, ProviderCallStatus, create_impact_provider
from app.ecn.llm.staged import (
    EXECUTIVE_SUMMARY_PATTERN,
    FactorStageResult,
    ReadinessStageResult,
    SummaryNarrativeStyle,
    SummaryStageResult,
    build_stage_prompt,
)
from app.ecn.llm.synthesis import build_synthesis_projection
from app.ecn.models import OUTPUT_SCHEMA_VERSION, PROMPT_VERSION
from app.ecn.services.snapshots import SnapshotService
from app.ecn.settings import load_ecn_settings
from app.persistence.artifact_repository import ArtifactRepository
from app.persistence.ecn_repository import EcnRepository
from app.persistence.hana import create_hana_engine
from app.persistence.reference_dataset_repository import ReferenceDatasetRepository
from app.persistence.records import PinnedGenerationConfig
from app.persistence.table_contracts import PersistenceTableNames, quote_identifier

pytestmark = [
    pytest.mark.llm_live,
    pytest.mark.skipif(
        os.getenv("RUN_REAL_SUMMARY_AB_TEST") != "1",
        reason="set RUN_REAL_SUMMARY_AB_TEST=1 to accept six real-data Gen AI Hub calls",
    ),
]

ECN_NUMBER = "ECN346863"


class ReadOnlySnapshotService(SnapshotService):
    """Build a normal snapshot while suppressing the standard plant-label cache write."""

    def _store_plant_labels(self, ecn_row_id, snapshot):
        """Keep this test read-only; the production cache UPDATE is deliberately disabled."""
        return None


def _lower_keys(row: object) -> dict[str, object]:
    """Normalize HANA driver's lower-case result mappings once at the boundary."""
    return {str(key).lower(): value for key, value in dict(row).items()}


def _latest_ecn_row_id(engine, tables: PersistenceTableNames) -> str:
    """Return the latest persisted occurrence of ECN346863 without reading source content."""
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                f'''SELECT E."ECN_ROW_ID"
                    FROM {quote_identifier(tables.ecn)} E
                    JOIN {quote_identifier(tables.upload)} U ON U."UPLOAD_ID" = E."UPLOAD_ID"
                    WHERE E."ECN_NUMBER" = :ecn_number
                    ORDER BY U."PUBLICATION_ORDER" DESC, E."SOURCE_ORDINAL" ASC'''
            ),
            {"ecn_number": ECN_NUMBER},
        ).mappings().all()
    if not rows:
        pytest.skip(f"{ECN_NUMBER} is not present in the configured HANA schema")
    return str(_lower_keys(rows[0])["ecn_row_id"])


def _real_projection():
    """Read ECN346863 and its pinned HANA inputs into one bounded, in-memory projection."""
    tables = PersistenceTableNames()
    engine = create_hana_engine()
    try:
        ecn_row_id = _latest_ecn_row_id(engine, tables)
        ecns = EcnRepository(engine, tables)
        workspace = ReadOnlySnapshotService(
            ecns,
            ArtifactRepository(engine, tables),
            ReferenceDatasetRepository(engine, tables),
            engine,
            tables,
        )
        evidence = ecns.get_evidence_bundle(ecn_row_id)
        if evidence is None:
            pytest.skip(f"{ECN_NUMBER} has no persisted evidence bundle")
        snapshot = workspace.get(ecn_row_id)
        assert snapshot["ecn_number"] == ECN_NUMBER
        assert snapshot["bom_changes"]["lines"]
        assert snapshot["datasets"]
        return build_synthesis_projection(evidence, snapshot=snapshot), snapshot
    finally:
        engine.dispose()


async def _generate(provider, config, prompt, stage: ProviderCallStage):
    """Call Gen AI Hub once and reject incomplete transport or schema responses."""
    result = await provider.generate(prompt, config, stage)
    assert result.status is ProviderCallStatus.RECEIVED
    assert result.response_status == "completed"
    assert not result.refusal_present
    assert result.raw_json
    return prompt.schema.model_validate_json(result.raw_json), result.usage


def _allowed_citations(projection) -> set[str]:
    """Return every evidence handle allowed for factor-stage citations."""
    return set(projection.header_citation_ids) | {
        citation_id for record in projection.records for citation_id in record.citation_ids
    } | {fact.evidence_id for fact in projection.discovered_facts}


def _print_summary(label: str, result: SummaryStageResult, usage) -> None:
    """Emit reviewer-readable summaries and token usage without logging raw source data."""
    print(f"\n=== {label} ===")
    print(f"Headline: {result.change_brief}")
    print(result.sections.executive_summary.assessment_markdown)
    print(f"Usage: input={usage.input_tokens}, output={usage.output_tokens}, total={usage.total_tokens}")


@pytest.mark.asyncio
async def test_ecn346863_description_first_summary_is_identifier_free_and_grounded():
    """Compare current and candidate summary wording over exactly one HANA-backed projection."""
    load_dotenv(Path(__file__).resolve().parents[2] / "api" / ".env")
    settings = load_ecn_settings()
    projection, snapshot = _real_projection()
    config = PinnedGenerationConfig(
        prompt_version=PROMPT_VERSION,
        output_schema_version=OUTPUT_SCHEMA_VERSION,
        model_name=settings.llm_model_name,
        request_timeout_seconds=settings.llm_request_timeout_seconds,
        max_prompt_evidence_items=settings.max_prompt_evidence_items,
        max_prompt_characters=settings.max_prompt_characters,
    )
    provider = create_impact_provider(settings)
    try:
        completed = {}
        permitted = _allowed_citations(projection)
        for name, schema in (
            ("MANUFACTURING", FactorStageResult),
            ("PROCUREMENT", FactorStageResult),
            ("CONFIGURATION", FactorStageResult),
            ("DATA_READINESS", ReadinessStageResult),
        ):
            prompt = build_stage_prompt(projection, name, snapshot=snapshot)
            result, _usage = await _generate(provider, config, prompt, ProviderCallStage(name))
            assert isinstance(result, schema)
            if isinstance(result, FactorStageResult):
                assert set(result.factor.evidence_ids) <= permitted
            if isinstance(result, ReadinessStageResult):
                assert {item.code for item in result.descriptions} == {
                    question["code"] for question in snapshot["questions"]
                }
            completed[name] = result.model_dump(mode="json")

        variants = {}
        for label, style in (
            ("A — current identifier-led summary", SummaryNarrativeStyle.IDENTIFIER_LED),
            ("B — description-first candidate", SummaryNarrativeStyle.DESCRIPTION_FIRST),
        ):
            prompt = build_stage_prompt(projection, "SUMMARY", completed=completed, summary_narrative_style=style)
            result, usage = await _generate(provider, config, prompt, ProviderCallStage.SUMMARY)
            assert isinstance(result, SummaryStageResult)
            executive_summary = result.sections.executive_summary.assessment_markdown
            assert re.fullmatch(EXECUTIVE_SUMMARY_PATTERN, executive_summary)
            _print_summary(label, result, usage)
            variants[label] = result
    finally:
        await provider.close()

    candidate = variants["B — description-first candidate"]
    candidate_text = candidate.sections.executive_summary.assessment_markdown
    forbidden_identifier = re.compile(r"\b(?:AC[WX][A-Z0-9]+|Q900VV\d+|Q957VV\d+|\d{6,}|\d+M\d+)\b")
    assert not forbidden_identifier.search(candidate_text), candidate_text
    assert re.search(r"sticker|decal|assembly|content", candidate_text, re.IGNORECASE)
    assert "Plant A" in candidate_text or "PL10" in candidate_text
    assert "2026-05-19" in candidate_text or "19 May 2026" in candidate_text
