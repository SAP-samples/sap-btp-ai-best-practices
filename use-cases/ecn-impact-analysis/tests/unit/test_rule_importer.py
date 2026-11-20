"""Pure normalization and publication tests for conditional-rule workbooks.

Run these tests in isolation with:

    PYTHONPATH=api python -m pytest -q tests/unit/test_rule_importer.py
"""

import json
from datetime import datetime, timezone
from io import BytesIO

import pytest
from app.ecn.errors import PersistenceStateError, SourceValidationError
from app.ecn.models import ArtifactKind, SourceArtifact
from app.ecn.rules.importer import (
    import_rule_workbook,
    load_parsed_rules,
    normalize_rule_workbook,
)
from app.ecn.rules.models import RuleArtifactMetadata, RuleParseState
from openpyxl import Workbook, load_workbook

HEADERS = (
    "Statut général",
    "Statut de modification",
    "Indicateur de type d'objet",
    "Numéro",
    "Fusionné",
    "Version",
    "Etat",
    "Définition de règle",
    "Description",
    "Modifié le",
)


def _workbook(*definitions: str) -> bytes:
    """Build a small rule workbook with the production header contract."""
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Default"
    sheet.append(HEADERS)
    for index, definition in enumerate(definitions, start=1):
        sheet.append(
            (
                "Valid",
                "Released",
                "Rule",
                f"RL{index:010d}",
                None,
                "1",
                "App For Series",
                definition,
                f"Rule {index}",
                "04.09.2026",
            )
        )
    output = BytesIO()
    workbook.save(output)
    return output.getvalue()


class _ArtifactRepository:
    """Record immutable rule-artifact publications for unit tests."""

    def __init__(self) -> None:
        """Initialize an empty ordered publication list."""
        self.published: list[SourceArtifact] = []

    def publish_rule_artifact(self, artifact: SourceArtifact) -> SourceArtifact:
        """Append and return the supplied artifact."""
        self.published.append(artifact)
        return artifact


def _metadata() -> RuleArtifactMetadata:
    """Return deterministic source metadata for hash and identity assertions."""
    return RuleArtifactMetadata(
        artifact_id="00000000-0000-0000-0000-000000000111",
        file_name="rules.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        created_at=datetime(2026, 9, 14, tzinfo=timezone.utc),
    )


def test_normalization_preserves_source_metadata_and_reloads_equal_rules() -> None:
    """Round-trip persisted normalized JSON without reparsing source semantics."""
    content = _workbook(
        "A = B --> Inclure (S = V)",
        "A = B + C = D / E = F --> Inclure (S = V)",
    )
    result = normalize_rule_workbook(content, _metadata())

    assert result.artifact.artifact_kind is ArtifactKind.CONDITIONAL_RULE_WORKBOOK
    assert result.artifact.content_bytes == content
    assert result.rules[0].rule_number == "RL0000000001"
    assert result.rules[0].source_row_number == 2
    assert result.parse_counts == {
        RuleParseState.PARSED: 1,
        RuleParseState.UNSUPPORTED: 0,
        RuleParseState.AMBIGUOUS: 1,
    }
    assert result.artifact.normalized_content is not None
    payload = json.loads(result.artifact.normalized_content)
    assert payload["rules"][0]["condition"]["type"] == "EQUALS"
    assert payload["rules"][0]["action"]["type"] == "INCLUDE"
    assert load_parsed_rules(result.artifact.normalized_content) == result.rules


def test_validate_only_writes_nothing_but_returns_all_counts() -> None:
    """Make dry validation independent of HANA mutation."""
    repository = _ArtifactRepository()
    result = import_rule_workbook(
        _workbook("A = B --> Inclure (S = V)"), _metadata(), repository, True
    )
    assert result.validate_only is True
    assert result.artifact is None
    assert result.total_rows == 1
    assert repository.published == []


def test_valid_import_appends_exactly_one_immutable_artifact() -> None:
    """Publish the original bytes and normalized representation together."""
    repository = _ArtifactRepository()
    result = import_rule_workbook(
        _workbook("A = B --> Inclure (S = V)"), _metadata(), repository, False
    )
    assert result.artifact is repository.published[0]
    assert len(repository.published) == 1


@pytest.mark.parametrize("mutation", ["sheet", "header", "corrupt"])
def test_workbook_level_failure_never_publishes(mutation: str) -> None:
    """Reject a broken workbook contract atomically."""
    content = _workbook("A = B --> Inclure (S = V)")
    if mutation != "corrupt":
        workbook = load_workbook(BytesIO(content))
        sheet = workbook["Default"]
        if mutation == "sheet":
            sheet.title = "Other"
        else:
            sheet.cell(1, 8, "Other definition")
        output = BytesIO()
        workbook.save(output)
        content = output.getvalue()
    else:
        content = b"not-an-xlsx"
    repository = _ArtifactRepository()

    with pytest.raises(SourceValidationError) as caught:
        import_rule_workbook(content, _metadata(), repository, False)

    assert caught.value.code in {
        "RULE_WORKBOOK_INVALID",
        "RULE_SHEET_MISSING",
        "RULE_HEADER_MISSING",
    }
    assert repository.published == []


def test_stored_rule_loader_rejects_untagged_actions() -> None:
    """Prevent historical artifacts from acquiring action semantics on reload."""
    normalized = normalize_rule_workbook(
        _workbook("A = B --> Inclure (S = V)"), _metadata()
    ).artifact.normalized_content
    assert normalized is not None
    payload = json.loads(normalized)
    payload["rules"][0]["action"] = [{"characteristic": "S", "value": "V"}]

    with pytest.raises(PersistenceStateError, match="Stored rule action"):
        load_parsed_rules(json.dumps(payload))
