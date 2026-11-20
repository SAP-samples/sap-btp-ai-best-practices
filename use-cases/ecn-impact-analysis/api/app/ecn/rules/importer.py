"""Normalize rule workbooks and optionally publish one immutable HANA artifact."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from io import BytesIO

from openpyxl import load_workbook

from app.ecn.errors import PersistenceStateError, SourceValidationError
from app.ecn.models import ArtifactKind, JsonValue, ParseState, SourceArtifact
from app.ecn.rules.models import (
    ParsedRule,
    RuleArtifactMetadata,
    RuleImportResult,
    RuleKind,
    RuleParseState,
    RuleSourceRow,
    RuleWorkbookNormalization,
)
from app.ecn.rules.parser import (
    action_from_data,
    action_to_data,
    expression_from_data,
    expression_to_data,
    parse_rule_definition,
    stable_rule_id,
)
from app.persistence.artifact_repository import ArtifactRepository

RULE_HEADERS = (
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
"""Required headers on the single admitted ``Default`` worksheet."""

NORMALIZED_RULE_FORMAT = "ecn-rules-normalized-v1"
"""Version tag for the lossless normalized rule artifact payload."""

_RULE_RECORD_KEYS = {
    "rule_id",
    "source_row_number",
    "rule_number",
    "general_status",
    "change_status",
    "object_type_indicator",
    "merged_raw",
    "version",
    "lifecycle_state",
    "description",
    "modified_at_raw",
    "source_text",
    "parse_text",
    "source_lexemes",
    "parse_state",
    "condition",
    "action",
    "canonical_text",
    "warnings",
    "reason_code",
    "reason_offset",
    "review_required",
    "rule_kind",
    "semantic_round_trip",
}


def _optional_text(value: JsonValue) -> str | None:
    """Normalize optional metadata while preserving definition text separately."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_cell(value: object) -> JsonValue:
    """Convert worksheet scalars to a lossless JSON-safe representation."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()  # type: ignore[union-attr,no-any-return]
    return str(value)


def _read_rows(content: bytes) -> tuple[RuleSourceRow, ...]:
    """Validate the exact worksheet/header contract and return source rows.

    Args:
        content: Original XLSX bytes supplied by the operator.

    Returns:
        Non-empty source rows with their original one-based worksheet numbers.

    Raises:
        SourceValidationError: If the workbook, sheet, or headers are invalid.
    """
    workbook = None
    try:
        workbook = load_workbook(BytesIO(content), read_only=True, data_only=False)
        if "Default" not in workbook.sheetnames:
            raise SourceValidationError(
                "RULE_SHEET_MISSING", "The required Default worksheet is missing."
            )
        sheet = workbook["Default"]
        iterator = sheet.iter_rows(values_only=True)
        try:
            raw_headers = next(iterator)
        except StopIteration as exc:
            raise SourceValidationError(
                "RULE_HEADER_MISSING", "The rule worksheet is empty."
            ) from exc
        headers = tuple(
            str(item).strip() if item is not None else "" for item in raw_headers
        )
        missing = tuple(header for header in RULE_HEADERS if headers.count(header) != 1)
        if missing:
            raise SourceValidationError(
                "RULE_HEADER_MISSING",
                "Required conditional-rule headers are missing or duplicated.",
                details={"headers": list(missing)},
            )
        positions = {header: headers.index(header) for header in RULE_HEADERS}
        rows: list[RuleSourceRow] = []
        for source_row_number, row in enumerate(iterator, start=2):
            values = {
                header: _safe_cell(row[index] if index < len(row) else None)
                for header, index in positions.items()
            }
            if any(value is not None and value != "" for value in values.values()):
                rows.append(RuleSourceRow(source_row_number, values))
        return tuple(rows)
    except SourceValidationError:
        raise
    except Exception as exc:
        raise SourceValidationError(
            "RULE_WORKBOOK_INVALID",
            "The conditional-rule workbook is unreadable.",
        ) from exc
    finally:
        if workbook is not None:
            workbook.close()


def _rule_to_data(rule: ParsedRule) -> dict[str, JsonValue]:
    """Serialize every rule field and ordered AST/action into canonical data."""
    return {
        "rule_id": rule.rule_id,
        "source_row_number": rule.source_row_number,
        "rule_number": rule.rule_number,
        "general_status": rule.general_status,
        "change_status": rule.change_status,
        "object_type_indicator": rule.object_type_indicator,
        "merged_raw": rule.merged_raw,
        "version": rule.version,
        "lifecycle_state": rule.lifecycle_state,
        "description": rule.description,
        "modified_at_raw": rule.modified_at_raw,
        "source_text": rule.source_text,
        "parse_text": rule.parse_text,
        "source_lexemes": list(rule.source_lexemes),
        "parse_state": rule.parse_state.value,
        "condition": (
            expression_to_data(rule.condition) if rule.condition is not None else None
        ),
        "action": action_to_data(rule.action) if rule.action is not None else None,
        "canonical_text": rule.canonical_text,
        "warnings": list(rule.warnings),
        "reason_code": rule.reason_code,
        "reason_offset": rule.reason_offset,
        "review_required": rule.review_required,
        "rule_kind": rule.rule_kind.value,
        "semantic_round_trip": rule.parse_state is RuleParseState.PARSED,
    }


def _optional_stored_text(value: JsonValue, field_name: str) -> str | None:
    """Validate one nullable text field from normalized rule JSON."""
    if value is None or isinstance(value, str):
        return value
    raise PersistenceStateError(
        "RULE_ARTIFACT_INVALID",
        f"Stored rule field {field_name} is invalid.",
        status_code=500,
    )


def _rule_from_data(value: dict[str, JsonValue]) -> ParsedRule:
    """Reconstruct one strictly shaped stored rule without semantic reparsing."""
    if set(value) != _RULE_RECORD_KEYS:
        raise PersistenceStateError(
            "RULE_ARTIFACT_INVALID",
            "A stored rule has unexpected fields.",
            status_code=500,
        )
    if not isinstance(value["rule_id"], str) or not isinstance(
        value["source_text"], str
    ) or not isinstance(value["parse_text"], str):
        raise PersistenceStateError(
            "RULE_ARTIFACT_INVALID",
            "Stored rule identity/text is invalid.",
            status_code=500,
        )
    row = value["source_row_number"]
    offset = value["reason_offset"]
    if (row is not None and not isinstance(row, int)) or (
        offset is not None and not isinstance(offset, int)
    ):
        raise PersistenceStateError(
            "RULE_ARTIFACT_INVALID",
            "Stored rule coordinates are invalid.",
            status_code=500,
        )
    warnings = value["warnings"]
    lexemes = value["source_lexemes"]
    if not isinstance(warnings, list) or not all(
        isinstance(item, str) for item in warnings
    ):
        raise PersistenceStateError(
            "RULE_ARTIFACT_INVALID",
            "Stored rule warnings are invalid.",
            status_code=500,
        )
    if not isinstance(lexemes, list) or not all(
        isinstance(item, Mapping)
        and set(item) == {"kind", "value", "offset"}
        and isinstance(item["kind"], str)
        and isinstance(item["value"], str)
        and isinstance(item["offset"], int)
        and item["offset"] >= 0
        for item in lexemes
    ):
        raise PersistenceStateError(
            "RULE_ARTIFACT_INVALID",
            "Stored rule lexemes are invalid.",
            status_code=500,
        )
    condition_data = value["condition"]
    if condition_data is not None and not isinstance(condition_data, Mapping):
        raise PersistenceStateError(
            "RULE_ARTIFACT_INVALID",
            "Stored rule condition is invalid.",
            status_code=500,
        )
    try:
        condition = (
            None if condition_data is None else expression_from_data(condition_data)
        )
    except (TypeError, ValueError) as exc:
        raise PersistenceStateError(
            "RULE_ARTIFACT_INVALID",
            "Stored rule condition is invalid.",
            status_code=500,
        ) from exc
    action_data = value["action"]
    if action_data is None:
        action = None
    elif isinstance(action_data, Mapping):
        try:
            action = action_from_data(action_data)
        except (TypeError, ValueError) as exc:
            raise PersistenceStateError(
                "RULE_ARTIFACT_INVALID",
                "Stored rule action is invalid.",
                status_code=500,
            ) from exc
    else:
        raise PersistenceStateError(
            "RULE_ARTIFACT_INVALID",
            "Stored rule action is invalid.",
            status_code=500,
        )
    if not isinstance(value["review_required"], bool) or not isinstance(
        value["semantic_round_trip"], bool
    ):
        raise PersistenceStateError(
            "RULE_ARTIFACT_INVALID",
            "Stored rule flags are invalid.",
            status_code=500,
        )
    if not isinstance(value["parse_state"], str) or not isinstance(
        value["rule_kind"], str
    ):
        raise PersistenceStateError(
            "RULE_ARTIFACT_INVALID",
            "Stored rule enum values are invalid.",
            status_code=500,
        )
    try:
        parse_state = RuleParseState(value["parse_state"])
        rule_kind = RuleKind(value["rule_kind"])
    except ValueError as exc:
        raise PersistenceStateError(
            "RULE_ARTIFACT_INVALID",
            "Stored rule enum values are unknown.",
            status_code=500,
        ) from exc
    if value["semantic_round_trip"] is not (parse_state is RuleParseState.PARSED):
        raise PersistenceStateError(
            "RULE_ARTIFACT_INVALID",
            "Stored semantic round-trip flag is inconsistent.",
            status_code=500,
        )
    return ParsedRule(
        rule_id=value["rule_id"],
        source_row_number=row,
        rule_number=_optional_stored_text(value["rule_number"], "rule_number"),
        general_status=_optional_stored_text(
            value["general_status"], "general_status"
        ),
        change_status=_optional_stored_text(value["change_status"], "change_status"),
        object_type_indicator=_optional_stored_text(
            value["object_type_indicator"], "object_type_indicator"
        ),
        merged_raw=value["merged_raw"],
        version=_optional_stored_text(value["version"], "version"),
        lifecycle_state=_optional_stored_text(
            value["lifecycle_state"], "lifecycle_state"
        ),
        description=_optional_stored_text(value["description"], "description"),
        modified_at_raw=_optional_stored_text(
            value["modified_at_raw"], "modified_at_raw"
        ),
        source_text=value["source_text"],
        parse_text=value["parse_text"],
        source_lexemes=tuple(dict(item) for item in lexemes),
        parse_state=parse_state,
        condition=condition,
        action=action,
        canonical_text=_optional_stored_text(
            value["canonical_text"], "canonical_text"
        ),
        warnings=tuple(warnings),
        reason_code=_optional_stored_text(value["reason_code"], "reason_code"),
        reason_offset=offset,
        review_required=value["review_required"],
        rule_kind=rule_kind,
    )


def normalize_rule_workbook(
    content: bytes,
    metadata: RuleArtifactMetadata,
    row_progress: Callable[
        [Iterable[RuleSourceRow]], Iterable[RuleSourceRow]
    ] | None = None,
) -> RuleWorkbookNormalization:
    """Return a validated lossless rule artifact without writing to HANA.

    Args:
        content: Original workbook bytes.
        metadata: Safe immutable artifact identity and source metadata.
        row_progress: Optional CLI-only wrapper for progress reporting.

    Returns:
        The immutable artifact, parsed rule records, and deterministic counts.
    """
    source_rows = _read_rows(content)
    iterable = row_progress(source_rows) if row_progress is not None else source_rows
    rules: list[ParsedRule] = []
    for source in iterable:
        definition = source.values["Définition de règle"]
        raw_definition = (
            definition
            if isinstance(definition, str)
            else "" if definition is None else str(definition)
        )
        parsed = parse_rule_definition(
            raw_definition,
            stable_rule_id(metadata.artifact_id, source.source_row_number),
            source.source_row_number,
        )
        rules.append(
            replace(
                parsed,
                rule_number=_optional_text(source.values["Numéro"]),
                general_status=_optional_text(source.values["Statut général"]),
                change_status=_optional_text(
                    source.values["Statut de modification"]
                ),
                object_type_indicator=_optional_text(
                    source.values["Indicateur de type d'objet"]
                ),
                merged_raw=source.values["Fusionné"],
                version=_optional_text(source.values["Version"]),
                lifecycle_state=_optional_text(source.values["Etat"]),
                description=_optional_text(source.values["Description"]),
                modified_at_raw=_optional_text(source.values["Modifié le"]),
            )
        )
    parse_counts = Counter(rule.parse_state for rule in rules)
    warning_counts = Counter(warning for rule in rules for warning in rule.warnings)
    normalized = json.dumps(
        {
            "format_version": NORMALIZED_RULE_FORMAT,
            "rules": [_rule_to_data(rule) for rule in rules],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    artifact = SourceArtifact(
        artifact_id=metadata.artifact_id,
        artifact_kind=ArtifactKind.CONDITIONAL_RULE_WORKBOOK,
        file_name=metadata.file_name,
        media_type=metadata.media_type,
        byte_size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        content_bytes=content,
        normalized_content=normalized,
        parse_state=ParseState.VALID,
        parse_error_code=None,
        created_at=metadata.created_at,
    )
    complete_counts = {state: parse_counts.get(state, 0) for state in RuleParseState}
    return RuleWorkbookNormalization(
        artifact, tuple(rules), complete_counts, dict(warning_counts)
    )


def load_parsed_rules(normalized_content: str) -> tuple[ParsedRule, ...]:
    """Load strict stored AST records without reparsing source definitions.

    Args:
        normalized_content: Versioned canonical rule artifact JSON.

    Returns:
        Immutable rule records exactly as stored.

    Raises:
        PersistenceStateError: If the format or any stored record is invalid.
    """
    try:
        payload = json.loads(normalized_content)
    except json.JSONDecodeError as exc:
        raise PersistenceStateError(
            "RULE_ARTIFACT_INVALID",
            "The stored normalized rule payload is invalid.",
            status_code=500,
        ) from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"format_version", "rules"}
        or payload.get("format_version") != NORMALIZED_RULE_FORMAT
    ):
        raise PersistenceStateError(
            "RULE_ARTIFACT_VERSION_UNSUPPORTED",
            "The stored rule format is unsupported.",
            status_code=500,
        )
    records = payload.get("rules")
    if not isinstance(records, list) or not all(
        isinstance(item, dict) for item in records
    ):
        raise PersistenceStateError(
            "RULE_ARTIFACT_INVALID",
            "The stored normalized rule payload is invalid.",
            status_code=500,
        )
    return tuple(_rule_from_data(item) for item in records)


def import_rule_workbook(
    content: bytes,
    metadata: RuleArtifactMetadata,
    repository: ArtifactRepository,
    validate_only: bool,
    row_progress: Callable[
        [Iterable[RuleSourceRow]], Iterable[RuleSourceRow]
    ] | None = None,
) -> RuleImportResult:
    """Validate all rows and append one artifact unless validation-only is set.

    Args:
        content: Original rule workbook bytes.
        metadata: Safe immutable source metadata.
        repository: HANA artifact repository used only after full normalization.
        validate_only: When true, perform no repository mutation.
        row_progress: Optional CLI-only progress wrapper.

    Returns:
        Counts and the published artifact, or no artifact in validation mode.
    """
    normalization = normalize_rule_workbook(content, metadata, row_progress)
    published = (
        None
        if validate_only
        else repository.publish_rule_artifact(normalization.artifact)
    )
    return RuleImportResult(
        artifact=published,
        validate_only=validate_only,
        total_rows=len(normalization.rules),
        parse_counts=normalization.parse_counts,
        warning_counts=normalization.warning_counts,
    )
