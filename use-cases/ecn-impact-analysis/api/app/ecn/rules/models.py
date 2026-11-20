"""Immutable domain records for conditional-rule parsing and evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TypeAlias

from app.ecn.models import JsonValue, SourceArtifact


class RuleParseState(StrEnum):
    """Describe the deterministic outcome of parsing one rule definition."""

    PARSED = "PARSED"
    UNSUPPORTED = "UNSUPPORTED"
    AMBIGUOUS = "AMBIGUOUS"


class TruthValue(StrEnum):
    """Represent a three-valued result when source assignments may be incomplete."""

    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class EqualsExpression:
    """Hold one exact characteristic/value comparison."""

    left: str
    right: str


@dataclass(frozen=True)
class AndExpression:
    """Hold a conjunction while preserving source operand order."""

    operands: tuple[RuleExpression, ...]


@dataclass(frozen=True)
class OrExpression:
    """Hold a disjunction while preserving source operand order."""

    operands: tuple[RuleExpression, ...]


@dataclass(frozen=True)
class NotExpression:
    """Hold unary negation over one explicitly grouped expression."""

    operand: RuleExpression


RuleExpression: TypeAlias = (
    EqualsExpression | AndExpression | OrExpression | NotExpression
)
"""Recursive expression nodes admitted by the verified rule grammar."""


@dataclass(frozen=True)
class IncludeAssignment:
    """Hold one exact characteristic/value assignment in an Include action."""

    characteristic: str
    value: str


@dataclass(frozen=True)
class IncludeAction:
    """Hold ordered assignments from one ``Inclure`` action."""

    assignments: tuple[IncludeAssignment, ...]


class RuleKind(StrEnum):
    """Classify source rule size without assigning business meaning."""

    STANDARD = "STANDARD"
    LARGE_MULTI_ASSIGNMENT_ACTION = "LARGE_MULTI_ASSIGNMENT_ACTION"


class RuleEvaluationState(StrEnum):
    """Describe why an evaluator did or did not evaluate a parsed rule."""

    EVALUATED = "EVALUATED"
    NOT_EVALUATED_PARSE_STATE = "NOT_EVALUATED_PARSE_STATE"
    NOT_EVALUATED_ACTIVATION_POLICY_MISSING = (
        "NOT_EVALUATED_ACTIVATION_POLICY_MISSING"
    )
    NOT_EVALUATED_LIFECYCLE_STATE = "NOT_EVALUATED_LIFECYCLE_STATE"
    NOT_EVALUATED_REVIEW_REQUIRED = "NOT_EVALUATED_REVIEW_REQUIRED"
    NOT_EVALUATED_LARGE_MULTI_ASSIGNMENT = (
        "NOT_EVALUATED_LARGE_MULTI_ASSIGNMENT"
    )


@dataclass(frozen=True)
class ParsedRule:
    """Preserve a source rule and its reversible AST when parsing succeeds."""

    rule_id: str
    source_row_number: int | None
    rule_number: str | None
    general_status: str | None
    change_status: str | None
    object_type_indicator: str | None
    merged_raw: JsonValue
    version: str | None
    lifecycle_state: str | None
    description: str | None
    modified_at_raw: str | None
    source_text: str
    parse_text: str
    source_lexemes: tuple[dict[str, JsonValue], ...]
    parse_state: RuleParseState
    condition: RuleExpression | None
    action: IncludeAction | None
    canonical_text: str | None
    warnings: tuple[str, ...]
    reason_code: str | None
    reason_offset: int | None
    review_required: bool
    rule_kind: RuleKind


@dataclass(frozen=True)
class RuleEvaluation:
    """Hold one activation-aware evaluation result with explicit provenance."""

    rule_id: str
    rule_number: str | None
    source_row_number: int | None
    parse_state: RuleParseState
    parse_reason_code: str | None
    parse_reason_offset: int | None
    state: RuleEvaluationState
    truth_value: TruthValue | None
    applied_assignments: tuple[IncludeAssignment, ...]
    missing_variables: tuple[str, ...]
    incomplete_variables: tuple[str, ...]
    warnings: tuple[str, ...]
    activation_policy_version: str | None
    dependency_path: tuple[str, ...]


@dataclass(frozen=True)
class RuleArtifactMetadata:
    """Hold safe source metadata supplied by rule setup code."""

    artifact_id: str
    file_name: str
    media_type: str
    created_at: datetime


@dataclass(frozen=True)
class RuleSourceRow:
    """Hold one validated worksheet row before definition parsing."""

    source_row_number: int
    values: Mapping[str, JsonValue]


@dataclass(frozen=True)
class RuleWorkbookNormalization:
    """Hold a pure normalized workbook result before optional publication."""

    artifact: SourceArtifact
    rules: tuple[ParsedRule, ...]
    parse_counts: Mapping[RuleParseState, int]
    warning_counts: Mapping[str, int]


@dataclass(frozen=True)
class RuleImportResult:
    """Summarize import results; omit the artifact for validate-only execution."""

    artifact: SourceArtifact | None
    validate_only: bool
    total_rows: int
    parse_counts: Mapping[RuleParseState, int]
    warning_counts: Mapping[str, int]


@dataclass(frozen=True)
class FactSet:
    """Hold known values for one characteristic plus source completeness."""

    values: frozenset[str]
    complete: bool


@dataclass(frozen=True)
class RuleActivationPolicy:
    """Define versioned business policy that admits stored rules for execution."""

    version: str
    admitted_lifecycle_states: frozenset[str]
    allow_large_multi_assignment_actions: bool = False
    allow_review_required: bool = False
