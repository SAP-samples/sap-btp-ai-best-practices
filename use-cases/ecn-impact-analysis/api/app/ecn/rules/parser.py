"""Deterministically parse and serialize the verified conditional-rule grammar."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum

from app.ecn.models import JsonValue
from app.ecn.rules.models import (
    AndExpression,
    EqualsExpression,
    IncludeAction,
    IncludeAssignment,
    NotExpression,
    OrExpression,
    ParsedRule,
    RuleExpression,
    RuleKind,
    RuleParseState,
)


class _TokenKind(StrEnum):
    """Enumerate internal lexical token kinds for the verified grammar."""

    SCALAR = "SCALAR"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    EQUALS = "EQUALS"
    PLUS = "PLUS"
    SLASH = "SLASH"
    COMMA = "COMMA"
    NOT = "NOT"
    ARROW = "ARROW"
    END = "END"


@dataclass(frozen=True)
class _Token:
    """Hold one decoded token and its source character offset."""

    kind: _TokenKind
    value: str
    offset: int


class _UnsupportedSyntax(ValueError):
    """Represent syntax outside the deliberately verified grammar."""

    def __init__(self, code: str, offset: int) -> None:
        """Initialize a stable reason code and source character offset."""
        super().__init__(code)
        self.code = code
        self.offset = offset


class _AmbiguousSyntax(_UnsupportedSyntax):
    """Represent syntax with multiple unverified interpretations."""


def stable_rule_id(artifact_id: str, source_row_number: int) -> str:
    """Return a stable SHA-256 identifier scoped to one artifact row.

    Args:
        artifact_id: Immutable source artifact identifier.
        source_row_number: One-based worksheet row number.

    Returns:
        A lowercase hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(
        f"{artifact_id}:{source_row_number}".encode()
    ).hexdigest()


def _is_standalone_slash(text: str, index: int) -> bool:
    """Recognize slash as OR only when whitespace delimits both sides."""
    return (
        text[index] == "/"
        and index > 0
        and index + 1 < len(text)
        and text[index - 1].isspace()
        and text[index + 1].isspace()
    )


def _lex(text: str) -> tuple[_Token, ...]:
    """Decode quoted and bare scalars without splitting embedded slash values."""
    tokens: list[_Token] = []
    index = 0
    punctuation = {
        "(": _TokenKind.LPAREN,
        ")": _TokenKind.RPAREN,
        "=": _TokenKind.EQUALS,
        "+": _TokenKind.PLUS,
        ",": _TokenKind.COMMA,
    }
    while index < len(text):
        if text[index].isspace():
            index += 1
            continue
        if text.startswith("-->", index):
            tokens.append(_Token(_TokenKind.ARROW, "-->", index))
            index += 3
            continue
        if text[index] in punctuation:
            tokens.append(_Token(punctuation[text[index]], text[index], index))
            index += 1
            continue
        if _is_standalone_slash(text, index):
            tokens.append(_Token(_TokenKind.SLASH, "/", index))
            index += 1
            continue
        if text[index] == "-" and text[index + 1 :].lstrip().startswith("("):
            tokens.append(_Token(_TokenKind.NOT, "-", index))
            index += 1
            continue
        # The source export sometimes represents one quoted scalar with paired
        # double quotes rather than JSON's single quote delimiters.
        if text.startswith('""', index):
            end = text.find('""', index + 2)
            if end < 0:
                raise _UnsupportedSyntax("UNTERMINATED_DOUBLED_QUOTE", index)
            tokens.append(_Token(_TokenKind.SCALAR, text[index + 2 : end], index))
            index = end + 2
            continue
        if text[index] == '"':
            try:
                value, consumed = json.JSONDecoder().raw_decode(text[index:])
            except json.JSONDecodeError as exc:
                raise _UnsupportedSyntax(
                    "INVALID_QUOTED_SCALAR", index + exc.pos
                ) from exc
            if not isinstance(value, str):
                raise _UnsupportedSyntax("SCALAR_MUST_BE_TEXT", index)
            tokens.append(_Token(_TokenKind.SCALAR, value, index))
            index += consumed
            continue
        start = index
        while index < len(text):
            if (
                text.startswith("-->", index)
                or text[index].isspace()
                or text[index] in "()=+,"
                or _is_standalone_slash(text, index)
            ):
                break
            index += 1
        if index == start:
            raise _UnsupportedSyntax("UNSUPPORTED_TOKEN", index)
        tokens.append(_Token(_TokenKind.SCALAR, text[start:index], start))
    tokens.append(_Token(_TokenKind.END, "", len(text)))
    return tuple(tokens)


class _Parser:
    """Parse rule tokens recursively while rejecting guessed precedence."""

    def __init__(self, tokens: tuple[_Token, ...]) -> None:
        """Initialize the token stream at its first token."""
        self.tokens = tokens
        self.index = 0

    @property
    def current(self) -> _Token:
        """Return the token at the current parse cursor."""
        return self.tokens[self.index]

    def take(self, kind: _TokenKind) -> _Token:
        """Consume one expected token or raise a positioned parse error."""
        token = self.current
        if token.kind is not kind:
            raise _UnsupportedSyntax(f"EXPECTED_{kind.value}", token.offset)
        self.index += 1
        return token

    def parse(self) -> tuple[RuleExpression, IncludeAction]:
        """Parse a complete condition-arrow-Include rule."""
        condition = self.expression()
        self.take(_TokenKind.ARROW)
        return condition, self.include_action(require_end=True)

    def include_action(self, require_end: bool = True) -> IncludeAction:
        """Parse the verified Include action independently of its condition."""
        action_name = self.take(_TokenKind.SCALAR)
        if action_name.value != "Inclure":
            raise _UnsupportedSyntax("UNSUPPORTED_ACTION", action_name.offset)
        self.take(_TokenKind.LPAREN)
        assignments = [self.assignment()]
        while self.current.kind is _TokenKind.COMMA:
            self.take(_TokenKind.COMMA)
            assignments.append(self.assignment())
        self.take(_TokenKind.RPAREN)
        if require_end:
            self.take(_TokenKind.END)
        return IncludeAction(tuple(assignments))

    def expression(self) -> RuleExpression:
        """Parse one operator family at the current nesting level."""
        operands = [self.unary()]
        operator = self.current.kind
        if operator not in {_TokenKind.PLUS, _TokenKind.SLASH}:
            return operands[0]
        while self.current.kind is operator:
            self.take(operator)
            operands.append(self.unary())
        other = _TokenKind.SLASH if operator is _TokenKind.PLUS else _TokenKind.PLUS
        if self.current.kind is other:
            raise _AmbiguousSyntax("MIXED_OPERATOR_PRECEDENCE", self.current.offset)
        if operator is _TokenKind.PLUS:
            return AndExpression(tuple(operands))
        return OrExpression(tuple(operands))

    def unary(self) -> RuleExpression:
        """Parse negation, an explicit group, or an equality."""
        if self.current.kind is _TokenKind.NOT:
            self.take(_TokenKind.NOT)
            self.take(_TokenKind.LPAREN)
            operand = self.expression()
            self.take(_TokenKind.RPAREN)
            return NotExpression(operand)
        if self.current.kind is _TokenKind.LPAREN:
            self.take(_TokenKind.LPAREN)
            expression = self.expression()
            self.take(_TokenKind.RPAREN)
            return expression
        equality = self.assignment()
        return EqualsExpression(equality.characteristic, equality.value)

    def assignment(self) -> IncludeAssignment:
        """Parse one scalar-equals-scalar pair."""
        left = self.take(_TokenKind.SCALAR)
        self.take(_TokenKind.EQUALS)
        right = self.take(_TokenKind.SCALAR)
        return IncludeAssignment(left.value.strip(), right.value.strip())


def _render_expression(expression: RuleExpression) -> str:
    """Render one AST node with explicit grouping and JSON scalar quoting."""
    if isinstance(expression, EqualsExpression):
        return (
            f"{json.dumps(expression.left, ensure_ascii=False)} = "
            f"{json.dumps(expression.right, ensure_ascii=False)}"
        )
    if isinstance(expression, AndExpression):
        return "(" + " + ".join(
            _render_expression(item) for item in expression.operands
        ) + ")"
    if isinstance(expression, OrExpression):
        return "(" + " / ".join(
            _render_expression(item) for item in expression.operands
        ) + ")"
    rendered = _render_expression(expression.operand)
    if rendered.startswith("(") and rendered.endswith(")"):
        rendered = rendered[1:-1]
    return f"-({rendered})"


def canonicalize_rule(rule: ParsedRule) -> str:
    """Render a parsed rule that round-trips to equal condition and action ASTs.

    Args:
        rule: A successfully parsed conditional rule.

    Returns:
        Canonical text with explicit grouping and JSON-quoted scalars.

    Raises:
        ValueError: If the rule has no complete parsed AST.
    """
    if (
        rule.parse_state is not RuleParseState.PARSED
        or rule.condition is None
        or rule.action is None
    ):
        raise ValueError("Only parsed rules have a canonical representation")
    assignments = ", ".join(
        f"{json.dumps(item.characteristic, ensure_ascii=False)} = "
        f"{json.dumps(item.value, ensure_ascii=False)}"
        for item in rule.action.assignments
    )
    return f"{_render_expression(rule.condition)} --> Inclure ({assignments})"


def _contains_suspicious_or(expression: RuleExpression) -> bool:
    """Find any OR whose direct operands are all negations."""
    if isinstance(expression, OrExpression) and expression.operands and all(
        isinstance(item, NotExpression) for item in expression.operands
    ):
        return True
    if isinstance(expression, (AndExpression, OrExpression)):
        return any(_contains_suspicious_or(item) for item in expression.operands)
    if isinstance(expression, NotExpression):
        return _contains_suspicious_or(expression.operand)
    return False


def _parse_action_after_arrow(tokens: tuple[_Token, ...]) -> IncludeAction:
    """Recover a verified action from a condition-ambiguous rule only."""
    arrow_positions = [
        index for index, token in enumerate(tokens) if token.kind is _TokenKind.ARROW
    ]
    if len(arrow_positions) != 1:
        raise _UnsupportedSyntax("EXPECTED_SINGLE_ARROW", tokens[-1].offset)
    action_tokens = tokens[arrow_positions[0] + 1 :]
    return _Parser(action_tokens).include_action(require_end=True)


def parse_rule_definition(
    raw: str, rule_id: str, source_row_number: int | None = None
) -> ParsedRule:
    """Return a reversible parsed rule or a preserved positioned failure record.

    Args:
        raw: Rule definition exactly as exported by the source workbook.
        rule_id: Stable caller-supplied identifier for this rule.
        source_row_number: Optional one-based source worksheet row.

    Returns:
        A lossless rule record classified as parsed, ambiguous, or unsupported.
    """
    source_text = raw
    parse_text = raw.removeprefix("'")
    warnings = ["LEADING_EXPORT_APOSTROPHE"] if source_text != parse_text else []
    base: dict[str, object] = {
        "rule_id": rule_id,
        "source_row_number": source_row_number,
        "rule_number": None,
        "general_status": None,
        "change_status": None,
        "object_type_indicator": None,
        "merged_raw": None,
        "version": None,
        "lifecycle_state": None,
        "description": None,
        "modified_at_raw": None,
        "source_text": source_text,
        "parse_text": parse_text,
        "source_lexemes": (),
    }
    tokens: tuple[_Token, ...] = ()
    try:
        tokens = _lex(parse_text.strip())
        base["source_lexemes"] = tuple(
            {"kind": token.kind.value, "value": token.value, "offset": token.offset}
            for token in tokens
            if token.kind is not _TokenKind.END
        )
        condition, action = _Parser(tokens).parse()
    except _AmbiguousSyntax as exc:
        try:
            action = _parse_action_after_arrow(tokens)
        except _UnsupportedSyntax as action_exc:
            return ParsedRule(
                **base,
                parse_state=RuleParseState.UNSUPPORTED,
                condition=None,
                action=None,
                canonical_text=None,
                warnings=tuple(warnings),
                reason_code=action_exc.code,
                reason_offset=action_exc.offset,
                review_required=False,
                rule_kind=RuleKind.STANDARD,
            )
        kind = (
            RuleKind.LARGE_MULTI_ASSIGNMENT_ACTION
            if len(action.assignments) > 100
            else RuleKind.STANDARD
        )
        return ParsedRule(
            **base,
            parse_state=RuleParseState.AMBIGUOUS,
            condition=None,
            action=action,
            canonical_text=None,
            warnings=tuple(warnings),
            reason_code=exc.code,
            reason_offset=exc.offset,
            review_required=False,
            rule_kind=kind,
        )
    except _UnsupportedSyntax as exc:
        return ParsedRule(
            **base,
            parse_state=RuleParseState.UNSUPPORTED,
            condition=None,
            action=None,
            canonical_text=None,
            warnings=tuple(warnings),
            reason_code=exc.code,
            reason_offset=exc.offset,
            review_required=False,
            rule_kind=RuleKind.STANDARD,
        )
    suspicious = _contains_suspicious_or(condition)
    if suspicious:
        warnings.append("SUSPICIOUS_OR_OF_NEGATIONS")
    kind = (
        RuleKind.LARGE_MULTI_ASSIGNMENT_ACTION
        if len(action.assignments) > 100
        else RuleKind.STANDARD
    )
    parsed = ParsedRule(
        **base,
        parse_state=RuleParseState.PARSED,
        condition=condition,
        action=action,
        canonical_text=None,
        warnings=tuple(warnings),
        reason_code=None,
        reason_offset=None,
        review_required=suspicious,
        rule_kind=kind,
    )
    canonical = canonicalize_rule(parsed)
    reparsed = (
        parse_rule_definition(canonical, rule_id, source_row_number)
        if source_text != canonical
        else parsed
    )
    if reparsed.condition != condition or reparsed.action != action:
        return replace(
            parsed,
            parse_state=RuleParseState.UNSUPPORTED,
            condition=None,
            action=None,
            canonical_text=None,
            reason_code="SEMANTIC_ROUND_TRIP_FAILED",
            reason_offset=None,
        )
    return replace(parsed, canonical_text=canonical)


def expression_to_data(expression: RuleExpression) -> dict[str, JsonValue]:
    """Serialize one expression to an explicitly tagged JSON-compatible object.

    Args:
        expression: A supported conditional-rule expression node.

    Returns:
        A strict object tagged as ``EQUALS``, ``AND``, ``OR``, or ``NOT``.
    """
    if isinstance(expression, EqualsExpression):
        return {"type": "EQUALS", "left": expression.left, "right": expression.right}
    if isinstance(expression, AndExpression):
        return {
            "type": "AND",
            "operands": [expression_to_data(item) for item in expression.operands],
        }
    if isinstance(expression, OrExpression):
        return {
            "type": "OR",
            "operands": [expression_to_data(item) for item in expression.operands],
        }
    return {"type": "NOT", "operand": expression_to_data(expression.operand)}


def action_to_data(action: IncludeAction) -> dict[str, JsonValue]:
    """Serialize one Include action with an explicit reversible node tag.

    Args:
        action: Ordered Include assignments to serialize.

    Returns:
        A strict JSON-compatible ``INCLUDE`` node.
    """
    return {
        "type": "INCLUDE",
        "assignments": [
            {"characteristic": item.characteristic, "value": item.value}
            for item in action.assignments
        ],
    }


def expression_from_data(data: JsonValue) -> RuleExpression:
    """Decode one strict tagged expression from persisted JSON-compatible data.

    Args:
        data: A tagged mapping produced by :func:`expression_to_data`.

    Returns:
        The corresponding immutable expression node.

    Raises:
        ValueError: If the tag, keys, or nested values do not match the schema.
    """
    if not isinstance(data, Mapping) or not isinstance(data.get("type"), str):
        # The public codec contract reports every malformed persisted shape as
        # ValueError, including a missing/non-text tag.
        raise ValueError(  # noqa: TRY004
            "expression must be an object with a string type tag"
        )
    node_type = data["type"]
    if node_type == "EQUALS":
        if set(data) != {"type", "left", "right"} or not isinstance(
            data.get("left"), str
        ) or not isinstance(data.get("right"), str):
            raise ValueError("EQUALS expression has an invalid shape")
        return EqualsExpression(data["left"], data["right"])
    if node_type in {"AND", "OR"}:
        operands_raw = data.get("operands")
        if set(data) != {"type", "operands"} or not isinstance(operands_raw, list):
            raise ValueError(f"{node_type} expression has an invalid shape")
        operands = tuple(expression_from_data(item) for item in operands_raw)
        if len(operands) < 2:
            raise ValueError(f"{node_type} expression requires at least two operands")
        return AndExpression(operands) if node_type == "AND" else OrExpression(operands)
    if node_type == "NOT":
        if set(data) != {"type", "operand"}:
            raise ValueError("NOT expression has an invalid shape")
        return NotExpression(expression_from_data(data["operand"]))
    raise ValueError(f"unsupported expression type: {node_type}")


def action_from_data(data: JsonValue) -> IncludeAction:
    """Decode only the strict versioned Include action shape.

    Args:
        data: A tagged mapping produced by :func:`action_to_data`.

    Returns:
        An immutable ordered Include action.

    Raises:
        ValueError: If the tag, keys, or assignments do not match the schema.
    """
    if (
        not isinstance(data, Mapping)
        or set(data) != {"type", "assignments"}
        or data.get("type") != "INCLUDE"
    ):
        raise ValueError("Invalid normalized rule action")
    assignments = data["assignments"]
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("Invalid normalized rule action")
    if not all(
        isinstance(item, Mapping)
        and set(item) == {"characteristic", "value"}
        and isinstance(item["characteristic"], str)
        and isinstance(item["value"], str)
        for item in assignments
    ):
        raise ValueError("Invalid normalized rule action")
    return IncludeAction(
        tuple(
            IncludeAssignment(item["characteristic"], item["value"])
            for item in assignments
        )
    )
