"""Deterministic evaluation of a customer's documented rules on the raw extraction.

Each registered customer has a playbook whose ``anchors`` (stored in HANA,
authored from the payee's conversion-rule documents) contain ordered rules:

- ``customer_account_rules`` + ``default_customer_account``: which payee
  customer account a line belongs to (e.g. a customer column "Rg" equals "A" ->
  10050955) and who the payer is.
- ``company_code_rules`` + ``default_company_code``: the payee company code.
  Playbooks authored before these keys existed state it as a constant header
  mapping (``field_mappings``: ``iln_recipient`` / payee = Fixed value "CA01");
  that constant is used when neither key gives a value.
- ``deduction_rules`` (+ ``default_rule``): the reason code of a deduction line
  (e.g. invoice number ends with "UL" -> 316).

A rule is ``{priority, field, operator, value, <target>}``. ``field`` names a
column as the rule author saw it ("Column Rg", "Rg", "invoice_number",
"description", "company_code"). It is resolved, in order, against:

1. the raw Document AI columns of the line, by technical name or printed label
   (``raw_extraction.schema_fields``), then the raw header columns;
2. canonical line fields, then canonical header fields, then header values this
   engine already derived (so account rules may test ``company_code``);
3. a small alias map (``invoice_number`` -> ``invoice_reference``, ...);
4. the single raw column whose label ends with the rule's word(s) ("description"
   -> "Invoice Description"); several candidates leave the field unresolved.

Operators: ``equals``, ``starts_with``, ``ends_with``, ``contains``, ``regex``;
case-insensitive, whitespace-trimmed. The lowest ``priority`` that matches wins
(ties keep list order). Rules whose field cannot be resolved, or whose operator
is unknown, never match and are reported in ``warnings`` instead of silently
doing nothing; a missing extracted column is visible that way.

The engine only decides what the documents state. Lines no specific rule covers
are left for the deduction agent (see ``interpretation_service``).

Example:
    outcome = apply_rules(payload, raw_extraction, playbook.anchors)
    outcome["lines"][0]["customer_account"]      # "10050955"
    outcome["lines"][0]["deduction"]["reason_code"]  # "316" when a rule matched
"""
from __future__ import annotations

import copy
import re
from typing import Any

from .aggregation import _normalize_value
from .canonical import CANONICAL_SCHEMA

OPERATORS = ("equals", "starts_with", "ends_with", "contains", "regex")
# Rule-author vocabulary -> canonical field names (after normalize_label).
FIELD_ALIASES = {
    "invoice_number": "invoice_reference",
    "invoice_no": "invoice_reference",
    "your_invoice_no": "invoice_reference",
    "invoice": "invoice_reference",
    "description": "deduction_reason",
    "invoice_description": "deduction_reason",
    "gross": "gross_amount",
    "net": "net_amount",
    "discount": "discount_amount",
}
_PREFIXES = ("column_", "item_", "line_", "header_")
# Header mapping targets that name the receiving payee entity (the company code).
_COMPANY_CODE_TARGETS = {"company_code", "iln_recipient", "iln_recipient_payee", "payee"}
# Item-record vocabulary of the payee's conversion documents -> canonical line field.
# Only these targets are applied from ``field_mappings``; the rest (market name,
# advice number, header totals) are not canonical line fields.
DOCUMENTED_LINE_TARGETS = {
    "invoice_number": "invoice_reference", "invoice_no": "invoice_reference",
    "inv_number": "invoice_reference", "item_characteristic_inv_number": "invoice_reference",
    "invoice_reference": "invoice_reference",
    "invoice_amount": "gross_amount", "gross_amount": "gross_amount",
    "payment_amount": "net_amount", "net_amount": "net_amount",
    "cash_discount": "discount_amount", "discount_amount": "discount_amount",
    "invoice_date": "invoice_date", "inv_date": "invoice_date",
    "deduction_amount": "deduction_amount", "deduction_reason": "deduction_reason",
    "customer_document_reference": "customer_document_reference",
}
_LINE_DEFS = {f["name"]: f for f in CANONICAL_SCHEMA["lineItemFields"]}


def normalize_label(value: Any) -> str:
    """Normalize a field reference: lower snake case without "column"/"item"/"header" prefixes.

    Examples: "Column Rg" -> "rg"; "item.invoice_number" -> "invoice_number";
    "Your Invoice No" -> "your_invoice_no".
    """
    label = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    for prefix in _PREFIXES:
        if label.startswith(prefix) and len(label) > len(prefix):
            label = label[len(prefix):]
    return label


def _text(value: Any) -> str:
    """Comparable text of a cell: trimmed, single-spaced, case-folded."""
    return " ".join(str(value).split()).casefold()


def matches(operator: str, cell: Any, expected: Any) -> bool:
    """Return True when ``cell`` satisfies ``operator`` against ``expected``.

    Raises:
        ValueError: For an unknown operator or an invalid regular expression.
    """
    if cell is None or (isinstance(cell, str) and not cell.strip()):
        return False
    op = normalize_label(operator)
    if op == "regex":
        try:
            return re.search(str(expected), " ".join(str(cell).split()), re.IGNORECASE) is not None
        except re.error as exc:
            raise ValueError(f"invalid regex {expected!r}: {exc}") from exc
    left, right = _text(cell), _text(expected)
    if op == "equals":
        return left == right
    if op == "starts_with":
        return left.startswith(right)
    if op == "ends_with":
        return left.endswith(right)
    if op == "contains":
        return right in left
    raise ValueError(f"unknown operator {operator!r} (supported: {', '.join(OPERATORS)})")


class _Resolver:
    """Resolve rule field references against raw columns, canonical fields and derived values."""

    def __init__(self, raw_extraction: dict[str, Any] | None) -> None:
        raw = raw_extraction or {}
        self.raw_header: dict[str, Any] = raw.get("header") or {}
        self.raw_lines: list[dict[str, Any]] = raw.get("line_items") or []
        # normalized name/label -> (scope, technical field name)
        self.columns: dict[str, tuple[str, str]] = {}
        for item in raw.get("schema_fields") or []:
            scope = "header" if item.get("scope") == "header" else "line"
            for key in (item.get("name"), item.get("label")):
                if key:
                    self.columns.setdefault(normalize_label(key), (scope, str(item["name"])))

    def raw_row(self, index: int) -> dict[str, Any]:
        """Raw columns of line ``index`` ({} when the raw view is shorter)."""
        return self.raw_lines[index] if index < len(self.raw_lines) else {}

    def resolve(self, field: str, raw_row: dict[str, Any], line: dict[str, Any],
                header: dict[str, Any], derived: dict[str, Any]) -> tuple[bool, Any]:
        """Return (resolved, value) for a rule field; resolved=False means the field is unknown."""
        key = normalize_label(field)
        if key in self.columns:
            scope, name = self.columns[key]
            return True, (raw_row if scope == "line" else self.raw_header).get(name)
        for source in (raw_row, self.raw_header):
            for name, value in source.items():
                if normalize_label(name) == key:
                    return True, value
        canonical = FIELD_ALIASES.get(key, key)
        for source in (line, header, derived):
            if canonical in source:
                return True, source.get(canonical)
        suffix = [column for column in self.columns if column.endswith("_" + key)]
        if len({self.columns[column] for column in suffix}) == 1:
            scope, name = self.columns[suffix[0]]
            return True, (raw_row if scope == "line" else self.raw_header).get(name)
        return False, None


def _is_constant(mapping: dict[str, Any]) -> bool:
    """True when a field mapping states a fixed value rather than a source column."""
    source = normalize_label(mapping.get("source_field"))
    return normalize_label(mapping.get("transformation")) == "constant" or source in {"constant", "fixed"} \
        or source.startswith("fixed_value")


def apply_documented_mappings(payload: dict[str, Any], raw_extraction: dict[str, Any] | None,
                              anchors: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    """Take line fields from the columns the customer's documents name, overriding the LLM mapper.

    The canonical mapper guesses which extracted column means what from technical
    names; on a dedicated schema it can swap e.g. "Document No" and "Your Invoice
    No". The conversion documents state it exactly (``field_mappings``: target
    ``invoice_number`` from source "Your Invoice No"), so those pairs win:

    1. Each non-constant mapping whose target is a canonical line field
       (``DOCUMENTED_LINE_TARGETS``) is resolved to a raw LINE column by printed
       label or technical name. Unresolvable sources are reported, not guessed.
    2. Mapper choices that conflict (same column or same target) are dropped. In a
       plain swap (one column displaced, one target freed) the displaced column
       takes the freed target, e.g. "Document No" -> ``customer_document_reference``.
    3. Every affected field is re-read from the raw row and coerced to its
       canonical type; a field left without a column is removed.

    Args:
        payload: Canonical advice (not mutated).
        raw_extraction: Raw view with ``schema_fields``, ``line_items`` and the
            mapper's choices in ``mapping`` (``{"line_items": {column: canonical}}``).
        anchors: Playbook anchors (``field_mappings``).

    Returns:
        (payload copy with documented fields applied, {canonical field: documented
        column label}, warnings). Without raw columns or documented line mappings
        the payload is returned unchanged.
    """
    raw = raw_extraction or {}
    resolver = _Resolver(raw)
    if not resolver.columns:
        return payload, {}, []
    labels = {str(f.get("name")): str(f.get("label") or f.get("name")) for f in raw.get("schema_fields") or []}
    documented: dict[str, str] = {}  # raw column -> canonical field
    warnings: list[str] = []
    for mapping in (anchors or {}).get("field_mappings") or []:
        target = DOCUMENTED_LINE_TARGETS.get(normalize_label(mapping.get("target_field")))
        if not target or _is_constant(mapping) or target in documented.values():
            continue
        scope, column = resolver.columns.get(normalize_label(mapping.get("source_field")), (None, None))
        if scope != "line":
            warnings.append(f"documented column '{mapping.get('source_field')}' for {target} was not found "
                            "among the extracted line columns; the mapper's choice was kept")
            continue
        documented[column] = target
    if not documented:
        return payload, {}, warnings

    mapper = dict((raw.get("mapping") or {}).get("line_items") or {})
    targets = set(documented.values())
    displaced = [column for column, field in mapper.items() if field in targets and column not in documented]
    freed = [field for column, field in mapper.items() if column in documented and field not in targets]
    final = {column: field for column, field in mapper.items() if column not in documented and field not in targets}
    final.update(documented)
    # ponytail: only a single swap is repaired; longer chains leave the displaced column unmapped.
    if len(displaced) == 1 and len(freed) == 1:
        final[displaced[0]] = freed[0]
    column_of = {field: column for column, field in final.items()}

    out = copy.deepcopy(payload)
    for index, line in enumerate(out.get("line_items") or []):
        row = resolver.raw_row(index)
        for field in targets | set(freed):
            if field in column_of:
                line[field] = _normalize_value(row.get(column_of[field]), _LINE_DEFS.get(field, {}))
            else:
                line.pop(field, None)
    return out, {field: labels.get(column, column) for column, field in documented.items()}, warnings


def mapped_company_code(anchors: dict[str, Any]) -> tuple[str, str] | None:
    """Company code stated as a constant header mapping, as (code, mapping target).

    Accepts both stored shapes: ``{"target_field": "iln_recipient", "source_field":
    'Fixed value "CA01"', "transformation": "constant"}`` and ``{"target_field":
    "payee", "source_field": "constant", "transformation": "CA01"}``. Only a
    4-character SAP company code is accepted; anything else is ignored.
    """
    for mapping in anchors.get("field_mappings") or []:
        target = normalize_label(mapping.get("target_field"))
        if target not in _COMPANY_CODE_TARGETS:
            continue
        source, transformation = str(mapping.get("source_field") or ""), str(mapping.get("transformation") or "")
        if normalize_label(transformation) == "constant":
            value = re.sub(r"^\s*fixed\s+value\s*", "", source, flags=re.IGNORECASE)
        elif normalize_label(source) in {"constant", "fixed", "fixed_value"}:
            value = transformation
        else:
            continue
        value = value.strip().strip("\"'“”").strip().upper()
        if re.fullmatch(r"[A-Z0-9]{4}", value):
            return value, target
    return None


def describe(rule: dict[str, Any]) -> str:
    """Human-readable rule text for rationales, e.g. "Rg equals 'A' (priority 3)"."""
    return f"{rule.get('field')} {rule.get('operator')} '{rule.get('value')}' (priority {rule.get('priority')})"


def first_match(rules: list[dict[str, Any]] | None, resolve, warnings: set[str]) -> dict[str, Any] | None:
    """Return the matching rule with the lowest priority (ties keep list order), or None.

    Args:
        rules: Ordered rule dicts.
        resolve: Callable ``field -> (resolved, value)`` for the current line/header.
        warnings: Collects unresolved fields and invalid rules (deduplicated).
    """
    ordered = sorted(enumerate(rules or []), key=lambda pair: (pair[1].get("priority") or 0, pair[0]))
    for _, rule in ordered:
        resolved, value = resolve(str(rule.get("field") or ""))
        if not resolved:
            warnings.add(f"rule field '{rule.get('field')}' was not found in the extracted columns")
            continue
        try:
            if matches(str(rule.get("operator") or ""), value, rule.get("value")):
                return rule
        except ValueError as exc:
            warnings.add(f"rule {describe(rule)} skipped: {exc}")
    return None


def apply_rules(payload: dict[str, Any], raw_extraction: dict[str, Any] | None,
                anchors: dict[str, Any] | None) -> dict[str, Any]:
    """Evaluate a customer's rules on every line of one advice.

    Args:
        payload: Canonical advice ``{"header", "line_items"}`` (not mutated).
        raw_extraction: ``{"schema_fields", "header", "line_items"}`` aligned with
            ``payload["line_items"]``; ``None`` evaluates on canonical fields only.
        anchors: Playbook anchors; ``None``/empty means no rules (all values None).

    Returns:
        ``{"header": {payer_account, company_code, company_code_rule},
           "lines": [{customer_account, customer_account_rule, flags,
                      deduction: {reason_code, rule, action} | None}],
           "warnings": [str]}``.
        ``deduction`` is set only when a specific ``deduction_rules`` entry matched;
        the ``default_rule`` is never applied here (the agent handles those lines).
    """
    anchors = anchors or {}
    resolver = _Resolver(raw_extraction)
    header = payload.get("header") or {}
    lines = payload.get("line_items") or []
    warnings: set[str] = set()
    derived: dict[str, Any] = {}

    def header_resolve(field: str) -> tuple[bool, Any]:
        return resolver.resolve(field, {}, {}, header, derived)

    company_rule = first_match(anchors.get("company_code_rules"), header_resolve, warnings)
    if company_rule:
        derived["company_code"], company_text = company_rule.get("company_code"), describe(company_rule)
    elif anchors.get("default_company_code"):
        derived["company_code"], company_text = anchors["default_company_code"], "default company code"
    elif mapped := mapped_company_code(anchors):
        derived["company_code"], company_text = mapped[0], f"constant header mapping '{mapped[1]}'"
    else:
        derived["company_code"], company_text = None, None

    account_rules = anchors.get("customer_account_rules") or []
    default_account = anchors.get("default_customer_account")
    out_lines: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        raw_row = resolver.raw_row(index)

        def line_resolve(field: str, raw_row=raw_row, line=line) -> tuple[bool, Any]:
            return resolver.resolve(field, raw_row, line, header, derived)

        flags: list[str] = []
        account_rule = first_match(account_rules, line_resolve, warnings)
        if account_rule:
            account, account_text = account_rule.get("customer_account"), describe(account_rule)
        elif default_account:
            account, account_text = default_account, "default customer account"
            if account_rules:
                flags.append("customer-account-defaulted")
        else:
            account, account_text = None, None
        reason_rule = first_match(anchors.get("deduction_rules"), line_resolve, warnings)
        deduction = None if reason_rule is None else {
            "reason_code": reason_rule.get("reason_code"), "rule": describe(reason_rule),
            "action": reason_rule.get("action"), "description": reason_rule.get("description")}
        out_lines.append({"customer_account": account, "customer_account_rule": account_text,
                          "flags": flags, "deduction": deduction})

    line_accounts = {line["customer_account"] for line in out_lines if line["customer_account"]}
    payer = default_account or (next(iter(line_accounts)) if len(line_accounts) == 1 else None)
    return {"header": {"payer_account": payer, "company_code": derived["company_code"],
                       "company_code_rule": company_text},
            "lines": out_lines, "warnings": sorted(warnings)}


def route_lines(payload: dict[str, Any], outcome: dict[str, Any]) -> dict[str, Any]:
    """Decide per line whether a documented rule settles it or the deduction agent must judge it.

    - Deduction line (negative, or carrying a deduction reason) with a specific rule
      match: settled by the rule (nature "chargeback", the rule's reason code).
    - Deduction line without a specific match: agent, told that only the default applies.
    - Positive line that matches a deduction rule: agent, with the rule as a hint; it
      may be the reversal of a customer deduction rather than a payee invoice, and
      a rule never silently turns an invoice into a deduction.
    - Any other line: plain invoice, no agent.

    Args:
        payload: Canonical advice.
        outcome: ``apply_rules`` result for the same payload.

    Returns:
        ``{"engine_entries": {index: interpretation}, "agent_indexes": [index],
           "hints": {index: text}}``.
    """
    from .deductions import needs_analysis

    engine_entries: dict[int, dict[str, Any]] = {}
    agent_indexes: list[int] = []
    hints: dict[int, str] = {}
    for index, line in enumerate(payload.get("line_items") or []):
        deduction = outcome["lines"][index]["deduction"] if index < len(outcome["lines"]) else None
        if needs_analysis(line) and deduction:
            reason = deduction.get("description")
            engine_entries[index] = {
                "document_nature": "chargeback", "reason_code": deduction["reason_code"],
                "residual_items": [], "flags": [],
                "rationale": f"Documented customer rule {deduction['rule']}" + (f": {reason}" if reason else "")}
        elif needs_analysis(line):
            agent_indexes.append(index)
            hints[index] = "No specific documented rule matched; the customer's default rule applies unless the evidence shows otherwise."
        elif deduction:
            agent_indexes.append(index)
            hints[index] = (f"Positive amount, but it matches documented deduction rule {deduction['rule']} "
                            f"(reason {deduction['reason_code']}). Decide whether it reverses a customer deduction "
                            "or is a genuine payee invoice.")
    return {"engine_entries": engine_entries, "agent_indexes": agent_indexes, "hints": hints}


def merge_rule_outcome(enriched: dict[str, Any], outcome: dict[str, Any], rule_resolved: int = 0) -> dict[str, Any]:
    """Write the rule-derived fields into an enriched advice (in place) and refresh its summary.

    Lines get ``customer_account`` / ``customer_account_rule`` (plus rule flags); the
    header gets ``payer_account``, ``company_code`` and ``company_code_rule``; the
    ``interpretation`` summary gets ``rule_resolved_count`` and ``rule_warnings``, and
    ``unresolved_count`` is recomputed because rule flags also need review.

    Returns:
        The same ``enriched`` dict.
    """
    for line, derived in zip(enriched.get("line_items") or [], outcome["lines"]):
        line["customer_account"] = derived["customer_account"]
        line["customer_account_rule"] = derived["customer_account_rule"]
        line["flags"] = list(dict.fromkeys([*(line.get("flags") or []), *derived["flags"]]))
    header = enriched.setdefault("header", {})
    header.update(outcome["header"])
    summary = header.setdefault("interpretation", {})
    summary["rule_resolved_count"] = rule_resolved
    summary["rule_warnings"] = outcome["warnings"]
    # {canonical field: documented column}, e.g. {"invoice_reference": "Your Invoice No"}
    summary["documented_mappings"] = outcome.get("documented_mappings") or {}
    summary["unresolved_count"] = sum(1 for line in enriched.get("line_items") or []
                                      if line.get("reason_code") == "323" or line.get("flags"))
    return enriched
