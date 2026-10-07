"""Administer a customer's dedicated Document AI schema: describe, plan, prepare, test, publish, discard.

Used by the chat agents (``deduction_agent/tools/schema_tools.py``). No agent
code lives here.

Lifecycle::

    describe -> plan (explicit changes, or proposed from a sample)
             -> prepare (new schema version, ACTIVE in Document AI, NOT bound)
             -> test on a sample (optional) -> publish (bind + retire old version) | discard

Why a prepared version cannot affect production: every extraction sends the
schema version bound to the customer in HANA (``CUSTOMER_SCHEMAS``), so a new
active version is only used after ``publish`` rebinds the customer. Document AI
versions are immutable once active, so every change is a NEW version and the
previous one stays available for a rollback (publish it again).

Rules enforced here:
- Only dedicated schemas are edited; the shared canonical schema is read-only.
- A dedicated schema is only used for a priority customer (``schema_select``
  ignores the binding otherwise), so ``publish`` requires priority.
- Field sets are always sent complete, with only name/label/description/
  formattingType (``SapDoxClient`` adds the setup defaults SAP requires) and
  labels unique across header and line fields (SAP error ES130 otherwise).
- The bound version is never reconfigured: configuring an active version first
  deactivates it, which would stop production extraction.

Example:
    current = current_fields(engine, dox, "northwind", "ai4u_payment_advice")
    plan = plan_changes(current, add=[{"name": "payee_name", "scope": "header", "description": "..."}])
    prepared = prepare(engine, dox, plan, "ai4u_payment_advice")
    report = test_extract(engine, dox, "northwind", prepared["version"], "advice.pdf", "ai4u_payment_advice")
    publish(engine, dox, "northwind", prepared["version"], "ai4u_payment_advice")
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import canonical as C
from .config import DEFAULT_MODEL
from .customers import CustomerSchema, bind_schema, get_bound_schema, get_customer
from .schema_select import _VALID_FORMATTING, _normalize_field_name, dedicated_schema_name, generate_schema_fields
from .schema_select import sample_document_text

FIELD_KEYS = ("name", "label", "description", "formattingType")
SCOPES = ("header", "line")


class SchemaAdminError(ValueError):
    """A refused or impossible schema operation; the message is meant for the user."""


@dataclass(frozen=True)
class CurrentSchema:
    """The customer's dedicated schema as the base of a change.

    Args:
        client_key: Customer key.
        schema_id: Dedicated schema id, or None when the customer has none yet.
        version: Bound dedicated version, or None.
        header: Header fields (dicts with FIELD_KEYS).
        line: Line-item fields.
    """

    client_key: str
    schema_id: str | None
    version: str | None
    header: list[dict[str, Any]] = field(default_factory=list)
    line: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class SchemaPlan:
    """A complete proposed field set for a customer's dedicated schema.

    Args:
        client_key: Customer key.
        base_version: Dedicated version bound when the plan was made (None for a first schema);
            ``prepare`` refuses a plan whose base is no longer bound.
        header: Complete header field set.
        line: Complete line-item field set.
        diff: ``{"added": [...], "changed": [...], "removed": [...]}`` against the base.
    """

    client_key: str
    base_version: str | None
    header: list[dict[str, Any]]
    line: list[dict[str, Any]]
    diff: dict[str, list[dict[str, Any]]]


def copy_fields(details: dict[str, Any], collection: str) -> list[dict[str, Any]]:
    """Copy one field collection of a version, keeping only the keys safe to send back."""
    return [{key: item[key] for key in FIELD_KEYS if item.get(key)}
            for item in details.get(collection) or [] if isinstance(item, dict) and item.get("name")]


def _versions(dox: Any, schema_id: str, client_id: str) -> list[dict[str, str]]:
    """Versions of a schema as ``[{version, state}]`` with lower-case states."""
    return [{"version": str(v.get("version")), "state": str(v.get("state", "")).lower()}
            for v in dox.list_schema_versions(schema_id, client_id=client_id)]


def _require_customer(engine: Any, client_key: str) -> Any:
    """Return the registered customer or refuse."""
    customer = get_customer(engine, client_key)
    if customer is None:
        raise SchemaAdminError(f"unknown customer {client_key!r}; register it first")
    return customer


def _dedicated_schema_id(engine: Any, dox: Any, client_key: str, client_id: str) -> str | None:
    """The customer's dedicated schema id: the bound one, else the one named for the customer."""
    bound = get_bound_schema(engine, client_key)
    if bound and not bound.is_canonical:
        return bound.schema_id
    existing = dox.get_schema_by_name(dedicated_schema_name(client_key), client_id=client_id)
    return existing["id"] if existing else None


def describe(engine: Any, dox: Any, client_key: str, client_id: str) -> dict[str, Any]:
    """Show the schema a customer's advices are extracted with, and whether it can be edited.

    Returns:
        ``{client_key, display_name, priority, schema: {schema_id, version, source, canonical} | None,
        header_fields, line_fields, versions, editable, note}``. ``versions`` lists the dedicated
        schema's versions (empty for the canonical schema).
    """
    customer = _require_customer(engine, client_key)
    bound = get_bound_schema(engine, client_key)
    report: dict[str, Any] = {"client_key": client_key, "display_name": customer.display_name,
                              "priority": customer.is_critical, "schema": None, "header_fields": [],
                              "line_fields": [], "versions": []}
    if bound:
        details = dox.get_schema_version_details(bound.schema_id, bound.schema_version, client_id=client_id)
        report["schema"] = {"schema_id": bound.schema_id, "version": bound.schema_version, "source": bound.source,
                            "canonical": bound.is_canonical}
        report["header_fields"] = copy_fields(details, "headerFields")
        report["line_fields"] = copy_fields(details, "lineItemFields")
    dedicated_id = _dedicated_schema_id(engine, dox, client_key, client_id)
    if dedicated_id:
        report["versions"] = _versions(dox, dedicated_id, client_id)
    if not customer.is_critical:
        report["editable"], report["note"] = False, (
            "Not a priority customer: its advices use the shared canonical schema, which is read-only. "
            "Make the customer priority to give it its own schema.")
    elif bound is None or bound.is_canonical:
        report["editable"], report["note"] = True, (
            "Priority customer without its own schema yet: propose one from a sample document.")
    else:
        report["editable"], report["note"] = True, "Own (dedicated) schema: changes create a new version."
    return report


def current_fields(engine: Any, dox: Any, client_key: str, client_id: str) -> CurrentSchema:
    """The bound dedicated schema's fields as the base of a plan (empty for a first schema)."""
    _require_customer(engine, client_key)
    bound = get_bound_schema(engine, client_key)
    if bound is None or bound.is_canonical:
        return CurrentSchema(client_key, _dedicated_schema_id(engine, dox, client_key, client_id), None)
    details = dox.get_schema_version_details(bound.schema_id, bound.schema_version, client_id=client_id)
    return CurrentSchema(client_key, bound.schema_id, bound.schema_version,
                         copy_fields(details, "headerFields"), copy_fields(details, "lineItemFields"))


def diff_fields(before: CurrentSchema, header: list[dict[str, Any]], line: list[dict[str, Any]]) -> dict[str, list]:
    """Added, changed (per key, before -> after) and removed fields between the base and a new set.

    Fields are identified by (scope, name): Document AI allows the same name in header and line
    fields (a customer layout can have ``document_no`` in both).
    """
    def index(head: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
        return {(scope, f["name"]): f for scope, fields in (("header", head), ("line", rows)) for f in fields}

    old, new = index(before.header, before.line), index(header, line)
    changed = []
    for key in sorted(old.keys() & new.keys()):
        changes = {k: [old[key].get(k), new[key].get(k)] for k in FIELD_KEYS[1:] if old[key].get(k) != new[key].get(k)}
        if changes:
            changed.append({"name": key[1], "scope": key[0], "changes": changes})
    return {"added": [{"scope": s, **f} for (s, n), f in new.items() if (s, n) not in old],
            "changed": changed,
            "removed": [{"scope": s, **f} for (s, n), f in old.items() if (s, n) not in new]}


def _validated(header: list[dict[str, Any]], line: list[dict[str, Any]]) -> None:
    """Refuse field sets SAP or the pipeline cannot use (duplicate names per scope, duplicate labels, no lines)."""
    labels: set[str] = set()
    for scope, fields in (("header", header), ("line", line)):
        names = [f["name"] for f in fields]
        duplicate = next((n for n in names if names.count(n) > 1), None)
        if duplicate:
            raise SchemaAdminError(f"{scope} field name {duplicate!r} is used twice")
        for item in fields:
            label = str(item.get("label") or item["name"]).casefold()
            if label in labels:
                raise SchemaAdminError(f"label {item.get('label') or item['name']!r} is used twice; labels must be "
                                       "unique across header and line fields")
            labels.add(label)
    if not line:
        raise SchemaAdminError("a payment advice schema needs at least one line-item field")


def _type(value: Any) -> str:
    """A formatting type the agent may set (existing fields keep theirs, e.g. country/region)."""
    ftype = str(value or "string").lower()
    if ftype not in _VALID_FORMATTING:
        raise SchemaAdminError(f"type must be one of {', '.join(sorted(_VALID_FORMATTING))}, not {ftype!r}")
    return ftype


def _new_field(spec: dict[str, Any]) -> dict[str, Any]:
    """Build a field dict from an agent/user spec ``{name, label?, description?, type?}``."""
    name = _normalize_field_name(spec.get("name") or "")
    if name == "field":
        raise SchemaAdminError("every added field needs a name")
    item = {"name": name, "label": str(spec.get("label") or name.replace("_", " ").title())[:200],
            "formattingType": _type(spec.get("type") or spec.get("formattingType"))}
    if spec.get("description"):
        item["description"] = str(spec["description"])[:500]
    return item


def _find(header: list[dict[str, Any]], line: list[dict[str, Any]], spec: dict[str, Any] | str,
          action: str) -> tuple[str, dict[str, Any]]:
    """Locate one existing field by name (and scope when the name exists in header and line)."""
    name, scope = (spec, None) if isinstance(spec, str) else (spec.get("name"), spec.get("scope"))
    matches = [(s, f) for s, fields in (("header", header), ("line", line)) for f in fields
               if f["name"] == name and scope in (None, s)]
    if not matches:
        raise SchemaAdminError(f"cannot {action} {name!r}: no such field")
    if len(matches) > 1:
        raise SchemaAdminError(f"{name!r} exists as header and line field; say which (scope 'header' or 'line')")
    return matches[0]


def plan_changes(current: CurrentSchema, *, add: list[dict[str, Any]] | None = None,
                 update: list[dict[str, Any]] | None = None,
                 remove: list[dict[str, Any] | str] | None = None) -> SchemaPlan:
    """Apply explicit field changes to the base field set (read-only; nothing is sent to SAP).

    Args:
        current: Base from ``current_fields``.
        add: New fields ``{name, scope: "header"|"line", label?, description?, type?}``.
        update: ``{name, scope?, label?, description?, type?}`` for existing fields.
        remove: ``{name, scope?}`` (or a plain name) of fields to drop.

    Returns:
        The complete new field set with its diff.

    Raises:
        SchemaAdminError: Unknown/ambiguous/duplicate fields, bad types or scopes, duplicate labels.
    """
    header = [dict(f) for f in current.header]
    line = [dict(f) for f in current.line]
    for spec in remove or []:
        scope, target = _find(header, line, spec, "remove")
        if scope == "header":
            header = [f for f in header if f is not target]
        else:
            line = [f for f in line if f is not target]
    for spec in update or []:
        _, target = _find(header, line, spec, "update")
        for key, source in (("label", "label"), ("description", "description")):
            if spec.get(source) is not None:
                target[key] = str(spec[source])
        if spec.get("type") is not None:
            target["formattingType"] = _type(spec["type"])
    for spec in add or []:
        scope = spec.get("scope", "header")
        if scope not in SCOPES:
            raise SchemaAdminError(f"scope must be 'header' or 'line', not {scope!r}")
        item = _new_field(spec)
        fields = header if scope == "header" else line
        if any(f["name"] == item["name"] for f in fields):
            raise SchemaAdminError(f"{scope} field {item['name']!r} already exists; update it instead")
        fields.append(item)
    _validated(header, line)
    return SchemaPlan(current.client_key, current.version, header, line, diff_fields(current, header, line))


def propose_from_sample(current: CurrentSchema, sample_path: str | Path, *, merge: bool = True,
                        model: str = DEFAULT_MODEL) -> SchemaPlan:
    """Let the LLM propose fields from a sample document (read-only).

    Args:
        current: Base from ``current_fields``.
        sample_path: Sample advice (PDF, spreadsheet, Word).
        merge: True keeps every current field and only adds proposed fields with new names and
            labels (safe for an existing schema whose fields customer rules may use); False replaces the set.
        model: LLM for the proposal.
    """
    proposed_header, proposed_line = generate_schema_fields(
        sample_document_text(sample_path), client_key=current.client_key, model=model)
    as_dicts = lambda fields: [{k: v for k, v in f.to_dict().items() if k in FIELD_KEYS and v} for f in fields]
    header, line = as_dicts(proposed_header), as_dicts(proposed_line)
    if merge and (current.header or current.line):
        labels = {str(f.get("label") or f["name"]).casefold() for f in current.header + current.line}

        def fresh(fields: list[dict[str, Any]], existing: list[dict[str, Any]]) -> list[dict[str, Any]]:
            names = {f["name"] for f in existing}
            return [f for f in fields if f["name"] not in names and str(f.get("label") or f["name"]).casefold() not in labels]

        header = [dict(f) for f in current.header] + fresh(header, current.header)
        line = [dict(f) for f in current.line] + fresh(line, current.line)
    _validated(header, line)
    return SchemaPlan(current.client_key, current.version, header, line, diff_fields(current, header, line))


def prepare(engine: Any, dox: Any, plan: SchemaPlan, client_id: str) -> dict[str, Any]:
    """Create a new ACTIVE, unbound version of the customer's dedicated schema from a plan.

    Production is unaffected: advices keep using the bound version until ``publish``.
    A first schema is created as ``dedicated_schema_name(client_key)``. A draft or
    inactive version newer than the bound one (left by an interrupted run) is reused.

    Returns:
        ``{schema_id, version, created_schema, reused_version}``.

    Raises:
        SchemaAdminError: Stale plan (another version was published meanwhile).
    """
    _require_customer(engine, plan.client_key)
    bound = get_bound_schema(engine, plan.client_key)
    bound_version = bound.schema_version if bound and not bound.is_canonical else None
    if bound_version != plan.base_version:
        raise SchemaAdminError(f"the plan was made from version {plan.base_version}, but version {bound_version} "
                               "is bound now; make a new plan")
    schema_id = _dedicated_schema_id(engine, dox, plan.client_key, client_id)
    created_schema = schema_id is None
    if created_schema:
        schema_id = dox.create_schema(
            client_id=client_id, schema_name=dedicated_schema_name(plan.client_key),
            schema_description=f"Dedicated payment advice schema for {plan.client_key}",
            document_type=C.CANONICAL_DOCUMENT_TYPE, document_type_description="Customer payment/remittance advice",
        )["id"]
    versions = _versions(dox, schema_id, client_id)
    leftover = next((v["version"] for v in versions if v["state"] in {"draft", "inactive"}
                     and v["version"] != bound_version
                     and (bound_version is None or int(v["version"]) > int(bound_version))), None)
    if leftover:
        version = leftover
    else:
        created = dox.create_schema_version(schema_id, client_id=client_id)
        version = str(created.get("version") or max(int(v["version"]) for v in _versions(dox, schema_id, client_id)))
    if version == bound_version:  # configuring it would deactivate production extraction
        raise SchemaAdminError(f"refusing to reconfigure the bound version {version}")
    dox.configure_schema_version(schema_id, version, client_id=client_id, header_fields=plan.header,
                                 line_item_fields=plan.line, replace=True, activate=True)
    return {"schema_id": schema_id, "version": version, "created_schema": created_schema,
            "reused_version": bool(leftover)}


def _owned_version(engine: Any, dox: Any, client_key: str, version: str, client_id: str) -> tuple[str, dict]:
    """The customer's dedicated schema id and the details of one of its versions (refuse otherwise)."""
    schema_id = _dedicated_schema_id(engine, dox, client_key, client_id)
    if not schema_id:
        raise SchemaAdminError(f"{client_key!r} has no dedicated schema")
    if str(version) not in {v["version"] for v in _versions(dox, schema_id, client_id)}:
        raise SchemaAdminError(f"version {version} does not exist in {client_key!r}'s schema")
    return schema_id, dox.get_schema_version_details(schema_id, version, client_id=client_id)


def test_extract(engine: Any, dox: Any, client_key: str, version: str, sample_path: str | Path, client_id: str,
                 *, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """Extract a sample with a (prepared) version and preview the canonical result. Writes nothing.

    Returns:
        ``{schema_id, version, header_values, empty_header_fields, line_count, first_lines,
        empty_line_fields, canonical_header, mapping}``: the raw values Document AI returned,
        the fields it left empty, and how the LLM mapper would fill the canonical header.
    """
    from .extract import extract_document
    from .mapper import map_to_canonical

    schema_id, details = _owned_version(engine, dox, client_key, version, client_id)
    if str(details.get("state", "")).lower() != "active":
        raise SchemaAdminError(f"version {version} is not active; prepare it again before testing")
    with tempfile.TemporaryDirectory(prefix="pa_schema_test_") as work:
        result = extract_document(sample_path, client=dox, schema={}, dox_client_id=client_id, out_dir=work,
                                  schema_id=schema_id, schema_version=version)
    aggregate = result.aggregate
    headers = aggregate.get("headers") or {}
    lines = aggregate.get("line_items") or []
    mapped = map_to_canonical(aggregate, model=model)
    filled_line_keys = {key for row in lines for key, value in row.items() if value not in (None, "")}
    return {"schema_id": schema_id, "version": str(version), "header_values": headers,
            "empty_header_fields": [f["name"] for f in copy_fields(details, "headerFields")
                                    if headers.get(f["name"]) in (None, "")],
            "line_count": len(lines), "first_lines": lines[:5],
            "empty_line_fields": [f["name"] for f in copy_fields(details, "lineItemFields")
                                  if f["name"] not in filled_line_keys],
            "canonical_header": mapped.get("header"), "mapping": mapped.get("mapping")}


def publish(engine: Any, dox: Any, client_key: str, version: str, client_id: str) -> dict[str, Any]:
    """Bind a version to the customer (production change) and retire the previous dedicated version.

    Binding happens before the old version is deactivated, so a failure leaves a working
    binding. Publishing an older version is a rollback (it is reactivated if needed).

    Returns:
        ``{schema_id, version, previous_version, previous_deactivated}``.

    Raises:
        SchemaAdminError: Not a priority customer, unknown version, already published.
    """
    from .pipeline import _SCHEMA_FIELDS_CACHE

    customer = _require_customer(engine, client_key)
    if not customer.is_critical:
        raise SchemaAdminError(f"{client_key!r} is not a priority customer, so a dedicated schema would be ignored; "
                               "make it priority first")
    schema_id, details = _owned_version(engine, dox, client_key, version, client_id)
    bound = get_bound_schema(engine, client_key)
    previous = bound.schema_version if bound and not bound.is_canonical and bound.schema_id == schema_id else None
    if previous == str(version):
        raise SchemaAdminError(f"version {version} is already published for {client_key!r}")
    if str(details.get("state", "")).lower() != "active":
        dox.activate_schema_version(schema_id, version, client_id=client_id)
    bind_schema(engine, CustomerSchema(client_key, schema_id, str(version), False, "dedicated"))
    if previous:
        dox.deactivate_schema_version(schema_id, previous, client_id=client_id)
    for key in ((schema_id, str(version)), (schema_id, str(previous))):
        _SCHEMA_FIELDS_CACHE.pop(key, None)
    return {"schema_id": schema_id, "version": str(version), "previous_version": previous,
            "previous_deactivated": bool(previous)}


def discard(engine: Any, dox: Any, client_key: str, version: str, client_id: str) -> dict[str, Any]:
    """Deactivate a prepared version that will not be published (never the bound one)."""
    schema_id, details = _owned_version(engine, dox, client_key, version, client_id)
    bound = get_bound_schema(engine, client_key)
    if bound and not bound.is_canonical and bound.schema_id == schema_id and bound.schema_version == str(version):
        raise SchemaAdminError(f"version {version} is the published version; publish another version instead")
    if str(details.get("state", "")).lower() == "active":
        dox.deactivate_schema_version(schema_id, version, client_id=client_id)
    return {"schema_id": schema_id, "version": str(version), "deactivated": True}
