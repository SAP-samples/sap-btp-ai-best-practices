"""
Schema selection and generation for the Payment Advice Extractor (UC-01).

Implements the criticality-tiered strategy:

- Critical client -> a dedicated, purpose-built Document AI schema. If none is bound
  yet, one is GENERATED from a sample document (LLM proposes client-native fields),
  created + activated under the project client id, and bound as ``dedicated``.
- Non-critical client -> the canonical ``payment_advice_canonical`` schema
  first (bound as ``canonical``). A per-client custom schema + LLM mapping is the
  documented fallback when canonical extraction is poor.

A dedicated schema uses client-native field names (better literal extraction); the
raw extraction is later normalized by ``mapper.map_to_canonical``. A canonical schema
needs no mapping (``is_canonical=True``).

Human-in-the-loop: generating a new dedicated schema is gated. With ``assume_yes``
the schema is created immediately; otherwise ``select_schema`` returns a proposal for
the CLI to confirm before anything is written to SAP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from contextlib import nullcontext
from sqlalchemy import text as sql_text
from .hana_schema import CUSTOMERS
from typing import Any, Callable

from dox_client import FieldDefinition
from pypdf import PdfReader

from . import canonical as C
from .config import DEFAULT_MODEL
from .customers import CustomerSchema, bind_schema, get_bound_schema, get_customer, normalize_client_key
from .llm import complete_json
from .splitter import FORMAT_OPAQUE, FORMAT_PDF, FORMAT_TABULAR, _read_table, detect_format

# Formatting types offered for a generated field (others fall back to string).
# "country/region" is deliberately excluded: Document AI normalizes such cells to ISO
# country codes and returns null for short codes like a region letter "A", which
# silently drops columns that customer rules depend on. Codes stay plain strings.
_VALID_FORMATTING = {"string", "number", "date", "discount", "currency"}

_GEN_SYSTEM_PROMPT = (
    "You design SAP Document AI extraction schemas for payment/remittance advices. "
    "A literal extractor fills a field only when the document plainly shows that label "
    "or column, so propose fields that match the document's own wording. "
    "Reply with a single JSON object and nothing else."
)


@dataclass
class SchemaSelection:
    """Result of selecting a schema for a client.

    Attributes:
        status: "ready" when a usable schema is bound; "proposed" when a dedicated
            schema must be generated and needs confirmation (no SAP write happened).
        schema_id / schema_version: The schema to extract with (when ready).
        is_canonical: Whether extraction output is already canonical (skip mapping).
        source: "dedicated" | "generated" | "canonical".
        proposed_header / proposed_line: Field definitions awaiting confirmation
            (only when status == "proposed").
    """

    status: str
    schema_id: str | None = None
    schema_version: str | None = None
    is_canonical: bool = False
    source: str | None = None
    proposed_header: list[FieldDefinition] = field(default_factory=list)
    proposed_line: list[FieldDefinition] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Document sampling
# --------------------------------------------------------------------------- #
def sample_document_text(path: str | Path, *, max_chars: int = 6000, max_rows: int = 15) -> str:
    """
    Extract a short text sample of a document to show the schema-generating LLM.

    PDF: text of the first two pages. Tabular: header plus the first ``max_rows``
    rows. Opaque: the first ``max_chars`` characters. Returns "" when no text can be
    sampled (e.g. an image), in which case schema generation is not possible.
    """
    path = Path(path)
    fmt = detect_format(path)
    text = ""
    if fmt == FORMAT_PDF:
        reader = PdfReader(str(path))
        text = "\n".join((page.extract_text() or "") for page in reader.pages[:2])
    elif fmt == FORMAT_TABULAR:
        header, rows = _read_table(path)
        lines = ["\t".join("" if c is None else str(c) for c in header)]
        lines += ["\t".join("" if c is None else str(c) for c in row) for row in rows[:max_rows]]
        text = "\n".join(lines)
    elif fmt == FORMAT_OPAQUE:
        try:
            if path.suffix.lower() == '.docx':
                from ..email_ingestion.preview import preview
                text = preview(path.name, path.read_bytes()) or ''
            else:
                text = path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            text = ""
    return text[:max_chars].strip()


# --------------------------------------------------------------------------- #
# Field generation
# --------------------------------------------------------------------------- #
def _normalize_field_name(raw: str) -> str:
    """Normalize a proposed field name to SAP's allowed technical-name charset."""
    import re

    name = re.sub(r"\s+", "_", str(raw).strip())
    # Keep letters, numbers, and the SAP-permitted punctuation set.
    name = re.sub(r"[^A-Za-z0-9_\-.,&$#~]", "", name)
    return name or "field"


def _to_field_definitions(items: Any, used_labels: set[str]) -> list[FieldDefinition]:
    """Convert LLM-proposed field dicts into unique, valid FieldDefinition objects."""
    definitions: list[FieldDefinition] = []
    seen_names: set[str] = set()
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        name = _normalize_field_name(item["name"])
        if name in seen_names:
            continue
        seen_names.add(name)

        ftype = str(item.get("type") or item.get("formattingType") or "string").strip().lower()
        if ftype == "integer":
            ftype = "number"
        if ftype not in _VALID_FORMATTING:
            ftype = "string"

        label = str(item.get("label") or name.replace("_", " ").title())[:200]
        # Labels must be globally unique across header + line items (case-insensitive).
        base_label, suffix = label, 2
        while label.casefold() in used_labels:
            label = f"{base_label} ({suffix})"
            suffix += 1
        used_labels.add(label.casefold())

        definitions.append(
            FieldDefinition(
                name=name,
                label=label,
                description=str(item.get("description") or "")[:500] or None,
                formattingType=ftype,
            )
        )
    return definitions


def retyped_field_definitions(
    details: dict[str, Any], field_name: str, formatting_type: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """
    Copy a schema version's fields with one field's formatting type changed.

    Only name/label/description/formattingType are kept: sending an explicit
    number ``formatting`` object makes the Document AI fields endpoint fail.
    ``SapDoxClient`` adds the required setup defaults (``setupType``,
    ``setup``) when the fields are sent.

    Args:
        details: ``get_schema_version_details`` response.
        field_name: Header or line-item field to retype.
        formatting_type: New SAP formatting type (e.g. ``string``).

    Returns:
        (header_fields, line_item_fields, previous_type).

    Raises:
        ValueError: When the field does not exist in the version.
    """
    def copy_fields(collection: str) -> list[dict[str, Any]]:
        return [{key: item[key] for key in ("name", "label", "description", "formattingType") if item.get(key)}
                for item in details.get(collection) or [] if isinstance(item, dict) and item.get("name")]

    header, lines = copy_fields("headerFields"), copy_fields("lineItemFields")
    target = next((f for f in header + lines if f["name"] == field_name), None)
    if target is None:
        names = ", ".join(f["name"] for f in header + lines)
        raise ValueError(f"field {field_name!r} not in schema version (fields: {names})")
    previous = str(target.get("formattingType") or "string")
    target["formattingType"] = formatting_type
    return header, lines, previous


def generate_schema_fields(
    sample_text: str,
    *,
    client_key: str,
    model: str = DEFAULT_MODEL,
    openai_create: Callable[..., Any] | None = None,
    gemini_generate: Callable[..., Any] | None = None,
) -> tuple[list[FieldDefinition], list[FieldDefinition]]:
    """
    Ask the LLM to propose header + line-item fields from a document sample.

    Returns:
        (header_fields, line_item_fields) as SAP FieldDefinition objects with unique
        labels and valid formatting types.

    Raises:
        ValueError: If ``sample_text`` is empty (nothing to base a schema on).
    """
    if not sample_text.strip():
        raise ValueError(f"no text could be sampled for {client_key!r}; cannot generate a schema")

    user_prompt = (
        f"Client: {client_key}\n\n"
        "From this payment-advice sample, propose SAP Document AI extraction fields that "
        "match the labels/columns literally present. Split header (document-level) from "
        "line-item (per-row) fields. Use type one of: string, number, date, currency, "
        "discount. Codes such as region, province, store, branch, DC, member or account "
        "columns are type string, even when they are short letters or digits.\n\n"
        f"SAMPLE:\n{sample_text}\n\n"
        'Return JSON: {"header": [{"name","label","description","type"}], '
        '"line_items": [{"name","label","description","type"}]}'
    )
    result = complete_json(
        _GEN_SYSTEM_PROMPT, user_prompt, model=model,
        openai_create=openai_create, gemini_generate=gemini_generate,
        route_label="schema_select:generate_fields",
    )
    used_labels: set[str] = set()
    header = _to_field_definitions(result.get("header"), used_labels)
    line = _to_field_definitions(result.get("line_items"), used_labels)
    if not header and not line:
        raise ValueError("schema generation produced no usable fields")
    return header, line


# --------------------------------------------------------------------------- #
# SAP schema creation
# --------------------------------------------------------------------------- #
def _create_and_activate_schema(
    client: Any,
    *,
    dox_client_id: str,
    schema_name: str,
    schema_description: str,
    header_fields: list[FieldDefinition],
    line_fields: list[FieldDefinition],
) -> tuple[str, str]:
    """Create (or reuse), configure, and activate a schema; return (id, version)."""
    existing = client.get_schema_by_name(schema_name, client_id=dox_client_id)
    if existing:
        schema_id = existing["id"]
    else:
        resp = client.create_schema(
            client_id=dox_client_id,
            schema_name=schema_name,
            schema_description=schema_description,  # SAP requires this
            document_type=C.CANONICAL_DOCUMENT_TYPE,
            document_type_description="Customer payment/remittance advice",
        )
        schema_id = resp["id"]

    versions = client.list_schema_versions(schema_id, client_id=dox_client_id)
    active = next((v for v in versions if str(v.get("state", "")).lower() == "active"), None)
    if active is not None:
        return schema_id, str(active["version"])
    version = next(
        (str(v["version"]) for v in versions if str(v.get("state", "")).lower() in {"inactive", "draft"}),
        str(versions[0]["version"]) if versions else "1",
    )
    client.configure_schema_version(
        schema_id,
        version,
        client_id=dox_client_id,
        header_fields=header_fields,
        line_item_fields=line_fields,
        replace=True,
        activate=True,
    )
    return schema_id, version


def ensure_canonical_schema(client: Any, *, dox_client_id: str, canonical_guard=None) -> tuple[str, str]:
    """Return an active canonical schema, optionally guarded by a context-manager factory."""
    header_fields, line_fields = C.build_field_definitions()
    # Inbox workers supply a cross-process HANA guard; existing CLI/API callers
    # retain their original behavior. Never hold this guard through extraction.
    with canonical_guard() if canonical_guard else nullcontext():
        return _create_and_activate_schema(
            client,
            dox_client_id=dox_client_id,
            schema_name=C.CANONICAL_SCHEMA_NAME,
            schema_description="Canonical payment advice schema (UC-01)",
            header_fields=header_fields,
            line_fields=line_fields,
        )


def dedicated_schema_name(client_key: str) -> str:
    """Document AI schema name of a customer's dedicated schema."""
    return f"payment_advice_client_{client_key}"


def create_dedicated_schema(
    client: Any,
    engine: Any,
    *,
    client_key: str,
    dox_client_id: str,
    header_fields: list[FieldDefinition],
    line_fields: list[FieldDefinition],
) -> CustomerSchema:
    """Create+activate a client's dedicated schema and bind it as ``dedicated``."""
    schema_id, version = _create_and_activate_schema(
        client,
        dox_client_id=dox_client_id,
        schema_name=dedicated_schema_name(client_key),
        schema_description=f"Dedicated payment advice schema for {client_key} (UC-01)",
        header_fields=header_fields,
        line_fields=line_fields,
    )
    binding = CustomerSchema(
        client_key=client_key,
        schema_id=schema_id,
        schema_version=version,
        is_canonical=False,
        source="dedicated",
    )
    bind_schema(engine, binding)
    return binding


# --------------------------------------------------------------------------- #
# Tiered selection
# --------------------------------------------------------------------------- #
def _select_schema(
    engine: Any,
    client: Any,
    raw_client: str,
    sample_path: str | Path,
    *,
    dox_client_id: str,
    assume_yes: bool = False,
    model: str = DEFAULT_MODEL,
    openai_create: Callable[..., Any] | None = None,
    gemini_generate: Callable[..., Any] | None = None,
    canonical_guard=None,
) -> SchemaSelection:
    """
    Select (or generate) the schema to extract a client's advice with.

    Args:
        engine: HANA engine for customer/schema lookups and bindings.
        client: SAP Document AI client.
        raw_client: The ``--client`` argument (normalized internally).
        sample_path: The advice file, used as a sample when generating a schema.
        dox_client_id: SAP Document AI client id.
        assume_yes: Create a generated dedicated schema without pausing.
        model / llm_create: Schema-generation LLM controls (fake-able).
        canonical_guard: Optional context-manager factory for shared canonical setup.

    Returns:
        A ``SchemaSelection``. ``status == "proposed"`` means a dedicated schema was
        generated but not created (awaiting confirmation).
    """
    client_key = normalize_client_key(raw_client)
    customer = get_customer(engine, client_key)
    is_critical = bool(customer and customer.is_critical)

    bound = get_bound_schema(engine, client_key)
    if bound is not None and bound.is_canonical != is_critical:
        return SchemaSelection(
            status="ready",
            schema_id=bound.schema_id,
            schema_version=bound.schema_version,
            is_canonical=bound.is_canonical,
            source=bound.source,
        )

    if not is_critical:
        # Non-critical, no binding yet: use the canonical schema.
        schema_id, version = ensure_canonical_schema(client, dox_client_id=dox_client_id, canonical_guard=canonical_guard)
        bind_schema(
            engine,
            CustomerSchema(client_key, schema_id, version, is_canonical=True, source="canonical"),
        )
        return SchemaSelection("ready", schema_id, version, True, "canonical")

    # Critical, no dedicated schema yet: generate one from the sample document.
    sample_text = sample_document_text(sample_path)
    if not sample_text:
        # Scans/images need OCR before schema generation; use canonical extraction as the sample.
        import json
        import tempfile
        from .extract import extract_document
        schema_id, version = ensure_canonical_schema(client, dox_client_id=dox_client_id, canonical_guard=canonical_guard)
        with tempfile.TemporaryDirectory(prefix='schema-sample-') as directory:
            sample = extract_document(sample_path, client=client, schema=C.CANONICAL_SCHEMA,
                dox_client_id=dox_client_id, out_dir=directory, schema_id=schema_id, schema_version=version)
        sample_text = json.dumps(sample.aggregate)[:20000]
    header_fields, line_fields = generate_schema_fields(
        sample_text, client_key=client_key, model=model,
        openai_create=openai_create, gemini_generate=gemini_generate,
    )
    if not assume_yes:
        return SchemaSelection(
            status="proposed",
            source="dedicated",
            proposed_header=header_fields,
            proposed_line=line_fields,
        )
    binding = create_dedicated_schema(
        client,
        engine,
        client_key=client_key,
        dox_client_id=dox_client_id,
        header_fields=header_fields,
        line_fields=line_fields,
    )
    return SchemaSelection(
        "ready", binding.schema_id, binding.schema_version, False, "dedicated"
    )


def select_schema(engine: Any, client: Any, raw_client: str, sample_path: str | Path, *,
                  dox_client_id: str, assume_yes: bool = False, model: str = DEFAULT_MODEL,
                  openai_create: Callable[..., Any] | None = None,
                  gemini_generate: Callable[..., Any] | None = None, canonical_guard=None) -> SchemaSelection:
    """Serialize schema selection/creation on the HANA customer row across API/CLI workers.

    An absent engine is supported only for injected offline schema-selection tests.
    Existing active schemas are reused; a promoted canonical binding is not dedicated.
    canonical_guard optionally serializes shared canonical setup across customers.
    """
    with engine.begin() if engine is not None else nullcontext() as connection:
        if connection is not None:
            connection.execute(sql_text(f'SELECT "CLIENT_KEY" FROM "{CUSTOMERS}" WHERE "CLIENT_KEY"=:key FOR UPDATE'),
                               {'key': normalize_client_key(raw_client)}).first()
        return _select_schema(engine, client, raw_client, sample_path, dox_client_id=dox_client_id,
            assume_yes=assume_yes, model=model, openai_create=openai_create, gemini_generate=gemini_generate,
            canonical_guard=canonical_guard)
