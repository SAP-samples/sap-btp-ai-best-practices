"""
UC-01 extraction as a reusable, non-interactive service function.

This module exposes the UC-01 pipeline as one blocking, non-interactive call
suitable for an HTTP handler or a worker: given an advice file on disk + a client name, it
resolves (and, for a critical client with no schema yet, generates) the schema
and runs the pipeline to a canonical ``{header, line_items}`` payload.

It is deliberately thin: every step reuses an existing UC-01 library function.
The only policy choice made here is ``assume_yes=True`` -- a browser upload
cannot answer the CLI's "create a dedicated schema? [y/N]" prompt, so schema
generation proceeds unattended, which matches UC-01's documented
"create a schema if necessary" behaviour.

This function performs blocking network I/O (SAP Document AI). Callers on an
async event loop MUST run it in a worker thread (e.g. ``asyncio.to_thread``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import pipeline
from .customers import normalize_client_key
from .schema_select import select_schema


def extract_to_canonical(
    file_path: str | Path,
    client: str,
    *,
    engine: Any,
    dox: Any,
    settings: Any,
    assume_yes: bool = True,
    out_dir: Path | None = None,
    canonical_guard: Any = None,
) -> dict[str, Any]:
    """
    Extract one advice file to a canonical payload, resolving the schema first.

    Args:
        file_path: Path to the advice file on disk (suffix must reflect the real
            format so the splitter can detect it, e.g. ``.pdf``).
        client: Client key or display name (normalized internally).
        engine: HANA engine for customer/schema lookups and bindings.
        dox: SAP Document AI client.
        settings: ``PaymentAdviceSettings`` (client id, models, limits, out dir).
        assume_yes: When True (default) a critical client with no bound schema
            gets a dedicated schema generated + activated without pausing.

        canonical_guard: Optional context-manager factory serializing shared canonical
            schema setup across inbox workers. Released before document extraction.

    Returns:
        A dict with:
            client_key: The normalized client key.
            canonical:  ``{"header": {...}, "line_items": [...]}``.
            raw_extraction: ``{"schema_fields", "header", "line_items"}`` with every
                        Document AI column as printed, index-aligned with canonical.
            verify:     Serializable confidence summary (confidence, needs_review,
                        issues, warnings) for surfacing in the UI.

    Raises:
        ValueError: If schema selection cannot produce a ready schema (e.g. a
            critical client whose document yields no sampleable text while
            ``assume_yes`` is False).
    """
    client_key = normalize_client_key(client) if client else None

    # With assume_yes=True, select_schema creates+binds a dedicated schema for a
    # critical unbound client and returns a "ready" selection directly; a
    # non-critical client falls back to the canonical schema. No separate
    # create_dedicated_schema call is needed.
    selection = select_schema(
        engine,
        dox,
        client,
        file_path,
        dox_client_id=settings.dox_client_id,
        assume_yes=assume_yes,
        model=settings.mapper_model,
        canonical_guard=canonical_guard,
    ) if client else None
    if selection is None:
        from .schema_select import ensure_canonical_schema, SchemaSelection
        schema_id, version = ensure_canonical_schema(dox, dox_client_id=settings.dox_client_id, canonical_guard=canonical_guard)
        selection = SchemaSelection("ready", schema_id, version, True, "canonical")
    if selection.status != "ready":
        # Only reachable when assume_yes is False; the HTTP path always passes True.
        raise ValueError(
            f"schema for {client_key!r} is not ready (status={selection.status!r}); "
            "cannot extract without confirming the proposed schema"
        )

    result = pipeline.run(
        file_path,
        client_key or "unconfirmed",
        selection,
        dox_client=dox,
        settings=settings,
        **({"out_dir": out_dir} if out_dir else {}),
    )

    return {
        "client_key": client_key,
        "canonical": result.canonical,
        "raw_extraction": result.raw_extraction,
        "verify": {
            "confidence": result.verify.confidence,
            "needs_review": result.verify.needs_review,
            "issues": result.verify.issues,
            "warnings": result.verify.warnings,
        },
    }
