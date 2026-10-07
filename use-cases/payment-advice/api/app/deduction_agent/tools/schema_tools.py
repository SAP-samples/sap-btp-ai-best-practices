"""Document AI schema tools for the chat agents.

The Rules chat gets them for any registered customer; the Advice chat gets them
bound to the advice's own customer (``bound_client``). All business logic lives
in ``payment_advice.schema_admin``; this module adds what a conversation needs:

- **Plans by id.** Chat memory keeps only text, never tool results, and an
  LLM proposal cannot be regenerated identically, so every plan is kept
  server-side under a short ``plan_id`` that the user confirms later.
- **Explicit requests for writes.** ``prepare``, ``publish`` and ``discard``
  check the raw latest user message (never document text): a matching verb, and
  in the Rules chat the customer (name, key or plan id). ``publish`` and
  ``discard`` also need the version number in the message, so the model cannot
  pick a version on its own.
- **Lazy dependencies.** The Document AI client and settings are created after
  the chat runtime at startup, so they are read through getters at call time.

Example (wiring):
    tools = build_schema_tools(engine, lambda: app.state.dox, lambda: app.state.settings,
                               current_request.get, get_bound_source_path)
"""
from __future__ import annotations

import re
import uuid
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool

from ...payment_advice import schema_admin as A
from ...payment_advice.customers import get_customer, normalize_client_key

WRITE_REQUEST = r"\b(prepare|create|apply|update|add|change|save|make|generate|build|prepara|crea|añade|cambia|genera|aplica)\b"
PUBLISH_REQUEST = r"\b(publish|activate|go live|apply|confirm|deploy|publica|activa|confirma)\b"
DISCARD_REQUEST = r"\b(discard|delete|drop|cancel|descarta|elimina|borra)\b"

# ponytail: in-process plan registry (like the chat uploads in chat_session): single API
# process, plans are lost on restart and the user plans again. Move to HANA if the API scales out.
_PLANS: OrderedDict[str, A.SchemaPlan] = OrderedDict()
_MAX_PLANS = 100


def _remember(plan: A.SchemaPlan) -> str:
    """Store a plan and return its id (oldest plans are evicted)."""
    plan_id = uuid.uuid4().hex[:8]
    _PLANS[plan_id] = plan
    while len(_PLANS) > _MAX_PLANS:
        _PLANS.popitem(last=False)
    return plan_id


def _summary(plan_id: str, plan: A.SchemaPlan) -> dict[str, Any]:
    """Compact plan view for the model: the diff plus the resulting field list."""
    fields = lambda items: [{"name": f["name"], "label": f.get("label"), "type": f.get("formattingType", "string")}
                            for f in items]
    return {"plan_id": plan_id, "client_key": plan.client_key, "base_version": plan.base_version,
            "diff": plan.diff, "header_fields": fields(plan.header), "line_fields": fields(plan.line),
            "next": "show the diff as a table; prepare_schema_version(plan_id) after an explicit request"}


def build_schema_tools(engine: Any, get_dox: Callable[[], Any], get_settings: Callable[[], Any],
                       get_request: Callable[[], str], resolve_sample: Callable[[str], str | Path],
                       bound_client: str | None = None) -> list[BaseTool]:
    """Build the schema tools for one chat.

    Args:
        engine: HANA engine (customers and schema bindings).
        get_dox: Returns the Document AI client (None when unavailable).
        get_settings: Returns ``PaymentAdviceSettings`` (Document AI client id, mapper model).
        get_request: Returns the raw latest user message of the current turn.
        resolve_sample: Maps a sample name to a local file path (uploaded file, or the advice's attachment).
        bound_client: Restrict every tool to this customer (Advice chat).

    Returns:
        LangChain tools: describe, plan, propose, prepare, test, publish, discard.
    """

    def services() -> tuple[Any, Any]:
        """Document AI client and settings, or a clear error when the service is not configured."""
        dox, settings = get_dox(), get_settings()
        if dox is None or settings is None:
            raise ValueError("Document AI is not available on this server; schemas cannot be read or changed")
        return dox, settings

    def customer_key(client_key: str) -> str:
        """Resolve the customer, enforcing the Advice chat's bound customer."""
        if bound_client:
            if client_key and normalize_client_key(client_key) != bound_client:
                raise ValueError(f"This chat can only work on the schema of {bound_client!r}")
            return bound_client
        if not client_key:
            raise ValueError("Name the customer (client key)")
        return normalize_client_key(client_key)

    def require_request(pattern: str, client_key: str, *, version: str | None = None, plan_id: str | None = None) -> None:
        """Refuse a write unless the latest user message explicitly asks for it."""
        request = get_request() or ""
        if not re.search(pattern, request, re.I):
            raise ValueError("An explicit request in the latest user message is required for this change")
        if version is not None and not re.search(rf"(?<!\d){re.escape(str(version))}(?!\d)", request):
            raise ValueError(f"The latest user message must name version {version}")
        if bound_client is None:
            customer = get_customer(engine, client_key)
            names = {client_key.casefold(), (customer.display_name if customer else client_key).casefold(),
                     (plan_id or "\0").casefold()}
            if not any(name in request.casefold() for name in names):
                raise ValueError("The latest user message must name the customer")

    def plan_view(plan: A.SchemaPlan) -> dict[str, Any]:
        """Remember a plan and return its summary."""
        return _summary(_remember(plan), plan)

    @tool
    def describe_extraction_schema(client_key: str = "") -> dict:
        """Show the Document AI schema a customer's advices are extracted with: fields, version, versions, editable."""
        dox, settings = services()
        return A.describe(engine, dox, customer_key(client_key), settings.dox_client_id)

    @tool
    def plan_schema_changes(client_key: str = "", add: list[dict] | None = None, update: list[dict] | None = None,
                            remove: list[dict] | None = None) -> dict:
        """Plan explicit field changes to a customer's dedicated schema (read-only; returns plan_id + diff).

        add: [{name, scope: "header"|"line", label?, description?, type?}]; update: [{name, scope?, label?,
        description?, type?}]; remove: [{name, scope?}]. scope is needed when a name exists in header and line.
        Types: string, number, date, currency, discount.
        """
        dox, settings = services()
        current = A.current_fields(engine, dox, customer_key(client_key), settings.dox_client_id)
        return plan_view(A.plan_changes(current, add=add, update=update, remove=remove))

    @tool
    def propose_schema_from_sample(sample_name: str, client_key: str = "", merge: bool = True) -> dict:
        """Propose schema fields from a sample document (read-only; returns plan_id + diff).

        merge=True keeps all current fields and only adds new ones; merge=False replaces the field set.
        """
        dox, settings = services()
        current = A.current_fields(engine, dox, customer_key(client_key), settings.dox_client_id)
        return plan_view(A.propose_from_sample(current, resolve_sample(sample_name), merge=merge,
                                               model=settings.mapper_model))

    @tool
    def prepare_schema_version(plan_id: str) -> dict:
        """Create a new ACTIVE but unpublished schema version from a plan. Advices keep the published version."""
        plan = _PLANS.get(plan_id)
        if plan is None:
            raise ValueError(f"Unknown or expired plan {plan_id!r}; plan the changes again")
        key = customer_key(plan.client_key)
        require_request(WRITE_REQUEST, key, plan_id=plan_id)
        dox, settings = services()
        prepared = A.prepare(engine, dox, plan, settings.dox_client_id)
        return {**prepared, "client_key": key,
                "next": "optionally test_schema_version on a sample, then publish_schema_version on an explicit request"}

    @tool
    def test_schema_version(version: str, sample_name: str, client_key: str = "") -> dict:
        """Extract a sample with a schema version and preview the values and canonical header. Writes nothing."""
        dox, settings = services()
        return A.test_extract(engine, dox, customer_key(client_key), version, resolve_sample(sample_name),
                              settings.dox_client_id, model=settings.mapper_model)

    @tool
    def publish_schema_version(version: str, client_key: str = "") -> dict:
        """Publish a version: all future advices of the customer use it; the previous version is retired."""
        key = customer_key(client_key)
        require_request(PUBLISH_REQUEST, key, version=version)
        dox, settings = services()
        return A.publish(engine, dox, key, version, settings.dox_client_id)

    @tool
    def discard_schema_version(version: str, client_key: str = "") -> dict:
        """Deactivate a prepared version that will not be published (never the published one)."""
        key = customer_key(client_key)
        require_request(DISCARD_REQUEST, key, version=version)
        dox, settings = services()
        return A.discard(engine, dox, key, version, settings.dox_client_id)

    return [describe_extraction_schema, plan_schema_changes, propose_schema_from_sample, prepare_schema_version,
            test_schema_version, publish_schema_version, discard_schema_version]
