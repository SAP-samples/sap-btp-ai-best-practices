"""Shared selective interpretation stream for HTTP and durable inbox processing."""
import json
import logging
from typing import Any, AsyncIterator
from ..deduction_agent.interpretation_schema import INTERPRETATION_SCHEMA
logger = logging.getLogger(__name__)

async def interpret_events(client: str, payload: dict[str, Any], *, runtime: Any, engine: Any,
                           playbook_revision: int | None = None, raw_extraction: dict[str, Any] | None = None,
                           anchors: dict[str, Any] | None = None) -> AsyncIterator[str]:
    """Generate NDJSON lines for the rules-first, selective-analysis interpretation run.

    Shared by the HTTP endpoint and the inbox worker:
    1. The customer's documented column mappings replace the LLM mapper's choices
       (``rule_engine.apply_documented_mappings``, e.g. invoice number = "Your
       Invoice No"); then the documented rules (playbook anchors) are evaluated
       deterministically on the raw extraction (``rule_engine.apply_rules``):
       customer account per line, payer, company code, and reason codes.
    2. ``rule_engine.route_lines`` settles deduction lines a specific rule covers;
       only the rest go to the agent (deduction lines with no specific rule, and
       positive lines that match a deduction rule), each with a ``rule_hint`` and
       its raw ``source_columns``. Plain invoice lines never reach the LLM.
    3. Emits an analysis envelope reporting which lines go to the agent and how
       many were settled by rules or passed through.
    4. If no line needs the agent, the runtime is skipped. Otherwise ONLY the
       routed subset is sent, and the full index-aligned list is reassembled.
    5. Emits a tools envelope, then enriches the payload, writes the rule-derived
       fields and emits the result envelope.

    Yields one JSON object per line:
        {"type": "analysis", "analyzed_refs": [...], "analyzed_indexes": [...],
         "passthrough_count": n, "rule_resolved_count": n, "rule_warnings": [...]}
        {"type": "tools",    "tools_called": [...]}
        {"type": "result",   "enriched": {...}, "unresolved_count": n}
    or on failure:
        {"type": "error",    "message": "<error description>"}

    The agent runtime is read from ``request.app.state.runtime`` which is
    set during server lifespan startup.  If the runtime is unavailable a
    single error line is emitted and the stream ends.

    Args:
        client:  Client identifier passed from the request body.
        payload: Canonical payment advice payload.
        runtime: Agent runtime with document and rule lookup tools.
        engine: Existing HANA engine for legacy revision/rule lookup.
        playbook_revision: Frozen rule revision for inbox runs; None keeps legacy lookup.
        raw_extraction: Raw Document AI columns aligned with ``payload["line_items"]``;
            None evaluates rules on canonical fields only.
        anchors: Frozen playbook anchors for inbox runs; None loads them with the
            legacy lookup (or applies no rules offline).
    """
    from ..deduction_agent.document_context import (
        reset_current_document,
        set_current_document,
    )
    from ..deduction_agent.trace import extract_tool_calls
    from ..payment_advice.customers import normalize_client_key
    from ..payment_advice.deductions import (
        assemble_interpretations,
        enrich_payload,
    )
    from ..payment_advice.rule_engine import apply_documented_mappings, apply_rules, merge_rule_outcome, route_lines


    client_key = normalize_client_key(client)

    # Bind the payload to the ContextVar so full-document tools can access it
    # server-side without requiring the payload as an explicit argument.  The
    # token is reset in the finally block so concurrent async-generator tasks
    # each see only their own document.
    token = set_current_document(payload)
    try:
        # ------------------------------------------------------------------
        # Playbook revision lookup for provenance (0 in offline/no-engine mode).
        # ------------------------------------------------------------------
        revision = playbook_revision or 0
        if playbook_revision is None and engine is not None:
            from ..payment_advice.deduction_rules import get_playbook
            pb = get_playbook(engine, client_key)
            if pb is not None:
                revision = pb.revision
                anchors = pb.anchors if anchors is None else anchors

        # ------------------------------------------------------------------
        # Step 0: documented customer rules, deterministically, on raw columns.
        # Documented column mappings first (they override the LLM mapper), so
        # rules, agent, review and S/4 all see e.g. "Your Invoice No" as the
        # invoice reference; then the account/company/reason rules.
        # ------------------------------------------------------------------
        payload, documented, mapping_warnings = apply_documented_mappings(payload, raw_extraction, anchors)
        set_current_document(payload)  # rebind; the finally block still restores the original context
        outcome = apply_rules(payload, raw_extraction, anchors)
        outcome["warnings"] = mapping_warnings + outcome["warnings"]
        outcome["documented_mappings"] = documented
        routing = route_lines(payload, outcome)
        raw_lines = (raw_extraction or {}).get("line_items") or []

        # ------------------------------------------------------------------
        # Step 1: identify lines that require agent analysis.
        # Plain positive invoice lines are excluded from the LLM call entirely.
        # Each dict in `analysis` carries a `line_index` field pointing back to
        # its original position in payload["line_items"].
        # This happens BEFORE the runtime guard so that a clean payload (zero
        # anomalous lines) can return a correct result even when no runtime is
        # available — the LLM is only needed for the anomalous branch.
        # ------------------------------------------------------------------
        line_items = payload.get("line_items", [])
        analysis = [
            {"line_index": i, **line_items[i], "rule_hint": routing["hints"][i],
             **({"source_columns": raw_lines[i]} if i < len(raw_lines) else {})}
            for i in routing["agent_indexes"]
        ]
        total_lines = len(line_items)
        rule_resolved = len(routing["engine_entries"])
        passthrough_count = total_lines - len(analysis) - rule_resolved

        # Emit an analysis envelope so the caller can see which lines were selected.
        # Fall back to the line_index string when invoice_reference is absent so
        # every entry in analyzed_refs is a non-null string.
        analyzed_refs = [
            line.get("invoice_reference") or str(line.get("line_index"))
            for line in analysis
        ]
        yield json.dumps({
            "type": "analysis",
            "analyzed_refs": analyzed_refs,
            "analyzed_indexes": routing["agent_indexes"],
            "passthrough_count": passthrough_count,
            "rule_resolved_count": rule_resolved,
            "rule_warnings": outcome["warnings"],
        }) + "\n"

        # ------------------------------------------------------------------
        # Step 2: LLM call (only when analysis lines exist).
        # Prompt carries the subset only — NOT the whole payload.
        # The runtime guard lives here, inside the anomalous branch, so a clean
        # document streams a proper result even on a runtime-less deployment.
        # ------------------------------------------------------------------
        if not analysis:
            # No anomalous lines: skip the runtime entirely.
            full = assemble_interpretations(payload, [], routing["engine_entries"], set())
            tools_called: list[Any] = []
        else:
            # Require the runtime only when there is actually something to interpret.
            if runtime is None:
                yield json.dumps({"type": "error", "message": "interpretation runtime not available"}) + "\n"
                return

            # Build the subset prompt with only the anomalous lines (each entry
            # already carries line_index so the model can echo it back).
            prompt_text = (
                f"Client: {client_key}\n"
                "Analyze the following payment-advice lines that require interpretation. "
                "Lines covered by a specific documented customer rule were already settled "
                "deterministically and are not included. Each line carries a rule_hint and, "
                "when available, its source_columns as printed on the remittance. "
                "Return EXACTLY ONE interpretation entry per provided line, "
                "echoing each line's line_index in your response.\n\n"
                f"{json.dumps(analysis)}"
            )

            result = await runtime.ainvoke(
                text=prompt_text,
                context_id=client_key,
                response_model=INTERPRETATION_SCHEMA,
            )

            # output_parsed is {"interpretations": [...]} from structured output.
            # Guard against None, non-dict, or missing key before consuming.
            parsed = result.output_parsed or {}
            agent_entries: list[Any] = (
                parsed.get("interpretations", [])
                if isinstance(parsed, dict)
                else []
            )

            # Reassemble the full index-aligned interpretation list using
            # the line_index keys the agent echoed back.
            full = assemble_interpretations(payload, agent_entries, routing["engine_entries"],
                                            set(routing["agent_indexes"]))
            tools_called = extract_tool_calls(getattr(result, "messages", []) or [])

        # Emit the tools envelope.
        yield json.dumps({"type": "tools", "tools_called": tools_called}) + "\n"

        # ------------------------------------------------------------------
        # Step 3: enrich the payload and emit the result envelope.
        # Use safe chained .get() for unresolved_count so a missing key does not
        # raise a mid-stream KeyError after earlier envelopes were already sent.
        # ------------------------------------------------------------------
        enriched = enrich_payload(
            payload,
            full,
            client_key=client_key,
            playbook_revision=revision,
        )
        merge_rule_outcome(enriched, outcome, rule_resolved)
        unresolved_count = (
            enriched.get("header", {})
            .get("interpretation", {})
            .get("unresolved_count", 0)
        )
        yield json.dumps({
            "type": "result",
            "enriched": enriched,
            "unresolved_count": unresolved_count,
        }) + "\n"

    except Exception as exc:
        # Log the full traceback server-side; send only the message to the caller
        # to avoid leaking internal stack details over the network.
        logger.exception("interpretation error for client %s: %s", client_key, exc)
        yield json.dumps({"type": "error", "message": str(exc)}) + "\n"
    finally:
        # Always restore the previous ContextVar binding so concurrent or
        # sequential requests sharing the same event-loop task cannot see a
        # stale document from an earlier invocation.
        reset_current_document(token)
