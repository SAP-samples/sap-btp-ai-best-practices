"""Shared output budget and ECN targeting for the chat tools.

Every tool result is fed back into the model context, so each one is capped
at ``TOOL_BUDGET`` characters (``_bounded``). ``_for_ecns`` runs one tool body
for the ECNs a call names (or the open ECN) and splits that budget when
several ECNs are read.

Example::

    async def run(row_id):
        return {"assessment_id": row_id, "items": [...]}

    return await _for_ecns(catalogue, ["ECN306072", "ECN304940"], run)
"""

import json
from dataclasses import asdict

from app.ecn.assistant.catalogue import EcnCatalogue
from app.ecn.rdf import GraphExpansionPage, GraphFact, GraphFactPage

TOOL_BUDGET = 24000
"""Most characters one tool result may add to the model context."""

MAX_DETAIL_ECNS = 5
"""Most ECNs a detail tool reads per call; they share one ``TOOL_BUDGET``."""


def _bounded(value, budget: int = TOOL_BUDGET):
    """Limit long source text while retaining exact totals and labeling omitted detail.

    Args:
        value: Tool result dictionary.
        budget: Most serialized characters allowed before detail is omitted.

    Returns:
        ``{"text_truncated", "result"}``, or totals and pagination only with
        ``detail_omitted`` when the result exceeds ``budget``.
    """
    truncated = False

    def visit(item):
        """Recursively bound free text without changing count fields or source identifiers."""
        nonlocal truncated
        if isinstance(item, str) and len(item) > 3000:
            truncated = True
            return item[:3000] + " [text truncated]"
        if isinstance(item, dict):
            return {k: visit(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [visit(v) for v in item]
        return item

    result = visit(value)
    encoded = json.dumps(result, ensure_ascii=False, default=str)
    if len(encoded) > budget:
        return {
            "assessment_id": value.get("assessment_id"),
            "detail_omitted": True,
            "message": "This result exceeds the tool budget. Narrow the query or inspect one object.",
            "total": value.get("total", value.get("location_total")),
            "counts": value.get("counts"),
            "page": value.get("page"),
            "page_size": value.get("page_size"),
            "total_matches": value.get("total_matches"),
            "has_next": value.get("has_next"),
        }
    return {"text_truncated": truncated, "result": result}


def _graph_page_response(page, numbers: dict[str, str] | None = None) -> dict:
    """Serialize a fixed graph result page before applying the shared text cap.

    Args:
        page: A graph fact, expansion or field page.
        numbers: ``assessment_id -> ECN number`` for multi-ECN pages; each
            fact then names its ``ecn_number``.

    Returns:
        The page as a dict without internal ECN identities.
    """
    result = asdict(page)
    for item in result["items"]:
        owner = item.pop("ecn_row_id", None)
        if numbers and owner:
            item["ecn_number"] = numbers.get(owner, owner)
        if item.get("ecn_count", 0) is None:
            item.pop("ecn_count")
    return result


async def _for_ecns(catalogue: EcnCatalogue, ecn, run, *, limit: int | None = MAX_DETAIL_ECNS,
                    compact=None) -> dict:
    """Run one tool body for the ECNs a call names, or the open ECN, and shape the output.

    Args:
        catalogue: Per-turn ``EcnCatalogue``.
        ecn: ECN numbers from the tool call (string or list; empty = open ECN).
        run: ``async (assessment_id) -> dict`` returning one ECN's raw result.
        limit: Most ECNs per call, or ``None`` for no cap.
        compact: Optional reducer applied to each result when several ECNs are read.

    Returns:
        One ECN: its bounded result, plus ``ecn_number`` when it was named.
        Several: ``{"results": [{"ecn_number", ...bounded result}]}`` with the
        budget split evenly; an ECN that cannot be read reports its error
        code instead of hiding the others. An error result when ``ecn``
        cannot be resolved.
    """
    targets, problem = await catalogue.targets(ecn, limit)
    if problem:
        return _bounded(problem)
    catalogue.note(targets)
    if len(targets) == 1:
        target = targets[0]
        result = await run(target["assessment_id"])
        return _bounded(result if target["ecn_number"] is None else {"ecn_number": target["ecn_number"], **result})
    results = []
    for target in targets:
        try:
            result = await run(target["assessment_id"])
            result = compact(result) if compact else result
        except Exception as error:  # noqa: BLE001 - one unreadable ECN must not hide the others.
            result = {"error_code": str(getattr(error, "code", type(error).__name__))[:80],
                      "message": "This ECN could not be read; no data was changed."}
        results.append({"ecn_number": target["ecn_number"], **_bounded(result, TOOL_BUDGET // len(targets))})
    return {"results": results}


def _capture_graph_facts(
    page: GraphFactPage | GraphExpansionPage,
    captured_facts: list[GraphFact],
    issued_graph_nodes: set[str],
) -> None:
    """Keep exact source facts and trusted node references scoped to one chat turn."""
    captured_facts.extend(page.items)
    for fact in page.items:
        issued_graph_nodes.add(fact.subject_ref)
        if fact.object_ref is not None:
            issued_graph_nodes.add(fact.object_ref)
