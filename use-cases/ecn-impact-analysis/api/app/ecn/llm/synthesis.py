"""Bounded synthesis input with complete aggregates and fairly sampled evidence."""

import json
from collections import defaultdict
from dataclasses import replace
from itertools import zip_longest

from app.ecn.assistant.discovery import DiscoveryResult
from app.ecn.evidence.business import business_evidence_records
from app.ecn.evidence.material_changes import change_summary
from app.ecn.llm.bom_context import bounded_bom_context, trim_bom_example
from app.ecn.llm.two_pass import (
    FINAL_MAX_CHARACTERS,
    FINAL_MAX_RECORDS,
    EvidencePlan,
    EvidenceSlice,
    TwoPassProjectionError,
    _build_final_projection,
    build_final_input,
)


def build_synthesis_projection(
    bundle,
    snapshot=None,
    discovery: DiscoveryResult | None = None,
):
    """Keep complete counts and round-robin evidence categories within a hard message cap."""
    groups = defaultdict(list)
    for record in business_evidence_records(bundle):
        groups[record.section].append(record)
    selected = tuple(
        record
        for group in zip_longest(*(groups[key] for key in sorted(groups)))
        for record in group
        if record is not None
    )
    plan = EvidencePlan(requested_slices=list(EvidenceSlice), focus_material_numbers=[])
    context = {}
    if snapshot is not None:
        context = {k: v for k, v in snapshot.items() if k not in {"objects", "changes", "programs", "parts", "reference_documents"}}
        context.update(bounded_bom_context(snapshot))
        # Full subjects remain in the UI/object API; model examples must respect
        # the same bounded budget as other source evidence.
        context["questions"] = [{**q, "subjects": [{**s, "details": s.get("details", [])[:3]}
            for s in q.get("subjects", [])[:25]]} for q in snapshot.get("questions", [])]
        context["material_change_total"] = len(snapshot.get("changes", []))
        context["material_changes"] = [change_summary(c) for c in snapshot.get("changes", [])[:25]]
        # Summaries expose representative references, while UI/tools retain every row.
        examples = []
        for obj in sorted(
            snapshot["objects"],
            key=lambda o: (
                not bool(o["reference_records"]),
                not bool(o["locations"]),
                o["number"] or "",
            ),
        )[:25]:
            examples.append(
                {
                    k: v
                    for k, v in obj.items()
                    if k not in {"locations", "reference_records", "source_references", "change"}
                }
                | {
                    "locations": obj["locations"][:2],
                    "reference_records": obj["reference_records"][:3],
                    "location_total": len(obj["locations"]),
                    "reference_total": len(obj["reference_records"]),
                }
            )
        context["illustrative_objects"] = examples
        context["illustrative_object_count"] = len(examples)
    included = selected[:FINAL_MAX_RECORDS]
    while True:
        projection = replace(
            _build_final_projection(bundle, plan, selected, included),
            snapshot_context=context,
        )
        if discovery is not None:
            allowed = {
                section: frozenset(
                    set(citation_ids)
                    | {fact.evidence_id for fact in discovery.facts}
                )
                for section, citation_ids in (
                    projection.allowed_evidence_ids_by_section.items()
                )
            }
            projection = replace(
                projection,
                allowed_evidence_ids_by_section=allowed,
                discovery_status=discovery.status.value,
                discovery_error_code=discovery.error_code,
                discovery_hash=discovery.discovery_hash,
                discovered_facts=discovery.facts,
            )
        if (
            len(json.dumps(build_final_input(projection), ensure_ascii=False))
            <= FINAL_MAX_CHARACTERS
        ):
            return projection
        if included:
            included = included[:-1]
        elif context.get("illustrative_objects"):
            context["illustrative_objects"].pop()
            context["illustrative_object_count"] = len(context["illustrative_objects"])
        elif context.get("material_changes"):
            context["material_changes"].pop()
        elif trim_bom_example(context):
            continue
        elif any(q.get("subjects") for q in context.get("questions", [])):
            next(q for q in reversed(context["questions"]) if q.get("subjects"))["subjects"].pop()
        else:
            raise TwoPassProjectionError("SYNTHESIS_INPUT_LIMIT_TOO_SMALL")
