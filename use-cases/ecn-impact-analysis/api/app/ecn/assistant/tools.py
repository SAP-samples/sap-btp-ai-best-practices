"""Bounded LangChain tools for the ECN chat.

Every read tool takes an optional ``ecn`` argument (ECN numbers). Empty means
the ECN open on the detail page; on other pages it names the ECNs to read.
``EcnCatalogue`` resolves the numbers server-side, so the model never supplies
an internal identity, SQL, SPARQL or file path. Review tools act only on the
open ECN.
"""

import json
import re
from typing import Annotated, Literal
from uuid import NAMESPACE_URL, uuid5

from langchain_core.tools import tool
from pydantic import Field

from app.ecn.assistant.catalogue import EcnCatalogue
from app.ecn.assistant.graph_tools import create_graph_tools, create_replacement_tools
from app.ecn.assistant.tool_scope import _bounded, _for_ecns
from app.ecn.evidence.business import (
    BusinessEvidenceSection,
    paginate_business_evidence,
)
from app.ecn.evidence.acceptance_plan import build_acceptance_plan
from app.ecn.evidence.material_changes import change_summary
from app.ecn.llm.bom_context import bounded_bom_context, trim_bom_example
from app.ecn.errors import PersistenceStateError
from app.ecn.services.blocking_io import run_blocking
from app.ecn.services.reviews import record_review
from app.ecn.rdf import GraphFact
from app.persistence.codec import business_evidence_page_response


def compact_impact(result: dict) -> dict:
    """Keep one ECN's overall impact and factor severities for a multi-ECN comparison."""
    keys = ("available", "is_current", "overall_impact", "completeness", "message")
    return {**{key: result[key] for key in keys if key in result},
            "impact_factors": {name: factor.get("severity")
                               for name, factor in (result.get("impact_factors") or {}).items()}}


def compact_review(result: dict) -> dict:
    """Keep one ECN's decision and acceptance block for a multi-ECN comparison."""
    acceptance = result.get("acceptance") or {}
    return {"current_decision": result.get("current_decision"), "latest": result.get("latest"),
            "blocked": acceptance.get("blocked"), "block_reason": acceptance.get("block_reason")}


def _timestamp(value) -> str | None:
    """Render a stored timestamp as ISO text so tool output stays JSON-serializable."""
    return value.isoformat() if hasattr(value, "isoformat") else value


def review_status(reviews: list[dict], overview: dict) -> dict:
    """Summarize the append-only engineering review history of one assessment.

    Args:
        reviews: Rows from ``WorkspaceRepository.reviews``, oldest first, with
            lowercase keys (reviewer, outcome, comments, created_at,
            evidence_hash, snapshot_version).
        overview: Current snapshot overview; its ``evidence_hash`` and
            ``version`` identify the evidence the user sees now.

    Returns:
        ``current_decision`` (the newest outcome, or ``NONE``), the newest
        review in ``latest``, up to five reviews newest first in ``history``,
        and ``total``. Each review states whether it was made on the current
        snapshot (``applies_to_current_snapshot``).
    """

    def summary(row: dict) -> dict:
        """Keep the fields a planner needs; the actor hash is never read."""
        return {
            "reviewer": row.get("reviewer"),
            "outcome": row.get("outcome"),
            "comments": (row.get("comments") or "")[:500],
            "created_at": _timestamp(row.get("created_at")),
            "applies_to_current_snapshot": row.get("evidence_hash") == overview.get("evidence_hash")
            and row.get("snapshot_version") == overview.get("version"),
        }

    latest = reviews[-1] if reviews else None
    return {
        "current_decision": latest["outcome"] if latest else "NONE",
        "latest": summary(latest) if latest else None,
        "history": [summary(row) for row in reversed(reviews[-5:])],
        "total": len(reviews),
    }


def acceptance_summary(plan: dict) -> dict:
    """Condense the acceptance plan for the chat tool budget.

    Args:
        plan: Output of ``build_acceptance_plan`` (parts, missing prices, block).

    Returns:
        ``blocked`` and ``block_reason`` (whether Accept is possible), at most
        ten parts with their material master change and price state, their
        total, and the new parts with no price. The SAP write is simulated.
    """
    return {
        "blocked": plan["blocked"],
        "block_reason": plan["block_reason"],
        "parts": plan["parts"][:10],
        "parts_total": len(plan["parts"]),
        "missing_price": plan["missing_price"][:10],
        "missing_price_total": len(plan["missing_price"]),
        "sap_write": "simulated",
    }


def analysis_summary(analysis: dict, current_evidence_hash: str | None) -> dict:
    """Condense a stored AI impact analysis for the chat tool budget.

    Args:
        analysis: ``StoredAnalysisView.model_dump(mode="json")``.
        current_evidence_hash: Evidence hash of the snapshot shown now.

    Returns:
        Overall impact, completeness, per-factor severity and rationale, each
        section's status and (trimmed) assessment, next steps, the model used,
        and ``is_current`` (false when the analysis predates the evidence).
    """
    return {
        "available": True,
        "is_current": analysis.get("evidence_hash") == current_evidence_hash,
        "overall_impact": analysis.get("overall_impact"),
        "completeness": analysis.get("completeness"),
        "model_name": analysis.get("model_name"),
        "created_at": analysis.get("created_at"),
        "impact_factors": {
            name: {"severity": factor.get("severity"), "assessment": factor.get("assessment")}
            for name, factor in (analysis.get("impact_factors") or {}).items()
        },
        "sections": {
            name: {"status": section.get("status"), "assessment": (section.get("assessment_markdown") or "")[:1500]}
            for name, section in (analysis.get("sections") or {}).items()
        },
        "next_steps": analysis.get("additional_next_steps") or {},
    }


def readiness_question(question: dict) -> dict:
    """Keep one readiness question with the exact part numbers it concerns.

    Args:
        question: One ``snapshot["questions"]`` entry (code, question, why,
            resolution, affected_count, subjects[{number, ...}]).

    Returns:
        The question text fields plus ``parts``: up to ten distinct subject
        numbers, so answers can name the parts instead of counting them.
    """
    parts = list(dict.fromkeys(s.get("number") for s in question.get("subjects") or [] if s.get("number")))
    return {
        **{key: question.get(key) for key in ("code", "question", "why", "resolution", "affected_count")},
        "parts": parts[:10],
    }


ReviewOutcome = Literal["ACCEPTED", "CHANGES_REQUESTED"]


def review_problem(outcome: str, reviewer: str, comments: str) -> dict | None:
    """Apply the review form's rules before a draft is shown or saved.

    Args:
        outcome: ``ACCEPTED`` or ``CHANGES_REQUESTED``.
        reviewer: Reviewer name given by the user.
        comments: Review comment written from the user's remarks.

    Returns:
        ``None`` when valid, otherwise ``{"error_code", "message"}`` telling
        the model what to ask the user for.
    """
    if not reviewer.strip():
        return {"error_code": "REVIEWER_REQUIRED", "message": "Ask the user for the reviewer name."}
    if outcome == "CHANGES_REQUESTED" and not comments.strip():
        return {"error_code": "COMMENTS_REQUIRED", "message": "Ask the user which changes are needed."}
    return None


def _shown_text(text: str) -> str:
    """Normalize Markdown quote markers and whitespace so a shown draft can be matched."""
    return " ".join(re.sub(r"(?m)^\s*>\s?", "", text).split())


def draft_was_shown(history, outcome: str, reviewer: str, comments: str) -> bool:
    """Check that the previous assistant turn showed this exact draft to the user.

    This is the server-side guard that stops the model from saving review text
    the user never saw. Whether the user's reply confirms it is judged by the
    model under the review-decision skill.

    Args:
        history: Browser-supplied turns ``[{role, content}]`` before this message.
        outcome / reviewer / comments: The review about to be saved.

    Returns:
        True when the latest assistant turn contains the comment text, the
        reviewer name and the decision wording ("accept", or "reject" /
        the older "return for epr", "send back" and "request" for a rejection).
    """
    previous = next((turn["content"] for turn in reversed(history) if turn.get("role") == "assistant"), "")
    shown = _shown_text(previous)
    wording = ("accept",) if outcome == "ACCEPTED" else ("reject", "return for epr", "send back", "request")
    return (_shown_text(comments) in shown and reviewer.strip() in shown
            and any(word in shown.casefold() for word in wording))


def create_tools(
    snapshots,
    ecns,
    rdf,
    ecn_row_id: str | None,
    consulted: set[tuple[str, bool]],
    captured_facts: list[GraphFact],
    issued_graph_nodes: set[str],
    workspace=None,
    history=(),
    actor: str = "",
    review_state: dict | None = None,
    catalogue: EcnCatalogue | None = None,
):
    """Return relational, graph and decision tools with no model-controlled identity.

    Args:
        snapshots: ``SnapshotService`` for the deterministic snapshot.
        ecns: ``EcnRepository`` for stored evidence and the stored AI analysis.
        rdf: RDF projection service for source-field tools.
        ecn_row_id: ECN open on the detail page, or ``None`` on other pages.
            Read tools default to it; review tools act only on it and are
            omitted when it is ``None``.
        consulted: Collects objects of the open ECN inspected during the turn.
        captured_facts / issued_graph_nodes: Per-turn graph tool state.
        workspace: ``WorkspaceRepository`` that stores engineering reviews;
            ``None`` means no review history is available and nothing can be saved.
        history: Prior browser turns, used to prove a review draft was shown.
        actor: Hashed caller identity stored with a saved review.
        review_state: Per-turn dict; ``submit_review_decision`` sets
            ``saved=True`` so the response can tell the UI to refresh.
        catalogue: Per-turn ``EcnCatalogue`` resolving the ``ecn`` argument;
            defaults to one that only knows the open ECN.

    Returns:
        The list of LangChain tools.
    """
    catalogue = catalogue or EcnCatalogue(ecns, ecn_row_id)

    @tool
    async def get_ecn_overview(ecn: list[str] = Field(default_factory=list, max_length=5)) -> dict:
        """Read intent, counts, datasets, BOM changes, plants, and open questions.

        Use this first for broad questions about what the ECN covers or what is
        unresolved. ``bom_changes`` lists what changes in each assembly
        (replaced, added, removed, quantity changed, with end/start dates and
        full phase-out / new-part flags); ``plant_impact`` lists plants.
        It returns the deterministic snapshot overview without the
        full object list. Do not use it to answer arbitrary source-field
        questions; use ``list_ecn_fields`` and ``find_ecn_facts`` instead.

        ``ecn``: up to five ECN numbers; leave empty for the open ECN
        (required when none is open). Several ECNs return ``results``.

        Example input: ``{}``.
        """

        async def run(row_id: str) -> dict:
            """Read one ECN's bounded overview."""
            overview = await run_blocking(snapshots.overview, row_id)
            # Complete UI arrays can exceed the tool budget and hide all useful facts.
            # Return totals/examples here; paged tools retrieve exact rows on demand.
            result = {k: v for k, v in overview.items() if k not in {"programs", "parts", "reference_documents"}}
            result.update(bounded_bom_context(overview))
            result["questions"] = [{k: v for k, v in q.items() if k not in {"subjects", "scopes"}}
                                   for q in overview.get("questions", [])]
            while len(json.dumps(result, default=str)) > 20000 and trim_bom_example(result):
                pass
            return result

        return await _for_ecns(catalogue, ecn, run)

    @tool
    async def list_bom_changes(number: str = "", page: Annotated[int, Field(ge=1)] = 1,
                               ecn: list[str] = Field(default_factory=list, max_length=5)) -> dict:
        """Read actual BOM replacements, removals, additions and quantity changes.

        Use FIRST for what changes, which plant, phase-out end dates, engineering
        release dates and manufacturing cut-over dates. These dates come from
        BOM rows, not ECN JSON fields. Optional number matches an exact old/new
        part or assembly; an empty number returns all changes, five per page.
        Output preserves sources, pairing qualifiers, separate EBOM/MBOM dates,
        plant/model, complete counts and lifecycle flags for the returned parts.
        Follow has_next for the remaining rows.

        ``ecn``: up to five ECN numbers to compare; leave empty for the open
        ECN (required when none is open). Several ECNs return ``results``.
        """
        number = number.strip()[:200]

        async def run(row_id: str) -> dict:
            """Page one ECN's BOM change lines."""
            snapshot = await run_blocking(snapshots.get, row_id)
            changes = snapshot.get("bom_changes") or {}
            rows = [line for line in changes.get("lines", []) if not number or number in
                    {line.get("assembly"), line.get("parent"),
                     (line.get("outgoing") or {}).get("number"), (line.get("incoming") or {}).get("number")}]
            items = rows[(page - 1) * 5:page * 5]
            numbers = {side["number"] for line in items for side in (line.get("outgoing"), line.get("incoming")) if side}
            return {"assessment_id": row_id, "assessment_date": snapshot.get("assessment_date"),
                    "counts": changes.get("counts", {}), "coverage_complete": changes.get("coverage_complete", False),
                    "total": len(rows), "page": page, "page_size": 5, "has_next": page * 5 < len(rows),
                    "items": items, "materials": [m for m in changes.get("materials", []) if m["number"] in numbers]}

        return await _for_ecns(catalogue, ecn, run)

    @tool
    async def list_referenced_objects(
        query: str = "",
        page: Annotated[int, Field(ge=1)] = 1,
        technical: bool = True,
        origin: Literal["ecn", "bom"] = "ecn",
        ecn: list[str] = Field(default_factory=list, max_length=5),
    ) -> dict:
        """Page ECN-referenced parts/assemblies, documents, or BOM-change rows.

        ``technical=true`` selects parts/assemblies; false selects supporting
        objects. ``origin="ecn"`` means explicitly referenced in the ECN JSON;
        ``origin="bom"`` means a relational BOM row carries the ECN number.
        ``query`` matches identifier, description, or type. Results contain ten
        objects per page with roles, levels, and occurrence counts; follow
        pagination when more rows exist.

        ``ecn``: up to five ECN numbers; leave empty for the open ECN
        (required when none is open). Several ECNs return ``results``.

        Example input: ``{"query": "bolt", "technical": true, "origin": "ecn", "page": 1}``.
        """

        async def run(row_id: str) -> dict:
            """Page one ECN's referenced objects."""
            return await run_blocking(
                snapshots.objects,
                row_id,
                query=query[:200],
                page=page,
                page_size=10,
                technical=technical,
                origin=origin,
            )

        return await _for_ecns(catalogue, ecn, run)

    @tool
    async def inspect_object(
        number: str, page: Annotated[int, Field(ge=1)] = 1, technical: bool = True,
        ecn: list[str] = Field(default_factory=list, max_length=5),
    ) -> dict:
        """Inspect one exact referenced object's evidence and source coordinates.

        Use a number or object key returned by ``list_referenced_objects``.
        ``technical`` must match that result. Output pages up to five observed
        pinned BOM locations, material/purchasing/stock rows, and ECN JSON source
        references, with independent totals for each collection. Missing rows
        mean unavailable evidence, never zero stock or no impact.

        ``ecn``: up to five ECN numbers to inspect the same object in (e.g. how
        one part is handled in two ECNs); leave empty for the open ECN.

        Example input: ``{"number": "255100200400", "technical": true, "page": 1}``.
        """

        async def run(row_id: str) -> dict:
            """Read one object's evidence in one ECN."""
            result = await run_blocking(
                snapshots.object_details,
                row_id,
                number[:200],
                page=page,
                page_size=5,
                technical=technical,
            )
            # Object buttons in the answer open the object on the current ECN page.
            if row_id == ecn_row_id:
                consulted.add((number, technical))
            if result.get("change"):
                result["change"] = change_summary(result["change"])
            return result

        return await _for_ecns(catalogue, ecn, run)

    @tool
    async def inspect_hierarchy(
        ancestor: str, page: Annotated[int, Field(ge=1)] = 1,
        ecn: Annotated[str, Field(max_length=80)] = "",
    ) -> dict:
        """Find referenced objects observed beneath one exact BOM ancestor.

        Use an ancestor identifier from an inspected occurrence path. The
        result pages referenced objects whose observed paths contain that
        ancestor; it does not mark other descendants as changed and remains
        limited to the supplied scoped population.

        ``ecn``: one ECN number; leave empty for the open ECN.

        Example input: ``{"ancestor": "VC-100", "page": 1}``.
        """

        async def run(row_id: str) -> dict:
            """Page one ECN's objects below the ancestor."""
            return await run_blocking(
                snapshots.objects,
                row_id,
                ancestor=ancestor[:200],
                page=page,
                page_size=10,
            )

        return await _for_ecns(catalogue, ecn, run, limit=1)

    @tool
    async def inspect_configuration(page: Annotated[int, Field(ge=1)] = 1,
                                    ecn: Annotated[str, Field(max_length=80)] = "") -> dict:
        """Read pinned configuration evidence and explicit limitations.

        Use for configuration applicability or conditional-rule questions, not
        arbitrary ECN source fields. Results are grouped five at a time and
        preserve source provenance. Unknown applicability remains unknown when
        required inputs or analysis dates are unavailable.

        ``ecn``: one ECN number; leave empty for the open ECN.

        Example input: ``{"page": 1}``.
        """

        async def run(row_id: str) -> dict:
            """Page one ECN's configuration evidence."""
            bundle = await run_blocking(ecns.get_evidence_bundle, row_id)
            projected = paginate_business_evidence(
                bundle, BusinessEvidenceSection.CONFIGURATION, page=page, page_size=5
            )
            return business_evidence_page_response(
                projected, include_technical=True
            ).model_dump(mode="json")

        return await _for_ecns(catalogue, ecn, run, limit=1)

    @tool
    async def list_material_changes(page: Annotated[int, Field(ge=1)] = 1,
                                    ecn: list[str] = Field(default_factory=list, max_length=5)) -> dict:
        """Page shared deterministic lifecycle, revision, pairing and scoped usage facts.

        Use this for ECN-declared revision intent; use list_bom_changes first for
        replacements and their actual start/end dates. Preserve pairing
        basis and qualifiers. Historical evidence is not current production use.

        ``ecn``: up to five ECN numbers; leave empty for the open ECN
        (required when none is open). Several ECNs return ``results``.
        """

        async def run(row_id: str) -> dict:
            """Page one ECN's material changes."""
            result = await run_blocking(snapshots.changes, row_id, page=page, page_size=5)
            result["items"] = [change_summary(change) for change in result["items"]]
            return result

        return await _for_ecns(catalogue, ecn, run)

    @tool
    async def get_review_status(ecn: list[str] = Field(default_factory=list, max_length=50)) -> dict:
        """Read whether an ECN is accepted, rejected, or unreviewed.

        Use FIRST for approval questions, "what should I do next?" and before
        drafting a review. ``current_decision`` is NONE when nobody has
        reviewed it; ``applies_to_current_snapshot`` is false when a decision
        was made on older evidence. ``acceptance`` lists each part's material
        master change and price state. The ECN cannot be accepted while
        ``blocked`` (a new part has no price); then it can only be
        rejected. The SAP write is simulated.

        ``ecn``: ECN numbers; leave empty for the open ECN (required when
        none is open). Several ECNs return compact ``results`` (decision,
        latest review, blocked); ask for one ECN to see its parts.

        Example input: ``{}``.
        """

        async def run(row_id: str) -> dict:
            """Read one ECN's review history and acceptance plan."""
            overview = await run_blocking(snapshots.overview, row_id)
            reviews = await run_blocking(workspace.reviews, row_id) if workspace else []
            return {
                "assessment_id": row_id,
                **review_status(reviews, overview),
                "acceptance": acceptance_summary(build_acceptance_plan(overview)),
            }

        return await _for_ecns(catalogue, ecn, run, limit=None, compact=compact_review)

    @tool
    async def get_impact_analysis(ecn: list[str] = Field(default_factory=list, max_length=50)) -> dict:
        """Read the stored AI impact analysis: overall impact, factors, sections, next steps.

        Use for "how big is the impact?", severity by manufacturing,
        procurement, configuration and data readiness, or the analysis
        summary. ``available`` is false when no analysis was generated yet.
        ``is_current`` is false when it was generated on older evidence; then
        prefer the BOM tools. Do not mention the analysis itself to the user.
        Use the BOM tools for exact parts and dates.

        ``ecn``: ECN numbers; leave empty for the open ECN (required when
        none is open). Several ECNs return compact ``results`` (overall
        impact and factor severities) for comparison.

        Example input: ``{}``.
        """

        async def run(row_id: str) -> dict:
            """Read one ECN's stored analysis."""
            detail = await run_blocking(ecns.get_ecn_detail, row_id)
            analysis = detail.analysis if detail is not None else None
            if analysis is None:
                return {"assessment_id": row_id, "available": False,
                        "message": "No AI impact analysis has been generated for this ECN yet."}
            overview = await run_blocking(snapshots.overview, row_id)
            return {"assessment_id": row_id,
                    **analysis_summary(analysis.model_dump(mode="json"), overview.get("evidence_hash"))}

        return await _for_ecns(catalogue, ecn, run, limit=None, compact=compact_impact)

    @tool
    async def get_data_quality_issues(page: Annotated[int, Field(ge=1)] = 1,
                                      ecn: list[str] = Field(default_factory=list, max_length=5)) -> dict:
        """Read what needs attention before an ECN can be trusted or accepted.

        Returns open readiness questions with the exact part numbers they
        concern (``parts``), plus stored data-quality warnings and missing
        evidence, five of each per page. Follow ``has_next`` for more.
        Missing data is unknown, never zero stock or no impact.

        ``ecn``: up to five ECN numbers; leave empty for the open ECN
        (required when none is open). Several ECNs return ``results``.

        Example input: ``{"page": 1}``.
        """

        async def run(row_id: str) -> dict:
            """Page one ECN's readiness questions and data-quality records."""
            overview = await run_blocking(snapshots.overview, row_id)
            questions = [readiness_question(q) for q in overview.get("questions", [])]
            start = (page - 1) * 5
            result = {"assessment_id": row_id, "page": page, "page_size": 5,
                      "readiness_total": len(questions), "readiness_questions": questions[start:start + 5]}
            has_next = start + 5 < len(questions)
            bundle = await run_blocking(ecns.get_evidence_bundle, row_id)
            if bundle is not None:
                for section in (BusinessEvidenceSection.DATA_QUALITY, BusinessEvidenceSection.MISSING_EVIDENCE):
                    records = business_evidence_page_response(
                        paginate_business_evidence(bundle, section, page=page, page_size=5),
                        include_technical=False,
                    ).model_dump(mode="json")
                    result[section.value] = records
                    has_next = has_next or page < records["total_pages"]
            result["has_next"] = has_next
            return result

        return await _for_ecns(catalogue, ecn, run)

    read_tools = [
        list_bom_changes,
        list_material_changes,
        get_ecn_overview,
        list_referenced_objects,
        inspect_object,
        inspect_hierarchy,
        inspect_configuration,
        get_review_status,
        get_impact_analysis,
        get_data_quality_issues,
        *create_graph_tools(rdf, ecn_row_id, captured_facts, issued_graph_nodes, catalogue),
        *create_replacement_tools(snapshots, rdf, ecn_row_id, catalogue),
    ]
    if ecn_row_id is None:
        # No open ECN: the chat only reads. Reviews are saved from the ECN page.
        return read_tools

    state = review_state if review_state is not None else {}

    @tool
    async def draft_review_decision(
        outcome: ReviewOutcome,
        reviewer: Annotated[str, Field(max_length=200)],
        comments: Annotated[str, Field(max_length=10000)] = "",
    ) -> dict:
        """Check an Accept / Reject review draft without saving anything.

        Load the review-decision skill before using. ``comments`` is the
        professional review comment you wrote from the user's remarks, naming
        exact parts, assemblies and plants. Returns the draft to show the user
        verbatim, the current decision and warnings to mention. Nothing is
        written; ask the user to confirm before ``submit_review_decision``.

        Example input: ``{"outcome": "CHANGES_REQUESTED", "reviewer": "Ana Diaz",
        "comments": "Confirm the PL10 cut-over date for QKX463881A."}``.
        """
        problem = review_problem(outcome, reviewer, comments)
        if problem:
            return _bounded({"assessment_id": ecn_row_id, "drafted": False, **problem})
        overview = await run_blocking(snapshots.overview, ecn_row_id)
        reviews = await run_blocking(workspace.reviews, ecn_row_id) if workspace else []
        current = review_status(reviews, overview)
        plan = build_acceptance_plan(overview)
        if outcome == "ACCEPTED" and plan["blocked"]:
            return _bounded({"assessment_id": ecn_row_id, "drafted": False, "error_code": "ECN_ACCEPT_BLOCKED",
                             "message": plan["block_reason"], "missing_price": plan["missing_price"][:10]})
        warnings = []
        if outcome == "ACCEPTED" and overview.get("questions"):
            warnings.append(f"{len(overview['questions'])} readiness questions are still open.")
        latest = current["latest"]
        if latest and latest["outcome"] == outcome and latest["applies_to_current_snapshot"]:
            warnings.append(f"{latest['reviewer']} already recorded this decision on the current snapshot.")
        return _bounded({
            "assessment_id": ecn_row_id, "drafted": True, "outcome": outcome,
            "reviewer": reviewer.strip(), "comments": comments.strip(),
            "current_decision": current["current_decision"], "warnings": warnings,
            "confirmation_required": True,
        })

    @tool
    async def submit_review_decision(
        outcome: ReviewOutcome,
        reviewer: Annotated[str, Field(max_length=200)],
        comments: Annotated[str, Field(max_length=10000)] = "",
    ) -> dict:
        """Save the confirmed engineering review. Only after explicit user confirmation.

        Load the review-decision skill before using. Call only when the
        user's latest message confirms the exact draft you showed in your
        previous answer; pass the same outcome, reviewer and comments. The
        save is refused when that draft was not shown. At most one save per
        answer. Accepting is refused while the ECN is blocked; the SAP write is simulated.

        Example input: ``{"outcome": "ACCEPTED", "reviewer": "Ana Diaz", "comments": ""}``.
        """
        problem = review_problem(outcome, reviewer, comments)
        if problem:
            return _bounded({"assessment_id": ecn_row_id, "saved": False, **problem})
        if state.get("saved"):
            return _bounded({"assessment_id": ecn_row_id, "saved": False, "error_code": "ALREADY_SAVED",
                             "message": "A review was already saved in this answer."})
        if workspace is None:
            return _bounded({"assessment_id": ecn_row_id, "saved": False, "error_code": "REVIEWS_UNAVAILABLE",
                             "message": "Reviews cannot be saved here; use the Engineering review card."})
        if not draft_was_shown(history, outcome, reviewer, comments):
            return _bounded({"assessment_id": ecn_row_id, "saved": False, "error_code": "DRAFT_NOT_SHOWN",
                             "message": "Show this exact draft with draft_review_decision and ask the user to confirm first."})
        overview = await run_blocking(snapshots.overview, ecn_row_id)
        request_id = str(uuid5(NAMESPACE_URL, "|".join(
            (ecn_row_id, overview["evidence_hash"], outcome, reviewer.strip(), comments.strip()))))
        try:
            reviews = await run_blocking(
                record_review, snapshots, workspace, ecn_row_id,
                request_id=request_id, reviewer=reviewer, outcome=outcome, comments=comments,
                actor=actor, evidence_hash=overview["evidence_hash"], snapshot_version=overview["version"],
            )
        except PersistenceStateError as error:
            return _bounded({"assessment_id": ecn_row_id, "saved": False,
                             "error_code": error.code, "message": error.public_message})
        state["saved"] = True
        latest = reviews[-1] if reviews else {}
        return _bounded({"assessment_id": ecn_row_id, "saved": True, "outcome": outcome,
                         "reviewer": reviewer.strip(), "created_at": _timestamp(latest.get("created_at")),
                         "sap_updated": False})

    return [*read_tools, draft_review_decision, submit_review_decision]
