"""Cross-ECN discovery tools for the chat.

``create_global_tools`` adds three discovery tools that work over all stored
ECNs: ``list_ecns`` (catalogue with impact, plants and review state),
``find_ecns_with_part`` and ``find_related_ecns`` (material matches between
ECNs, from ``SnapshotService.part_index``).

Example::

    catalogue = EcnCatalogue(ecn_repository, open_ecn_row_id=None)
    tools = create_global_tools(catalogue, snapshot_service)
"""

from typing import Annotated, Literal

from langchain_core.tools import tool
from pydantic import Field

from app.ecn.assistant.catalogue import EcnCatalogue
from app.ecn.assistant.tool_scope import _bounded
from app.ecn.services.blocking_io import run_blocking

_LIST_PAGE_SIZE = 20
_MATCH_PAGE_SIZE = 10


async def _part_indexes(snapshots, entries: list[dict]) -> tuple[list[tuple[dict, dict]], list[str]]:
    """Read the part index of each ECN, skipping ECNs whose snapshot cannot be built.

    Returns:
        ``([(entry, index)], unavailable_ecn_numbers)``.
    """
    indexed, unavailable = [], []
    # ponytail: sequential reads keep HANA connection use flat; run them
    # concurrently only if first-time cross-ECN searches are too slow.
    for entry in entries:
        try:
            indexed.append((entry, await run_blocking(snapshots.part_index, entry["assessment_id"])))
        except Exception:  # noqa: BLE001 - one broken ECN must not hide matches in the others.
            unavailable.append(entry["ecn_number"])
    return indexed, unavailable


def _page(rows: list[dict], page: int, page_size: int) -> dict:
    """Return one page of rows with totals and continuation metadata."""
    start = (page - 1) * page_size
    return {"total": len(rows), "page": page, "page_size": page_size,
            "has_next": start + page_size < len(rows), "items": rows[start:start + page_size]}


def create_global_tools(catalogue: EcnCatalogue, snapshots) -> list:
    """Return the cross-ECN discovery tools bound to one chat turn.

    Args:
        catalogue: Per-turn ``EcnCatalogue`` (also records consulted ECNs).
        snapshots: ``SnapshotService`` providing ``part_index``.

    Returns:
        ``list_ecns``, ``find_ecns_with_part`` and ``find_related_ecns``.
    """
    @tool
    async def list_ecns(
        query: str = "",
        plant: str = "",
        impact: Literal["HIGH", "MEDIUM", "LOW", "UNKNOWN", "NOT_APPLICABLE"] | None = None,
        review: Literal["ACCEPTED", "CHANGES_REQUESTED", "NOT_REVIEWED"] | None = None,
        page: Annotated[int, Field(ge=1)] = 1,
    ) -> dict:
        """List stored ECNs with their impact, plants, review decision and counts.

        Use FIRST for questions over all ECNs: which ECNs exist, which have
        high impact, which affect a plant, which are accepted, rejected
        (CHANGES_REQUESTED) or not reviewed. ``query`` matches ECN number,
        title or summary; ``plant`` matches a plant code or name; ``impact``
        and ``review`` filter exactly. Twenty ECNs per page, newest upload
        first; follow ``has_next``. ``plants`` is null when the ECN was never
        assessed (not "no plant"). Pass returned ECN numbers in ``ecn`` to
        the other tools for details.

        Example input: ``{"impact": "HIGH"}`` or ``{"plant": "PL10", "review": "NOT_REVIEWED"}``.
        """
        text, plant_text = query.strip().casefold()[:200], plant.strip().casefold()[:80]
        rows = [
            entry for entry in await catalogue.entries()
            if (not text or any(text in (entry[key] or "").casefold() for key in ("ecn_number", "title", "summary")))
            and (not plant_text or any(plant_text in label.casefold() for label in entry["plants"] or ()))
            and (impact is None or entry["overall_impact"] == impact)
            and (review is None or entry["review_outcome"] == review)
        ]
        result = _page(rows, page, _LIST_PAGE_SIZE)
        catalogue.note(result["items"])
        result["items"] = [{k: v for k, v in entry.items() if k != "assessment_id"} for entry in result["items"]]
        return _bounded(result)

    @tool
    async def find_ecns_with_part(
        number: Annotated[str, Field(min_length=1, max_length=200)],
        ecn: list[str] = Field(default_factory=list, max_length=50),
        page: Annotated[int, Field(ge=1)] = 1,
    ) -> dict:
        """Find every ECN that touches one exact part number, and how.

        Use for "which ECNs include / change / replace part X?". Searches all
        stored ECNs (or only the ECN numbers in ``ecn``), covering parts named
        in the ECN and parts in its BOM changes. Each hit gives the ECN number,
        ``roles`` (ECN_OBJECT = named in the ECN; OUTGOING / INCOMING = old or
        new side of a BOM change), ``changes``, ``assemblies``, ``plants`` and
        EBOM release / MBOM cut-over / end dates. Ten hits per page.
        ``unavailable_ecns`` could not be checked; never treat them as no match.

        Example input: ``{"number": "QKX7702260"}``.
        """
        target = number.strip().casefold()
        if ecn:
            entries, problem = await catalogue.targets(ecn, None)
            if problem:
                return _bounded(problem)
        else:
            entries = await catalogue.entries()
        indexed, unavailable = await _part_indexes(snapshots, entries)
        hits = []
        for entry, index in indexed:
            part = next((value for key, value in index["parts"].items() if key.casefold() == target), None)
            if part is not None:
                hits.append((entry, {"ecn_number": entry["ecn_number"], "title": entry["title"],
                                     "overall_impact": entry["overall_impact"], **part}))
        result = _page([row for _, row in hits], page, _MATCH_PAGE_SIZE)
        catalogue.note([entry for entry, row in hits if row in result["items"]])
        return _bounded({"number": number.strip(), "ecns_searched": len(indexed),
                         "unavailable_ecns": unavailable, **result})

    @tool
    async def find_related_ecns(
        ecn: Annotated[str, Field(max_length=80)] = "",
        page: Annotated[int, Field(ge=1)] = 1,
    ) -> dict:
        """Find other ECNs that share part numbers with one ECN.

        Use for "which ECNs have the same materials as this ECN?". ``ecn`` is
        one ECN number; leave it empty for the open ECN. Returns other ECNs
        ranked by how many parts they share, each with up to fifteen shared
        part numbers and the part's roles in both ECNs (ECN_OBJECT, OUTGOING,
        INCOMING). Ten ECNs per page. ``unavailable_ecns`` could not be checked.

        Example input: ``{}`` or ``{"ecn": "ECN306072"}``.
        """
        targets, problem = await catalogue.targets(ecn, 1)
        if problem:
            return _bounded(problem)
        source = targets[0]
        source_index = await run_blocking(snapshots.part_index, source["assessment_id"])
        source_number = source["ecn_number"] or source_index["ecn_number"]
        others = [entry for entry in await catalogue.entries()
                  if entry["assessment_id"] != source["assessment_id"]
                  and entry["ecn_number"].casefold() != (source_number or "").casefold()]
        indexed, unavailable = await _part_indexes(snapshots, others)
        mine = source_index["parts"]
        rows, matched = [], {}
        for entry, index in indexed:
            shared = sorted(set(mine) & set(index["parts"]))
            if not shared:
                continue
            matched[entry["ecn_number"]] = entry
            rows.append({
                "ecn_number": entry["ecn_number"], "title": entry["title"],
                "overall_impact": entry["overall_impact"], "shared_count": len(shared),
                "shared_parts": [{"number": n, "description": mine[n]["description"],
                                  "roles_in_source": mine[n]["roles"], "roles_in_other": index["parts"][n]["roles"]}
                                 for n in shared[:15]],
            })
        rows.sort(key=lambda row: (-row["shared_count"], row["ecn_number"]))
        result = _page(rows, page, _MATCH_PAGE_SIZE)
        catalogue.note([matched[row["ecn_number"]] for row in result["items"]])
        return _bounded({"source_ecn": source_number, "source_part_count": len(mine),
                         "ecns_searched": len(indexed), "unavailable_ecns": unavailable, **result})

    return [list_ecns, find_ecns_with_part, find_related_ecns]
