"""Per-turn catalogue of stored ECNs for the chat.

The chat tools take an ``ecn`` argument (ECN numbers). ``EcnCatalogue`` turns
those numbers into stored occurrences, decides which ECNs one call reads (the
named ones, the open ECN, or none), and records every ECN it hands out so the
answer can link to them.

Example::

    catalogue = EcnCatalogue(ecn_repository, open_ecn_row_id=None)
    targets, problem = await catalogue.targets(["ECN306072"], limit=5)
"""

from app.ecn.api_models import EcnListQuery
from app.ecn.services.blocking_io import run_blocking

CATALOGUE_LIMIT = 1000
"""Most stored ECNs the chat can see in one turn (newest uploads first)."""


def _iso(value) -> str | None:
    """Render an optional datetime as ISO text so tool output stays JSON-serializable."""
    return value.isoformat() if value is not None else None


def catalogue_entry(item) -> dict:
    """Keep the list fields a chat answer needs from one ``EcnListItem``.

    Args:
        item: One ``EcnListItem`` from ``EcnRepository.list_ecns``.

    Returns:
        ECN number, title, short summary, dates, plants (``None`` = not
        assessed yet), overall impact, review outcome (``NOT_REVIEWED`` when
        none), analysis state, counts and the internal ``assessment_id``.
    """
    counts = item.counts
    return {
        "assessment_id": str(item.ecn_row_id),
        "ecn_number": item.display_ecn_number,
        "title": item.title,
        "summary": (item.description_summary or "")[:300] or None,
        "resolution_date": _iso(item.resolution_date),
        "uploaded_at": _iso(item.uploaded_at),
        "plants": item.affected_plants,
        "overall_impact": item.overall_impact.value if item.overall_impact else None,
        "review_outcome": item.review_outcome or "NOT_REVIEWED",
        "analysis_state": item.analysis_state.value,
        "counts": {
            "parts": counts.affected_item_count,
            "ebom_matches": counts.ebom_match_count,
            "mbom_matches": counts.mbom_match_count,
            "warnings": counts.warning_count,
        },
    }


class EcnCatalogue:
    """Resolve ECN numbers for one chat turn and record which ECNs were read.

    Args:
        ecn_repository: ``EcnRepository`` providing the ECN list.
        open_ecn_row_id: ECN open on the detail page, or ``None`` on other pages.
    """

    def __init__(self, ecn_repository, open_ecn_row_id: str | None = None):
        """Bind the repository and the open ECN; nothing is read until needed."""
        self.ecns = ecn_repository
        self.open_ecn_row_id = open_ecn_row_id
        self.consulted: dict[str, str] = {}
        self._entries: list[dict] | None = None

    def _load(self) -> list[dict]:
        """Read the ECN list newest first; keep the newest occurrence per ECN number."""
        entries, seen, page, read = [], set(), 1, 0
        # ponytail: the whole catalogue (up to CATALOGUE_LIMIT rows, 100 per
        # SQL page) is read once per chat turn; add server-side filtering if
        # stored ECNs grow far beyond that.
        while read < CATALOGUE_LIMIT:
            response = self.ecns.list_ecns(
                EcnListQuery(page=page, page_size=100, sort="uploaded_at", order="desc"))
            for item in response.items:
                entry = catalogue_entry(item)
                read += 1
                key = entry["ecn_number"].casefold()
                if key not in seen:
                    seen.add(key)
                    entries.append(entry)
            if page >= response.total_pages:
                break
            page += 1
        return entries

    async def entries(self) -> list[dict]:
        """Return the newest occurrence of every stored ECN (read once per turn)."""
        if self._entries is None:
            self._entries = await run_blocking(self._load)
        return self._entries

    def note(self, entries: list[dict]) -> None:
        """Record ECNs whose data was returned, for the answer's ECN links."""
        for entry in entries:
            if entry.get("ecn_number"):
                self.consulted[entry["assessment_id"]] = entry["ecn_number"]

    async def resolve(self, numbers: list[str]) -> tuple[list[dict], list[str]]:
        """Map ECN numbers to stored occurrences.

        Args:
            numbers: ECN numbers as written by the user (case-insensitive).

        Returns:
            ``(entries, unknown)``: catalogue entries in input order (the
            newest upload when a number was imported more than once) and the
            numbers that match no stored ECN.
        """
        by_number = {entry["ecn_number"].casefold(): entry for entry in await self.entries()}
        found, unknown = [], []
        for number in dict.fromkeys(numbers):
            entry = by_number.get(number.casefold())
            if entry is None:
                unknown.append(number)
            else:
                found.append(entry)
        return found, unknown

    async def targets(self, ecn: str | list[str], limit: int | None) -> tuple[list[dict], dict | None]:
        """Decide which ECNs one tool call reads.

        Args:
            ecn: ECN numbers from the tool call (one string or a list).
            limit: Most ECNs this tool may read at once, or ``None`` for no cap.

        Returns:
            ``(targets, problem)``. Empty ``ecn`` targets the open ECN as
            ``{"assessment_id", "ecn_number": None}`` (no catalogue read);
            otherwise the resolved entries. ``problem`` is an error result for
            the model: ``ECN_REQUIRED`` (no ECN open or named),
            ``TOO_MANY_ECNS`` or ``ECN_NOT_FOUND``.
        """
        numbers = [ecn] if isinstance(ecn, str) else list(ecn)
        numbers = [number.strip()[:80] for number in numbers if number and number.strip()]
        if not numbers:
            if self.open_ecn_row_id is not None:
                return [{"assessment_id": self.open_ecn_row_id, "ecn_number": None}], None
            return [], {"error_code": "ECN_REQUIRED",
                        "message": "No ECN is open. Pass ECN numbers in `ecn`; find them with "
                                   "list_ecns, find_ecns_with_part or find_ecn_facts."}
        if self.ecns is None:
            # Impact-analysis discovery binds the tools to one ECN only.
            return [], {"error_code": "ECN_SCOPE_FIXED",
                        "message": "Only the selected ECN is available here; leave `ecn` empty."}
        if limit is not None and len(numbers) > limit:
            return [], {"error_code": "TOO_MANY_ECNS",
                        "message": f"This tool reads at most {limit} ECNs per call; split the request."}
        entries, unknown = await self.resolve(numbers)
        if unknown:
            return [], {"error_code": "ECN_NOT_FOUND",
                        "message": f"No stored ECN is numbered {', '.join(unknown)}. Use list_ecns to find ECN numbers."}
        return entries, None
