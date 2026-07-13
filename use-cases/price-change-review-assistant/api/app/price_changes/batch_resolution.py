from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from .completion_model import CompletionDecision, CompletionResponse
from .models import (
    AgentProposal,
    GmailEmail,
    MaterialCandidate,
    PriceMode,
    RawExtraction,
    RawExtractionItem,
    SupplierCandidate,
)
from .tools import validate_price_change_proposal


STRICT_ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
YEAR_ONLY_DATE_PATTERN = re.compile(r"^\d{4}$")
ABSOLUTE_PRICE_PATTERN = re.compile(r"^\d+(?:\.\d+)?$")
SHEET_SUPPLIER_PATTERN = re.compile(
    r"\bSheet:?\s+(?P<name>[^;]+);\s*supplier\s+entity\s+(?P<supplier_id>SUP[0-9A-Z_-]+)\b",
    re.IGNORECASE,
)
SHEET_NAME_PATTERN = re.compile(r"\bSheet:?\s+(?P<name>[^;]+)", re.IGNORECASE)
SUPPLIER_NAME_ID_PATTERN = re.compile(
    r"(?P<name>[A-Z][A-Za-z0-9 &'._-]{1,80}?)\s*\((?P<supplier_id>SUP[0-9A-Z_-]+)\)",
    re.IGNORECASE,
)
SUPPLIER_ID_NAME_PATTERN = re.compile(
    r"(?P<supplier_id>SUP[0-9A-Z_-]+)\s+(?P<name>[A-Z][A-Za-z0-9 &'._-]{1,80}?)(?=\s+and\s+SUP|\)|,|;|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CanonicalPriceChangeItem:
    """Canonical price-change item used by batch completion.

    Attributes:
        item_index: Zero-based source item index.
        supplier_id: Supplier identifier extracted at the email level.
        supplier_name: Supplier name extracted at the email level.
        supplier_email: Supplier email extracted at the email level.
        material_number: Material number extracted for this item.
        material_description: Material description extracted for this item.
        original_price: Existing S/4 price, when known before completion.
        requested_price: Requested target price value.
        currency: Requested or mentioned currency.
        uom: Unit of measure mentioned in the source email.
        effective_from: Raw effective-from date text.
        effective_to: Raw effective-to date text.
        notes: Item-level extraction notes.
        confidence: Extractor confidence for the canonical row.
        price_change_mode: Requested price-change mode.
        price_change_value: Requested price-change value.
    """

    item_index: int
    supplier_id: str | None
    supplier_name: str | None
    supplier_email: str | None
    material_number: str | None
    material_description: str | None
    original_price: str | None
    requested_price: str | None
    currency: str | None
    uom: str | None
    effective_from: str | None
    effective_to: str | None
    notes: str | None
    confidence: float | None
    price_change_mode: PriceMode | None
    price_change_value: str | None

    def model_context(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the canonical row.

        Returns:
            Dictionary containing all canonical fields.
        """
        return asdict(self)


@dataclass
class S4ContextPacket:
    """S/4 lookup context assembled for one canonical price-change item.

    Attributes:
        item: Canonical source row this packet enriches.
        selected_supplier: Singular supplier selected from S/4, when available.
        supplier_candidates: Supplier candidates returned when lookup is not singular.
        selected_material: Singular material selected from S/4, when available.
        material_candidates: Material candidates returned when lookup is not singular.
        current_price: Exact current price for a singular supplier/material context.
        price_context_candidates: Possible price rows for ambiguous contexts.
        lookup_trace: Compact records of S/4 lookup methods called for this row.
        conflicts: Human-readable context conflicts detected while planning.
    """

    item: CanonicalPriceChangeItem
    selected_supplier: dict[str, Any] | None = None
    supplier_candidates: list[dict[str, Any]] = field(default_factory=list)
    selected_material: dict[str, Any] | None = None
    material_candidates: list[dict[str, Any]] = field(default_factory=list)
    current_price: dict[str, Any] | None = None
    price_context_candidates: list[dict[str, Any]] = field(default_factory=list)
    lookup_trace: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)

    def model_context(self) -> dict[str, Any]:
        """Return JSON-serializable context for completion model prompting.

        Returns:
            Dictionary containing the canonical row, selected/candidate S/4
            supplier and material context, price context, trace, and conflicts.
        """
        context = {
            "item_index": self.item.item_index,
            "canonical_item": self.item.model_context(),
            "selected_supplier": self.selected_supplier,
            "supplier_candidates": self.supplier_candidates,
            "selected_material": self.selected_material,
            "material_candidates": self.material_candidates,
            "current_price": self.current_price,
            "price_context_candidates": self.price_context_candidates,
            "lookup_trace": self.lookup_trace,
            "conflicts": self.conflicts,
        }
        safe_context = _json_safe_value(context)
        return safe_context if isinstance(safe_context, dict) else {}


@dataclass(frozen=True)
class PersistedDraftResult:
    """Persisted draft metadata returned by batch price-change processing.

    Attributes:
        item_index: Zero-based canonical item index that produced the draft.
        proposal: Validated proposal persisted for review.
        draft_id: Repository-assigned draft identifier.
    """

    item_index: int
    proposal: AgentProposal
    draft_id: str


class BatchPriceChangeProcessingError(Exception):
    """Raised when batch processing fails after persisting some draft results.

    Args:
        partial_results: Draft results persisted before the processing failure.
        original_error: Original exception raised by the processor dependency or persistence path.

    Attributes:
        partial_results: Persisted draft results that callers should still summarize.
        original_error: Original exception preserved for diagnostics and exception chaining.
    """

    def __init__(self, partial_results: list[PersistedDraftResult], original_error: Exception) -> None:
        """Create a partial batch-processing failure.

        Args:
            partial_results: Draft results persisted before the processing failure.
            original_error: Original exception raised by the failed operation.

        Returns:
            None.
        """
        self.partial_results = list(partial_results)
        self.original_error = original_error
        super().__init__(str(original_error))


class LookupCache:
    """Per-run cache for S/4 lookup repository calls.

    Args:
        lookup_repository: Repository or facade exposing S/4 lookup methods.
    """

    def __init__(self, lookup_repository: Any) -> None:
        """Create a cache bound to one lookup repository.

        Args:
            lookup_repository: Object exposing callable S/4 lookup methods.

        Returns:
            None.
        """
        self.lookup_repository = lookup_repository
        self._cache: dict[tuple[str, tuple[Any, ...]], dict[str, Any]] = {}

    def call(self, method_name: str, *args: Any) -> dict[str, Any]:
        """Call and cache one lookup method by method name and arguments.

        Args:
            method_name: Name of the lookup method on the repository.
            *args: Positional arguments to pass to the lookup method.

        Returns:
            Lookup result dictionary. Non-dict method results are wrapped in a
            failed response containing the raw result.
        """
        key = (method_name, tuple(_freeze_cache_value(arg) for arg in args))
        if key not in self._cache:
            method = getattr(self.lookup_repository, method_name)
            result = method(*args)
            if not isinstance(result, dict):
                result = {"status": "failed", "result": result}
            self._cache[key] = result
        return self._cache[key]


def _freeze_cache_value(value: Any) -> Any:
    """Convert mutable lookup arguments into hashable cache-key values.

    Args:
        value: Value passed to a cached lookup call.

    Returns:
        Hashable representation preserving dict/list/tuple/set content.
    """
    if isinstance(value, dict):
        return tuple(
            sorted(
                ((_freeze_cache_value(key), _freeze_cache_value(item)) for key, item in value.items()),
                key=lambda pair: repr(pair[0]),
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_cache_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((_freeze_cache_value(item) for item in value), key=repr))
    return value


def _json_safe_value(value: Any) -> Any:
    """Return a JSON-safe deep copy of common lookup payload values.

    Args:
        value: Arbitrary value from an extraction, lookup result, or packet.

    Returns:
        JSON-serializable copy. Dictionaries, lists, tuples, and sets are copied
        recursively; Decimal, date, datetime, and unknown objects are converted
        to stable strings.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (Decimal, date, datetime)):
        return str(value)
    if isinstance(value, dict):
        return {str(_json_safe_value(key)): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, set):
        return [_json_safe_value(item) for item in sorted(value, key=repr)]
    if hasattr(value, "model_dump"):
        return _json_safe_value(value.model_dump())
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe_value(asdict(value))
    return str(value)


def _single_candidate(result: dict[str, Any], candidate_key: str) -> dict[str, Any] | None:
    """Return a selected lookup row when a result is singular.

    Args:
        result: Lookup response dictionary.
        candidate_key: Singular key to prefer, such as `supplier` or `material`.

    Returns:
        Selected candidate row when exactly one row is available; otherwise None.
    """
    direct_candidate = result.get(candidate_key)
    if isinstance(direct_candidate, dict):
        safe_candidate = _json_safe_value(direct_candidate)
        return safe_candidate if isinstance(safe_candidate, dict) else None
    candidates = _candidate_rows(result)
    if len(candidates) == 1:
        return candidates[0]
    return None


def _candidate_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract list-shaped candidate rows from a lookup result.

    Args:
        result: Lookup response dictionary.

    Returns:
        Candidate dictionaries from the `candidates` field, excluding non-dict rows.
    """
    candidates = result.get("candidates")
    if not isinstance(candidates, list):
        return []
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if isinstance(candidate, dict):
            safe_candidate = _json_safe_value(candidate)
            if isinstance(safe_candidate, dict):
                rows.append(safe_candidate)
    return rows


def _has_singular_price_context(lookup_repository: Any) -> bool:
    """Return whether exact current-price lookup config is singular.

    Args:
        lookup_repository: Lookup repository or facade whose `.config` or
            `.lookup_repository.config` may contain S/4 context settings.

    Returns:
        False when purchasing organization, info-record category, or plant is a
        list/tuple/set; otherwise True. Missing config is treated as singular.
    """
    config = getattr(getattr(lookup_repository, "lookup_repository", lookup_repository), "config", None)
    if config is None:
        return True
    for field_name in ("purchasing_organization", "info_record_category", "plant"):
        if isinstance(getattr(config, field_name, None), (list, tuple, set)):
            return False
    return True


def _non_singular_price_context_values(
    lookup_repository: Any,
) -> tuple[list[Any] | None, list[Any] | None, list[Any] | None]:
    """Extract list-valued S/4 purchasing context filters for fallback lookups.

    Args:
        lookup_repository: Lookup repository or facade whose `.config` or
            `.lookup_repository.config` may contain S/4 context settings.

    Returns:
        Tuple of optional purchasing organizations, info-record categories, and
        plants. List/tuple/set config values are returned as lists; singular or
        missing values return None so repository defaults still apply.
    """
    config = getattr(getattr(lookup_repository, "lookup_repository", lookup_repository), "config", None)
    if config is None:
        return (None, None, None)

    def list_value(field_name: str) -> list[Any] | None:
        """Return a deterministic list for non-singular config values.

        Args:
            field_name: S/4 lookup config field name.

        Returns:
            List of configured values when the field is list-like; otherwise None.
        """
        value = getattr(config, field_name, None)
        if isinstance(value, set):
            return sorted(value, key=repr)
        if isinstance(value, (list, tuple)):
            return list(value)
        return None

    return (
        list_value("purchasing_organization"),
        list_value("info_record_category"),
        list_value("plant"),
    )


def _supplier_lookup_request(
    email: Any,
    item: CanonicalPriceChangeItem,
) -> tuple[str, str] | None:
    """Choose the deterministic supplier lookup method and value for an item.

    Args:
        email: Source email object that may expose `sender_email`.
        item: Canonical item containing extracted supplier clues.

    Returns:
        Tuple of lookup method name and lookup value, or None when no clue exists.
    """
    supplier_id_from_email, _supplier_name_from_email = _supplier_identity_from_item_notes(
        item.notes,
        supplier_name_map=_supplier_identity_map_from_texts(getattr(email, "body", None)),
    )
    if supplier_id_from_email:
        return ("find_supplier_by_id", supplier_id_from_email)
    if item.supplier_id:
        return ("find_supplier_by_id", item.supplier_id)
    if item.supplier_email:
        return ("find_supplier_by_email", item.supplier_email)
    if item.supplier_name:
        return ("find_supplier_by_name", item.supplier_name)
    sender_email = getattr(email, "sender_email", None)
    if sender_email:
        return ("find_supplier_by_email", sender_email)
    return None


def _material_lookup_request(
    item: CanonicalPriceChangeItem,
    selected_supplier: dict[str, Any] | None,
) -> tuple[str, tuple[Any, ...]] | None:
    """Choose the deterministic material lookup method and arguments.

    Args:
        item: Canonical item containing extracted material clues.
        selected_supplier: Selected supplier used to scope description search.

    Returns:
        Tuple of lookup method name and positional arguments, or None when no
        material clue exists.
    """
    if item.material_number:
        return ("find_material_by_number", (item.material_number,))
    if item.material_description:
        supplier_scope = None
        if selected_supplier is not None:
            supplier_scope = selected_supplier.get("supplier_id")
        return ("search_materials_by_description", (item.material_description, supplier_scope))
    return None


def _selected_supplier_id(selected_supplier: dict[str, Any]) -> str | None:
    """Extract the supplier identifier for exact price lookup.

    Args:
        selected_supplier: Singular supplier row selected from S/4.

    Returns:
        Supplier id or S/4 supplier number as a string, or None when missing.
    """
    value = selected_supplier.get("supplier_id") or selected_supplier.get("supplier")
    return str(value) if value is not None else None


def _selected_material_number(selected_material: dict[str, Any]) -> str | None:
    """Extract the material identifier for exact price lookup.

    Args:
        selected_material: Singular material row selected from S/4.

    Returns:
        Material code or material number as a string, or None when missing.
    """
    value = selected_material.get("material_code") or selected_material.get("material_number")
    return str(value) if value is not None else None


def _rows_for_price_context(
    selected: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return selected row or candidate rows for price-context lookup.

    Args:
        selected: Singular selected row, when available.
        candidates: Candidate rows from an ambiguous lookup.

    Returns:
        One-item selected row list or all candidate rows.
    """
    return [selected] if selected is not None else candidates


def _trace_entry(method_name: str, value: Any, result: dict[str, Any]) -> dict[str, Any]:
    """Build one compact lookup trace entry.

    Args:
        method_name: Lookup method name.
        value: Lookup value or argument summary.
        result: Lookup response.

    Returns:
        JSON-serializable trace dictionary with status and candidate count.
    """
    return {
        "method": method_name,
        "value": _json_safe_value(value),
        "status": _json_safe_value(result.get("status")),
        "candidate_count": len(_candidate_rows(result)),
    }


def _record_price_context_candidates(
    packet: S4ContextPacket,
    cache: LookupCache,
    supplier_rows: list[dict[str, Any]],
    material_rows: list[dict[str, Any]],
    purchasing_organizations: list[Any] | None,
    info_record_categories: list[Any] | None,
    plants: list[Any] | None,
) -> None:
    """Populate ambiguous price-context candidates on a packet.

    Args:
        packet: S/4 context packet being assembled.
        cache: Per-run lookup cache.
        supplier_rows: Supplier rows available for candidate lookup.
        material_rows: Material rows available for candidate lookup.
        purchasing_organizations: Optional purchasing organizations for fallback filtering.
        info_record_categories: Optional info-record categories for fallback filtering.
        plants: Optional plants for fallback filtering.

    Returns:
        None.
    """
    if not supplier_rows or not material_rows:
        return
    result = cache.call(
        "find_supplier_material_price_candidates",
        supplier_rows,
        material_rows,
        purchasing_organizations,
        info_record_categories,
        plants,
    )
    packet.price_context_candidates = _candidate_rows(result)
    trace_value = {
        "supplier_count": len(supplier_rows),
        "material_count": len(material_rows),
    }
    if purchasing_organizations is not None:
        trace_value["purchasing_organizations"] = purchasing_organizations
    if info_record_categories is not None:
        trace_value["info_record_categories"] = info_record_categories
    if plants is not None:
        trace_value["plants"] = plants
    packet.lookup_trace.append(
        {
            **_trace_entry(
                "find_supplier_material_price_candidates",
                trace_value,
                result,
            ),
            "price_context_candidate_count": len(packet.price_context_candidates),
        }
    )


def build_s4_context_packets(
    email: Any,
    canonical_items: list[CanonicalPriceChangeItem],
    lookup_repository: Any,
) -> list[S4ContextPacket]:
    """Build deterministic S/4 context packets for canonical price-change rows.

    Args:
        email: Source email object used for sender-email supplier fallback.
        canonical_items: Canonical rows produced by extraction normalization.
        lookup_repository: Repository or facade exposing S/4 lookup methods.

    Returns:
        One S4ContextPacket per canonical item with selected or candidate S/4
        supplier/material/price context.
    """
    cache = LookupCache(lookup_repository)
    purchasing_organizations, info_record_categories, plants = _non_singular_price_context_values(lookup_repository)
    packets: list[S4ContextPacket] = []

    for item in canonical_items:
        packet = S4ContextPacket(item=item)

        supplier_request = _supplier_lookup_request(email, item)
        if supplier_request is not None:
            supplier_method, supplier_value = supplier_request
            supplier_result = cache.call(supplier_method, supplier_value)
            packet.lookup_trace.append(_trace_entry(supplier_method, supplier_value, supplier_result))
            packet.selected_supplier = _single_candidate(supplier_result, "supplier")
            if packet.selected_supplier is None:
                packet.supplier_candidates = _candidate_rows(supplier_result)

        material_request = _material_lookup_request(item, packet.selected_supplier)
        if material_request is not None:
            material_method, material_args = material_request
            material_result = cache.call(material_method, *material_args)
            material_value: Any = material_args[0] if len(material_args) == 1 else list(material_args)
            packet.lookup_trace.append(_trace_entry(material_method, material_value, material_result))
            packet.selected_material = _single_candidate(material_result, "material")
            if packet.selected_material is None:
                packet.material_candidates = _candidate_rows(material_result)

        supplier_rows = _rows_for_price_context(packet.selected_supplier, packet.supplier_candidates)
        material_rows = _rows_for_price_context(packet.selected_material, packet.material_candidates)
        has_ambiguous_supplier_or_material = packet.selected_supplier is None or packet.selected_material is None

        if packet.selected_supplier is not None and packet.selected_material is not None:
            supplier_id = _selected_supplier_id(packet.selected_supplier)
            material_number = _selected_material_number(packet.selected_material)
            if supplier_id and material_number and _has_singular_price_context(lookup_repository):
                price_result = cache.call("get_current_supplier_material_price", supplier_id, material_number)
                packet.lookup_trace.append(
                    _trace_entry(
                        "get_current_supplier_material_price",
                        {"supplier_id": supplier_id, "material_number": material_number},
                        price_result,
                    )
                )
                if price_result.get("status") == "found" and isinstance(price_result.get("price"), dict):
                    safe_price = _json_safe_value(price_result["price"])
                    packet.current_price = safe_price if isinstance(safe_price, dict) else None
                elif price_result.get("status") in {"info_record_ambiguous", "price_ambiguous"}:
                    _record_price_context_candidates(
                        packet,
                        cache,
                        supplier_rows,
                        material_rows,
                        purchasing_organizations,
                        info_record_categories,
                        plants,
                    )
            else:
                _record_price_context_candidates(
                    packet,
                    cache,
                    supplier_rows,
                    material_rows,
                    purchasing_organizations,
                    info_record_categories,
                    plants,
                )
        elif has_ambiguous_supplier_or_material:
            _record_price_context_candidates(
                packet,
                cache,
                supplier_rows,
                material_rows,
                purchasing_organizations,
                info_record_categories,
                plants,
            )

        packets.append(packet)

    return packets


def _canonical_item_from_extraction(
    extraction: RawExtraction,
    item: RawExtractionItem | None,
    item_index: int,
    supplier_name_map: dict[str, tuple[str, str]] | None = None,
) -> CanonicalPriceChangeItem:
    """Build one canonical row from email-level data and an optional item.

    Args:
        extraction: Raw extractor output for the source email.
        item: Raw extracted item, or None when no item-level details exist.
        item_index: Zero-based canonical item index.
        supplier_name_map: Optional normalized supplier-name mapping derived
            from extraction-level context.

    Returns:
        CanonicalPriceChangeItem with missing values represented as None.
    """
    requested_price = item.requested_price if item is not None else None
    item_supplier_id, item_supplier_name = _supplier_identity_from_item_notes(
        item.notes if item is not None else None,
        supplier_name_map=supplier_name_map,
    )

    return CanonicalPriceChangeItem(
        item_index=item_index,
        supplier_id=extraction.supplier_id or item_supplier_id,
        supplier_name=item_supplier_name or extraction.supplier_name,
        supplier_email=extraction.supplier_email,
        material_number=item.material_number if item is not None else None,
        material_description=item.material_description if item is not None else None,
        original_price=None,
        requested_price=requested_price.value if requested_price is not None else None,
        currency=item.currency if item is not None else None,
        uom=item.uom if item is not None else None,
        effective_from=item.valid_from_raw if item is not None else None,
        effective_to=item.valid_to_raw if item is not None else None,
        notes=item.notes if item is not None else None,
        confidence=item.confidence if item is not None else extraction.confidence,
        price_change_mode=requested_price.mode if requested_price is not None else None,
        price_change_value=requested_price.value if requested_price is not None else None,
    )


def _normalized_supplier_name_key(value: str | None) -> str | None:
    """Normalize supplier names for deterministic sheet-to-supplier matching.

    Args:
        value: Supplier or sheet name text.

    Returns:
        Case-normalized key, or None when the text is empty.
    """
    if value is None:
        return None
    normalized = re.sub(r"\s+", " ", value.strip()).casefold()
    return normalized or None


def _supplier_identity_map_from_texts(*texts: str | None) -> dict[str, tuple[str, str]]:
    """Build a supplier-name map from free-text supplier context.

    Args:
        *texts: Free-text values that may list supplier ids and names.

    Returns:
        Mapping from normalized supplier name to supplier id and display name.
    """
    mapping: dict[str, tuple[str, str]] = {}

    def add(supplier_id: str | None, supplier_name: str | None) -> None:
        """Add one supplier mapping when both fields are present."""
        key = _normalized_supplier_name_key(supplier_name)
        if not supplier_id or key is None:
            return
        clean_id = supplier_id.strip().upper()
        clean_name = supplier_name.strip()
        mapping[key] = (clean_id, clean_name)

    for text in texts:
        if not text:
            continue
        for match in SUPPLIER_NAME_ID_PATTERN.finditer(text):
            add(match.group("supplier_id"), match.group("name"))
        for match in SUPPLIER_ID_NAME_PATTERN.finditer(text):
            add(match.group("supplier_id"), match.group("name"))
    return mapping


def _supplier_identity_map_from_extraction(extraction: RawExtraction) -> dict[str, tuple[str, str]]:
    """Build a supplier-name map from extraction-level supplier context.

    Args:
        extraction: Raw extraction whose reason may list supplier entities.

    Returns:
        Mapping from normalized supplier name to supplier id and display name.
    """
    mapping = _supplier_identity_map_from_texts(extraction.reason)

    def add(supplier_id: str | None, supplier_name: str | None) -> None:
        """Add one supplier mapping when both fields are present."""
        key = _normalized_supplier_name_key(supplier_name)
        if not supplier_id or key is None:
            return
        clean_id = supplier_id.strip().upper()
        clean_name = supplier_name.strip()
        mapping[key] = (clean_id, clean_name)

    add(extraction.supplier_id, extraction.supplier_name)
    return mapping


def _supplier_identity_from_item_notes(
    notes: str | None,
    supplier_name_map: dict[str, tuple[str, str]] | None = None,
) -> tuple[str | None, str | None]:
    """Extract item-scoped supplier identity from attachment notes.

    Args:
        notes: Item notes emitted by the extractor, often containing sheet and
            supplier-entity labels from tabular attachments.
        supplier_name_map: Optional mapping from sheet/supplier names to
            supplier ids derived from extraction-level context.

    Returns:
        Tuple of supplier id and supplier name, with None for values not found.
    """
    if not notes:
        return (None, None)
    match = SHEET_SUPPLIER_PATTERN.search(notes)
    if match is not None:
        supplier_name = match.group("name").strip()
        supplier_id = match.group("supplier_id").strip().upper()
        return (supplier_id or None, supplier_name or None)
    sheet_match = SHEET_NAME_PATTERN.search(notes)
    if sheet_match is None or not supplier_name_map:
        return (None, None)
    sheet_name = sheet_match.group("name").strip()
    supplier = supplier_name_map.get(_normalized_supplier_name_key(sheet_name) or "")
    if supplier is None:
        return (None, sheet_name or None)
    return supplier



def normalize_extraction_to_canonical_items(
    extraction: RawExtraction,
) -> list[CanonicalPriceChangeItem]:
    """Normalize raw extractor output into canonical price-change rows.

    Args:
        extraction: Raw extractor output for one email.

    Returns:
        Canonical rows, one per extracted item. Price requests with no item-level
        details return one email-level row with null item fields.
    """
    if not extraction.items and extraction.is_price_request:
        return [
            _canonical_item_from_extraction(
                extraction,
                item=None,
                item_index=0,
                supplier_name_map=_supplier_identity_map_from_extraction(extraction),
            )
        ]

    supplier_name_map = _supplier_identity_map_from_extraction(extraction)
    return [
        _canonical_item_from_extraction(
            extraction,
            item=item,
            item_index=item_index,
            supplier_name_map=supplier_name_map,
        )
        for item_index, item in enumerate(extraction.items)
    ]


def build_completion_payload(
    email: GmailEmail,
    extraction: RawExtraction,
    packets: list[S4ContextPacket],
) -> dict[str, Any]:
    """Assemble a JSON-safe Stage 2 completion payload for one packet batch.

    Args:
        email: Source Gmail message associated with the extraction.
        extraction: Raw extraction output being completed.
        packets: S/4 context packets for the canonical rows in this batch.

    Returns:
        JSON-serializable dictionary containing source context, item-indexed
        canonical rows, S/4 context, and completion rules.
    """
    payload = {
        "task": "complete_price_change_draft_rows",
        "email": email.model_dump(mode="json"),
        "extraction": extraction.model_dump(mode="json"),
        "items": [
            {
                "item_index": packet.item.item_index,
                "canonical_item": packet.item.model_context(),
                "s4_context": packet.model_context(),
            }
            for packet in packets
        ],
        "rules": [
            "Return exactly one completion item for each input item_index.",
            "Use deterministic S/4 context as authoritative when present.",
            "Leave mandatory fields null when no decisive value exists.",
            "Explain unresolved fields for human review.",
            "Do not approve or post price changes.",
        ],
    }
    safe_payload = _json_safe_value(payload)
    return safe_payload if isinstance(safe_payload, dict) else {}


def supplier_candidate_from_context_row(row: dict[str, Any], score: float = 0.5) -> SupplierCandidate | None:
    """Convert one S/4 supplier context row to a review candidate.

    Args:
        row: Supplier row from selected or candidate S/4 lookup context.
        score: Confidence score assigned to the review candidate.

    Returns:
        SupplierCandidate when an identifier is present; otherwise None.
    """
    supplier_id = row.get("supplier_id") or row.get("supplier")
    if supplier_id is None:
        return None
    supplier_name = row.get("supplier_name") or row.get("company") or row.get("name") or row.get("supplier")
    return SupplierCandidate(
        supplier_id=str(supplier_id),
        supplier_name=str(supplier_name or supplier_id),
        supplier_email=row.get("supplier_email") or row.get("email"),
        score=score,
    )


def material_candidate_from_context_row(row: dict[str, Any], score: float = 0.5) -> MaterialCandidate | None:
    """Convert one S/4 material or price context row to a review candidate.

    Args:
        row: Material or supplier-material price row from S/4 context.
        score: Confidence score assigned to the review candidate.

    Returns:
        MaterialCandidate when a material identifier is present; otherwise None.
    """
    material_number = row.get("material_number") or row.get("material_code")
    if material_number is None:
        return None
    material_description = row.get("material_description") or row.get("description") or row.get("material_name")
    return MaterialCandidate(
        material_number=str(material_number),
        material_description=str(material_description or material_number),
        supplier_id=str(row.get("supplier_id")) if row.get("supplier_id") is not None else None,
        current_price=str(row.get("current_price")) if row.get("current_price") is not None else None,
        currency=row.get("currency"),
        score=score,
    )


def _append_unique_error(errors: list[str], error: str) -> None:
    """Append one validation error when it is not already present.

    Args:
        errors: Validation error list to mutate.
        error: Error message to append.

    Returns:
        None.
    """
    if error not in errors:
        errors.append(error)


def _normalized_identifier(value: Any) -> str | None:
    """Normalize S/4 identifiers for backend grounding comparisons.

    Args:
        value: Supplier, material, or other identifier value.

    Returns:
        Case-normalized text, or None when the value is empty.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text.casefold() if text else None


def _normalized_price_text(value: Any) -> str | None:
    """Normalize localized unsigned price text to a decimal-dot string.

    Args:
        value: S/4, extraction, or completion price value.

    Returns:
        Decimal-dot numeric text when parseable, otherwise None.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if re.search(r"[A-Za-z%+\-]", text):
        return None
    text = text.replace("€", "").replace("$", "").replace(" ", "").replace("'", "")
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")
    if not ABSOLUTE_PRICE_PATTERN.fullmatch(text):
        return None
    try:
        return str(Decimal(text))
    except InvalidOperation:
        return None


def _normalized_decimal(value: Any) -> Decimal | None:
    """Normalize numeric text for current-price grounding comparisons.

    Args:
        value: S/4 or completion price value.

    Returns:
        Decimal value when parseable, otherwise None.
    """
    normalized = _normalized_price_text(value)
    if normalized is None:
        return None
    return Decimal(normalized)


def _row_material_number(row: dict[str, Any]) -> str | None:
    """Read a material identifier from an S/4 material or price row.

    Args:
        row: S/4 context row.

    Returns:
        Material number as text, or None when missing.
    """
    value = row.get("material_number") or row.get("material_code")
    return str(value) if value is not None else None


def _row_supplier_id(row: dict[str, Any]) -> str | None:
    """Read a supplier identifier from an S/4 supplier or price row.

    Args:
        row: S/4 context row.

    Returns:
        Supplier identifier as text, or None when missing.
    """
    value = row.get("supplier_id") or row.get("supplier")
    return str(value) if value is not None else None


def _row_supplier_identifiers(row: dict[str, Any]) -> set[str]:
    """Read all supplier identifiers from an S/4 supplier or price row.

    Args:
        row: S/4 context row.

    Returns:
        Normalized supplier identifiers from supported supplier-id fields.
    """
    identifiers = {
        _normalized_identifier(row.get("supplier_id")),
        _normalized_identifier(row.get("supplier")),
    }
    return {identifier for identifier in identifiers if identifier is not None}


def _supported_supplier_ids(packet: S4ContextPacket) -> set[str]:
    """Return supplier identifiers grounded in this item's S/4 context.

    Args:
        packet: S/4 context packet for one canonical item.

    Returns:
        Normalized supplier identifiers from selected suppliers, supplier
        candidates, exact current-price context, and price-context candidates.
    """
    rows: list[dict[str, Any]] = []
    if packet.selected_supplier is not None:
        rows.append(packet.selected_supplier)
    if packet.current_price is not None:
        rows.append(packet.current_price)
    rows.extend(packet.supplier_candidates)
    rows.extend(packet.price_context_candidates)
    supported: set[str] = set()
    for row in rows:
        supported.update(_row_supplier_identifiers(row))
    return supported


def _supported_material_numbers(packet: S4ContextPacket) -> set[str]:
    """Return material numbers grounded in this item's S/4 context.

    Args:
        packet: S/4 context packet for one canonical item.

    Returns:
        Normalized material identifiers from selected materials, material
        candidates, and price-context candidates.
    """
    rows: list[dict[str, Any]] = []
    if packet.selected_material is not None:
        rows.append(packet.selected_material)
    rows.extend(packet.material_candidates)
    rows.extend(packet.price_context_candidates)
    return {
        normalized
        for row in rows
        if (normalized := _normalized_identifier(_row_material_number(row))) is not None
    }


def _matching_price_context_candidate(
    packet: S4ContextPacket,
    supplier_id: str | None,
    material_number: str | None,
    original_price: str | None,
) -> dict[str, Any] | None:
    """Find a price-context candidate matching completion-selected values.

    Args:
        packet: S/4 context packet containing possible price rows.
        supplier_id: Completion or selected supplier id.
        material_number: Completion or selected material number.
        original_price: Completion-provided current/original price.

    Returns:
        Matching S/4 price-context candidate, or None when no candidate supports
        the selected supplier, material, and current price.
    """
    normalized_supplier = _normalized_identifier(supplier_id)
    normalized_material = _normalized_identifier(material_number)
    normalized_price = _normalized_decimal(original_price)
    if normalized_material is None or normalized_price is None:
        return None

    for row in packet.price_context_candidates:
        row_material = _normalized_identifier(_row_material_number(row))
        row_supplier = _normalized_identifier(_row_supplier_id(row))
        row_price = _normalized_decimal(row.get("current_price"))
        supplier_matches = normalized_supplier is None or row_supplier is None or row_supplier == normalized_supplier
        if row_material == normalized_material and row_price == normalized_price and supplier_matches:
            return row
    return None


def _append_completion_grounding_errors(
    packet: S4ContextPacket,
    supplier_id: str | None,
    decision_supplier_id: str | None,
    material_number: str | None,
    original_price: str | None,
    validation_errors: list[str],
) -> None:
    """Validate that completion-selected material and current price are S/4-grounded.

    Args:
        packet: S/4 context packet for the completed item.
        supplier_id: Proposal supplier id chosen from completion/S/4 context.
        decision_supplier_id: Completion-provided supplier id, if any.
        material_number: Proposal material number chosen from completion/S/4 context.
        original_price: Proposal current/original price chosen for review.
        validation_errors: Proposal validation errors to update.

    Returns:
        None.
    """
    normalized_supplier = _normalized_identifier(supplier_id)
    selected_supplier_ids = (
        _row_supplier_identifiers(packet.selected_supplier) if packet.selected_supplier is not None else set()
    )
    normalized_decision_supplier = _normalized_identifier(decision_supplier_id)
    if selected_supplier_ids and normalized_decision_supplier is not None:
        if normalized_decision_supplier not in selected_supplier_ids:
            _append_unique_error(
                validation_errors,
                "supplier_id does not match the selected S/4 supplier for this item",
            )
    else:
        supported_suppliers = _supported_supplier_ids(packet)
        if normalized_supplier is not None and normalized_supplier not in supported_suppliers:
            _append_unique_error(
                validation_errors,
                "supplier_id is not grounded in S/4 supplier candidates for this item",
            )

    normalized_material = _normalized_identifier(material_number)
    supported_materials = _supported_material_numbers(packet)
    if normalized_material is not None and normalized_material not in supported_materials:
        _append_unique_error(
            validation_errors,
            "material_number is not grounded in S/4 material candidates for this item",
        )

    if original_price is None:
        return

    if packet.current_price is not None:
        current_price = _normalized_decimal(packet.current_price.get("current_price"))
        if current_price is None:
            _append_unique_error(
                validation_errors,
                "original_price is not grounded in a usable S/4 current price",
            )
            return
        proposal_price = _normalized_decimal(original_price)
        if proposal_price != current_price:
            _append_unique_error(
                validation_errors,
                "original_price does not match the S/4 current price for this item",
            )
        return

    if _matching_price_context_candidate(packet, supplier_id, material_number, original_price) is None:
        _append_unique_error(
            validation_errors,
            "original_price is not grounded in S/4 price context for this item",
        )


def _packet_supplier_candidates(packet: S4ContextPacket) -> list[SupplierCandidate]:
    """Build unique supplier review candidates from packet context.

    Args:
        packet: S/4 context packet containing selected and ambiguous suppliers.

    Returns:
        Supplier candidates suitable for AgentProposal review metadata.
    """
    rows = []
    if packet.selected_supplier is not None:
        rows.append((packet.selected_supplier, 1.0))
    rows.extend((row, 0.5) for row in packet.supplier_candidates)
    candidates: list[SupplierCandidate] = []
    seen: set[str] = set()
    for row, score in rows:
        candidate = supplier_candidate_from_context_row(row, score=score)
        if candidate is None or candidate.supplier_id in seen:
            continue
        candidates.append(candidate)
        seen.add(candidate.supplier_id)
    return candidates


def _packet_material_candidates(packet: S4ContextPacket) -> list[MaterialCandidate]:
    """Build unique material review candidates from packet context.

    Args:
        packet: S/4 context packet containing selected, ambiguous, and price rows.

    Returns:
        Material candidates suitable for AgentProposal review metadata.
    """
    rows = []
    if packet.selected_material is not None:
        rows.append((packet.selected_material, 1.0))
    rows.extend((row, 0.5) for row in packet.material_candidates)
    rows.extend((row, 0.4) for row in packet.price_context_candidates)
    candidates: list[MaterialCandidate] = []
    seen: set[str] = set()
    for row, score in rows:
        candidate = material_candidate_from_context_row(row, score=score)
        if candidate is None or candidate.material_number in seen:
            continue
        candidates.append(candidate)
        seen.add(candidate.material_number)
    return candidates


def _price_context_value(packet: S4ContextPacket, field_name: str) -> str | None:
    """Read one field from exact current-price context as text.

    Args:
        packet: S/4 context packet that may contain current price.
        field_name: Current-price dictionary key to read.

    Returns:
        Text value from current price context, or None when missing.
    """
    if packet.current_price is None:
        return None
    value = packet.current_price.get(field_name)
    return str(value) if value is not None else None


def _selected_supplier_value(packet: S4ContextPacket, *field_names: str) -> str | None:
    """Read the first available selected supplier value as text.

    Args:
        packet: S/4 context packet that may contain a selected supplier.
        *field_names: Ordered supplier row keys to try.

    Returns:
        First non-null supplier value as text, or None.
    """
    if packet.selected_supplier is None:
        return None
    for field_name in field_names:
        value = packet.selected_supplier.get(field_name)
        if value is not None:
            return str(value)
    return None


def _selected_material_value(packet: S4ContextPacket, *field_names: str) -> str | None:
    """Read the first available selected material value as text.

    Args:
        packet: S/4 context packet that may contain a selected material.
        *field_names: Ordered material row keys to try.

    Returns:
        First non-null material value as text, or None.
    """
    if packet.selected_material is None:
        return None
    for field_name in field_names:
        value = packet.selected_material.get(field_name)
        if value is not None:
            return str(value)
    return None


def _is_absolute_price_text(value: str) -> bool:
    """Return whether text is a clean absolute decimal price.

    Args:
        value: Completion-provided requested price text.

    Returns:
        True when the text is unsigned decimal money-like text without percent,
        delta signs, currency words, or other non-decimal characters.
    """
    return _normalized_price_text(value) is not None


def _requested_new_price_value(
    item: CanonicalPriceChangeItem,
    decision: CompletionDecision,
    validation_errors: list[str],
) -> str | None:
    """Return a final requested price only when it is actually resolved.

    Args:
        item: Canonical extraction row containing extracted price clues.
        decision: Completion decision that may contain the final requested price.
        validation_errors: Validation error list to update for invalid relative prices.

    Returns:
        Completion-decided final price, extracted absolute price, or None for
        relative changes whose final price still needs calculation.
    """
    if decision.requested_new_price and item.price_change_mode in {"relative_percent", "relative_amount"}:
        normalized = _normalized_price_text(decision.requested_new_price)
        if normalized is not None:
            return normalized
        _append_unique_error(
            validation_errors,
            "requested_new_price must be a resolved absolute numeric price for relative price changes",
        )
        return None
    if decision.requested_new_price:
        normalized = _normalized_price_text(decision.requested_new_price)
        if normalized is not None:
            return normalized
        _append_unique_error(validation_errors, "requested_new_price must be a numeric price")
        return None
    if item.price_change_mode == "absolute":
        normalized = _normalized_price_text(item.requested_price)
        if normalized is not None:
            return normalized
        if item.requested_price is not None:
            _append_unique_error(validation_errors, "requested_new_price must be a numeric price")
        return None
    return None


def _is_iso_date_text(value: str) -> bool:
    """Return whether text is a normalized ISO calendar date.

    Args:
        value: Date text to validate.

    Returns:
        True when value parses as YYYY-MM-DD; otherwise False.
    """
    if not STRICT_ISO_DATE_PATTERN.fullmatch(value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _normalized_date_value(
    value: str | None,
    field_name: str,
    validation_errors: list[str],
) -> str | None:
    """Accept only normalized date values for proposal date fields.

    Args:
        value: Completion or extraction date text.
        field_name: Proposal field name receiving the date.
        validation_errors: Validation error list to update for raw dates.

    Returns:
        ISO date text when valid; otherwise None.
    """
    if value is None or not value.strip():
        return None
    normalized = value.strip()
    if YEAR_ONLY_DATE_PATTERN.fullmatch(normalized):
        return f"{normalized}-01-01"
    if _is_iso_date_text(normalized):
        return normalized
    _append_unique_error(validation_errors, f"{field_name} must be a normalized ISO date")
    return None


def proposal_from_completion(
    email: GmailEmail,
    extraction: RawExtraction,
    packet: S4ContextPacket,
    decision: CompletionDecision,
) -> AgentProposal:
    """Build and validate an AgentProposal from completion and S/4 context.

    Args:
        email: Source Gmail message for trusted metadata.
        extraction: Raw extraction output used for confidence fallback.
        packet: Canonical row and deterministic S/4 context for the item.
        decision: Stage 2 completion decision for this item.

    Returns:
        AgentProposal with backend validation errors appended and status set to
        ready_for_review only when all mandatory fields are resolved.
    """
    item = packet.item
    confidence = decision.confidence or item.confidence or extraction.confidence
    notes = decision.notes.strip() if decision.notes else ""
    unresolved_fields = ", ".join(decision.unresolved_fields)
    explanation = notes or "Completed from canonical extraction and S/4 context."
    if unresolved_fields:
        explanation = f"{explanation} Unresolved fields: {unresolved_fields}."
    validation_errors: list[str] = []
    requested_new_price = _requested_new_price_value(item, decision, validation_errors)
    if requested_new_price is None and item.price_change_mode in {"relative_percent", "relative_amount"}:
        _append_unique_error(
            validation_errors,
            "requested_new_price must be computed for relative price changes",
        )
    effective_from = _normalized_date_value(
        decision.effective_from or item.effective_from,
        "effective_from",
        validation_errors,
    )
    effective_to = _normalized_date_value(
        decision.effective_to or item.effective_to,
        "effective_to",
        validation_errors,
    )
    supplier_id = _selected_supplier_value(packet, "supplier_id", "supplier") or decision.supplier_id or item.supplier_id
    supplier_name = (
        _selected_supplier_value(packet, "supplier_name", "company", "name")
        or decision.supplier_name
        or item.supplier_name
    )
    supplier_email = (
        _selected_supplier_value(packet, "supplier_email", "email")
        or decision.supplier_email
        or item.supplier_email
        or email.sender_email
    )
    material_number = (
        decision.material_number
        or _selected_material_value(packet, "material_number", "material_code")
        or item.material_number
    )
    material_description = (
        decision.material_description
        or _selected_material_value(packet, "material_description", "description", "material_name")
        or item.material_description
    )
    original_price = _price_context_value(packet, "current_price") or decision.original_price or item.original_price
    if original_price is not None:
        normalized_original_price = _normalized_price_text(original_price)
        if normalized_original_price is None:
            _append_unique_error(validation_errors, "original_price must be a numeric price")
        original_price = normalized_original_price
    _append_completion_grounding_errors(
        packet=packet,
        supplier_id=supplier_id,
        decision_supplier_id=decision.supplier_id,
        material_number=material_number,
        original_price=original_price,
        validation_errors=validation_errors,
    )

    proposal = AgentProposal(
        status="needs_human_review",
        supplier_id=supplier_id,
        supplier_name=supplier_name,
        supplier_email=supplier_email,
        material_number=material_number,
        material_description=material_description,
        original_price=original_price,
        requested_new_price=requested_new_price,
        currency=decision.currency or _price_context_value(packet, "currency") or item.currency,
        uom=decision.uom or _price_context_value(packet, "uom") or item.uom,
        price_change_mode=item.price_change_mode,
        price_change_value=item.price_change_value,
        effective_from=effective_from,
        effective_to=effective_to,
        email_date=email.email_date.isoformat() if email.email_date is not None else "",
        gmail_message_id=email.gmail_message_id,
        confidence=confidence,
        explanation=explanation,
        candidate_materials=_packet_material_candidates(packet),
        candidate_suppliers=_packet_supplier_candidates(packet),
        validation_errors=validation_errors,
    )

    for field_name in decision.unresolved_fields:
        if getattr(proposal, field_name, None) in {None, ""}:
            _append_unique_error(proposal.validation_errors, f"{field_name} is unresolved")
    for error in validate_price_change_proposal(proposal):
        _append_unique_error(proposal.validation_errors, error)
    proposal.status = "needs_human_review" if proposal.validation_errors else "ready_for_review"
    return proposal


def _missing_completion_decision(packet: S4ContextPacket) -> CompletionDecision:
    """Create a review-required decision for a missing completion response item.

    Args:
        packet: S/4 context packet whose item was omitted by completion.

    Returns:
        CompletionDecision containing the packet item index and explanatory note.
    """
    return CompletionDecision(
        item_index=packet.item.item_index,
        confidence=packet.item.confidence or 0.0,
        notes="Completion response omitted this item; human review is required.",
        unresolved_fields=["completion_response"],
    )


def _completion_decisions_by_index(response: CompletionResponse) -> dict[int, CompletionDecision]:
    """Index completion response items by item index, keeping the first decision.

    Args:
        response: Completion model response for one packet batch.

    Returns:
        Mapping from item index to completion decision.
    """
    decisions: dict[int, CompletionDecision] = {}
    for decision in response.items:
        decisions.setdefault(decision.item_index, decision)
    return decisions


def _chunks(values: list[S4ContextPacket], size: int) -> list[list[S4ContextPacket]]:
    """Split packets into positive-size chunks.

    Args:
        values: S/4 context packets to chunk.
        size: Desired chunk size.

    Returns:
        List of packet chunks.
    """
    chunk_size = max(1, size)
    return [values[index : index + chunk_size] for index in range(0, len(values), chunk_size)]


class BatchPriceChangeProcessor:
    """Batch processor that completes and persists price-change draft proposals."""

    def __init__(self, tools: Any, completion_client: Any, completion_batch_size: int = 8) -> None:
        """Create a batch processor.

        Args:
            tools: Price-change tool facade exposing S/4 lookups and draft persistence.
            completion_client: Client exposing ``complete(payload, ...)`` for Stage 2 completion.
            completion_batch_size: Maximum number of canonical rows per completion call.

        Returns:
            None.
        """
        self.tools = tools
        self.completion_client = completion_client
        self.completion_batch_size = max(1, completion_batch_size)

    def process_email(
        self,
        email: GmailEmail,
        extraction: RawExtraction,
        extraction_id: str,
        token_usage_tracker: Any | None = None,
        progress_reporter: Any | None = None,
        llm_usage_context: Any | None = None,
    ) -> list[PersistedDraftResult]:
        """Complete and persist one draft for each canonical item in an email.

        Args:
            email: Source Gmail message being processed.
            extraction: Raw extraction output for the email.
            extraction_id: Persisted extraction identifier associated with drafts.
            token_usage_tracker: Optional tracker passed through to completion calls.
            progress_reporter: Optional reporter receiving coarse processing events.
            llm_usage_context: Optional Cloud Logging request context.

        Returns:
            Persisted draft results, one per canonical price-change item.
        """
        if progress_reporter is not None:
            progress_reporter.event("normalizing_extraction", "Preparing extracted price-change rows")
        canonical_items = normalize_extraction_to_canonical_items(extraction)

        if progress_reporter is not None:
            progress_reporter.event("gathering_s4_context", "Building deterministic S/4 context")
        packets = build_s4_context_packets(email, canonical_items, lookup_repository=self.tools)

        results: list[PersistedDraftResult] = []
        try:
            for batch_number, packet_batch in enumerate(_chunks(packets, self.completion_batch_size), start=1):
                if progress_reporter is not None:
                    progress_reporter.event(
                        "completing_rows",
                        "Completing price-change drafts",
                        metadata={"batch_number": batch_number, "item_count": len(packet_batch)},
                    )
                payload = build_completion_payload(email, extraction, packet_batch)
                completion_response = self.completion_client.complete(
                    payload,
                    token_usage_tracker=token_usage_tracker,
                    gmail_message_id=email.gmail_message_id,
                    extraction_id=extraction_id,
                    llm_usage_context=llm_usage_context,
                )
                decisions = _completion_decisions_by_index(completion_response)

                for packet in packet_batch:
                    decision = decisions.get(packet.item.item_index)
                    missing_completion = decision is None
                    if decision is None:
                        decision = _missing_completion_decision(packet)
                    proposal = proposal_from_completion(email, extraction, packet, decision)
                    if missing_completion:
                        _append_unique_error(
                            proposal.validation_errors,
                            "completion response omitted this item",
                        )
                        proposal.status = "needs_human_review"
                    raw_agent_output = {
                        "canonical_item": packet.item.model_context(),
                        "s4_context": packet.model_context(),
                        "completion_decision": decision.model_dump(mode="json"),
                    }
                    draft_id = self.tools.persist_price_change_draft(
                        proposal=proposal,
                        extraction_id=extraction_id,
                        item_index=packet.item.item_index,
                        raw_agent_output=raw_agent_output,
                    )
                    results.append(
                        PersistedDraftResult(
                            item_index=packet.item.item_index,
                            proposal=proposal,
                            draft_id=draft_id,
                        )
                    )
                    if progress_reporter is not None:
                        progress_reporter.event(
                            "draft_saved",
                            "Draft saved for review",
                            metadata={"draft_id": draft_id, "item_index": packet.item.item_index},
                        )
        except Exception as exc:
            if results:
                raise BatchPriceChangeProcessingError(results, exc) from exc
            raise

        return results
