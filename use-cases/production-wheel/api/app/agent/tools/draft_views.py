"""Compact, model-facing views of optimization drafts and their validation.

A draft stores the complete optimizer configuration, including two lists that
the plant profile owns and that grow with the plant portfolio:

* ``matrix_pairs``: every ordered package-volume pair of the compatibility
  matrix (51 volumes give 2,601 rows, about 140 KB of JSON);
* ``group_size_overrides``: one group-size cap per plant/SEFI block (hundreds
  of rows for a full portfolio).

The draft also embeds a frozen copy of the plant profile with the same matrix
again (``plant_profile.matrix_rows``). Returned unchanged, a draft or its
validation is several times larger than the context budget's per-tool limit
(``context_budget.max_tool_result_bytes``), and the model only sees an
"output omitted" placeholder. It then cannot read ``valid``, ``errors`` or
``pv_mode_scope`` and refuses to launch.

The functions here replace only those large lists with exact summaries
(counts, statuses, volumes). Every validation field the agent must reason about
is kept unchanged. The service, the HTTP API, the browser events and the
launch check keep using the full records; only tool results sent to the model
are compacted.

The summarized lists are fixed by the plant profile: ``update_draft`` rejects a
patch that changes them, so a summary echoed back by the model can never
silently replace the real list.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

# Lists longer than this in a draft config are summarized generically, so any
# future large field cannot push a draft over the per-tool budget again.
LARGE_LIST_ITEMS = 50
# Number of example items kept for a generically summarized list.
SAMPLE_ITEMS = 5

PROFILE_OWNED_NOTE = (
    "Summarized for the assistant: the full list is fixed by the plant profile "
    "and stays unchanged in the draft. Do not copy this summary into a draft "
    "patch; change the plant profile instead."
)


def _volume_key(volume: str) -> tuple[int, Decimal | str]:
    """Sort package volumes numerically, keeping non-numeric labels last.

    Args:
        volume: Package volume label in litres, e.g. ``"2.5"``.

    Returns:
        A sort key: numeric volumes first in ascending order, then other labels.
    """
    try:
        return (0, Decimal(volume))
    except (InvalidOperation, ValueError):
        return (1, volume)


def summarize_matrix_pairs(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize a compatibility matrix without returning its rows.

    Args:
        rows: Matrix rows with ``volume_a``, ``volume_b`` and ``status``
            (``Y``, ``AVOID`` or ``N``), as stored in a draft config or profile.

    Returns:
        ``summarized`` (always true), ``pair_count`` (stored ordered rows),
        ``pairs_by_status`` (row count per status), ``volume_count``,
        ``volumes`` (ascending) and the profile-owned ``note``.
    """
    rows = list(rows)
    volumes = {str(row.get(side)) for row in rows for side in ("volume_a", "volume_b") if row.get(side) is not None}
    return {
        "summarized": True,
        "pair_count": len(rows),
        "pairs_by_status": dict(sorted(Counter(str(row.get("status")) for row in rows).items())),
        "volume_count": len(volumes),
        "volumes": sorted(volumes, key=_volume_key),
        "note": PROFILE_OWNED_NOTE,
    }


def summarize_group_size_overrides(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize per-block group-size caps without returning every block.

    Args:
        rows: Override rows with ``plant``, ``sefi`` and ``maximum``.

    Returns:
        ``summarized`` (always true), ``block_count``, ``blocks_by_maximum``
        (how many blocks use each cap), ``sample`` (first rows) and the
        profile-owned ``note``.
    """
    rows = list(rows)
    by_maximum = Counter(str(row.get("maximum")) for row in rows)
    return {
        "summarized": True,
        "block_count": len(rows),
        "blocks_by_maximum": dict(sorted(by_maximum.items(), key=lambda item: _volume_key(item[0]))),
        "sample": [dict(row) for row in rows[:SAMPLE_ITEMS]],
        "note": PROFILE_OWNED_NOTE,
    }


def _summarize_list(items: list[Any]) -> dict[str, Any]:
    """Return a count and the first items of any other oversized config list.

    Args:
        items: A config list longer than ``LARGE_LIST_ITEMS``.

    Returns:
        ``summarized``, ``item_count`` and ``sample`` (first ``SAMPLE_ITEMS``).
    """
    return {"summarized": True, "item_count": len(items), "sample": items[:SAMPLE_ITEMS]}


def compact_config(config: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return an optimizer config with its large lists summarized.

    Args:
        config: Draft request config or compiled effective config (JSON form).

    Returns:
        A new dict (the input is not modified) where ``matrix_pairs`` and
        ``group_size_overrides`` are replaced by their summaries and any other
        list longer than ``LARGE_LIST_ITEMS`` by a count and a sample. Scalar
        settings, small lists (scope, constraints, business rules) and nested
        objects are kept as they are. ``None`` is returned unchanged.
    """
    if config is None:
        return None
    result: dict[str, Any] = {}
    for key, value in config.items():
        if key == "matrix_pairs" and isinstance(value, list):
            result[key] = summarize_matrix_pairs(value)
        elif key == "group_size_overrides" and isinstance(value, list) and len(value) > SAMPLE_ITEMS:
            result[key] = summarize_group_size_overrides(value)
        elif isinstance(value, list) and len(value) > LARGE_LIST_ITEMS:
            result[key] = _summarize_list(value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _compact_request(request: Mapping[str, Any] | None) -> Any:
    """Compact the ``config`` of a draft request and keep its other fields.

    Args:
        request: Draft ``request`` or ``source_request`` (scope, constraints, config).

    Returns:
        A copy with ``config`` compacted, or the input unchanged when it is not a dict.
    """
    if not isinstance(request, Mapping):
        return request
    compacted = {key: copy.deepcopy(value) for key, value in request.items() if key != "config"}
    if "config" in request:
        compacted["config"] = compact_config(request["config"])
    return compacted


def _compact_profile(profile: Mapping[str, Any] | None) -> Any:
    """Replace a frozen profile's matrix rows with their count and volumes.

    Args:
        profile: The draft's frozen ``plant_profile`` snapshot.

    Returns:
        A copy without ``matrix_rows`` but with ``matrix_pair_count``,
        ``matrix_pairs_by_status`` and ``matrix_volumes``; settings, rules and
        revision are kept. Non-dict input is returned unchanged.
    """
    if not isinstance(profile, Mapping):
        return profile
    compacted = {key: copy.deepcopy(value) for key, value in profile.items() if key != "matrix_rows"}
    if "matrix_rows" in profile:
        summary = summarize_matrix_pairs(profile.get("matrix_rows") or [])
        compacted["matrix_pair_count"] = summary["pair_count"]
        compacted["matrix_pairs_by_status"] = summary["pairs_by_status"]
        compacted["matrix_volumes"] = summary["volumes"]
    return compacted


def compact_draft(draft: Mapping[str, Any]) -> dict[str, Any]:
    """Return a model-facing draft with profile-owned lists summarized.

    Args:
        draft: Draft record as returned by the workspace service (``draft_id``,
            ``revision``, ``request``, optional ``source_request``,
            ``plant_profile``, ``budget``, ``parameter_sources``, ...).

    Returns:
        A new dict with every field kept, except that ``request.config`` and
        ``source_request.config`` are compacted with ``compact_config`` and the
        frozen ``plant_profile`` matrix rows are replaced by counts and volumes.
    """
    result = {key: copy.deepcopy(value) for key, value in draft.items()
              if key not in {"request", "source_request", "plant_profile"}}
    for key in ("request", "source_request"):
        if key in draft:
            result[key] = _compact_request(draft[key])
    if "plant_profile" in draft:
        result["plant_profile"] = _compact_profile(draft["plant_profile"])
    return result


def compact_validation(validation: Mapping[str, Any]) -> dict[str, Any]:
    """Return a model-facing draft validation with the effective config compacted.

    Args:
        validation: ``WorkspaceService.validate_draft`` output.

    Returns:
        The same fields (``valid``, ``errors``, ``warnings``, ``pv_mode_scope``,
        ``counts``, ``applied_constraints``, ``matrix_coverage``,
        ``profile_status``, ``revision``) unchanged, with only
        ``effective_request.config`` compacted.
    """
    result = {key: copy.deepcopy(value) for key, value in validation.items() if key != "effective_request"}
    if "effective_request" in validation:
        result["effective_request"] = _compact_request(validation["effective_request"])
    return result
