"""Compact, model-facing dataset list entries.

``WorkspaceService.list_datasets`` returns complete dataset records: per-table
source descriptions, row counts, input hashes, parameter sources and every
review finding. Each record is about 5 KB, so a handful of imports already
exceeds the assistant's per-tool context limit and the whole list is replaced
by an "output omitted" placeholder. The assistant only needs the list to pick a
dataset; ``inspect_dataset`` returns the full record of the chosen one.

``dataset_list_entry`` keeps the identifying and decision-relevant fields and
replaces the findings with counts per severity and the checks that raised them.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

# Top-level record fields kept as they are.
ENTRY_FIELDS = ("dataset_id", "name", "status", "revision", "created_at", "published_at", "removed_at")
# Metadata fields small enough and useful for choosing a dataset.
METADATA_FIELDS = ("plant", "block_count", "settings", "admission_counts")


def dataset_list_entry(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return one compact dataset entry for the assistant's dataset list.

    Args:
        record: Dataset record from ``WorkspaceService.list_datasets`` with
            ``dataset_id``, ``name``, ``status``, ``revision``, ``created_at``,
            ``metadata`` and ``issues``.

    Returns:
        The fields in ``ENTRY_FIELDS`` that are present, the metadata fields in
        ``METADATA_FIELDS``, ``issue_counts`` (findings per severity) and
        ``issue_rule_ids`` (sorted distinct checks with findings). Source tables,
        hashes, row counts and full findings are left to ``inspect_dataset``.
    """
    metadata = record.get("metadata") or {}
    issues = record.get("issues") or []
    entry = {key: record[key] for key in ENTRY_FIELDS if record.get(key) is not None}
    entry.update({key: metadata[key] for key in METADATA_FIELDS if key in metadata})
    entry["issue_counts"] = dict(sorted(Counter(str(issue.get("severity")) for issue in issues).items()))
    entry["issue_rule_ids"] = sorted({str(issue.get("rule_id")) for issue in issues if issue.get("rule_id")})
    return entry
