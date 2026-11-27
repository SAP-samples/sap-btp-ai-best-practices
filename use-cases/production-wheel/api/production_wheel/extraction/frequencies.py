"""Production frequency inventory of a primary workbook.

The import admits FINIs by weekly production frequency code (``01W``, ``02W``,
...). Which codes exist depends on the uploaded workbook, so planners must not be
asked to type them. This module reads the codes straight from the FINI table and
resolves the ``all_available`` request into the exact list found in the source.

Example::

    inventory = frequency_inventory(find_table(workbook, "FINI_grouping_for_wheel"))
    codes = [item["code"] for item in inventory["codes"]]
"""

from __future__ import annotations

import collections

from production_wheel.extraction.common import TableData, as_text
from production_wheel.extraction.primary_workbook import normalize_production_frequency

# Source column that carries the planner-maintained frequency label.
FREQUENCY_COLUMN = "Production Frequency"
# Settings value meaning "every weekly code present in the uploaded workbook".
ALL_AVAILABLE = "all_available"
# Display value for rows whose frequency cell is empty.
BLANK_VALUE = "(blank)"


def is_all_available(value: object) -> bool:
    """Return whether a ``modeled_frequencies`` setting asks for every workbook code.

    Args:
        value: Raw ``modeled_frequencies`` setting (list of codes or the keyword).
    Returns:
        True only for the case-insensitive string ``all_available``.
    """
    return isinstance(value, str) and value.strip().lower() == ALL_AVAILABLE


def frequency_inventory(main: TableData) -> dict:
    """Count the production frequency values of the FINI table.

    Args:
        main: Extracted ``FINI_grouping_for_wheel`` table.
    Returns:
        Dict with ``codes`` (recognized weekly codes in code order, each with the
        first source ``label`` seen and its ``rows`` count), ``unrecognized``
        (source values without a weekly code, such as "not possible to determine",
        with their ``rows`` count) and ``total_rows``.
    """
    recognized: collections.Counter[str] = collections.Counter()
    labels: dict[str, str] = {}
    unrecognized: collections.Counter[str] = collections.Counter()
    for record in main.records:
        raw = as_text(record.get(FREQUENCY_COLUMN))
        code = normalize_production_frequency(raw)
        if code is None:
            unrecognized[raw or BLANK_VALUE] += 1
            continue
        recognized[code] += 1
        # Keep the planner-facing label ("01W. Weekly") for display only.
        labels.setdefault(code, raw)
    return {
        "codes": [
            {"code": code, "label": labels[code], "rows": recognized[code]}
            for code in sorted(recognized)
        ],
        "unrecognized": [
            {"value": value, "rows": count}
            for value, count in sorted(unrecognized.items())
        ],
        "total_rows": len(main.records),
    }


def available_codes(inventory: dict) -> list[str]:
    """Return the recognized weekly codes of an inventory, raising when there are none.

    Args:
        inventory: Result of ``frequency_inventory``.
    Returns:
        Sorted list of codes such as ``["01W", "02W", "16W"]``.
    Raises:
        ValueError: The workbook has no recognizable weekly frequency code.
    """
    codes = [item["code"] for item in inventory["codes"]]
    if not codes:
        raise ValueError(
            "The workbook has no recognizable weekly production frequency code "
            f"in the '{FREQUENCY_COLUMN}' column"
        )
    return codes


def undetermined_frequency_issues(inventory: dict) -> list[dict]:
    """Report source frequency values that cannot be modeled as review warnings.

    Args:
        inventory: Result of ``frequency_inventory``.
    Returns:
        One warning per unrecognized source value, in the validation-issue shape.
        These FINIs stay out of scope whatever frequency setting is chosen.
    """
    return [
        {
            "rule_id": "PRODUCTION_FREQUENCY_UNDETERMINED",
            "severity": "warning",
            "entity": "fini_master",
            "entity_key": item["value"],
            "observed": f"{item['rows']} FINI rows",
            "expected": "a weekly frequency code such as 01W",
            "action": "These FINIs stay out of scope. Correct the source frequency to model them.",
        }
        for item in inventory["unrecognized"]
    ]
