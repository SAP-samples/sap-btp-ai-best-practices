"""
Text Parser — parse SAP mixed-format fields that combine an ID with a description.

SAP sometimes returns fields in the format:
    "63001000 (Electricity and other Utilities)"
    "63001000 Electricity and other Utilities"
    "PC-1001 (South Region)"

This module extracts the ID and description separately WITHOUT destroying either.
"""

import logging
import re
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Pattern: "ID (Description)" — ID is the first non-space token, description is in parens
_PAREN_RE = re.compile(r'^(\S+)\s+\((.+)\)\s*$')

# Pattern: "ID Description" — ID is the first non-space token, rest is description
_SPACE_RE = re.compile(r'^(\S+)\s+(.+)$')


def parse_id_description(value: str) -> Tuple[str, str]:
    """
    Parse a mixed SAP field value into (id_part, description_part).

    Examples:
        "63001000 (Electricity and other Utilities)" → ("63001000", "Electricity and other Utilities")
        "63001000 Electricity and other Utilities"   → ("63001000", "Electricity and other Utilities")
        "63001000"                                   → ("63001000", "")
        ""                                           → ("", "")

    Rules:
    - The ID is ALWAYS the first whitespace-delimited token.
    - The description is everything after the ID (parentheses stripped if present).
    - NEVER modifies the ID part (no digit stripping, no numeric casting).
    """
    if not value or not isinstance(value, str):
        return ("", "")

    stripped = value.strip()
    if not stripped:
        return ("", "")

    # Try "ID (Description)" format first
    m = _PAREN_RE.match(stripped)
    if m:
        return (m.group(1).strip(), m.group(2).strip())

    # Try "ID Description" format
    m = _SPACE_RE.match(stripped)
    if m:
        return (m.group(1).strip(), m.group(2).strip())

    # No description — just an ID
    return (stripped, "")


def split_id_description_column(
    series: pd.Series,
    id_col_name: str,
    desc_col_name: str,
) -> pd.DataFrame:
    """
    Split a Series of mixed "ID (Description)" values into two separate columns.

    Returns a DataFrame with columns [id_col_name, desc_col_name].
    NaN values produce ("", "") pairs.

    This is the safe replacement for any regex digit-stripping logic.
    """
    def _safe_parse(val) -> Tuple[str, str]:
        if pd.isna(val):
            return ("", "")
        return parse_id_description(str(val))

    parsed = series.apply(_safe_parse)
    result = pd.DataFrame(
        parsed.tolist(),
        index=series.index,
        columns=[id_col_name, desc_col_name],
    )

    # Log if any values were split (i.e., had a description component)
    had_description = result[desc_col_name] != ""
    if had_description.any():
        logger.info(
            "split_id_description_column: %d/%d values in source column had "
            "embedded descriptions (e.g. 'ID (text)'). "
            "IDs stored in '%s', descriptions in '%s'.",
            had_description.sum(), len(series), id_col_name, desc_col_name,
        )

    return result