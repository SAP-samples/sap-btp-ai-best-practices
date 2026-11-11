"""Render canonical line alternatives in the source workbook's notation."""

from __future__ import annotations

import re

from .dataset_tables import parse_string_lines


def format_lines(lines: tuple[str, ...] | list[str], source_pattern: str | None) -> str:
    """Format exact line IDs using source slots/separators, without assigning a line.

    Inputs are the eligible alternatives and original cell text. Identical sets
    retain the cell verbatim. Numeric positional patterns retain whitespace and
    dash slots; slash/named alternatives retain their delimiter and source order.
    Unknown templates use explicit slash-separated string identifiers.
    """
    selected = set(map(str, lines))
    source = str(source_pattern or "")
    if selected == set(parse_string_lines(source) or ()):
        return source
    tokens = re.findall(r"\S+", source)
    positional = "-" in tokens and all(
        token == "-" or token == str(i + 1) for i, token in enumerate(tokens)
    )
    if positional and selected <= {str(i + 1) for i in range(len(tokens))}:
        positions = iter(range(1, len(tokens) + 1))

        def replace_slot(match):
            """Render the next positional slot while preserving surrounding whitespace."""
            line = str(next(positions))
            return line if line in selected else "-"

        return re.sub(r"\S+", replace_slot, source)
    order = [token for token in re.split(r"[\s/|;,]+", source) if token in selected]
    order = list(dict.fromkeys(order))
    order.extend(sorted(selected - set(order)))
    separator = re.search(r"\s*[/|;,]\s*", source)
    return (separator.group() if separator else " / ").join(order)
