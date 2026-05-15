import re
from typing import Any


def parse_csv(text: str) -> dict[str, Any]:
    text = text.lstrip("﻿")
    rows: list[dict[str, str]] = []
    pos = 0
    length = len(text)

    def parse_field() -> str:
        nonlocal pos
        if pos < length and text[pos] == '"':
            pos += 1
            field = []
            while pos < length:
                if text[pos] == '"':
                    pos += 1
                    if pos < length and text[pos] == '"':
                        field.append('"')
                        pos += 1
                    else:
                        break
                else:
                    field.append(text[pos])
                    pos += 1
            return "".join(field)
        else:
            field = []
            while pos < length and text[pos] not in (",", "\n", "\r"):
                field.append(text[pos])
                pos += 1
            return "".join(field).strip()

    def parse_line() -> list[str]:
        nonlocal pos
        fields = []
        while pos < length and text[pos] not in ("\n", "\r"):
            fields.append(parse_field())
            if pos < length and text[pos] == ",":
                pos += 1
        if pos < length and text[pos] == "\r":
            pos += 1
        if pos < length and text[pos] == "\n":
            pos += 1
        return fields

    headers = [h.strip() for h in parse_line()]

    while pos < length:
        if text[pos] in ("\r", "\n"):
            if text[pos] == "\r":
                pos += 1
            if pos < length and text[pos] == "\n":
                pos += 1
            continue
        fields = parse_line()
        if not fields or (len(fields) == 1 and fields[0] == ""):
            continue
        row = {}
        for i, h in enumerate(headers):
            row[h] = fields[i] if i < len(fields) else ""
        rows.append(row)

    return {"headers": headers, "rows": rows}


def parse_amount(s: Any) -> float:
    if s is None or s == "":
        return float("nan")
    cleaned = re.sub(r",(?=\d)", "", str(s)).strip()
    try:
        return float(cleaned)
    except ValueError:
        return float("nan")
