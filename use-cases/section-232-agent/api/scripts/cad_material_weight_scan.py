#!/usr/bin/env python3
"""
Proof-of-concept scanner for CAD files exported from Creo/STEP.

It is intentionally conservative:
- STEP: searches for explicit material/mass/density/weight text/properties.
- Creo .prt/.asm: extracts simple Creo user parameters that are stored as text
  in the binary file, e.g. MATERIAL and WEIGHT, and lists material-like tokens.

For production Creo extraction, prefer Creo Toolkit / J-Link / OTK APIs because
mass properties and material assignment can be stored in proprietary binary
structures that are not safe to parse by regex alone.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

PARAMS = ["MATERIAL", "WEIGHT", "VIKT", "TREATMENT", "REMARK"]
STEP_TERMS = re.compile(
    r"\b(MATERIAL|MATERIAL_DESIGNATION|DENSITY|MASS|WEIGHT|STEEL|STAINLESS|"
    r"ALUMINIUM|ALUMINUM|AISI|ASTM|BRASS|BRONZE|COPPER|TITANIUM|ZINC|MAGNESIUM)\b",
    re.IGNORECASE,
)
MATERIAL_TOKEN_RE = re.compile(
    rb"\b(?:RUBBER_[A-Z0-9_]+|TMPMATERIAL|PTC_SYSTEM_MTRL_PROPS|[0-9A-Z_]*HPDC[0-9A-Z_\.]*?)\b"
)


def ascii_safe(raw: bytes) -> str | None:
    if raw and all(32 <= b < 127 for b in raw):
        return raw.decode("latin1", errors="replace")
    return None


def extract_creo_param(data: bytes, name: str) -> list[str]:
    """Extract simple text Creo parameters.

    Creo binary files often contain a serialized parameter table like:
        e3 NAME 00 e2 33 VALUE 00
    where e2 33 means a string parameter. Numeric parameters are detected but
    not decoded here because the internal representation is not plain IEEE-754.
    """
    key = name.encode("ascii")
    out: list[str] = []

    # Preferred exact serialized parameter table.
    pattern = rb"\xe3" + re.escape(key) + rb"\x00\xe2(.)"
    for m in re.finditer(pattern, data):
        typ = m.group(1)
        if typ == b"3":  # string parameter
            j = m.end()
            k = data.find(b"\x00", j, min(j + 200, len(data)))
            if k > j:
                raw = data[j:k]
                value = ascii_safe(raw)
                if value is None:
                    continue
            else:
                continue
        elif typ == b"2":  # numeric/double parameter in Creo internal encoding
            j = m.end()
            k = data.find(b"\xf6", j, min(j + 80, len(data)))
            raw = data[j:k if k > j else min(j + 24, len(data))]
            value = "<binary numeric value: " + raw[:24].hex() + ">"
        elif typ == b"4":
            value = "<integer parameter>"
        elif typ == b"5":
            value = "<logical parameter>"
        else:
            value = "<unknown parameter type>"
        if value not in out:
            out.append(value)

    if out:
        return out

    # Fallback for another nearby table structure in some files.
    start = 0
    while True:
        i = data.find(key, start)
        if i < 0:
            break
        start = i + len(key)
        if i > 0 and (65 <= data[i - 1] <= 90 or 97 <= data[i - 1] <= 122 or data[i - 1] == 95):
            continue
        chunk = data[i : i + 160]
        value = None
        m = chunk.find(b"\xe3\x33")
        if m >= 0:
            j = i + m + 2
            k = data.find(b"\x00", j, min(j + 80, len(data)))
            if k > j:
                value = ascii_safe(data[j:k])
        if value is not None and value not in out:
            out.append(value)
    return out


def scan_file(path: Path) -> dict[str, str]:
    data = path.read_bytes()
    name = path.name
    is_step = name.lower().endswith((".stp", ".step"))
    row = {
        "file": name,
        "kind": "STEP" if is_step else "Creo binary",
        "material_param": "",
        "weight_param": "",
        "vikt_param": "",
        "material_like_tokens": "",
        "explicit_step_material_mass_density_weight_terms": "",
        "assessment": "",
    }

    if is_step:
        text = data.decode("latin1", errors="ignore")
        # Only report actual textual terms, not numeric product IDs.
        terms = sorted(set(m.group(0).upper() for m in STEP_TERMS.finditer(text)))
        row["explicit_step_material_mass_density_weight_terms"] = "; ".join(terms)
        row["assessment"] = "no explicit material/mass/density/weight metadata found" if not terms else "review STEP terms"
        return row

    for p in PARAMS:
        vals = extract_creo_param(data, p)
        if p == "MATERIAL":
            row["material_param"] = "; ".join(vals[:5])
        elif p == "WEIGHT":
            row["weight_param"] = "; ".join(vals[:5])
        elif p == "VIKT":
            row["vikt_param"] = "; ".join(vals[:5])

    tokens = sorted(set(t.decode("latin1", errors="replace") for t in MATERIAL_TOKEN_RE.findall(data)))
    row["material_like_tokens"] = "; ".join(tokens[:20])
    material_is_useful = row["material_param"] and row["material_param"] != "-"
    weight_is_useful = row["weight_param"] and row["weight_param"] not in {"-", "."}
    if material_is_useful or weight_is_useful or row["vikt_param"]:
        row["assessment"] = "some extractable CAD parameters, but no metal composition in text layer"
    else:
        row["assessment"] = "no useful text material/weight parameter found"
    return row


def main(argv: list[str]) -> int:
    folder = Path(argv[1]) if len(argv) > 1 else Path(".")
    paths = sorted(
        p for p in folder.iterdir()
        if p.is_file()
        and p.name != Path(__file__).name
        and not p.name.lower().endswith((".csv", ".json", ".md"))
    )
    rows = [scan_file(p) for p in paths]
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0].keys()) if rows else [])
    writer.writeheader()
    writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
