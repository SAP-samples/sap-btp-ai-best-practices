"""Check that the material search finds the right material for every line of the test orders.

The agent can only choose among the candidates ``search_materials`` returns, so every expected
material must be in the candidate list (recall). This check runs the same ranking as the tool
(``rank_materials`` with the customer's own codes and the sales-org filter), without the LLM,
over ``data/seed/expected_lines.json`` (generated together with the master-data seed).

Examples (run from the api/ folder, credentials in api/.env):

    ../.venv/bin/python -m app.cli.check_material_matching
    ../.venv/bin/python -m app.cli.check_material_matching --limit 5 --show-all
    ../.venv/bin/python -m app.cli.check_material_matching --file ../data/seed/expected_lines.json

Each expected line: {document, line, customer, sales_org, printed_code, description, unit_price,
expected_matnr, optional search_query}. ``search_query`` is the English text the agent is told to
retry with when the printed description finds nothing; when present, the better of both
searches counts. Exit code 1 when any expected material is missing from the candidates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from app.ecc import master_data  # noqa: E402
from app.orders.matching import alpha_in, alpha_out, compact_id, rank_materials  # noqa: E402

DEFAULT_FILE = Path(__file__).resolve().parents[3] / "data" / "seed" / "expected_lines.json"


def rank_of(expected: str, line: dict, materials: list[dict], customer_codes: dict, limit: int) -> int | None:
    """Return the 1-based rank of the expected material among the candidates (None if absent).

    Args:
        expected: Expected MATNR (ALPHA format).
        line: Expected-line record.
        materials: Materials sold in the line's sales organisation.
        customer_codes: {normalised KDMAT: MATNR} of the line's customer.
        limit: Candidates per search (same as the tool's ``limit_per_line``).

    Returns:
        Best rank over the printed search and the optional English retry, or None.
    """
    ranks = []
    for description in filter(None, [line.get("description"), line.get("search_query")]):
        ranked = rank_materials(materials, limit=limit, codes=[line["printed_code"]], description=description,
                                customer_codes=customer_codes)
        ranks += [i for i, m in enumerate(ranked, start=1) if m["MATNR"] == expected]
    return min(ranks) if ranks else None


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Command-line arguments (defaults to sys.argv).

    Returns:
        0 when every expected material is among the candidates, otherwise 1.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE, help="expected lines JSON")
    parser.add_argument("--limit", type=int, default=8, help="candidates per line (tool default)")
    parser.add_argument("--show-all", action="store_true", help="print every line, not only misses")
    args = parser.parse_args(argv)

    lines = json.loads(args.file.read_text(encoding="utf-8"))
    all_materials = master_data.load_materials()
    codes_cache: dict[tuple[str, str], dict] = {}
    top1 = found = 0
    report = []
    for line in tqdm(lines, desc="Checking lines", unit="line"):
        vkorg = line["sales_org"]
        key = (line["customer"], vkorg)
        if key not in codes_cache:
            codes_cache[key] = {compact_id(k["KDMAT"]): k["MATNR"]
                                for k in master_data.get_customer_material_codes(line["customer"], vkorg) if k.get("KDMAT")}
        materials = [m for m in all_materials if any(s["VKORG"] == vkorg for s in m.get("SALES", []))]
        rank = rank_of(alpha_in(line["expected_matnr"], 18), line, materials, codes_cache[key], args.limit)
        top1 += rank == 1
        found += rank is not None
        if rank is None or args.show_all:
            report.append(f"{'MISS' if rank is None else f'#{rank}':>5}  {line['document'][:32]:<32} line {line['line']:>2}  "
                          f"{line['printed_code']:<22} {line['description'][:50]:<50} -> {alpha_out(line['expected_matnr'])}")
    print(*report, sep="\n")
    print(f"\n{len(lines)} lines: expected material among candidates {found}/{len(lines)}, ranked first {top1}/{len(lines)}")
    return 0 if found == len(lines) else 1


if __name__ == "__main__":
    sys.exit(main())
