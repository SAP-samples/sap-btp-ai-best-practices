#!/usr/bin/env python3
"""
Quality gate for a handover Markdown file before (and after) building the .docx.

Usage:
    python check_handover.py handover.md [--inventory inventory.json] [--repo <repo_path>] [--json report.json]
    python check_handover.py handover/handover.md --inventory handover/inventory.json --repo . --bpd handover/bpd/bpd.json
    python check_handover.py --self-test

Checks
  1. Required chapters present (overview, usage, architecture, backend/API, AI capabilities,
     security, configuration/deployment, path to production, glossary)
  2. Figures exist on disk; an architecture figure is present
  3. API coverage: every HTTP route in the inventory is mentioned (path match, params normalised)
  4. Env-var coverage (warning only)
  5. Secret leakage: generic secret patterns + literal secret VALUES found in the repo's
     .env / manifest files (values are compared in memory and never printed)
  6. Leftover placeholders (TODO, TBD, lorem, XXX, <fill ...>)
  7. House rules: no citations of repo docs/ files; no mention of the SAP AI Ethics self-assessment or of prompt
     data privacy; the platform is 'SAP Business AI Platform' ("BTP" is allowed only inside a service name such as
     'SAP BTP Destination Service', never for the platform itself); no time/effort estimates in the path-to-production
     part; gap/finding IDs used outside their table must say where they are defined
  8. --forbid terms (anonymisation) across markdown, figure sources, file names and the built .docx
  9. ROM cost calculation: section present (warning), every row classified Shared / Solution-specific,
     total equals the sum of the rows, no scenarios or annual figures, no estimate-tool name
 10. --bpd bpd.json (from bpd_extract.py): the handover absorbs the BPD, so "BPD" (also in file names and labels) and
     "Business Scenario Design" are errors anywhere in the markdown; a 'Business scenario' chapter and a BR-xx
     requirements table must exist (errors); every BPD requirement must have its mirrored BR ID (BPD 5 -> BR-05) and every BPD out-of-scope process
     must be named in Part V or VI (warnings). Files next to bpd.json are skipped by the --forbid scan.
Exit code 1 on any error (secrets, missing chapters, house-rule violations, forbidden terms).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REQUIRED = {
    "overview": r"at a glance|overview|executive summary|introduction",
    "usage": r"how to use|using the application|user guide|usage|walkthrough|end-to-end",
    "architecture": r"architecture",
    "backend_api": r"\bapi\b|endpoint|backend",
    "ai_capabilities": r"\bai\b|genai|llm|agent|machine learning|optimi[sz]ation|model",
    "security": r"security|authentication|authori[sz]ation",
    "configuration": r"configuration|deployment|operations|environment",
    "production": r"production|next steps|roadmap|path to",
    "glossary": r"glossary",
}
SECRET_PATTERNS = [
    (r"(?i)\b(api[_-]?key|secret|password|passwd|client[_-]?secret|token)\b\s*[:=]\s*['\"]?(?!<|\(\(|\$\{|\*{3}|x{4}|\.\.\.)[A-Za-z0-9/+_\-\.]{16,}", "assignment of a secret-looking value"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "private key"),
    (r"\beyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{10,}", "JWT"),
    (r"\bAKIA[0-9A-Z]{16}\b", "AWS access key"),
    (r"\bsk-[A-Za-z0-9]{20,}", "OpenAI-style key"),
    (r"://[^/\s:@]+:[^@\s/]{6,}@", "credentials in URL"),
]
# Words that make "BTP <Word>" name the platform rather than a service ("BTP Platform", "BTP Subaccount")
BTP_GENERIC = "Platform|Application|Applications|Layer|Subaccount|Subaccounts|Account|Global|Landscape|Cockpit|Environment|Tenant"
# Content rules the author of this skill wants enforced in every handover
BANNED_TOPICS = [
    (r"(?i)\bAI[- ]ethics\b|ethics (self-)?assessment|responsible[- ]AI assessment", "SAP AI Ethics self-assessment must not be mentioned"),
    (r"(?i)privacy[^.\n]{0,60}\bprompts?\b|\bprompts?\b[^.\n]{0,60}privacy", "data privacy of prompts must not be discussed"),
    (r"\bBTP\b(?!,?[ \t]+(?!(?:" + BTP_GENERIC + r")\b)[A-Z])|(?i:business technology platform)|\bBAIP\b",
     "use 'SAP Business AI Platform' for the platform (\"BTP\" only inside a service name such as 'SAP BTP Destination Service'; never BAIP)"),
    (r"(?i)discovery center", "the ROM cost section must not name the tool that produced the estimate"),
]
# With --bpd: the handover never names its source document (bpd_flow.png, fig-bpd-1 and "BPD role" all count)
BPD_MENTION = r"(?i)(?<![a-z])bpd(?![a-z])|business scenario design"
DOCS_CITATION = r"(?<![\w/.-])docs/[\w\-./ ]+?\.(?:md|markdown|rst|txt|json|adoc)\b"
TIME_ESTIMATE = (r"(?i)\b\d+(?:[.,]\d+)?\s*(?:[-–]|to)?\s*\d*\s*(?:person[- ]?|man[- ]?)?(?:days?|weeks?|months?|sprints?|quarters?|PDs?|FTEs?|hours?)\b"
                 r"|\bQ[1-4]\s*20\d\d\b|\beffort\b\s*[|:]|\|\s*effort\s*\|")
PLACEHOLDERS = r"\bTODO\b|\bTBD\b|lorem ipsum|\bXXX\b|<fill[^>]*>|\[insert[^\]]*\]"
SECRET_NAME = re.compile(r"(?i)(secret|password|passwd|pwd|token|api[_-]?key|private[_-]?key|client[_-]?secret|credential)")


def norm_path(p: str) -> str:
    p = re.sub(r"\{[^}]+\}|<[^>]+>|:[A-Za-z_]\w*", "{}", p.strip())
    return p.rstrip("/") or "/"


def repo_secret_values(repo: Path):
    vals = {}
    for p in list(repo.rglob(".env*")) + list(repo.rglob("manifest*.y*ml")) + list(repo.rglob("default-env.json")) + list(repo.rglob("vars*.y*ml")):
        if any(part in ("node_modules", ".venv", ".git") for part in p.parts) or p.name.endswith((".example", ".sample", ".template")):
            continue
        try:
            t = p.read_text(errors="ignore")
        except Exception:
            continue
        for m in re.finditer(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*(.+?)\s*$", t, re.M):
            k, v = m.group(1), m.group(2).strip().strip("\"'")
            if SECRET_NAME.search(k) and len(v) >= 8 and not v.startswith(("((", "${", "<", "$")):
                vals[v] = k
        for m in re.finditer(r'"(clientsecret|password|secret|apikey|api_key|client_secret)"\s*:\s*"([^"]{8,})"', t, re.I):
            vals[m.group(2)] = m.group(1)
    return vals


def _norm(s: str) -> str:
    """Lower-case, hyphens and punctuation to spaces, collapsed whitespace (for loose name matching)."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", s.lower())).strip()


def oos_keys(name: str) -> tuple[str, list[str]]:
    """Short label and search keys for a BPD out-of-scope process name.

    "Global CIB (Change Implementation Board) Review Coordination" -> label "Global CIB", keys: its first two
    words, its first and last word, and the parenthetical ("change implementation board"). The process counts as
    named when any key appears in Part V/VI.

    Returns:
        (label, normalised keys)
    """
    paren = re.findall(r"\(([^)]+)\)", name)
    words = re.sub(r"\([^)]*\)", " ", name).split()
    label = " ".join(words[:2])
    keys = {_norm(label), _norm(f"{words[0]} {words[-1]}") if words else ""} | {_norm(x) for x in paren}
    return label, [k for k in keys if k]


def check_bpd(code_free: str, bpd: dict, report: dict, full_text: str = "") -> None:
    """Check the handover against a BPD extract (bpd.json from bpd_extract.py).

    Errors: "BPD" or "Business Scenario Design" anywhere in the markdown incl. front matter, file names and labels
    (full_text, fenced code removed); no 'Business scenario' heading; no requirements table (rows whose first cell
    is BR-xx).
    Warnings: a BPD requirement without its mirrored BR ID (BPD "5" -> BR-05); a BPD out-of-scope process not
    named between the Part V heading (or 'Known limits' if there is no Part V) and the first appendix.
    """
    scan = re.sub(r"```.*?```", "", full_text or code_free, flags=re.S)
    for i, ln in enumerate(scan.splitlines(), 1):
        m = re.search(BPD_MENTION, ln)
        if m:
            report["errors"].append(f"line {i}: names the source document ('{ln.strip()[:60]}'); write the content in the "
                                    "handover's own voice (see references/bpd-merge.md)")
    if not re.search(r"^#{1,2}\s+.*\bbusiness scenario\b", code_free, re.M | re.I):
        report["errors"].append("--bpd given but no 'Business scenario' chapter (see references/bpd-merge.md)")
    brs = set(re.findall(r"^\|\s*\**(BR-\d+)\**\s*\|", code_free, re.M))
    if not brs:
        report["errors"].append("--bpd given but no requirements table with BR-xx IDs in the first column (see references/bpd-merge.md)")
    report["stats"]["bpd_requirements"] = len(bpd.get("requirements", []))
    report["stats"]["br_ids"] = len(brs)
    missing = []
    for r in bpd.get("requirements", []):
        m = re.fullmatch(r"\D*(\d+)", r["id"].strip())
        expected = f"BR-{int(m.group(1)):02d}" if m else None
        if brs and expected not in brs and not (expected is None and r["id"] in code_free):
            missing.append(f"{r['id']} ({expected or 'no BR ID'})")
    if missing:
        report["warnings"].append(f"{len(missing)} BPD requirements not in the requirements table: " + ", ".join(missing))
    start = re.search(r"^#\s+Part V\b.*$", code_free, re.M) or re.search(r"^#{1,2}\s+.*known limits.*$", code_free, re.M | re.I)
    if not start:
        report["warnings"].append("no Part V / 'Known limits' heading: BPD out-of-scope processes not checked")
        return
    end = re.search(r"^#\s+Appendix", code_free[start.end():], re.M)
    seg = _norm(code_free[start.start(): start.end() + end.start()] if end else code_free[start.start():])
    unnamed = []
    for o in bpd.get("out_of_scope", []):
        label, keys = oos_keys(o["name"])
        if not any(k in seg for k in keys):
            unnamed.append(label)
    if unnamed:
        report["warnings"].append("BPD out-of-scope processes not named in Part V/VI (add them to Known limits and the roadmap): "
                                  + ", ".join(unnamed))


def self_test() -> None:
    """Assert the BTP naming rule and the out-of-scope key logic on known cases."""
    btp = next(pat for pat, _ in BANNED_TOPICS if "BTP" in pat)
    for ok in ["SAP BTP Destination Service", "SAP BTP Audit Log", "SAP BTP, Cloud Foundry Runtime", "SAP BTPEA"]:
        assert not re.search(btp, ok), ok
    for bad in ["hosted on SAP BTP.", "the BTP application layer", "SAP BTP subaccount", "SAP BTP Subaccount",
                "SAP Business Technology Platform (BTP)", "BTP-based", "SAP BTP\nDestination", "BAIP"]:
        assert re.search(btp, bad), bad
    label, keys = oos_keys("Global CIB (Change Implementation Board) Review Coordination")
    assert label == "Global CIB" and "global cib" in keys and "change implementation board" in keys, (label, keys)
    assert "make or buy recommendation" in oos_keys("Make-or-Buy Recommendation Engine")[1]
    for bad in ["This chapter summarises the BPD.", "| BPD role |", "figures/bpd_flow_1.png", "id=fig-bpd-flow",
                "the Business Scenario Design document"]:
        assert re.search(BPD_MENTION, bad), bad
    for ok in ["the business process flow", "UPDATE handling", "bpdx"]:
        assert not re.search(BPD_MENTION, ok), ok
    print("self-test passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("markdown", nargs="?")
    ap.add_argument("--inventory")
    ap.add_argument("--repo")
    ap.add_argument("--json")
    ap.add_argument("--forbid", action="append", default=[], help="term that must not appear anywhere (e.g. a customer name); repeatable, case-insensitive")
    ap.add_argument("--docx", help="built .docx to scan for forbidden terms (all XML parts incl. metadata)")
    ap.add_argument("--bpd", help="bpd.json from bpd_extract.py: check the Business scenario chapter and BPD coverage")
    ap.add_argument("--self-test", action="store_true", help="run the rule self-checks and exit")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.markdown:
        ap.error("the handover markdown file is required")
    md_path = Path(a.markdown)
    text = md_path.read_text(encoding="utf-8")
    body = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.S)
    headings = re.findall(r"^(#{1,3})\s+(.+)$", body, re.M)
    h_text = [h[1] for h in headings]
    report = {"errors": [], "warnings": [], "stats": {}}

    # 1 chapters
    h1 = [h for lvl, h in headings if lvl in ("#", "##")]
    missing = [k for k, pat in REQUIRED.items() if not any(re.search(pat, h, re.I) for h in h1)]
    if missing:
        report["errors"].append(f"missing required chapters: {', '.join(missing)}")
    report["stats"]["chapters_h1"] = sum(1 for lvl, _ in headings if lvl == "#")
    report["stats"]["sections_h2"] = sum(1 for lvl, _ in headings if lvl == "##")

    # 2 figures
    figs = re.findall(r"!\[([^\]]*)\]\(([^)\s]+)", body)
    report["stats"]["figures"] = len(figs)
    for alt, src in figs:
        if not (md_path.parent / src).exists():
            report["errors"].append(f"figure file missing: {src}")
    if not any(re.search(r"architect", alt + src, re.I) for alt, src in figs):
        report["warnings"].append("no figure whose caption/path mentions 'architecture'")
    mock = [f for f in figs if re.search(r"scr_|screenshot|placeholder", f[0] + f[1], re.I)]
    report["stats"]["ui_figures"] = len(mock)

    # 3/4 coverage
    if a.inventory:
        inv = json.loads(Path(a.inventory).read_text())
        md_norm = norm_path(body)
        paths = sorted({norm_path(r["path"]) for r in inv.get("http_routes", []) if r.get("framework") != "cap"})
        body_paths = {norm_path(m) for m in re.findall(r"(/[A-Za-z0-9_\-\{\}/<>:\.]+)", body)}
        uncovered = [p for p in paths if p not in body_paths and p not in md_norm]
        # method+path: the method must appear on a line (table row) that mentions the path
        lines = [norm_path(l) for l in body.splitlines()]
        pairs = sorted({(r["method"].split(",")[0].upper(), norm_path(r["path"])) for r in inv.get("http_routes", []) if r.get("framework") != "cap"})
        pair_missing = [f"{m} {p}" for m, p in pairs
                        if not any(re.search(rf"\b{m}\b", l) and re.search(re.escape(p) + r"(?![\w/{])", l) for l in lines)]
        report["stats"]["api_paths"] = len(paths)
        report["stats"]["api_paths_documented"] = len(paths) - len(uncovered)
        report["stats"]["api_operations"] = len(pairs)
        report["stats"]["api_operations_documented"] = len(pairs) - len(pair_missing)
        if uncovered:
            report["warnings"].append(f"{len(uncovered)} paths not mentioned: " + ", ".join(uncovered[:40]))
        if pair_missing:
            report["warnings"].append(f"{len(pair_missing)} method+path operations not found on one line (endpoint table row): " + ", ".join(pair_missing[:40]))
        envs = [e["name"] for e in inv.get("env_vars", [])]
        env_missing = [e for e in envs if e not in body]
        report["stats"]["env_vars"] = len(envs)
        report["stats"]["env_vars_documented"] = len(envs) - len(env_missing)
        if env_missing:
            report["warnings"].append(f"{len(env_missing)} env vars not mentioned: " + ", ".join(env_missing[:40]))

    # 5 secrets
    leaks = []
    for pat, what in SECRET_PATTERNS:
        for m in re.finditer(pat, body):
            line = body[: m.start()].count("\n") + 1
            leaks.append(f"line {line}: {what}")
    if a.repo:
        for val, key in repo_secret_values(Path(a.repo)).items():
            if val in body:
                leaks.append(f"literal value of repo secret '{key}' appears in the document")
    if leaks:
        report["errors"].append("possible secrets: " + "; ".join(leaks))

    # 5b forbidden terms (anonymisation)
    if a.forbid:
        hits = []
        pat = re.compile("|".join(re.escape(t) for t in a.forbid), re.I)
        n = len(pat.findall(text))
        if n:
            hits.append(f"{n}x in markdown")
        figdir = md_path.parent
        bpd_dir = Path(a.bpd).resolve().parent if a.bpd else None
        for f in figdir.rglob("*"):
            if bpd_dir and bpd_dir in f.resolve().parents:
                continue  # the BPD extract is customer source material, not part of the deliverable
            if f.is_file() and f.suffix.lower() in (".json", ".html", ".md", ".svg", ".css", ".txt") and f != md_path:
                k = len(pat.findall(f.read_text(errors="ignore")))
                if k:
                    hits.append(f"{k}x in {f.relative_to(figdir)}")
            if f.is_file() and pat.search(f.name):
                hits.append(f"file name {f.relative_to(figdir)}")
        if a.docx:
            import zipfile
            z = zipfile.ZipFile(a.docx)
            for nme in z.namelist():
                if nme.endswith((".xml", ".rels")):
                    k = len(pat.findall(z.read(nme).decode("utf-8", "ignore")))
                    if k:
                        hits.append(f"{k}x in docx:{nme}")
        if hits:
            report["errors"].append("forbidden terms found: " + "; ".join(hits))
        report["warnings"].append("forbidden-term scan does not read text inside PNG images: check figures visually")

    # 5c content rules
    code_free = re.sub(r"```.*?```", "", body, flags=re.S)
    for pat, why in BANNED_TOPICS:
        for m in re.finditer(pat, code_free):
            report["errors"].append(f"{why}: line {code_free[:m.start()].count(chr(10)) + 1} '{m.group(0)[:60]}'")
    for m in re.finditer(DOCS_CITATION, code_free):
        report["errors"].append(f"cites a repository docs/ file (use its content, not the file name): line {code_free[:m.start()].count(chr(10)) + 1} '{m.group(0)}'")
    # no time/effort estimates in the path-to-production part
    heads = [(m.start(), m.group(1), m.group(2)) for m in re.finditer(r"^(#{1,2})\s+(.+)$", code_free, re.M)]
    prod_end = None
    prod_start = next((pos for pos, lvl, h in heads if lvl == "#" and re.search(r"path to production|production readiness|part vi", h, re.I)), None)
    if prod_start is not None:
        prod_end = next((pos for pos, lvl, h in heads if pos > prod_start and lvl == "#" and re.match(r"(?i)appendix", h)), len(code_free))
        seg = code_free[prod_start:prod_end]
        for m in re.finditer(TIME_ESTIMATE, seg):
            line = code_free[:prod_start + m.start()].count(chr(10)) + 1
            report["errors"].append(f"time/effort estimate in path to production (not allowed): line {line} '{m.group(0).strip()[:40]}'")
    # IDs (G-01, SEC-3, Q-07 …) used outside their defining table must say where they are defined
    defined = {}
    for m in re.finditer(r"^\|\s*\**([A-Z]{1,4}-\d{1,3})\**\s*\|", code_free, re.M):
        defined.setdefault(m.group(1), code_free[:m.start()].count(chr(10)) + 1)
    if defined:
        prod_lines = (code_free[:prod_start].count(chr(10)) + 1, code_free[:prod_end].count(chr(10)) + 1) if prod_start is not None else (0, -1)
        for i, ln in enumerate(code_free.splitlines(), 1):
            if prod_lines[0] <= i <= prod_lines[1]:
                continue  # inside Part VI the register is next to the reference
            if ln.lstrip().startswith("|") and re.match(r"^\|\s*\**[A-Z]{1,4}-\d{1,3}", ln.strip()):
                continue  # a defining row
            ids = [x for x in re.findall(r"\b([A-Z]{1,4}-\d{1,3})\b", ln) if x in defined]
            if ids and not re.search(r"@tbl-|Table \d|\]\(#|Part [IVX]+|[Cc]hapter \d|[Ss]ection \d|path to production|gap register|open questions", ln):
                report["errors"].append(f"line {i}: {', '.join(sorted(set(ids)))} referenced without saying where it is defined "
                                          f"(add e.g. 'see @tbl-gaps' or a link to the chapter)")

    # ROM cost calculation (Part VI): present, every row classified, total = sum of rows
    rom = re.search(r"^#{1,3}\s+.*\bROM cost\b.*$", code_free, re.M | re.I)
    if not rom:
        report["warnings"].append("no 'ROM cost calculation' section in Part VI (see references/rom-cost.md)")
    else:
        nxt = re.search(r"^#{1,3}\s+", code_free[rom.end():], re.M)
        sec = code_free[rom.end(): rom.end() + nxt.start()] if nxt else code_free[rom.end():]
        rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in sec.splitlines()
                if l.strip().startswith("|") and not re.match(r"^\|?\s*:?-{3,}", l.strip())]
        if rows:
            hdr = [h.lower() for h in rows[0]]
            ci = next((i for i, h in enumerate(hdr) if h == "cost"), None)
            ti = next((i for i, h in enumerate(hdr) if "cost type" in h), None)
            amt = lambda v: float(re.sub(r"[*,\s]|EUR", "", v) or "nan")
            total_rows = [r for r in rows[1:] if re.search(r"(?i)total", r[0])]
            lines_ = [r for r in rows[1:] if r not in total_rows]
            if ci is None or ti is None:
                report["errors"].append("ROM cost table needs 'Cost' and 'Cost type' columns (see references/rom-cost.md)")
            else:
                for r in lines_:
                    if len(r) <= ti or not re.search(r"(?i)shared|solution-specific", r[ti]):
                        report["errors"].append(f"ROM cost row '{r[0]}' is not classified as Shared / Solution-specific")
                try:
                    s_ = round(sum(amt(r[ci]) for r in lines_), 2)
                    if total_rows and abs(amt(total_rows[0][ci]) - s_) > 0.01:
                        report["errors"].append(f"ROM cost total {total_rows[0][ci]} != sum of rows {s_:,.2f}")
                    if not total_rows:
                        report["errors"].append("ROM cost table has no total row")
                except (ValueError, IndexError):
                    report["errors"].append("ROM cost table: could not parse the Cost column (use 1,020.00 style amounts)")
        if re.search(r"(?i)what this means|greenfield|per year|annual", sec):
            report["errors"].append("ROM cost section must not add scenarios or annual figures (see references/rom-cost.md)")

    # 10 BPD coverage
    if a.bpd:
        check_bpd(code_free, json.loads(Path(a.bpd).read_text(encoding="utf-8")), report, text)

    # 6 placeholders
    ph = [(body[: m.start()].count("\n") + 1, m.group(0)) for m in re.finditer(PLACEHOLDERS, body)]
    if ph:
        report["warnings"].append("placeholders: " + ", ".join(f"line {l} '{t}'" for l, t in ph[:20]))

    report["stats"]["words"] = len(re.findall(r"\w+", body))
    report["stats"]["tables"] = len(re.findall(r"^\|?\s*:?-{3,}", body, re.M))
    report["stats"]["callouts"] = len(re.findall(r"^>\s*\[!\w+\]", body, re.M))
    report["stats"]["gap_callouts"] = len(re.findall(r"^>\s*\[!GAP\]", body, re.M))
    report["ok"] = not report["errors"]
    out = json.dumps(report, indent=2)
    print(out)
    if a.json:
        Path(a.json).write_text(out)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
