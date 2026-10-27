#!/usr/bin/env python3
"""
Extract a Business Process / Scenario Design Document (BPD, .docx) into clean
Markdown + JSON, save its images, and cut framed process-flow figures into panels.

Usage:
    python bpd_extract.py <bpd.docx> --out handover/bpd
    python bpd_extract.py --self-test

Outputs (in --out):
  bpd.md       headings (numbered like the BPD's TOC), paragraphs, bullets and tables in document order
  bpd.json     the same content structured: sections, tables (caption, kind, header, rows), images,
               requirements (BPD's own IDs) and out-of-scope processes; check_handover.py --bpd reads it
  media/       every embedded image (img_NN.<ext>), plus img_NN_panel_K.png for figures made of
               several framed panels (e.g. a wide process flow holding two processes side by side)

Tables are cleaned: horizontally merged cells are de-duplicated, empty rows dropped, a full-width
title row becomes the caption, and a table that stacks several sub-tables (a new title row, or a
row with a different number of cells) is split into blocks. Each block gets a best-guess `kind`
from its section heading (requirements, objectives, pain_points, kpis, roles, data_model,
data_sources, process_scope, architecture, other). The script targets the SAP Business Scenario
Design template; adapt KINDS and the ID / in-scope column patterns for other templates.

Requires python-docx and Pillow (both installed in Step 0). EMF/WMF images are converted to PNG
with LibreOffice (`soffice`) when it is available; otherwise they are saved as-is with a warning.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "v": "urn:schemas-microsoft-com:vml",
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
}
# (pattern on the section heading or table caption, kind); first match wins
KINDS = [
    (r"requirement", "requirements"),
    (r"pain point", "pain_points"),     # before "objective": BPD pain-point headings end "(to achieve business objective)"
    (r"objective", "objectives"),
    (r"\bkpis?\b|performance indicator", "kpis"),
    (r"\broles?\b", "roles"),
    (r"data model", "data_model"),
    (r"data source|related system", "data_sources"),
    (r"process scope|\bscope\b", "process_scope"),
    (r"architecture", "architecture"),
]
ID_HEADER = r"(?i)\bid\b|\bno\.?$|^#$|req"      # column holding the requirement number
IN_SCOPE_HEADER = r"(?i)in[ -]?scope"            # process-scope column: "Yes" / "No - POC" / ...


# ----------------------------------------------------------------------------- panels
def _rules(binary: Image.Image, min_len: int) -> list[tuple[int, int, int]]:
    """Find straight horizontal rules in a binarised image.

    Args:
        binary: mode "L" image where ink is 0 and background 255.
        min_len: minimum run length in pixels for a rule.
    Returns:
        (y, x0, x1) per rule; adjacent rows with the same run (a 2-px line) are merged.
    """
    w, h = binary.size
    data = binary.tobytes()
    pat = re.compile(rb"\x00{%d,}" % min_len)
    out: list[tuple[int, int, int]] = []
    for y in range(h):
        for m in pat.finditer(data, y * w, (y + 1) * w):
            x0, x1 = m.start() - y * w, m.end() - y * w
            k = len(out) - 1
            dup = False
            while k >= 0 and y - out[k][0] <= 3:  # rules found in the last 3 pixel rows
                if abs(out[k][1] - x0) <= 3 and abs(out[k][2] - x1) <= 3:
                    dup = True  # same rule, next pixel row of a thick line
                    break
                k -= 1
            if not dup:
                out.append((y, x0, x1))
    return out


def find_panels(img: Image.Image, min_frac: float = 0.25, tol: float = 0.02) -> list[tuple[int, int, int, int]]:
    """Find framed panels (large outlined rectangles) in a figure.

    A panel is two vertical rules with the same vertical span joined by a horizontal rule at their
    top or bottom (one edge may be broken by a title chip). Nested rectangles are dropped, so boxes
    inside a panel never split it; white space inside a panel is ignored.

    Args:
        img: any PIL image (transparency is flattened on white).
        min_frac: a rule must be at least this fraction of the image height (vertical) or width
            (horizontal) long; also the minimum panel size.
        tol: position tolerance as a fraction of the image size.
    Returns:
        Panel boxes (x0, y0, x1, y1) in reading order; empty unless at least two panels exist.
    """
    rgba = img.convert("RGBA")
    flat = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    flat.alpha_composite(rgba)
    binary = flat.convert("L").point(lambda v: 0 if v < 245 else 255)
    w, h = binary.size
    tx, ty = tol * w, tol * h
    horiz = _rules(binary, int(min_frac * w * 0.5))  # side-by-side panels can each be < half the width
    vert = [(x, y0, y1) for x, y0, y1 in _rules(binary.transpose(Image.Transpose.TRANSPOSE), int(min_frac * h))]
    if len(vert) > 200:
        return []  # Photos and dense bitmaps give thousands of runs; framed diagrams give a handful

    def has_edge(y: int, xa: int, xb: int) -> bool:
        return any(abs(hy - y) <= ty and hx0 <= xa + tx and hx1 >= xb - tx for hy, hx0, hx1 in horiz)

    boxes = []
    for i, (xa, ya0, ya1) in enumerate(vert):
        for xb, yb0, yb1 in vert[i + 1:]:
            if xb - xa < min_frac * w * 0.5 or abs(ya0 - yb0) > ty or abs(ya1 - yb1) > ty:
                continue
            y0, y1 = min(ya0, yb0), max(ya1, yb1)
            if has_edge(y0, xa, xb) or has_edge(y1, xa, xb):
                boxes.append((xa, y0, xb, y1))
    inside = lambda a, b: a != b and a[0] >= b[0] - tx and a[1] >= b[1] - ty and a[2] <= b[2] + tx and a[3] <= b[3] + ty
    top = [b for b in boxes if not any(inside(b, o) for o in boxes)]
    top = sorted(set(top), key=lambda b: (b[1] // max(1, h // 3), b[0]))  # rows of panels, then left to right
    return top if len(top) >= 2 else []


def crop_panel(img: Image.Image, box: tuple[int, int, int, int], pad: int = 12) -> Image.Image:
    """Crop a panel, growing the frame box outwards until a blank line so titles on the frame edge stay in.

    Args:
        img: the source figure.
        box: (x0, y0, x1, y1) from find_panels.
        pad: extra white margin in pixels.
    Returns:
        The cropped RGB panel.
    """
    rgba = img.convert("RGBA")
    flat = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    flat.alpha_composite(rgba)
    g = flat.convert("L")
    w, h = g.size
    x0, y0, x1, y1 = box

    def blank(x_a, y_a, x_b, y_b):
        return g.crop((x_a, y_a, x_b + 1, y_b + 1)).getextrema()[0] >= 245

    cap_y, cap_x = int(0.1 * h), int(0.1 * w)
    # Top edge: container titles often float a few pixels above the frame, so jump small blank gaps.
    gap, run, y = max(16, int(0.03 * (box[3] - box[1]))), 0, y0
    while y > 0 and box[1] - y < cap_y and run < gap:
        y -= 1
        if blank(x0, y, x1, y):
            run += 1
        else:
            run, y0 = 0, y
    # Other edges stop at the first blank line, so a panel never grows into its neighbour.
    while y1 < h - 1 and y1 - box[3] < cap_y and not blank(x0, y1, x1, y1):
        y1 += 1
    while x0 > 0 and box[0] - x0 < cap_x and not blank(x0, y0, x0, y1):
        x0 -= 1
    while x1 < w - 1 and x1 - box[2] < cap_x and not blank(x1, y0, x1, y1):
        x1 += 1
    return flat.convert("RGB").crop((max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad + 1), min(h, y1 + pad + 1)))


# ----------------------------------------------------------------------------- docx walking
def cell_text(cell) -> str:
    """Return a table cell's text: paragraphs joined by newlines, whitespace collapsed."""
    parts = [re.sub(r"\s+", " ", p.text).strip() for p in cell.paragraphs]
    return "\n".join(p for p in parts if p)


def table_blocks(table) -> list[dict]:
    """Split a Word table into clean blocks.

    Args:
        table: python-docx Table.
    Returns:
        [{"caption", "header", "rows"}]; a full-width single-cell row starts a new block and becomes
        its caption, the next row is the header, and a row with a different cell count starts a
        new block whose header is that row.
    """
    blocks: list[dict] = []
    cur = None
    ncols = len(table.columns)
    for row in table.rows:
        seen, cells = set(), []
        for c in row.cells:  # horizontally merged cells repeat the same <w:tc>
            if id(c._tc) not in seen:
                seen.add(id(c._tc))
                cells.append(cell_text(c))
        if not any(cells):
            continue
        if len(cells) == 1 and ncols > 1:
            cur = {"caption": cells[0], "header": None, "rows": []}
            blocks.append(cur)
        elif cur is None or cur["header"] is None:
            if cur is None:
                cur = {"caption": "", "header": None, "rows": []}
                blocks.append(cur)
            cur["header"] = cells
        elif len(cells) != len(cur["header"]):
            cur = {"caption": "", "header": cells, "rows": []}
            blocks.append(cur)
        else:
            cur["rows"].append(cells)
    return [b for b in blocks if b["header"]]


def kind_of(*texts: str) -> str:
    """Classify a table block from its section heading and caption (see KINDS)."""
    for t in texts:
        for pat, kind in KINDS:
            if re.search(pat, t or "", re.I):
                return kind
    return "other"


def save_image(part, out: Path, n: int) -> tuple[Path, list[str]]:
    """Write an image part to media/img_NN.<ext>, converting EMF/WMF to PNG when soffice exists.

    Returns:
        (saved path, warnings)
    """
    ext = Path(part.partname).suffix.lower() or ".bin"
    dst = out / f"img_{n:02d}{ext}"
    dst.write_bytes(part.blob)
    if ext in (".emf", ".wmf"):
        soffice = shutil.which("soffice") or shutil.which("libreoffice")
        if not soffice:
            return dst, [f"{dst.name}: vector image kept as {ext}; install LibreOffice to convert it"]
        subprocess.run([soffice, "--headless", "--convert-to", "png", "--outdir", str(out), str(dst)],
                       capture_output=True, timeout=120)
        png = dst.with_suffix(".png")
        if png.exists():
            dst.unlink()
            return png, []
        return dst, [f"{dst.name}: LibreOffice could not convert it"]
    return dst, []


def extract(docx_path: Path, out: Path) -> dict:
    """Walk the BPD in body order and write bpd.md, bpd.json and media/.

    Args:
        docx_path: the BPD .docx.
        out: output folder (created).
    Returns:
        The bpd.json content.
    """
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = docx.Document(str(docx_path))
    media = out / "media"
    media.mkdir(parents=True, exist_ok=True)
    res = {"source": docx_path.name, "title": "", "sections": [], "tables": [], "images": [],
           "requirements": [], "out_of_scope": [], "warnings": []}
    md: list[str] = []
    counters = [0] * 9
    section = ""       # "2.3 High Level Business Requirements"
    started = False    # False until the first heading: cover page and TOC are skipped
    n_img = 0

    def images_in(el) -> list[str]:
        rids = [b.get(f"{{{NS['r']}}}embed") for b in el.iter(f"{{{NS['a']}}}blip")]
        rids += [v.get(f"{{{NS['r']}}}id") for v in el.iter(f"{{{NS['v']}}}imagedata")]
        return [r for r in rids if r]

    def add_images(el):
        nonlocal n_img
        for rid in images_in(el):
            part = doc.part.related_parts.get(rid)
            if part is None or not getattr(part, "blob", None):
                continue
            n_img += 1
            path, warns = save_image(part, media, n_img)
            res["warnings"] += warns
            info = {"file": f"media/{path.name}", "section": section, "cover": not started, "panels": []}
            try:
                with Image.open(path) as im:
                    info["size"] = list(im.size)
                    for k, box in enumerate([] if info["cover"] else find_panels(im), 1):
                        p = media / f"{path.stem}_panel_{k}.png"
                        crop_panel(im, box).save(p)
                        info["panels"].append(f"media/{p.name}")
            except Exception as e:  # unreadable format: keep the file, skip panels
                res["warnings"].append(f"{path.name}: not analysed ({e})")
            res["images"].append(info)
            if started:
                md.append(f"![{path.stem} ({'x'.join(map(str, info.get('size', [])))})]({info['file']})\n")
                for p in info["panels"]:
                    md.append(f"![{Path(p).stem}]({p})\n")

    for el in doc.element.body.iterchildren():
        tag = el.tag.split("}")[1]
        if tag == "p":
            p = Paragraph(el, doc)
            style = p.style.name if p.style is not None else ""
            text = re.sub(r"\s+", " ", p.text).strip()
            if style.lower().startswith("toc"):
                continue
            m = re.match(r"Heading (\d)$", style)
            if m:
                lvl = int(m.group(1))
                counters[lvl - 1] += 1
                counters[lvl:] = [0] * (9 - lvl)
                num = ".".join(str(c) for c in counters[:lvl])
                section = f"{num} {text}"
                started = True
                res["sections"].append({"number": num, "title": text, "level": lvl})
                md.append(f"{'#' * lvl} {section}\n")
            elif not started:
                if text and style.lower().startswith("_cover_title") and not res["title"]:
                    res["title"] = text
            elif text:
                runs = [r for r in p.runs if r.text.strip()]
                if runs and all(r.bold for r in runs) and len(text) < 80:
                    md.append(f"**{text}**\n")
                elif p._p.pPr is not None and p._p.pPr.numPr is not None:
                    md.append(f"- {text}")
                else:
                    md.append(f"{text}\n")
            add_images(el)
        elif tag == "tbl" and started:
            t = Table(el, doc)
            for b in table_blocks(t):
                kind = kind_of(section, b["caption"])
                res["tables"].append({"section": section, "kind": kind, **b})
                esc = lambda c: c.replace("|", "\\|").replace("\n", "<br>")
                md.append(f"\n**{b['caption'] or 'Table'}** (kind: {kind})\n")
                md.append("| " + " | ".join(map(esc, b["header"])) + " |")
                md.append("|" + "---|" * len(b["header"]))
                md += ["| " + " | ".join(map(esc, r)) + " |" for r in b["rows"]]
                md.append("")
                if kind == "requirements":
                    ci = next((i for i, hd in enumerate(b["header"]) if re.search(ID_HEADER, hd)), 0)
                    for r in b["rows"]:
                        if r[ci]:
                            rest = [c for i, c in enumerate(r) if i != ci]
                            res["requirements"].append({"id": r[ci], "text": rest[0] if rest else "", "row": r})
                if kind == "process_scope":
                    si = next((i for i, hd in enumerate(b["header"]) if re.search(IN_SCOPE_HEADER, hd)), None)
                    for r in b["rows"]:
                        if si is not None and re.match(r"(?i)\s*no\b", r[si]):
                            res["out_of_scope"].append({"name": r[0], "in_scope": r[si], "row": r})
            add_images(el)
    if not res["title"]:
        res["title"] = doc.core_properties.title or ""
    (out / "bpd.md").write_text(f"# {res['title'] or docx_path.stem}\n\n" + "\n".join(md) + "\n", encoding="utf-8")
    (out / "bpd.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return res


# ----------------------------------------------------------------------------- self-test
def self_test() -> None:
    """Assert that find_panels splits a synthetic two-panel flow and nothing else."""
    im = Image.new("RGBA", (1600, 400), (0, 0, 0, 0))  # transparent, like exported flows
    d = ImageDraw.Draw(im)
    d.rectangle((30, 60, 760, 370), outline=(230, 230, 230, 255), width=2)       # panel 1
    d.rectangle((820, 30, 1570, 390), outline=(230, 230, 230, 255), width=2)     # panel 2 (other span)
    d.rectangle((40, 30, 300, 50), fill=(200, 200, 200, 255))                    # title chip 10 px above the frame
    for x in (80, 300, 520, 870, 1300):                                          # flow boxes, with a wide
        d.rectangle((x, 180, x + 140, 240), outline=(0, 90, 200, 255), width=3)  # white gap inside panel 2
    panels = find_panels(im)
    assert len(panels) == 2, panels
    assert panels[0][0] < 40 and panels[1][0] > 800, panels
    crop = crop_panel(im, panels[0])
    assert crop.size[1] >= 340 + 24, crop.size  # grew over the gap to keep the title chip (y 30..370 + padding)
    assert find_panels(Image.new("RGB", (800, 200), "white")) == []
    print("self-test passed")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("docx", nargs="?")
    ap.add_argument("--out", default="bpd", help="output folder (default: ./bpd)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.docx:
        ap.error("a BPD .docx is required")
    res = extract(Path(a.docx), Path(a.out))
    kinds = {}
    for t in res["tables"]:
        kinds[t["kind"]] = kinds.get(t["kind"], 0) + 1
    print(f"sections: {len(res['sections'])}  tables: {len(res['tables'])} {kinds}")
    print(f"requirements: {len(res['requirements'])} ({', '.join(r['id'] for r in res['requirements'])})")
    print(f"out of scope: {len(res['out_of_scope'])} ({'; '.join(o['name'] for o in res['out_of_scope'])})")
    for i in res["images"]:
        tag = "cover" if i["cover"] else i["section"]
        print(f"image {i['file']} {i.get('size')} [{tag}]" + (f" -> {len(i['panels'])} panels" if i["panels"] else ""))
    for w in res["warnings"]:
        print("WARNING:", w, file=sys.stderr)
    print(f"wrote {Path(a.out) / 'bpd.md'}, {Path(a.out) / 'bpd.json'}")


if __name__ == "__main__":
    main()
