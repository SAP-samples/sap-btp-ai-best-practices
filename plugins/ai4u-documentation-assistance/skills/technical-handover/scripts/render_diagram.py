#!/usr/bin/env python3
"""
Render SAP-styled diagrams (architecture graphs, step flows, sequence diagrams) from a JSON spec to PNG.

Usage:
    python render_diagram.py spec.json out.png [--html-only] [--brand ../assets/brand.json]

Spec types (full reference with examples: references/diagrams.md):
  "graph"    - components on a grid (col,row) with labelled arrows and optional boundary groups
  "steps"    - numbered process flow, wraps into rows
  "sequence" - participants with lifelines and ordered messages
Common keys: type, title, subtitle, width (default 1800), note (footer band), legend
"""
from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from render_png import screenshot  # noqa: E402

BRAND = json.loads((HERE.parent / "assets" / "brand.json").read_text())


def esc(s):
    return html.escape(str(s))


def wrap_count(text, chars_per_line):
    """number of lines after greedy word wrap"""
    if not text:
        return 0
    lines, cur = 1, 0
    for w in str(text).split():
        L = len(w)
        if cur == 0:
            cur = L
        elif cur + 1 + L <= chars_per_line:
            cur += 1 + L
        else:
            lines += 1
            cur = L
        while cur > chars_per_line:
            lines += 1
            cur -= chars_per_line
    return lines


class Canvas:
    def __init__(self, spec, brand):
        self.spec = spec
        self.brand = brand
        self.C = brand["colors"]
        self.P = brand["diagram_palette"]
        self.W = int(spec.get("width", 1800))
        self.m = 70
        self.header_h = 130 if spec.get("title") else 0
        self.svg = []       # below nodes
        self.svg_top = []   # arrows (above groups, below nodes)
        self.divs = []      # nodes
        self.labels = []    # labels above everything
        self.markers = set()

    def pal(self, name):
        return self.P.get(name or "blue", self.P["blue"])

    def color_hex(self, name):
        if name in self.P:
            return self.P[name]["bar"]
        if name in self.C:
            return self.C[name]
        return name.lstrip("#") if name else self.C["blue"]

    # ---------------------------------------------------------------- primitives
    def marker(self, hexc):
        self.markers.add(hexc)
        return f"url(#arr-{hexc})"

    def path(self, pts, color="blue", dashed=False, both=False, width=4):
        hexc = self.color_hex(color)
        d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
        dash = ' stroke-dasharray="12 9"' if dashed else ""
        ms = f' marker-start="{self.marker(hexc)}"' if both else ""
        self.svg_top.append(f'<path d="{d}" fill="none" stroke="#{hexc}" stroke-width="{width}"{dash}{ms} '
                            f'marker-end="{self.marker(hexc)}" stroke-linejoin="round"/>')

    def label(self, x, y, text, color="blue", size=19):
        if not text:
            return
        hexc = self.color_hex(color)
        self.labels.append(f'<div class="elabel" style="left:{x:.0f}px;top:{y:.0f}px;color:#{hexc};font-size:{size}px">{esc(text)}</div>')

    def node_div(self, x, y, w, h, n):
        p = self.pal(n.get("color"))
        status = n.get("status", "")
        border = "dashed" if status in ("planned", "new", "external") else "solid"
        badge = ""
        if status in ("planned", "new"):
            badge = f'<span class="badge" style="background:#{p["bar"]}">{"TO BE BUILT" if status == "planned" else "NEW"}</span>'
        elif status == "external":
            badge = '<span class="badge" style="background:#5B738B">EXTERNAL</span>'
        tech = f'<div class="tech">{badge}{esc(n.get("tech", ""))}</div>' if (n.get("tech") or badge) else ""
        badge = ""
        lines = "".join(f'<div class="ln">{esc(l)}</div>' for l in n.get("lines", []))
        nt_style = ""
        self.divs.append(
            f'<div class="node" style="left:{x:.0f}px;top:{y:.0f}px;width:{w:.0f}px;height:{h:.0f}px;'
            f'background:#{p["fill"]};border-style:{border}">'
            f'<div class="bar" style="background:#{p["bar"]}"></div>{badge}{tech}'
            f'<div class="nt"{nt_style}>{esc(n.get("title", ""))}</div>{lines}</div>')

    def est_node_h(self, n, w):
        inner = w - 10 - 56
        h = 30
        if n.get("tech") or n.get("status") in ("planned", "new", "external"):
            h += 28
        tw = inner
        h += wrap_count(n.get("title", ""), max(6, int(tw / 14.5))) * 33 + 10
        for l in n.get("lines", []):
            h += wrap_count(l, max(8, int(inner / 10.6))) * 29
        return h + 26

    # ---------------------------------------------------------------- footer
    def footer_h(self):
        h = 0
        if self.spec.get("legend"):
            h += 60
        if self.spec.get("note"):
            h += 30 + 28 * wrap_count(self.spec["note"], int((self.W - 2 * self.m - 60) / 10.5)) + 20
        return h + (20 if h else 0)

    def draw_footer(self, y):
        leg = self.spec.get("legend")
        if leg:
            x = self.m
            items = []
            for it in leg:
                c = self.color_hex(it.get("color", "grey"))
                if it.get("style") in ("dashed", "solid") or it.get("kind") == "edge":
                    dash = "stroke-dasharray:10 7;" if it.get("style") == "dashed" else ""
                    sw = f'<svg width="54" height="14"><line x1="2" y1="7" x2="52" y2="7" style="stroke:#{c};stroke-width:4;{dash}"/></svg>'
                else:
                    sw = f'<span class="sw" style="background:#{c}"></span>'
                items.append(f'<span class="li">{sw}{esc(it["label"])}</span>')
            self.divs.append(f'<div class="legend" style="left:{x}px;top:{y}px;width:{self.W - 2 * self.m}px">{"".join(items)}</div>')
            y += 60
        if self.spec.get("note"):
            self.divs.append(f'<div class="note" style="left:{self.m}px;top:{y}px;width:{self.W - 2 * self.m}px">{esc(self.spec["note"])}</div>')

    # ---------------------------------------------------------------- html
    def html(self, H):
        s = self.spec
        font = self.brand.get("font_css", "Arial, sans-serif")
        header = ""
        if s.get("title"):
            header = (f'<div class="hdr"><div class="t">{esc(s["title"])}</div>'
                      f'<div class="st">{esc(s.get("subtitle", ""))}</div></div><div class="grad"></div>')
        markers = "".join(
            f'<marker id="arr-{c}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#{c}"/></marker>' for c in self.markers)
        C = self.C
        return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="render-size" content="{self.W},{H}">
<style>
*{{box-sizing:border-box}} body{{margin:0;width:{self.W}px;height:{H}px;background:#fff;font-family:{font};color:#{C['text']};position:relative;overflow:hidden}}
.hdr{{position:absolute;left:0;top:0;width:100%;height:{self.header_h - 6}px;background:#{C['navy']};padding:28px {self.m}px}}
.hdr .t{{color:#fff;font-size:44px;font-weight:600;letter-spacing:-0.5px}} .hdr .st{{color:#D1EFFF;font-size:24px;margin-top:6px}}
.grad{{position:absolute;left:0;top:{self.header_h - 6}px;width:100%;height:6px;background:linear-gradient(90deg,#{C['blue']},#1DB8F2 60%,#{C['teal']})}}
svg.layer{{position:absolute;left:0;top:0}}
.node{{position:absolute;border:2px solid #C9D2DB;border-radius:16px;padding:24px 26px 20px 38px;overflow:hidden;z-index:1}}
.node .bar{{position:absolute;left:0;top:0;bottom:0;width:10px}}
.node .nt{{font-size:26px;font-weight:600;line-height:33px;margin-bottom:10px;color:#{C['navy']}}}
.node .ln{{font-size:20px;line-height:29px;color:#{C['muted']}}}
.node .tech{{font-size:15px;letter-spacing:1.2px;text-transform:uppercase;font-weight:700;color:#{C['grey']};margin-bottom:6px}}
.node .nt{{display:flex;align-items:center;gap:12px}} .nicon{{display:inline-flex;flex:none}}
.node .nsub{{font-size:17px;color:#{C['muted']};margin:-4px 0 10px;font-style:italic}}
.pills{{display:grid;gap:10px;margin-top:6px}}
.pill{{border:2px solid;border-radius:12px;background:#fff;padding:8px 10px;text-align:center;font-size:18px;line-height:23px;color:#{C['navy']}}}
.sec{{border:2px solid #C9D2DB;border-radius:14px;background:#fff;padding:10px 12px 12px;margin-top:12px}}
.sec .st{{text-align:center;font-weight:700;font-size:19px;margin-bottom:8px;color:#{C['navy']}}}
.svc{{position:absolute;border:3px solid;border-radius:50%;display:flex;align-items:center;justify-content:center;z-index:2;background:#fff;box-shadow:0 2px 6px rgba(29,45,62,.12)}}
.sbadge{{position:absolute;bottom:-12px;left:50%;transform:translateX(-50%);color:#fff;font-size:11px;font-weight:700;letter-spacing:1px;padding:2px 7px;border-radius:9px;white-space:nowrap}}
.stitle{{position:absolute;text-align:center;z-index:2}} .stitle .t{{font-size:20px;line-height:26px;color:#{C['navy']};font-weight:600}}
.stitle .ssub{{font-size:16px;line-height:21px;color:#{C['muted']};margin-top:2px}}
.ftitle{{position:absolute;font-weight:700;z-index:1;display:flex;flex-direction:column;gap:2px;line-height:1.15}}
.ftitle .nicon{{margin-bottom:4px}} .fsub{{font-size:17px;font-weight:400;color:#{C['muted']}}}
.dlabel{{position:absolute;font-size:20px;font-weight:700;letter-spacing:1px}}
.epill{{position:absolute;transform:translate(-50%,-50%);background:#fff;border:2px solid;border-radius:14px;padding:1px 12px;font-size:16px;font-weight:700;white-space:nowrap;z-index:3}}
.badge{{display:inline-block;color:#fff;font-size:13px;font-weight:700;letter-spacing:1px;padding:2px 9px;border-radius:10px;margin-right:10px;vertical-align:1px}}
.elabel{{position:absolute;transform:translate(-50%,-50%);background:#fff;padding:1px 8px;border-radius:8px;white-space:nowrap;z-index:3;font-weight:500}}
.glabel{{position:absolute;transform:translateY(-50%);background:#fff;padding:2px 12px;font-size:18px;font-weight:700;letter-spacing:0.8px;text-transform:uppercase;z-index:0}}
.legend{{position:absolute;display:flex;gap:36px;flex-wrap:wrap;font-size:19px;color:#{C['muted']};align-items:center}}
.legend .li{{display:flex;align-items:center;gap:10px}} .legend .sw{{display:inline-block;width:22px;height:22px;border-radius:5px}}
.note{{position:absolute;background:#FFF8E1;border:2px solid #F2C94C;border-radius:14px;padding:16px 28px;font-size:19px;line-height:28px;color:#{C['navy']}}}
.step{{position:absolute;border:2px solid #C9D2DB;border-radius:16px;padding:70px 24px 18px 24px;z-index:1}}
.step .num{{position:absolute;left:22px;top:18px;width:42px;height:42px;border-radius:21px;color:#fff;font-weight:700;font-size:22px;display:flex;align-items:center;justify-content:center}}
.step .nt{{font-size:23px;font-weight:600;line-height:29px;color:#{C['navy']};margin-bottom:8px}}
.step .ln{{font-size:18px;line-height:26px;color:#{C['muted']}}}
.phead{{position:absolute;border:2px solid #C9D2DB;border-radius:14px;text-align:center;padding:14px 10px;z-index:1}}
.phead .nt{{font-size:22px;font-weight:600;color:#{C['navy']};line-height:27px}} .phead .sub{{font-size:16px;color:#{C['muted']};margin-top:4px}}
.phead .bar{{position:absolute;left:0;right:0;top:0;height:8px;border-radius:12px 12px 0 0}}
.mlabel{{position:absolute;transform:translate(-50%,-100%);font-size:18px;white-space:nowrap;padding:0 6px 3px;z-index:3;color:#{C['navy']}}}
.mnote{{position:absolute;font-size:16px;color:#{C['muted']};font-style:italic;transform:translateX(-50%);white-space:nowrap}}
.sect{{position:absolute;font-size:16px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#{C['grey']};background:#fff;padding:2px 10px;transform:translateY(-50%);z-index:2}}
</style></head><body>
{header}
<svg class="layer" width="{self.W}" height="{H}" style="z-index:0"><defs>{markers}</defs>{''.join(self.svg)}{''.join(self.svg_top)}</svg>
{''.join(self.divs)}
{''.join(self.labels)}
</body></html>"""


# ============================================================================ icons
# Generic line icons (24x24, stroke). Deliberately NOT the official SAP icon set.
ICONS = {
    "user": '<circle cx="12" cy="8" r="4"/><path d="M4 21c0-4.4 3.6-7 8-7s8 2.6 8 7"/>',
    "users": '<circle cx="9" cy="8" r="3.5"/><path d="M2.5 20c0-3.8 3-6 6.5-6s6.5 2.2 6.5 6"/><circle cx="17" cy="9" r="2.8"/><path d="M16 14.2c3 .2 5.5 2 5.5 5"/>',
    "device": '<rect x="2.5" y="4" width="15" height="11" rx="1.5"/><path d="M7 19h6M10 15v4"/><rect x="18" y="9" width="4" height="10" rx="1"/>',
    "browser": '<rect x="2.5" y="4" width="19" height="16" rx="2"/><path d="M2.5 8.5h19"/><circle cx="5.5" cy="6.3" r=".6"/><circle cx="7.7" cy="6.3" r=".6"/>',
    "router": '<circle cx="12" cy="12" r="9"/><path d="M7 12h5l4-4M12 12l4 4M14 8h2v2M14 16h2v-2"/>',
    "shield": '<path d="M12 3l7.5 3v5.5c0 4.6-3.2 8.3-7.5 9.5-4.3-1.2-7.5-4.9-7.5-9.5V6z"/><path d="M9 12l2 2 4-4"/>',
    "key": '<circle cx="8" cy="15" r="4"/><path d="M11 12l8-8M16 7l2 2M14 9l2 2"/>',
    "destination": '<path d="M12 3v18M12 5h7l2 2.5-2 2.5h-7M12 12H5l-2 2.5L5 17h7"/>',
    "plug": '<path d="M9 3v5M15 3v5M6 8h12v3a6 6 0 01-12 0zM12 17v4"/>',
    "database": '<ellipse cx="12" cy="5.5" rx="7.5" ry="2.8"/><path d="M4.5 5.5v13c0 1.5 3.4 2.8 7.5 2.8s7.5-1.3 7.5-2.8v-13M4.5 12c0 1.5 3.4 2.8 7.5 2.8s7.5-1.3 7.5-2.8"/>',
    "ai": '<path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z"/><path d="M18.5 15l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8z"/>',
    "agent": '<rect x="4.5" y="7" width="15" height="11" rx="3"/><circle cx="9.5" cy="12.5" r="1.3"/><circle cx="14.5" cy="12.5" r="1.3"/><path d="M12 7V4M10 4h4M2.5 12v2M21.5 12v2"/>',
    "cloud": '<path d="M7 18h10.5a4 4 0 00.6-7.95A6 6 0 006.4 9.1 4.5 4.5 0 007 18z"/>',
    "building": '<path d="M3 21h18M5 21V10l7-6 7 6v11M10 21v-6h4v6"/>',
    "gear": '<circle cx="12" cy="12" r="3"/><path d="M12 2.5v3M12 18.5v3M21.5 12h-3M5.5 12h-3M18.7 5.3l-2.1 2.1M7.4 16.6l-2.1 2.1M18.7 18.7l-2.1-2.1M7.4 7.4L5.3 5.3"/>',
    "code": '<path d="M8 7l-5 5 5 5M16 7l5 5-5 5M14 4l-4 16"/>',
    "chat": '<path d="M4 5h16v11H9l-5 4z"/><path d="M8 9.5h8M8 12.5h5"/>',
    "cpu": '<rect x="6" y="6" width="12" height="12" rx="1.5"/><rect x="9.5" y="9.5" width="5" height="5"/><path d="M9 3v3M15 3v3M9 18v3M15 18v3M3 9h3M3 15h3M18 9h3M18 15h3"/>',
    "optimizer": '<path d="M4 20V4M4 20h16"/><path d="M7 16l4-5 3 3 5-7"/><circle cx="19" cy="7" r="1.6"/>',
    "queue": '<rect x="4" y="4" width="16" height="4" rx="1"/><rect x="4" y="10" width="16" height="4" rx="1"/><rect x="4" y="16" width="16" height="4" rx="1"/>',
    "file": '<path d="M6 3h8l4 4v14H6z"/><path d="M14 3v4h4M9 12h6M9 15.5h6"/>',
    "chart": '<path d="M4 20h16M7 17v-6M11 17V7M15 17v-4M19 17V9"/>',
    "lock": '<rect x="5" y="10.5" width="14" height="10" rx="2"/><path d="M8 10.5V7.5a4 4 0 018 0v3"/>',
    "globe": '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c2.6 2.7 2.6 15.3 0 18M12 3c-2.6 2.7-2.6 15.3 0 18"/>',
    "monitor": '<path d="M3 12h4l2-6 4 12 2-6h6"/>',
    "pipeline": '<circle cx="5" cy="6" r="2.2"/><circle cx="5" cy="18" r="2.2"/><circle cx="19" cy="12" r="2.2"/><path d="M5 8.2v7.6M7.2 6c6 0 5 6 9.6 6M7.2 18c6 0 5-6 9.6-6"/>',
    "api": '<path d="M4 8h16M4 16h16"/><circle cx="8" cy="8" r="2"/><circle cx="16" cy="16" r="2"/>',
    "mail": '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3.5 6l8.5 7 8.5-7"/>',
    "search": '<circle cx="10.5" cy="10.5" r="6"/><path d="M15 15l5.5 5.5"/>',
}


def icon_svg(name, color, size=40):
    body = ICONS.get(name or "gear", ICONS["gear"])
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="#{color}" '
            f'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">{body}</svg>')


# ============================================================================ graph
KIND_ICON_SIZE = {"service": 84, "actor": 96}


def est_block_h(cv, n, w):
    """Estimated height of a box node incl. items (pills) and sections (nested blocks with pills)."""
    kind = n.get("kind", "box")
    if kind in ("service", "actor"):
        d = KIND_ICON_SIZE[kind]
        h = d + 14 + wrap_count(n.get("title", ""), max(6, int(w / 11.5))) * 27
        if n.get("sub"):
            h += wrap_count(n["sub"], max(6, int(w / 9))) * 22
        return h + 8
    inner = w - 10 - 56
    h = 28
    if n.get("tech") or n.get("status") in ("planned", "new", "external"):
        h += 28
    title_w = inner - (54 if n.get("icon") else 0)
    h += wrap_count(n.get("title", ""), max(6, int(title_w / 14.5))) * 33 + 10
    if n.get("sub"):
        h += wrap_count(n["sub"], max(6, int(inner / 9.5))) * 24
    for l in n.get("lines", []):
        h += wrap_count(l, max(8, int(inner / 10.6))) * 29
    h += pills_h(n.get("items", []), inner, n.get("item_cols", 1))
    for sec in n.get("sections", []):
        sw = inner - 28
        h += 14 + 36 + pills_h(sec.get("items", []), sw, sec.get("cols", 2)) + 14
    return h + 24


def pills_h(items, width, cols):
    if not items:
        return 0
    cols = max(1, cols)
    pw = (width - (cols - 1) * 12) / cols
    rows = [items[i:i + cols] for i in range(0, len(items), cols)]
    h = 0
    for r in rows:
        h += max(wrap_count(t, max(6, int((pw - 24) / 9.6))) for t in r) * 25 + 20 + 10
    return h + 4


def pills_html(items, cols, color):
    if not items:
        return ""
    cells = "".join(f'<div class="pill" style="border-color:#{color}">{esc(t)}</div>' for t in items)
    return f'<div class="pills" style="grid-template-columns:repeat({max(1, cols)},1fr)">{cells}</div>'


def render_graph(spec, brand):
    cv = Canvas(spec, brand)
    nodes = spec["nodes"]
    groups = spec.get("groups", [])
    gmap = {g["id"]: g for g in groups if "id" in g}
    ncols = max(n["col"] + n.get("colspan", 1) for n in nodes)
    nrows = max(n["row"] + n.get("rowspan", 1) for n in nodes)

    # --- group membership (nested groups contain nodes of their child groups)
    def members(g, seen=()):
        ids = set(g.get("nodes", []))
        for cid in g.get("groups", []):
            if cid in gmap and cid not in seen:
                ids |= members(gmap[cid], seen + (cid,))
        return ids
    for g in groups:
        g["_members"] = members(g)
        g.setdefault("style", "boundary")
        g.setdefault("pad", 30 if g["style"] == "boundary" else 34)
        g["_title_h"] = (0 if g["style"] == "boundary" else (78 if g.get("sub") else 58)) + g.get("title_extra", 0)

    def chain(nid):
        return [g for g in groups if nid in g["_members"]]

    # --- layout
    gap_default = spec.get("col_gap", 130)
    col_gaps = spec.get("col_gaps") or [gap_default] * (ncols - 1)
    row_gap_default = spec.get("row_gap", 110)
    row_gaps = spec.get("row_gaps") or [row_gap_default] * (nrows - 1)
    weights = spec.get("col_weights") or [1] * ncols
    left_need = max([sum(g["pad"] for g in chain(n["id"])) for n in nodes if n["col"] == 0] + [0])
    right_need = max([sum(g["pad"] for g in chain(n["id"])) for n in nodes if n["col"] + n.get("colspan", 1) == ncols] + [0])
    ml = cv.m - 20 + left_need
    mr = cv.m - 20 + right_need
    usable = cv.W - ml - mr - sum(col_gaps)
    col_w = [usable * w / sum(weights) for w in weights]
    col_x = []
    x = ml
    for i in range(ncols):
        col_x.append(x)
        x += col_w[i] + (col_gaps[i] if i < ncols - 1 else 0)

    def span_w(c, cs):
        return col_x[c + cs - 1] + col_w[c + cs - 1] - col_x[c]

    top_need = max([sum(g["pad"] + g["_title_h"] for g in chain(n["id"])) for n in nodes if n["row"] == 0] + [0])
    top = cv.header_h + 40 + top_need
    row_h = [0.0] * nrows
    for n in nodes:
        if n.get("rowspan", 1) == 1:
            row_h[n["row"]] = max(row_h[n["row"]], est_block_h(cv, n, span_w(n["col"], n.get("colspan", 1))))
    row_h = [max(h, spec.get("min_row_height", 130)) for h in row_h]
    for n in nodes:  # multi-row nodes may need more room than the rows give
        rs = n.get("rowspan", 1)
        if rs > 1:
            need = est_block_h(cv, n, span_w(n["col"], n.get("colspan", 1)))
            have = sum(row_h[n["row"]:n["row"] + rs]) + sum(row_gaps[n["row"]:n["row"] + rs - 1])
            if need > have:
                row_h[n["row"] + rs - 1] += need - have
    row_y = []
    y = top
    for i, h in enumerate(row_h):
        row_y.append(y)
        y += h + (row_gaps[i] if i < nrows - 1 else 0)
    content_bottom = y

    # --- nodes
    box = {}
    label_bottom = {}
    for n in nodes:
        cs, rs = n.get("colspan", 1), n.get("rowspan", 1)
        x0, w = col_x[n["col"]], span_w(n["col"], cs)
        y0 = row_y[n["row"]]
        h = sum(row_h[n["row"]:n["row"] + rs]) + sum(row_gaps[n["row"]:n["row"] + rs - 1])
        kind = n.get("kind", "box")
        if kind in ("service", "actor"):
            d = KIND_ICON_SIZE[kind]
            eh = est_block_h(cv, n, w)
            oy = y0 + max(0, (h - eh) / 2) if n.get("valign", "middle") == "middle" else y0
            cx = x0 + w / 2
            box[n["id"]] = (cx - d / 2, oy, cx + d / 2, oy + d)
            label_bottom[n["id"]] = oy + eh + 4   # edges leaving/entering the bottom start below the label
            draw_service(cv, n, cx, oy, d, w)
        else:
            if n.get("fit", spec.get("fit_default", "cell")) == "content":
                h = min(h, est_block_h(cv, n, w))
            box[n["id"]] = (x0, y0, x0 + w, y0 + h)
            draw_box(cv, n, x0, y0, w, h)

    # --- groups: inner first for geometry, outer first for painting
    def depth(g, seen=()):
        parents = [p for p in groups if g.get("id") and g["id"] in p.get("groups", [])]
        return 0 if not parents else 1 + max(depth(p) for p in parents)
    rects = {}
    order = sorted(groups, key=lambda g: -depth(g))  # deepest first
    for g in order:
        # service/actor nodes: box is the icon only, but their label hangs below it and must stay inside the frame
        bs = [(*box[i][:3], max(box[i][3], label_bottom.get(i, box[i][3]))) for i in g["_members"] if i in box]
        bs += [rects[c] for c in g.get("groups", []) if c in rects]
        if not bs:
            continue
        # direct members' own chain padding is handled by nesting: child rects already include their pads
        x1 = min(b[0] for b in bs) - g["pad"]
        y1 = min(b[1] for b in bs) - g["pad"] - g["_title_h"]
        x2 = max(b[2] for b in bs) + g["pad"]
        y2 = max(b[3] for b in bs) + g["pad"]
        rects[g.get("id", id(g))] = (x1, y1, x2, y2)
        g["_rect"] = (x1, y1, x2, y2)
    for g in sorted(groups, key=lambda g: depth(g)):
        if "_rect" in g:
            draw_group(cv, g, g["_rect"])

    # --- dividers (e.g. NETWORK line between the platform and external systems)
    all_rects = list(box.items()) + [(g.get("id"), g["_rect"]) for g in groups if "_rect" in g]
    for dv in spec.get("dividers", []):
        k = dv["after_col"]
        left_ids = {n["id"] for n in nodes if n["col"] + n.get("colspan", 1) - 1 <= k}
        L = max([b[2] for i, b in box.items() if i in left_ids] +
                [g["_rect"][2] for g in groups if "_rect" in g and g["_members"] and g["_members"] <= left_ids])
        R = min([b[0] for i, b in box.items() if i not in left_ids] +
                [g["_rect"][0] for g in groups if "_rect" in g and g["_members"] and not (g["_members"] & left_ids)])
        xd = (L + R) / 2
        c = cv.color_hex(dv.get("color", "grey"))
        cv.svg.append(f'<line x1="{xd}" y1="{cv.header_h + 20}" x2="{xd}" y2="{content_bottom + 60}" stroke="#{c}" stroke-width="5"/>')
        if dv.get("label"):
            cv.divs.append(f'<div class="dlabel" style="left:{xd + 18}px;top:{cv.header_h + 22}px;color:#{c}">{esc(dv["label"])}</div>')

    # --- edges
    route_edges(cv, spec.get("edges", []), box, label_bottom)

    bottom = max([r[3] for r in rects.values()] + [content_bottom]) + 40
    H = bottom
    fy = H
    H += cv.footer_h()
    cv.draw_footer(fy)
    return cv.html(int(H))


def draw_box(cv, n, x, y, w, h):
    p = cv.pal(n.get("color"))
    status = n.get("status", "")
    border = "dashed" if status in ("planned", "new", "external") else "solid"
    badge = ""
    if status in ("planned", "new"):
        badge = f'<span class="badge" style="background:#{p["bar"]}">{"TO BE BUILT" if status == "planned" else "NEW"}</span>'
    elif status == "external":
        badge = '<span class="badge" style="background:#5B738B">EXTERNAL</span>'
    tech = f'<div class="tech">{badge}{esc(n.get("tech", ""))}</div>' if (n.get("tech") or badge) else ""
    ic = f'<span class="nicon">{icon_svg(n["icon"], p["bar"], 30)}</span>' if n.get("icon") else ""
    sub = f'<div class="nsub">{esc(n["sub"])}</div>' if n.get("sub") else ""
    lines = "".join(f'<div class="ln">{esc(l)}</div>' for l in n.get("lines", []))
    items = pills_html(n.get("items", []), n.get("item_cols", 1), p["bar"])
    secs = ""
    for sec in n.get("sections", []):
        secs += (f'<div class="sec"><div class="st">{esc(sec.get("title", ""))}</div>'
                 f'{pills_html(sec.get("items", []), sec.get("cols", 2), p["bar"])}</div>')
    bar = "" if n.get("bar") is False else f'<div class="bar" style="background:#{p["bar"]}"></div>'
    fill = "FFFFFF" if n.get("fill") == "white" else p["fill"]
    cv.divs.append(
        f'<div class="node" style="left:{x:.0f}px;top:{y:.0f}px;width:{w:.0f}px;height:{h:.0f}px;'
        f'background:#{fill};border-style:{border};border-color:#{n.get("border", "C9D2DB") if not n.get("accent_border") else p["bar"]}">'
        f'{bar}{tech}<div class="nt">{ic}<span>{esc(n.get("title", ""))}</span></div>{sub}{lines}{items}{secs}</div>')


def draw_service(cv, n, cx, y, d, w):
    p = cv.pal(n.get("color"))
    dashed = n.get("status") in ("planned", "new")
    style = "dashed" if dashed else "solid"
    badge = ""
    if n.get("status") == "planned":
        badge = f'<div class="sbadge" style="background:#{p["bar"]}">TO BE BUILT</div>'
    elif n.get("status") == "new":
        badge = f'<div class="sbadge" style="background:#{p["bar"]}">NEW</div>'
    size = int(d * 0.52)
    cv.divs.append(
        f'<div class="svc" style="left:{cx - d / 2:.0f}px;top:{y:.0f}px;width:{d}px;height:{d}px;border-style:{style};'
        f'background:#{p["fill"]};border-color:#{p["bar"]}">{icon_svg(n.get("icon", "user" if n.get("kind") == "actor" else "gear"), p["bar"], size)}{badge}</div>')
    sub = f'<div class="ssub">{esc(n["sub"])}</div>' if n.get("sub") else ""
    cv.divs.append(f'<div class="stitle" style="left:{cx - w / 2:.0f}px;top:{y + d + 10:.0f}px;width:{w:.0f}px">'
                   f'<div class="t">{esc(n.get("title", ""))}</div>{sub}</div>')


def draw_group(cv, g, r):
    x1, y1, x2, y2 = r
    c = cv.color_hex(g.get("color", "grey"))
    if g["style"] == "boundary":
        cv.svg.append(f'<rect x="{x1}" y="{y1}" width="{x2 - x1}" height="{y2 - y1}" rx="22" fill="#{c}" fill-opacity="0.03" '
                      f'stroke="#{c}" stroke-width="2.5" stroke-dasharray="14 9"/>')
        if g.get("label"):
            cv.divs.append(f'<div class="glabel" style="left:{x1 + 26}px;top:{y1}px;color:#{c}">{esc(g["label"])}</div>')
        return
    fill = g.get("fill") or ("F2F9FF" if g.get("tone", "light") == "light" else "FFFFFF")
    sw = g.get("stroke_width", 3)
    cv.svg.append(f'<rect x="{x1}" y="{y1}" width="{x2 - x1}" height="{y2 - y1}" rx="24" fill="#{fill}" stroke="#{c}" stroke-width="{sw}"/>')
    size = g.get("title_size", 26)
    sub = f'<div class="fsub">{esc(g["sub"])}</div>' if g.get("sub") else ""
    ic = f'<span class="nicon">{icon_svg(g["icon"], c, int(size * 1.2))}</span>' if g.get("icon") else ""
    cv.divs.append(f'<div class="ftitle" style="left:{x1 + 28}px;top:{y1 + 18}px;color:#{g.get("title_color", cv.C["navy"])};font-size:{size}px">'
                   f'{ic}<span>{esc(g.get("label", ""))}</span>{sub}</div>')


def route_edges(cv, edges, box, label_bottom=None):
    label_bottom = label_bottom or {}

    def anchor_box(nid, side):
        b = box[nid]
        if side == "bottom" and nid in label_bottom:
            return (b[0], b[1], b[2], label_bottom[nid])
        return b

    plans = []
    for e in edges:
        a, b = box[e["from"]], box[e["to"]]
        fs, ts = e.get("from_side"), e.get("to_side")
        if not fs or not ts:
            afs, ats = auto_sides(a, b)
            fs, ts = fs or afs, ts or ats
        plans.append([e, a, b, fs, ts])
    ports = {}
    for i, (e, a, b, fs, ts) in enumerate(plans):
        ports.setdefault((e["from"], fs), []).append((i, "s", b))
        ports.setdefault((e["to"], ts), []).append((i, "t", a))
    pos = {}
    for (nid, side), lst in ports.items():
        bx = box[nid]
        horiz = side in ("top", "bottom")
        key = (lambda it: (it[2][0] + it[2][2]) / 2) if horiz else (lambda it: (it[2][1] + it[2][3]) / 2)
        lst.sort(key=key)
        k = len(lst)
        length = (bx[2] - bx[0]) if horiz else (bx[3] - bx[1])
        spacing = min(48, length / (k + 1))
        for j, (i, end_, _) in enumerate(lst):
            pos[(i, end_)] = (j - (k - 1) / 2) * spacing
    corridors = {}
    for i, (e, a, b, fs, ts) in enumerate(plans):
        s = side_point(anchor_box(e["from"], fs), fs, pos[(i, "s")])
        t = side_point(anchor_box(e["to"], ts), ts, pos[(i, "t")])
        if "via_y" in e or "via_x" in e:
            pts = via_path(s, t, fs, ts, e)
        else:
            pts = orth_path(s, t, fs, ts, corridors)
        color = e.get("color", "blue")
        dashed = e.get("style") == "dashed" or e.get("status") == "planned"
        cv.path(pts, color, dashed, e.get("both", False), width=e.get("width", 4))
        if e.get("label"):
            lab = label_point(pts, e.get("label_at"))
            if e.get("label_style", cv.spec.get("edge_label_style", "plain")) == "pill":
                cv.labels.append(f'<div class="epill" style="left:{lab[0]:.0f}px;top:{lab[1]:.0f}px;color:#{cv.color_hex(color)};'
                                 f'border-color:#{cv.color_hex(color)}">{esc(e["label"])}</div>')
            else:
                cv.label(lab[0], lab[1], e["label"], color)


def via_path(s, t, fs, ts, e):
    """Force the corridor through an explicit absolute y (via_y) or x (via_x) coordinate."""
    if "via_y" in e:
        y = e["via_y"]
        return [s, (s[0], y), (t[0], y), t] if fs in ("top", "bottom") else [s, (s[0] + (30 if fs == "right" else -30), s[1]), (s[0] + (30 if fs == "right" else -30), y), (t[0], y), t]
    x = e["via_x"]
    return [s, (x, s[1]), (x, t[1]), t]


def side_point(b, side, off):
    x1, y1, x2, y2 = b
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return {"left": (x1, cy + off), "right": (x2, cy + off), "top": (cx + off, y1), "bottom": (cx + off, y2)}[side]


def auto_sides(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    oy = min(ay2, by2) - max(ay1, by1)
    if oy > 30 and (bx1 >= ax2 or ax1 >= bx2):
        return ("right", "left") if bx1 >= ax2 else ("left", "right")
    ox = min(ax2, bx2) - max(ax1, bx1)
    if by1 >= ay2:
        return "bottom", "top"
    if ay1 >= by2:
        return "top", "bottom"
    return ("right", "left") if (bx1 + bx2) > (ax1 + ax2) else ("left", "right")


def orth_path(s, t, fs, ts, corridors):
    hs, ht = fs in ("left", "right"), ts in ("left", "right")
    if hs and ht and abs(s[1] - t[1]) < 60 and fs != ts:
        t = (t[0], s[1])
    if not hs and not ht and abs(s[0] - t[0]) < 60 and fs != ts:
        t = (s[0], t[1])
    if hs and ht:
        if abs(s[1] - t[1]) < 2:
            return [s, t]
        mx = (s[0] + t[0]) / 2
        n = corridors.get(("x", round(mx / 10)), 0); corridors[("x", round(mx / 10))] = n + 1
        mx += (n % 2 * 2 - 1) * ((n + 1) // 2) * 16
        return [s, (mx, s[1]), (mx, t[1]), t]
    if not hs and not ht:
        if abs(s[0] - t[0]) < 2:
            return [s, t]
        my = (s[1] + t[1]) / 2
        n = corridors.get(("y", round(my / 10)), 0); corridors[("y", round(my / 10))] = n + 1
        my += (n % 2 * 2 - 1) * ((n + 1) // 2) * 16
        return [s, (s[0], my), (t[0], my), t]
    if hs:
        return [s, (t[0], s[1]), t]
    return [s, (s[0], t[1]), t]


def label_point(pts, at=None):
    segs = list(zip(pts, pts[1:]))
    if at is not None and isinstance(at, (int, float)):
        # fraction along the total path length
        total = sum(abs(p[0] - q[0]) + abs(p[1] - q[1]) for p, q in segs)
        goal, run = total * at, 0
        for p, q in segs:
            L = abs(p[0] - q[0]) + abs(p[1] - q[1])
            if run + L >= goal and L > 0:
                f = (goal - run) / L
                return (p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f)
            run += L
    if len(pts) == 4:
        p, q = pts[1], pts[2]
        if abs(p[0] - q[0]) + abs(p[1] - q[1]) > 120:
            return ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
    best, lab = -1, pts[0]
    for p, q in segs:
        L = abs(p[0] - q[0]) + abs(p[1] - q[1])
        if L > best:
            best, lab = L, ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
    return lab


# ============================================================================ platform preset
def platform_to_graph(spec):
    """SAP Business AI Platform reference-architecture layout:
       clients (left) | SAP Business AI Platform > Subaccount (inner grid) | NETWORK | external systems (right)."""
    inner = spec.get("nodes", [])
    icols = max([n["col"] + n.get("colspan", 1) for n in inner] + [1])
    clients = spec.get("clients", [])
    external = spec.get("external", [])
    nodes = []
    has_clients = bool(clients)
    off = 1 if has_clients else 0
    for i, c in enumerate(clients):
        nodes.append({"kind": "actor", "icon": c.get("icon", "user"), "color": c.get("color", "grey"),
                      "row": c.get("row", i), "col": 0, **c})
    for n in inner:
        m = dict(n)
        m["col"] = n["col"] + off
        nodes.append(m)
    ext_col = icols + off
    for i, e in enumerate(external):
        m = {"kind": "box", "icon": e.get("icon", "cloud"), "color": e.get("color", "navy"), "row": e.get("row", i),
             "col": ext_col, "fill": "white", "accent_border": True, "bar": False, **e}
        m["col"] = ext_col
        nodes.append(m)
    ncols = ext_col + (1 if external else 0)
    inner_ids = [n["id"] for n in inner]
    groups = [dict(g) for g in spec.get("groups", [])]
    for g in groups:
        g.setdefault("style", "frame")
        g.setdefault("tone", "white")
    child_ids = {c for g in groups for c in g.get("groups", [])}
    top_inner_groups = [g["id"] for g in groups if "id" in g and g["id"] not in child_ids]
    sub = spec.get("subaccount", {})
    groups.append({"id": "_subaccount", "style": "frame", "label": sub.get("title", "Subaccount"), "sub": sub.get("sub"),
                   "nodes": inner_ids, "groups": top_inner_groups, "color": "blue", "tone": "white", "pad": 34, "title_size": 24})
    groups.append({"id": "_platform", "style": "frame", "label": spec.get("platform_label", "SAP Business AI Platform"), "groups": ["_subaccount"],
                   "color": "blue", "tone": "light", "pad": 26, "title_size": 34, "title_color": cv_blue(), "icon": None})
    if external:
        groups.append({"id": "_external", "style": "boundary", "label": spec.get("external_label", ""),
                       "nodes": [e["id"] for e in external], "color": "grey", "pad": 22})
    weights = ([spec.get("client_weight", 0.7)] if has_clients else []) + (spec.get("col_weights") or [1] * icols) + \
              ([spec.get("external_weight", 1.1)] if external else [])
    inner_gaps = spec.get("col_gaps") or [spec.get("col_gap", 110)] * (icols - 1)
    gaps = ([spec.get("client_gap", 160)] if has_clients else []) + inner_gaps + ([spec.get("network_gap", 210)] if external else [])
    g = {k: v for k, v in spec.items() if k not in ("nodes", "clients", "external", "groups", "col_gaps", "col_weights")}
    g.update({"type": "graph", "nodes": nodes, "groups": groups, "col_weights": weights, "col_gaps": gaps,
              "edge_label_style": spec.get("edge_label_style", "pill")})
    if external:
        g["dividers"] = [{"after_col": ext_col - 1, "label": spec.get("network_label", "NETWORK")}]
    g.setdefault("width", 2600 if icols >= 4 else 2300)
    g.setdefault("fit_default", "content")
    g.setdefault("row_gap", 90)
    return g


def cv_blue():
    return BRAND["colors"]["blue"]


def render_platform(spec, brand):
    return render_graph(platform_to_graph(spec), brand)


# ============================================================================ steps
def render_steps(spec, brand):
    cv = Canvas(spec, brand)
    steps = spec["steps"]
    n = spec.get("per_row", min(4, len(steps)))
    gap = 90
    top = cv.header_h + 60
    cw = (cv.W - 2 * cv.m - (n - 1) * gap) / n
    rows = [steps[i:i + n] for i in range(0, len(steps), n)]

    def est(s):
        inner = cw - 48
        h = 70 + wrap_count(s.get("title", ""), int(inner / 12.8)) * 29 + 8
        body = s.get("lines") or ([s["text"]] if s.get("text") else [])
        for l in body:
            h += wrap_count(l, int(inner / 9.4)) * 26
        return h + 22

    row_h = [max(est(s) for s in r) for r in rows]
    gap_y = 110
    y = top
    boxes = []
    for ri, r in enumerate(rows):
        for ci, s in enumerate(r):
            x = cv.m + ci * (cw + gap)
            p = cv.pal(s.get("color", "blue"))
            body = s.get("lines") or ([s["text"]] if s.get("text") else [])
            lines = "".join(f'<div class="ln">{esc(l)}</div>' for l in body)
            idx = len(boxes) + 1
            cv.divs.append(f'<div class="step" style="left:{x:.0f}px;top:{y:.0f}px;width:{cw:.0f}px;height:{row_h[ri]:.0f}px;background:#{p["fill"]}">'
                           f'<div class="num" style="background:#{p["bar"]}">{idx}</div><div class="nt">{esc(s.get("title", ""))}</div>{lines}</div>')
            boxes.append((x, y, x + cw, y + row_h[ri], ri))
        y += row_h[ri] + gap_y
    for a, b in zip(boxes, boxes[1:]):
        color = spec.get("arrow_color", "blue")
        if a[4] == b[4]:
            yy = (a[1] + a[3]) / 2
            cv.path([(a[2], yy), (b[0], yy)], color)
        else:
            my = a[3] + gap_y / 2
            ax = (a[0] + a[2]) / 2
            bx = (b[0] + b[2]) / 2
            cv.path([(ax, a[3]), (ax, my), (bx, my), (bx, b[1])], color)
    H = y - gap_y + 60
    fy = H
    H += cv.footer_h()
    cv.draw_footer(fy)
    return cv.html(int(H))


# ============================================================================ sequence
def check_sequence_loops(spec):
    """Every synchronous request must be answered: A->B (sync) needs a later message B->A.
    Mark genuinely fire-and-forget calls with "kind": "async" (or "oneway": true)."""
    msgs = spec["messages"]
    names = {p["id"]: p.get("title", p["id"]) for p in spec["participants"]}
    problems = []
    for i, m in enumerate(msgs):
        a, b = m["from"], m["to"]
        if a == b or m.get("kind", "sync") in ("async", "return") or m.get("oneway"):
            continue
        if not any(x["from"] == b and x["to"] == a for x in msgs[i + 1:]):
            problems.append(f"message {i + 1} '{m.get('label', '')}' ({names[a]} -> {names[b]}) never gets a response back "
                            f"from {names[b]} to {names[a]}")
    first = msgs[0]["from"] if msgs else None
    if first and msgs and not any(x["to"] == first for x in msgs[1:]):
        problems.append(f"the initiator {names[first]} never receives the final result")
    for pr in problems:
        print("WARNING (sequence):", pr, file=sys.stderr)
    return problems


def render_sequence(spec, brand):
    check_sequence_loops(spec)
    cv = Canvas(spec, brand)
    parts = spec["participants"]
    msgs = spec["messages"]
    n = len(parts)
    top = cv.header_h + 50
    span = (cv.W - 2 * cv.m) / n
    pw = min(300, span - 36)
    head_h = 96 if any(p.get("sub") for p in parts) else 74
    xs = {}
    for i, p in enumerate(parts):
        cx = cv.m + span * (i + 0.5)
        xs[p["id"]] = cx
        pal = cv.pal(p.get("color", "blue"))
        sub = f'<div class="sub">{esc(p["sub"])}</div>' if p.get("sub") else ""
        cv.divs.append(f'<div class="phead" style="left:{cx - pw / 2:.0f}px;top:{top}px;width:{pw:.0f}px;height:{head_h}px;background:#{pal["fill"]}">'
                       f'<div class="bar" style="background:#{pal["bar"]}"></div><div class="nt">{esc(p["title"])}</div>{sub}</div>')
    row = 70
    y = top + head_h + 70
    sections = {s["at"]: s["label"] for s in spec.get("sections", [])}
    auto = spec.get("autonumber", True)
    num = 0
    for i, msg in enumerate(msgs):
        if i in sections:
            y += 16
            cv.svg.append(f'<line x1="{cv.m - 20}" y1="{y - 34}" x2="{cv.W - cv.m + 20}" y2="{y - 34}" stroke="#{cv.C["grey"]}" stroke-width="2" stroke-dasharray="6 6"/>')
            cv.labels.append(f'<div class="sect" style="left:{cv.m - 10}px;top:{y - 34}px">{esc(sections[i])}</div>')
            y += 30
        kind = msg.get("kind", "sync")
        color = "grey" if kind == "return" else msg.get("color", "blue")
        x1, x2 = xs[msg["from"]], xs[msg["to"]]
        num += 1
        text = (f"{num}. " if auto else "") + msg.get("label", "")
        if msg["from"] == msg["to"]:
            cv.path([(x1, y - 14), (x1 + 70, y - 14), (x1 + 70, y + 16), (x1 + 4, y + 16)], color, kind == "return", width=3)
            if x1 + 90 + len(text) * 10 > cv.W:
                cv.labels.append(f'<div class="mlabel" style="left:{x1 - 12}px;top:{y + 8}px;transform:translate(-100%,-100%)">{esc(text)}</div>')
            else:
                cv.labels.append(f'<div class="mlabel" style="left:{x1 + 80}px;top:{y + 8}px;transform:translate(0,-100%)">{esc(text)}</div>')
            y += 20
        else:
            cv.path([(x1, y), (x2 - (6 if x2 > x1 else -6), y)], color, kind in ("return", "async"), width=3)
            cv.labels.append(f'<div class="mlabel" style="left:{(x1 + x2) / 2:.0f}px;top:{y - 2}px">{esc(text)}</div>')
        if msg.get("note"):
            cv.labels.append(f'<div class="mnote" style="left:{(x1 + x2) / 2 + (40 if x1 == x2 else 0):.0f}px;top:{y + 8}px">{esc(msg["note"])}</div>')
            y += 22
        y += row
    bottom = y - row + 50
    for p in parts:
        cx = xs[p["id"]]
        cv.svg.append(f'<line x1="{cx}" y1="{top + head_h}" x2="{cx}" y2="{bottom}" stroke="#AAB6C2" stroke-width="2.5" stroke-dasharray="8 8"/>')
    H = bottom + 50
    fy = H
    H += cv.footer_h()
    cv.draw_footer(fy)
    return cv.html(int(H))


RENDERERS = {"graph": render_graph, "platform": render_platform, "steps": render_steps, "sequence": render_sequence}


def main():
    global BRAND
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec")
    ap.add_argument("out", help="output .png (the .html is written next to it)")
    ap.add_argument("--html-only", action="store_true")
    ap.add_argument("--brand")
    a = ap.parse_args()
    if a.brand:
        BRAND = json.loads(Path(a.brand).read_text())
    spec = json.loads(Path(a.spec).read_text())
    kind = spec.get("type", "graph")
    doc = RENDERERS[kind](spec, BRAND)
    out = Path(a.out)
    html_path = out.with_suffix(".html")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(doc, encoding="utf-8")
    if a.html_only:
        print(html_path)
        return
    screenshot(html_path, out)
    print(out)


if __name__ == "__main__":
    main()
