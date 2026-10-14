#!/usr/bin/env python3
"""
Create a WIP screenshot placeholder PNG that the author later replaces with a real screenshot.

Usage:
    python make_placeholder.py out.png --title "Datasets page — upload and publish" \
        --capture "Datasets list with one published dataset" --capture "Upload dialog open" [--id SCR-03]

Only needs Pillow (no browser). 2000x1250 (16:10) so a real screenshot can replace it 1:1.
"""
import argparse
import os
from PIL import Image, ImageDraw, ImageFont

FONTS = ["/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf", "C:/Windows/Fonts/arial.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
BOLD = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf", "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]


def font(size, bold=False):
    for f in (BOLD if bold else FONTS):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def wrap(draw, text, fnt, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=fnt) <= width:
            cur = t
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--title", required=True, help="screen / state shown")
    ap.add_argument("--capture", action="append", default=[], help="what the real screenshot must show (repeatable)")
    ap.add_argument("--id", default="", help="optional placeholder id, e.g. SCR-03")
    ap.add_argument("--width", type=int, default=2000)
    ap.add_argument("--height", type=int, default=1250)
    a = ap.parse_args()
    W, H = a.width, a.height
    im = Image.new("RGB", (W, H), (245, 246, 247))
    d = ImageDraw.Draw(im)
    for x in range(-H, W, 46):  # hatch
        d.line([(x, H), (x + H, 0)], fill=(233, 236, 239), width=10)
    m = 40
    for i in range(m, W - m, 36):  # dashed border
        d.line([(i, m), (min(i + 20, W - m), m)], fill=(91, 115, 139), width=5)
        d.line([(i, H - m), (min(i + 20, W - m), H - m)], fill=(91, 115, 139), width=5)
    for j in range(m, H - m, 36):
        d.line([(m, j), (m, min(j + 20, H - m))], fill=(91, 115, 139), width=5)
        d.line([(W - m, j), (W - m, min(j + 20, H - m))], fill=(91, 115, 139), width=5)
    cw = W - 2 * 170
    card_h = 330 + 58 * max(1, len(a.capture)) + (60 if len(a.title) > 60 else 0)
    x0, y0 = 170, (H - card_h) // 2
    d.rounded_rectangle([x0, y0, x0 + cw, y0 + card_h], radius=28, fill=(255, 255, 255), outline=(201, 210, 219), width=3)
    d.rounded_rectangle([x0, y0, x0 + 14, y0 + card_h], radius=7, fill=(231, 101, 0))
    y = y0 + 44
    tag = "SCREENSHOT PLACEHOLDER" + (f"  ·  {a.id}" if a.id else "") + "  ·  WIP"
    f_tag = font(30, True)
    tw = d.textlength(tag, font=f_tag)
    d.rounded_rectangle([x0 + 60, y, x0 + 60 + tw + 40, y + 52], radius=26, fill=(231, 101, 0))
    d.text((x0 + 80, y + 9), tag, font=f_tag, fill="white")
    y += 90
    f_title = font(52, True)
    for line in wrap(d, a.title, f_title, cw - 120):
        d.text((x0 + 60, y), line, font=f_title, fill=(29, 45, 62)); y += 64
    y += 14
    f_b = font(34)
    if a.capture:
        d.text((x0 + 60, y), "Capture:", font=font(34, True), fill=(85, 107, 130)); y += 52
        for c in a.capture:
            for k, line in enumerate(wrap(d, c, f_b, cw - 180)):
                d.text((x0 + 90, y), ("•  " if k == 0 else "    ") + line, font=f_b, fill=(29, 45, 62)); y += 48
    f_s = font(28)
    d.text((x0 + 60, y0 + card_h - 58), "Replace with a real screenshot (synthetic or approved data only) — same 16:10 size.",
           font=f_s, fill=(85, 107, 130))
    im.save(a.out)
    print(a.out)


if __name__ == "__main__":
    main()
