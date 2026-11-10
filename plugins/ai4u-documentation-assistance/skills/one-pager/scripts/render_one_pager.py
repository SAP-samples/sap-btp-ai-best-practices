"""Render an AI4U one-pager image from a content JSON file.

The static artwork (background, SAP and AI4U logos, laptop frame, arrows,
persona circles, Challenge/Solution/Benefits panels) is a pre-rendered base
plate in ``assets/base_plate.png``. This script pastes the app screenshot into
the laptop screen, the two persona photos into the circles, and draws every
text slot on top. Output is a single PNG, 3840x2160 by default.

Examples:
    # Render with the default 4K size
    python scripts/render_one_pager.py --content one-pager/content.json \
        --output one-pager/one-pager.png

    # 1080p output
    python scripts/render_one_pager.py --content one-pager/content.json \
        --output one-pager/one-pager.png --width 1920

    # Take the screenshot from the clipboard (macOS / Windows)
    python scripts/render_one_pager.py --content one-pager/content.json \
        --output one-pager/one-pager.png --screenshot clipboard

    # Use fonts from a custom folder
    python scripts/render_one_pager.py --content c.json --output o.png --font-dir ~/fonts/72

Content JSON keys: title, subtitle (optional), app_name, screenshot (path or
"clipboard", optional), left_persona/right_persona {name, role, photo,
caption}, challenge, solution, benefits (list of 3 strings; the first word of
each is drawn in the accent color). Relative paths resolve against the JSON
file's folder. ``photo`` is a file name from assets/personas/manifest.json or
a path to a user-supplied image.

Exit codes: 0 on success, 2 when a text slot overflows (message says which
slot and roughly how many words to cut), 1 on other input errors.
"""

import argparse
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageGrab

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout  # noqa: E402
from text_fit import SlotOverflow, draw_slot, fallback_font  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"
DEFAULT_SUBTITLE = "Powered by SAP Business AI Platform (BAIP)"

# Folders searched for the SAP 72 font family, per operating system.
FONT_DIRS = [
    Path("/Library/Fonts"),
    Path.home() / "Library/Fonts",
    Path("/System/Library/Fonts"),
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path.home() / ".fonts",
    Path.home() / ".local/share/fonts",
    Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/Windows/Fonts",
]


def find_font_file(prefixes, extra_dir=None):
    """Find the first font file whose name starts with one of ``prefixes``.

    Installed 72 files often carry a suffix (``72-Regular_43188850.ttf``), so
    matching is by prefix and the name must continue with "." or "_".

    Args:
        prefixes: candidate file-name prefixes, in preference order.
        extra_dir: optional folder searched first (``--font-dir``).

    Returns:
        Path to the font file, or None when nothing matches.
    """
    dirs = ([Path(extra_dir).expanduser()] if extra_dir else []) + FONT_DIRS
    files = [f for d in dirs if d.is_dir() for f in d.rglob("*") if f.suffix.lower() in (".ttf", ".otf")]
    for prefix in prefixes:
        for f in files:
            rest = f.name[len(prefix):]
            if f.name.startswith(prefix) and rest[:1] in (".", "_"):
                return f
    return None


def make_font_loader(font_dir):
    """Build a cached ``load_font(role, size)`` for the roles in layout.FONTS.

    Prints one warning per missing role and falls back to Pillow's bundled
    font, so a machine without the 72 family still renders (less on-brand).
    """
    paths = {}
    for role, prefixes in layout.FONTS.items():
        paths[role] = find_font_file(prefixes, font_dir)
        if paths[role] is None:
            print(f"warning: no 72 font for '{role}' ({prefixes}); using fallback font. "
                  "Install the SAP 72 fonts or pass --font-dir.", file=sys.stderr)

    @lru_cache(maxsize=None)
    def load_font(role, size):
        path = paths[role]
        return ImageFont.truetype(str(path), size) if path else fallback_font(size)

    return load_font


def scaled(box, s):
    """Scale an (x, y, w, h) box from the 1920 basis to output pixels."""
    return tuple(v * s for v in box)


def cover(img, w, h):
    """Resize and crop ``img`` to exactly w x h, keeping its top edge.

    Used for persona photos (faces sit in the upper part of the frame), so
    the crop removes rows from the bottom and equal columns from both sides.
    """
    w, h = int(round(w)), int(round(h))
    ratio = max(w / img.width, h / img.height)
    img = img.resize((max(w, round(img.width * ratio)), max(h, round(img.height * ratio))), Image.LANCZOS)
    left = (img.width - w) // 2
    return img.crop((left, 0, left + w, h))


def fit_screen(img, w, h):
    """Fit a screenshot into the laptop screen without losing its sides.

    Scales to the screen width. A taller result is cropped at the bottom; a
    shorter one is padded below with the screenshot's own bottom-edge color,
    which blends with typical app footers. Left navigation and right-hand
    actions are never cut and nothing is stretched.
    """
    w, h = int(round(w)), int(round(h))
    img = img.resize((w, max(1, round(img.height * w / img.width))), Image.LANCZOS)
    if img.height >= h:
        return img.crop((0, 0, w, h))
    edge = img.crop((0, img.height - 1, w, img.height)).resize((1, 1), Image.BOX).getpixel((0, 0))
    out = Image.new("RGB", (w, h), edge)
    out.paste(img, (0, 0))
    return out


def load_screenshot(source, base_dir):
    """Return the screenshot as an RGB image, or None for the placeholder.

    Args:
        source: path, "clipboard", or None.
        base_dir: folder relative paths resolve against.
    """
    if not source:
        return None
    if source == "clipboard":
        img = ImageGrab.grabclipboard()
        if not isinstance(img, Image.Image):
            raise ValueError("clipboard holds no image; copy the screenshot first")
        return img.convert("RGB")
    return Image.open(base_dir / source).convert("RGB")


def placeholder(w, h, load_font):
    """Neutral grey screen with 'Screenshot pending' text."""
    img = Image.new("RGB", (int(round(w)), int(round(h))), (226, 228, 233))
    d = ImageDraw.Draw(img)
    d.text((img.width / 2, img.height / 2), "Screenshot pending",
           font=load_font("regular", max(12, img.height // 10)), fill=(90, 96, 110), anchor="mm")
    return img


def load_photo(photo, base_dir):
    """Open a persona photo from the bundled library or a user path (RGBA)."""
    bundled = ASSETS / "personas" / photo
    path = bundled if bundled.is_file() else base_dir / photo
    return Image.open(path).convert("RGBA")


def paste_circle(canvas, photo, box):
    """Paste ``photo`` clipped to the circle inscribed in ``box``.

    The circle mask is drawn at 4x and downsampled for a smooth edge, then
    multiplied with the photo's own alpha (bundled photos are cut-outs).
    """
    x, y, w, h = (int(round(v)) for v in box)
    size = min(w, h)
    photo = cover(photo.convert("RGBA"), size, size)
    big = Image.new("L", (size * 4, size * 4), 0)
    ImageDraw.Draw(big).ellipse((0, 0, size * 4 - 1, size * 4 - 1), fill=255)
    mask = big.resize((size, size), Image.LANCZOS)
    alpha = Image.composite(photo.getchannel("A"), Image.new("L", mask.size, 0), mask)
    canvas.paste(photo, (x + (w - size) // 2, y + (h - size) // 2), alpha)


def benefit_runs(text):
    """Split a benefit into [(first word, ACCENT), (rest, WHITE)]."""
    first, _, rest = text.strip().partition(" ")
    return [(first, layout.ACCENT), (rest, layout.WHITE)]


def render(content, base_dir, width=layout.DEFAULT_WIDTH, font_dir=None, screenshot=None):
    """Compose the one-pager and return it as an RGB image.

    Args:
        content: parsed content JSON (see module docstring).
        base_dir: folder for resolving relative paths in ``content``.
        width: output width in px (>= layout.MIN_WIDTH), height is 9/16 of it.
        font_dir: optional extra folder with the 72 fonts.
        screenshot: overrides content["screenshot"] when given.

    Returns:
        PIL.Image in RGB mode.

    Raises:
        SlotOverflow: some text slot does not fit.
        ValueError: invalid width or content.
    """
    if width < layout.MIN_WIDTH:
        raise ValueError(f"width must be at least {layout.MIN_WIDTH}px")
    benefits = content.get("benefits", [])
    if len(benefits) != 3:
        raise ValueError("benefits must contain exactly 3 lines")

    s = width / layout.BASE_WIDTH
    load_font = make_font_loader(font_dir)
    canvas = Image.open(ASSETS / "base_plate.png").convert("RGB")
    canvas = canvas.resize((width, round(width * 9 / 16)), Image.LANCZOS)

    sx, sy, sw, sh = scaled(layout.SCREEN, s)
    shot = load_screenshot(screenshot or content.get("screenshot"), base_dir)
    canvas.paste(fit_screen(shot, sw, sh) if shot else placeholder(sw, sh, load_font),
                 (int(round(sx)), int(round(sy))))

    left, right = content["left_persona"], content["right_persona"]
    paste_circle(canvas, load_photo(left["photo"], base_dir), scaled(layout.LEFT_CIRCLE, s))
    paste_circle(canvas, load_photo(right["photo"], base_dir), scaled(layout.RIGHT_CIRCLE, s))

    W = layout.WHITE
    texts = {
        "title": [[(content["title"], W)]],
        "subtitle": [[(content.get("subtitle") or DEFAULT_SUBTITLE, layout.ACCENT)]],
        "app_name": [[(content["app_name"], W)]],
        "left_name": [[(left["name"], W)]],
        "left_role": [[(left["role"], layout.ACCENT)]],
        "right_name": [[(right["name"], W)]],
        "right_role": [[(right["role"], layout.ACCENT)]],
        "left_caption": [[(left["caption"], W)]],
        "right_caption": [[(right["caption"], W)]],
        "challenge": [[(content["challenge"], W)]],
        "solution": [[(content["solution"], W)]],
        "benefits": [benefit_runs(b) for b in benefits],
    }
    draw = ImageDraw.Draw(canvas)
    errors = []
    for name, spec in layout.SLOTS.items():
        spec = dict(spec, box=scaled(spec["box"], s), size=spec["size"] * s, min_size=spec["min_size"] * s)
        try:
            draw_slot(draw, name, spec, texts[name], load_font)
        except SlotOverflow as exc:
            errors.append(str(exc))
    if errors:
        raise SlotOverflow("\n".join(errors))
    return canvas


def main():
    """CLI entry point: parse arguments, render, save the PNG."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--content", required=True, help="content JSON file")
    ap.add_argument("--output", required=True, help="output PNG path")
    ap.add_argument("--width", type=int, default=layout.DEFAULT_WIDTH,
                    help=f"output width in px, minimum {layout.MIN_WIDTH} (default {layout.DEFAULT_WIDTH})")
    ap.add_argument("--font-dir", help="extra folder that holds the SAP 72 fonts")
    ap.add_argument("--screenshot", help='override the screenshot: a path or "clipboard"')
    args = ap.parse_args()

    content_path = Path(args.content)
    content = json.loads(content_path.read_text(encoding="utf-8"))
    shot = args.screenshot
    if shot and shot != "clipboard":
        shot = str(Path(shot).resolve())
    try:
        img = render(content, content_path.parent, args.width, args.font_dir, shot)
    except SlotOverflow as exc:
        print(f"overflow:\n{exc}", file=sys.stderr)
        sys.exit(2)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"wrote {out} ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
