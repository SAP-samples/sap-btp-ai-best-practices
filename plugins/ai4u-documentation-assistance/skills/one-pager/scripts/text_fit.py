"""Word wrapping, auto-shrink and colored-run drawing for the one-pager.

A slot holds one or more paragraphs. Each paragraph is a list of runs
``(text, color)``; this is how a benefit line gets its first verb in the accent
color and the rest in white. The fitter tries the slot's nominal font size,
then shrinks one pixel at a time down to ``min_size``. If the text still does
not fit, it raises ``SlotOverflow`` with an estimate of how many words to cut,
so the agent rewrites the copy instead of shipping a clipped image.

Inputs: a PIL ImageDraw, a slot spec from ``layout.SLOTS`` (already scaled to
the output width), paragraphs of runs, and a font loader.
Outputs: text drawn on the image; ``SlotOverflow`` when it cannot fit.
"""

from PIL import ImageFont


class SlotOverflow(Exception):
    """Raised when a slot's text does not fit even at its minimum font size."""


def _tokens(paragraph):
    """Split a paragraph of (text, color) runs into (word, color) tokens.

    Args:
        paragraph: list of (text, color) tuples.

    Returns:
        list of (word, color) tuples, whitespace removed.
    """
    return [(w, color) for text, color in paragraph for w in text.split()]


def _wrap(tokens, font, max_width):
    """Greedy word wrap of colored tokens.

    Args:
        tokens: list of (word, color).
        font: PIL FreeTypeFont used for measuring.
        max_width: available line width in px.

    Returns:
        list of lines, each a list of (word, color). A single word wider than
        the box gets its own line (the fitter then shrinks the font).
    """
    space = font.getlength(" ")
    lines, line, width = [], [], 0.0
    for word, color in tokens:
        w = font.getlength(word)
        needed = w if not line else width + space + w
        if line and needed > max_width:
            lines.append(line)
            line, width = [(word, color)], w
        else:
            line.append((word, color))
            width = needed
    if line:
        lines.append(line)
    return lines


def _line_width(line, font):
    """Return the rendered width of one wrapped line in px."""
    return sum(font.getlength(w) for w, _ in line) + font.getlength(" ") * (len(line) - 1)


def _layout(paragraphs, font, spec, size):
    """Wrap all paragraphs at one font size and report whether they fit.

    Returns:
        (fits, wrapped_paragraphs, total_height)
    """
    x, y, w, h = spec["box"]
    line_h = size * spec.get("line", 1.2)
    gap = size * spec.get("para_gap", 0)
    wrapped = [_wrap(_tokens(p), font, w) for p in paragraphs]
    if spec.get("single_line"):
        fits = all(len(p) <= 1 for p in wrapped)
    else:
        fits = all(_line_width(l, font) <= w for p in wrapped for l in p)
    total = sum(len(p) for p in wrapped) * line_h + gap * (len(wrapped) - 1)
    return fits and total <= h, wrapped, total


def draw_slot(draw, name, spec, paragraphs, load_font):
    """Fit and draw one slot, centered horizontally.

    Args:
        draw: PIL ImageDraw.Draw of the output image.
        name: slot name, used in error messages.
        spec: slot spec with box/size/min_size already scaled to output px.
        paragraphs: list of paragraphs, each a list of (text, color) runs.
        load_font: callable(font_role, size_px) -> FreeTypeFont.

    Returns:
        The font size in px that was used.

    Raises:
        SlotOverflow: text does not fit at ``min_size``.
    """
    x, y, w, h = spec["box"]
    size = int(round(spec["size"]))
    min_size = int(round(spec["min_size"]))
    while True:
        font = load_font(spec["font"], size)
        fits, wrapped, total = _layout(paragraphs, font, spec, size)
        if fits or size <= min_size:
            break
        size -= 1
    if not fits:
        words = sum(len(_tokens(p)) for p in paragraphs)
        cut = max(1, round(words * (1 - h / total))) if total > h else 1
        raise SlotOverflow(
            f"slot '{name}' overflows at the minimum font size; "
            f"shorten it by about {cut} of its {words} words"
        )

    line_h = size * spec.get("line", 1.2)
    gap = size * spec.get("para_gap", 0)
    top = y + (h - total) / 2 if spec.get("valign") == "middle" else y
    space = font.getlength(" ")
    cy = top + line_h / 2
    for para in wrapped:
        for line in para:
            cx = x + (w - _line_width(line, font)) / 2
            for word, color in line:
                # "lm" anchor = left edge, vertical middle of the line box.
                draw.text((cx, cy), word, font=font, fill=color, anchor="lm")
                cx += font.getlength(word) + space
            cy += line_h
        cy += gap
    return size


def fallback_font(size):
    """Return Pillow's bundled scalable font when the 72 family is missing."""
    return ImageFont.load_default(size=size)
