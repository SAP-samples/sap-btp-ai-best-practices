"""Self-check for the one-pager renderer and persona picker.

Renders assets/example_content.json at the default (4K) and minimum (1080p)
widths, checks the size and aspect ratio, checks that an over-long challenge
is rejected with a "shorten" message instead of being clipped, and runs the
persona picker self-check. Writes nothing outside a temp folder.

Example:
    python scripts/check_one_pager.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout  # noqa: E402
import pick_personas  # noqa: E402
from render_one_pager import ASSETS, render  # noqa: E402
from text_fit import SlotOverflow  # noqa: E402


def main():
    """Run all checks; exit non-zero (AssertionError) on the first failure."""
    content = json.loads((ASSETS / "example_content.json").read_text())

    for width in (layout.DEFAULT_WIDTH, layout.MIN_WIDTH):
        img = render(content, ASSETS, width)
        assert img.size == (width, width * 9 // 16), img.size
    with tempfile.TemporaryDirectory() as tmp:
        img.save(Path(tmp) / "check.png")

    long = dict(content, challenge=" ".join(["Manual reconciliation slows every review."] * 30))
    try:
        render(long, ASSETS, layout.MIN_WIDTH)
        raise AssertionError("over-long challenge was not rejected")
    except SlotOverflow as exc:
        assert "challenge" in str(exc) and "shorten" in str(exc), exc

    try:
        render(content, ASSETS, 1280)
        raise AssertionError("width below 1920 was accepted")
    except ValueError:
        pass

    pick_personas._self_check()
    print("one-pager self-check passed")


if __name__ == "__main__":
    main()
