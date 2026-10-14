#!/usr/bin/env python3
"""
Screenshot an HTML file to PNG with a headless browser.

Usage:
    python render_png.py mockup.html out.png [--width 2000] [--height 1250] [--full-page]

Used by render_diagram.py; also usable for any HTML page you need as an image.
Order of attempts: Playwright (python) -> Chrome/Chromium CLI. If neither exists:
    pip install playwright && python -m playwright install chromium
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def _chrome_binaries():
    cands = [shutil.which(n) for n in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome")]
    cands += [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    cands += sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome"))
    cands += sorted(glob.glob(os.path.expanduser("~/Library/Caches/ms-playwright/chromium-*/chrome-mac*/Chromium.app/Contents/MacOS/Chromium")))
    cands += sorted(glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux/chrome")))
    return [c for c in cands if c and os.path.exists(c)]


def read_render_size(html_path: Path):
    m = re.search(r'<meta\s+name="render-size"\s+content="(\d+),(\d+)"', html_path.read_text(encoding="utf-8", errors="ignore"))
    return (int(m.group(1)), int(m.group(2))) if m else None


def screenshot(html_path, png_path, width=2000, height=1250, full_page=False, scale=1):
    html_path, png_path = Path(html_path).resolve(), Path(png_path).resolve()
    size = read_render_size(html_path)
    if size:
        width, height = size
        full_page = False
    png_path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
        with sync_playwright() as p:
            kw = {}
            exe = os.environ.get("CHROMIUM_PATH")
            if exe:
                kw["executable_path"] = exe
            try:
                browser = p.chromium.launch(**kw)
            except Exception:
                bins = _chrome_binaries()
                if not bins:
                    raise
                browser = p.chromium.launch(executable_path=bins[0])
            page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=scale)
            page.goto(html_path.as_uri())
            page.wait_for_timeout(300)
            need = page.evaluate("() => Math.max(...[...document.body.querySelectorAll('*')].map(e => e.getBoundingClientRect().bottom))")
            if need > height + 2 and not full_page:
                print(f"WARNING: content extends to {int(need)}px but the canvas is {height}px tall and will be cropped. "
                      f"Trim content or set <meta name=\"render-size\" content=\"{width},{int(need) + 20}\">", file=sys.stderr)
            page.screenshot(path=str(png_path), full_page=full_page)
            browser.close()
        return png_path
    except Exception as e:  # noqa
        errors.append(f"playwright: {e}")
    for b in _chrome_binaries():
        try:
            subprocess.run([b, "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
                            f"--force-device-scale-factor={scale}", f"--window-size={width},{height}",
                            f"--screenshot={png_path}", html_path.as_uri()],
                           check=True, capture_output=True, timeout=120)
            if png_path.exists():
                return png_path
        except Exception as e:  # noqa
            errors.append(f"{b}: {e}")
    raise RuntimeError("No headless browser available. Install one with:\n"
                       "  pip install playwright && python -m playwright install chromium\n" + "\n".join(errors))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("html")
    ap.add_argument("png")
    ap.add_argument("--width", type=int, default=2000)
    ap.add_argument("--height", type=int, default=1250)
    ap.add_argument("--full-page", action="store_true")
    ap.add_argument("--scale", type=float, default=1)
    a = ap.parse_args()
    out = screenshot(a.html, a.png, a.width, a.height, a.full_page, a.scale)
    print(out)


if __name__ == "__main__":
    sys.exit(main())
