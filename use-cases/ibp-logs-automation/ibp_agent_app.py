"""
IBP Local Agent — system tray app.
Runs local_agent.py Flask server on port 5001 and shows a system tray icon.
Double-click or distribute as IBP-Agent.app (Mac) / IBP-Agent.exe (Windows).
"""
import os
import sys
import threading
import webbrowser
from pathlib import Path

# ── Determine base path (works both from source and PyInstaller bundle) ───────
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys._MEIPASS)          # PyInstaller temp dir
    APP_DIR  = Path(sys.executable).parent  # folder next to the .exe/.app
else:
    BASE_DIR = Path(__file__).parent
    APP_DIR  = BASE_DIR

CF_UI_URL = os.getenv("CF_UI_URL", "https://<your-cf-ui-host>")
PORT      = 5001


def _run_flask():
    """Start the Flask local agent server."""
    # Add BASE_DIR to path so local_agent can import its deps
    sys.path.insert(0, str(BASE_DIR))
    os.chdir(str(APP_DIR))

    # Load .env from app directory if present
    env_file = APP_DIR / ".env"
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)

    # Kill any process already on port 5001
    import subprocess
    try:
        result = subprocess.run(["lsof", "-ti", f":{PORT}"],
                                capture_output=True, text=True)
        for pid in result.stdout.strip().split():
            if pid:
                subprocess.run(["kill", "-9", pid], capture_output=True)
    except Exception:
        pass

    import local_agent as agent
    agent.app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)


def _create_tray_icon():
    """Create a system tray icon with menu."""
    from PIL import Image, ImageDraw
    import pystray

    # Simple colored square icon
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([4, 4, 60, 60], fill="#0a6ed1")
    draw.text((18, 20), "IBP", fill="white")

    def on_open(_):
        webbrowser.open(CF_UI_URL)

    def on_quit(icon):
        icon.stop()
        os._exit(0)

    menu = pystray.Menu(
        pystray.MenuItem("Open IBP Extractor", on_open, default=True),
        pystray.MenuItem(f"Running on localhost:{PORT}", None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", on_quit),
    )
    icon = pystray.Icon("IBP Agent", img, "IBP Agent", menu)
    return icon


def main():
    # Start Flask in background thread
    flask_thread = threading.Thread(target=_run_flask, daemon=True)
    flask_thread.start()

    # Give Flask a moment to start, then open the browser
    import time
    time.sleep(1.5)
    webbrowser.open(CF_UI_URL)

    # Show tray icon (blocks until quit)
    try:
        icon = _create_tray_icon()
        icon.run()
    except Exception:
        # Fallback if pystray not available — just keep Flask running
        print(f"IBP Agent running on http://localhost:{PORT}")
        print(f"Opening {CF_UI_URL}")
        print("Press Ctrl+C to stop.")
        flask_thread.join()


if __name__ == "__main__":
    main()
