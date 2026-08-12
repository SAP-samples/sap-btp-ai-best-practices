#!/usr/bin/env python3
"""
upload_session.py — Upload your local browser profile to the CF backend.

Run this AFTER logging into IBP manually in your local browser
(via ./start.sh or directly). This sends your cookies/session to CF
so the headless extraction can authenticate automatically.

Usage:
    python upload_session.py
    python upload_session.py --backend https://<your-cf-backend-host>
"""
import argparse
import os
import io
import sys
import zipfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / "backend" / ".env")

BROWSER_PROFILE = Path(os.getenv("BROWSER_PROFILE", "./backend/browser_profile")).resolve()
CF_BACKEND      = os.getenv("CF_BACKEND_URL",
                             "https://<your-cf-backend-host>")


def zip_profile(profile_path: Path) -> bytes:
    """Zip the entire browser_profile directory in memory."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in profile_path.rglob("*"):
            if f.is_file():
                # Skip lock files and singleton files that cause issues
                if f.name in ("SingletonLock", "SingletonCookie", "SingletonSocket",
                              "lockfile", ".com.google.Chrome"):
                    continue
                arcname = f.relative_to(profile_path)
                zf.write(f, arcname)
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser(description="Upload browser session to CF backend")
    parser.add_argument("--backend", default=CF_BACKEND, help="CF backend URL")
    args = parser.parse_args()

    import requests

    if not BROWSER_PROFILE.exists():
        print(f"ERROR: Browser profile not found at {BROWSER_PROFILE}")
        print("Run ./start.sh first and log into IBP at least once.")
        sys.exit(1)

    # Check profile has cookies
    cookies_exist = (
        (BROWSER_PROFILE / "Default" / "Cookies").exists() or
        (BROWSER_PROFILE / "Default" / "Network" / "Cookies").exists()
    )
    if not cookies_exist:
        print("WARNING: No cookies found in browser profile.")
        print("Make sure you logged into IBP at least once with ./start.sh")

    print(f"Zipping browser profile from: {BROWSER_PROFILE}")
    profile_zip = zip_profile(BROWSER_PROFILE)
    size_mb = len(profile_zip) / 1_048_576
    print(f"Profile zip size: {size_mb:.1f} MB")

    url = f"{args.backend.rstrip('/')}/api/session/upload"
    print(f"Uploading to: {url}")

    resp = requests.post(
        url,
        files={"file": ("browser_profile.zip", profile_zip, "application/zip")},
        timeout=120,
    )

    if resp.status_code == 200:
        result = resp.json()
        uploaded_mb = result.get("profileSizeBytes", 0) / 1_048_576
        print(f"✓ Session uploaded successfully ({uploaded_mb:.1f} MB on server)")
        print("The CF backend will now use your cookies for IBP authentication.")
    else:
        print(f"✗ Upload failed ({resp.status_code}): {resp.text[:300]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
