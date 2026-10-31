#!/usr/bin/env python3
"""
extract.py — Local extraction script for IBP Optimizer Log Extractor.

Run this on your Mac or Windows machine.
It opens Chrome, downloads the Supply Planning Logs from SAP IBP,
then uploads the result to the CF backend.

Usage:
    python extract.py
    python extract.py --job "My Job Name"
    python extract.py --backend https://<your-cf-backend-host>
    python extract.py --no-upload   # extract only, don't upload
"""
import argparse
import hashlib
import io
import json
import os
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / "backend" / ".env")

DEFAULT_JOB     = os.getenv("JOB_NAME", "<your-ibp-job-name>")
IBP_URL         = os.getenv("IBP_URL",  "https://<your-ibp-tenant>.scmibp.ondemand.com/")
DOWNLOAD_DIR    = Path(os.getenv("DOWNLOAD_DIR", "./backend/downloads")).resolve()
BROWSER_PROFILE = Path(os.getenv("BROWSER_PROFILE", "./backend/browser_profile")).resolve()
CF_BACKEND      = os.getenv("CF_BACKEND_URL",
                             "https://<your-cf-backend-host>")

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
BROWSER_PROFILE.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def extract(job_name: str) -> Path:
    """Run Playwright and download the Supply Planning Logs zip/csv. Returns saved path."""
    import platform
    is_win = platform.system() == "Windows"

    log(f"Job: {job_name}")
    log("Launching Chrome...")

    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            user_data_dir=str(BROWSER_PROFILE),
            channel="chrome",
            headless=False,
            slow_mo=150,
            accept_downloads=True,
            ignore_https_errors=False,
        )
        page = ctx.pages[0] if ctx.pages else ctx.new_page()

        log(f"Navigating to {IBP_URL} ...")
        page.goto(IBP_URL, wait_until="domcontentloaded", timeout=90_000)

        log("Waiting for Fiori shell...")
        page.wait_for_selector(
            ".sapUiShellHead, .sapShellHead, .sapUshellShellHeadTitle, "
            "#shell-header, [id*='shellHeader'], [class*='ShellHeader']",
            timeout=120_000,
        )
        log("Session active.")

        log("Opening Application Jobs...")
        try:
            page.locator("text=Application Jobs").first.click(timeout=10_000)
        except PWTimeout:
            page.goto(IBP_URL + "#Shell-home", timeout=30_000)
            page.wait_for_load_state("domcontentloaded")
            page.locator("text=Application Jobs").first.click(timeout=30_000)
        page.wait_for_load_state("domcontentloaded")

        log(f"Filtering by: '{job_name}'...")
        try:
            search = page.locator(
                "input[placeholder*='Search'], input[placeholder*='Filter'], "
                ".sapMSearchField input"
            ).first
            search.fill(job_name, timeout=10_000)
            search.press("Enter")
            page.wait_for_load_state("domcontentloaded")
        except PWTimeout:
            log("Search field not found — continuing without filter.")

        log("Verifying Status = Finished...")
        finished_row = page.locator(
            f"tr:has-text('Finished'):has-text('{job_name[:30]}')"
        ).first
        finished_row.wait_for(state="visible", timeout=15_000)
        log("Finished job found.")

        # Wait for any busy indicator to disappear before clicking
        log("Waiting for table to finish loading...")
        try:
            page.wait_for_selector(
                ".sapUiLocalBusyIndicator, [role='progressbar']",
                state="hidden", timeout=15_000
            )
        except PWTimeout:
            pass  # No busy indicator found — proceed anyway
        page.wait_for_timeout(500)

        log("Clicking Log icon...")
        log_cell = finished_row.locator("td").nth(1)
        log_btn  = log_cell.locator("button, a, [role='button']").first
        log_btn.click(timeout=15_000)
        page.wait_for_load_state("domcontentloaded")

        log("Locating 'Supply Planning Logs' attachment...")
        supply_link = page.locator(
            "tr:has-text('Optimizer explanation log created') >> text=Supply Planning Logs"
        ).first
        try:
            supply_link.wait_for(state="visible", timeout=20_000)
        except PWTimeout:
            supply_link = page.locator("text=Supply Planning Logs").nth(1)
            supply_link.wait_for(state="visible", timeout=10_000)

        log("Downloading...")
        with page.expect_download(timeout=60_000) as dl_info:
            supply_link.click()

        download  = dl_info.value
        orig_name = download.suggested_filename
        ts        = datetime.now().strftime("%Y%m%d_%H%M")
        save_name = f"IBP1_TS_OPTIMIZER_LOG_{ts}_{orig_name}"
        save_path = DOWNLOAD_DIR / save_name
        download.save_as(str(save_path))
        log(f"Saved: {save_path}")

        ctx.close()

    return save_path


def upload(file_path: Path, job_name: str, backend_url: str):
    """Upload the downloaded file to the CF backend."""
    url = f"{backend_url.rstrip('/')}/api/upload"
    log(f"Uploading to {url} ...")
    with open(file_path, "rb") as f:
        resp = requests.post(
            url,
            files={"file": (file_path.name, f, "application/octet-stream")},
            data={"jobName": job_name},
            timeout=120,
        )
    if resp.status_code == 200:
        result = resp.json()
        log(f"Upload successful: {result.get('saved')}")
        hs = result.get("hanaStatus", {})
        for csv_name, info in hs.items():
            if csv_name != "error":
                log(f"  HANA: {csv_name} → {info.get('table')}: {info.get('inserted')} rows")
        if "error" in hs:
            log(f"  HANA error: {hs['error']}")
    else:
        log(f"Upload failed ({resp.status_code}): {resp.text[:300]}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IBP Optimizer Log extractor (local)")
    parser.add_argument("--job",       default=DEFAULT_JOB,  help="Application Job name")
    parser.add_argument("--backend",   default=CF_BACKEND,   help="CF backend URL")
    parser.add_argument("--no-upload", action="store_true",  help="Skip upload to CF")
    args = parser.parse_args()

    file_path = extract(args.job)

    if not args.no_upload:
        upload(file_path, args.job, args.backend)
    else:
        log("Skipping upload (--no-upload). File saved locally.")

    log("Done.")
