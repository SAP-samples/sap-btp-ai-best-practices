"""Eval result persistence — saves timestamped JSON to tests/results/."""
from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

_RESULTS_DIR = Path(__file__).parent / "results"


def _git_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def save_batch_result(
    test_type: str,
    data: Dict[str, Any],
    *,
    runs_total: int = 1,
) -> str:
    """Save a quality scorecard result.

    Writes to:
      - tests/results/{test_type}/{timestamp}_{commit}_batch.json
      - tests/results/{test_type}/latest_batch.json
    """
    out_dir = _RESULTS_DIR / test_type
    out_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    commit = _git_commit_short()

    data["timestamp"] = now.isoformat(timespec="seconds")
    data["git_commit"] = commit
    data["test_type"] = test_type
    data["batch_id"] = now.strftime("%Y%m%d%H%M%S")
    data["runs_total"] = runs_total

    filename = f"{now.strftime('%Y-%m-%dT%H-%M-%S')}_{commit}_batch.json"
    run_path = out_dir / filename

    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    latest_path = out_dir / "latest_batch.json"
    shutil.copy2(run_path, latest_path)

    return str(run_path)
