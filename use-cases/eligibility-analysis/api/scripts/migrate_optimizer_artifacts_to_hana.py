#!/usr/bin/env python3
"""Backfill optimizer run artifacts into the configured database backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from app.services.optimizer.process_manager import DATA_DIR, ProcessManager


def _counts_for_process(process_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    try:
        import pandas as pd
    except Exception:
        return counts

    for name, bucket in (
        ("selected.xlsx", "selected"),
        ("excluded.xlsx", "excluded"),
        ("weekly_plan.xlsx", "weekly_plan"),
        ("weekly_exposure.xlsx", "weekly_exposure"),
    ):
        path = process_dir / name
        if not path.exists():
            counts[bucket] = 0
            continue
        try:
            counts[bucket] = int(len(pd.read_excel(path, engine="openpyxl")))
        except Exception:
            counts[bucket] = -1
    return counts


def migrate_process(process_id: str, manager: ProcessManager, force: bool = False) -> Dict[str, Any]:
    record = manager.get_process(process_id)
    if record is None:
        return {"process_id": process_id, "status": "missing_process_record"}

    process_dir = Path(record["process_dir"])
    if not process_dir.exists():
        return {"process_id": process_id, "status": "missing_process_dir"}

    source_counts = _counts_for_process(process_dir)
    persisted = manager.sync_process_artifacts(process_id, force=force)
    return {
        "process_id": process_id,
        "status": "migrated",
        "source_counts": source_counts,
        "persisted_counts": persisted,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--process-id", help="Only migrate a single process id")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Optimizer runs directory")
    parser.add_argument("--force", action="store_true", help="Replace existing persisted artifacts")
    parser.add_argument("--dry-run", action="store_true", help="List candidate processes without writing")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    manager = ProcessManager()
    results = []

    if args.process_id:
        process_ids = [args.process_id]
    elif data_dir.exists():
        process_ids = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
    else:
        process_ids = []

    for process_id in process_ids:
        if args.dry_run:
            record = manager.get_process(process_id)
            results.append(
                {
                    "process_id": process_id,
                    "status": "dry_run",
                    "has_process_record": record is not None,
                    "process_dir": str(Path(record["process_dir"])) if record else None,
                }
            )
            continue
        results.append(migrate_process(process_id, manager, force=args.force))

    print(json.dumps({"results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
