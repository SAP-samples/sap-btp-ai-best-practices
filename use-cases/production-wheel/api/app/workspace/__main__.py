"""Operate the HANA-backed production-wheel workspace.

Examples (from api/):
    ../.venv/bin/python -m app.workspace worker
    ../.venv/bin/python -m app.workspace worker --run-id RUN_ID
    ../.venv/bin/python -m app.workspace import-bundle job_runs/JOB/solution --dataset-directory job_runs/SNAPSHOT
    ../.venv/bin/python -m app.workspace retry-publication JOB
    ../.venv/bin/python -m app.workspace check-hana
"""

import argparse
import json
import logging
from pathlib import Path
from .dependencies import get_service


def main():
    """Dispatch a documented administrative command and return shell status."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("worker").add_argument("--run-id", help="Execute only this queued run, then exit")
    sub.add_parser("check-hana")
    for name in ("execute", "retry-publication"):
        sub.add_parser(name).add_argument("run_id")
    migration = sub.add_parser("import-bundle")
    migration.add_argument("solution_directory", type=Path)
    migration.add_argument("--dataset-directory", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("pyomo.contrib.appsi.solvers.highs").setLevel(logging.WARNING)
    service = get_service()
    if args.command == "worker":
        import signal
        import threading
        import uuid
        from .worker import worker_loop, supervise_once

        stop = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        if args.run_id:
            supervise_once(service, uuid.uuid4().hex, run_id=args.run_id, stop=stop)
        else:
            worker_loop(service, stop=stop)
    elif args.command == "execute":
        from .worker import execute_run

        execute_run(service, args.run_id)
    elif args.command == "retry-publication":
        from .worker import retry_publication

        print(json.dumps(retry_publication(service, args.run_id), default=str))
    elif args.command == "import-bundle":
        from .migration import import_history
        from tqdm import tqdm

        with tqdm(total=1, desc="Validate and import historical bundle") as bar:
            result = import_history(
                service, args.solution_directory, args.dataset_directory
            )
            bar.update(1)
        print(
            json.dumps(
                {
                    "run_id": result["run_id"],
                    "dataset_id": result["dataset_id"],
                    "results_ready": result["results_ready"],
                }
            )
        )
    else:
        service.repo.ensure()
        print("HANA workspace schema ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
