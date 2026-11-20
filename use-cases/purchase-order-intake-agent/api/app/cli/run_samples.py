"""Run the purchase-order agent end to end on sample files and print the outcome per order.

Each file becomes its own order (intake), the agent processes up to --concurrency orders in
parallel, then the deterministic gate sets the final status. Without --post, orders that pass
the gate stay NEEDS_REVIEW with the message "Ready to post" (nothing is written to the mock ECC).

Examples (run from the api/ folder, credentials in api/.env):

    ../.venv/bin/python -m app.cli.run_samples "../data/orders/PO to MRY on the 20-12-2024.pdf"
    ../.venv/bin/python -m app.cli.run_samples ../data/orders --concurrency 4
    ../.venv/bin/python -m app.cli.run_samples "../data/orders/6k Ilokuw 55285196.pdf" --sales-org DE10
    ../.venv/bin/python -m app.cli.run_samples ../data/orders --recursive --skip "price list"
    ../.venv/bin/python -m app.cli.run_samples ../data/orders --post
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from app.db.engine import get_engine  # noqa: E402
from app.db.schema import ensure_schema  # noqa: E402
from app.intake.service import ACCEPTED_SUFFIXES, UploadedFile, intake_upload  # noqa: E402
from app.order_agent.runner import close_runtime, run_order_agent  # noqa: E402
from app.orders import repository  # noqa: E402
from app.orders.matching import alpha_out  # noqa: E402
from app.orders.service import finalize_after_agent  # noqa: E402


def collect(paths: list[str], recursive: bool, skip: list[str]) -> list[Path]:
    """Expand files and folders into supported sample files.

    Args:
        paths: Files or folders.
        recursive: Descend into sub-folders.
        skip: Case-insensitive substrings of file names to ignore (e.g. "price list").

    Returns:
        Sorted list of files.
    """
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        candidates = (path.rglob("*") if recursive else path.glob("*")) if path.is_dir() else [path]
        files += [p for p in candidates if p.is_file() and p.suffix.lower() in ACCEPTED_SUFFIXES]
    return sorted({f for f in files if not any(s.lower() in f.name.lower() for s in skip)})


async def process(order_id: str, semaphore: asyncio.Semaphore, post) -> tuple[str, object]:
    """Run agent + gate for one order under the concurrency limit."""
    async with semaphore:
        await asyncio.to_thread(repository.set_status, order_id, "PROCESSING", "Agent running (CLI)", ATTEMPTS=1)
        outcome = await run_order_agent(order_id)
        try:
            if outcome.outcome in ("ERROR", "TIMEOUT"):
                await asyncio.to_thread(repository.set_status, order_id, "FAILED", outcome.error)
            else:
                await asyncio.to_thread(finalize_after_agent, order_id, outcome.summary, post)
        except Exception as exc:  # noqa: BLE001 - one broken order must not stop the batch
            await asyncio.to_thread(repository.set_status, order_id, "FAILED", f"{type(exc).__name__}: {exc}"[:2000])
        return order_id, outcome


async def main_async(args: argparse.Namespace) -> int:
    """Intake all files, run the agent concurrently and print a result table."""
    files = collect(args.paths, args.recursive, args.skip)
    if not files:
        print("No supported files found.")
        return 1
    await asyncio.to_thread(ensure_schema, get_engine())
    orders: dict[str, Path] = {}
    for path in files:
        created = await asyncio.to_thread(
            intake_upload, [UploadedFile(path.name, None, path.read_bytes())], None, args.sales_org)
        for order in created:
            orders[order["order_id"]] = path
    post = None
    if args.post:
        from app.orders.posting import post_order  # available from the posting phase on
        post = post_order
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [asyncio.create_task(process(order_id, semaphore, post)) for order_id in orders]
    outcomes = {}
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing orders", unit="order"):
        order_id, outcome = await task
        outcomes[order_id] = outcome
    await close_runtime()

    print()
    for order_id, path in orders.items():
        header = repository.get_header(order_id)
        outcome = outcomes[order_id]
        issues = [s for s in repository.get_field_states(order_id) if s.get("ISSUE_SEVERITY") in ("ERROR", "REVIEW")]
        print(f"{path.name[:48]:<48} {header['STATUS']:<13} {header.get('CLASSIFICATION') or '-':<12} "
              f"PO={header.get('PO_NUMBER') or '-'}  sold-to={alpha_out(header.get('SOLD_TO')) or '-'}  "
              f"ship-to={alpha_out(header.get('SHIP_TO')) or '-'}  lines={len(repository.get_items(order_id))}  "
              f"{outcome.seconds:.0f}s  calls={outcome.llm_calls} in={outcome.input_tokens} "
              f"cached={outcome.cached_input_tokens} out={outcome.output_tokens}")
        print(f"    {header.get('STATUS_MESSAGE') or ''}")
        for issue in issues[:6]:
            print(f"    - {issue['ISSUE_SEVERITY']:<6} item {issue['ITEM_NO']:>3} {issue['FIELD_NAME']:<18} {issue.get('ISSUE_TEXT')}"[:220])
        if args.verbose and header.get("AGENT_SUMMARY"):
            print(f"    summary: {header['AGENT_SUMMARY']}"[:1200])
        print(f"    order id: {order_id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="sample files or folders")
    parser.add_argument("--recursive", action="store_true", help="descend into sub-folders")
    parser.add_argument("--skip", nargs="*", default=["price list"], help="file-name substrings to ignore")
    parser.add_argument("--sales-org", default=None, help="sales organisation hint (simulated inbox mapping)")
    parser.add_argument("--concurrency", type=int, default=4, help="orders processed in parallel")
    parser.add_argument("--post", action="store_true", help="post orders that pass the gate to the mock ECC")
    parser.add_argument("--verbose", action="store_true", help="print the agent summary")
    return asyncio.run(main_async(parser.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
