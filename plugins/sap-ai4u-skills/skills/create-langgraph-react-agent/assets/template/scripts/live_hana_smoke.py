"""Validate the real HANA memory table, leases, and versioned saves.

Two store objects stand in for two Cloud Foundry instances sharing one HANA
table. The script uses one temporary context row and deletes it at the end.

Examples:
    .venv/bin/python -m scripts.live_hana_smoke
    .venv/bin/python -m scripts.live_hana_smoke --table LANGGRAPH_AGENT_MEMORY_SMOKE
"""

from __future__ import annotations

import argparse
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from tqdm import tqdm

from template_agent.config import MemorySettings
from template_agent.memory import ConcurrentUpdateError, HanaConversationStore


def _check(condition: bool, message: str) -> None:
    """Raise a readable failure when one smoke assertion does not hold."""

    if not condition:
        raise RuntimeError(message)


def main() -> int:
    """Run every live check against one temporary context and clean it up."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table", default="LANGGRAPH_AGENT_MEMORY")
    args = parser.parse_args()

    load_dotenv(override=False)
    settings = MemorySettings(enabled=True, table_name=args.table)
    instance_a = HanaConversationStore(settings)
    instance_b = HanaConversationStore(settings)
    context_id = f"template-smoke-{uuid4()}"
    owner_a, owner_b = f"a-{uuid4().hex[:8]}", f"b-{uuid4().hex[:8]}"
    messages = [
        HumanMessage(content="memory smoke"),
        AIMessage(
            content="",
            tool_calls=[{"name": "echo", "args": {"v": 1}, "id": "c1", "type": "tool_call"}],
        ),
        ToolMessage(content="1", tool_call_id="c1"),
        AIMessage(content="memory ok"),
    ]
    pending = {"tool_calls": [{"name": "echo", "args": {"v": 2}, "id": "c2"}]}

    def create_and_migrate() -> None:
        """Create or migrate the table from both instances."""

        instance_a.ensure()
        instance_b.ensure()

    def lease_is_exclusive() -> None:
        """A creates and leases the row; B cannot take it while A holds it."""

        _check(instance_a.acquire(context_id, owner_a, 60), "A could not take a new lease")
        _check(not instance_b.acquire(context_id, owner_b, 60), "B took A's live lease")

    def save_releases_and_versions() -> None:
        """A saves full messages and a pending action; the lease is freed."""

        version = instance_a.load(context_id).version
        instance_a.save(context_id, owner_a, version, messages, pending)
        stored = instance_b.load(context_id)
        _check(stored.version == version + 1, f"Version did not advance: {stored.version}")
        _check(stored.pending_action == pending, "Pending action did not round-trip")
        _check(
            [type(m) for m in stored.messages] == [type(m) for m in messages],
            "Message types did not round-trip",
        )
        _check(stored.messages[1].tool_calls[0]["id"] == "c1", "Tool call lost")

    def stale_save_is_rejected() -> None:
        """B takes the lease; A's stale save with the old version must fail."""

        _check(instance_b.acquire(context_id, owner_b, 60), "B could not take freed lease")
        try:
            instance_a.save(context_id, owner_a, 0, [], None)
        except ConcurrentUpdateError:
            pass
        else:
            raise RuntimeError("Stale save from A overwrote B's turn")
        instance_b.release(context_id, owner_b)

    def expired_lease_is_taken_over() -> None:
        """An expired lease can be taken by the other instance."""

        _check(instance_a.acquire(context_id, owner_a, -1), "A could not take lease")
        _check(instance_b.acquire(context_id, owner_b, 60), "B could not take expired lease")
        instance_b.release(context_id, owner_b)

    checks = [
        create_and_migrate,
        lease_is_exclusive,
        save_releases_and_versions,
        stale_save_is_rejected,
        expired_lease_is_taken_over,
    ]
    initialized = False
    try:
        for check in tqdm(checks, desc="HANA memory checks", unit="check"):
            check()
            initialized = True
        instance_a.clear(context_id)
        _check(not instance_a.load(context_id).messages, "Temporary context was not removed")
    finally:
        # A second parameterized delete is harmless if the first one succeeded.
        if initialized:
            instance_a.clear(context_id)
        instance_a.close()
        instance_b.close()
    print("HANA memory smoke test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
