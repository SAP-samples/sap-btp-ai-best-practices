"""Validate the real HANA memory table with one temporary context row.

Example:
    .venv/bin/python -m scripts.live_hana_smoke
"""

from __future__ import annotations

from uuid import uuid4

from dotenv import load_dotenv

from template_agent.config import MemorySettings
from template_agent.memory import HanaConversationStore


def main() -> int:
    """Create/validate storage, round-trip one row, and remove that row."""

    load_dotenv(override=False)
    context_id = f"template-smoke-{uuid4()}"
    store = HanaConversationStore(MemorySettings(enabled=True))
    initialized = False
    expected = [
        {"role": "user", "content": "memory smoke"},
        {"role": "assistant", "content": "memory ok"},
    ]
    try:
        store.ensure()
        initialized = True
        store.save(context_id, expected)
        actual = store.load(context_id)
        if actual != expected:
            raise RuntimeError(f"HANA round-trip mismatch: {actual!r}")
        store.clear(context_id)
        if store.load(context_id):
            raise RuntimeError("Temporary HANA context was not removed")
    finally:
        # A second parameterized delete is harmless if the first one succeeded.
        if initialized:
            store.clear(context_id)
        store.close()
    print("HANA memory smoke test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
