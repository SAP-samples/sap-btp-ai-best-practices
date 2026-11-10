"""
Unit tests for the rules-chat session helper (offline, fake runtime).

Verifies the per-session behaviour that HANA conversation memory does NOT cover:
uploaded rule documents accumulate across turns, are bound to the rule-source
allowlist for the duration of one invocation, are named back to the agent in the
prompt, and the allowlist is always reset afterwards. Also checks that an
untrusted session id is rejected before any filesystem work.

Run from the repo root:
    PYTHONPATH=api python -m unittest discover -s tests/unit -q
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

_API_DIR = Path(__file__).resolve().parents[2] / "api"
if str(_API_DIR) not in sys.path:
    sys.path.insert(0, str(_API_DIR))

from app.deduction_agent import chat_session, rule_sources  # noqa: E402


class _FakeRuntime:
    """Record each ainvoke call and return an AgentResult-like object."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def ainvoke(self, text: str, context_id: str, **_kw):
        self.calls.append({"text": text, "context_id": context_id})
        return SimpleNamespace(output_text=f"ok:{context_id}", output_parsed=None, messages=[])


class ChatSession(unittest.TestCase):
    def setUp(self) -> None:
        # Isolate module-global session state and clean any temp dirs between tests.
        self._reset_sessions()
        self.addCleanup(self._reset_sessions)
        # The rule-source allowlist is a ContextVar; clear any leakage from other
        # tests so the "reset after turn" assertions compare against a clean None.
        rule_sources._BOUND_SOURCES.set(None)

    @staticmethod
    def _reset_sessions() -> None:
        for state in chat_session._sessions.values():
            shutil.rmtree(state["dir"], ignore_errors=True)
        chat_session._sessions.clear()

    def test_uploads_accumulate_across_turns_and_allowlist_resets(self) -> None:
        rt = _FakeRuntime()

        # Turn 1: attach a PDF (byte content is irrelevant; bind only checks the
        # suffix + that the file exists, it does not parse here).
        out1 = asyncio.run(rt_turn(rt, "sess-1", "here are the rules", [("rulesA.pdf", b"%PDF-1.4 fake")]))
        self.assertEqual(out1["reply"], "ok:sess-1")
        self.assertEqual(out1["session_id"], "sess-1")
        self.assertEqual(rt.calls[-1]["context_id"], "sess-1")  # memory keyed by session
        self.assertIn("rulesA.pdf", rt.calls[-1]["text"])       # attachment named to agent
        self.assertIsNone(rule_sources._BOUND_SOURCES.get())    # allowlist reset after turn

        # Turn 2: attach a second file -> both must now be available.
        out2 = asyncio.run(rt_turn(rt, "sess-1", "and these", [("rulesB.xlsx", b"PK fake xlsx")]))
        self.assertIn("rulesA.pdf", rt.calls[-1]["text"])
        self.assertIn("rulesB.xlsx", rt.calls[-1]["text"])
        self.assertEqual(out2["session_id"], "sess-1")

        # Turn 3: no new upload -> prior files persist for the session.
        asyncio.run(rt_turn(rt, "sess-1", "what did I upload?", []))
        self.assertIn("rulesA.pdf", rt.calls[-1]["text"])
        self.assertIn("rulesB.xlsx", rt.calls[-1]["text"])
        self.assertIsNone(rule_sources._BOUND_SOURCES.get())

    def test_pure_text_turn_binds_nothing(self) -> None:
        rt = _FakeRuntime()
        asyncio.run(rt_turn(rt, "sess-2", "show the rules for contoso", []))
        # No attachment note appended, and no allowlist bound.
        self.assertEqual(rt.calls[-1]["text"], "show the rules for contoso")
        self.assertIsNone(rule_sources._BOUND_SOURCES.get())

    def test_unsafe_session_id_rejected(self) -> None:
        rt = _FakeRuntime()
        with self.assertRaises(ValueError):
            asyncio.run(rt_turn(rt, "../escape", "hi", []))


def rt_turn(runtime, session_id, message, uploads):
    """Thin wrapper so each asyncio.run drives one run_chat_turn coroutine."""
    return chat_session.run_chat_turn(runtime, session_id, message, uploads)


if __name__ == "__main__":
    unittest.main()
