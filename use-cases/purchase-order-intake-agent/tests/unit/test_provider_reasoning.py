"""Unit tests: reasoning blocks are dropped from model answers (SAP AI Core rejects replayed reasoning).

Run from the api/ folder:  ../.venv/bin/python -m unittest discover -s ../tests -t ..
"""

import unittest

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.agent_core.providers import _without_reasoning


def result_with(content) -> ChatResult:
    """Wrap one AI message in a ChatResult as the model returns it."""
    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


class WithoutReasoning(unittest.TestCase):
    """Only reasoning blocks are removed; everything the agent needs stays."""

    def test_reasoning_dropped_other_blocks_kept(self):
        content = [
            {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "gAAAA..."},
            {"type": "function_call", "name": "search_materials", "call_id": "call_1", "arguments": "{}"},
            {"type": "text", "text": "Done."},
        ]
        message = _without_reasoning(result_with(content)).generations[0].message
        self.assertEqual([block["type"] for block in message.content], ["function_call", "text"])

    def test_plain_text_unchanged(self):
        message = _without_reasoning(result_with("Done.")).generations[0].message
        self.assertEqual(message.content, "Done.")


if __name__ == "__main__":
    unittest.main()
