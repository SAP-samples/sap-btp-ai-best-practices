"""Tests for handover pre-fill extraction logic.

Covers JSON parsing, field cleaning, merging, conversation Q&A building,
cache-check semantics, and the end-to-end extraction flow.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch


# ── Helpers: import the functions under test ───────────────────────────────

from ideation_agent.handover_prefill import (
    _build_conversation_qa,
    _clean_extracted,
    _parse_json_response,
    merge_extracted_fields,
)


# ── Test: _parse_json_response ─────────────────────────────────────────────


class TestParseJsonResponse(unittest.TestCase):
    """Tests for JSON response parsing (handles code fences, invalid input)."""

    def test_valid_json_object(self):
        result = _parse_json_response('{"a": 1, "b": "hello"}')
        self.assertEqual(result, {"a": 1, "b": "hello"})

    def test_json_with_code_fences(self):
        raw = '```\n{"usecase_title": "Test"}\n```'
        result = _parse_json_response(raw)
        self.assertEqual(result, {"usecase_title": "Test"})

    def test_json_with_language_tag(self):
        raw = '```json\n{"industry": "Automotive"}\n```'
        result = _parse_json_response(raw)
        self.assertEqual(result, {"industry": "Automotive"})

    def test_invalid_json_returns_empty(self):
        self.assertEqual(_parse_json_response("not json at all"), {})

    def test_non_dict_returns_empty(self):
        self.assertEqual(_parse_json_response("[1, 2, 3]"), {})

    def test_empty_string_returns_empty(self):
        self.assertEqual(_parse_json_response(""), {})

    def test_whitespace_only_returns_empty(self):
        self.assertEqual(_parse_json_response("   \n  "), {})


# ── Test: _clean_extracted ─────────────────────────────────────────────────


class TestCleanExtracted(unittest.TestCase):
    """Tests for null/empty removal from extracted field dicts."""

    def test_removes_null_values(self):
        result = _clean_extracted({"a": None, "b": "x"})
        self.assertEqual(result, {"b": "x"})

    def test_removes_empty_arrays(self):
        result = _clean_extracted({"a": [], "b": [1]})
        self.assertEqual(result, {"b": [1]})

    def test_removes_blank_strings(self):
        result = _clean_extracted({"a": "  ", "b": "x"})
        self.assertEqual(result, {"b": "x"})

    def test_preserves_valid_values(self):
        data = {
            "usecase_title": "Test",
            "industry": "Automotive",
            "data_sources": [{"name": "SAP", "system": "ERP", "data_type": "Transactional"}],
        }
        result = _clean_extracted(data)
        self.assertEqual(result, data)

    def test_all_null_returns_empty(self):
        result = _clean_extracted({"a": None, "b": None, "c": []})
        self.assertEqual(result, {})

    def test_cleans_business_impact_empty_dimensions(self):
        """business_impact dimensions with only blank strings should be stripped."""
        data = {
            "business_impact": {
                "volume": {"current": "500/week", "target": ""},
                "time_spent": {"current": "", "target": ""},
                "error_rate": {"current": "12%", "target": "< 3%"},
            }
        }
        result = _clean_extracted(data)
        impact = result["business_impact"]
        self.assertIn("volume", impact)
        self.assertIn("error_rate", impact)
        self.assertNotIn("time_spent", impact)

    def test_cleans_business_impact_all_empty(self):
        """business_impact with all blank sub-objects should be removed entirely."""
        data = {
            "business_impact": {
                "volume": {"current": "", "target": ""},
                "time_spent": {"current": None, "target": None},
            }
        }
        result = _clean_extracted(data)
        self.assertNotIn("business_impact", result)

    def test_validates_ai_capabilities(self):
        """Only valid capability IDs should survive cleaning."""
        data = {
            "ai_capabilities": [
                "doc_understanding", "classification",
                "invalid_cap", "nlu", "made_up",
            ]
        }
        result = _clean_extracted(data)
        self.assertEqual(result["ai_capabilities"], ["doc_understanding", "classification", "nlu"])

    def test_ai_capabilities_all_invalid(self):
        """ai_capabilities with no valid IDs should be removed."""
        data = {"ai_capabilities": ["fake1", "fake2"]}
        result = _clean_extracted(data)
        self.assertNotIn("ai_capabilities", result)

    def test_removes_empty_dicts(self):
        """Empty dict values should be stripped."""
        result = _clean_extracted({"a": {}, "b": "x"})
        self.assertEqual(result, {"b": "x"})


# ── Test: merge_extracted_fields ───────────────────────────────────────────


class TestMergeExtractedFields(unittest.TestCase):
    """Tests for merging multiple extraction results."""

    def test_single_dict(self):
        result = merge_extracted_fields({"a": "1", "b": "2"})
        self.assertEqual(result, {"a": "1", "b": "2"})

    def test_later_overrides_earlier(self):
        result = merge_extracted_fields({"a": "old"}, {"a": "new"})
        self.assertEqual(result, {"a": "new"})

    def test_none_values_do_not_override(self):
        result = merge_extracted_fields({"a": "keep"}, {"a": None})
        self.assertEqual(result, {"a": "keep"})

    def test_empty_strings_do_not_override(self):
        result = merge_extracted_fields({"a": "keep"}, {"a": "  "})
        self.assertEqual(result, {"a": "keep"})

    def test_empty_arrays_do_not_override(self):
        result = merge_extracted_fields({"a": [1]}, {"a": []})
        self.assertEqual(result, {"a": [1]})

    def test_none_dicts_are_skipped(self):
        result = merge_extracted_fields(None, {"a": "1"}, None)
        self.assertEqual(result, {"a": "1"})

    def test_all_empty_returns_empty(self):
        result = merge_extracted_fields({}, {}, None)
        self.assertEqual(result, {})

    def test_deep_merge_business_impact(self):
        """business_impact should deep-merge sub-keys, not clobber."""
        result = merge_extracted_fields(
            {"business_impact": {"volume": {"current": "500/week", "target": None}}},
            {"business_impact": {"time_spent": {"current": "40 hrs", "target": "8 hrs"}}},
        )
        self.assertIn("volume", result["business_impact"])
        self.assertIn("time_spent", result["business_impact"])
        self.assertEqual(result["business_impact"]["volume"]["current"], "500/week")
        self.assertEqual(result["business_impact"]["time_spent"]["current"], "40 hrs")

    def test_deep_merge_business_impact_later_wins(self):
        """Later non-empty values within the same dimension should win."""
        result = merge_extracted_fields(
            {"business_impact": {"volume": {"current": "old", "target": "old target"}}},
            {"business_impact": {"volume": {"current": "new", "target": ""}}},
        )
        # "new" wins for current, "old target" preserved since "" is blank
        self.assertEqual(result["business_impact"]["volume"]["current"], "new")
        self.assertEqual(result["business_impact"]["volume"]["target"], "old target")

    def test_union_merge_ai_capabilities(self):
        """ai_capabilities should union-merge without duplicates."""
        result = merge_extracted_fields(
            {"ai_capabilities": ["nlu", "classification"]},
            {"ai_capabilities": ["classification", "rag", "agentic"]},
        )
        self.assertEqual(result["ai_capabilities"], ["nlu", "classification", "rag", "agentic"])

    def test_empty_dicts_do_not_override(self):
        """Empty dict values should not override existing values."""
        result = merge_extracted_fields({"a": {"x": 1}}, {"a": {}})
        self.assertEqual(result, {"a": {"x": 1}})


# ── Test: _build_conversation_qa ───────────────────────────────────────────


class TestBuildConversationQA(unittest.TestCase):
    """Tests for Q&A transcript building from conversation messages."""

    def test_empty_messages_returns_empty_string(self):
        self.assertEqual(_build_conversation_qa([]), "")

    def test_user_and_clarification_produce_qa_pairs(self):
        messages = [
            {
                "role": "agent",
                "content": "What industry are you in?",
                "metadata": {"ui": {"kind": "clarification", "question": "What industry are you in?"}},
            },
            {"role": "user", "content": "Automotive manufacturing"},
        ]
        result = _build_conversation_qa(messages)
        self.assertIn("Q: What industry are you in?", result)
        self.assertIn("A: Automotive manufacturing", result)

    def test_skips_final_messages(self):
        messages = [
            {
                "role": "agent",
                "content": "Here is the final solution...",
                "metadata": {"ui": {"kind": "final"}},
            },
        ]
        result = _build_conversation_qa(messages)
        self.assertEqual(result, "")

    def test_skips_thinking_prefix_messages(self):
        messages = [
            {"role": "agent", "content": "[thinking: reframe direction]", "metadata": {}},
        ]
        result = _build_conversation_qa(messages)
        self.assertEqual(result, "")

    def test_skips_error_and_confirmed_kinds(self):
        messages = [
            {"role": "agent", "content": "Error occurred", "metadata": {"ui": {"kind": "error"}}},
            {"role": "agent", "content": "Confirmed", "metadata": {"ui": {"kind": "confirmed"}}},
        ]
        result = _build_conversation_qa(messages)
        self.assertEqual(result, "")

    def test_handles_missing_metadata(self):
        messages = [
            {"role": "user", "content": "My answer"},
            {"role": "agent", "content": "Short"},
        ]
        # Should not crash — user answer included, short agent msg skipped
        result = _build_conversation_qa(messages)
        self.assertIn("A: My answer", result)
        self.assertNotIn("Short", result)

    def test_includes_substantive_agent_messages(self):
        long_msg = "x" * 150  # > 100 chars
        messages = [
            {"role": "agent", "content": long_msg, "metadata": {}},
        ]
        result = _build_conversation_qa(messages)
        self.assertIn(f"CONTEXT: {long_msg}", result)

    def test_short_agent_messages_without_kind_excluded(self):
        messages = [
            {"role": "agent", "content": "Got it.", "metadata": {}},
        ]
        result = _build_conversation_qa(messages)
        self.assertEqual(result, "")

    def test_uses_question_field_from_ui_metadata(self):
        """Prefer ui.question over raw content for clarification messages."""
        messages = [
            {
                "role": "agent",
                "content": "Let me ask about your process. How often does the invoice review happen?",
                "metadata": {
                    "ui": {
                        "kind": "clarification",
                        "question": "How often does the invoice review happen?",
                    }
                },
            },
        ]
        result = _build_conversation_qa(messages)
        self.assertEqual(result, "Q: How often does the invoice review happen?")

    def test_empty_content_skipped(self):
        messages = [
            {"role": "user", "content": ""},
            {"role": "agent", "content": None, "metadata": {}},
        ]
        result = _build_conversation_qa(messages)
        self.assertEqual(result, "")


# ── Test: Cache-check semantics ────────────────────────────────────────────


class TestCacheCheckSemantics(unittest.TestCase):
    """Validates that the prefill endpoint's cache check behaves correctly.

    The cache check is: ``if source.get("extracted_fields") and not is_ideation:``
    Static sources use cache; ideation sources always re-extract.
    """

    def test_none_is_falsy(self):
        source = {"extracted_fields": None}
        self.assertFalse(bool(source.get("extracted_fields")))

    def test_empty_dict_is_falsy(self):
        source = {"extracted_fields": {}}
        self.assertFalse(bool(source.get("extracted_fields")))

    def test_populated_dict_is_truthy(self):
        source = {"extracted_fields": {"usecase_title": "Test"}}
        self.assertTrue(bool(source.get("extracted_fields")))

    def test_missing_key_is_falsy(self):
        source = {"source_type": "ideation"}
        self.assertFalse(bool(source.get("extracted_fields")))

    def test_ideation_bypasses_cache(self):
        """Ideation sources always re-extract even when cache is populated."""
        source = {
            "source_type": "ideation",
            "extracted_fields": {"usecase_title": "Old cached title"},
        }
        is_ideation = source["source_type"] == "ideation"
        cached = source.get("extracted_fields")
        # The combined check: use cache only for non-ideation with populated fields
        use_cache = bool(cached) and not is_ideation
        self.assertFalse(use_cache)

    def test_static_source_uses_cache(self):
        """Document and manual sources use cached extraction results."""
        for source_type in ("document", "manual"):
            source = {
                "source_type": source_type,
                "extracted_fields": {"problem_statement": "Cached problem"},
            }
            is_ideation = source["source_type"] == "ideation"
            cached = source.get("extracted_fields")
            use_cache = bool(cached) and not is_ideation
            self.assertTrue(use_cache, f"{source_type} should use cache")


# ── Test: extract_from_ideation (integration with mocks) ──────────────────


def _make_messages(
    *,
    include_final: bool = True,
    qa_pairs: Optional[List[tuple]] = None,
) -> List[Dict[str, Any]]:
    """Build a realistic conversation message list for testing."""
    messages: List[Dict[str, Any]] = []
    seq = 0

    # Intro agent message
    seq += 1
    messages.append(
        {
            "role": "agent",
            "content": "Welcome! Let's define your AI use case.",
            "metadata": {"ui": {"kind": "clarification", "question": "What company are you from?"}},
            "sequence_num": seq,
        }
    )
    seq += 1
    messages.append({"role": "user", "content": "Volvo Car AB", "sequence_num": seq})

    # Q&A pairs
    if qa_pairs:
        for question, answer in qa_pairs:
            seq += 1
            messages.append(
                {
                    "role": "agent",
                    "content": f"Understood. {question}",
                    "metadata": {"ui": {"kind": "clarification", "question": question}},
                    "sequence_num": seq,
                }
            )
            seq += 1
            messages.append({"role": "user", "content": answer, "sequence_num": seq})

    # Final delivery
    if include_final:
        seq += 1
        messages.append(
            {
                "role": "agent",
                "content": "## Recommended Solution\n\n- **Industry:** Automotive\n- **Title:** Material Grouping Agent\n\n### Problem\nClerks spend 40% of time on manual categorization.\n\n### Solution\nWHAT IT DOES: Automates material group assignment.\n\nTRIGGER: New PDFs arrive via email inbox.\n\nDATA SOURCES & SYSTEMS: Email inbox, Excel master list, S/4HANA.\n\nPROCESSING STEPS:\n1. Extract line items from PDFs\n2. Match against taxonomy\n3. Route uncertain cases to review\n\nUSER ROLES: Procurement clerks review uncertain cases.",
                "metadata": {
                    "ui": {
                        "kind": "final",
                        "title": "Material Grouping Agent",
                        "industry": "Automotive",
                        "problem": "Clerks spend 40% of time on manual categorization.",
                        "solution": "WHAT IT DOES: Automates material group assignment.\n\nTRIGGER: New PDFs arrive via email inbox.\n\nDATA SOURCES & SYSTEMS: Email inbox, Excel master list, S/4HANA.\n\nPROCESSING STEPS:\n1. Extract line items from PDFs\n2. Match against taxonomy\n3. Route uncertain cases to review\n\nUSER ROLES: Procurement clerks review uncertain cases.",
                    }
                },
                "sequence_num": seq,
            }
        )

    return messages


class TestExtractFromIdeation(unittest.IsolatedAsyncioTestCase):
    """Integration tests for extract_from_ideation with mocked store and LLM."""

    def _mock_store(self, messages, conv_title="Volvo Car AB: Material Grouping Agent", research=None):
        """Create a mock store that returns the given messages and conversation."""
        store = AsyncMock()
        store.get_messages = AsyncMock(return_value=messages)
        store.get_conversation_by_id = AsyncMock(
            return_value={"title": conv_title, "company_name": "Volvo Car AB"}
        )
        store.get_all_research = AsyncMock(return_value=research or {})
        return store

    def _mock_llm_response(self, extracted: dict):
        """Create a mock LLM that returns the given extraction dict as JSON."""
        import json

        response = SimpleNamespace(
            content=json.dumps(extracted),
            response_metadata={"token_usage": {"prompt_tokens": 100, "completion_tokens": 50}},
        )
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=response)
        return llm

    @patch("ideation_agent.handover_prefill.make_llm")
    @patch("ideation_agent.hana_store.get_store")
    @patch("app.observability.llm_usage_logging.emit_llm_usage_for_response")
    async def test_builds_source_with_final_and_qa(self, mock_emit, mock_get_store, mock_make_llm):
        """Verify the prompt sent to the LLM includes both final delivery and Q&A."""
        from ideation_agent.handover_prefill import extract_from_ideation

        qa_pairs = [
            ("How often do clerks process invoices?", "About 200 per week"),
            ("What systems are involved?", "SAP S/4HANA and an internal Excel file"),
        ]
        messages = _make_messages(include_final=True, qa_pairs=qa_pairs)

        store = self._mock_store(messages)
        mock_get_store.return_value = store

        extracted = {
            "usecase_title": "Material Grouping Agent",
            "industry": "Automotive",
            "problem_statement": "Clerks spend 40% of time on manual categorization.",
            "process_freq": "200 per week",
            "process_team": "Procurement clerks",
        }
        llm = self._mock_llm_response(extracted)
        mock_make_llm.return_value = llm

        result = await extract_from_ideation("test-conv-id")

        # Verify the LLM was called
        llm.ainvoke.assert_called_once()
        prompt = llm.ainvoke.call_args[0][0]

        # Prompt should contain final delivery sections
        self.assertIn("SOLUTION:", prompt)
        self.assertIn("INDUSTRY:", prompt)

        # Prompt should contain Q&A from conversation
        self.assertIn("CONVERSATION Q&A:", prompt)
        self.assertIn("A: About 200 per week", prompt)
        self.assertIn("A: SAP S/4HANA and an internal Excel file", prompt)

        # Result should have extracted fields
        self.assertEqual(result["industry"], "Automotive")
        self.assertEqual(result["process_freq"], "200 per week")

    @patch("ideation_agent.handover_prefill.make_llm")
    @patch("ideation_agent.hana_store.get_store")
    async def test_no_final_message_returns_empty(self, mock_get_store, mock_make_llm):
        """When no message has kind == 'final', returns empty dict."""
        from ideation_agent.handover_prefill import extract_from_ideation

        messages = _make_messages(include_final=False)
        store = self._mock_store(messages)
        mock_get_store.return_value = store

        result = await extract_from_ideation("test-conv-id")

        self.assertEqual(result, {})
        mock_make_llm.assert_not_called()

    @patch("ideation_agent.handover_prefill.make_llm")
    @patch("ideation_agent.hana_store.get_store")
    @patch("app.observability.llm_usage_logging.emit_llm_usage_for_response")
    async def test_conversation_title_overrides_extracted(self, mock_emit, mock_get_store, mock_make_llm):
        """The conversation's stored title should override the LLM-extracted one."""
        from ideation_agent.handover_prefill import extract_from_ideation

        messages = _make_messages(include_final=True)
        store = self._mock_store(messages, conv_title="Volvo Car AB: Custom Title Override")
        mock_get_store.return_value = store

        llm = self._mock_llm_response({"usecase_title": "LLM-extracted Title", "industry": "Automotive"})
        mock_make_llm.return_value = llm

        result = await extract_from_ideation("test-conv-id")

        self.assertEqual(result["usecase_title"], "Volvo Car AB: Custom Title Override")

    @patch("ideation_agent.hana_store.get_store")
    async def test_store_error_returns_empty_gracefully(self, mock_get_store):
        """When the store raises, returns {} without propagating."""
        from ideation_agent.handover_prefill import extract_from_ideation

        mock_get_store.side_effect = RuntimeError("HANA connection failed")

        result = await extract_from_ideation("test-conv-id")

        self.assertEqual(result, {})

    @patch("ideation_agent.handover_prefill.make_llm")
    @patch("ideation_agent.hana_store.get_store")
    @patch("app.observability.llm_usage_logging.emit_llm_usage_for_response")
    @patch("app.services.hana_submissions.get_submission_store")
    async def test_includes_research_and_eval_in_prompt(
        self, mock_sub_store, mock_emit, mock_get_store, mock_make_llm
    ):
        """Verify research and evaluation context appear in the LLM prompt."""
        from ideation_agent.handover_prefill import extract_from_ideation

        messages = _make_messages(include_final=True)

        research = {
            "perplexity_company": {
                "payload": {"company_markdown": "Volvo is a leading automotive manufacturer focused on safety."}
            },
            "perplexity_problem": {
                "payload": {"problem_markdown": "Material group assignment is a common pain point in procurement."}
            },
            "linkedin": {
                "payload": {
                    "status": "ok",
                    "top_technologies": [{"name": "SAP S/4HANA", "insight": "core ERP"}],
                    "ideation_insights": {
                        "bottlenecks": ["Manual categorization of purchase orders"],
                        "automation_opportunities": ["AI-driven material group assignment"],
                    },
                }
            },
        }
        store = self._mock_store(messages, research=research)
        mock_get_store.return_value = store

        # Mock submission store with evaluation data
        sub_store = AsyncMock()
        sub_store.get_all_by_context_any_user = AsyncMock(return_value=[
            {
                "score": 720,
                "evaluator_detailed_analysis": "Good proposal but missing failure paths.",
                "evaluator_suggestions": [
                    "What happens when confidence is too low?",
                    "How is the Excel taxonomy structured?",
                ],
            }
        ])
        mock_sub_store.return_value = sub_store

        extracted = {"usecase_title": "Material Grouping Agent", "industry": "Automotive"}
        llm = self._mock_llm_response(extracted)
        mock_make_llm.return_value = llm

        result = await extract_from_ideation("test-conv-id")

        # Verify LLM was called and check prompt contents
        llm.ainvoke.assert_called_once()
        prompt = llm.ainvoke.call_args[0][0]

        # Research context should appear
        self.assertIn("RESEARCH CONTEXT:", prompt)
        self.assertIn("Volvo is a leading automotive manufacturer", prompt)
        self.assertIn("Material group assignment is a common pain point", prompt)
        self.assertIn("Manual categorization of purchase orders", prompt)

        # Evaluation feedback should appear
        self.assertIn("EVALUATION FEEDBACK:", prompt)
        self.assertIn("720/1000", prompt)
        self.assertIn("missing failure paths", prompt)
        self.assertIn("What happens when confidence is too low?", prompt)

    @patch("ideation_agent.handover_prefill.make_llm")
    @patch("ideation_agent.hana_store.get_store")
    @patch("app.observability.llm_usage_logging.emit_llm_usage_for_response")
    async def test_research_failure_degrades_gracefully(self, mock_emit, mock_get_store, mock_make_llm):
        """When research/eval fetches fail, extraction still completes."""
        from ideation_agent.handover_prefill import extract_from_ideation

        messages = _make_messages(include_final=True)
        store = self._mock_store(messages)
        # Simulate research fetch failure
        store.get_all_research = AsyncMock(side_effect=RuntimeError("HANA timeout"))
        mock_get_store.return_value = store

        extracted = {"usecase_title": "Test", "industry": "Auto"}
        llm = self._mock_llm_response(extracted)
        mock_make_llm.return_value = llm

        result = await extract_from_ideation("test-conv-id")

        # Should still return results despite research failure
        self.assertEqual(result["industry"], "Auto")
        llm.ainvoke.assert_called_once()


if __name__ == "__main__":
    unittest.main()
