from __future__ import annotations

import json
import unittest

from ideation_agent.response_contract import (
    DEFAULT_CLARIFICATION_ERROR,
    REFINEMENT_ACTION_LABELS,
    extract_question_candidate,
    extract_structured_candidate,
    finalize_normalized_turn,
    looks_like_clarification_candidate,
    make_error_turn,
    strict_normalize_confirmed,
    strict_normalize_refinement,
    strict_normalize_turn,
    to_ui_payload,
)


class ResponseContractTests(unittest.TestCase):
    def test_valid_clarification_turn_is_normalized_and_increments_count(self) -> None:
        raw = json.dumps(
            {
                "kind": "clarification",
                "message": "I have enough to narrow this down.",
                "question": "Which audience should the gamified flow target first?",
                "suggested_answers": [
                    "New hires during onboarding",
                    "Managers driving adoption",
                    "Sales teams in enablement",
                ],
            }
        )

        normalized = strict_normalize_turn(raw, clarification_count=0)
        self.assertIsNotNone(normalized)

        turn = finalize_normalized_turn(normalized or {}, clarification_count=0)
        self.assertEqual(turn["kind"], "clarification")
        self.assertEqual(turn["clarification_count"], 1)
        self.assertEqual(len(turn["suggested_replies"]), 3)
        self.assertIn(
            "**Question:** Which audience should the gamified flow target first?",
            turn["display_text"],
        )
        self.assertIn(
            "*Choose one of the options below or write your own answer.*",
            turn["display_text"],
        )

    def test_double_barreled_clarification_turn_passes_validation(self) -> None:
        """Double-barreled questions are allowed through (the system prompt
        instructs the model to avoid them, but hard-rejecting causes user-facing
        errors that are worse than a mildly compound question).
        """
        raw = json.dumps(
            {
                "kind": "clarification",
                "message": "I need one more detail.",
                "question": "How many distinct material groups does Volvo procurement work with, and do the descriptions use standardized terminology or free-text variations?",
                "suggested_answers": [
                    "Fewer than 100, mostly standardized",
                    "100-500 with mixed terminology",
                    "More than 500 with many free-text variations",
                ],
            }
        )

        turn = strict_normalize_turn(raw, clarification_count=1)
        self.assertIsNotNone(turn)
        self.assertEqual(turn["kind"], "clarification")

    def test_valid_final_turn_is_rendered_to_human_readable_format(self) -> None:
        raw = json.dumps(
            {
                "kind": "final",
                "app_type": "Agent",
                "industry": "Learning and Development",
                "title": "AI Learning Quest",
                "problem": "Employees struggle to find the right enablement content and complete onboarding tasks consistently.",
                "solution": "Build a guided assistant that coaches users through learning missions, ranks next best enablement steps based on role and progress, and tracks onboarding task completion. Input: employee role, completed modules, available content catalog. Output: personalized learning path with progress tracking. Triggered on first login and weekly check-ins.",
            }
        )

        normalized = strict_normalize_turn(raw, clarification_count=2)
        self.assertIsNotNone(normalized)

        turn = finalize_normalized_turn(normalized or {}, clarification_count=2)
        self.assertEqual(turn["kind"], "final")
        self.assertEqual(turn["clarification_count"], 2)
        self.assertEqual(turn["industry"], "Learning and Development")
        self.assertEqual(turn["title"], "AI Learning Quest")
        self.assertEqual(turn["app_type"], "Agent")
        self.assertIn("## Recommended Solution", turn["display_text"])
        self.assertIn("### Overview", turn["display_text"])
        self.assertIn("- **Industry:** Learning and Development", turn["display_text"])
        self.assertIn("- **Type:** Agent", turn["display_text"])
        self.assertIn("- **Title:** *AI Learning Quest*", turn["display_text"])
        self.assertIn("### Solution", turn["display_text"])

        ui_payload = to_ui_payload(turn)
        self.assertEqual(ui_payload["kind"], "final")
        self.assertEqual(ui_payload["app_type"], "Agent")
        self.assertEqual(ui_payload["industry"], "Learning and Development")
        self.assertEqual(ui_payload["title"], "AI Learning Quest")

    def test_malformed_json_returns_none(self) -> None:
        self.assertIsNone(strict_normalize_turn("This is not valid JSON.", clarification_count=1))

    def test_invalid_clarification_missing_suggested_answers_returns_none(self) -> None:
        raw = json.dumps(
            {
                "kind": "clarification",
                "message": "One more detail will help.",
                "question": "Which region should this launch in first?",
                "suggested_answers": ["EMEA", "North America"],
            }
        )

        self.assertIsNone(strict_normalize_turn(raw, clarification_count=0))

    def test_final_without_app_type_defaults_to_agent(self) -> None:
        raw = json.dumps(
            {
                "kind": "final",
                "industry": "Manufacturing",
                "title": "PDF Classifier",
                "problem": "Manual classification is slow.",
                "solution": "Extract items from PDFs and match against taxonomy.",
            }
        )

        normalized = strict_normalize_turn(raw, clarification_count=0)
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["app_type"], "Agent")

    def test_clarification_after_three_is_still_returned(self) -> None:
        raw = json.dumps(
            {
                "kind": "clarification",
                "message": "One more detail would help.",
                "question": "What KPI should improve first?",
                "suggested_answers": [
                    "Course completion rate",
                    "Feature adoption rate",
                    "Time to first successful task",
                ],
            }
        )

        normalized = strict_normalize_turn(raw, clarification_count=3)
        self.assertIsNotNone(normalized)

        turn = finalize_normalized_turn(normalized or {}, clarification_count=3)
        self.assertEqual(turn["kind"], "clarification")
        self.assertEqual(turn["clarification_count"], 4)
        self.assertEqual(len(turn["suggested_replies"]), 3)

    def test_make_error_turn_exposes_error_ui_payload(self) -> None:
        turn = make_error_turn(DEFAULT_CLARIFICATION_ERROR, clarification_count=2)

        self.assertEqual(turn["kind"], "error")
        self.assertIn("**I need one more try.**", turn["display_text"])
        self.assertIn(DEFAULT_CLARIFICATION_ERROR, turn["display_text"])

        ui_payload = to_ui_payload(turn)
        self.assertEqual(ui_payload["kind"], "error")
        self.assertEqual(ui_payload["clarificationCount"], 2)
        self.assertEqual(ui_payload["messageType"], "structured_generation_error")

    def test_question_only_prose_is_detected_as_clarification_candidate(self) -> None:
        raw = """
Perfect. Now I understand you want full automation with direct S/4HANA write-back. One more clarification will help me tailor the solution:

How many quotations does your procurement team process per week?
""".strip()

        self.assertTrue(looks_like_clarification_candidate(raw))
        self.assertEqual(
            extract_question_candidate(raw),
            "How many quotations does your procurement team process per week?",
        )

    def test_invalid_json_clarification_is_detected_as_clarification_candidate(self) -> None:
        raw = json.dumps(
            {
                "kind": "clarification",
                "message": "Perfect-so the AI will suggest material groups and the clerk will review before posting.",
                "question": "How many material groups does your internal Excel contain, and how detailed are the descriptions?",
                "suggested_answers": [
                    "50-200 groups with short codes and brief descriptions",
                    "200-500 groups with detailed long-text descriptions",
                    "500+ groups with multi-language or highly technical descriptions",
                ],
            },
            indent=2,
        )

        self.assertTrue(looks_like_clarification_candidate(raw))
        self.assertEqual(
            extract_structured_candidate(raw)["question"],
            "How many material groups does your internal Excel contain, and how detailed are the descriptions?",
        )

    def test_final_formatted_text_is_not_detected_as_clarification_candidate(self) -> None:
        raw = """
Industry: Automotive Manufacturing
Title: Material Classification Copilot
Problem: Manual classification is slow and error-prone.
Selected AI Capabilities:
- [8] Document & Image Analysis
Solution: Use AI to classify quotation items and route exceptions for review.
""".strip()

        self.assertFalse(looks_like_clarification_candidate(raw))

    # --- Refinement turn tests ---

    def test_valid_refinement_turn_is_normalized(self) -> None:
        raw = json.dumps({"kind": "refinement", "message": "Got it, I'll focus more on the batch processing aspect."})

        normalized = strict_normalize_refinement(raw, clarification_count=5)
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["kind"], "refinement")
        self.assertEqual(normalized["display_text"], "Got it, I'll focus more on the batch processing aspect.")
        self.assertEqual(normalized["suggested_replies"], REFINEMENT_ACTION_LABELS)
        self.assertTrue(normalized["allow_freeform"])
        self.assertEqual(normalized["clarification_count"], 5)

    def test_valid_confirmed_turn_is_normalized(self) -> None:
        raw = json.dumps({"kind": "confirmed", "message": "Great! Your AI use case is ready for the builder."})

        normalized = strict_normalize_confirmed(raw, clarification_count=5)
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["kind"], "confirmed")
        self.assertEqual(normalized["display_text"], "Great! Your AI use case is ready for the builder.")
        self.assertEqual(normalized["suggested_replies"], [])
        self.assertFalse(normalized["allow_freeform"])

    def test_refinement_missing_message_returns_none(self) -> None:
        raw = json.dumps({"kind": "refinement"})
        self.assertIsNone(strict_normalize_refinement(raw))

        raw_empty = json.dumps({"kind": "refinement", "message": ""})
        self.assertIsNone(strict_normalize_refinement(raw_empty))

    def test_refinement_ui_payload_includes_action_labels(self) -> None:
        raw = json.dumps({"kind": "refinement", "message": "I understand the feedback."})
        normalized = strict_normalize_refinement(raw, clarification_count=4)
        self.assertIsNotNone(normalized)

        ui = to_ui_payload(normalized)
        self.assertEqual(ui["kind"], "refinement")
        self.assertEqual(ui["suggestedReplies"], REFINEMENT_ACTION_LABELS)
        self.assertTrue(ui["allowFreeform"])
        self.assertEqual(ui["clarificationCount"], 4)

    def test_confirmed_ui_payload_disables_freeform(self) -> None:
        raw = json.dumps({"kind": "confirmed", "message": "All done!"})
        normalized = strict_normalize_confirmed(raw, clarification_count=5)
        self.assertIsNotNone(normalized)

        ui = to_ui_payload(normalized)
        self.assertEqual(ui["kind"], "confirmed")
        self.assertFalse(ui["allowFreeform"])
        self.assertEqual(ui["suggestedReplies"], [])

    def test_refinement_not_accepted_without_refinement_mode(self) -> None:
        raw = json.dumps({"kind": "refinement", "message": "Some reply"})
        turn = strict_normalize_turn(raw, clarification_count=3, refinement_mode=False)
        self.assertIsNone(turn)

    def test_refinement_accepted_with_refinement_mode(self) -> None:
        raw = json.dumps({"kind": "refinement", "message": "Some reply"})
        turn = strict_normalize_turn(raw, clarification_count=3, refinement_mode=True)
        self.assertIsNotNone(turn)
        self.assertEqual(turn["kind"], "refinement")

    # --- Multi-question clarification tests ---

    def test_multi_question_clarification_is_normalized(self) -> None:
        """Multi-question format with 2 questions normalizes correctly."""
        raw = json.dumps({
            "kind": "clarification",
            "message": "I'd like to understand a few things before updating:",
            "questions": [
                {"question": "What database fits your current infrastructure?", "suggested_answers": ["SAP HANA Cloud", "PostgreSQL", "Keep Excel for now"]},
                {"question": "What does a correction look like in your workflow?", "suggested_answers": ["Full re-assignment", "Confidence override", "Category split"]},
            ],
            "problem_clear": True,
        })

        normalized = strict_normalize_turn(raw, clarification_count=5)
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["kind"], "clarification")
        self.assertEqual(len(normalized["questions"]), 2)
        self.assertEqual(normalized["questions"][0]["question"], "What database fits your current infrastructure?")
        self.assertEqual(normalized["questions"][1]["question"], "What does a correction look like in your workflow?")
        self.assertEqual(len(normalized["questions"][0]["suggested_replies"]), 3)
        self.assertEqual(len(normalized["questions"][1]["suggested_replies"]), 3)
        # Backward-compat fields: first question surfaces at top level
        self.assertEqual(normalized["question"], "What database fits your current infrastructure?")
        self.assertEqual(normalized["suggested_replies"], ["SAP HANA Cloud", "PostgreSQL", "Keep Excel for now"])
        # Multi-question bubble shows only lead-in (questions are in the interactive card)
        self.assertEqual(normalized["display_text"], "I'd like to understand a few things before updating:")
        self.assertNotIn("**Question 1:**", normalized["display_text"])

    def test_multi_question_single_item_treated_as_single_question(self) -> None:
        """A questions array with 1 item renders like the single-question format."""
        raw = json.dumps({
            "kind": "clarification",
            "message": "One more thing.",
            "questions": [
                {"question": "Which region should this launch in first?", "suggested_answers": ["EMEA", "North America", "APAC"]},
            ],
            "problem_clear": False,
        })

        normalized = strict_normalize_turn(raw, clarification_count=2)
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["question"], "Which region should this launch in first?")
        self.assertEqual(len(normalized["suggested_replies"]), 3)
        # Uses single-question display format
        self.assertIn("**Question:** Which region should this launch in first?", normalized["display_text"])
        self.assertNotIn("**Question 1:**", normalized["display_text"])

    def test_multi_question_rejects_empty_questions_array(self) -> None:
        """An empty questions array falls through to legacy format check (and fails)."""
        raw = json.dumps({
            "kind": "clarification",
            "message": "Let me ask...",
            "questions": [],
            "problem_clear": True,
        })
        self.assertIsNone(strict_normalize_turn(raw, clarification_count=0))

    def test_multi_question_rejects_wrong_suggested_count(self) -> None:
        """Each question must have exactly 3 suggested_answers."""
        raw = json.dumps({
            "kind": "clarification",
            "message": "A few questions:",
            "questions": [
                {"question": "What scale do you operate at?", "suggested_answers": ["Small", "Large"]},
                {"question": "What format are the inputs?", "suggested_answers": ["PDF", "CSV", "API"]},
            ],
            "problem_clear": True,
        })
        self.assertIsNone(strict_normalize_turn(raw, clarification_count=0))

    def test_multi_question_rejects_more_than_four(self) -> None:
        """Maximum 4 questions per turn."""
        raw = json.dumps({
            "kind": "clarification",
            "message": "Many questions:",
            "questions": [
                {"question": f"Question {i}?", "suggested_answers": ["A", "B", "C"]}
                for i in range(5)
            ],
            "problem_clear": True,
        })
        self.assertIsNone(strict_normalize_turn(raw, clarification_count=0))

    def test_multi_question_ui_payload_has_questions_array(self) -> None:
        """to_ui_payload exposes the questions array for the frontend."""
        raw = json.dumps({
            "kind": "clarification",
            "message": "Before updating:",
            "questions": [
                {"question": "Which system stores the data?", "suggested_answers": ["SAP S/4HANA", "Custom DB", "Excel files"]},
                {"question": "How often do corrections happen?", "suggested_answers": ["Daily", "Weekly", "Rarely"]},
            ],
            "problem_clear": True,
        })

        normalized = strict_normalize_turn(raw, clarification_count=3)
        self.assertIsNotNone(normalized)

        turn = finalize_normalized_turn(normalized, clarification_count=3)
        ui = to_ui_payload(turn)

        self.assertEqual(ui["kind"], "clarification")
        self.assertIn("questions", ui)
        self.assertEqual(len(ui["questions"]), 2)
        self.assertEqual(ui["questions"][0]["question"], "Which system stores the data?")
        self.assertEqual(ui["questions"][0]["suggestedReplies"], ["SAP S/4HANA", "Custom DB", "Excel files"])
        self.assertEqual(ui["questions"][1]["question"], "How often do corrections happen?")
        self.assertEqual(ui["questions"][1]["suggestedReplies"], ["Daily", "Weekly", "Rarely"])
        # Backward compat: top-level suggestedReplies is first question's
        self.assertEqual(ui["suggestedReplies"], ["SAP S/4HANA", "Custom DB", "Excel files"])

    def test_legacy_single_question_still_has_questions_array(self) -> None:
        """Old format (question + suggested_answers at top level) also gets a questions array."""
        raw = json.dumps({
            "kind": "clarification",
            "message": "One detail needed.",
            "question": "What scale are we talking about?",
            "suggested_answers": ["10-50 per week", "50-200 per week", "200+ per week"],
        })

        normalized = strict_normalize_turn(raw, clarification_count=1)
        self.assertIsNotNone(normalized)
        self.assertIn("questions", normalized)
        self.assertEqual(len(normalized["questions"]), 1)
        self.assertEqual(normalized["questions"][0]["question"], "What scale are we talking about?")

        ui = to_ui_payload(finalize_normalized_turn(normalized, clarification_count=1))
        self.assertIn("questions", ui)
        self.assertEqual(len(ui["questions"]), 1)

    def test_multi_question_double_barreled_passes(self) -> None:
        """Double-barreled question inside multi-question array is now allowed."""
        raw = json.dumps({
            "kind": "clarification",
            "message": "A few things:",
            "questions": [
                {"question": "How many users do you have, and how often do they submit requests?", "suggested_answers": ["Small scale", "Medium scale", "Large scale"]},
                {"question": "What format are the inputs?", "suggested_answers": ["PDF", "CSV", "API"]},
            ],
            "problem_clear": True,
        })
        result = strict_normalize_turn(raw, clarification_count=0)
        self.assertIsNotNone(result)
        self.assertEqual(result["kind"], "clarification")
        self.assertEqual(len(result["questions"]), 2)


if __name__ == "__main__":
    unittest.main()
