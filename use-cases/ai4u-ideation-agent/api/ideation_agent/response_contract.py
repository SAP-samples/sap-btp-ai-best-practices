from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

MAX_SUGGESTED_REPLIES = 3
MAX_MULTI_QUESTIONS = 4
MAX_CLARIFICATION_QUESTIONS = 9
MIN_CLARIFICATION_QUESTIONS = 5
DEFAULT_CLARIFICATION_ERROR = "I could not reliably structure the next clarification step. Please resend your last answer."
DEFAULT_FINAL_ERROR = (
    "I could not generate a reliable recommendation right now. Please try again."
)
_DOUBLE_BARRELED_PATTERNS = (
    "how many",
    "how often",
    "do the",
    "does the",
    "are the",
    "is the",
)


def _strip_code_fences(text: str) -> str:
    stripped = str(text or "").strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _normalize_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    normalized: List[str] = []
    for item in values:
        text = str(item or "").strip()
        if text:
            normalized.append(text)
    return normalized


def _is_single_focus_question(question: str) -> bool:
    text = str(question or "").strip()
    if not text:
        return False
    if text.count("?") != 1:
        return False

    lowered = text.lower()
    if (
        ", and " not in lowered
        and " and do " not in lowered
        and " and are " not in lowered
        and " and is " not in lowered
    ):
        return True

    matched = sum(1 for pattern in _DOUBLE_BARRELED_PATTERNS if pattern in lowered)
    return matched <= 1


def extract_question_candidate(raw_text: str) -> Optional[str]:
    lines = [line.strip() for line in str(raw_text or "").splitlines()]
    non_empty_lines = [line for line in lines if line]
    for line in reversed(non_empty_lines):
        if line.endswith("?"):
            return line
    return None


def extract_structured_candidate(raw_text: str) -> Dict[str, str]:
    parsed_text = _strip_code_fences(raw_text)
    try:
        payload = json.loads(parsed_text)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    # Support both old "question" field and new "questions" array
    question = str(payload.get("question", "")).strip()
    if not question and isinstance(payload.get("questions"), list) and payload["questions"]:
        first_q = payload["questions"][0]
        if isinstance(first_q, dict):
            question = str(first_q.get("question", "")).strip()
    return {
        "kind": str(payload.get("kind", "")).strip().lower(),
        "message": str(payload.get("message", "")).strip(),
        "question": question,
    }


def looks_like_clarification_candidate(raw_text: str) -> bool:
    text = str(raw_text or "").strip()
    if not text:
        return False

    structured_candidate = extract_structured_candidate(text)
    if structured_candidate.get("kind") == "clarification" and structured_candidate.get(
        "question"
    ):
        return True

    question = extract_question_candidate(text)
    if not question:
        return False

    lowered_lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    final_markers = (
        "industry:",
        "title:",
        "problem:",
        "selected ai capabilities:",
        "solution:",
    )
    return not any(
        line.startswith(marker) for line in lowered_lines for marker in final_markers
    )


def _load_payload(raw_text: str) -> Optional[Dict[str, Any]]:
    parsed_text = _strip_code_fences(raw_text)
    try:
        payload = json.loads(parsed_text)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _render_final_display_text(
    industry: str,
    title: str,
    problem: str,
    solution: str,
    app_type: str = "Agent",
    company_context: str = "",
    sources: Optional[List[str]] = None,
) -> str:
    text = (
        "## Recommended Solution\n\n"
        "### Overview\n\n"
        f"- **Industry:** {industry}\n"
        f"- **Type:** {app_type}\n"
        f"- **Title:** *{title}*\n\n"
        "### Problem\n\n"
        f"{problem}\n\n"
        "### Solution\n\n"
        f"{solution}"
    )
    if company_context:
        text += f"\n\n### Company Context\n\n{company_context}"
    if sources:
        source_lines = "\n".join(f"- {src}" for src in sources)
        text += f"\n\n### Sources\n\n{source_lines}"
    return text


def _render_clarification_display_text(message: str, question: str) -> str:
    sections: List[str] = []
    if message:
        sections.append(message)
    sections.append(f"**Question:** {question}")
    sections.append("*Choose one of the options below or write your own answer.*")
    return "\n\n".join(section for section in sections if section).strip()


def _render_multi_clarification_display_text(message: str, questions: List[Dict[str, Any]]) -> str:
    """For multi-question, bubble only shows the lead-in message.

    Questions are rendered interactively in the card below the bubble,
    so duplicating them here would create visual noise.
    """
    if message:
        return message
    return "I have a few questions before proceeding:"


def _render_error_display_text(message: str) -> str:
    return f"**I need one more try.**\n\n*{message}*"


def _normalize_questions_array(raw_questions: Any) -> Optional[List[Dict[str, Any]]]:
    """Validate and normalize a questions array from multi-question clarification.

    Returns None if validation fails.
    """
    if not isinstance(raw_questions, list) or not raw_questions:
        return None
    if len(raw_questions) > MAX_MULTI_QUESTIONS:
        return None

    normalized: List[Dict[str, Any]] = []
    for item in raw_questions:
        if not isinstance(item, dict):
            return None
        question = str(item.get("question", "")).strip()
        suggested_replies = _normalize_list(item.get("suggested_answers"))
        if not question or len(suggested_replies) != MAX_SUGGESTED_REPLIES:
            return None
        normalized.append({"question": question, "suggested_replies": suggested_replies})
    return normalized


def strict_normalize_clarification(
    raw_text: str,
    clarification_count: int = 0,
) -> Optional[Dict[str, Any]]:
    payload = _load_payload(raw_text)
    if payload is None:
        return None
    if str(payload.get("kind", "")).strip().lower() != "clarification":
        return None

    message = str(payload.get("message", "")).strip()

    # --- Multi-question format: "questions" array ---
    raw_questions = payload.get("questions")
    if isinstance(raw_questions, list) and raw_questions:
        questions = _normalize_questions_array(raw_questions)
        if questions is None:
            return None

        # Single-item questions array → treat as single-question for backward compat
        if len(questions) == 1:
            return {
                "kind": "clarification",
                "display_text": _render_clarification_display_text(message, questions[0]["question"]),
                "question": questions[0]["question"],
                "questions": questions,
                "suggested_replies": questions[0]["suggested_replies"],
                "allow_freeform": True,
                "clarification_count": int(clarification_count),
                "normalization_source": "json",
                "problem_clear": bool(payload.get("problem_clear", False)),
            }

        # Multi-question (2-4 items)
        return {
            "kind": "clarification",
            "display_text": _render_multi_clarification_display_text(message, questions),
            "question": questions[0]["question"],
            "questions": questions,
            "suggested_replies": questions[0]["suggested_replies"],
            "allow_freeform": True,
            "clarification_count": int(clarification_count),
            "normalization_source": "json",
            "problem_clear": bool(payload.get("problem_clear", False)),
        }

    # --- Legacy single-question format: "question" + "suggested_answers" at top level ---
    question = str(payload.get("question", "")).strip()
    suggested_replies = _normalize_list(payload.get("suggested_answers"))
    if not question or len(suggested_replies) != MAX_SUGGESTED_REPLIES:
        return None

    questions_array = [{"question": question, "suggested_replies": suggested_replies}]
    return {
        "kind": "clarification",
        "display_text": _render_clarification_display_text(message, question),
        "question": question,
        "questions": questions_array,
        "suggested_replies": suggested_replies,
        "allow_freeform": True,
        "clarification_count": int(clarification_count),
        "normalization_source": "json",
        "problem_clear": bool(payload.get("problem_clear", False)),
    }


def strict_normalize_final(
    raw_text: str,
    clarification_count: int = 0,
) -> Optional[Dict[str, Any]]:
    payload = _load_payload(raw_text)
    if payload is None:
        return None
    if str(payload.get("kind", "")).strip().lower() != "final":
        return None

    industry = str(payload.get("industry", "")).strip()
    title = str(payload.get("title", "")).strip()
    problem = str(payload.get("problem", "")).strip()
    solution = str(payload.get("solution", "")).strip()
    app_type = str(payload.get("app_type", "Agent")).strip()
    if app_type not in ("Agent", "Application"):
        app_type = "Agent"
    if (
        not industry
        or not title
        or not problem
        or not solution
    ):
        return None

    # Optional enrichment fields (do not break validation if absent)
    company_context = str(payload.get("company_context", "") or "").strip() or ""
    raw_sources = payload.get("sources")
    sources = (
        [str(s).strip() for s in raw_sources if str(s).strip()]
        if isinstance(raw_sources, list)
        else None
    )

    return {
        "kind": "final",
        "display_text": _render_final_display_text(
            industry=industry,
            title=title,
            problem=problem,
            solution=solution,
            app_type=app_type,
            company_context=company_context,
            sources=sources,
        ),
        "question": None,
        "suggested_replies": [],
        "allow_freeform": True,
        "clarification_count": int(clarification_count),
        "normalization_source": "json",
        "industry": industry,
        "title": title,
        "problem": problem,
        "solution": solution,
        "app_type": app_type,
        "company_context": company_context,
        "sources": sources or [],
    }


REFINEMENT_ACTION_LABELS = ["Update Report"]


def strict_normalize_refinement(
    raw_text: str,
    clarification_count: int = 0,
) -> Optional[Dict[str, Any]]:
    payload = _load_payload(raw_text)
    if payload is None:
        return None
    if str(payload.get("kind", "")).strip().lower() != "refinement":
        return None

    message = str(payload.get("message", "")).strip()
    if not message:
        return None

    return {
        "kind": "refinement",
        "display_text": message,
        "question": None,
        "suggested_replies": list(REFINEMENT_ACTION_LABELS),
        "allow_freeform": True,
        "clarification_count": int(clarification_count),
        "normalization_source": "json",
    }


def strict_normalize_confirmed(
    raw_text: str,
    clarification_count: int = 0,
) -> Optional[Dict[str, Any]]:
    payload = _load_payload(raw_text)
    if payload is None:
        return None
    if str(payload.get("kind", "")).strip().lower() != "confirmed":
        return None

    message = str(payload.get("message", "")).strip()
    if not message:
        return None

    return {
        "kind": "confirmed",
        "display_text": message,
        "question": None,
        "suggested_replies": [],
        "allow_freeform": False,
        "clarification_count": int(clarification_count),
        "normalization_source": "json",
    }


def strict_normalize_turn(
    raw_text: str,
    clarification_count: int = 0,
    *,
    refinement_mode: bool = False,
) -> Optional[Dict[str, Any]]:
    clarification_turn = strict_normalize_clarification(
        raw_text,
        clarification_count=clarification_count,
    )
    if clarification_turn is not None:
        return clarification_turn
    final_turn = strict_normalize_final(raw_text, clarification_count=clarification_count)
    if final_turn is not None:
        return final_turn
    if refinement_mode:
        refinement_turn = strict_normalize_refinement(raw_text, clarification_count=clarification_count)
        if refinement_turn is not None:
            return refinement_turn
    return None


def make_error_turn(
    message: str,
    clarification_count: int = 0,
    *,
    message_type: str = "structured_generation_error",
) -> Dict[str, Any]:
    return {
        "kind": "error",
        "display_text": _render_error_display_text(
            str(message or DEFAULT_FINAL_ERROR).strip()
        ),
        "question": None,
        "suggested_replies": [],
        "allow_freeform": True,
        "clarification_count": int(clarification_count),
        "normalization_source": "error",
        "message_type": message_type,
    }


def finalize_normalized_turn(
    turn: Dict[str, Any],
    clarification_count: int = 0,
) -> Dict[str, Any]:
    normalized_turn = dict(turn)
    kind = str(normalized_turn.get("kind", "")).strip().lower()
    if kind == "clarification":
        normalized_turn["clarification_count"] = clarification_count + 1
        return normalized_turn

    normalized_turn["clarification_count"] = int(clarification_count)
    return normalized_turn


def to_ui_payload(turn: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "kind": str(turn.get("kind", "error")).strip() or "error",
        "question": turn.get("question"),
        "suggestedReplies": list(turn.get("suggested_replies", []) or []),
        "allowFreeform": bool(turn.get("allow_freeform", True)),
        "clarificationCount": int(turn.get("clarification_count", 0) or 0),
    }
    # Multi-question support: include questions array for the UI
    questions = turn.get("questions")
    if questions and isinstance(questions, list):
        payload["questions"] = [
            {"question": q["question"], "suggestedReplies": q["suggested_replies"]}
            for q in questions
        ]
    if payload["kind"] == "final":
        payload.update(
            {
                "app_type": str(turn.get("app_type", "Agent")).strip(),
                "industry": str(turn.get("industry", "")).strip(),
                "title": str(turn.get("title", "")).strip(),
                "problem": str(turn.get("problem", "")).strip(),
                "solution": str(turn.get("solution", "")).strip(),
            }
        )
        company_context = str(turn.get("company_context", "") or "").strip()
        if company_context:
            payload["companyContext"] = company_context
        sources = turn.get("sources")
        if sources:
            payload["sources"] = sources
    elif payload["kind"] == "confirmed":
        payload["allowFreeform"] = False
    elif payload["kind"] == "error":
        payload["messageType"] = (
            str(turn.get("message_type", "structured_generation_error")).strip()
            or "structured_generation_error"
        )
    return payload
