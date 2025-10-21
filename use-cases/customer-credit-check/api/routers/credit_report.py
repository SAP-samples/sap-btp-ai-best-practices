"""
Credit Report API router.

Generates an executive credit report using a Vertex AI (Gemini 2.5 Pro)
LLM via the shared `make_llm` helper. The report is based on:
- The original EvaluateRequest payload sent to the policy engine
- The evaluation result produced by the policy engine
- The content of the credit policy rules stored in `routers/tools/rules_prompt.md`

Endpoint:
- POST /api/credit/report
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, List

from fastapi import APIRouter
from pydantic import BaseModel, Field

from extraction.common import make_llm

from langchain.schema import HumanMessage, SystemMessage


router = APIRouter()


class CreditReportRequest(BaseModel):
    """Request payload to generate a credit report.

    - payload: EvaluateRequest payload sent to /api/credit/evaluate
    - evaluation: Result returned by /api/credit/evaluate
    - language: Desired language for the generated report (default: "en")
    - max_policy_chars: Safety limit for how much policy text to include
    """

    payload: Dict[str, Any]
    evaluation: Dict[str, Any]
    language: str = Field(default="en")
    max_policy_chars: int = Field(default=60000)


def _find_rules_prompt() -> Optional[Path]:
    """Locate the rules markdown file in `routers/tools`.

    Returns the path if found, otherwise None.
    """
    tools_dir = Path(__file__).parent / "tools"
    rules_path = tools_dir / "rules_prompt.md"
    return rules_path if rules_path.exists() else None


def _read_rules_text(rules_path: Path, max_chars: Optional[int] = None) -> str:
    """Read the full text of the rules markdown file.

    Args:
        rules_path: Path to the markdown file with credit policy rules
        max_chars: Optional character cap for safety

    Returns:
        File contents as a string (possibly truncated)
    """
    text = rules_path.read_text(encoding="utf-8")
    if max_chars is not None and max_chars > 0:
        return text[:max_chars]
    return text


def _build_messages(
    policy_text: str, evaluation: Dict[str, Any], request_payload: Dict[str, Any], language: str
) -> List[Any]:
    """Build LangChain messages for the LLM invocation.

    The system message defines the task, and the human message includes the
    contextual JSONs and policy text.
    """
    # Ensure English by default unless explicitly requested otherwise
    lang_hint = "Respond ONLY in Spanish." if not language.lower().startswith("en") else "Respond ONLY in English."

    system_msg = SystemMessage(
        content=(
            "You are an expert credit policy analyst. Generate a clear, structured, and actionable "
            "executive report about this credit evaluation. "
            "The report must include: \n"
            "Header: Current Date, Name and ID of client, Type of request and other key relevant information in a table format.\n"
            "- Include 'Usuario CxC 1' (as Responsable Cuentas por Cobrar) and 'Vendedor 1' (as Vendedor) in the header table when present in the request payload (under customer.usuario_cxc_1 and customer.vendedor_1).\n"
            "1) Approval status (Approved/Rejected/Requires Director),\n"
            "2) Detailed reasons referencing computed scores and checks (CAL, C3M, table D, docs, caps, other relevant information),\n"
            "3) Risk level (Low/Medium/High) with justification,\n"
            "4) Concrete next steps.\n\n"
            "Use the provided policy text as authoritative constraints; do not invent rules. "
            "If something is ambiguous, call it out and state reasonable assumptions. "
            "Do not mention the code variables or check names."
            f"{lang_hint}"
        )
    )

    # Pretty JSON for readability in the prompt
    eval_json = json.dumps(evaluation, ensure_ascii=False, indent=2, default=str)
    req_json = json.dumps(request_payload, ensure_ascii=False, indent=2, default=str)

    human_msg = HumanMessage(
        content=(
            "CONTEXT:\n\n"
            "EVALUATION_RESULT_JSON:\n" + eval_json + "\n\n"
            "EVALUATE_REQUEST_JSON:\n" + req_json + "\n\n"
            "POLICY_TEXT (may be truncated for length):\n" + policy_text
        )
    )

    return [system_msg, human_msg]


@router.post("/report")
def generate_report(req: CreditReportRequest) -> Dict[str, Any]:
    """Generate an executive credit report using Gemini 2.5 Pro (Vertex AI).

    This endpoint internally loads the policy rules from `rules_prompt.md` so clients only need
    to provide the evaluation result and original request payload.
    """
    # 1) Locate and read the rules markdown
    rules_path = _find_rules_prompt()
    if not rules_path:
        return {"success": False, "error": "rules_prompt.md not found in routers/tools"}

    policy_text = _read_rules_text(rules_path, max_chars=req.max_policy_chars)
    if not policy_text:
        return {"success": False, "error": "Failed to read policy rules from rules_prompt.md"}

    # 2) Build LLM and messages
    llm = make_llm(
        provider="vertex",
        model_name="gemini-2.5-pro",
        temperature=0.2,
        # max_tokens=3000,
    )

    messages = _build_messages(
        policy_text=policy_text,
        evaluation=req.evaluation,
        request_payload=req.payload,
        language=req.language,
    )

    # 3) Invoke LLM
    try:
        response = llm.invoke(messages)
        report_text = getattr(response, "content", None) or str(response)
        return {"success": True, "report": report_text}
    except Exception as e:
        return {"success": False, "error": f"LLM invocation failed: {e}"}

