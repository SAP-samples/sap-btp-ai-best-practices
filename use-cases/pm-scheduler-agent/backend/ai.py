"""SAP Generative AI Hub integration and prompt builders."""
from datetime import date


def init_llm():
    """Connect to SAP Gen AI Hub. Returns (llm, error_message)."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
        from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
        proxy_client = get_proxy_client("gen-ai-hub")
        llm = ChatOpenAI(proxy_model_name="gpt-5.4", proxy_client=proxy_client)
        return llm, None
    except Exception as exc:
        return None, str(exc)


_llm_singleton = None
_llm_error = None


def get_llm():
    global _llm_singleton, _llm_error
    if _llm_singleton is None and _llm_error is None:
        _llm_singleton, _llm_error = init_llm()
    return _llm_singleton, _llm_error


def ask_llm(prompt: str) -> str:
    llm, err = get_llm()
    if llm is None:
        return f"LLM unavailable: {err}"
    try:
        return llm.invoke(prompt).content
    except Exception as exc:
        return f"LLM call failed: {exc}"


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_explanation_prompt(schedule_json: list, week_start: str,
                              opp_added: list | None = None) -> str:
    lines = [
        "You are a senior maintenance scheduler at FMI .",
        f"Proposed schedule for the week of {week_start}.",
        "Only P3 (Medium) and P4 (Low) priority orders are included.",
        "MN03 orders are scheduled on their exact BASIC_START_DATE week (no sliding).",
        "MN01 orders use a flexible window.",
        "Explain WHY each work center's orders were scheduled or deferred.",
        "Reference order type, criticality, capacity, and scheduling windows.",
        "Provide 2-3 concrete recommendations to improve schedule attainment.",
        "Format with one section per work center. Be concise.",
        "",
    ]
    for wc in schedule_json:
        load = wc["load_pct"]
        lines.append(f"=== {wc['work_center']} === {wc['capacity_used']:.1f}h / {wc['capacity_available']:.1f}h ({load:.0f}%)")
        for op in wc["scheduled"][:5]:
            tag = " [OPPORTUNISTIC]" if op.get("_OPPORTUNISTIC") else ""
            lines.append(f"  + {op.get('ORDER_NO')}{tag} | {op.get('OPER_SHORT_TEXT')} | "
                         f"Type:{op.get('ORDER_TYPE_CODE')} | Crit:{op.get('EQUIPMENT_CRITICALITY')} | "
                         f"{op.get('ACTIVITY_WORK_INVOLVE',0):.1f}h")
        for op in wc["unscheduled"][:3]:
            lines.append(f"  - DEFERRED {op.get('ORDER_NO')} | {op.get('OPER_SHORT_TEXT')} | "
                         f"{op.get('ACTIVITY_WORK_INVOLVE',0):.1f}h")
        lines.append("")

    if opp_added:
        lines.append("=== Opportunistic Orders (Equipment Down) ===")
        for op in opp_added[:5]:
            lines.append(f"  * {op.get('ORDER_NO')} | {op.get('OPER_SHORT_TEXT')} | "
                         f"Orig week: {str(op.get('BASIC_START_DATE',''))[:10]}")
        lines.append("Explain the scheduling benefit of grouping these with the downtime.")
        lines.append("")

    return "\n".join(lines)


def build_conflict_prompt(schedule_json: list) -> str:
    overloaded = [w for w in schedule_json
                  if w["capacity_available"] > 0 and w["load_pct"] > 90]
    underused = [w for w in schedule_json
                 if w["capacity_available"] > 0 and w["load_pct"] < 50 and w["capacity_used"] > 0]
    deferred = [w for w in schedule_json if w["unscheduled"]]

    lines = [
        "You are a maintenance scheduling AI at FMI.",
        "Analyse the following conflicts and provide a numbered list of 3-5 specific actions.",
        "",
    ]
    if overloaded:
        lines.append("OVERLOADED (>90%):")
        for w in overloaded:
            lines.append(f"  {w['work_center']}: {w['capacity_used']:.0f}h/{w['capacity_available']:.0f}h — {len(w['unscheduled'])} deferred")
    if underused:
        lines.append("UNDER-UTILISED (<50%):")
        for w in underused:
            lines.append(f"  {w['work_center']}: {w['capacity_used']:.0f}h/{w['capacity_available']:.0f}h")
    if deferred:
        lines.append("DEFERRED HIGH-PRIORITY:")
        for w in deferred:
            for op in [o for o in w["unscheduled"] if o.get("PRIORITY") in ("High","Medium")][:3]:
                lines.append(f"  {op.get('ORDER_NO')} ({w['work_center']}) | {op.get('OPER_SHORT_TEXT')} | Crit:{op.get('EQUIPMENT_CRITICALITY')}")
    if not (overloaded or underused or deferred):
        lines.append("No major conflicts. All work centers within acceptable utilisation.")
    return "\n".join(lines)


def build_query_prompt(question: str, context: str) -> str:
    return (
        "You are a maintenance scheduling AI at FMI (Freeport-McMoRan mines). "
        f"Schedule context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely and professionally."
    )
