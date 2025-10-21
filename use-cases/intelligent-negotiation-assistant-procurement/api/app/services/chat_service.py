from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.core.llm import create_llm
from app.services.kg_service import KGService


@dataclass
class SupplierRef:
    """Reference to a supplier KG by id or inline payload.

    Attributes:
        id: Optional supplier identifier; if provided, server resolves KG from static storage.
        name: Human-readable supplier name (optional; defaults from env if id is used).
        kg_json: Inline knowledge graph (optional alternative to id).
    """

    id: Optional[str] = None
    name: Optional[str] = None
    kg_json: Optional[Dict[str, Any]] = None


class ChatService:
    """Encapsulates RFQ chatbot orchestration on the server side."""

    def __init__(self) -> None:
        self.kg_service = KGService()

    def _ensure_kg(self, ref: SupplierRef) -> Tuple[Dict[str, Any], str]:
        """Load KG JSON and resolve supplier display name for a reference."""
        if ref.kg_json is not None:
            name = ref.name or (ref.id or "Supplier")
            return ref.kg_json, name
        if not ref.id:
            raise ValueError("Supplier reference must include either id or kg_json")
        kg = self.kg_service.load_static_supplier_kg(ref.id)
        # Prefer name from env mapping
        if ref.id == self.kg_service.supplier1_id:
            name = ref.name or self.kg_service.supplier1_name
        elif ref.id == self.kg_service.supplier2_id:
            name = ref.name or self.kg_service.supplier2_name
        else:
            name = ref.name or ref.id
        return kg, name

    def _extract_text(self, response: Any) -> str:
        if response is None:
            return ""
        try:
            content = response.content if hasattr(response, "content") else str(response)
            if isinstance(content, list):
                return content[0] if len(content) == 1 else " ".join(str(x) for x in content)
            return str(content)
        except Exception:
            return str(response)

    def _clean_fenced_json(self, text: str) -> str:
        t = str(text or "").strip()
        if t.startswith("```json"):
            t = t[7:]
        if t.startswith("```"):
            t = t[3:]
        if t.endswith("```"):
            t = t[:-3]
        return t.strip()

    def _build_supplier_prompt(self, question: str, supplier_name: str, kg_data: Dict[str, Any]) -> str:
        return (
            f"You are analyzing an RFQ supplier knowledge graph for: {supplier_name}.\n"
            "Answer the user question ONLY using information from the provided knowledge graph.\n"
            "Return ONLY valid JSON (no commentary) with this exact schema:\n"
            "{\n"
            "  \"supplier\": \"<supplier_name>\",\n"
            "  \"answer\": \"<concise answer based on the graph>\",\n"
            "  \"sources\": [ {\n"
            "    \"filename\": \"<source filename>\",\n"
            "    \"chunk_id\": \"<page or chunk id>\"\n"
            "  } ]\n"
            "}\n\n"
            f"Question:\n{question}\n\n"
            "Knowledge Graph (JSON):\n"
            f"{json.dumps(kg_data, ensure_ascii=False)}"
        )

    def _build_consolidation_prompt(self, question: str, s1: Dict[str, Any], s2: Dict[str, Any]) -> str:
        return (
            "You are consolidating two supplier-specific answers derived from their respective "
            "knowledge graphs. Produce a concise, well-structured Markdown response that:\n"
            "- Answers the user's question\n"
            "- Clearly distinguishes supplier-specific facts when helpful\n"
            "- Does not invent information\n"
            "- Includes a final 'Sources' section with deduplicated citations in the form 'filename:chunk_id'\n\n"
            f"Question:\n{question}\n\n"
            "Supplier 1 JSON:\n"
            f"{json.dumps(s1, ensure_ascii=False, indent=2)}\n\n"
            "Supplier 2 JSON:\n"
            f"{json.dumps(s2, ensure_ascii=False, indent=2)}\n\n"
            "Return only the Markdown answer (no JSON)."
        )

    def ask(self, question: str, supplier1: SupplierRef, supplier2: SupplierRef, model: Optional[str] = None) -> Dict[str, Any]:
        kg1, name1 = self._ensure_kg(supplier1)
        kg2, name2 = self._ensure_kg(supplier2)

        model_name = model or "gemini-2.5-flash"
        llm = create_llm(model_name=model_name, temperature=0.0)

        # Per-supplier calls
        s1_prompt = self._build_supplier_prompt(question, name1, kg1)
        s2_prompt = self._build_supplier_prompt(question, name2, kg2)
        s1_text = self._clean_fenced_json(self._extract_text(llm.invoke(s1_prompt)))
        s2_text = self._clean_fenced_json(self._extract_text(llm.invoke(s2_prompt)))
        try:
            s1_json = json.loads(s1_text) if s1_text else {"supplier": name1, "answer": "", "sources": []}
        except Exception:
            s1_json = {"supplier": name1, "answer": s1_text or "", "sources": []}
        try:
            s2_json = json.loads(s2_text) if s2_text else {"supplier": name2, "answer": "", "sources": []}
        except Exception:
            s2_json = {"supplier": name2, "answer": s2_text or "", "sources": []}

        # Consolidation
        consolidation_prompt = self._build_consolidation_prompt(question, s1_json, s2_json)
        final_text = self._extract_text(llm.invoke(consolidation_prompt)).strip()
        if not final_text:
            final_text = f"{name1}: {s1_json.get('answer','')}\n\n{name2}: {s2_json.get('answer','')}"

        return {"answer_markdown": final_text, "sources": [*s1_json.get("sources", []), *s2_json.get("sources", [])]}
