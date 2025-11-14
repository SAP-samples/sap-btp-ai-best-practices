"""Service responsible for turning diagram images into BPMN XML via LLM.

V2 implementation: deterministic three-stage pipeline using image input
  1) image → graph JSON (VISION_TO_GRAPH_PROMPT)
  2) graph JSON → BPMN XML with DI (GRAPH_TO_BPMN_PROMPT)
  3) graph JSON + BPMN XML → validated BPMN XML (BPMN_VALIDATION_PROMPT)

All three stages use the same provider/model. Non-reasoning models run with
temperature=0; reasoning models (e.g., gpt-5) keep reasoning mode on.
Stage C always runs to ensure structural integrity and completeness.
"""

from __future__ import annotations

import base64
import logging
from functools import lru_cache
import re
from typing import Callable, Dict, Optional, Tuple
import xml.etree.ElementTree as ET

from gen_ai_hub.proxy.native.amazon.clients import Session
from gen_ai_hub.proxy.native.google_vertexai.clients import GenerativeModel
from gen_ai_hub.proxy.native.openai import chat

from ..models.chat import ChatResponse
from ..prompts.bpmn_prompt_v2 import (
    VISION_TO_GRAPH_PROMPT,
    GRAPH_TO_BPMN_PROMPT,
    BPMN_VALIDATION_PROMPT,
)

logger = logging.getLogger(__name__)

DEFAULT_PROVIDER = "anthropic"

PROVIDER_DEFAULT_MODELS: Dict[str, str] = {
    "openai": "gpt-4.1",
    "anthropic": "anthropic--claude-4-sonnet",
    "gemini": "gemini-2.5-pro",
}


def normalize_provider(provider: Optional[str]) -> str:
    """Normalize and validate the requested provider."""
    if provider is None:
        return DEFAULT_PROVIDER

    normalized = provider.strip().lower()
    if not normalized:
        normalized = DEFAULT_PROVIDER

    if normalized not in PROVIDER_DEFAULT_MODELS:
        raise ValueError(
            f"Unsupported provider '{provider}'. "
            f"Supported providers: {', '.join(PROVIDER_DEFAULT_MODELS)}"
        )
    return normalized


class BPMNGenerator:
    """Encapsulates prompt loading and LLM invocation."""

    def __init__(self, prompt_text: str):
        self.prompt_text = prompt_text

    def generate(
        self,
        image_bytes: bytes,
        provider: Optional[str],
        model_name: Optional[str] = None,
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> Tuple[str, str, ChatResponse]:
        """Generate BPMN XML for a given diagram image.

        Args:
            image_bytes: Raw byte content of the uploaded image.
            provider: Provider identifier (openai, anthropic, gemini).
            model_name: Optional model to override the provider default.
            filename: Optional filename for logging context.

        Returns:
            Tuple containing the resolved provider key, model name, and ChatResponse.
        """
        if not image_bytes:
            raise ValueError("Image content is empty.")

        provider_key = normalize_provider(provider)
        resolved_model = self._resolve_model(provider_key, model_name)

        # --- Stage A: image -> graph JSON ---
        try:
            stage_a_text = self._build_stage_a_user_text(filename)
            resp_a = self._invoke_with_image(
                provider_key,
                resolved_model,
                VISION_TO_GRAPH_PROMPT,
                stage_a_text,
                image_bytes,
                content_type,
            )
            graph_json_str = _extract_json_graph(resp_a.text)
            if not graph_json_str:
                raise ValueError("Stage A returned no parseable JSON graph.")

            # --- Stage B: graph JSON -> BPMN XML ---
            stage_b_text = self._build_stage_b_user_text(graph_json_str)
            resp_b = self._invoke_text(
                provider_key,
                resolved_model,
                GRAPH_TO_BPMN_PROMPT,
                stage_b_text,
            )

            xml = _extract_bpmn_xml(resp_b.text)
            if not xml:
                raise ValueError("Stage B returned no BPMN XML.")

            # --- Stage C: BPMN validation against Graph JSON (always runs) ---
            stage_c_text = self._build_stage_c_user_text(graph_json_str, xml)
            resp_c = self._invoke_text(
                provider_key,
                resolved_model,
                BPMN_VALIDATION_PROMPT,
                stage_c_text,
            )
            raw_c = resp_c.text or ""
            # Try JSON/text patch first; fallback to full corrected XML
            patch_str = _extract_json_patch(raw_c)
            validated_xml = ""
            if patch_str:
                applied = _apply_bpmn_patch(xml, patch_str)
                if applied:
                    validated_xml = applied
            if not validated_xml:
                validated_xml = _extract_bpmn_xml(raw_c)
            if not validated_xml:
                raise ValueError("Stage C failed to produce BPMN XML.")
            if not _has_bpmndi(validated_xml):
                raise ValueError("Stage C output still missing BPMN DI section.")

            usage = {"stage_a": resp_a.usage, "stage_b": resp_b.usage, "stage_c": resp_c.usage}
            return provider_key, resolved_model, ChatResponse(
                text=validated_xml,
                model=resolved_model,
                success=True,
                usage=usage,
            )
        except Exception as exc:  # pragma: no cover - LLM exceptions are runtime
            logger.exception("BPMN generation failed via LLM: %s", exc)
            return provider_key, resolved_model, ChatResponse(
                text="",
                model=resolved_model,
                success=False,
                error=str(exc),
            )

    def get_default_model(self, provider: Optional[str]) -> str:
        """Expose the default model used for a provider."""
        provider_key = normalize_provider(provider)
        return PROVIDER_DEFAULT_MODELS[provider_key]

    def _resolve_model(self, provider: str, model_name: Optional[str]) -> str:
        """Resolve the model name, falling back to provider defaults."""
        if model_name:
            candidate = model_name.strip()
            if candidate:
                return candidate
        return PROVIDER_DEFAULT_MODELS[provider]

    def _build_stage_a_user_text(self, filename: Optional[str]) -> str:
        suffix = f"Filename: {filename}\n" if filename else ""
        return (
            "Analyze the following diagram IMAGE and return a STRICT JSON graph "
            "according to the schema in the system prompt. No markdown, no preamble.\n"
            + suffix
        )

    def _build_stage_b_user_text(self, graph_json: str) -> str:
        return (
            "Convert this GRAPH JSON into BPMN 2.0 XML compatible with Signavio. "
            "Return only the XML.\n\nGRAPH JSON:\n" + graph_json
        )

    def _build_stage_c_user_text(self, graph_json: str, xml: str) -> str:
        return (
            "Validate and correct the BPMN XML against the source GRAPH JSON. "
            "Ensure structural integrity, reference validity, XML well-formedness, and complete BPMN DI. "
            "Prefer a minimal JSON patch per the system prompt (with optional text_edits then edits). "
            "If a patch cannot express the fixes, return the full corrected XML only.\n\n"
            "GRAPH JSON (source of truth):\n"
            + graph_json
            + "\n\nBPMN XML (to be validated and corrected):\n"
            + xml
        )

    # --- Generic invoke helpers ---
    def _invoke_text(
        self,
        provider: str,
        model_name: str,
        system_text: str,
        user_text: str,
    ) -> ChatResponse:
        if provider == "openai":
            return self._invoke_openai(model_name, {"system": system_text, "user": user_text})
        if provider == "anthropic":
            return self._invoke_anthropic(model_name, {"system": system_text, "user": user_text})
        return self._invoke_gemini(model_name, {"system": system_text, "user": user_text})

    def _invoke_with_image(
        self,
        provider: str,
        model_name: str,
        system_text: str,
        user_text: str,
        image_bytes: bytes,
        mime: Optional[str],
    ) -> ChatResponse:
        if provider == "openai":
            return self._invoke_openai_with_image(model_name, system_text, user_text, image_bytes, mime)
        if provider == "anthropic":
            return self._invoke_anthropic_with_image(model_name, system_text, user_text, image_bytes, mime)
        return self._invoke_gemini_with_image(model_name, system_text, user_text, image_bytes, mime)

    def _invoke_openai(self, model_name: str, payload: Dict[str, str]) -> ChatResponse:
        """Invoke OpenAI compatible models."""
        messages = [
            {"role": "system", "content": payload["system"]},
            {"role": "user", "content": payload["user"]},
        ]

        if model_name == "gpt-5":
            reasoning_effort = "high"
            response = chat.completions.create(
                messages=messages,
                model=model_name,
                reasoning_effort=reasoning_effort,
            )
        else:
            response = chat.completions.create(
                messages=messages,
                model=model_name,
                temperature=0,
            )

        text = response.choices[0].message.content
        usage = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
            "completion_tokens": getattr(response.usage, "completion_tokens", None),
            "total_tokens": getattr(response.usage, "total_tokens", None),
        }

        return ChatResponse(
            text=text,
            model=model_name,
            success=True,
            usage=usage,
        )

    def _invoke_anthropic(self, model_name: str, payload: Dict[str, str]) -> ChatResponse:
        """Invoke Anthropic Sonnet (via Bedrock)."""
        bedrock = Session().client(model_name=model_name)
        messages = [
            {
                "role": "user",
                "content": [{"text": payload["user"]}],
            }
        ]

        response = bedrock.converse(
            system=[{"text": payload["system"]}],
            messages=messages,
            inferenceConfig={"temperature": 0},
        )

        text = response["output"]["message"]["content"][0]["text"]
        usage = response.get("usage", {})

        return ChatResponse(
            text=text,
            model=model_name,
            success=True,
            usage=usage,
        )

    def _invoke_gemini(self, model_name: str, payload: Dict[str, str]) -> ChatResponse:
        """Invoke Gemini via Vertex AI."""
        prompt = f"{payload['system']}\n\n{payload['user']}"
        model = GenerativeModel(model_name)

        response = model.generate_content(
            contents=prompt,
            generation_config={"temperature": 0},
        )

        text = response.text
        usage = {
            "prompt_tokens": getattr(
                getattr(response, "usage_metadata", None), "prompt_token_count", None
            ),
            "completion_tokens": getattr(
                getattr(response, "usage_metadata", None), "candidates_token_count", None
            ),
            "total_tokens": getattr(
                getattr(response, "usage_metadata", None), "total_token_count", None
            ),
        }

        return ChatResponse(
            text=text,
            model=model_name,
            success=True,
            usage=usage,
        )

    # --- Provider helpers with IMAGE inputs ---
    def _invoke_openai_with_image(
        self,
        model_name: str,
        system_text: str,
        user_text: str,
        image_bytes: bytes,
        mime: Optional[str],
    ) -> ChatResponse:
        from gen_ai_hub.proxy.native.openai import chat

        mime = mime or "image/png"
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime};base64,{b64}"

        messages = [
            {"role": "system", "content": system_text},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]

        if model_name == "gpt-5":
            response = chat.completions.create(
                messages=messages,
                model=model_name,
                reasoning_effort="medium",
            )
        else:
            response = chat.completions.create(
                messages=messages,
                model=model_name,
                temperature=0,
            )

        text = response.choices[0].message.content
        usage = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
            "completion_tokens": getattr(response.usage, "completion_tokens", None),
            "total_tokens": getattr(response.usage, "total_tokens", None),
        }
        return ChatResponse(text=text, model=model_name, success=True, usage=usage)

    def _invoke_anthropic_with_image(
        self,
        model_name: str,
        system_text: str,
        user_text: str,
        image_bytes: bytes,
        mime: Optional[str],
    ) -> ChatResponse:
        from gen_ai_hub.proxy.native.amazon.clients import Session

        bedrock = Session().client(model_name=model_name)

        fmt = "png"
        if (mime or "").endswith("jpeg") or (mime or "").endswith("jpg"):
            fmt = "jpeg"
        elif (mime or "").endswith("webp"):
            fmt = "webp"

        messages = [
            {
                "role": "user",
                "content": [
                    {"image": {"format": fmt, "source": {"bytes": image_bytes}}},
                    {"text": user_text},
                ],
            }
        ]

        response = bedrock.converse(
            system=[{"text": system_text}],
            messages=messages,
            inferenceConfig={"temperature": 0},
        )

        text = response["output"]["message"]["content"][0]["text"]
        usage = response.get("usage", {})
        return ChatResponse(text=text, model=model_name, success=True, usage=usage)

    def _invoke_gemini_with_image(
        self,
        model_name: str,
        system_text: str,
        user_text: str,
        image_bytes: bytes,
        mime: Optional[str],
    ) -> ChatResponse:
        from gen_ai_hub.proxy.native.google_vertexai.clients import GenerativeModel

        mime_type = mime or "image/png"
        model = GenerativeModel(model_name, system_instruction=system_text)

        contents = [
            {
                "role": "user",
                "parts": [
                    {"text": user_text},
                    {"inline_data": {"mime_type": mime_type, "data": image_bytes}},
                ],
            }
        ]

        response = model.generate_content(
            contents=contents,
            generation_config={"temperature": 0},
        )

        text = getattr(response, "text", "")
        usage = {
            "prompt_tokens": getattr(
                getattr(response, "usage_metadata", None), "prompt_token_count", None
            ),
            "completion_tokens": getattr(
                getattr(response, "usage_metadata", None), "candidates_token_count", None
            ),
            "total_tokens": getattr(
                getattr(response, "usage_metadata", None), "total_token_count", None
            ),
        }
        return ChatResponse(text=text, model=model_name, success=True, usage=usage)


@lru_cache()
def get_bpmn_generator() -> BPMNGenerator:
    """Cached dependency for FastAPI using LLM BPMN generator."""
    # prompt_text is unused in v2, kept for backward compatibility
    return BPMNGenerator(prompt_text="")


def _extract_bpmn_xml(raw_text: Optional[str]) -> str:
    """Normalize LLM output so only the BPMN XML remains."""
    if not raw_text:
        return ""

    text = raw_text.strip()

    # If XML is inside a fenced code block
    code_block = re.search(r"```(?:xml)?\s*(.+?)```", text, re.DOTALL | re.IGNORECASE)
    if code_block:
        text = code_block.group(1).strip()
    else:
        # Attempt to slice between XML start and definitions end
        xml_start = text.find("<?xml")
        xml_end = text.rfind("</definitions>")
        if xml_start != -1 and xml_end != -1:
            xml_end += len("</definitions>")
            text = text[xml_start:xml_end].strip()

    return text


def _extract_json_graph(raw_text: Optional[str]) -> str:
    """Extract a JSON object from model output.

    Tries code fences first; otherwise attempts to slice between the first '{'
    and the last '}' and returns that substring if it parses as JSON.
    """
    if not raw_text:
        return ""
    text = raw_text.strip()

    # Code fence with optional json tag
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    candidate = m.group(1).strip() if m else text

    # If not fenced, try full text first
    import json

    def _try(s: str) -> Optional[str]:
        try:
            json.loads(s)
            return s
        except Exception:
            return None

    out = _try(candidate)
    if out:
        return out

    # Fallback: slice between first '{' and last '}'
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        out = _try(snippet)
        if out:
            return out

    return ""


def _has_bpmndi(xml_text: str) -> bool:
    """Check whether BPMN DI elements exist in the BPMN XML."""
    if not xml_text:
        return False
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return False

    namespaces = {
        "bpmndi": "http://www.omg.org/spec/BPMN/20100524/DI",
        "bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL",
    }

    return root.find(".//bpmndi:BPMNDiagram", namespaces) is not None


def _extract_json_patch(raw_text: Optional[str]) -> str:
    """Extract a JSON patch object (with 'edits' and/or 'text_edits') from model output."""
    if not raw_text:
        return ""
    text = raw_text.strip()

    # Prefer fenced JSON
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    candidate = m.group(1).strip() if m else text

    import json

    def _looks_like_patch(obj) -> bool:
        if not isinstance(obj, dict):
            return False
        if "edits" in obj and isinstance(obj.get("edits"), list):
            return True
        if "text_edits" in obj and isinstance(obj.get("text_edits"), list):
            return True
        return False

    def _try(s: str) -> Optional[str]:
        try:
            obj = json.loads(s)
            return s if _looks_like_patch(obj) else None
        except Exception:
            return None

    out = _try(candidate)
    if out:
        return out

    # Fallback slice
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        out = _try(snippet)
        if out:
            return out
    return ""


def _apply_bpmn_patch(base_xml: str, patch_json: str) -> Optional[str]:
    """Apply a two-phase patch: text edits (pre-parse) then element edits (id-targeted)."""
    import json
    try:
        patch = json.loads(patch_json)
    except Exception:
        return None

    text_edits = patch.get("text_edits", []) or []
    element_edits = patch.get("edits", []) or []

    def _apply_text_edits(xml_text: str, edits) -> str:
        for e in edits:
            op = e.get("op")
            if op == "regex_sub":
                pattern = e.get("pattern", "")
                repl = e.get("repl", "")
                flags_str = (e.get("flags") or "").lower()
                flags = 0
                if "i" in flags_str:
                    flags |= re.IGNORECASE
                if "m" in flags_str:
                    flags |= re.MULTILINE
                if "s" in flags_str:
                    flags |= re.DOTALL
                try:
                    xml_text = re.sub(pattern, repl, xml_text, flags=flags)
                except re.error:
                    # ignore invalid regex; continue
                    pass
            elif op == "insert_before":
                needle = e.get("needle", "")
                txt = e.get("text", "")
                idx = xml_text.find(needle)
                if idx != -1:
                    xml_text = xml_text[:idx] + txt + xml_text[idx:]
            elif op == "insert_after":
                needle = e.get("needle", "")
                txt = e.get("text", "")
                idx = xml_text.find(needle)
                if idx != -1:
                    pos = idx + len(needle)
                    xml_text = xml_text[:pos] + txt + xml_text[pos:]
            # unknown ops ignored
        return xml_text

    # Phase 1: attempt to fix well-formedness with text edits
    working_xml = _apply_text_edits(base_xml, text_edits) if text_edits else base_xml

    # If no element edits, return the possibly text-fixed XML
    if not element_edits:
        return working_xml

    # Phase 2: element edits (requires parseable XML)
    try:
        root = ET.fromstring(working_xml)
    except ET.ParseError:
        return None

    NS_DECL = (
        "xmlns:bpmn='http://www.omg.org/spec/BPMN/20100524/MODEL' "
        "xmlns:bpmndi='http://www.omg.org/spec/BPMN/20100524/DI' "
        "xmlns:di='http://www.omg.org/spec/DD/20100524/DI' "
        "xmlns:dc='http://www.omg.org/spec/DD/20100524/DC'"
    )

    def parse_snippet(snippet: str) -> Optional[ET.Element]:
        try:
            wrapper = f"<tmp {NS_DECL}>" + snippet + "</tmp>"
            tmp = ET.fromstring(wrapper)
            children = list(tmp)
            return children[0] if children else None
        except ET.ParseError:
            return None

    def find_by_id(elem: ET.Element, elem_id: str) -> Optional[ET.Element]:
        return elem.find(f".//*[@id='{elem_id}']")

    def find_parent(current: ET.Element, target: ET.Element) -> Optional[ET.Element]:
        for child in list(current):
            if child is target:
                return current
            found = find_parent(child, target)
            if found is not None:
                return found
        return None

    for e in element_edits:
        op = e.get("op")
        if op == "set_attr":
            target = find_by_id(root, e.get("id", ""))
            if target is not None:
                name = e.get("name", "")
                value = e.get("value", "")
                if name:
                    target.set(name, value)
        elif op == "delete_element":
            target = find_by_id(root, e.get("id", ""))
            if target is not None:
                parent = find_parent(root, target)
                if parent is not None:
                    parent.remove(target)
        elif op == "replace_element":
            target = find_by_id(root, e.get("id", ""))
            new_elem = parse_snippet(e.get("new_xml", ""))
            if target is not None and new_elem is not None:
                parent = find_parent(root, target)
                if parent is not None:
                    idx = list(parent).index(target)
                    parent.remove(target)
                    parent.insert(idx, new_elem)
        elif op == "insert_child":
            parent = find_by_id(root, e.get("parent_id", ""))
            new_elem = parse_snippet(e.get("new_xml", ""))
            if parent is not None and new_elem is not None:
                before_id = e.get("before_id")
                after_id = e.get("after_id")
                if before_id or after_id:
                    siblings = list(parent)
                    pos = None
                    for i, s in enumerate(siblings):
                        sid = s.attrib.get("id")
                        if sid == (before_id or after_id):
                            pos = i if before_id else i + 1
                            break
                    if pos is not None:
                        parent.insert(pos, new_elem)
                    else:
                        parent.append(new_elem)
                else:
                    parent.append(new_elem)
        # unknown ops ignored

    try:
        xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        return xml_bytes.decode("utf-8")
    except Exception:
        return None
