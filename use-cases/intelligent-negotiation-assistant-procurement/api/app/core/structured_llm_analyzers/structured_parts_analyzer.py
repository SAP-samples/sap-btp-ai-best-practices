"""
Structured Parts Analyzer - Returns JSON for Dashboard

This module analyzes parts information from knowledge graphs and returns
structured JSON data with source traceability for parts comparison across
any domain-specific components.


Embedding-based blocking is intentionally gated behind future flags to
avoid runtime dependency changes; symbolic blocking is used by default.
"""
import json
import re
import hashlib
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set
from dotenv import load_dotenv

load_dotenv()

from app.core.llm import create_llm
from .business_context import format_prompt_with_business_context


# ==== POC-specific constants and helpers (strict categories) ====
CORE_PART_CATEGORIES_DEFAULT = [
    "Part A",
    "Part B",
    "Part C",
]
OTHERS_CATEGORY = "Others"


def _slugify(text: str) -> str:
    """Create a lowercase underscore slug for keys."""
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(text)).strip("_")
    return text.lower() or "item"


def _normalize_category(raw: Optional[str], core_categories: List[str]) -> str:
    """
    Map any incoming category to one of the strict core categories or Others.
    This generic version performs only a case-insensitive direct match against
    provided categories and otherwise returns Others.
    """
    if not raw:
        return OTHERS_CATEGORY
    val = str(raw).strip().lower()

    # Try direct match against provided categories (case-insensitive)
    for c in core_categories:
        if val == c.lower():
            return c
    return OTHERS_CATEGORY


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith('```json'):
        text = text[7:]
    elif text.startswith('```'):
        text = text[3:]
    if text.endswith('```'):
        text = text[:-3]
    return text.strip()


def _ensure_sources(sources: Any) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if isinstance(sources, list):
        for s in sources:
            if isinstance(s, dict):
                out.append({
                    "filename": str(s.get("filename", "")),
                    "chunk_id": str(s.get("chunk_id", "")),
                })
    return out


def _stringify_other_spec(item: Any) -> str:
    """Convert a spec entry (dict or scalar) into a concise string."""
    if isinstance(item, dict):
        name = str(item.get("name") or item.get("metric") or "").strip()
        value = str(item.get("value") or "").strip()
        notes = str(item.get("notes") or "").strip()
        pieces: List[str] = []
        if name and value:
            pieces.append(f"{name}: {value}")
        elif name:
            pieces.append(name)
        elif value:
            pieces.append(value)
        if notes:
            pieces.append(notes)
        return " - ".join(pieces) if pieces else json.dumps(item, ensure_ascii=False)
    return str(item)


def _normalize_tech_specs_to_strings(ts: Any) -> Dict[str, Any]:
    """Ensure technical_specifications.other_specs is a list[str]."""
    default = {
        "dimensions": "",
        "weight": "",
        "material": "",
        "torque_capacity": "",
        "operating_temperature": "",
        "other_specs": [],
    }
    if not isinstance(ts, dict):
        return default
    out = {
        "dimensions": ts.get("dimensions", ""),
        "weight": ts.get("weight", ""),
        "material": ts.get("material", ""),
        "torque_capacity": ts.get("torque_capacity", ""),
        "operating_temperature": ts.get("operating_temperature", ""),
        "other_specs": [],
    }
    raw_other = ts.get("other_specs")
    if isinstance(raw_other, list):
        out["other_specs"] = [
            _stringify_other_spec(x) for x in raw_other if x is not None
        ]
    elif raw_other is not None:
        out["other_specs"] = [_stringify_other_spec(raw_other)]
    return out


def _looks_like_year_or_range(value: Any) -> bool:
    """Detect if a string represents a year or year range (e.g., "2029-2032")."""
    if not isinstance(value, str):
        return False
    text = value.strip()
    return bool(re.fullmatch(r"(?i)\s*(?:19|20)\d{2}(?:\s*[-–]\s*(?:19|20)\d{2})?\s*", text))


def _strip_year_parentheticals(text: str) -> str:
    """Remove parenthetical year ranges from text, e.g., "70000 units (2027-2028)" -> "70000 units"."""
    if not isinstance(text, str):
        return text
    return re.sub(r"\s*\((?:(?:19|20)\d{2})(?:\s*[-–]\s*(?:19|20)\d{2})?\)\s*", "", text).strip()


def _coerce_price_to_float(val: Any) -> Optional[float]:
    """Try to parse a price value that may include currency symbols or be a string."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val)
    m = re.search(r"[-+]?[0-9]*\.?[0-9]+", s.replace(',', ''))
    if m:
        try:
            return float(m.group(0))
        except Exception:
            return None
    return None


def _normalize_volume_pricing_list(items: Any) -> List[Dict[str, Any]]:
    """
    Normalize volume_pricing entries to ensure 'volume' captures quantities, not years.
    - Remove parenthetical year ranges
    - Drop entries where volume is only a year/range and no meaningful price
    - Coerce price_per_unit to float when possible
    - Keep discount_percentage if numeric, else None
    """
    normalized: List[Dict[str, Any]] = []
    if not isinstance(items, list):
        return normalized
    for entry in items:
        if not isinstance(entry, dict):
            continue
        raw_volume = entry.get('volume')
        volume_text = _strip_year_parentheticals(str(raw_volume)) if raw_volume is not None else ""
        price_val = _coerce_price_to_float(entry.get('price_per_unit'))
        discount = entry.get('discount_percentage')
        if _looks_like_year_or_range(volume_text):
            if price_val is None:
                continue
            volume_text = ""
        if (volume_text and volume_text.strip()) or (price_val is not None):
            normalized.append({
                'volume': volume_text.strip(),
                'price_per_unit': price_val,
                'discount_percentage': discount if isinstance(discount, (int, float)) else None,
            })
    return normalized


def _build_concentrated_from_parts(parts: List[Dict[str, Any]], supplier_name: str, core_categories: List[str]) -> Dict[str, Any]:
    """Build UI-friendly concentrated_parts from flat parts list."""
    buckets: Dict[str, List[Dict[str, Any]]] = {c: [] for c in core_categories}
    buckets[OTHERS_CATEGORY] = []

    for p in parts:
        cat = _normalize_category(p.get("category"), core_categories)
        primary_num = p.get("part_number") or p.get("part_name") or ""
        basis = f"{supplier_name}|{primary_num}|{p.get('part_name', '')}"
        canon_id = f"canon:{hashlib.md5(basis.encode('utf-8')).hexdigest()[:12]}"
        item = {
            "canonical_part_id": canon_id,
            "primary_part_number": primary_num,
            "part_name": p.get("part_name") or primary_num,
            "synonyms": list(dict.fromkeys([
                *(p.get("alternative_numbers") or []),
                p.get("part_name") or "",
            ])),
            "source_parts": [],
            "sources": _ensure_sources(p.get("sources")),
            "confidence": 0.8,
            "rationale": "LLM-consolidated",
        }
        buckets[cat].append(item)

    categories_payload: List[Dict[str, Any]] = []
    for cat in core_categories:
        if buckets[cat]:
            categories_payload.append({
                "category": cat,
                "key": _slugify(cat),
                "items": buckets[cat],
                "summary": {"count": len(buckets[cat])},
            })

    other_bucket = {
        "category": "Other",
        "key": "other",
        "items": buckets[OTHERS_CATEGORY],
        "summary": {"count": len(buckets[OTHERS_CATEGORY])},
    }

    return {
        "categories": categories_payload,
        "other": other_bucket,
        "meta": {
            "supplier_name": supplier_name,
            "provided_categories": core_categories,
        },
        "version": 1,
    }


def _build_prompt_for_llm(kg_data: Dict[str, Any], supplier_name: Optional[str], allowed_categories: List[str]) -> str:
    categories_list = ", ".join(allowed_categories + [OTHERS_CATEGORY])
    supplier_context = f"Supplier: {supplier_name}\n" if supplier_name else ""
    return f"""
You are an expert technical analyst extracting parts information from a supplier knowledge graph.
{supplier_context}
STRICT CATEGORY RULES:
- You MUST assign each part to exactly one of: [{categories_list}]. Never invent categories.
- If you are unsure or category is not in the provided set, use "{OTHERS_CATEGORY}".

DEDUPLICATION & CONSOLIDATION RULES:
- Multiple nodes can describe the same physical part (different names/numbers). Merge them into a single part entry.
- Use the most specific, informative part name as part_name.
- Include all alternative identifiers in alternative_numbers.
- Always include precise source references: exact filename and chunk_id.

VOLUME PRICING RULES (CRITICAL):
- volume_pricing.volume MUST be a quantity descriptor (e.g., "70,000 units/year", "500 pcs", "100 sets").
- NEVER put years or date ranges in volume (e.g., "2029-2032" is INVALID for volume).
- price_per_unit MUST be a numeric price per unit (EUR). If unavailable, use null.
- discount_percentage is optional and numeric; use null if not applicable.

ALSO EXTRACT the system-level object: overall_system, representing the complete solution or assembly relevant to the parts in scope (domain-agnostic),
including consolidated specs/pricing if present and sources.

Return ONLY valid JSON with this exact structure (no markdown, no prose):
{{
  "parts": [
    {{
      "part_number": "primary part number or identifier",
      "alternative_numbers": ["other identifiers or part numbers"],
      "part_name": "concise human-readable name",
      "category": "one of {categories_list}",
      "description": "brief description",
      "technical_specifications": {{
        "dimensions": "",
        "weight": "",
        "material": "",
        "torque_capacity": "",
        "operating_temperature": "",
        "other_specs": []
      }},
      "pricing": {{
        "unit_price": null,
        "currency": "EUR",
        "volume_pricing": [{{"volume": "quantity only (e.g., 70000 units/year)", "price_per_unit": null, "discount_percentage": null}}],
        "cost_breakdown": [{{"component": "", "amount": null, "percentage": null}}]
      }},
      "certifications": [{{"standard": "", "description": "", "validity": ""}}],
      "capacity": {{
        "production_capacity": "",
        "current_utilization": "",
        "lead_time": "",
        "min_order_quantity": ""
      }},
      "sources": [{{"filename": "file.ext", "chunk_id": "page_X"}}]
    }}
  ],
  "overall_system": {{
    "title": "",
    "description": "",
    "components": [],
    "technical_specifications": {{
      "torque_capacity": "",
      "weight": "",
      "dimensions": "",
      "other_specs": []
    }},
    "pricing_summary": {{
      "unit_price": null,
      "currency": "EUR",
      "volume_pricing": [{{"volume": "quantity only (e.g., 70000 units/year)", "price_per_unit": null}}],
      "notes": ""
    }},
    "sources": [{{"filename": "file.ext", "chunk_id": "page_X"}}]
  }},
  "parts_summary": {{
    "total_parts_count": 0,
    "categories": []
  }}
}}

Knowledge Graph (part-focused subset):
{json.dumps(kg_data, ensure_ascii=False)}
"""

def _normalize_text(text: str) -> str:
    """
    Normalize free text for deterministic comparisons.
    - Uppercase
    - Strip surrounding whitespace
    - Collapse internal whitespace and standardize delimiters
    """
    if text is None:
        return ""
    cleaned = re.sub(r"\s+", " ", str(text)).strip().upper()
    return cleaned


def _normalize_part_number_key(text: str) -> str:
    """
    Normalize part numbers for matching keys by removing non-alphanumerics.
    Keep only A–Z and 0–9 to generate a compact matching key.
    """
    if not text:
        return ""
    return re.sub(r"[^A-Z0-9]", "", _normalize_text(text))


def _pn_prefix_key(norm_pn: str, length: int = 8) -> str:
    """
    Create a blocking key using the first N alphanumeric characters of a
    normalized part number.
    """
    return norm_pn[:max(0, length)] if norm_pn else ""


def _sequence_ratio(a: str, b: str) -> float:
    """
    Deterministic fuzzy similarity using difflib.SequenceMatcher.
    Returns a value in [0, 1].
    """
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _extract_part_nodes(kg_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract Part nodes from the KG and derive basic attributes used for
    canonicalization.
    Returns a list of dictionaries with:
      - id, raw_id_token, name, part_number, aliases, category, metadata
    """
    nodes: List[Dict[str, Any]] = kg_data.get("nodes", [])
    part_nodes: List[Dict[str, Any]] = []
    for node in nodes:
        if node.get("type") != "Part":
            continue
        node_id = node.get("id", "")
        props = node.get("properties", {}) or {}
        metadata = node.get("metadata", []) or []

        # Derive candidate fields
        name = props.get("name") or props.get("part_name") or ""
        part_number = props.get("part_number") or ""
        category = props.get("part_type") or ""

        # Token from ID after 'part:'
        raw_id_token = node_id.split(":", 1)[-1] if ":" in node_id else node_id

        # Build alias set: include id token, explicit part_number, and name
        aliases: Set[str] = set()
        for candidate in [raw_id_token, part_number, name]:
            norm_candidate = _normalize_text(candidate)
            if norm_candidate:
                aliases.add(norm_candidate)

        part_nodes.append(
            {
                "id": node_id,
                "raw_id_token": raw_id_token,
                "name": _normalize_text(name),
                "part_number": _normalize_text(part_number),
                "aliases": sorted(aliases),
                "category": _normalize_text(category),
                "metadata": metadata,
                "_node": node,
            }
        )
    return part_nodes


def _build_symbolic_blocks(parts: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    """
    Build simple blocking groups using normalized part number prefixes.
    If no part number is present, fall back to raw ID token prefix.
    """
    blocks: Dict[str, List[int]] = {}
    for idx, item in enumerate(parts):
        # Choose the best PN-like alias for blocking
        all_aliases = item.get("aliases", [])
        pn_keys = [_normalize_part_number_key(a) for a in all_aliases]
        pn_keys = [k for k in pn_keys if k]
        primary_key = pn_keys[0] if pn_keys else _normalize_part_number_key(item.get("raw_id_token", ""))
        block_key = _pn_prefix_key(primary_key, length=8)
        if not block_key:
            block_key = f"NO_KEY::{idx}"
        blocks.setdefault(block_key, []).append(idx)
    return blocks


def _score_pair(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """
    Compute a deterministic similarity score in [0, 1] using:
    - Exact normalized PN key equality (strong signal)
    - Fuzzy PN similarity
    - Name fuzzy similarity
    Conservative weights to avoid over-merging.
    """
    # PN keys
    a_pn_keys = [_normalize_part_number_key(x) for x in a.get("aliases", [])]
    b_pn_keys = [_normalize_part_number_key(x) for x in b.get("aliases", [])]
    a_pn_keys = [k for k in a_pn_keys if k]
    b_pn_keys = [k for k in b_pn_keys if k]

    exact_signal = 0.0
    if a_pn_keys and b_pn_keys and (set(a_pn_keys) & set(b_pn_keys)):
        exact_signal = 0.8  # strong signal for exact PN match

    # Fuzzy PN: compare best-to-best
    pn_fuzzy = 0.0
    if a_pn_keys and b_pn_keys:
        best = 0.0
        for ak in a_pn_keys:
            for bk in b_pn_keys:
                best = max(best, _sequence_ratio(ak, bk))
        pn_fuzzy = best

    # Name fuzzy
    name_fuzzy = _sequence_ratio(a.get("name", ""), b.get("name", ""))

    # Combine with conservative cap
    score = exact_signal + 0.35 * pn_fuzzy + 0.2 * name_fuzzy
    return max(0.0, min(1.0, score))


def _union_find_build(n: int) -> Tuple[List[int], List[int]]:
    parent = list(range(n))
    rank = [0] * n
    return parent, rank


def _union_find_find(parent: List[int], i: int) -> int:
    if parent[i] != i:
        parent[i] = _union_find_find(parent, parent[i])
    return parent[i]


def _union_find_union(parent: List[int], rank: List[int], x: int, y: int) -> None:
    rx = _union_find_find(parent, x)
    ry = _union_find_find(parent, y)
    if rx == ry:
        return
    if rank[rx] < rank[ry]:
        parent[rx] = ry
    elif rank[rx] > rank[ry]:
        parent[ry] = rx
    else:
        parent[ry] = rx
        rank[rx] += 1


def _cluster_parts(parts: List[Dict[str, Any]], blocks: Dict[str, List[int]], merge_threshold: float = 0.75) -> Dict[int, List[int]]:
    """
    Within each block, connect pairs with score >= threshold, then return
    connected components as clusters: root_index -> [member_indices].
    """
    n = len(parts)
    parent, rank = _union_find_build(n)

    for _, indices in blocks.items():
        # pairwise within block
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a_idx = indices[i]
                b_idx = indices[j]
                score = _score_pair(parts[a_idx], parts[b_idx])
                if score >= merge_threshold:
                    _union_find_union(parent, rank, a_idx, b_idx)

    # Aggregate by root
    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        r = _union_find_find(parent, i)
        clusters.setdefault(r, []).append(i)
    return clusters


def _choose_primary_pn(aliases: List[str]) -> str:
    """
    Choose a canonical primary PN: prefer the longest alphanumeric token.
    Fallback to the lexicographically smallest alias.
    """
    if not aliases:
        return ""
    # Rank by (length of alphanumerics, then lexicographically)
    def score(alias: str) -> Tuple[int, str]:
        return (len(_normalize_part_number_key(alias)), alias)

    sorted_aliases = sorted(aliases, key=score, reverse=True)
    return sorted_aliases[0]


def _consolidate_cluster(parts: List[Dict[str, Any]], member_indices: List[int], supplier_name: Optional[str]) -> Dict[str, Any]:
    """
    Build a canonical part record from a cluster of member items.
    Conservative consolidation: aggregate aliases and evidence; prefer
    the most informative names and categories when available.
    """
    aliases: Set[str] = set()
    names: Set[str] = set()
    categories: Set[str] = set()
    evidence: List[Dict[str, str]] = []

    for idx in member_indices:
        item = parts[idx]
        aliases.update(item.get("aliases", []))
        if item.get("name"):
            names.add(item["name"])
        if item.get("category"):
            categories.add(item["category"])
        # Collect evidence from metadata
        for md in item.get("metadata", []) or []:
            filename = md.get("filename")
            chunk_id = md.get("chunk_id")
            if filename or chunk_id:
                evidence.append({"filename": filename or "", "chunk_id": chunk_id or ""})

    synonyms = sorted(a for a in aliases if a)
    primary_pn = _choose_primary_pn(synonyms)
    part_name = max(names, key=len) if names else primary_pn or (list(aliases)[0] if aliases else "")
    category = max(categories, key=len) if categories else ""

    # Stable canonical id from supplier + primary PN + top 3 synonyms
    id_basis = f"{supplier_name or 'UNKNOWN'}|{primary_pn}|{','.join(synonyms[:3])}"
    canon_hash = hashlib.md5(id_basis.encode("utf-8")).hexdigest()[:16]
    canonical_part_id = f"canon:{_normalize_part_number_key(primary_pn) or 'PART'}:{canon_hash}"

    canonical = {
        "canonical_part_id": canonical_part_id,
        "primary_part_number": primary_pn or part_name,
        "synonyms": synonyms,
        "category": category or "",
        "part_name": part_name,
        "technical_specifications": {
            "torque_capacity_Nm": None,
            "weight_kg": None,
            "outer_diameter_mm": None,
            "other_specs": []
        },
        "pricing": {
            "unit_price_eur": None,
            "volume_pricing": [],
            "pricing_sources": []
        },
        "evidence": evidence,
        "confidence": 0.8 if primary_pn else 0.7,
    }
    return canonical


def _canonicalize_parts(kg_data: Dict[str, Any], supplier_name: Optional[str]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Deterministic canonicalization pipeline:
    - Extract part nodes
    - Build blocking groups
    - Pairwise score and union within blocks
    - Consolidate clusters into canonical parts
    Returns (canonical_parts, raw_to_canonical_map)
    """
    extracted = _extract_part_nodes(kg_data)
    if not extracted:
        return [], {}

    blocks = _build_symbolic_blocks(extracted)
    clusters = _cluster_parts(extracted, blocks, merge_threshold=0.75)

    canonical_parts: List[Dict[str, Any]] = []
    raw_to_canonical: Dict[str, str] = {}

    for root_idx, members in clusters.items():
        canonical = _consolidate_cluster(extracted, members, supplier_name)
        canonical_parts.append(canonical)
        for idx in members:
            raw_to_canonical[extracted[idx]["id"]] = canonical["canonical_part_id"]

    return canonical_parts, raw_to_canonical


def filter_part_related_data(kg_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter KG to only include part nodes and their direct connections.
    
    This significantly reduces the data sent to the LLM while preserving
    all relevant information for parts analysis.
    
    Strategy:
    1. Get all nodes with ID starting with "part:"
    2. Get all relationships involving these parts
    3. Get all nodes connected to parts (1-hop)
    4. Preserve metadata structure
    
    Args:
        kg_data: Full knowledge graph dictionary
        
    Returns:
        Filtered KG with same structure but only part-related data
    """
    nodes = kg_data.get('nodes', [])
    relationships = kg_data.get('relationships', [])
    
    # Step 1: Find all part node IDs
    part_node_ids = {n['id'] for n in nodes if n.get('id', '').startswith('part:')}
    
    if not part_node_ids:
        print("Warning: No part nodes found in the knowledge graph")
        return kg_data  # Return original if no parts found
    
    # Step 2: Find all relationships involving parts
    part_relationships = []
    connected_node_ids = set()
    
    for rel in relationships:
        source = rel.get('source')
        target = rel.get('target')
        
        # If either end connects to a part, include it
        if source in part_node_ids or target in part_node_ids:
            part_relationships.append(rel)
            # Track the connected nodes
            connected_node_ids.add(source)
            connected_node_ids.add(target)
    
    # Step 3: Get all relevant nodes (parts + connected nodes)
    relevant_nodes = [
        n for n in nodes 
        if n['id'] in connected_node_ids
    ]
    
    # Step 4: Return filtered KG with same structure
    return {
        'nodes': relevant_nodes,
        'relationships': part_relationships,
        'metadata': kg_data.get('metadata', {}),
        'export_metadata': kg_data.get('export_metadata', {})
    }


def analyze_parts_structured(
    kg_json_path: str,
    supplier_name: Optional[str] = None,
    save_to_file: bool = False,
    output_path: Optional[str] = None,
    use_part_filter: bool = True,
    canonicalize: bool = True,
    model_name: Optional[str] = None,
    core_categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generic parts analysis:
    - Uses provided core categories (or defaults) with a strict mapping plus "Others"
    - LLM consolidates duplicate nodes per part and extracts a domain-agnostic overall system
    - Returns a concentrated_parts structure used directly by the UI
    """
    print(f"Loading KG from: {kg_json_path}")

    # Load KG JSON
    with open(kg_json_path, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)

    allowed_categories = list(core_categories) if core_categories else list(CORE_PART_CATEGORIES_DEFAULT)

    # Optional filtering for prompt efficiency
    original_size = len(json.dumps(kg_data))
    filtered = filter_part_related_data(kg_data) if use_part_filter else kg_data
    if use_part_filter:
        filtered_size = len(json.dumps(filtered))
        reduction_pct = (1 - filtered_size / max(original_size, 1)) * 100
        print(f"Applied part filtering: size reduced by {reduction_pct:.1f}%")

    # Build prompt and call LLM
    base_prompt = _build_prompt_for_llm(filtered, supplier_name, allowed_categories)
    prompt = format_prompt_with_business_context(base_prompt)
    selected_model = model_name or "gpt-4.1"
    llm = create_llm(model_name=selected_model, temperature=0.0)
    print(f"Sending to {selected_model} for parts analysis (~{len(prompt):,} chars)...")

    raw_result: Dict[str, Any] = {}
    try:
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, 'content') else str(response)
        if isinstance(text, list):
            text = text[0] if text else ""
        text = _strip_code_fences(str(text))
        raw_result = json.loads(text)
    except Exception as e:
        print(f"Failed to parse LLM response: {e}")
        raw_result = {}

    # Normalize parts
    parts_in = raw_result.get('parts') if isinstance(raw_result.get('parts'), list) else []
    normalized_parts: List[Dict[str, Any]] = []
    for p in parts_in:
        if not isinstance(p, dict):
            continue
        cat = _normalize_category(p.get('category'), allowed_categories)
        # Normalize technical specs to strings
        tech_specs = _normalize_tech_specs_to_strings(p.get("technical_specifications"))
        # Normalize pricing to avoid None currency artifacts downstream
        pricing_in = p.get("pricing") if isinstance(p.get("pricing"), dict) else {}
        unit_price_val = _coerce_price_to_float(pricing_in.get("unit_price", None))
        pricing = {
            "unit_price": unit_price_val,
            "currency": pricing_in.get("currency", "EUR") or "EUR",
            "volume_pricing": _normalize_volume_pricing_list(pricing_in.get("volume_pricing", []) or []),
            "cost_breakdown": pricing_in.get("cost_breakdown", []) or [],
        }
        normalized_parts.append({
            "part_number": p.get("part_number") or "",
            "alternative_numbers": p.get("alternative_numbers") or [],
            "part_name": p.get("part_name") or (p.get("part_number") or ""),
            "category": cat,
            "description": p.get("description") or "",
            "technical_specifications": tech_specs,
            "pricing": pricing,
            "certifications": p.get("certifications") or [],
            "capacity": p.get("capacity") or {
                "production_capacity": "",
                "current_utilization": "",
                "lead_time": "",
                "min_order_quantity": "",
            },
            "sources": _ensure_sources(p.get("sources")),
        })

    # Overall system (generic). Backward compatibility for 'overall_clutch_system'
    overall_in = None
    if isinstance(raw_result.get('overall_system'), dict):
        overall_in = raw_result.get('overall_system')
    elif isinstance(raw_result.get('overall_clutch_system'), dict):
        overall_in = raw_result.get('overall_clutch_system')
    # Default empty structure
    overall = overall_in if isinstance(overall_in, dict) else {
        "title": "",
        "description": "",
        "components": [],
        "technical_specifications": {"torque_capacity": "", "weight": "", "dimensions": "", "other_specs": []},
        "pricing_summary": {"unit_price": None, "currency": "EUR", "volume_pricing": [], "notes": ""},
        "sources": [],
    }

    # Sanitize overall pricing summary as well
    if isinstance(overall, dict):
        ps = overall.get('pricing_summary') if isinstance(overall.get('pricing_summary'), dict) else {}
        ps_unit = _coerce_price_to_float(ps.get('unit_price'))
        ps_vol = _normalize_volume_pricing_list(ps.get('volume_pricing', []) or [])
        overall['pricing_summary'] = {
            'unit_price': ps_unit,
            'currency': ps.get('currency', 'EUR') or 'EUR',
            'volume_pricing': ps_vol,
            'notes': ps.get('notes', ''),
        }

    # Build summary and concentrated view
    categories_present = sorted(list({p.get('category', OTHERS_CATEGORY) for p in normalized_parts}))
    parts_summary = {
        "total_parts_count": len(normalized_parts),
        "categories": categories_present,
    }
    concentrated = _build_concentrated_from_parts(normalized_parts, supplier_name or "Unknown Supplier", allowed_categories)

    result: Dict[str, Any] = {
        "parts": normalized_parts,
        # New generic key
        "overall_system": overall,
        # Backward compatible alias for downstream consumers
        "overall_clutch_system": overall,
        "parts_summary": parts_summary,
        "concentrated_parts": concentrated,
        "supplier_name": supplier_name or "Unknown Supplier",
    }

    if save_to_file and output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved parts analysis to: {output_path}")

    return result


def _invert_raw_to_canonical(raw_to_canonical: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Build canonical_part_id -> [raw_node_ids] mapping from raw_to_canonical map.
    """
    canonical_to_raw: Dict[str, List[str]] = {}
    for raw_id, canon_id in (raw_to_canonical or {}).items():
        canonical_to_raw.setdefault(canon_id, []).append(raw_id)
    return canonical_to_raw


def _lookup_raw_nodes(kg_data: Dict[str, Any], raw_ids: Set[str]) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Build a minimal lookup for raw node properties: number and name.
    """
    lookup: Dict[str, Dict[str, Optional[str]]] = {}
    for node in kg_data.get('nodes', []) or []:
        nid = node.get('id')
        if nid in raw_ids:
            props = node.get('properties', {}) or {}
            lookup[nid] = {
                'raw_part_number': props.get('part_number') or props.get('name') or None,
                'raw_part_name': props.get('name') or props.get('part_name') or None,
            }
    return lookup


def _build_concentration_input(
    canonical_parts: List[Dict[str, Any]],
    raw_to_canonical: Dict[str, str],
    kg_data: Dict[str, Any],
    supplier_name: str,
) -> Dict[str, Any]:
    """
    Create a compact input payload for the LLM grouping prompt.
    """
    canonical_to_raw = _invert_raw_to_canonical(raw_to_canonical)
    all_raw_ids: Set[str] = set()
    for canon_id, raw_ids in canonical_to_raw.items():
        for r in raw_ids:
            all_raw_ids.add(r)
    raw_lookup = _lookup_raw_nodes(kg_data, all_raw_ids)

    slim_parts: List[Dict[str, Any]] = []
    for cp in canonical_parts:
        slim_parts.append({
            'canonical_part_id': cp.get('canonical_part_id'),
            'primary_part_number': cp.get('primary_part_number'),
            'part_name': cp.get('part_name'),
            'synonyms': (cp.get('synonyms') or [])[:8],
            'category': cp.get('category') or '',
            'evidence': (cp.get('evidence') or [])[:2],
        })

    # Build source_parts per canonical by joining canonical_to_raw with raw_lookup
    canonical_sources: Dict[str, List[Dict[str, Optional[str]]]] = {}
    for cp in canonical_parts:
        cid = cp.get('canonical_part_id')
        source_list: List[Dict[str, Optional[str]]] = []
        for rid in canonical_to_raw.get(cid, []) if cid in canonical_to_raw else []:
            raw = raw_lookup.get(rid) or {}
            source_list.append({
                'raw_node_id': rid,
                'raw_part_number': raw.get('raw_part_number'),
                'raw_part_name': raw.get('raw_part_name'),
            })
        canonical_sources[cid] = source_list

    return {
        'supplier_name': supplier_name,
        'canonical_parts': slim_parts,
        'canonical_sources': canonical_sources,
    }


def _concentrate_canonical_parts_with_llm(
    canonical_parts: List[Dict[str, Any]],
    raw_to_canonical: Dict[str, str],
    kg_data: Dict[str, Any],
    supplier_name: str,
    core_categories: Optional[List[str]],
    model_name: Optional[str],
) -> Dict[str, Any]:
    """
    Use LLM to group canonical parts into provided categories (+ Other) or
    propose categories if none provided. Always return JSON with a stable shape.
    """
    concentration_input = _build_concentration_input(
        canonical_parts=canonical_parts,
        raw_to_canonical=raw_to_canonical,
        kg_data=kg_data,
        supplier_name=supplier_name,
    )

    grouping_llm = create_llm(model_name=model_name or "gemini-2.5-pro", temperature=0.0)

    if core_categories and len(core_categories) > 0:
        cats_list = ", ".join(core_categories)
        grouping_prompt = f"""
You are grouping canonical parts for supplier: {supplier_name}.
You MUST assign each canonical part to exactly ONE of these categories or to "Other": [{cats_list}].
Do not invent new categories. Use "Other" only if none apply.

Return ONLY valid JSON with this exact structure:
{{
  "categories": [
    {{
      "category": "<one of {cats_list}>",
      "key": "<slug, lowercase with underscores, e.g., clutch_disk>",
      "items": [
        {{
          "canonical_part_id": "...",
          "primary_part_number": "...",
          "part_name": "...",
          "synonyms": ["..."],
          "source_parts": [{{"raw_node_id":"...","raw_part_number":"...","raw_part_name":"..."}}],
          "sources": [{{"filename":"...","chunk_id":"..."}}],
          "confidence": 0.0,
          "rationale": "<=20 words"
        }}
      ],
      "summary": {{"count": <number>}}
    }}
  ],
  "other": {{
    "category": "Other",
    "key": "other",
    "items": [ ... same item schema ... ],
    "summary": {{"count": <number>}}
  }},
  "meta": {{
    "supplier_name": "{supplier_name}",
    "input_counts": {{"canonical_parts": {len(canonical_parts)}}},
    "provided_categories": {json.dumps(core_categories)}
  }},
  "version": 1
}}

Canonical Parts Input:
{json.dumps(concentration_input, ensure_ascii=False)}
"""
    else:
        grouping_prompt = f"""
You are grouping canonical parts for supplier: {supplier_name}.
Propose 3-6 high-level categories that best capture these parts.
For each category include a stable slug key (lowercase with underscores), and assign every part to exactly one category.

Return ONLY valid JSON with this exact structure:
{{
  "categories": [
    {{
      "category": "<name>",
      "key": "<slug>",
      "items": [
        {{
          "canonical_part_id": "...",
          "primary_part_number": "...",
          "part_name": "...",
          "synonyms": ["..."],
          "source_parts": [{{"raw_node_id":"...","raw_part_number":"...","raw_part_name":"..."}}],
          "sources": [{{"filename":"...","chunk_id":"..."}}],
          "confidence": 0.0,
          "rationale": "<=20 words"
        }}
      ],
      "summary": {{"count": <number>}}
    }}
  ],
  "meta": {{
    "supplier_name": "{supplier_name}",
    "input_counts": {{"canonical_parts": {len(canonical_parts)}}},
    "provided_categories": []
  }},
  "version": 1
}}

Canonical Parts Input:
{json.dumps(concentration_input, ensure_ascii=False)}
"""

    # Invoke LLM and parse JSON
    raw = grouping_llm.invoke(format_prompt_with_business_context(grouping_prompt))
    text = raw.content if hasattr(raw, 'content') else str(raw)
    if isinstance(text, list):
        text = text[0] if text else ""
    text = str(text).strip()
    if text.startswith('```json'):
        text = text[7:]
    elif text.startswith('```'):
        text = text[3:]
    if text.endswith('```'):
        text = text[:-3]

    try:
        parsed = json.loads(text)
        # Validate minimal shape
        if 'categories' not in parsed:
            raise ValueError('Missing categories in LLM output')

        # Enrich each item with sources from canonical evidence and raw_lookup if absent
        # Join back using concentration_input canonical_sources
        canon_sources = concentration_input.get('canonical_sources', {})
        # Build quick evidence map from canonical_parts
        evidence_map: Dict[str, List[Dict[str, str]]] = {
            cp.get('canonical_part_id'): (cp.get('evidence') or []) for cp in canonical_parts
        }
        for cat in parsed.get('categories', []) or []:
            for item in cat.get('items', []) or []:
                cid = item.get('canonical_part_id')
                if cid:
                    if 'source_parts' not in item or not item.get('source_parts'):
                        item['source_parts'] = canon_sources.get(cid, [])
                    if 'sources' not in item or not item.get('sources'):
                        item['sources'] = evidence_map.get(cid, [])
        # If strict categories mode, ensure only allowed categories present
        if core_categories:
            allowed = set(core_categories)
            # Move unknown categories to 'other'
            other_bucket = {'category': 'Other', 'key': 'other', 'items': [], 'summary': {'count': 0}}
            filtered_categories = []
            for cat in parsed.get('categories', []) or []:
                if cat.get('category') in allowed:
                    filtered_categories.append(cat)
                else:
                    other_bucket['items'].extend(cat.get('items', []) or [])
            other_bucket['summary']['count'] = len(other_bucket['items'])
            parsed['categories'] = filtered_categories
            parsed['other'] = other_bucket
            parsed.setdefault('meta', {})['provided_categories'] = core_categories
        # Ensure counts
        for cat in parsed.get('categories', []) or []:
            cat['summary'] = {'count': len(cat.get('items', []) or [])}
        return parsed
    except Exception as e:
        print(f"Failed to group with LLM: {e}")
        # Fallback: assign all to Other
        fallback_items = []
        canon_sources = concentration_input.get('canonical_sources', {})
        evidence_map: Dict[str, List[Dict[str, str]]] = {
            cp.get('canonical_part_id'): (cp.get('evidence') or []) for cp in canonical_parts
        }
        for cp in canonical_parts:
            cid = cp.get('canonical_part_id')
            fallback_items.append({
                'canonical_part_id': cid,
                'primary_part_number': cp.get('primary_part_number'),
                'part_name': cp.get('part_name'),
                'synonyms': (cp.get('synonyms') or [])[:8],
                'source_parts': canon_sources.get(cid, []),
                'sources': evidence_map.get(cid, []),
                'confidence': 0.5,
                'rationale': 'Fallback grouping',
            })
        return {
            'categories': [],
            'other': {'category': 'Other', 'key': 'other', 'items': fallback_items, 'summary': {'count': len(fallback_items)}},
            'meta': {
                'supplier_name': supplier_name,
                'input_counts': {'canonical_parts': len(canonical_parts)},
                'provided_categories': core_categories or [],
            },
            'version': 1,
        }


def main():
    """Test the parts analyzer with sample data"""
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Analyze parts from KG')
    parser.add_argument('kg_path', help='Path to KG JSON file')
    parser.add_argument('--supplier', help='Supplier name', default=None)
    parser.add_argument('--output', help='Output JSON file path', default=None)
    
    args = parser.parse_args()
    
    # Generate output path if not provided
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        supplier_suffix = f"_{args.supplier.replace(' ', '_')}" if args.supplier else ""
        args.output = f"output/llm_analysis/parts_analysis_{timestamp}{supplier_suffix}.json"
    
    # Run analysis
    result = analyze_parts_structured(
        kg_json_path=args.kg_path,
        supplier_name=args.supplier,
        save_to_file=True,
        output_path=args.output
    )
    
    # Print summary
    print("\n=== Parts Analysis Summary ===")
    print(f"Supplier: {result.get('supplier_name', 'Unknown')}")
    print(f"Total Parts: {result.get('parts_summary', {}).get('total_parts_count', 0)}")
    print(f"Categories: {', '.join(result.get('parts_summary', {}).get('categories', []))}")
    print(f"Part Families: {len(result.get('part_families', []))}")
    
    if result.get('parts'):
        print("\n=== Sample Parts ===")
        for part in result['parts'][:3]:
            print(f"- {part.get('part_number', 'N/A')}: {part.get('part_name', 'N/A')}")
            if part.get('pricing', {}).get('unit_price'):
                print(f"  Price: {part['pricing']['currency']} {part['pricing']['unit_price']}")


if __name__ == "__main__":
    main()