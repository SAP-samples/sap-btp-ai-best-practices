"""
Verification API router.

Provides endpoints to compare attributes between two documents using
flexible comparators (name, id, text, or auto heuristic).

This integrates the logic from `temp/kyc_verifier_api.py` into the main API
as a router mounted under `/api/verification`.
"""

from typing import Any, Dict, List, Optional, Tuple, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

import unicodedata
import re


# Router instance for verification endpoints
router = APIRouter()


# --------------------------
# Utility normalization helpers
# --------------------------

def strip_accents(text: str) -> str:
    """Remove diacritics from a string while preserving base characters."""
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )


def canonical_text(val: Any) -> str:
    """Lowercase, remove accents, and collapse whitespace for text comparison."""
    if val is None:
        return ""
    value_as_text = str(val)
    value_as_text = strip_accents(value_as_text).lower()
    value_as_text = " ".join(value_as_text.split())
    return value_as_text


def canonical_id(val: Any) -> str:
    """Uppercase alphanumeric-only form for identifier comparison (e.g., RFC, CURP)."""
    if val is None:
        return ""
    value_as_text = str(val)
    value_as_text = strip_accents(value_as_text)
    value_as_text = "".join(ch for ch in value_as_text if ch.isalnum())
    return value_as_text.upper()


def tokenize_name(val: Any) -> List[str]:
    """Tokenize names or razon social fields into lowercase alphanumeric tokens."""
    normalized = canonical_text(val)
    return re.findall(r"[a-z0-9]+", normalized)


def tokenize_text(val: Any) -> List[str]:
    """Generic tokenization for fuzzy comparison."""
    normalized = canonical_text(val)
    return re.findall(r"[a-z0-9]+", normalized)


def jaccard_similarity(a_tokens: List[str], b_tokens: List[str]) -> float:
    """Jaccard similarity between two token sets in [0,1]."""
    set_a = set(a_tokens)
    set_b = set(b_tokens)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def compare_name(value_a: Any, value_b: Any) -> bool:
    """Compare two names by checking equality of token sets (order-insensitive)."""
    tokens_a = set(tokenize_name(value_a))
    tokens_b = set(tokenize_name(value_b))
    return (len(tokens_a) > 0) and (tokens_a == tokens_b)


def compare_text(value_a: Any, value_b: Any) -> bool:
    """Compare two text values after canonical normalization (order preserved)."""
    return canonical_text(value_a) == canonical_text(value_b)


def compare_id(value_a: Any, value_b: Any) -> bool:
    """Compare two identifiers after alphanumeric-only, case-insensitive normalization."""
    return canonical_id(value_a) == canonical_id(value_b)


def compare_fuzzy(value_a: Any, value_b: Any) -> float:
    """Return token Jaccard similarity in [0,1] after normalization."""
    return jaccard_similarity(tokenize_text(value_a), tokenize_text(value_b))


def compare_contains(value_a: Any, value_b: Any) -> bool:
    """Return True if either normalized value is a substring of the other."""
    a = canonical_text(value_a)
    b = canonical_text(value_b)
    if not a or not b:
        return False
    return a in b or b in a


# --------------------------
# Number comparison helpers
# --------------------------
def _extract_integers(val: Any) -> List[int]:
    """Extract all integer numbers from a value string.

    Examples:
    - '30' -> [30]
    - '30 días' -> [30]
    - '30 (días)' -> [30]
    - '30-60 días' -> [30, 60]
    """
    text = canonical_text(val)
    return [int(x) for x in re.findall(r"\d+", text)]


def compare_number(value_a: Any, value_b: Any) -> bool:
    """Compare values by their first extracted integer number.

    Returns True if both values contain at least one integer and the first
    extracted integer matches. Otherwise returns False.
    """
    nums_a = _extract_integers(value_a)
    nums_b = _extract_integers(value_b)
    if not nums_a or not nums_b:
        return False
    return nums_a[0] == nums_b[0]

# --------------------------
# Address comparison helpers
# --------------------------
# Expanded stopwords to remove non-informative address label tokens and fillers
ADDRESS_STOPWORDS = {
    # generic/address fillers
    "cp", "c", "p", "entre", "y", "no", "num", "numero", "n", "int", "ext", "mz", "lt",
    "de", "del", "la", "las", "los", "el", "al",
    # address type labels
    "calle", "col", "colonia", "col", "municipio", "demarcacion", "territorial", "delegacion",
    "alcaldia", "estado", "entidad", "federativa", "pais", "provincia",
    # csf label words
    "codigo", "postal", "tipo", "vialidad", "nombre",
    # country common token (kept sometimes in values; we drop to avoid penalizing when missing)
    "mexico",
}

# Patterns to strip CSF/CGV style labels like "Código Postal:", "Tipo de Vialidad:", etc.
ADDRESS_LABEL_PATTERNS = [
    r"\bcodigo\s+postal\s*:\s*",
    r"\bc\.?\s*p\.?\s*:\s*",  # C.P.:
    r"\btipo\s+de\s+vialidad\s*:\s*",
    r"\bnombre\s+de\s+la\s+vialidad\s*:\s*",
    r"\bnombre\s+de\s+la\s+colonia\s*:\s*",
    r"\bnombre\s+del\s+municipio\s+o\s+demarcacion\s+territorial\s*:\s*",
    r"\bnombre\s+de\s+la\s+entidad\s+federativa\s*:\s*",
    r"\bmunicipio\s*:\s*",
    r"\bestado\s*:\s*",
    r"\bpais\s*:\s*",
    r"\bcolonia\s*:\s*",
    r"\bentre\s+calle\s*:\s*",
    r"\by\s+calle\s*:\s*",
    r"\bnumero\s+exterior\s*:\s*",
    r"\bno\.?\s*:\s*",
]


def _address_core_text(val: Any) -> str:
    """Normalize address text for comparison.

    - Lowercase, strip accents, collapse whitespace
    - Remove trailing 'entre ...' clauses which are non-essential
    - Normalize common variants (e.g., 'avenida'/'av.' -> 'av')
    - Normalize 'c.p.' forms to 'cp'
    """
    t = canonical_text(val)
    # Remove common CSF/CGV label prefixes, keeping only the values
    for pat in ADDRESS_LABEL_PATTERNS:
        t = re.sub(pat, "", t)
    # Drop trailing 'entre ...' and anything that follows
    t = re.sub(r"\bentre\b.*$", "", t)
    # Normalize avenida/av. to av
    t = re.sub(r"\bavenida\b", "av", t)
    t = re.sub(r"\bav\.\b", "av", t)
    # Normalize c.p. to cp
    t = re.sub(r"\bc\.??\s*p\.??\b", "cp", t)
    return t


def _tokenize_address(val: Any) -> List[str]:
    """Tokenize address and remove non-informative tokens."""
    t = _address_core_text(val)
    tokens = re.findall(r"[a-z0-9]+", t)
    return [tok for tok in tokens if tok not in ADDRESS_STOPWORDS]


def _extract_postal_code(val: Any) -> str:
    """Extract a 5-digit postal code if present, else empty string."""
    t = canonical_text(val)
    m = re.search(r"\b\d{5}\b", t)
    return m.group(0) if m else ""


def compare_address(value_a: Any, value_b: Any, threshold: float = 0.6) -> bool:
    """Compare addresses using structured heuristics + token similarity.

    Rules:
    - If both have a 5-digit CP and they differ -> no match
    - Otherwise compute a composite score:
      - Base Jaccard similarity over filtered tokens (0..1)
      - Bonus for matching street number (if both present)
      - Bonus for matching municipality/city token overlap (>0)
      - Bonus for matching state token overlap (>0)
    - Consider a match when composite score >= threshold.
    """
    cp_a = _extract_postal_code(value_a)
    cp_b = _extract_postal_code(value_b)
    if cp_a and cp_b and cp_a != cp_b:
        return False

    tokens_a = _tokenize_address(value_a)
    tokens_b = _tokenize_address(value_b)
    base = jaccard_similarity(tokens_a, tokens_b)

    # Extract street numbers (first integer not equal to CP)
    def first_house_number(tokens: List[str], cp: str) -> str:
        for tok in tokens:
            if tok.isdigit() and tok != cp and len(tok) <= 6:
                return tok
        return ""

    num_a = first_house_number(tokens_a, cp_a)
    num_b = first_house_number(tokens_b, cp_b)

    bonus = 0.0
    if num_a and num_b and num_a == num_b:
        bonus += 0.15

    # Very light geographic bonuses based on overlap of likely city/state tokens
    # Heuristic: tokens near CP or following municipio/estado labels were already normalized; rely on overlap
    overlap = len(set(tokens_a) & set(tokens_b))
    if overlap >= 1:
        # If we already have a strong base, small bonus; otherwise slightly larger to help structured vs free-text
        bonus += 0.10 if base >= 0.4 else 0.20

    score = min(1.0, base + bonus)
    return score >= threshold

def normalize_currency(value: Any) -> str:
    """
    Normalize currency representations to a canonical code.

    Maps common Spanish/English descriptions and symbols to ISO-like codes.
    Currently supports: MXN, USD, EUR.
    """
    text = canonical_text(value)
    if not text:
        return ""

    # MXN synonyms
    if (
        "mxn" in text
        or "mxp" in text
        or "peso" in text
        or "pesos" in text
        or text == "$"
        or "moneda nacional" in text
        or text.replace(" ", "") in {"mn", "m.n", "m.n."}
    ):
        return "MXN"

    # USD synonyms
    if (
        "usd" in text
        or "dolar" in text
        or "dolares" in text
        or "dólar" in text
        or "dlls" in text
        or "us$" in text
    ):
        return "USD"

    # EUR synonyms
    if "eur" in text or "euro" in text or "euros" in text:
        return "EUR"

    # Fallback: alphanumeric upper
    return canonical_id(text)


def compare_currency(value_a: Any, value_b: Any) -> bool:
    """Compare two currency values by their normalized currency codes."""
    return normalize_currency(value_a) == normalize_currency(value_b)


def smart_compare(key_a: str, val_a: Any, val_b: Any) -> Tuple[bool, str]:
    """
    Choose comparison strategy based on the key and value characteristics.

    - If the key suggests a name field -> use name comparator
    - Else if values are mostly alphanumeric with few spaces -> use id comparator
    - Else -> use text comparator

    Returns a tuple (match_result, comparator_used).
    """
    key_lower = (key_a or "").lower()
    name_keywords = [
        "name",
        "nombre",
        "apellido",
        "razon social",
        "razón social",
        "legal name",
        "cliente",
        "customer",
    ]
    is_name_key = any(keyword in key_lower for keyword in name_keywords)

    if is_name_key:
        return compare_name(val_a, val_b), "name"

    a_text = str(val_a or "")
    b_text = str(val_b or "")
    space_count = a_text.count(" ") + b_text.count(" ")
    non_alnum = re.findall(r"[^a-zA-Z0-9]", a_text + b_text)
    if space_count <= 1 and (
        len(non_alnum) <= max(2, int(0.1 * max(1, len(a_text) + len(b_text))))
    ):
        return compare_id(val_a, val_b), "id"

    return compare_text(val_a, val_b), "text"


# --------------------------
# Request/Response models
# --------------------------

Comparator = Literal["auto", "name", "id", "text", "fuzzy", "currency", "contains", "address", "number"]


class FieldSpec(BaseModel):
    """Mapping specification from a key in doc_a to a key in doc_b with comparator."""

    to: str = Field(..., description="Key in doc_b to compare with this key from doc_a.")
    comparator: Comparator = Field(
        "auto", description="Comparison type: auto | name | id | text | fuzzy | currency | contains | address | number"
    )
    threshold: float = Field(
        0.82, ge=0.0, le=1.0, description="Match threshold for fuzzy/address comparators."
    )


class VerifyRequest(BaseModel):
    """Request payload with documents and a per-field comparison map."""

    doc_a: Dict[str, Any] = Field(
        ..., description="Extracted key-value pairs from document A."
    )
    doc_b: Dict[str, Any] = Field(
        ..., description="Extracted key-value pairs from document B."
    )
    # Map keys in A to keys in B. Either "BKey" string or {"to": "BKey", "comparator": "..."}
    key_map: Dict[str, Any] = Field(
        ..., description="Mapping from keys in doc_a to corresponding keys/specs in doc_b."
    )


class ItemResult(BaseModel):
    """Per-field verification outcome."""

    key_a: str
    key_b: str
    value_a: Any = None
    value_b: Any = None
    comparator: str
    match: bool
    reason: Optional[str] = None


class VerifyResponse(BaseModel):
    """Aggregated verification results and a summary."""

    verified: List[ItemResult]
    failed: List[ItemResult]
    summary: Dict[str, int]


# --------------------------
# Helpers
# --------------------------

def parse_field_spec(raw_spec: Any) -> FieldSpec:
    """Normalize a key_map raw spec into a FieldSpec model."""
    if isinstance(raw_spec, str):
        return FieldSpec(to=raw_spec, comparator="auto")
    if isinstance(raw_spec, dict):
        to_value = raw_spec.get("to")
        comp_value = raw_spec.get("comparator", "auto")
        thr_value = raw_spec.get("threshold", 0.82)
        if to_value is None:
            raise ValueError("key_map spec objects must include a 'to' field.")
        if comp_value not in {"auto", "name", "id", "text", "fuzzy", "currency", "contains", "address", "number"}:
            raise ValueError(f"Unsupported comparator: {comp_value!r}")
        try:
            thr_value = float(thr_value)
        except Exception:
            raise ValueError(f"Invalid threshold: {thr_value!r}")
        if not (0.0 <= thr_value <= 1.0):
            raise ValueError("threshold must be between 0.0 and 1.0")
        return FieldSpec(to=to_value, comparator=comp_value, threshold=thr_value) 
    raise ValueError(
        "key_map values must be either a string or an object with {to, comparator}."
    )


# --------------------------
# API endpoints
# --------------------------


@router.post("/verify", response_model=VerifyResponse)
def verify(req: VerifyRequest) -> VerifyResponse:
    """Compare attributes between doc_a and doc_b according to the provided key_map."""
    verified: List[ItemResult] = []
    failed: List[ItemResult] = []

    for key_a, raw_spec in req.key_map.items():
        try:
            spec = parse_field_spec(raw_spec)
        except Exception as exc:
            failed.append(
                ItemResult(
                    key_a=key_a,
                    key_b="",
                    value_a=None,
                    value_b=None,
                    comparator="n/a",
                    match=False,
                    reason=f"invalid key_map spec: {exc}",
                )
            )
            continue

        value_a = req.doc_a.get(key_a, None)
        value_b = req.doc_b.get(spec.to, None)

        if value_a is None or value_b is None:
            failed.append(
                ItemResult(
                    key_a=key_a,
                    key_b=spec.to,
                    value_a=value_a,
                    value_b=value_b,
                    comparator=spec.comparator,
                    match=False,
                    reason="missing value in one or both documents",
                )
            )
            continue

        if spec.comparator == "name":
            match, comp_used = compare_name(value_a, value_b), "name"
        elif spec.comparator == "id":
            match, comp_used = compare_id(value_a, value_b), "id"
        elif spec.comparator == "text":
            match, comp_used = compare_text(value_a, value_b), "text"
        elif spec.comparator == "fuzzy":
            score = compare_fuzzy(value_a, value_b)
            match, comp_used = (score >= spec.threshold), "fuzzy"
        elif spec.comparator == "currency":
            match, comp_used = compare_currency(value_a, value_b), "currency"
        elif spec.comparator == "contains":
            match, comp_used = compare_contains(value_a, value_b), "contains"
        elif spec.comparator == "address":
            match, comp_used = compare_address(value_a, value_b, spec.threshold), "address"
        elif spec.comparator == "number":
            match, comp_used = compare_number(value_a, value_b), "number"
        else:
            match, comp_used = smart_compare(key_a, value_a, value_b)

        result_item = ItemResult(
            key_a=key_a,
            key_b=spec.to,
            value_a=value_a,
            value_b=value_b,
            comparator=comp_used,
            match=match,
            reason=None if match else "values differ after normalization",
        )
        (verified if match else failed).append(result_item)

    return VerifyResponse(
        verified=verified,
        failed=failed,
        summary={
            "verified": len(verified),
            "failed": len(failed),
            "total": len(req.key_map),
        },
    )


@router.get("/health")
def health() -> Dict[str, str]:
    """Simple health endpoint for the verification router."""
    return {"status": "ok", "component": "verification", "version": "0.3.0"}


