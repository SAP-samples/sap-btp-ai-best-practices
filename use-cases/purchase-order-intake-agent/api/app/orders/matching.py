"""Pure matching helpers: SAP number formats, text normalisation, customer and material ranking.

Nothing here touches the database; callers pass master-data rows (dicts with SAP column
names, as returned by ``app.ecc.master_data``). That keeps the scoring rules unit-testable
and identical for the agent tools, the validator and the UI value help.

Scoring scale (0..1):

* 1.00  exact identifier hit (material number, sales code, VAT/tax id)
* 0.99  customer-material info record (KNMT) code for this customer
* 0.95  manufacturer part number (MARA.MFRPN) printed as a code or inside the description
* 0.97  strong secondary identifier (sender e-mail domain, phone number)
* 0.80-0.95  sales code with tolerated variations (missing letter prefix, missing suffix,
        pack size confirmed) or very close name/address
* <0.80 fuzzy description / name similarity only

A candidate list is *ambiguous* when the best two scores are closer than ``AMBIGUITY_GAP``;
the agent must then flag the field instead of choosing.

ponytail: Python scoring over the full candidate list is fine for mock-sized master data;
at real ECC scale push candidate retrieval into HANA (CONTAINS ... FUZZY) and score the top-N here.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Iterable

AMBIGUITY_GAP = 0.05

# Legal-form and filler tokens that carry no identifying signal in company names.
LEGAL_FORM_TOKENS = {
    "gmbh", "co", "kg", "ag", "ltd", "limited", "pte", "pvt", "private", "inc", "corp", "corporation",
    "llc", "plc", "jsc", "sa", "srl", "bv", "nv", "company", "cong", "ty", "tnhh", "cp", "phan", "the", "and",
}
# Public mail providers: their domain says nothing about the customer.
PUBLIC_MAIL_DOMAINS = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "live.com", "icloud.com", "qq.com"}

# Litres per pack unit used when comparing pack sizes printed on documents.
_LITRES_PER_UNIT = {"ML": 0.001, "L": 1.0, "LT": 1.0, "LTR": 1.0, "LTRS": 1.0, "LITRE": 1.0, "LITER": 1.0,
                    "GL": 3.785, "GAL": 3.785, "GALLON": 3.785, "HG": 1.893, "QT": 0.946, "QL": 0.946, "QUART": 0.946}
_PACK_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(ML|LTRS|LTR|LITRE|LITER|LT|L|GALLON|GAL|GL|HG|QUART|QT|QL)(?![A-Z])")
_BARE_PACK = {"GL", "GAL", "GALLON", "QT", "QUART", "QL", "HG"}


# ---------------------------------------------------------------------------
# SAP number formats
# ---------------------------------------------------------------------------
def alpha_in(value: Any, length: int) -> str | None:
    """Convert an external number to SAP internal ALPHA format.

    Purely numeric values are left-padded with zeros to ``length``; anything else is
    upper-cased and stripped (alphanumeric material numbers are not padded in ECC either).

    Args:
        value: Customer or material number as printed / entered.
        length: Field length (10 for KUNNR, 18 for MATNR).

    Returns:
        Internal representation, or None for empty input.
    """
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    return text.zfill(length) if text.isdigit() else text


def alpha_out(value: Any) -> str | None:
    """Convert an internal ALPHA number back to its display form (leading zeros removed).

    Args:
        value: Internal number.

    Returns:
        Display form, or None for empty input.
    """
    if value is None:
        return None
    text = str(value).strip()
    if text.isdigit():
        return text.lstrip("0") or "0"
    return text


# ---------------------------------------------------------------------------
# Text normalisation and similarity
# ---------------------------------------------------------------------------
def strip_accents(text: str) -> str:
    """Remove diacritics (incl. Vietnamese) and fold case.

    Args:
        text: Any text.

    Returns:
        ASCII-ish lower-case text ("Đường" -> "duong", "Straße" -> "strasse").
    """
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()


def tokens(text: str | None, drop_legal_forms: bool = True) -> list[str]:
    """Split text into normalised alphanumeric tokens.

    Args:
        text: Input text (None allowed).
        drop_legal_forms: Remove tokens such as "gmbh", "pte", "tnhh".

    Returns:
        Token list in original order.
    """
    if not text:
        return []
    parts = re.findall(r"[a-z0-9]+", strip_accents(text))
    return [p for p in parts if not (drop_legal_forms and p in LEGAL_FORM_TOKENS)]


def text_similarity(a: str | None, b: str | None) -> float:
    """Similarity of two names/addresses in [0, 1].

    The maximum of a character-level ratio (robust to typos and hyphenation) and a
    token Dice coefficient (robust to word order and extra words).

    Args:
        a: First text.
        b: Second text.

    Returns:
        Similarity score; 0.0 when either side is empty.
    """
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    seq = SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio()
    sa, sb = set(ta), set(tb)
    dice = 2 * len(sa & sb) / (len(sa) + len(sb))
    return max(seq, dice)


def token_coverage(document_text: str | None, master_text: str | None) -> float:
    """Share of the master record's tokens that appear in the document text.

    Better than symmetric similarity for addresses, where documents add words
    ("old address ...", city prefixes) that the master record does not have.

    Args:
        document_text: Text from the purchase order.
        master_text: Text from master data.

    Returns:
        Fraction in [0, 1]; 0.0 when either side is empty.
    """
    doc, master = set(tokens(document_text)), set(tokens(master_text))
    if not doc or not master:
        return 0.0
    return len(doc & master) / len(master)


def digits(text: str | None) -> str:
    """Return only the digits of a text (for phone and tax-id comparison)."""
    return re.sub(r"\D", "", text or "")


def compact_id(text: str | None) -> str:
    """Normalise an identifier (VAT, registration no.) to upper-case alphanumerics."""
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def email_domain(address: str | None) -> str | None:
    """Extract the lower-case domain of an e-mail address.

    Args:
        address: E-mail address, optionally in "Name <addr>" form.

    Returns:
        Domain, or None if no address is found.
    """
    match = re.search(r"@([A-Za-z0-9.-]+)", address or "")
    return match.group(1).lower().rstrip(".") if match else None


# ---------------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------------
def score_customer(
    customer: dict[str, Any],
    *,
    name: str | None = None,
    address: str | None = None,
    country: str | None = None,
    sender_email: str | None = None,
    tax_id: str | None = None,
    phone: str | None = None,
) -> tuple[float, list[str]]:
    """Score one KNA1 customer against what a purchase order says about its buyer.

    Args:
        customer: KNA1 row plus ``EMAILS`` (list of contact e-mail addresses).
        name: Buyer name as printed.
        address: Buyer address as printed (one line).
        country: ISO-2 country code, if known.
        sender_email: E-mail address of the message sender.
        tax_id: VAT / GST / registration number printed on the document.
        phone: Phone number printed in the buyer block.

    Returns:
        (score, matched_on) where matched_on lists the signals that contributed.
    """
    matched: list[str] = []
    strong = 0.0
    if tax_id and customer.get("STCEG") and compact_id(tax_id) == compact_id(customer["STCEG"]):
        strong, matched = 1.0, ["tax_id"]
    domain = email_domain(sender_email)
    if domain and domain not in PUBLIC_MAIL_DOMAINS and domain in {email_domain(e) for e in customer.get("EMAILS", [])}:
        strong = max(strong, 0.97)
        matched.append("email_domain")
    phone_digits, own_phone = digits(phone), digits(customer.get("TELF1"))
    if len(phone_digits) >= 6 and len(own_phone) >= 6 and (phone_digits.endswith(own_phone) or own_phone.endswith(phone_digits)):
        strong = max(strong, 0.97)
        matched.append("phone")

    full_name = " ".join(filter(None, [customer.get("NAME1"), customer.get("NAME2")]))
    name_sim = text_similarity(name, full_name)
    if address:
        full_address = " ".join(filter(None, [customer.get("STRAS"), customer.get("PSTLZ"), customer.get("ORT01")]))
        fuzzy = 0.75 * name_sim + 0.25 * text_similarity(address, full_address)
    else:
        fuzzy = 0.9 * name_sim
    if country and customer.get("LAND1") and country.strip().upper() != customer["LAND1"]:
        fuzzy -= 0.15
    if name_sim >= 0.6:
        matched.append("name")
    # A strong identifier wins, but an obviously different name lowers trust in it.
    score = strong if strong and (not name or name_sim >= 0.4) else max(fuzzy, strong - 0.2 if strong else 0.0)
    return round(max(0.0, min(score, 1.0)), 3), matched


def name_key(text: str | None) -> str:
    """Normalised key of a printed name/code for learned mappings (accents, case, legal forms removed)."""
    return " ".join(tokens(text))


def rank_customers(customers: Iterable[dict[str, Any]], limit: int = 5, learned: dict[str, str] | None = None,
                   **query: Any) -> list[dict[str, Any]]:
    """Rank customers for a buyer description; see ``score_customer`` for query keys.

    Args:
        customers: KNA1 rows with ``EMAILS``.
        limit: Maximum candidates returned.
        learned: Reviewer-confirmed mappings {name_key(printed buyer name): KUNNR}; a hit scores 0.98.
        **query: name, address, country, sender_email, tax_id, phone.

    Returns:
        Candidates sorted by score (descending), each the customer row plus
        ``SCORE``, ``MATCHED_ON`` and, on the first entry, ``AMBIGUOUS``.
    """
    learned_kunnr = (learned or {}).get(name_key(query.get("name")))
    scored = []
    for customer in customers:
        score, matched_on = score_customer(customer, **query)
        if learned_kunnr and customer["KUNNR"] == learned_kunnr and score < 0.98:
            score, matched_on = 0.98, ["learned", *matched_on]
        if score >= 0.3:
            scored.append({**customer, "SCORE": score, "MATCHED_ON": matched_on})
    scored.sort(key=lambda c: c["SCORE"], reverse=True)
    return mark_ambiguity(scored[:limit])


def mark_ambiguity(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Set ``AMBIGUOUS`` on every candidate: True when the top-2 scores are too close.

    Args:
        candidates: Candidates sorted by ``SCORE`` descending.

    Returns:
        The same list (mutated) for chaining.
    """
    ambiguous = len(candidates) > 1 and candidates[0]["SCORE"] - candidates[1]["SCORE"] < AMBIGUITY_GAP
    for candidate in candidates:
        candidate["AMBIGUOUS"] = ambiguous
    return candidates


# ---------------------------------------------------------------------------
# Material codes and packs
# ---------------------------------------------------------------------------
def pack_litres(text: str | None) -> float | None:
    """Read a pack size in litres from text such as "20L", "A4LT", "1GL", "946ML", "QT".

    Args:
        text: Pack text, code segment or description.

    Returns:
        Litres, or None when no pack size is recognisable.
    """
    if not text:
        return None
    upper = strip_accents(text).upper()
    match = _PACK_RE.search(upper)
    if match:
        return round(float(match.group(1).replace(",", ".")) * _LITRES_PER_UNIT[match.group(2)], 3)
    for word in re.findall(r"[A-Z]+", upper):
        if word in _BARE_PACK:
            return _LITRES_PER_UNIT[word]
    return None


def same_pack(a: float | None, b: float | None) -> bool | None:
    """Compare two pack sizes within 3 %; None when either is unknown."""
    if a is None or b is None:
        return None
    return abs(a - b) <= 0.03 * max(a, b)


def parse_code(code: str) -> tuple[str, float | None, set[str]]:
    """Split a sales code like "IPJ715/20L/XA" into base, pack and other suffixes.

    Args:
        code: Printed or master-data sales code.

    Returns:
        (base, pack in litres or None, set of non-pack suffix segments).
    """
    parts = [p for p in re.split(r"\s*/\s*", code.strip().upper()) if p]
    if not parts:
        return "", None, set()
    base, pack, suffixes = parts[0], None, set()
    for segment in parts[1:]:
        litres = pack_litres(segment)
        if litres is not None and pack is None:
            pack = litres
        else:
            suffixes.add(segment)
    return base, pack, suffixes


def code_variants(printed: str) -> list[str]:
    """Candidate codes hidden in a printed value ("Brandname 455991" -> ["BRANDNAME 455991", "455991"]).

    Args:
        printed: Code as printed, possibly with a brand word in front.

    Returns:
        Distinct upper-case variants, original first.
    """
    original = re.sub(r"\s+", " ", printed.strip().upper())
    variants = [original]
    for word in original.split(" "):
        if any(ch.isdigit() for ch in word) and word not in variants:
            variants.append(word)
    return variants


def score_code(printed: str, material: dict[str, Any], line_pack: float | None) -> tuple[float, str | None, list[str]]:
    """Score one printed code against a material's number and sales code.

    Args:
        printed: One code variant from the document.
        material: MARA row (``MATNR``, ``BISMT``).
        line_pack: Pack size in litres stated elsewhere on the line, if any.

    Returns:
        (score, matched_on, conflict notes).
    """
    if alpha_in(printed, 18) == material["MATNR"]:
        return 1.0, "matnr", []
    bismt = material.get("BISMT")
    if not bismt:
        return 0.0, None, []
    p_base, p_pack, p_suffix = parse_code(printed)
    m_base, m_pack, m_suffix = parse_code(bismt)
    notes: list[str] = []
    if p_base == m_base and (p_pack == m_pack or p_pack is None and m_pack is None) and p_suffix == m_suffix:
        score, how = 1.0, "sales_code"
    elif p_base == m_base:
        score, how = 0.85, "sales_code_partial"
    elif len(m_base) == len(p_base) + 1 and m_base.endswith(p_base) and m_base[0].isalpha():
        score, how = 0.82, "sales_code_prefix"          # e.g. printed G1421, master OG1421
    else:
        return 0.0, None, []
    if score < 1.0:
        pack_check = same_pack(p_pack, m_pack)
        if pack_check is True:
            score += 0.1
        elif pack_check is False:
            score -= 0.25
            notes.append(f"code pack {p_pack} L differs from material pack {m_pack} L")
        missing = p_suffix - m_suffix
        if missing:
            score -= 0.1
        elif p_suffix and p_suffix <= m_suffix:
            score += 0.04
    line_check = same_pack(line_pack, m_pack if m_pack is not None else _volume_litres(material))
    if line_check is False:
        score -= 0.2
        notes.append(f"line states {line_pack} L but material pack is {m_pack or _volume_litres(material)} L")
    elif line_check is True and score < 1.0:
        score += 0.05
    return round(max(0.0, min(score, 1.0)), 3), how, notes


def _volume_litres(material: dict[str, Any]) -> float | None:
    """Fill volume in litres from MARA.VOLUM/VOLEH (only litre-based units)."""
    volume = material.get("VOLUM")
    if volume is None:
        return None
    return float(volume) * {"L": 1.0, "ML": 0.001}.get(material.get("VOLEH") or "L", 1.0)


def score_material(
    material: dict[str, Any],
    *,
    codes: Iterable[str] = (),
    description: str | None = None,
    colour: str | None = None,
    pack_text: str | None = None,
    customer_codes: dict[str, str] | None = None,
    learned: dict[str, str] | None = None,
) -> tuple[float, list[str], list[str]]:
    """Score one material for an order line.

    Args:
        material: MARA row joined with ``MAKTX`` (``MFRPN`` = manufacturer part number, optional).
        codes: Codes printed on the line (material no., sales code, customer code...).
        description: Line description as printed (translated to English if needed).
        colour: Colour name/code on the line.
        pack_text: Pack size text on the line (e.g. "4 Ltr", "QT").
        customer_codes: Customer-material info map {normalised KDMAT: MATNR} for the sold-to.
        learned: Reviewer corrections for the sold-to {normalised printed code: MATNR}; a hit scores 0.98.

    Returns:
        (score, matched_on, conflict notes).
    """
    codes = list(codes)  # iterated twice (codes, manufacturer part number)
    line_pack = pack_litres(pack_text)
    best, matched_on, notes = 0.0, [], []
    for code in codes:
        for variant in code_variants(code):
            if customer_codes and customer_codes.get(compact_id(variant)) == material["MATNR"]:
                if 0.99 > best:
                    best, matched_on, notes = 0.99, ["customer_code"], []
            if learned and learned.get(compact_id(variant)) == material["MATNR"] and 0.98 > best:
                best, matched_on, notes = 0.98, ["learned"], []
            score, how, conflicts = score_code(variant, material, line_pack)
            if score > best:
                best, matched_on, notes = score, [how], conflicts
    # Manufacturer part number of traded goods: dealers print it as a code or inside the description.
    # ponytail: substring test on the compacted description; a 4+ character part number rarely collides.
    mfrpn = compact_id(material.get("MFRPN"))
    if len(mfrpn) >= 4 and 0.95 > best and (
            any(compact_id(code) == mfrpn for code in codes) or mfrpn in compact_id(description)):
        best, matched_on, notes = 0.95, ["manufacturer_part"], []
    if description:
        desc_score = 0.85 * text_similarity(description, material.get("MAKTX"))
        maktx_tokens = set(tokens(material.get("MAKTX")))
        if colour and set(tokens(colour)) & maktx_tokens:
            desc_score += 0.05
        pack_check = same_pack(line_pack, _volume_litres(material) or pack_litres(material.get("BISMT")))
        if pack_check is True:
            desc_score += 0.05
        elif pack_check is False:
            desc_score -= 0.15
        desc_score = min(desc_score, 0.9)
        if desc_score > best:
            best, matched_on, notes = desc_score, ["description"], []
        elif desc_score >= 0.6 and matched_on:
            matched_on = matched_on + ["description"]
    return round(max(best, 0.0), 3), matched_on, notes


def rank_materials(materials: Iterable[dict[str, Any]], limit: int = 5, **line: Any) -> list[dict[str, Any]]:
    """Rank materials for one order line; see ``score_material`` for line keys.

    Args:
        materials: Material rows (deleted materials should already be excluded).
        limit: Maximum candidates returned.
        **line: codes, description, colour, pack_text, customer_codes, learned.

    Returns:
        Candidates sorted by score with ``SCORE``, ``MATCHED_ON``, ``NOTES`` and ``AMBIGUOUS``.
    """
    scored = []
    for material in materials:
        score, matched_on, notes = score_material(material, **line)
        if score >= 0.3:
            scored.append({**material, "SCORE": score, "MATCHED_ON": matched_on, "NOTES": notes})
    scored.sort(key=lambda m: m["SCORE"], reverse=True)
    return mark_ambiguity(scored[:limit])
