"""
Dataset search module.
Searches the in-memory cached dataset for records relevant to the user query.
Operates ONLY on the already-loaded FinancialProcessor cache — no HANA queries.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column constants (must match services/financial_processor.py)
# ---------------------------------------------------------------------------
COL_GL_ACCOUNT       = "G/L Account"
COL_GL_ACCOUNT_NAME  = "G/L Account Name"
COL_COST_CENTER      = "Cost Center"
COL_COST_CENTER_NAME = "Cost Center Name"
COL_PROFIT_CENTER    = "Profit Center"
COL_PROFIT_CENTER_NAME = "Profit Center Name"
COL_WBS_ELEMENT      = "WBS Element"
COL_PURCHASING_DOC   = "Purchasing Document"
COL_SUPPLIER         = "Supplier"
COL_SUPPLIER_NAME    = "Name of Supplier"
COL_JE_ITEM_TEXT     = "Journal Entry Item Text"
COL_SEGMENT          = "Segment"
COL_COMPANY_CODE     = "Company Code"
COL_ASSIGNMENT_REF   = "Assignment Reference"
COL_JE_TYPE          = "Journal Entry Type"
COL_JE_TYPE_NAME     = "JE Type Name"
COL_PARTNER_PC       = "Partner Profit Center"
COL_OFFSETTING_ACCT  = "Offsetting Account"
COL_OFFSETTING_NAME  = "Offsetting Acct Name"
COL_FUNCTIONAL_AREA  = "Functional Area Name"
COL_ACCOUNT_GROUPING = "Account Grouping"
COL_FS_LINE_ITEM     = "Financial Statement Line Item"
COL_FLASH_LINE_ITEM  = "Flash Line Item"          # authoritative from v_glaccount
COL_FS_ITEM          = "FS Item"
COL_FS_ITEM_NAME     = "FS Item Name"
COL_AMOUNT           = "Amount in Company Code Currency"
AMOUNT_NUMERIC       = "amount_numeric"
FISCAL_YEAR          = "fiscal_year"
FISCAL_MONTH         = "fiscal_month"

# ---------------------------------------------------------------------------
# Flash Line Item canonical categories (from v_glaccount)
# ---------------------------------------------------------------------------
FLASH_CATEGORIES: Dict[str, List[str]] = {
    "travel and entertainment": [
        "travel and entertainment", "travel & entertainment", "t&e",
        "travel", "hotel", "flight", "airfare", "lodging", "meal",
        "entertainment", "per diem", "mileage", "car rental",
    ],
    "employee expense": [
        "employee expense", "employee expenses", "salary", "salaries",
        "payroll", "wage", "wages", "benefit", "bonus", "compensation",
        "pension", "healthcare", "hr", "human resources",
    ],
    "repair and maintenance": [
        "repair and maintenance", "repair & maintenance", "r&m",
        "repair", "maintenance", "fix", "overhaul", "spare part",
        "equipment service", "upkeep", "preventive",
    ],
    "outside services": [
        "outside services", "consulting", "consultant", "contractor",
        "professional service", "outsource", "vendor service",
        "third party", "advisory", "legal fee", "audit fee",
    ],
    "depreciation": [
        "depreciation", "amortization", "depr",
    ],
    "utilities": [
        "utilities", "utility", "electricity", "water", "gas", "power",
    ],
    "insurance": [
        "insurance", "premium", "coverage",
    ],
    "rent": [
        "rent", "lease", "rental", "facility",
    ],
}

# ---------------------------------------------------------------------------
# Semantic aliases for common financial terms / abbreviations
# (kept for backward compatibility — Flash categories take priority)
# ---------------------------------------------------------------------------
SEMANTIC_ALIASES: Dict[str, List[str]] = {
    "travel": ["travel", "hotel", "flight", "airfare", "airline", "lodging",
               "meal", "entertainment", "conference", "training", "seminar",
               "per diem", "mileage", "car rental"],
    "t&e":    ["travel", "hotel", "flight", "airfare", "lodging", "meal",
               "entertainment", "per diem"],
    "repair": ["repair", "maintenance", "fix", "overhaul", "spare part",
               "equipment service", "facility", "upkeep", "preventive"],
    "r&m":    ["repair", "maintenance", "fix", "overhaul"],
    "outside services": ["consulting", "consultant", "contractor",
                         "professional service", "outsource", "vendor service",
                         "third party", "advisory", "legal fee", "audit fee"],
    "employee": ["salary", "salaries", "payroll", "wage", "wages", "employee",
                 "benefit", "bonus", "compensation", "pension", "healthcare"],
    "depreciation": ["depreciation", "amortization", "depr"],
    "utilities": ["utility", "utilities", "electricity", "water", "gas", "power"],
    "insurance": ["insurance", "premium", "coverage"],
    "rent": ["rent", "lease", "rental", "facility"],
}


def _resolve_flash_category(msg_lower: str) -> Optional[str]:
    """
    Detect if the user message refers to a Flash Line Item category.
    Returns the canonical Flash category name or None.
    Priority: exact Flash category match > semantic alias match.
    """
    # 1. Exact or partial match against Flash category names
    for category, keywords in FLASH_CATEGORIES.items():
        if any(kw in msg_lower for kw in keywords):
            return category
    return None


# ---------------------------------------------------------------------------
# Words that must NOT be captured as entity IDs (dimension names, adjectives, etc.)
# ---------------------------------------------------------------------------
_ENTITY_BLOCKLIST = {
    "variance", "variances", "analysis", "summary", "biggest", "largest",
    "highest", "lowest", "top", "bottom", "increase", "decrease", "change",
    "delta", "difference", "comparison", "report", "breakdown", "detail",
    "details", "overview", "total", "totals", "all", "any", "the", "for",
    "and", "with", "from", "into", "that", "this", "which", "what", "why",
    "how", "show", "list", "find", "get", "give", "tell", "explain",
    "analyze", "analyse", "compare", "drill", "drilldown", "account",
    "accounts", "center", "centres", "centers", "profit", "cost", "element",
    "document", "supplier", "vendor", "segment", "group", "grouping",
    "period", "quarter", "year", "month", "fiscal", "financial", "data",
    "records", "entries", "entry", "journal", "posting", "amount", "value",
    "number", "name", "code", "type", "category", "class", "level",
    "driver", "drivers", "cause", "causes", "reason", "reasons",
}


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

# Month name → number mapping
_MONTH_NAMES: Dict[str, int] = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


def extract_entities(message: str, context_entities: Optional[Dict] = None) -> Dict:
    """
    Extract financial entities from the user message using regex + keyword matching.
    Merges with context_entities for follow-up question resolution.

    Extracted entities:
      gl_account, years, quarter, month, cost_center, profit_center,
      wbs_element, purchasing_doc, supplier_name, search_text,
      flash_category, semantic_category,
      segment, functional_area, je_type
    """
    msg_lower = message.lower()
    entities: Dict = {}

    # GL Account number — 8-digit numeric
    gl_matches = re.findall(r'\b(\d{8})\b', message)
    if gl_matches:
        entities["gl_account"] = gl_matches[0]

    # Fiscal year — 4-digit year 2020-2030
    year_matches = re.findall(r'\b(20[2-3]\d)\b', message)
    if year_matches:
        entities["years"] = list(set(year_matches))

    # Quarter — Q1..Q4
    quarter_matches = re.findall(r'\bq([1-4])\b', msg_lower)
    if quarter_matches:
        entities["quarter"] = int(quarter_matches[0])

    # Month name — "March 2026", "in January", "for March"
    if not entities.get("quarter"):
        for month_name, month_num in _MONTH_NAMES.items():
            if re.search(r'\b' + month_name + r'\b', msg_lower):
                entities["month"] = month_num
                break

    # Cost center — only when followed by an explicit ID (not a common word)
    cc_matches = re.findall(
        r'\b(?:cc|cost\s*center)\s*[:\-]\s*([A-Za-z0-9]{3,12})\b', msg_lower
    )
    if not cc_matches:
        cc_matches = re.findall(
            r'\bcost\s*center\s+([A-Za-z0-9]{3,12})\b', msg_lower
        )
    cc_matches = [m for m in cc_matches if m.lower() not in _ENTITY_BLOCKLIST]
    if cc_matches:
        entities["cost_center"] = cc_matches[0].upper()

    # Profit center — same strict matching
    pc_matches = re.findall(
        r'\b(?:pc|profit\s*center)\s*[:\-]\s*([A-Za-z0-9]{3,12})\b', msg_lower
    )
    if not pc_matches:
        pc_matches = re.findall(
            r'\bprofit\s*center\s+([A-Za-z0-9]{3,12})\b', msg_lower
        )
    pc_matches = [m for m in pc_matches if m.lower() not in _ENTITY_BLOCKLIST]
    if pc_matches:
        entities["profit_center"] = pc_matches[0].upper()

    # WBS element
    wbs_matches = re.findall(
        r'\b(?:wbs|wbs\s*element)\s*[:\-]?\s*([A-Za-z0-9\-\.]{4,20})\b', msg_lower
    )
    if wbs_matches:
        entities["wbs_element"] = wbs_matches[0].upper()

    # Purchasing document
    po_matches = re.findall(
        r'\b(?:po|purchasing\s*doc(?:ument)?)\s*[:\-]?\s*(\d{8,10})\b', msg_lower
    )
    if po_matches:
        entities["purchasing_doc"] = po_matches[0]

    # Supplier name — after "supplier:" or "vendor:"
    supplier_matches = re.findall(
        r'(?:supplier|vendor)\s*[:\-]\s*([A-Za-z][A-Za-z\s&,\.]{2,40}?)(?:\s*$|\s*[,\.])',
        msg_lower
    )
    if supplier_matches:
        entities["supplier_name"] = supplier_matches[0].strip()

    # Quoted search text
    quoted = re.findall(r'"([^"]+)"', message)
    if quoted:
        entities["search_text"] = quoted[0]

    # ------------------------------------------------------------------
    # Segment extraction — patterns:
    #   "Corporate Segment", "Corporate segment", "segment: Corporate",
    #   "for Corporate", "in the Corporate segment"
    # Captures 1-3 word phrase before/after "segment" keyword.
    # ------------------------------------------------------------------
    seg_match = None
    # Pattern 1: "<phrase> segment" — e.g. "Corporate Segment", "Retail segment"
    m = re.search(
        r'\b([A-Za-z][A-Za-z\s]{1,30}?)\s+segment\b',
        msg_lower
    )
    if m:
        candidate = m.group(1).strip()
        # Exclude generic words that are not segment names
        if candidate not in {"the", "a", "this", "that", "each", "any", "all", "by", "per"}:
            seg_match = candidate
    # Pattern 2: "segment: <phrase>" or "segment = <phrase>"
    if not seg_match:
        m2 = re.search(r'\bsegment\s*[:\=]\s*([A-Za-z][A-Za-z\s]{1,30})', msg_lower)
        if m2:
            seg_match = m2.group(1).strip().rstrip(".,;")
    if seg_match:
        # Title-case for consistent matching against SAP data
        entities["segment"] = seg_match.title()
        logger.debug("extract_entities: segment='%s'", entities["segment"])

    # ------------------------------------------------------------------
    # Functional Area extraction — patterns:
    #   "functional area: Finance", "Finance functional area"
    # ------------------------------------------------------------------
    fa_match = None
    m3 = re.search(
        r'\b([A-Za-z][A-Za-z\s]{1,30}?)\s+functional\s+area\b',
        msg_lower
    )
    if m3:
        candidate = m3.group(1).strip()
        if candidate not in {"the", "a", "this", "that", "each", "any", "all"}:
            fa_match = candidate
    if not fa_match:
        m4 = re.search(
            r'\bfunctional\s+area\s*[:\=]\s*([A-Za-z][A-Za-z\s]{1,30})',
            msg_lower
        )
        if m4:
            fa_match = m4.group(1).strip().rstrip(".,;")
    if fa_match:
        entities["functional_area"] = fa_match.title()
        logger.debug("extract_entities: functional_area='%s'", entities["functional_area"])

    # ------------------------------------------------------------------
    # Journal Entry Type extraction — e.g. "JE type RC", "document type KR"
    # ------------------------------------------------------------------
    je_match = re.search(
        r'\b(?:je\s*type|journal\s*entry\s*type|doc(?:ument)?\s*type)\s*[:\-]?\s*([A-Z]{1,3})\b',
        message,
        re.IGNORECASE,
    )
    if je_match:
        entities["je_type"] = je_match.group(1).upper()

    # Flash Line Item category detection (authoritative — from v_glaccount)
    flash_cat = _resolve_flash_category(msg_lower)
    if flash_cat:
        entities["flash_category"] = flash_cat
    else:
        # Fallback: legacy semantic alias detection
        for category, keywords in SEMANTIC_ALIASES.items():
            if any(kw in msg_lower for kw in keywords):
                entities["semantic_category"] = category
                break

    # Merge with context entities for follow-up resolution
    # (context entities fill in gaps — current message takes priority)
    if context_entities:
        for key, value in context_entities.items():
            if key not in entities and value is not None and value != "":
                entities[key] = value

    return entities


# ---------------------------------------------------------------------------
# Dataset search
# ---------------------------------------------------------------------------

def _fuzzy_match_segment(df: pd.DataFrame, segment_query: str) -> pd.Series:
    """
    Match segment_query against the Segment column using:
    1. Exact match (case-insensitive)
    2. Starts-with match
    3. Contains match
    Returns a boolean mask.
    """
    if COL_SEGMENT not in df.columns:
        return pd.Series([False] * len(df), index=df.index)

    col = df[COL_SEGMENT].fillna("").astype(str).str.strip()
    q   = segment_query.strip().lower()

    # Exact match first
    mask = col.str.lower() == q
    if mask.any():
        return mask

    # Starts-with
    mask = col.str.lower().str.startswith(q)
    if mask.any():
        return mask

    # Contains
    mask = col.str.lower().str.contains(re.escape(q), na=False, regex=True)
    return mask


def search_dataset(
    df: pd.DataFrame,
    entities: Dict,
    current_year: Optional[int] = None,
    previous_year: Optional[int] = None,
    company_code: Optional[str] = None,
    segment: Optional[str] = None,
    functional_area: Optional[str] = None,
    limit: int = 500,
    hierarchy=None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Filter the cached DataFrame based on extracted entities.

    Filter application order (per spec):
      1. Company code
      2. Segment (from entities OR explicit param)
         - Uses HierarchyService when available: Segment -> PCs -> CCs (authoritative)
         - Falls back to Segment column direct filter
      3. Functional Area (from entities OR explicit param)
      4. Fiscal year / quarter / month
      5. GL Account / Cost Center / Profit Center / WBS / PO
      6. Supplier / Flash / Semantic / Free text

    Args:
        hierarchy: Optional HierarchyService instance. When provided and built,
                   segment filtering uses the PC/CC hierarchy rather than the
                   Segment column directly. This ensures only CCs belonging to
                   the requested Segment are included.

    Returns (filtered_df, search_metadata).
    """
    if df is None or df.empty:
        return pd.DataFrame(), {"matched": False, "reason": "No data in cache"}

    result = df.copy()
    total_rows = len(result)
    meta: Dict = {"filters_applied": [], "matched": False, "total_rows": total_rows}

    logger.info(
        "search_dataset: START — total_rows=%d, entities=%s, "
        "current_year=%s, previous_year=%s, company=%s, segment=%s, fa=%s, "
        "hierarchy_built=%s",
        total_rows, {k: v for k, v in entities.items() if k != "years"},
        current_year, previous_year, company_code, segment, functional_area,
        hierarchy.is_built if hierarchy is not None else False,
    )

    # --- 1. Company code ---
    if company_code and COL_COMPANY_CODE in result.columns:
        before = len(result)
        result = result[result[COL_COMPANY_CODE].astype(str) == str(company_code)]
        meta["filters_applied"].append(f"company={company_code}")
        logger.info("search_dataset: company_code='%s': %d -> %d rows", company_code, before, len(result))

    # --- 2. Segment filter (entities take priority over explicit param) ---
    # Hierarchy: Segment -> Profit Centers -> Cost Centers (authoritative)
    seg_value = entities.get("segment") or segment
    if seg_value:
        before = len(result)

        # Collect all CCs before filter for exclusion logging
        all_ccs_before = set()
        if COL_COST_CENTER in result.columns:
            all_ccs_before = set(result[COL_COST_CENTER].dropna().astype(str).str.strip().unique())

        if hierarchy is not None and hierarchy.is_built:
            # Hierarchy-based filtering: Segment -> PCs -> CCs
            # This is the authoritative path — excludes CCs outside the segment
            result = hierarchy.apply_segment_filter(result, seg_value, context="search_dataset")
            meta["filters_applied"].append(f"segment={seg_value}[hierarchy]")

            # Narrative logging: top cost centers within segment
            if COL_COST_CENTER in result.columns and not result.empty:
                top_ccs = (
                    result[COL_COST_CENTER].dropna().astype(str)
                    .value_counts().head(10).index.tolist()
                )
                all_ccs_after = set(result[COL_COST_CENTER].dropna().astype(str).str.strip().unique())
                excluded_ccs = all_ccs_before - all_ccs_after
                logger.info(
                    "narrative_service[segment=%s]: "
                    "top_cost_centers=%s, "
                    "excluded_cost_centers_outside_segment=%d",
                    seg_value, top_ccs, len(excluded_ccs),
                )

            # Hierarchy stats for this segment
            if hierarchy.is_built:
                stats = hierarchy.get_hierarchy_stats(seg_value)
                logger.info(
                    "hierarchy_service[search_dataset]: %s",
                    {
                        "segment":                stats["segment"],
                        "matched_profit_centers": stats["matched_profit_centers"],
                        "matched_cost_centers":   stats["matched_cost_centers"],
                        "filtered_transaction_rows": len(result),
                    },
                )

        elif COL_SEGMENT in result.columns:
            # Fallback: direct Segment column fuzzy match
            seg_mask = _fuzzy_match_segment(result, seg_value)
            matched_segs = result.loc[seg_mask, COL_SEGMENT].dropna().unique().tolist()
            if seg_mask.any():
                result = result[seg_mask]
                meta["filters_applied"].append(f"segment={seg_value}")
                logger.info(
                    "search_dataset: segment='%s' matched SAP values %s: %d -> %d rows "
                    "(direct column filter — hierarchy not available)",
                    seg_value, matched_segs, before, len(result),
                )
            else:
                logger.warning(
                    "search_dataset: segment='%s' matched 0 rows. "
                    "Available segments: %s",
                    seg_value,
                    sorted(df[COL_SEGMENT].dropna().astype(str).unique().tolist())[:20],
                )
        else:
            logger.warning(
                "search_dataset: segment filter requested but column '%s' not in dataset "
                "and hierarchy not available.",
                COL_SEGMENT,
            )

    # --- 3. Functional Area filter ---
    fa_value = entities.get("functional_area") or functional_area
    if fa_value and COL_FUNCTIONAL_AREA in result.columns:
        before = len(result)
        fa_col = result[COL_FUNCTIONAL_AREA].fillna("").astype(str).str.strip()
        fa_mask = fa_col.str.lower() == fa_value.strip().lower()
        if not fa_mask.any():
            fa_mask = fa_col.str.lower().str.contains(re.escape(fa_value.strip().lower()), na=False)
        if fa_mask.any():
            result = result[fa_mask]
            meta["filters_applied"].append(f"fa={fa_value}")
            logger.info("search_dataset: functional_area='%s': %d → %d rows", fa_value, before, len(result))
        else:
            logger.warning(
                "search_dataset: functional_area='%s' matched 0 rows. "
                "Available: %s",
                fa_value,
                sorted(df[COL_FUNCTIONAL_AREA].dropna().astype(str).unique().tolist())[:20],
            )

    # --- 4a. Year filter ---
    years_to_filter = set()
    if current_year:
        years_to_filter.add(int(current_year))
    if previous_year:
        years_to_filter.add(int(previous_year))
    if entities.get("years"):
        for y in entities["years"]:
            years_to_filter.add(int(y))

    if years_to_filter and FISCAL_YEAR in result.columns:
        before = len(result)
        result = result[result[FISCAL_YEAR].isin(years_to_filter)]
        meta["filters_applied"].append(f"years={sorted(years_to_filter)}")
        logger.info("search_dataset: years=%s: %d → %d rows", sorted(years_to_filter), before, len(result))

    # --- 4b. Quarter filter ---
    if entities.get("quarter") and FISCAL_MONTH in result.columns:
        before = len(result)
        q = int(entities["quarter"])
        months = list(range((q - 1) * 3 + 1, q * 3 + 1))
        result = result[result[FISCAL_MONTH].isin(months)]
        meta["filters_applied"].append(f"Q{q}")
        logger.info("search_dataset: Q%d (months=%s): %d → %d rows", q, months, before, len(result))

    # --- 4c. Month filter (only if no quarter) ---
    elif entities.get("month") and FISCAL_MONTH in result.columns:
        before = len(result)
        m_num = int(entities["month"])
        result = result[result[FISCAL_MONTH] == m_num]
        meta["filters_applied"].append(f"month={m_num}")
        logger.info("search_dataset: month=%d: %d → %d rows", m_num, before, len(result))

    # --- 5a. GL Account ---
    if entities.get("gl_account") and COL_GL_ACCOUNT in result.columns:
        before = len(result)
        result = result[result[COL_GL_ACCOUNT].astype(str) == str(entities["gl_account"])]
        meta["filters_applied"].append(f"gl={entities['gl_account']}")
        logger.info("search_dataset: gl_account='%s': %d → %d rows", entities["gl_account"], before, len(result))

    # --- 5b. Cost Center ---
    if entities.get("cost_center") and COL_COST_CENTER in result.columns:
        before = len(result)
        result = result[
            result[COL_COST_CENTER].astype(str).str.upper() == entities["cost_center"]
        ]
        meta["filters_applied"].append(f"cc={entities['cost_center']}")
        logger.info("search_dataset: cost_center='%s': %d → %d rows", entities["cost_center"], before, len(result))

    # --- 5c. Profit Center ---
    if entities.get("profit_center") and COL_PROFIT_CENTER in result.columns:
        before = len(result)
        result = result[
            result[COL_PROFIT_CENTER].astype(str).str.upper() == entities["profit_center"]
        ]
        meta["filters_applied"].append(f"pc={entities['profit_center']}")
        logger.info("search_dataset: profit_center='%s': %d → %d rows", entities["profit_center"], before, len(result))

    # --- 5d. WBS Element ---
    if entities.get("wbs_element") and COL_WBS_ELEMENT in result.columns:
        before = len(result)
        result = result[
            result[COL_WBS_ELEMENT].astype(str).str.upper() == entities["wbs_element"]
        ]
        meta["filters_applied"].append(f"wbs={entities['wbs_element']}")
        logger.info("search_dataset: wbs='%s': %d → %d rows", entities["wbs_element"], before, len(result))

    # --- 5e. Purchasing Document ---
    if entities.get("purchasing_doc") and COL_PURCHASING_DOC in result.columns:
        before = len(result)
        result = result[
            result[COL_PURCHASING_DOC].astype(str) == str(entities["purchasing_doc"])
        ]
        meta["filters_applied"].append(f"po={entities['purchasing_doc']}")
        logger.info("search_dataset: po='%s': %d → %d rows", entities["purchasing_doc"], before, len(result))

    # --- 5f. Journal Entry Type ---
    if entities.get("je_type") and COL_JE_TYPE in result.columns:
        before = len(result)
        result = result[
            result[COL_JE_TYPE].astype(str).str.upper() == entities["je_type"]
        ]
        meta["filters_applied"].append(f"je_type={entities['je_type']}")
        logger.info("search_dataset: je_type='%s': %d → %d rows", entities["je_type"], before, len(result))

    # --- 6a. Supplier name (fuzzy) ---
    if entities.get("supplier_name") and COL_SUPPLIER_NAME in result.columns:
        before = len(result)
        sname = entities["supplier_name"].lower()
        result = result[
            result[COL_SUPPLIER_NAME].fillna("").astype(str).str.lower().str.contains(
                sname, na=False, regex=False
            )
        ]
        meta["filters_applied"].append(f"supplier~{entities['supplier_name']}")
        logger.info("search_dataset: supplier~'%s': %d → %d rows", entities["supplier_name"], before, len(result))

    # --- Flash Line Item category filter (authoritative from v_glaccount) ---
    if entities.get("flash_category") and not entities.get("gl_account"):
        flash_cat = entities["flash_category"]
        flash_keywords = FLASH_CATEGORIES.get(flash_cat, [flash_cat])
        pattern = "|".join(re.escape(kw) for kw in flash_keywords)

        # Priority 1: match on Flash Line Item column (exact SAP source)
        if COL_FLASH_LINE_ITEM in result.columns:
            mask = result[COL_FLASH_LINE_ITEM].fillna("").astype(str).str.lower().str.contains(
                pattern, na=False, regex=True
            )
            # Priority 2: also match FS Item Name
            if COL_FS_ITEM_NAME in result.columns:
                mask |= result[COL_FS_ITEM_NAME].fillna("").astype(str).str.lower().str.contains(
                    pattern, na=False, regex=True
                )
            # Priority 3: also match Financial Statement Line Item
            if COL_FS_LINE_ITEM in result.columns:
                mask |= result[COL_FS_LINE_ITEM].fillna("").astype(str).str.lower().str.contains(
                    pattern, na=False, regex=True
                )
            # Priority 4: fallback to JE text + account name
            if COL_JE_ITEM_TEXT in result.columns:
                mask |= result[COL_JE_ITEM_TEXT].fillna("").astype(str).str.lower().str.contains(
                    pattern, na=False, regex=True
                )
            if COL_GL_ACCOUNT_NAME in result.columns:
                mask |= result[COL_GL_ACCOUNT_NAME].fillna("").astype(str).str.lower().str.contains(
                    pattern, na=False, regex=True
                )
        else:
            # Flash Line Item column not available — fall back to text search
            mask = pd.Series([False] * len(result), index=result.index)
            for col in [COL_JE_ITEM_TEXT, COL_GL_ACCOUNT_NAME, COL_FS_LINE_ITEM]:
                if col in result.columns:
                    mask |= result[col].fillna("").astype(str).str.lower().str.contains(
                        pattern, na=False, regex=True
                    )

        result = result[mask]
        meta["filters_applied"].append(f"flash={flash_cat}")

    # --- Semantic category / free text search (legacy fallback) ---
    elif entities.get("semantic_category") and not entities.get("gl_account"):
        category = entities["semantic_category"]
        keywords = SEMANTIC_ALIASES.get(category, [category])
        if COL_JE_ITEM_TEXT in result.columns:
            pattern = "|".join(re.escape(kw) for kw in keywords)
            mask = result[COL_JE_ITEM_TEXT].fillna("").astype(str).str.lower().str.contains(
                pattern, na=False, regex=True
            )
            if COL_GL_ACCOUNT_NAME in result.columns:
                mask |= result[COL_GL_ACCOUNT_NAME].fillna("").astype(str).str.lower().str.contains(
                    pattern, na=False, regex=True
                )
            result = result[mask]
            meta["filters_applied"].append(f"semantic={category}")

    # --- Free text search ---
    if entities.get("search_text") and not entities.get("gl_account"):
        stext = entities["search_text"].lower()
        text_mask = pd.Series([False] * len(result), index=result.index)
        for col in [COL_JE_ITEM_TEXT, COL_GL_ACCOUNT_NAME, COL_SUPPLIER_NAME]:
            if col in result.columns:
                text_mask |= result[col].fillna("").astype(str).str.lower().str.contains(
                    stext, na=False, regex=False
                )
        result = result[text_mask]
        meta["filters_applied"].append(f"text~{entities['search_text']}")

    meta["matched"] = not result.empty
    meta["matched_rows"] = len(result)

    # Limit result size
    if len(result) > limit:
        result = result.head(limit)

    return result, meta


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def build_gl_summary(df: pd.DataFrame, year: Optional[int] = None) -> List[Dict]:
    """Top GL accounts by absolute amount."""
    if df.empty or COL_GL_ACCOUNT not in df.columns or AMOUNT_NUMERIC not in df.columns:
        return []
    sub = df[df[FISCAL_YEAR] == year] if year and FISCAL_YEAR in df.columns else df
    if sub.empty:
        return []
    agg = (
        sub.groupby([COL_GL_ACCOUNT, COL_GL_ACCOUNT_NAME] if COL_GL_ACCOUNT_NAME in sub.columns
                    else [COL_GL_ACCOUNT])[AMOUNT_NUMERIC]
        .sum()
        .reset_index()
        .sort_values(AMOUNT_NUMERIC, key=abs, ascending=False)
        .head(10)
    )
    results = []
    for _, row in agg.iterrows():
        entry = {
            "gl_account": str(row[COL_GL_ACCOUNT]),
            "amount": float(row[AMOUNT_NUMERIC]),
        }
        if COL_GL_ACCOUNT_NAME in row.index:
            entry["name"] = str(row[COL_GL_ACCOUNT_NAME])
        results.append(entry)
    return results


def build_cost_center_summary(df: pd.DataFrame, year: Optional[int] = None) -> List[Dict]:
    """Top cost centers by absolute amount."""
    if df.empty or COL_COST_CENTER not in df.columns or AMOUNT_NUMERIC not in df.columns:
        return []
    sub = df[df[FISCAL_YEAR] == year] if year and FISCAL_YEAR in df.columns else df
    if sub.empty:
        return []
    agg = (
        sub.groupby(COL_COST_CENTER)[AMOUNT_NUMERIC]
        .sum()
        .reset_index()
        .sort_values(AMOUNT_NUMERIC, key=abs, ascending=False)
        .head(10)
    )
    return [
        {"cost_center": str(r[COL_COST_CENTER]), "amount": float(r[AMOUNT_NUMERIC])}
        for _, r in agg.iterrows()
    ]


def build_supplier_summary(df: pd.DataFrame, year: Optional[int] = None) -> List[Dict]:
    """Top suppliers by amount."""
    if df.empty or COL_SUPPLIER_NAME not in df.columns or AMOUNT_NUMERIC not in df.columns:
        return []
    sub = df[df[FISCAL_YEAR] == year] if year and FISCAL_YEAR in df.columns else df
    if sub.empty:
        return []
    sub = sub[sub[COL_SUPPLIER_NAME].notna() & (sub[COL_SUPPLIER_NAME] != "")]
    if sub.empty:
        return []
    agg = (
        sub.groupby(COL_SUPPLIER_NAME)[AMOUNT_NUMERIC]
        .sum()
        .reset_index()
        .sort_values(AMOUNT_NUMERIC, key=abs, ascending=False)
        .head(10)
    )
    return [
        {"supplier": str(r[COL_SUPPLIER_NAME]), "amount": float(r[AMOUNT_NUMERIC])}
        for _, r in agg.iterrows()
    ]


def build_je_text_summary(df: pd.DataFrame, year: Optional[int] = None) -> List[Dict]:
    """Most frequent journal entry item texts."""
    if df.empty or COL_JE_ITEM_TEXT not in df.columns:
        return []
    sub = df[df[FISCAL_YEAR] == year] if year and FISCAL_YEAR in df.columns else df
    if sub.empty:
        return []
    counts = sub[COL_JE_ITEM_TEXT].dropna().value_counts().head(10)
    return [{"text": str(k), "count": int(v)} for k, v in counts.items()]


def build_flash_summary(df: pd.DataFrame, year: Optional[int] = None) -> List[Dict]:
    """
    Aggregate amounts by Flash Line Item category (from v_glaccount).
    Returns list of {flash_category, amount, records} sorted by absolute amount.
    Falls back to Account Grouping if Flash Line Item is unavailable.
    """
    if df.empty or AMOUNT_NUMERIC not in df.columns:
        return []

    sub = df[df[FISCAL_YEAR] == year] if year and FISCAL_YEAR in df.columns else df
    if sub.empty:
        return []

    # Priority 1: Flash Line Item (authoritative from v_glaccount)
    if COL_FLASH_LINE_ITEM in sub.columns:
        group_col = COL_FLASH_LINE_ITEM
    # Priority 2: FS Item Name
    elif COL_FS_ITEM_NAME in sub.columns:
        group_col = COL_FS_ITEM_NAME
    # Priority 3: Financial Statement Line Item
    elif COL_FS_LINE_ITEM in sub.columns:
        group_col = COL_FS_LINE_ITEM
    # Priority 4: Account Grouping (legacy fallback)
    elif COL_ACCOUNT_GROUPING in sub.columns:
        group_col = COL_ACCOUNT_GROUPING
    else:
        return []

    sub = sub[sub[group_col].notna() & (sub[group_col].astype(str).str.strip() != "")]
    if sub.empty:
        return []

    agg = (
        sub.groupby(group_col)[AMOUNT_NUMERIC]
        .agg(amount="sum", records="count")
        .reset_index()
        .sort_values("amount", key=abs, ascending=False)
        .head(15)
    )
    return [
        {
            "flash_category": str(r[group_col]),
            "amount":         round(float(r["amount"]), 2),
            "records":        int(r["records"]),
            "source":         group_col,
        }
        for _, r in agg.iterrows()
    ]


def build_variance_summary(
    df: pd.DataFrame,
    current_year: int,
    previous_year: int,
    group_col: str = COL_GL_ACCOUNT,
) -> List[Dict]:
    """Year-over-year variance for the filtered dataset."""
    if df.empty or FISCAL_YEAR not in df.columns or AMOUNT_NUMERIC not in df.columns:
        return []
    if group_col not in df.columns:
        return []

    cur  = df[df[FISCAL_YEAR] == current_year].groupby(group_col)[AMOUNT_NUMERIC].sum()
    prev = df[df[FISCAL_YEAR] == previous_year].groupby(group_col)[AMOUNT_NUMERIC].sum()
    combined = cur.rename("current").to_frame().join(
        prev.rename("previous"), how="outer"
    ).fillna(0)
    combined["delta"] = combined["current"] - combined["previous"]
    combined["delta_pct"] = combined.apply(
        lambda r: round((r["delta"] / abs(r["previous"])) * 100, 1)
        if r["previous"] != 0 else None,
        axis=1,
    )
    combined = combined.reindex(
        combined["delta"].abs().sort_values(ascending=False).index
    ).head(10)

    results = []
    for key, row in combined.iterrows():
        results.append({
            "key": str(key),
            "current": round(float(row["current"]), 2),
            "previous": round(float(row["previous"]), 2),
            "delta": round(float(row["delta"]), 2),
            "delta_pct": row["delta_pct"],
        })
    return results