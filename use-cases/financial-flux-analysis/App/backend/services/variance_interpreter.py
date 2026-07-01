"""
Variance Interpreter — accounting-aware variance direction classification.

Business Rule (P&L impact):
  EXPENSE accounts:  decrease (delta < 0) = FAVORABLE  (GREEN)
                     increase (delta > 0) = UNFAVORABLE (RED)
  REVENUE accounts:  increase (delta > 0) = FAVORABLE  (GREEN)
                     decrease (delta < 0) = UNFAVORABLE (RED)
  BALANCE SHEET:     direction is NEUTRAL — no direct P&L impact
  UNKNOWN / MIXED:   direction is NEUTRAL — cannot determine without SAP data

Account nature is determined EXCLUSIVELY from SAP master data fields already
loaded into the enriched DataFrame:
  - G/L Account Type          (from v_glaccount / v__glaccountgrouping)
  - FS Item Name              (from v_glaccount)
  - Financial Statement Line Item (from v_glaccount)
  - Account Grouping          (from v__glaccountgrouping)
  - Flash Line Item           (from v_glaccount)

NO hardcoded account lists.
NO amount-sign heuristics.
NO synthetic mappings.
NO fallbacks that assume "positive delta = good".
"""

import logging
from enum import Enum
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Account Nature Enum
# ---------------------------------------------------------------------------

class AccountNature(str, Enum):
    EXPENSE       = "EXPENSE"
    REVENUE       = "REVENUE"
    BALANCE_SHEET = "BALANCE_SHEET"
    MIXED         = "MIXED"       # grouping contains both expense and revenue
    UNKNOWN       = "UNKNOWN"     # SAP data insufficient to classify


# ---------------------------------------------------------------------------
# SAP keyword sets — derived from standard SAP Financial Statement Version
# terminology. Case-insensitive substring matching.
# ---------------------------------------------------------------------------

# Substrings that indicate a REVENUE / INCOME account
_REVENUE_KEYWORDS = frozenset([
    "revenue",
    "income",
    "sales",
    "turnover",
    "gain",
    "proceeds",
    "receipts",
    "interest income",
    "dividend",
    "rental income",
    "other income",
    "net revenue",
    "gross revenue",
    "service revenue",
    "product revenue",
    "operating income",
])

# Substrings that indicate an EXPENSE / COST account
_EXPENSE_KEYWORDS = frozenset([
    "expense",
    "cost",
    "depreciation",
    "amortization",
    "loss",
    "charge",
    "write-off",
    "write-down",
    "write off",
    "write down",
    "impairment",
    "provision",
    "wages",
    "salary",
    "salaries",
    "labor",
    "labour",
    "payroll",
    "rent",
    "utilities",
    "maintenance",
    "material",
    "supplies",
    "insurance",
    "travel",
    "advertising",
    "marketing",
    "research",
    "development",
    "overhead",
    "interest expense",
    "tax expense",
    "income tax",
    "operating expense",
    "selling expense",
    "general and administrative",
    "g&a",
    "cogs",
    "cost of goods",
    "cost of sales",
    "freight",
    "logistics",
    "consulting",
    "professional fees",
    "legal",
    "audit",
    "it expense",
    "software",
    "hardware",
    "training",
    "recruitment",
    "benefits",
    "pension",
    "bonus",
    "commission",
])

# SAP G/L Account Type values that indicate Balance Sheet accounts
_BALANCE_SHEET_ACCOUNT_TYPES = frozenset(["X", "B", "BS", "BALANCE", "BALANCE SHEET"])

# SAP G/L Account Type values that indicate P&L accounts
_PNL_ACCOUNT_TYPES = frozenset(["P", "P&L", "PL", "INCOME STATEMENT", "IS"])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _keyword_match(text: str, keywords: frozenset) -> bool:
    """Return True if any keyword is a substring of text (case-insensitive)."""
    if not text:
        return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def _classify_from_text_fields(
    fs_item_name: str,
    fs_line_item: str,
    account_grouping: str,
    flash_line_item: str,
    gl_account_key: str,
) -> Optional[AccountNature]:
    """
    Attempt to classify account nature from SAP text fields using keyword matching.
    Returns None if classification is inconclusive.
    """
    # Combine all available text fields for matching
    # Priority: FS Item Name > Financial Statement Line Item > Account Grouping > Flash Line Item
    fields_to_check = [
        ("FS Item Name",                  fs_item_name),
        ("Financial Statement Line Item", fs_line_item),
        ("Account Grouping",              account_grouping),
        ("Flash Line Item",               flash_line_item),
    ]

    for field_name, field_value in fields_to_check:
        if not field_value or str(field_value).strip() in ("", "nan", "None"):
            continue

        val = str(field_value).strip()

        is_revenue = _keyword_match(val, _REVENUE_KEYWORDS)
        is_expense = _keyword_match(val, _EXPENSE_KEYWORDS)

        if is_revenue and not is_expense:
            logger.debug(
                "Account '%s': classified as REVENUE via %s='%s'",
                gl_account_key, field_name, val,
            )
            return AccountNature.REVENUE

        if is_expense and not is_revenue:
            logger.debug(
                "Account '%s': classified as EXPENSE via %s='%s'",
                gl_account_key, field_name, val,
            )
            return AccountNature.EXPENSE

        if is_revenue and is_expense:
            # Ambiguous field — try next field
            logger.debug(
                "Account '%s': ambiguous keywords in %s='%s' — trying next field.",
                gl_account_key, field_name, val,
            )
            continue

    return None  # inconclusive


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_account_nature(
    gl_account_key: str,
    account_type: Optional[str],
    fs_item_name: Optional[str],
    fs_line_item: Optional[str],
    account_grouping: Optional[str],
    flash_line_item: Optional[str],
) -> AccountNature:
    """
    Classify the nature of a G/L account using SAP master data fields.

    Classification priority:
      1. G/L Account Type — if explicitly "X" (Balance Sheet), return BALANCE_SHEET
      2. Text field keyword matching (FS Item Name, FS Line Item, Account Grouping, Flash)
      3. If G/L Account Type is "P" (P&L) but text fields are inconclusive → UNKNOWN
      4. If no SAP data available → UNKNOWN

    Args:
        gl_account_key:   G/L Account number (for logging only)
        account_type:     SAP G/L Account Type field value
        fs_item_name:     FS Item Name from v_glaccount
        fs_line_item:     Financial Statement Line Item from v_glaccount
        account_grouping: Account Grouping from v__glaccountgrouping
        flash_line_item:  Flash Line Item from v_glaccount

    Returns:
        AccountNature enum value
    """
    # Normalize inputs
    acct_type = str(account_type).strip().upper() if account_type else ""

    # Step 1: Check G/L Account Type for Balance Sheet
    if acct_type in _BALANCE_SHEET_ACCOUNT_TYPES:
        logger.debug(
            "Account '%s': classified as BALANCE_SHEET via G/L Account Type='%s'.",
            gl_account_key, acct_type,
        )
        return AccountNature.BALANCE_SHEET

    # Step 2: Keyword matching on text fields
    nature = _classify_from_text_fields(
        fs_item_name     = str(fs_item_name     or "").strip(),
        fs_line_item     = str(fs_line_item     or "").strip(),
        account_grouping = str(account_grouping or "").strip(),
        flash_line_item  = str(flash_line_item  or "").strip(),
        gl_account_key   = gl_account_key,
    )

    if nature is not None:
        logger.debug(
            "Account '%s': classified as %s via keyword matching "
            "(G/L Account Type='%s', FS Item Name='%s', FS Line Item='%s', "
            "Account Grouping='%s', Flash='%s').",
            gl_account_key, nature.value, acct_type,
            fs_item_name or "", fs_line_item or "",
            account_grouping or "", flash_line_item or "",
        )
        return nature

    # Step 3: G/L Account Type is P&L but text fields inconclusive
    if acct_type in _PNL_ACCOUNT_TYPES:
        logger.warning(
            "Account '%s': G/L Account Type='%s' (P&L) but text fields are "
            "inconclusive for expense/revenue classification. "
            "Returning UNKNOWN. SAP master data may need enrichment.",
            gl_account_key, acct_type,
        )
        return AccountNature.UNKNOWN

    # Step 4: No usable SAP data
    logger.warning(
        "Account '%s': insufficient SAP master data to classify account nature "
        "(G/L Account Type='%s', all text fields empty/null). Returning UNKNOWN.",
        gl_account_key, acct_type,
    )
    return AccountNature.UNKNOWN


def compute_variance_direction(
    delta: float,
    account_nature: AccountNature,
    gl_account_key: str = "",
) -> Dict:
    """
    Compute whether a variance is favorable or unfavorable based on
    account nature and P&L impact.

    Business Rules:
      EXPENSE:       delta < 0 → FAVORABLE  (lower expense = better profitability)
                     delta > 0 → UNFAVORABLE (higher expense = worse profitability)
      REVENUE:       delta > 0 → FAVORABLE  (higher revenue = better profitability)
                     delta < 0 → UNFAVORABLE (lower revenue = worse profitability)
      BALANCE_SHEET: NEUTRAL (no direct P&L impact)
      MIXED/UNKNOWN: NEUTRAL (cannot determine without full account breakdown)

    Args:
        delta:          Variance amount (current - previous)
        account_nature: AccountNature enum value
        gl_account_key: Account key (for logging only)

    Returns:
        dict with keys:
          is_favorable      (bool | None)  — True=favorable, False=unfavorable, None=neutral
          direction_label   (str)          — "FAVORABLE" | "UNFAVORABLE" | "NEUTRAL"
          narrative_adjective (str)        — human-readable adjective for narratives
          narrative_context   (str)        — accounting context phrase for AI prompts
    """
    if delta == 0.0:
        return {
            "is_favorable":       None,
            "direction_label":    "NEUTRAL",
            "narrative_adjective": "unchanged",
            "narrative_context":   "no change versus prior year",
        }

    if account_nature == AccountNature.EXPENSE:
        if delta < 0:
            result = {
                "is_favorable":       True,
                "direction_label":    "FAVORABLE",
                "narrative_adjective": "favorable",
                "narrative_context":   (
                    "expense decreased versus prior year — favorable impact on Net Income. "
                    "Lower expense burden improves profitability."
                ),
            }
        else:
            result = {
                "is_favorable":       False,
                "direction_label":    "UNFAVORABLE",
                "narrative_adjective": "unfavorable",
                "narrative_context":   (
                    "expense increased versus prior year — unfavorable impact on Net Income. "
                    "Higher expense burden reduces profitability."
                ),
            }
        logger.debug(
            "Account '%s' [EXPENSE]: delta=%.2f → %s",
            gl_account_key, delta, result["direction_label"],
        )
        return result

    if account_nature == AccountNature.REVENUE:
        if delta > 0:
            result = {
                "is_favorable":       True,
                "direction_label":    "FAVORABLE",
                "narrative_adjective": "favorable",
                "narrative_context":   (
                    "revenue increased versus prior year — favorable impact on Net Income. "
                    "Higher revenue improves profitability."
                ),
            }
        else:
            result = {
                "is_favorable":       False,
                "direction_label":    "UNFAVORABLE",
                "narrative_adjective": "unfavorable",
                "narrative_context":   (
                    "revenue decreased versus prior year — unfavorable impact on Net Income. "
                    "Lower revenue reduces profitability."
                ),
            }
        logger.debug(
            "Account '%s' [REVENUE]: delta=%.2f → %s",
            gl_account_key, delta, result["direction_label"],
        )
        return result

    # BALANCE_SHEET, MIXED, UNKNOWN — neutral
    logger.debug(
        "Account '%s' [%s]: delta=%.2f → NEUTRAL (no P&L direction assigned).",
        gl_account_key, account_nature.value, delta,
    )
    return {
        "is_favorable":       None,
        "direction_label":    "NEUTRAL",
        "narrative_adjective": "neutral",
        "narrative_context":   (
            f"account nature is {account_nature.value} — "
            "P&L impact direction cannot be determined from available SAP data."
        ),
    }


def build_account_nature_lookup(
    df: pd.DataFrame,
    key_col: str,
    account_type_col: str   = "G/L Account Type",
    fs_item_name_col: str   = "FS Item Name",
    fs_line_item_col: str   = "Financial Statement Line Item",
    account_grouping_col: str = "Account Grouping",
    flash_line_item_col: str  = "Flash Line Item",
) -> Dict[str, AccountNature]:
    """
    Build a lookup dict {key → AccountNature} from the enriched DataFrame.

    For each unique key value, takes the first non-null row's master data fields
    and classifies the account nature.

    Only meaningful when key_col is G/L Account. For other dimensions
    (Profit Center, Cost Center, etc.) the result will be UNKNOWN for most keys
    since those dimensions aggregate multiple account types.

    Args:
        df:                  Enriched DataFrame
        key_col:             Column to group by (e.g., "G/L Account")
        account_type_col:    Column name for G/L Account Type
        fs_item_name_col:    Column name for FS Item Name
        fs_line_item_col:    Column name for Financial Statement Line Item
        account_grouping_col: Column name for Account Grouping
        flash_line_item_col: Column name for Flash Line Item

    Returns:
        Dict mapping each key value to its AccountNature
    """
    if df.empty or key_col not in df.columns:
        return {}

    # Build a deduplicated master data view (one row per key)
    master_cols = [key_col]
    for col in [account_type_col, fs_item_name_col, fs_line_item_col,
                account_grouping_col, flash_line_item_col]:
        if col in df.columns:
            master_cols.append(col)

    master_df = (
        df[master_cols]
        .dropna(subset=[key_col])
        .drop_duplicates(subset=[key_col], keep="first")
        .set_index(key_col)
    )

    lookup: Dict[str, AccountNature] = {}
    for key_val, row in master_df.iterrows():
        key_str = str(key_val).strip()
        nature = classify_account_nature(
            gl_account_key   = key_str,
            account_type     = row.get(account_type_col)     if account_type_col     in row.index else None,
            fs_item_name     = row.get(fs_item_name_col)     if fs_item_name_col     in row.index else None,
            fs_line_item     = row.get(fs_line_item_col)     if fs_line_item_col     in row.index else None,
            account_grouping = row.get(account_grouping_col) if account_grouping_col in row.index else None,
            flash_line_item  = row.get(flash_line_item_col)  if flash_line_item_col  in row.index else None,
        )
        lookup[key_str] = nature

    # Summary log
    from collections import Counter
    counts = Counter(n.value for n in lookup.values())
    logger.info(
        "build_account_nature_lookup: %d accounts classified — %s",
        len(lookup),
        dict(counts),
    )
    return lookup