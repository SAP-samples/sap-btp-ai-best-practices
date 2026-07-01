"""
Dataset Enricher — builds the analytical enriched DataFrame from raw source tables.

Pipeline:
  STEP 1  Normalize all ID columns (string, trim, fix float artefacts)
  STEP 2  Parse mixed "ID (Description)" fields in master data tables
  STEP 3  Deduplicate master data tables (prevent join explosion)
  STEP 4  LEFT JOIN basedata ← gl_grouping  (on G/L Account)
  STEP 5  LEFT JOIN result   ← profitcenter (on Profit Center)
  STEP 6  LEFT JOIN result   ← costcenter   (on Cost Center)
  STEP 7  Derive fiscal_year, fiscal_month from Posting Date
  STEP 8  Parse amount → amount_numeric
  STEP 9  Validate and return enriched DataFrame

All joins are LEFT JOINs — no transactional records are ever dropped.
All IDs remain exactly as they appear in the SAP source.
"""

import logging
from typing import Dict

import pandas as pd

from normalizers.id_normalizer import normalize_id_columns
from normalizers.text_parser import split_id_description_column
from utils.currency_parser import apply_currency_parser
from validators.data_validator import (
    validate_enriched_dataset,
    validate_join_integrity,
    validate_null_keys,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column name constants — source tables
# ---------------------------------------------------------------------------

# v_basedata columns
BD_GL_ACCOUNT      = "G/L Account"
BD_POSTING_DATE    = "Posting Date"
BD_AMOUNT          = "Amount in Company Code Currency"
BD_COMPANY_CODE    = "Company Code"
BD_PROFIT_CENTER   = "Profit Center"
BD_COST_CENTER     = "Cost Center"
BD_WBS_ELEMENT     = "WBS Element"
BD_PURCH_DOC       = "Purchasing Document"
BD_FISCAL_PERIOD   = "Fiscal Period"
BD_JE_TYPE         = "Journal Entry Type"
BD_JE_TYPE_NAME    = "JE Type Name"
BD_JE_ITEM_TEXT    = "Journal Entry Item Text"
BD_ASSIGNMENT_REF  = "Assignment Reference"
BD_SUPPLIER        = "Supplier"
BD_SUPPLIER_NAME   = "Name of Supplier"
BD_SEGMENT         = "Segment"
BD_PARTNER_PC      = "Partner Profit Center"

# v__glaccountgrouping columns
GL_EXT_ID           = "G/L Acct External ID"   # join key → matches BD_GL_ACCOUNT
GL_LONG_TEXT        = "G/L Account Long Text"
GL_ACCOUNT_TYPE     = "G/L Account Type"
GL_FS_ITEM          = "FS Item"
GL_FS_ITEM_NAME     = "FS Item Name"
GL_FS_LINE_ITEM     = "Financial Statement Line Item"
GL_FLASH_LINE_ITEM  = "Flash Line Item"
GL_ACCOUNT_GROUPING = "Account Grouping"

# v_glaccount columns (AUTHORITATIVE for Flash Line Item and FS hierarchy)
GLA_EXT_ID          = "G/L Acct External ID"   # join key → matches BD_GL_ACCOUNT
GLA_LONG_TEXT       = "G/L Account Long Text"
GLA_ACCOUNT_TYPE    = "G/L Account Type"
GLA_FS_ITEM         = "FS Item"
GLA_FS_ITEM_NAME    = "FS Item Name"
GLA_FS_LINE_ITEM    = "Financial Statement Line Item"
GLA_FLASH_LINE_ITEM = "Flash Line Item"          # PRIMARY Flash grouping source

# v_profitcenter columns
PC_PROFIT_CENTER   = "Profit Center"           # join key
PC_CONSUNIT        = "CONSUNIT"
PC_CONSUNIT_NAME   = "CONSUNIT NAME"
PC_SEGMENT         = "Segment"

# v_costcenter columns
CC_COST_CENTER     = "Cost Center"             # join key
CC_NAME            = "Name"
CC_FUNCTIONAL_AREA = "Functional Area"
CC_SEGMENT         = "Segment"

# ---------------------------------------------------------------------------
# Output column names (must match financial_processor.py constants)
# ---------------------------------------------------------------------------
OUT_GL_ACCOUNT      = "G/L Account"
OUT_GL_ACCOUNT_NAME = "G/L Account Name"
OUT_POSTING_DATE    = "Posting Date"
OUT_AMOUNT          = "Amount in Company Code Currency"
OUT_COMPANY_CODE    = "Company Code"
OUT_PROFIT_CENTER   = "Profit Center"
OUT_COST_CENTER     = "Cost Center"
OUT_WBS_ELEMENT     = "WBS Element"
OUT_PURCH_DOC       = "Purchasing Document"
OUT_FISCAL_PERIOD   = "Fiscal Period"
OUT_JE_TYPE         = "Journal Entry Type"
OUT_JE_TYPE_NAME    = "JE Type Name"
OUT_JE_ITEM_TEXT    = "Journal Entry Item Text"
OUT_ASSIGNMENT_REF  = "Assignment Reference"
OUT_SUPPLIER        = "Supplier"
OUT_SUPPLIER_NAME   = "Name of Supplier"
OUT_SEGMENT         = "Segment"
OUT_PARTNER_PC      = "Partner Profit Center"
OUT_ACCOUNT_TYPE    = "G/L Account Type"
OUT_FS_ITEM         = "FS Item"
OUT_FS_ITEM_NAME    = "FS Item Name"
OUT_FS_LINE_ITEM    = "Financial Statement Line Item"
OUT_FLASH_LINE_ITEM = "Flash Line Item"
OUT_ACCOUNT_GROUPING = "Account Grouping"
OUT_FUNCTIONAL_AREA = "Functional Area Name"
OUT_CONSUNIT        = "CONSUNIT"
OUT_CONSUNIT_NAME   = "CONSUNIT NAME"
AMOUNT_NUMERIC      = "amount_numeric"
FISCAL_YEAR         = "fiscal_year"
FISCAL_MONTH        = "fiscal_month"

# ID columns that must always be treated as strings
_BASEDATA_ID_COLS = [
    BD_GL_ACCOUNT, BD_COMPANY_CODE, BD_PROFIT_CENTER, BD_COST_CENTER,
    BD_WBS_ELEMENT, BD_PURCH_DOC, BD_PARTNER_PC,
]
_GL_ID_COLS   = [GL_EXT_ID]
_GLA_ID_COLS  = [GLA_EXT_ID]
_PC_ID_COLS   = [PC_PROFIT_CENTER]
_CC_ID_COLS   = [CC_COST_CENTER]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _first_existing(df: pd.DataFrame, candidates: list, default: str = "") -> str:
    """Return the first column name from candidates that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return default


def _safe_left_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_key: str,
    right_key: str,
    join_name: str,
    suffixes=("", "_right"),
) -> pd.DataFrame:
    """
    Perform a LEFT JOIN and validate integrity.
    Drops duplicate right-side key column after merge.
    """
    if right.empty:
        logger.warning("JOIN '%s': right-side table is empty — skipping.", join_name)
        return left

    before_count = len(left)

    result = left.merge(
        right,
        left_on=left_key,
        right_on=right_key,
        how="left",
        suffixes=suffixes,
    )

    # Drop the right-side key column if it's a duplicate
    if right_key != left_key and right_key in result.columns:
        result = result.drop(columns=[right_key])

    # Drop any "_right" suffix columns (duplicates from merge)
    right_cols = [c for c in result.columns if c.endswith("_right")]
    if right_cols:
        result = result.drop(columns=right_cols)

    validate_join_integrity(
        left_df=pd.DataFrame(index=range(before_count)),
        result_df=result,
        join_key=left_key,
        join_name=join_name,
    )

    return result


# ---------------------------------------------------------------------------
# Master data preparation
# ---------------------------------------------------------------------------

def _prepare_glaccount(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize and deduplicate the v_glaccount table.
    This is the AUTHORITATIVE source for Flash Line Item and FS hierarchy.
    """
    if df.empty:
        return df

    df = df.copy()

    # Parse GLA_EXT_ID if it contains mixed "ID (Description)" format
    if GLA_EXT_ID in df.columns:
        parsed = split_id_description_column(
            df[GLA_EXT_ID],
            id_col_name="_gla_id_parsed",
            desc_col_name="_gla_desc_parsed",
        )
        df["_gla_id_parsed"]   = parsed["_gla_id_parsed"]
        df["_gla_desc_parsed"] = parsed["_gla_desc_parsed"]

        if GLA_LONG_TEXT not in df.columns or df[GLA_LONG_TEXT].isna().all():
            df[GLA_LONG_TEXT] = df["_gla_desc_parsed"]

        df[GLA_EXT_ID] = df["_gla_id_parsed"]
        df = df.drop(columns=["_gla_id_parsed", "_gla_desc_parsed"])

    df = normalize_id_columns(df, _GLA_ID_COLS)

    # Deduplicate on join key
    before = len(df)
    df = df.drop_duplicates(subset=[GLA_EXT_ID], keep="first")
    if len(df) < before:
        logger.info(
            "v_glaccount: deduplicated %d → %d rows on '%s'.",
            before, len(df), GLA_EXT_ID,
        )

    # Log Flash Line Item coverage
    if GLA_FLASH_LINE_ITEM in df.columns:
        null_flash = df[GLA_FLASH_LINE_ITEM].isna().sum()
        total = len(df)
        logger.info(
            "v_glaccount: Flash Line Item coverage %d/%d accounts (%.1f%%).",
            total - null_flash, total,
            100.0 * (total - null_flash) / total if total > 0 else 0,
        )

    return df


def _prepare_gl_grouping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize and deduplicate the GL account grouping table.

    The GL_EXT_ID column may contain mixed "ID (Description)" values.
    We parse it to extract the clean account ID.
    """
    if df.empty:
        return df

    df = df.copy()

    # Parse GL_EXT_ID if it contains mixed format
    if GL_EXT_ID in df.columns:
        parsed = split_id_description_column(
            df[GL_EXT_ID],
            id_col_name="_gl_id_parsed",
            desc_col_name="_gl_desc_parsed",
        )
        # Use parsed ID as the join key; if description was embedded and
        # GL_LONG_TEXT is missing, use the parsed description as fallback
        df["_gl_id_parsed"]   = parsed["_gl_id_parsed"]
        df["_gl_desc_parsed"] = parsed["_gl_desc_parsed"]

        if GL_LONG_TEXT not in df.columns or df[GL_LONG_TEXT].isna().all():
            df[GL_LONG_TEXT] = df["_gl_desc_parsed"]

        # Replace the original column with the clean ID
        df[GL_EXT_ID] = df["_gl_id_parsed"]
        df = df.drop(columns=["_gl_id_parsed", "_gl_desc_parsed"])

    # Normalize ID column
    df = normalize_id_columns(df, _GL_ID_COLS)

    # Deduplicate on join key — keep first occurrence
    before = len(df)
    df = df.drop_duplicates(subset=[GL_EXT_ID], keep="first")
    if len(df) < before:
        logger.info(
            "GL grouping: deduplicated %d → %d rows on '%s'.",
            before, len(df), GL_EXT_ID,
        )

    return df


def _prepare_profitcenter(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and deduplicate the profit center master data table."""
    if df.empty:
        return df
    df = df.copy()
    df = normalize_id_columns(df, _PC_ID_COLS)
    before = len(df)
    df = df.drop_duplicates(subset=[PC_PROFIT_CENTER], keep="first")
    if len(df) < before:
        logger.info(
            "Profit center: deduplicated %d → %d rows.", before, len(df)
        )
    return df


def _prepare_costcenter(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and deduplicate the cost center master data table."""
    if df.empty:
        return df
    df = df.copy()
    df = normalize_id_columns(df, _CC_ID_COLS)
    before = len(df)
    df = df.drop_duplicates(subset=[CC_COST_CENTER], keep="first")
    if len(df) < before:
        logger.info(
            "Cost center: deduplicated %d → %d rows.", before, len(df)
        )
    return df


# ---------------------------------------------------------------------------
# Main enrichment pipeline
# ---------------------------------------------------------------------------

def build_enriched_dataset(raw_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build the fully enriched analytical DataFrame from the five raw source tables.

    Args:
        raw_tables: dict with keys
          'basedata', 'profitcenter', 'costcenter', 'gl_grouping', 'glaccount'

    Returns:
        Enriched DataFrame with all columns needed by FinancialProcessor.
        Column names match the existing API contracts exactly.
        Flash Line Item comes from v_glaccount (authoritative).
    """
    basedata     = raw_tables.get("basedata",     pd.DataFrame())
    profitcenter = raw_tables.get("profitcenter", pd.DataFrame())
    costcenter   = raw_tables.get("costcenter",   pd.DataFrame())
    gl_grouping  = raw_tables.get("gl_grouping",  pd.DataFrame())
    glaccount    = raw_tables.get("glaccount",    pd.DataFrame())

    if basedata.empty:
        logger.error("build_enriched_dataset: basedata is empty — cannot build dataset.")
        return pd.DataFrame()

    logger.info("build_enriched_dataset: starting pipeline with %d base rows.", len(basedata))

    # ------------------------------------------------------------------
    # STEP 1 — Normalize basedata ID columns
    # ------------------------------------------------------------------
    df = normalize_id_columns(basedata.copy(), _BASEDATA_ID_COLS)

    # ------------------------------------------------------------------
    # STEP 1b — Parse mixed "ID (Description)" format in G/L Account
    #
    # v_basedata may contain: "61002000 (3rd party material)"
    # v__glaccountgrouping contains: "61002000"
    #
    # We MUST extract the pure account ID before any join.
    # The description is stored separately and NEVER used as a join key.
    # Leading zeros are preserved. No numeric casting.
    # ------------------------------------------------------------------
    if BD_GL_ACCOUNT in df.columns:
        sample_vals = df[BD_GL_ACCOUNT].dropna().astype(str).head(5).tolist()
        has_mixed = any(" " in v or "(" in v for v in sample_vals)

        if has_mixed:
            parsed_gl = split_id_description_column(
                df[BD_GL_ACCOUNT],
                id_col_name="_gl_id_clean",
                desc_col_name="_gl_desc_from_basedata",
            )
            df["_gl_id_clean"]          = parsed_gl["_gl_id_clean"]
            df["_gl_desc_from_basedata"] = parsed_gl["_gl_desc_from_basedata"]

            # Log the parsing
            changed = df["_gl_id_clean"] != df[BD_GL_ACCOUNT].astype(str)
            if changed.any():
                samples = list(zip(
                    df.loc[changed, BD_GL_ACCOUNT].head(3).tolist(),
                    df.loc[changed, "_gl_id_clean"].head(3).tolist(),
                ))
                logger.info(
                    "Parsed mixed G/L Account values in v_basedata: %s "
                    "(%d values normalized for join)",
                    samples, int(changed.sum()),
                )

            # Replace the join key with the clean ID
            # Keep the original mixed value in a separate column for display
            df["_gl_account_original"] = df[BD_GL_ACCOUNT].copy()
            df[BD_GL_ACCOUNT]          = df["_gl_id_clean"]
            df = df.drop(columns=["_gl_id_clean"])

            # If GL Account Name is not yet available, use the parsed description
            if OUT_GL_ACCOUNT_NAME not in df.columns:
                df[OUT_GL_ACCOUNT_NAME] = df["_gl_desc_from_basedata"]
            df = df.drop(columns=["_gl_desc_from_basedata"])
        else:
            logger.debug("G/L Account in v_basedata: no mixed format detected — no parsing needed.")

    # ------------------------------------------------------------------
    # STEP 1c — Parse mixed "ID (Description)" format in Profit Center
    #           and Cost Center (same logic as STEP 1b for G/L Account)
    # ------------------------------------------------------------------
    for _bd_col in [BD_PROFIT_CENTER, BD_COST_CENTER]:
        if _bd_col in df.columns:
            _sample = df[_bd_col].dropna().astype(str).head(5).tolist()
            _has_mixed = any(" " in v or "(" in v for v in _sample)
            if _has_mixed:
                _parsed = split_id_description_column(
                    df[_bd_col],
                    id_col_name=f"_{_bd_col}_id_clean",
                    desc_col_name=f"_{_bd_col}_desc",
                )
                _changed = _parsed[f"_{_bd_col}_id_clean"] != df[_bd_col].astype(str)
                if _changed.any():
                    logger.info(
                        "Parsed mixed '%s' values in v_basedata: %d values normalized for join.",
                        _bd_col, int(_changed.sum()),
                    )
                df[_bd_col] = _parsed[f"_{_bd_col}_id_clean"]

    # Validate null keys before joins
    validate_null_keys(df, BD_GL_ACCOUNT,    context="basedata")
    validate_null_keys(df, BD_PROFIT_CENTER, context="basedata")
    validate_null_keys(df, BD_COST_CENTER,   context="basedata")

    # ------------------------------------------------------------------
    # STEP 2 — Parse Posting Date → fiscal_year, fiscal_month
    # ------------------------------------------------------------------
    if BD_POSTING_DATE in df.columns:
        df[BD_POSTING_DATE] = pd.to_datetime(df[BD_POSTING_DATE], errors="coerce")
        df[FISCAL_YEAR]  = df[BD_POSTING_DATE].dt.year.astype("Int64")
        df[FISCAL_MONTH] = df[BD_POSTING_DATE].dt.month.astype("Int64")
    else:
        logger.error("'%s' column not found in basedata.", BD_POSTING_DATE)
        df[FISCAL_YEAR]  = pd.NA
        df[FISCAL_MONTH] = pd.NA

    # ------------------------------------------------------------------
    # STEP 3 — Parse amount → amount_numeric
    # ------------------------------------------------------------------
    if BD_AMOUNT in df.columns:
        df[AMOUNT_NUMERIC] = apply_currency_parser(df[BD_AMOUNT])
    else:
        logger.warning("'%s' column not found in basedata.", BD_AMOUNT)
        df[AMOUNT_NUMERIC] = 0.0

    # ------------------------------------------------------------------
    # STEP 4 — Prepare master data tables
    # ------------------------------------------------------------------
    gla_prep = _prepare_glaccount(gl_grouping if glaccount.empty else glaccount)
    gl_prep  = _prepare_gl_grouping(gl_grouping)
    pc_prep  = _prepare_profitcenter(profitcenter)
    cc_prep  = _prepare_costcenter(costcenter)

    # ------------------------------------------------------------------
    # STEP 5a — LEFT JOIN: basedata ← v_glaccount (AUTHORITATIVE)
    #           Brings: Flash Line Item, FS Item, FS Item Name,
    #                   Financial Statement Line Item, G/L Account Type,
    #                   G/L Account Long Text
    # ------------------------------------------------------------------
    if not gla_prep.empty and GLA_EXT_ID in gla_prep.columns:
        gla_cols = [GLA_EXT_ID]
        for c in [GLA_LONG_TEXT, GLA_ACCOUNT_TYPE, GLA_FS_ITEM,
                  GLA_FS_ITEM_NAME, GLA_FS_LINE_ITEM, GLA_FLASH_LINE_ITEM]:
            if c in gla_prep.columns:
                gla_cols.append(c)
        gla_subset = gla_prep[gla_cols].copy()

        df = _safe_left_join(
            left=df, right=gla_subset,
            left_key=BD_GL_ACCOUNT, right_key=GLA_EXT_ID,
            join_name="basedata←v_glaccount",
        )

        # Log join match rate
        if GLA_FLASH_LINE_ITEM in df.columns:
            matched = df[GLA_FLASH_LINE_ITEM].notna().sum()
            logger.info(
                "v_glaccount join: %d/%d rows matched Flash Line Item (%.1f%%).",
                matched, len(df), 100.0 * matched / len(df) if len(df) > 0 else 0,
            )
    else:
        logger.warning("v_glaccount table empty or missing key — Flash Line Item unavailable.")

    # ------------------------------------------------------------------
    # STEP 5b — LEFT JOIN: basedata ← v__glaccountgrouping (FALLBACK)
    #           Only brings columns NOT already provided by v_glaccount.
    #           Priority: v_glaccount > v__glaccountgrouping
    # ------------------------------------------------------------------
    if not gl_prep.empty and GL_EXT_ID in gl_prep.columns:
        # Only bring columns from gl_grouping that are not already in df
        gl_cols = [GL_EXT_ID]
        for c in [GL_LONG_TEXT, GL_ACCOUNT_TYPE, GL_FS_ITEM, GL_FS_ITEM_NAME,
                  GL_FS_LINE_ITEM, GL_FLASH_LINE_ITEM, GL_ACCOUNT_GROUPING]:
            if c in gl_prep.columns and c not in df.columns:
                gl_cols.append(c)

        if len(gl_cols) > 1:  # more than just the key
            gl_subset = gl_prep[gl_cols].copy()
            # Rename to avoid collision with v_glaccount columns
            rename_fallback = {c: f"_glg_{c}" for c in gl_cols if c != GL_EXT_ID}
            gl_subset = gl_subset.rename(columns=rename_fallback)

            df = _safe_left_join(
                left=df, right=gl_subset,
                left_key=BD_GL_ACCOUNT, right_key=GL_EXT_ID,
                join_name="basedata←v__glaccountgrouping(fallback)",
            )

            # Promote fallback values only where primary is null
            for orig_col, fallback_col in rename_fallback.items():
                if fallback_col in df.columns:
                    if orig_col not in df.columns:
                        df = df.rename(columns={fallback_col: orig_col})
                    else:
                        # Fill nulls in primary with fallback
                        null_mask = df[orig_col].isna() | (df[orig_col].astype(str).str.strip() == "")
                        df.loc[null_mask, orig_col] = df.loc[null_mask, fallback_col]
                        df = df.drop(columns=[fallback_col])
        else:
            logger.info("v__glaccountgrouping: all columns already provided by v_glaccount — skipping.")
    else:
        logger.warning("GL grouping table empty or missing key — skipping fallback GL enrichment.")

    # ------------------------------------------------------------------
    # STEP 6 — LEFT JOIN: result ← profitcenter (on Profit Center)
    # ------------------------------------------------------------------
    if not pc_prep.empty and PC_PROFIT_CENTER in pc_prep.columns:
        pc_cols = [PC_PROFIT_CENTER]
        for c in [PC_CONSUNIT, PC_CONSUNIT_NAME, PC_SEGMENT]:
            if c in pc_prep.columns:
                pc_cols.append(c)
        pc_subset = pc_prep[pc_cols].copy()

        # Rename Segment to avoid collision with basedata Segment
        if PC_SEGMENT in pc_subset.columns:
            pc_subset = pc_subset.rename(columns={PC_SEGMENT: "_pc_segment"})

        df = _safe_left_join(
            left=df, right=pc_subset,
            left_key=BD_PROFIT_CENTER, right_key=PC_PROFIT_CENTER,
            join_name="result←profitcenter",
        )
    else:
        logger.warning("Profit center table empty or missing key — skipping PC enrichment.")

    # ------------------------------------------------------------------
    # STEP 7 — LEFT JOIN: result ← costcenter (on Cost Center)
    # ------------------------------------------------------------------
    if not cc_prep.empty and CC_COST_CENTER in cc_prep.columns:
        cc_cols = [CC_COST_CENTER]
        for c in [CC_NAME, CC_FUNCTIONAL_AREA, CC_SEGMENT]:
            if c in cc_prep.columns:
                cc_cols.append(c)
        cc_subset = cc_prep[cc_cols].copy()

        # Rename to avoid collision
        rename_map = {}
        if CC_NAME in cc_subset.columns:
            rename_map[CC_NAME] = "Cost Center Name"
        if CC_SEGMENT in cc_subset.columns:
            rename_map[CC_SEGMENT] = "_cc_segment"
        if CC_FUNCTIONAL_AREA in cc_subset.columns:
            rename_map[CC_FUNCTIONAL_AREA] = OUT_FUNCTIONAL_AREA
        if rename_map:
            cc_subset = cc_subset.rename(columns=rename_map)

        df = _safe_left_join(
            left=df, right=cc_subset,
            left_key=BD_COST_CENTER, right_key=CC_COST_CENTER,
            join_name="result←costcenter",
        )
    else:
        logger.warning("Cost center table empty or missing key — skipping CC enrichment.")

    # ------------------------------------------------------------------
    # STEP 8 — Resolve Segment (prefer profit center, fall back to cost center / basedata)
    # ------------------------------------------------------------------
    if "_pc_segment" in df.columns or "_cc_segment" in df.columns or BD_SEGMENT in df.columns:
        def _resolve_segment(row):
            for col in ["_pc_segment", "_cc_segment", BD_SEGMENT]:
                val = row.get(col, "")
                if val and str(val).strip() and str(val).strip().lower() not in ("nan", "none"):
                    return str(val).strip()
            return ""
        df[OUT_SEGMENT] = df.apply(_resolve_segment, axis=1)
        # Drop temp columns
        for tmp in ["_pc_segment", "_cc_segment"]:
            if tmp in df.columns:
                df = df.drop(columns=[tmp])
    elif BD_SEGMENT in df.columns:
        df[OUT_SEGMENT] = df[BD_SEGMENT]

    # ------------------------------------------------------------------
    # STEP 9 — Rename enriched columns to match existing API contracts
    # ------------------------------------------------------------------
    rename_map = {}
    if GL_LONG_TEXT in df.columns and OUT_GL_ACCOUNT_NAME not in df.columns:
        rename_map[GL_LONG_TEXT] = OUT_GL_ACCOUNT_NAME
    if rename_map:
        df = df.rename(columns=rename_map)

    # ------------------------------------------------------------------
    # STEP 10 — Final validation
    # ------------------------------------------------------------------
    validate_enriched_dataset(df)

    logger.info(
        "build_enriched_dataset: pipeline complete — %d rows, %d columns.",
        len(df), len(df.columns),
    )
    return df