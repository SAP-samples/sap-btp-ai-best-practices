"""
Financial Processor — in-memory data processing for FY vs FY comparison.

Architecture (new multi-table pipeline):
  loaders/raw_loader.py        → load v_basedata, v_profitcenter, v_costcenter, v__glaccountgrouping
  normalizers/id_normalizer.py → safe string normalization (no int/float casting)
  normalizers/text_parser.py   → parse "ID (Description)" mixed fields
  enrichers/dataset_enricher.py → LEFT JOIN pipeline → enriched DataFrame
  validators/data_validator.py  → integrity checks

FinancialProcessor caches the enriched DataFrame and exposes the same API
as before — all existing endpoints and frontend contracts are preserved.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd

from services.variance_interpreter import (
    AccountNature,
    build_account_nature_lookup,
    compute_variance_direction,
)
from services.hierarchy_service import HierarchyService

logger = logging.getLogger(__name__)


def _normalize_lookup_key(key: str) -> str:
    """
    Normalize a lookup key that may contain a mixed SAP format like
    "61002000 (3rd party material)" → "61002000".

    Uses the same parser as the enrichment pipeline so join keys always match.
    Leading zeros are preserved. No numeric casting.
    """
    try:
        from normalizers.text_parser import parse_id_description
        clean_id, _ = parse_id_description(str(key).strip())
        if clean_id and clean_id != key.strip():
            logger.info(
                "Parsed lookup key '%s' → '%s'",
                key.strip(), clean_id,
            )
        return clean_id if clean_id else str(key).strip()
    except Exception:
        return str(key).strip()

# ---------------------------------------------------------------------------
# Column name constants — must match enrichers/dataset_enricher.py output
# ---------------------------------------------------------------------------
COL_COMPANY_CODE      = "Company Code"
COL_GL_ACCOUNT        = "G/L Account"
COL_GL_ACCOUNT_NAME   = "G/L Account Name"
COL_POSTING_DATE      = "Posting Date"
COL_AMOUNT            = "Amount in Company Code Currency"
COL_PROFIT_CENTER     = "Profit Center"
COL_COST_CENTER       = "Cost Center"
COL_WBS_ELEMENT       = "WBS Element"
COL_PURCHASING_DOC    = "Purchasing Document"
COL_FISCAL_PERIOD     = "Fiscal Period"
COL_SEGMENT           = "Segment"
COL_ACCOUNT_GROUPING  = "Account Grouping"
COL_FS_LINE_ITEM      = "Financial Statement Line Item"
COL_FS_ITEM           = "FS Item"
COL_FS_ITEM_NAME      = "FS Item Name"
COL_JE_TYPE           = "Journal Entry Type"
COL_JE_TYPE_NAME      = "JE Type Name"
COL_JE_ITEM_TEXT      = "Journal Entry Item Text"
COL_ASSIGNMENT_REF    = "Assignment Reference"
COL_SUPPLIER          = "Supplier"
COL_SUPPLIER_NAME     = "Name of Supplier"
COL_FUNCTIONAL_AREA   = "Functional Area Name"
COL_FLASH_LINE_ITEM   = "Flash Line Item"
COL_PARTNER_PC        = "Partner Profit Center"
COL_CONSUNIT          = "CONSUNIT"
COL_CONSUNIT_NAME     = "CONSUNIT NAME"
COL_COST_CENTER_NAME  = "Cost Center Name"

# Derived columns (added by enricher)
AMOUNT_NUMERIC = "amount_numeric"
FISCAL_YEAR    = "fiscal_year"
FISCAL_MONTH   = "fiscal_month"

ALLOWED_GROUP_BY: Dict[str, str] = {
    "G/L Account":                   COL_GL_ACCOUNT,
    "Profit Center":                 COL_PROFIT_CENTER,
    "Cost Center":                   COL_COST_CENTER,
    "WBS Element":                   COL_WBS_ELEMENT,
    "Purchasing Document":           COL_PURCHASING_DOC,
    "Financial Statement Line Item": COL_FS_LINE_ITEM,
}

# Name column for each group-by dimension (optional label enrichment)
GROUP_NAME_COL: Dict[str, Optional[str]] = {
    "G/L Account":                   COL_GL_ACCOUNT_NAME,
    "Profit Center":                 None,
    "Cost Center":                   COL_COST_CENTER_NAME,
    "WBS Element":                   None,
    "Purchasing Document":           None,
    "Financial Statement Line Item": None,
}

MONTH_LABELS = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


class FinancialProcessor:
    """
    In-memory financial data processor.
    Caches the enriched dataset built from the multi-table pipeline.
    All public methods preserve the existing API contracts.
    """

    def __init__(self) -> None:
        self._cache: Optional[pd.DataFrame] = None
        self._hierarchy = HierarchyService()

    # ------------------------------------------------------------------
    # Data loading / cache management
    # ------------------------------------------------------------------

    def _get_data(self) -> pd.DataFrame:
        """Return cached enriched data, loading from HANA if not yet loaded."""
        if self._cache is None:
            self._load_and_enrich()
        return self._cache if self._cache is not None else pd.DataFrame()

    def _load_and_enrich(self) -> None:
        """
        Full ingestion pipeline:
          1. Load raw tables from HANA independently
          2. Build enriched dataset via LEFT JOINs
          3. Cache result
        """
        try:
            from loaders.raw_loader import load_all_raw
            from enrichers.dataset_enricher import build_enriched_dataset
        except ImportError as exc:
            logger.error("Pipeline import failed: %s", exc)
            self._cache = pd.DataFrame()
            return

        raw_tables = load_all_raw()
        enriched   = build_enriched_dataset(raw_tables)

        if enriched.empty:
            logger.warning("FinancialProcessor: enriched dataset is empty.")
        else:
            logger.info("FinancialProcessor: cached %d enriched rows.", len(enriched))

        self._cache = enriched

        # Build organizational hierarchy from master data tables
        # Hierarchy: Segment -> Profit Centers -> Cost Centers
        try:
            pc_table = raw_tables.get("v_profitcenter")
            cc_table = raw_tables.get("v_costcenter")
            self._hierarchy.build(
                pc_table=pc_table,
                cc_table=cc_table,
                enriched_df=enriched if not enriched.empty else None,
            )
        except Exception as hier_err:
            logger.error(
                "FinancialProcessor: hierarchy build failed: %s. "
                "Segment filtering will fall back to Segment column.",
                hier_err,
            )

    def refresh(self) -> int:
        """Force reload from HANA. Returns number of rows loaded."""
        self._cache = None
        self._load_and_enrich()
        return len(self._cache) if self._cache is not None else 0

    @property
    def hierarchy(self) -> HierarchyService:
        """Return the organizational hierarchy service (Segment -> PC -> CC)."""
        if self._cache is None:
            self._load_and_enrich()
        return self._hierarchy

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_available_years(self, company_code: Optional[str] = None) -> List[int]:
        """Return distinct fiscal years (ascending)."""
        df = self._get_data()
        if df.empty or FISCAL_YEAR not in df.columns:
            return []
        if company_code:
            df = df[df[COL_COMPANY_CODE].astype(str) == str(company_code)]
        return sorted(df[FISCAL_YEAR].dropna().astype(int).unique().tolist())

    def get_company_codes(self) -> List[str]:
        """Return distinct company codes (sorted)."""
        df = self._get_data()
        if df.empty or COL_COMPANY_CODE not in df.columns:
            return []
        return sorted(df[COL_COMPANY_CODE].dropna().astype(str).unique().tolist())

    def get_row_count(self) -> int:
        """Return number of cached rows."""
        return len(self._cache) if self._cache is not None else 0

    def get_filter_values(self) -> Dict:
        """
        Return all available filter values from the enriched dataset.
        Used by /api/init-data to populate frontend filter dropdowns.
        Values come exclusively from SAP data — no hardcoding.
        """
        df = self._get_data()
        if df.empty:
            return {"segments": [], "functionalAreas": [], "companyCodes": []}

        def _distinct(col: str) -> List[str]:
            if col not in df.columns:
                return []
            return sorted(
                v for v in df[col].dropna().astype(str).str.strip().unique()
                if v and v.lower() not in ("nan", "none", "")
            )

        return {
            "segments":       _distinct(COL_SEGMENT),
            "functionalAreas": _distinct(COL_FUNCTIONAL_AREA),
            "companyCodes":   _distinct(COL_COMPANY_CODE),
        }

    # ------------------------------------------------------------------
    # Global filter helper
    # ------------------------------------------------------------------

    def _apply_global_filters(
        self,
        df: pd.DataFrame,
        company_code: Optional[str] = None,
        segment: Optional[str] = None,
        functional_area: Optional[str] = None,
        context: str = "",
    ) -> pd.DataFrame:
        """
        Apply all global dimension filters to a DataFrame.
        Logs row counts before and after each filter for diagnostics.
        """
        before = len(df)

        if company_code:
            if COL_COMPANY_CODE in df.columns:
                df = df[df[COL_COMPANY_CODE].astype(str) == str(company_code)]
                logger.info(
                    "[%s] company_code='%s': %d → %d rows",
                    context, company_code, before, len(df),
                )
            else:
                logger.warning("[%s] company_code filter: column '%s' not found.", context, COL_COMPANY_CODE)

        if segment:
            if self._hierarchy.is_built:
                # Use hierarchy-based filtering:
                # Segment -> Profit Centers -> Cost Centers (authoritative)
                df = self._hierarchy.apply_segment_filter(df, segment, context=context)
            elif COL_SEGMENT in df.columns:
                # Fallback: direct Segment column filter
                before_seg = len(df)
                df = df[df[COL_SEGMENT].astype(str).str.strip() == str(segment).strip()]
                logger.info(
                    "[%s] segment='%s' (direct column filter, hierarchy not built): %d -> %d rows",
                    context, segment, before_seg, len(df),
                )
            else:
                logger.warning(
                    "[%s] segment filter: hierarchy not built and column '%s' not found.",
                    context, COL_SEGMENT,
                )

        if functional_area:
            if COL_FUNCTIONAL_AREA in df.columns:
                before_fa = len(df)
                df = df[df[COL_FUNCTIONAL_AREA].astype(str).str.strip() == str(functional_area).strip()]
                logger.info(
                    "[%s] functional_area='%s': %d → %d rows",
                    context, functional_area, before_fa, len(df),
                )
            else:
                logger.warning("[%s] functional_area filter: column '%s' not found.", context, COL_FUNCTIONAL_AREA)

        logger.info(
            "[%s] global filters applied: %d → %d rows (company=%s, segment=%s, fa=%s)",
            context, before, len(df), company_code or "ALL", segment or "ALL", functional_area or "ALL",
        )
        return df

    # ------------------------------------------------------------------
    # AI Explain — account-level data extraction
    # ------------------------------------------------------------------

    def get_ai_explanation_data(
        self,
        group_by: str,
        key: str,
        current_year: int,
        previous_year: int,
        company_code: Optional[str] = None,
        segment: Optional[str] = None,
        functional_area: Optional[str] = None,
    ) -> Dict:
        """Return structured data for a specific key to feed the AI prompt."""
        gc_col = ALLOWED_GROUP_BY.get(group_by, COL_GL_ACCOUNT)
        key    = _normalize_lookup_key(key)

        df = self._get_data()
        if df.empty:
            return {}

        # Apply global filters BEFORE key lookup — ensures consistent dataset
        df = self._apply_global_filters(
            df, company_code=company_code, segment=segment,
            functional_area=functional_area, context=f"ai_explain[{key}]",
        )

        before_key = len(df)
        df_key = df[df[gc_col].astype(str) == key]
        logger.info(
            "ai_explain[%s=%s]: key filter '%s'='%s': %d → %d rows",
            group_by, key, gc_col, key, before_key, len(df_key),
        )

        cur_df  = df_key[df_key[FISCAL_YEAR] == int(current_year)]
        prev_df = df_key[df_key[FISCAL_YEAR] == int(previous_year)]

        # ------------------------------------------------------------------
        # For Profit Center / Cost Center analysis:
        # Identify the REAL G/L accounts driving the variance.
        # These are used for account metadata lookup (Flash Line Item,
        # account nature, account name) — NOT the PC/CC key itself.
        # ------------------------------------------------------------------
        gl_account_breakdown: List[Dict] = []
        top_gl_key: Optional[str]  = None
        top_gl_name: Optional[str] = None

        if group_by != "G/L Account" and COL_GL_ACCOUNT in df_key.columns:
            # Step 1: already filtered by PC/CC via df_key
            # Step 2: aggregate by real G/L Account to find top variance drivers
            gl_cur  = cur_df.groupby(COL_GL_ACCOUNT)[AMOUNT_NUMERIC].sum()
            gl_prev = prev_df.groupby(COL_GL_ACCOUNT)[AMOUNT_NUMERIC].sum()
            gl_combined = (
                gl_cur.rename("current").to_frame()
                .join(gl_prev.rename("previous"), how="outer")
                .fillna(0)
            )
            gl_combined["delta"] = gl_combined["current"] - gl_combined["previous"]
            gl_combined = gl_combined.reindex(
                gl_combined["delta"].abs().sort_values(ascending=False).index
            ).head(10)

            # Step 3: enrich with GL account name and Flash Line Item
            gl_name_map: Dict[str, str] = {}
            if COL_GL_ACCOUNT_NAME in df_key.columns:
                gl_name_map = (
                    df_key.dropna(subset=[COL_GL_ACCOUNT_NAME])
                    .groupby(COL_GL_ACCOUNT)[COL_GL_ACCOUNT_NAME]
                    .first().to_dict()
                )

            gl_flash_map: Dict[str, str] = {}
            if COL_FLASH_LINE_ITEM in df_key.columns:
                flash_valid = df_key[
                    df_key[COL_FLASH_LINE_ITEM].notna()
                    & (df_key[COL_FLASH_LINE_ITEM].astype(str).str.strip() != "")
                ]
                if not flash_valid.empty:
                    gl_flash_map = (
                        flash_valid.groupby(COL_GL_ACCOUNT)[COL_FLASH_LINE_ITEM]
                        .first().to_dict()
                    )

            for gl_k, row in gl_combined.iterrows():
                gl_str = str(gl_k)
                gl_account_breakdown.append({
                    "glAccount":     gl_str,
                    "glAccountName": gl_name_map.get(gl_str, ""),
                    "flashLineItem": gl_flash_map.get(gl_str, ""),
                    "current":       round(float(row["current"]), 2),
                    "previous":      round(float(row["previous"]), 2),
                    "delta":         round(float(row["delta"]), 2),
                })

            if gl_account_breakdown:
                top_gl_key  = gl_account_breakdown[0]["glAccount"]
                top_gl_name = gl_account_breakdown[0]["glAccountName"]
                logger.info(
                    "ai_explain[%s=%s]: top GL drivers: %s",
                    group_by, key,
                    [
                        {
                            "gl":    g["glAccount"],
                            "name":  g["glAccountName"],
                            "flash": g["flashLineItem"],
                            "delta": g["delta"],
                        }
                        for g in gl_account_breakdown[:5]
                    ],
                )
            else:
                logger.warning(
                    "ai_explain[%s=%s]: No G/L accounts found in filtered dataset.",
                    group_by, key,
                )

        # Validation: log Flash Line Item join status (using correct dimension label)
        dim_label = "G/L Account" if group_by == "G/L Account" else group_by
        if COL_FLASH_LINE_ITEM in df_key.columns:
            flash_vals = df_key[COL_FLASH_LINE_ITEM].dropna().astype(str).str.strip()
            flash_vals = flash_vals[flash_vals != ""]
            unique_flash = flash_vals.unique().tolist()
            if unique_flash:
                logger.info(
                    "ai_explain: %s '%s' → Flash Line Item(s) from SAP: %s",
                    dim_label, key, unique_flash[:10],
                )
            else:
                logger.warning(
                    "ai_explain: %s '%s' → Flash Line Item is NULL/empty in SAP. "
                    "v_glaccount join may have failed or account has no Flash classification.",
                    dim_label, key,
                )
        else:
            logger.warning(
                "ai_explain: Flash Line Item column not present in enriched dataset "
                "for %s '%s'. v_glaccount was not joined.",
                dim_label, key,
            )

        cur_total  = float(cur_df[AMOUNT_NUMERIC].sum())  if not cur_df.empty  else 0.0
        prev_total = float(prev_df[AMOUNT_NUMERIC].sum()) if not prev_df.empty else 0.0
        delta      = cur_total - prev_total
        pct        = ((delta / abs(prev_total)) * 100) if prev_total != 0 else 0.0

        # Monthly breakdown
        monthly = []
        for month in range(1, 13):
            cur_m  = float(cur_df[cur_df[FISCAL_MONTH]  == month][AMOUNT_NUMERIC].sum())
            prev_m = float(prev_df[prev_df[FISCAL_MONTH] == month][AMOUNT_NUMERIC].sum())
            if cur_m != 0 or prev_m != 0:
                monthly.append({
                    "month":    month,
                    "label":    MONTH_LABELS.get(month, str(month)),
                    "current":  cur_m,
                    "previous": prev_m,
                    "delta":    cur_m - prev_m,
                })

        # Top profit centers
        pc_breakdown: List[Dict] = []
        if COL_PROFIT_CENTER in cur_df.columns and not cur_df.empty:
            pc_agg = (
                cur_df.groupby(COL_PROFIT_CENTER)[AMOUNT_NUMERIC]
                .sum().sort_values(ascending=False).head(5)
            )
            pc_breakdown = [{"profitCenter": str(k), "amount": float(v)} for k, v in pc_agg.items()]

        # Top cost centers
        cc_breakdown: List[Dict] = []
        if COL_COST_CENTER in cur_df.columns and not cur_df.empty:
            cc_agg = (
                cur_df.groupby(COL_COST_CENTER)[AMOUNT_NUMERIC]
                .sum().sort_values(ascending=False).head(5)
            )
            cc_breakdown = [{"costCenter": str(k), "amount": float(v)} for k, v in cc_agg.items()]

        # Segment
        segment = ""
        if COL_SEGMENT in df_key.columns and not df_key.empty:
            segs = df_key[COL_SEGMENT].dropna().unique().tolist()
            segment = ", ".join(str(s) for s in segs[:3])

        # Account name — for GL analysis use GL account name;
        # for PC/CC analysis use the dimension's own name column.
        account_name = ""
        if group_by == "G/L Account":
            if COL_GL_ACCOUNT_NAME in df_key.columns and not df_key.empty:
                names = df_key[COL_GL_ACCOUNT_NAME].dropna().unique().tolist()
                account_name = str(names[0]) if names else ""
        elif group_by == "Profit Center":
            pc_name_col = "Profit Center Name"
            if pc_name_col in df_key.columns and not df_key.empty:
                pc_names = df_key[pc_name_col].dropna().unique().tolist()
                account_name = str(pc_names[0]) if pc_names else key
            else:
                account_name = key
        elif group_by == "Cost Center":
            if COL_COST_CENTER_NAME in df_key.columns and not df_key.empty:
                cc_names = df_key[COL_COST_CENTER_NAME].dropna().unique().tolist()
                account_name = str(cc_names[0]) if cc_names else key
            else:
                account_name = key
        else:
            account_name = key

        # Journal entry texts (top 10)
        je_texts: List[Dict] = []
        if COL_JE_ITEM_TEXT in cur_df.columns and not cur_df.empty:
            texts = cur_df[COL_JE_ITEM_TEXT].dropna().value_counts().head(10)
            je_texts = [{"text": str(k), "count": int(v)} for k, v in texts.items()]

        # Top suppliers
        supplier_breakdown: List[Dict] = []
        if COL_SUPPLIER_NAME in cur_df.columns and not cur_df.empty:
            sup_agg = (
                cur_df[cur_df[COL_SUPPLIER_NAME].notna() & (cur_df[COL_SUPPLIER_NAME] != "")]
                .groupby(COL_SUPPLIER_NAME)[AMOUNT_NUMERIC]
                .sum().sort_values(ascending=False).head(5)
            )
            supplier_breakdown = [{"supplier": str(k), "amount": float(v)} for k, v in sup_agg.items()]

        # Document type breakdown
        doc_type_breakdown: List[Dict] = []
        if COL_JE_TYPE in cur_df.columns and COL_JE_TYPE_NAME in cur_df.columns and not cur_df.empty:
            dt_agg = (
                cur_df.groupby([COL_JE_TYPE, COL_JE_TYPE_NAME])[AMOUNT_NUMERIC]
                .agg(total="sum", count="count")
                .reset_index()
                .sort_values("total", ascending=False)
                .head(8)
            )
            for _, row in dt_agg.iterrows():
                doc_type_breakdown.append({
                    "jeType":     str(row[COL_JE_TYPE]),
                    "jeTypeName": str(row[COL_JE_TYPE_NAME]),
                    "amount":     float(row["total"]),
                    "count":      int(row["count"]),
                })

        # Flash grouping summary — ONLY from SAP Flash Line Item (v_glaccount)
        # NO fallbacks, NO heuristics, NO "Other" defaults
        flash_summary: Dict[str, float] = {}
        if not cur_df.empty:
            if COL_FLASH_LINE_ITEM in cur_df.columns:
                # Use ONLY real SAP values — null means unclassified in SAP
                flash_series = cur_df[COL_FLASH_LINE_ITEM].astype(object)
                null_flash = flash_series.isna().sum()
                empty_flash = (flash_series.fillna("").astype(str).str.strip() == "").sum()
                unclassified = null_flash + empty_flash

                if unclassified > 0:
                    logger.warning(
                        "Flash Line Item: %d/%d records have no SAP Flash Line Item "
                        "for G/L Account '%s'. These are excluded from flash summary. "
                        "Check v_glaccount join for this account.",
                        unclassified, len(cur_df), key,
                    )

                # Only aggregate rows that have a real SAP Flash Line Item value
                valid_mask = flash_series.notna() & (flash_series.astype(str).str.strip() != "")
                if valid_mask.any():
                    flash_df  = pd.DataFrame({
                        "flash":  flash_series[valid_mask].astype(str).str.strip(),
                        "amount": cur_df.loc[valid_mask, AMOUNT_NUMERIC].values,
                    })
                    flash_agg = flash_df.groupby("flash")["amount"].sum().sort_values(ascending=False)
                    flash_summary = {str(k): float(v) for k, v in flash_agg.items()}
            else:
                logger.warning(
                    "Flash Line Item column '%s' not found in enriched dataset for G/L Account '%s'. "
                    "Verify that v_glaccount was loaded and joined correctly.",
                    COL_FLASH_LINE_ITEM, key,
                )

        # Flash variance: Current Year vs Previous Year delta per category.
        # Uses signed SAP values exactly as stored — no absolute-value flattening,
        # no removal of reversals, RC reclasses, CO repostings, or offsetting entries.
        # Delta = CurrentYear - PreviousYear (positive = more expense, negative = less expense)
        #
        # Extended structure includes:
        #   - isFavorable: expense decrease = True, increase = False
        #   - periods: month-by-month breakdown per category
        #   - netZeroActivity: internal JE types (RC/CO/RK/AB) that net to zero per category
        flash_variance: List[Dict] = []

        # SAP internal/reallocation document types (same set as api.py _REALLOCATION_DOC_TYPES)
        _INTERNAL_JE_TYPES = {"RC", "CO", "AB", "SA", "JE", "KP", "RK", "WA"}

        if COL_FLASH_LINE_ITEM in df_key.columns:

            def _valid_flash_rows(year_df: pd.DataFrame) -> pd.DataFrame:
                """Return rows with a non-null, non-empty Flash Line Item."""
                if year_df.empty:
                    return pd.DataFrame()
                mask = (
                    year_df[COL_FLASH_LINE_ITEM].notna()
                    & (year_df[COL_FLASH_LINE_ITEM].astype(str).str.strip() != "")
                )
                result = year_df[mask].copy()
                if not result.empty:
                    result["_flash"] = result[COL_FLASH_LINE_ITEM].astype(str).str.strip()
                return result

            def _flash_agg_year(year_df: pd.DataFrame) -> pd.Series:
                """Aggregate amount_numeric by Flash Line Item for one year's rows."""
                valid = _valid_flash_rows(year_df)
                if valid.empty:
                    return pd.Series(dtype=float)
                return valid.groupby("_flash")[AMOUNT_NUMERIC].sum()

            def _flash_agg_by_period(year_df: pd.DataFrame) -> pd.DataFrame:
                """Aggregate amount_numeric by (Flash Line Item, fiscal_month)."""
                valid = _valid_flash_rows(year_df)
                if valid.empty or FISCAL_MONTH not in valid.columns:
                    return pd.DataFrame()
                return (
                    valid.groupby(["_flash", FISCAL_MONTH])[AMOUNT_NUMERIC]
                    .sum()
                    .reset_index()
                )

            cur_flash_agg   = _flash_agg_year(cur_df)
            prev_flash_agg  = _flash_agg_year(prev_df)
            cur_period_agg  = _flash_agg_by_period(cur_df)
            prev_period_agg = _flash_agg_by_period(prev_df)

            all_flash_cats = set(cur_flash_agg.index) | set(prev_flash_agg.index)

            for cat in sorted(all_flash_cats):
                cur_amt  = float(cur_flash_agg.get(cat, 0.0))
                prev_amt = float(prev_flash_agg.get(cat, 0.0))
                delta_v  = cur_amt - prev_amt
                delta_pct = (
                    round((delta_v / abs(prev_amt)) * 100, 1)
                    if prev_amt != 0 else None
                )
                # Expense accounts: decrease is favorable (less spend = better P&L)
                is_favorable = (
                    True  if delta_v < 0 else
                    False if delta_v > 0 else None
                )

                # Period breakdown for this category
                periods: List[Dict] = []
                if FISCAL_MONTH in df_key.columns:
                    cur_cat_map: Dict[int, float] = {}
                    if not cur_period_agg.empty and "_flash" in cur_period_agg.columns:
                        cat_rows = cur_period_agg[cur_period_agg["_flash"] == cat]
                        cur_cat_map = dict(zip(
                            cat_rows[FISCAL_MONTH].astype(int),
                            cat_rows[AMOUNT_NUMERIC].astype(float),
                        ))
                    prev_cat_map: Dict[int, float] = {}
                    if not prev_period_agg.empty and "_flash" in prev_period_agg.columns:
                        cat_rows = prev_period_agg[prev_period_agg["_flash"] == cat]
                        prev_cat_map = dict(zip(
                            cat_rows[FISCAL_MONTH].astype(int),
                            cat_rows[AMOUNT_NUMERIC].astype(float),
                        ))
                    all_months = set(cur_cat_map.keys()) | set(prev_cat_map.keys())
                    for month in sorted(all_months):
                        m     = int(month)
                        c_amt = float(cur_cat_map.get(m, 0.0))
                        p_amt = float(prev_cat_map.get(m, 0.0))
                        d_v   = c_amt - p_amt
                        d_pct = (
                            round((d_v / abs(p_amt)) * 100, 1)
                            if p_amt != 0 else None
                        )
                        periods.append({
                            "month":        m,
                            "monthLabel":   MONTH_LABELS.get(m, str(m)),
                            "current":      round(c_amt, 2),
                            "previous":     round(p_amt, 2),
                            "delta":        round(d_v, 2),
                            "deltaPercent": d_pct,
                            "isFavorable":  (
                                True  if d_v < 0 else
                                False if d_v > 0 else None
                            ),
                        })

                # Net-zero internal reallocation detection per flash category
                net_zero_activity: List[Dict] = []
                if COL_JE_TYPE in df_key.columns:
                    cat_all_rows = df_key[
                        df_key[COL_FLASH_LINE_ITEM].astype(str).str.strip() == cat
                    ]
                    if not cat_all_rows.empty:
                        internal_rows = cat_all_rows[
                            cat_all_rows[COL_JE_TYPE].astype(str).str.upper()
                            .isin(_INTERNAL_JE_TYPES)
                        ]
                        if not internal_rows.empty:
                            je_totals = (
                                internal_rows.groupby(COL_JE_TYPE)[AMOUNT_NUMERIC]
                                .sum()
                            )
                            for je_type, je_total in je_totals.items():
                                if abs(float(je_total)) < 0.01:
                                    net_zero_activity.append({
                                        "jeType":      str(je_type),
                                        "total":       round(float(je_total), 2),
                                        "description": "Internal reallocation (net zero)",
                                        "isNetZero":   True,
                                    })

                flash_variance.append({
                    "category":           cat,
                    "current":            round(cur_amt, 2),
                    "previous":           round(prev_amt, 2),
                    "delta":              round(delta_v, 2),
                    "deltaPercent":       delta_pct,
                    "isFavorable":        is_favorable,
                    "periods":            periods,
                    "netZeroActivity":    net_zero_activity,
                    "hasNetZeroActivity": len(net_zero_activity) > 0,
                })

            # Sort by absolute delta descending (largest movers first)
            flash_variance.sort(key=lambda x: abs(x["delta"]), reverse=True)

        # Accounting-aware variance direction for AI prompt.
        # For G/L Account: use the account's own nature.
        # For Profit Center / Cost Center: derive nature from the REAL G/L accounts
        # found in the filtered dataset — NOT from the PC/CC key itself.
        nature = AccountNature.UNKNOWN
        if group_by == "G/L Account" and not df_key.empty:
            nature_lookup_single = build_account_nature_lookup(df_key, COL_GL_ACCOUNT)
            nature = nature_lookup_single.get(key, AccountNature.UNKNOWN)
            logger.info(
                "ai_explain[GL=%s]: account nature = %s", key, nature.value,
            )
        elif group_by != "G/L Account" and not df_key.empty and COL_GL_ACCOUNT in df_key.columns:
            # Build nature lookup from all GL accounts in the filtered PC/CC dataset
            nature_lookup = build_account_nature_lookup(df_key, COL_GL_ACCOUNT)
            if nature_lookup:
                if top_gl_key and top_gl_key in nature_lookup:
                    # Use the nature of the top variance-driving GL account
                    nature = nature_lookup[top_gl_key]
                    logger.info(
                        "ai_explain[%s=%s]: account nature resolved from top GL '%s': %s",
                        group_by, key, top_gl_key, nature.value,
                    )
                else:
                    # Fallback: use the most common nature across all GL accounts
                    from collections import Counter
                    nature_counts = Counter(nature_lookup.values())
                    nature = nature_counts.most_common(1)[0][0]
                    logger.info(
                        "ai_explain[%s=%s]: account nature resolved from dominant GL nature: %s",
                        group_by, key, nature.value,
                    )
            else:
                logger.warning(
                    "ai_explain[%s=%s]: could not resolve account nature — "
                    "no GL accounts with SAP nature data found.",
                    group_by, key,
                )

        interp = compute_variance_direction(delta, nature, gl_account_key=top_gl_key or key)

        return {
            "key":                   key,
            "accountName":           account_name,
            "groupBy":               group_by,
            "currentYear":           current_year,
            "previousYear":          previous_year,
            "currentTotal":          cur_total,
            "previousTotal":         prev_total,
            "delta":                 delta,
            "deltaPercent":          pct,
            "segment":               segment,
            "monthly":               monthly,
            "profitCenterBreakdown": pc_breakdown,
            "costCenterBreakdown":   cc_breakdown,
            "journalEntryTexts":     je_texts,
            "supplierBreakdown":     supplier_breakdown,
            "docTypeBreakdown":      doc_type_breakdown,
            "flashSummary":          flash_summary,
            "flashVariance":         flash_variance,
            "currentRecords":        int(len(cur_df)),
            "previousRecords":       int(len(prev_df)),
            # Accounting-aware variance direction
            "isFavorable":           interp["is_favorable"],
            "accountNature":         nature.value,
            "varianceDirection":     interp["direction_label"],
            "narrativeContext":      interp["narrative_context"],
            # GL account breakdown for PC/CC analysis
            # (empty list when group_by == "G/L Account")
            "glAccountBreakdown":    gl_account_breakdown,
            "topGlAccount":          top_gl_key,
            "topGlAccountName":      top_gl_name,
        }


    def get_account_line_items(
        self,
        group_by: str,
        key: str,
        current_year: int,
        previous_year: int,
        company_code: Optional[str] = None,
        segment: Optional[str] = None,
        functional_area: Optional[str] = None,
        limit: int = 300,
    ) -> List[Dict]:
        """Return enriched line items for a specific key (both years) with Flash Grouping."""
        gc_col = ALLOWED_GROUP_BY.get(group_by, COL_GL_ACCOUNT)
        key    = _normalize_lookup_key(key)

        df = self._get_data()
        if df.empty:
            return []

        df = self._apply_global_filters(
            df, company_code=company_code, segment=segment,
            functional_area=functional_area, context=f"line_items[{key}]",
        )
        df = df[df[FISCAL_YEAR].isin([int(current_year), int(previous_year)])].copy()
        df = df[df[gc_col].astype(str) == key]

        # Flash Grouping — ONLY from SAP Flash Line Item (v_glaccount)
        # Null means SAP has no classification — never substitute with heuristics
        if COL_FLASH_LINE_ITEM in df.columns:
            # Expose the real SAP value as-is; null stays null (shown as empty string in output)
            df["Flash Grouping"] = df[COL_FLASH_LINE_ITEM].astype(object)
            null_count = df["Flash Grouping"].isna().sum()
            if null_count > 0:
                logger.warning(
                    "Flash Line Item: %d/%d line items have no SAP Flash Line Item "
                    "for key '%s'. These will appear blank in the output.",
                    null_count, len(df), key,
                )
        else:
            logger.warning(
                "Flash Line Item column '%s' not found in enriched dataset for key '%s'. "
                "Verify that v_glaccount was loaded and joined correctly.",
                COL_FLASH_LINE_ITEM, key,
            )
            df["Flash Grouping"] = None

        # Build JE Type display
        if COL_JE_TYPE_NAME in df.columns and COL_JE_TYPE in df.columns:
            df["JE Type Display"] = df[COL_JE_TYPE_NAME].fillna("") + " (" + df[COL_JE_TYPE].fillna("") + ")"
        elif COL_JE_TYPE_NAME in df.columns:
            df["JE Type Display"] = df[COL_JE_TYPE_NAME].fillna("")
        elif COL_JE_TYPE in df.columns:
            df["JE Type Display"] = df[COL_JE_TYPE].fillna("")
        else:
            df["JE Type Display"] = ""

        desired = [
            "JE Type Display", COL_JE_ITEM_TEXT, COL_SUPPLIER_NAME, COL_SUPPLIER,
            COL_ASSIGNMENT_REF, COL_AMOUNT, "Flash Grouping",
            COL_GL_ACCOUNT, COL_GL_ACCOUNT_NAME, COL_PROFIT_CENTER, COL_COST_CENTER,
            COL_FUNCTIONAL_AREA, COL_WBS_ELEMENT, COL_ACCOUNT_GROUPING, COL_SEGMENT,
            FISCAL_YEAR, FISCAL_MONTH, AMOUNT_NUMERIC,
        ]
        available = [c for c in desired if c in df.columns]
        result = df[available].head(limit).copy()
        result = result.fillna("")

        for col in [FISCAL_YEAR, FISCAL_MONTH]:
            if col in result.columns:
                result[col] = result[col].astype(object).where(result[col].notna(), None)

        return result.to_dict(orient="records")

    # ------------------------------------------------------------------
    # FY vs FY grouped analysis
    # ------------------------------------------------------------------

    def get_grouped_analysis(
        self,
        group_by: str,
        current_year: int,
        previous_year: int,
        company_code: Optional[str] = None,
        segment: Optional[str] = None,
        functional_area: Optional[str] = None,
    ) -> List[Dict]:
        """
        FY vs FY comparison aggregated by group_by dimension.
        Returns list of {key, name, currentAmount, previousAmount, variance, variancePercent, records}.
        """
        if group_by not in ALLOWED_GROUP_BY:
            logger.warning("Invalid group_by: %s", group_by)
            return []

        gc_col   = ALLOWED_GROUP_BY[group_by]
        name_col = GROUP_NAME_COL.get(group_by)

        df = self._get_data()
        if df.empty or FISCAL_YEAR not in df.columns:
            return []

        if gc_col not in df.columns:
            logger.warning("Column '%s' not found in cache.", gc_col)
            return []

        # Log unique values for the grouping dimension before filtering
        unique_keys_before = df[gc_col].dropna().astype(str).nunique()
        logger.info(
            "get_grouped_analysis[%s]: %d unique '%s' values before filters, %d total rows",
            group_by, unique_keys_before, gc_col, len(df),
        )

        df = self._apply_global_filters(
            df, company_code=company_code, segment=segment,
            functional_area=functional_area, context=f"grouped_analysis[{group_by}]",
        )

        unique_keys_after = df[gc_col].dropna().astype(str).nunique()
        logger.info(
            "get_grouped_analysis[%s]: %d unique '%s' values after filters, %d total rows",
            group_by, unique_keys_after, gc_col, len(df),
        )

        cur_df  = df[df[FISCAL_YEAR] == int(current_year)]
        prev_df = df[df[FISCAL_YEAR] == int(previous_year)]

        logger.info(
            "get_grouped_analysis[%s]: cur_year=%s → %d rows, prev_year=%s → %d rows",
            group_by, current_year, len(cur_df), previous_year, len(prev_df),
        )

        cur_agg  = cur_df.groupby(gc_col)[AMOUNT_NUMERIC].agg(total="sum", records="count")
        prev_agg = prev_df.groupby(gc_col)[AMOUNT_NUMERIC].sum().rename("prev_total")
        combined = cur_agg.join(prev_agg, how="outer").fillna(0)

        # Build name lookup
        name_lookup: Dict = {}
        if name_col and name_col in df.columns:
            name_lookup = (
                df[[gc_col, name_col]]
                .dropna(subset=[gc_col])
                .drop_duplicates(subset=[gc_col])
                .set_index(gc_col)[name_col]
                .to_dict()
            )

        # Build account nature lookup — only meaningful for G/L Account grouping.
        # For other dimensions (Profit Center, Cost Center, etc.) nature is UNKNOWN.
        nature_lookup: Dict[str, AccountNature] = {}
        if group_by == "G/L Account":
            nature_lookup = build_account_nature_lookup(df, gc_col)

        results: List[Dict] = []
        for key, row in combined.iterrows():
            ca      = float(row.get("total", 0.0))
            pa      = float(row.get("prev_total", 0.0))
            records = int(row.get("records", 0))
            var     = ca - pa
            var_pct = round((var / abs(pa)) * 100, 2) if pa != 0 else None
            key_str = str(key).strip()

            nature = nature_lookup.get(key_str, AccountNature.UNKNOWN)
            interp = compute_variance_direction(var, nature, gl_account_key=key_str)

            results.append({
                "key":               key_str,
                "name":              str(name_lookup.get(key, "")),
                "currentAmount":     round(ca, 2),
                "previousAmount":    round(pa, 2),
                "variance":          round(var, 2),
                "variancePercent":   var_pct,
                "records":           records,
                # Accounting-aware variance direction fields
                "isFavorable":       interp["is_favorable"],
                "accountNature":     nature.value,
                "varianceDirection": interp["direction_label"],
            })

        results.sort(key=lambda r: str(r["key"]))
        logger.info(
            "get_grouped_analysis: group=%s cur=%s prev=%s → %d rows",
            group_by, current_year, previous_year, len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Monthly drilldown
    # ------------------------------------------------------------------

    def get_group_detail(
        self,
        group_by: str,
        key: str,
        current_year: int,
        previous_year: int,
        company_code: Optional[str] = None,
        segment: Optional[str] = None,
        functional_area: Optional[str] = None,
    ) -> List[Dict]:
        """Monthly breakdown for a specific key."""
        if group_by not in ALLOWED_GROUP_BY:
            return []

        gc_col = ALLOWED_GROUP_BY[group_by]
        key    = _normalize_lookup_key(key)

        df = self._get_data()
        if df.empty or FISCAL_YEAR not in df.columns or gc_col not in df.columns:
            return []

        df = self._apply_global_filters(
            df, company_code=company_code, segment=segment,
            functional_area=functional_area, context=f"group_detail[{group_by}:{key}]",
        )

        before_key = len(df)
        df = df[df[gc_col].astype(str) == key]
        logger.info(
            "group_detail[%s]: key='%s' matched %d/%d rows",
            group_by, key, len(df), before_key,
        )

        cur_df  = df[df[FISCAL_YEAR] == int(current_year)]
        prev_df = df[df[FISCAL_YEAR] == int(previous_year)]

        cur_monthly  = cur_df.groupby(FISCAL_MONTH)[AMOUNT_NUMERIC].sum()
        prev_monthly = prev_df.groupby(FISCAL_MONTH)[AMOUNT_NUMERIC].sum()
        all_months   = set(cur_monthly.index) | set(prev_monthly.index)

        results: List[Dict] = []
        for month in sorted(all_months):
            m  = int(month)
            ca = float(cur_monthly.get(month, 0.0))
            pa = float(prev_monthly.get(month, 0.0))
            results.append({
                "month":          m,
                "monthLabel":     MONTH_LABELS.get(m, str(m)),
                "currentAmount":  round(ca, 2),
                "previousAmount": round(pa, 2),
                "delta":          round(ca - pa, 2),
            })

        return results