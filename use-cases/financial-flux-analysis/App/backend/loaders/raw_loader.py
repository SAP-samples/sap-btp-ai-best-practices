"""
Raw table loader — loads individual SAP HANA views independently.
No transformations, no joins. Returns raw DataFrames exactly as HANA returns them.

Transactional union:
  v_basedata + v_newcc + ceco1711 are vertically appended (UNION ALL) before
  enrichment. All three share the same transactional schema. The combined result
  is returned under the key 'basedata' so the enrichment pipeline requires no
  changes.

Schema validation:
  If v_newcc or ceco1711 are loaded and their column sets do not match
  v_basedata, an explicit SchemaCompatibilityError is raised.
  Silent continuation is NOT permitted.
"""

import logging
import os
from typing import Dict, Set

import pandas as pd

from database.hana_connection import HANAConnection, HDBCLI_AVAILABLE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# View names (configurable via environment variables)
# ---------------------------------------------------------------------------
VIEW_BASEDATA     = os.getenv("HANA_VIEW_BASEDATA",     "t_v_basedata")
VIEW_NEWCC        = os.getenv("HANA_VIEW_NEWCC",        "t_v_newcc")
VIEW_CECO1711     = os.getenv("HANA_VIEW_CECO1711",     "t_ceco1711")
VIEW_PROFITCENTER = os.getenv("HANA_VIEW_PROFITCENTER", "t_v_profitcenter")
VIEW_COSTCENTER   = os.getenv("HANA_VIEW_COSTCENTER",   "t_v_costcenter")
VIEW_GL_GROUPING  = os.getenv("HANA_VIEW_GL_GROUPING",  "t_v__glaccountgrouping")
VIEW_GLACCOUNT    = os.getenv("HANA_VIEW_GLACCOUNT",    "t_v_glaccount")


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class SchemaCompatibilityError(Exception):
    """
    Raised when a transactional source table has an incompatible column set
    relative to v_basedata. The union is aborted — no silent continuation.
    """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_view(view_name: str) -> pd.DataFrame:
    """
    Load all rows from a single HANA view.
    Returns an empty DataFrame on any error.
    All column values are returned as Python native types (str/int/float/Decimal).
    """
    if not HDBCLI_AVAILABLE:
        logger.warning("hdbcli not available — cannot load '%s'.", view_name)
        return pd.DataFrame()
    try:
        with HANAConnection() as hana:
            if hana.connection is None:
                logger.warning("No HANA connection — cannot load '%s'.", view_name)
                return pd.DataFrame()
            sql = 'SELECT * FROM "{}"'.format(view_name)
            df = hana.execute_query(sql)
            logger.info("Loaded %d rows from '%s'.", len(df), view_name)
            return df
    except Exception as exc:
        logger.error("Failed to load '%s': %s", view_name, exc)
        return pd.DataFrame()


def _validate_schema_compatibility(
    df_base: pd.DataFrame,
    df_newcc: pd.DataFrame,
    base_name: str = VIEW_BASEDATA,
    newcc_name: str = VIEW_NEWCC,
) -> None:
    """
    Validate that df_newcc has the same column set as df_base.

    Rules:
    - Column names are compared case-insensitively (SAP HANA may return
      uppercase or mixed-case depending on the view definition).
    - Extra columns in df_newcc that are absent from df_base are logged as
      WARNING and will be dropped during union (to prevent column explosion).
    - Columns present in df_base but MISSING from df_newcc raise
      SchemaCompatibilityError — the union cannot proceed safely.

    Raises:
        SchemaCompatibilityError: if df_newcc is missing columns that exist
            in df_base.
    """
    if df_base.empty:
        # Nothing to validate against — skip
        logger.warning(
            "Schema validation skipped: '%s' is empty.", base_name
        )
        return

    if df_newcc.empty:
        # v_newcc returned no rows — not an error, just log and skip union
        logger.info(
            "Schema validation skipped: '%s' is empty (0 rows). "
            "Union will not be performed.",
            newcc_name,
        )
        return

    base_cols:  Set[str] = set(c.upper() for c in df_base.columns)
    newcc_cols: Set[str] = set(c.upper() for c in df_newcc.columns)

    missing_in_newcc = base_cols - newcc_cols
    extra_in_newcc   = newcc_cols - base_cols

    if missing_in_newcc:
        raise SchemaCompatibilityError(
            f"Schema mismatch: '{newcc_name}' is missing columns that exist in "
            f"'{base_name}': {sorted(missing_in_newcc)}. "
            f"Union aborted. Verify the HANA view definition for '{newcc_name}'."
        )

    if extra_in_newcc:
        logger.warning(
            "Schema warning: '%s' has %d extra column(s) not present in '%s': %s. "
            "These columns will be included in the union (appended with NaN for "
            "rows from '%s').",
            newcc_name, len(extra_in_newcc), base_name,
            sorted(extra_in_newcc), base_name,
        )

    logger.info(
        "Schema validation passed: '%s' (%d cols) is compatible with '%s' (%d cols).",
        newcc_name, len(df_newcc.columns),
        base_name,  len(df_base.columns),
    )


def _align_schema(
    df_incoming: pd.DataFrame,
    df_master: pd.DataFrame,
    incoming_name: str,
    master_name: str,
) -> pd.DataFrame:
    """
    Align df_incoming schema to match df_master by adding NULL columns for
    any columns present in df_master but absent from df_incoming.

    Rules:
    - Missing columns are added explicitly as pd.NA (object dtype).
    - Column order follows df_master (master schema is authoritative).
    - Extra columns in df_incoming not present in df_master are kept at the end.
    - Every auto-added column is logged individually (no silent continuation).
    - If df_master or df_incoming is empty, df_incoming is returned unchanged.

    Returns:
        df_incoming with missing columns added and reindexed to master column order.
    """
    if df_master.empty or df_incoming.empty:
        return df_incoming

    master_cols        = list(df_master.columns)
    master_cols_upper  = {c.upper(): c for c in master_cols}
    incoming_cols_upper = {c.upper(): c for c in df_incoming.columns}

    missing_cols = [
        original_col
        for upper_col, original_col in master_cols_upper.items()
        if upper_col not in incoming_cols_upper
    ]

    if missing_cols:
        logger.warning(
            "Schema alignment: '%s' is missing %d column(s) present in '%s'. "
            "Auto-adding as NULL: %s",
            incoming_name, len(missing_cols), master_name, sorted(missing_cols),
        )
        df_aligned = df_incoming.copy()
        for col in missing_cols:
            df_aligned[col] = pd.NA
            logger.info(
                "Schema alignment: added NULL column '%s' to '%s'.",
                col, incoming_name,
            )
    else:
        df_aligned = df_incoming
        logger.info(
            "Schema alignment: '%s' already has all required columns from '%s'. "
            "No alignment needed.",
            incoming_name, master_name,
        )

    # Reindex: master columns first, then any extra columns from df_incoming
    extra_cols = [
        c for c in df_aligned.columns
        if c.upper() not in master_cols_upper
    ]
    final_col_order = [c for c in master_cols if c in df_aligned.columns] + extra_cols
    df_aligned = df_aligned[final_col_order]

    logger.info(
        "Schema alignment complete: '%s' → %d columns "
        "(master: %d | auto-added NULL: %d | extra kept: %d).",
        incoming_name, len(df_aligned.columns),
        len(master_cols), len(missing_cols), len(extra_cols),
    )

    return df_aligned


def _union_transactional(
    df_base: pd.DataFrame,
    df_newcc: pd.DataFrame,
    base_name: str = VIEW_BASEDATA,
    newcc_name: str = VIEW_NEWCC,
) -> pd.DataFrame:
    """
    Vertically append (UNION ALL) df_base and df_newcc.

    Strategy:
    - Align schema first: any columns present in df_base but absent from
      df_newcc are added as NULL columns (explicit, logged, never silent).
    - Validate schema compatibility after alignment (raises on hard mismatch).
    - Use pd.concat with ignore_index=True — no deduplication unless rows are
      EXACT duplicates across ALL columns.
    - Log row counts before and after.
    - Validate that combined count == base_count + newcc_count (no rows lost).

    Returns:
        Combined DataFrame. If df_newcc is empty, returns df_base unchanged.
    """
    base_count  = len(df_base)
    newcc_count = len(df_newcc)

    logger.info(
        "Transactional union: '%s' = %d rows | '%s' = %d rows",
        base_name, base_count, newcc_name, newcc_count,
    )

    # Align schema: add NULL columns for any missing in df_newcc before validation
    df_newcc = _align_schema(df_newcc, df_base, newcc_name, base_name)

    # Validate schema after alignment (strict — raises if still incompatible)
    _validate_schema_compatibility(df_base, df_newcc, base_name, newcc_name)

    if df_newcc.empty:
        logger.info(
            "Union skipped: '%s' is empty. Using '%s' alone (%d rows).",
            newcc_name, base_name, base_count,
        )
        return df_base

    if df_base.empty:
        logger.warning(
            "Union: '%s' is empty. Using '%s' alone (%d rows).",
            base_name, newcc_name, newcc_count,
        )
        return df_newcc

    # Add a source-tracking column for diagnostics (not exposed to enricher)
    df_base_tagged  = df_base.copy()
    df_newcc_tagged = df_newcc.copy()
    df_base_tagged["_source"]  = base_name
    df_newcc_tagged["_source"] = newcc_name

    combined = pd.concat(
        [df_base_tagged, df_newcc_tagged],
        ignore_index=True,
        sort=False,          # preserve column order from df_base
    )

    combined_count = len(combined)
    expected_count = base_count + newcc_count

    if combined_count != expected_count:
        raise RuntimeError(
            f"Union integrity check FAILED: expected {expected_count} rows "
            f"({base_count} + {newcc_count}) but got {combined_count}. "
            f"No rows should be lost during a vertical append."
        )

    logger.info(
        "Union complete: %d + %d = %d rows (integrity check passed).",
        base_count, newcc_count, combined_count,
    )

    # Log unique Cost Centers and Profit Centers contributed by v_newcc
    _log_new_dimension_values(df_base, df_newcc, "Cost Center",   newcc_name)
    _log_new_dimension_values(df_base, df_newcc, "Profit Center", newcc_name)

    return combined


def _log_new_dimension_values(
    df_base: pd.DataFrame,
    df_newcc: pd.DataFrame,
    col: str,
    newcc_name: str,
) -> None:
    """
    Log dimension values (Cost Center / Profit Center) that appear in
    df_newcc but not in df_base. Purely informational.
    """
    if col not in df_base.columns or col not in df_newcc.columns:
        return

    base_vals  = set(df_base[col].dropna().astype(str).str.strip().unique())
    newcc_vals = set(df_newcc[col].dropna().astype(str).str.strip().unique())
    new_vals   = newcc_vals - base_vals

    if new_vals:
        logger.info(
            "Union: '%s' contributes %d new %s value(s) not in '%s': %s",
            newcc_name, len(new_vals), col,
            VIEW_BASEDATA,
            sorted(new_vals)[:20],   # cap log output at 20 values
        )
    else:
        logger.info(
            "Union: '%s' adds no new %s values beyond '%s'.",
            newcc_name, col, VIEW_BASEDATA,
        )


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_basedata() -> pd.DataFrame:
    """Load v_basedata — base transactional financial records."""
    return _load_view(VIEW_BASEDATA)


def load_newcc() -> pd.DataFrame:
    """Load v_newcc — additional transactional records (same schema as v_basedata)."""
    return _load_view(VIEW_NEWCC)


def load_ceco1711() -> pd.DataFrame:
    """Load ceco1711 — additional transactional records (same schema as v_basedata)."""
    return _load_view(VIEW_CECO1711)


def load_profitcenter() -> pd.DataFrame:
    """Load v_profitcenter — profit center master data."""
    return _load_view(VIEW_PROFITCENTER)


def load_costcenter() -> pd.DataFrame:
    """Load v_costcenter — cost center master data."""
    return _load_view(VIEW_COSTCENTER)


def load_gl_grouping() -> pd.DataFrame:
    """Load v__glaccountgrouping — G/L account master data and groupings."""
    return _load_view(VIEW_GL_GROUPING)


def load_glaccount() -> pd.DataFrame:
    """
    Load v_glaccount — authoritative G/L account master data.
    Provides: Flash Line Item, FS Item, FS Item Name,
              Financial Statement Line Item, G/L Account Type.
    This is the PRIMARY source for Flash grouping semantics.
    """
    return _load_view(VIEW_GLACCOUNT)


def check_connectivity() -> Dict:
    """
    Lightweight connectivity check — verifies HANA is reachable and
    v_basedata is accessible. Used by the /health endpoint.
    """
    if not HDBCLI_AVAILABLE:
        return {"connected": False, "message": "hdbcli not installed."}
    try:
        with HANAConnection() as hana:
            if hana.connection is None:
                return {"connected": False, "message": "Connection is None."}
            hana.execute_query('SELECT COUNT(*) FROM "{}"'.format(VIEW_BASEDATA))
        return {"connected": True, "message": "SAP HANA connection successful (v_basedata)."}
    except Exception as exc:
        return {"connected": False, "message": str(exc)}


def load_all_raw() -> Dict[str, pd.DataFrame]:
    """
    Load all source tables independently, then union the three transactional
    sources (v_basedata + v_newcc + ceco1711) before returning.

    Returns a dict with keys:
      'basedata'     — combined v_basedata UNION ALL v_newcc UNION ALL ceco1711
      'profitcenter' — v_profitcenter master data
      'costcenter'   — v_costcenter master data
      'gl_grouping'  — v__glaccountgrouping master data
      'glaccount'    — v_glaccount master data (authoritative Flash Line Item)

    The 'basedata' key always contains the fully combined transactional dataset.
    The enrichment pipeline (dataset_enricher.py) requires no changes.

    Union strategy (two sequential UNION ALL operations):
      Step 1: v_basedata UNION ALL v_newcc   → df_combined_1
      Step 2: df_combined_1 UNION ALL ceco1711 → df_combined

    Each step validates schema compatibility and performs an integrity row-count
    check. SchemaCompatibilityError or RuntimeError propagate to the caller.

    Raises:
        SchemaCompatibilityError: if v_newcc or ceco1711 schema is incompatible
            with v_basedata.
        RuntimeError: if any union row count does not match the sum of inputs.
    """
    logger.info("Loading all raw source tables from HANA...")

    # --- Transactional sources (loaded independently) ---
    df_basedata = load_basedata()
    df_newcc    = load_newcc()
    df_ceco1711 = load_ceco1711()

    # --- Union transactional sources (two sequential UNION ALL steps) ---
    # Step 1: v_basedata UNION ALL v_newcc
    df_combined_1 = _union_transactional(df_basedata, df_newcc)

    # Step 2: (v_basedata + v_newcc) UNION ALL ceco1711
    # Uses the same schema validation and integrity checks as Step 1.
    df_combined = _union_transactional(
        df_combined_1,
        df_ceco1711,
        base_name="v_basedata+v_newcc",
        newcc_name=VIEW_CECO1711,
    )

    # --- Master data sources ---
    df_profitcenter = load_profitcenter()
    df_costcenter   = load_costcenter()
    df_gl_grouping  = load_gl_grouping()
    df_glaccount    = load_glaccount()

    tables = {
        "basedata":     df_combined,
        "profitcenter": df_profitcenter,
        "costcenter":   df_costcenter,
        "gl_grouping":  df_gl_grouping,
        "glaccount":    df_glaccount,
    }

    logger.info("Raw table summary:")
    logger.info(
        "  %-20s → %d rows, %d columns  [v_basedata=%d + v_newcc=%d + ceco1711=%d]",
        "basedata (combined)",
        len(df_combined), len(df_combined.columns),
        len(df_basedata), len(df_newcc), len(df_ceco1711),
    )
    for name in ("profitcenter", "costcenter", "gl_grouping", "glaccount"):
        df = tables[name]
        logger.info("  %-20s → %d rows, %d columns", name, len(df), len(df.columns))

    return tables
