"""Commodity code embedding, similarity search, and LLM verification helpers.

This module ports the reusable pieces from the exploratory notebooks so that
the pipeline can be executed programmatically.
"""
from __future__ import annotations

import json
import math
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz

try:  # dotenv is optional at runtime
    from dotenv import load_dotenv

    # Load from the single .env file at api/.env
    _env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(_env_path)
except Exception:  # pragma: no cover - optional dependency
    pass

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm optional
    tqdm = None

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4.1")
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "low")
TOP_K_CODES = 5
LINEITEM_DESC_CANDIDATES = ["Description", "Item Description", "Material Description", "Short Text"]
POSSIBLE_JOIN_KEYS = ["LineItemID", "Line Item", "Item", "ItemID", "ID", "PO Item", "Position"]
CODE_COLUMN = "CODE"
DESCRIPTION_LOW_LEVEL_COLUMN = "DESCRIPTION_LOW_LEVEL"
DESCRIPTION_HIGH_LEVEL_COLUMN = "DESCRIPTION_HIGH_LEVEL"
PROCUREMENT_GROUP_COLUMN = "PROCUREMENT_GROUP"
KEYWORDS_COLUMN = "DETAILED_DESCRIPTION_KEYWORDS"
UNSPSC_CONTEXT_COLUMN = "UNSPSC_CODE_DESCRIPTION"
UNSPSC_CONTEXT_CODE_COLUMN = "REFERENCE_CODE"
UNSPSC_CONTEXT_DESCRIPTION_COLUMN = "REFERENCE_CODE_DESCRIPTION"
SUPPLIER_NAME_COLUMN = "SUPPLIER_NAME"
BUSINESS_PARTNER_ID_COLUMN = "BUSINESS_PARTNER_ID"
MATERIAL_GROUP_COLUMN = "MATERIAL_GROUP"
DATA_VERSION_COLUMN = "DATA_VERSION"
UNSPSC_CONTEXT_HEADER_CANDIDATES: Sequence[int] = (1, 0)
UNSPSC_CONTEXT_MAX_ENTRIES = 25
UNSPSC_CONTEXT_CHAR_LIMIT = 800
LLM_FIELD_CHAR_LIMIT = 1000
DEFAULT_COMMUNITY_CATALOG_PATH = Path("doc_extraction/embedding/Copy of Commodity codes list Jan 2021.xlsx")
DEFAULT_UNSPSC_CONTEXT_PATH = Path(
    "doc_extraction/embedding/UNSPSC_COMM CODE_BUYER MAPPING - tree structure - updated.xlsx"
)
DEFAULT_SUPPLIER_GROUPS_PATH = Path("data/supplier_material_groups.csv")
TAXONOMY_FIELDS = [
    CODE_COLUMN,
    DESCRIPTION_LOW_LEVEL_COLUMN,
    DESCRIPTION_HIGH_LEVEL_COLUMN,
    PROCUREMENT_GROUP_COLUMN,
]

DataFrameOrPath = Union[pd.DataFrame, str, os.PathLike]

CATALOG_COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    CODE_COLUMN: ("CODE", "Code"),
    DESCRIPTION_LOW_LEVEL_COLUMN: (
        DESCRIPTION_LOW_LEVEL_COLUMN,
        "Description Low Level",
        "Description - low level",
    ),
    DESCRIPTION_HIGH_LEVEL_COLUMN: (
        DESCRIPTION_HIGH_LEVEL_COLUMN,
        "Description High Level",
        "Description - high level",
    ),
    PROCUREMENT_GROUP_COLUMN: (
        PROCUREMENT_GROUP_COLUMN,
        "Procurement Group",
    ),
    KEYWORDS_COLUMN: (
        KEYWORDS_COLUMN,
        "Detailed description. Keywords",
    ),
    UNSPSC_CONTEXT_COLUMN: (
        UNSPSC_CONTEXT_COLUMN,
        "UNSPSC Code description",
    ),
}

SUPPLIER_COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    SUPPLIER_NAME_COLUMN: (SUPPLIER_NAME_COLUMN, "Supplier name"),
    BUSINESS_PARTNER_ID_COLUMN: (BUSINESS_PARTNER_ID_COLUMN, "Business partner ID"),
    MATERIAL_GROUP_COLUMN: (MATERIAL_GROUP_COLUMN, "Material Group"),
}

UNSPSC_COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "LEVEL": ("LEVEL", "Level"),
    "UNSPSC_CODE": ("UNSPSC_CODE", "UNSPSC Code"),
    "SORT": ("SORT", "Sort"),
    UNSPSC_CONTEXT_COLUMN: (UNSPSC_CONTEXT_COLUMN, "UNSPSC Code description"),
    "ACTIVE_STATUS_IN_ARIBA": ("ACTIVE_STATUS_IN_ARIBA", "Active status in Ariba"),
    UNSPSC_CONTEXT_CODE_COLUMN: (UNSPSC_CONTEXT_CODE_COLUMN,),
    UNSPSC_CONTEXT_DESCRIPTION_COLUMN: (UNSPSC_CONTEXT_DESCRIPTION_COLUMN,),
    DATA_VERSION_COLUMN: (DATA_VERSION_COLUMN,),
}


def _stringify_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all DataFrame column names to strings.

    Args:
        df: Input DataFrame with potentially non-string column names.

    Returns:
        A copy of the DataFrame with all column names converted to strings.
    """
    df = df.copy()
    df.columns = [str(col) for col in df.columns]
    return df


def _read_any(
    path: str,
    csv_sep: str = ",",
    csv_encoding: str = "utf-8",
    csv_header: Union[int, str, None] = "infer",
) -> pd.DataFrame:
    """Read a DataFrame from various file formats (Excel, CSV, TSV).

    Supports .xlsx, .xls, .csv, .txt, and .tsv file extensions.
    Automatically converts all column names to strings.

    Args:
        path: File path to read from.
        csv_sep: Separator for CSV files (default: ",").
        csv_encoding: Encoding for CSV files (default: "utf-8").
        csv_header: Row number to use as header for CSV files (default: "infer").

    Returns:
        DataFrame with string column names.

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in {".xlsx", ".xls"}:
        return _stringify_columns(pd.read_excel(path))
    if ext in {".csv", ".txt", ".tsv"}:
        sep = "\t" if ext == ".tsv" else csv_sep
        return _stringify_columns(pd.read_csv(path, sep=sep, encoding=csv_encoding, header=csv_header))
    raise ValueError(f"Unsupported file extension: {path}")


def _coerce_df(
    value: DataFrameOrPath,
    csv_sep: str,
    csv_encoding: str,
    csv_header: Union[int, str, None],
) -> pd.DataFrame:
    """Convert a DataFrame or file path to a DataFrame with string columns.

    If the input is already a DataFrame, returns a copy with stringified columns.
    If the input is a path, reads the file and returns a DataFrame with stringified columns.

    Args:
        value: Either a DataFrame or a file path (str or PathLike).
        csv_sep: Separator for CSV files.
        csv_encoding: Encoding for CSV files.
        csv_header: Row number to use as header for CSV files.

    Returns:
        DataFrame with all column names as strings.

    Raises:
        TypeError: If the input is neither a DataFrame nor a path.
    """
    if isinstance(value, pd.DataFrame):
        return _stringify_columns(value)
    if isinstance(value, (str, os.PathLike)):
        return _read_any(str(value), csv_sep=csv_sep, csv_encoding=csv_encoding, csv_header=csv_header)
    raise TypeError(f"Expected DataFrame or path, received {type(value)!r}")


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _canonicalize_columns(
    df: pd.DataFrame,
    aliases: Dict[str, Sequence[str]],
    *,
    required: Sequence[str],
    dataset_name: str,
) -> pd.DataFrame:
    normalized_columns = {_normalize_name(col): str(col) for col in df.columns}
    rename_map: Dict[str, str] = {}

    for canonical, candidates in aliases.items():
        for candidate in candidates:
            current = normalized_columns.get(_normalize_name(candidate))
            if current:
                rename_map[current] = canonical
                break

    renamed = df.rename(columns=rename_map)
    missing = [column for column in required if column not in renamed.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    return _stringify_columns(renamed)


def _find_join_keys(df_a: pd.DataFrame, df_b: pd.DataFrame) -> List[Tuple[str, str]]:
    """Find common column names between two DataFrames for joining.

    First attempts to match columns from a predefined list of common join key candidates.
    If no matches are found, falls back to finding any common column names.

    Args:
        df_a: First DataFrame to compare.
        df_b: Second DataFrame to compare.

    Returns:
        List of tuples containing (column_name_in_df_a, column_name_in_df_b) pairs.
        Returns empty list if no common keys are found.
    """
    keys: List[Tuple[str, str]] = []
    a_cols = {str(col).lower(): str(col) for col in df_a.columns}
    b_cols = {str(col).lower(): str(col) for col in df_b.columns}
    for candidate in POSSIBLE_JOIN_KEYS:
        lowered = candidate.lower()
        if lowered in a_cols and lowered in b_cols:
            keys.append((a_cols[lowered], b_cols[lowered]))
    if not keys:
        common = list(set(df_a.columns) & set(df_b.columns))
        if common:
            first = str(common[0])
            keys.append((first, first))
    return keys


def _concat_kv_text(row: pd.Series) -> str:
    """Concatenate all non-empty column-value pairs from a DataFrame row into a text string.

    Each pair is formatted as "column: value" and pairs are joined with " | ".

    Args:
        row: A pandas Series representing a single row from a DataFrame.

    Returns:
        A string containing all non-empty key-value pairs joined with " | ".
    """
    parts: List[str] = []
    for column, value in row.items():
        if pd.notna(value) and str(value).strip():
            parts.append(f"{column}: {value}")
    return " | ".join(parts)


def _join_unique_non_empty(
    values: Iterable[Any],
    *,
    max_entries: Optional[int] = None,
    char_limit: Optional[int] = None,
) -> str:
    """Join unique non-empty values with optional truncation.

    Args:
        values: An iterable containing raw values.
        max_entries: Optional limit on how many unique values to keep.
        char_limit: Optional maximum length for the final concatenated string.

    Returns:
        A semicolon-separated string of unique values, optionally truncated.
    """
    ordered: Dict[str, None] = {}
    for value in values:
        text = str(value).strip()
        if not text or text in ordered:
            continue
        ordered[text] = None
        if max_entries is not None and len(ordered) >= max_entries:
            break

    combined = "; ".join(ordered.keys())
    if char_limit is not None and len(combined) > char_limit:
        combined = combined[:char_limit].rstrip("; ,") + "…"
    return combined


def _stringify_chat_content(content: Any) -> str:
    """Coerce chat completion content (string or segmented list) to plain text."""
    if isinstance(content, list):
        pieces: List[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("text"):
                    pieces.append(str(part["text"]))
                elif part.get("content"):
                    pieces.append(str(part["content"]))
            elif part:
                pieces.append(str(part))
        return " ".join(pieces).strip()
    if content is None:
        return ""
    return str(content)


CONFIDENCE_WORD_MAP = {
    "high": 0.9,
    "medium": 0.6,
    "med": 0.6,
    "low": 0.35,
    "very high": 0.95,
    "very low": 0.2,
}


def _coerce_confidence(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        lowered = text.lower()
        if lowered in CONFIDENCE_WORD_MAP:
            return CONFIDENCE_WORD_MAP[lowered]
        match = re.search(r"\d+(?:\.\d+)?", text)
        if match:
            number = float(match.group(0))
            if number > 1.0:
                number = number / 100.0
            return number
    return 0.0


def _load_unspsc_context_map(source: Optional[DataFrameOrPath]) -> Dict[str, str]:
    """Load UNSPSC context descriptions and create a code-to-description mapping.

    Reads a DataFrame or workbook containing reference codes and their UNSPSC descriptions,
    groups multiple descriptions per code, and returns a dictionary mapping codes to descriptions.

    Args:
        source: Optional DataFrame or path to the UNSPSC context data.

    Returns:
        Dictionary mapping commodity codes (uppercase strings) to their context descriptions.
        Returns empty dictionary if the source is not found or doesn't contain required columns.
    """
    if source is None:
        return {}

    context_df: Optional[pd.DataFrame] = None
    if isinstance(source, pd.DataFrame):
        context_df = _stringify_columns(source.copy())
    else:
        file_path = Path(source).expanduser()
        if not file_path.exists():
            warnings.warn(f"UNSPSC context file not found: {file_path}")
            return {}

        for header in UNSPSC_CONTEXT_HEADER_CANDIDATES:
            try:
                candidate = pd.read_excel(file_path, header=header)
            except Exception as exc:  # pragma: no cover - depends on Excel contents
                warnings.warn(f"Failed to read UNSPSC context file '{file_path}' (header={header}): {exc}")
                continue
            try:
                context_df = _canonicalize_columns(
                    candidate,
                    UNSPSC_COLUMN_ALIASES,
                    required=(UNSPSC_CONTEXT_CODE_COLUMN, UNSPSC_CONTEXT_COLUMN),
                    dataset_name="UNSPSC context data",
                )
                break
            except ValueError:
                continue

    if context_df is None:
        warnings.warn(
            "UNSPSC context data is missing the expected columns "
            f"({UNSPSC_CONTEXT_CODE_COLUMN!r}, {UNSPSC_CONTEXT_COLUMN!r}); skipping enrichment."
        )
        return {}

    try:
        context_df = _canonicalize_columns(
            context_df,
            UNSPSC_COLUMN_ALIASES,
            required=(UNSPSC_CONTEXT_CODE_COLUMN, UNSPSC_CONTEXT_COLUMN),
            dataset_name="UNSPSC context data",
        )
    except ValueError as exc:
        warnings.warn(f"{exc}; skipping enrichment.")
        return {}

    trimmed = context_df[[UNSPSC_CONTEXT_CODE_COLUMN, UNSPSC_CONTEXT_COLUMN]].dropna()
    if trimmed.empty:
        return {}

    trimmed[UNSPSC_CONTEXT_CODE_COLUMN] = trimmed[UNSPSC_CONTEXT_CODE_COLUMN].astype(str).str.strip().str.upper()
    trimmed[UNSPSC_CONTEXT_COLUMN] = trimmed[UNSPSC_CONTEXT_COLUMN].astype(str).str.strip()
    trimmed = trimmed[(trimmed[UNSPSC_CONTEXT_CODE_COLUMN] != "") & (trimmed[UNSPSC_CONTEXT_COLUMN] != "")]
    if trimmed.empty:
        return {}

    grouped = trimmed.groupby(UNSPSC_CONTEXT_CODE_COLUMN)[UNSPSC_CONTEXT_COLUMN].apply(
        lambda series: _join_unique_non_empty(
            series,
            max_entries=UNSPSC_CONTEXT_MAX_ENTRIES,
            char_limit=UNSPSC_CONTEXT_CHAR_LIMIT,
        )
    )
    grouped = grouped[grouped.str.strip().astype(bool)]
    return grouped.to_dict()


def _pick_description_column(df: pd.DataFrame) -> Optional[str]:
    """Find the most appropriate description column in a DataFrame.

    First checks for columns matching predefined description column candidates.
    If no match is found, searches for any column containing "desc" in its name (case-insensitive).

    Args:
        df: DataFrame to search for description columns.

    Returns:
        The name of the first matching description column, or None if no match is found.
    """
    lowered = {str(col).lower(): str(col) for col in df.columns}
    for candidate in LINEITEM_DESC_CANDIDATES:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    for column in df.columns:
        if "desc" in str(column).lower():
            return str(column)
    return None


def _find_sheet_like(xls: pd.ExcelFile, target_fragment: str) -> Optional[str]:
    """Find an Excel sheet name that contains a target fragment (case-insensitive).

    First attempts a direct substring match, then tries matching with whitespace removed.

    Args:
        xls: An open ExcelFile object to search.
        target_fragment: The text fragment to search for in sheet names.

    Returns:
        The first matching sheet name, or None if no match is found.
    """
    fragment = target_fragment.lower()
    for name in xls.sheet_names:
        if fragment in str(name).lower():
            return name
    squeezed_fragment = re.sub(r"\s+", "", fragment)
    for name in xls.sheet_names:
        squeezed_name = re.sub(r"\s+", "", str(name).lower())
        if squeezed_fragment in squeezed_name:
            return name
    return None


def _detect_code_column(catalog: pd.DataFrame, explicit_hint: Optional[str]) -> Optional[str]:
    """Detect the code column in a catalog DataFrame.

    First checks for an exact match with the explicit hint (case-insensitive).
    Then searches for a column named exactly "code" (case-insensitive).
    Finally searches for any column containing "code" in its name (case-insensitive).

    Args:
        catalog: DataFrame to search for code columns.
        explicit_hint: Optional explicit column name hint to check first.

    Returns:
        The name of the detected code column, or None if no match is found.
    """
    columns = [str(col) for col in catalog.columns]
    if explicit_hint:
        for column in columns:
            if column.lower() == explicit_hint.lower():
                return column
    for column in columns:
        if column.lower() == "code":
            return column
    for column in columns:
        if "code" in column.lower():
            return column
    return None


def _load_catalog_df(
    source: DataFrameOrPath,
    *,
    product_structure_sheet_name: Optional[str],
    product_structure_sheet_hint: str,
) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        catalog = _stringify_columns(source.copy())
    else:
        catalog_path = Path(source).expanduser()
        if not catalog_path.exists():
            raise FileNotFoundError(f"Community codes file not found: {catalog_path}")
        xls = pd.ExcelFile(catalog_path)
        sheet_name = _select_sheet(xls, product_structure_sheet_name, product_structure_sheet_hint)
        catalog = _stringify_columns(pd.read_excel(catalog_path, sheet_name=sheet_name))

    return _canonicalize_columns(
        catalog,
        CATALOG_COLUMN_ALIASES,
        required=(CODE_COLUMN, DESCRIPTION_LOW_LEVEL_COLUMN, DESCRIPTION_HIGH_LEVEL_COLUMN, PROCUREMENT_GROUP_COLUMN),
        dataset_name="Commodity catalog",
    )


def _load_supplier_material_groups(source: DataFrameOrPath) -> Dict[str, Dict[str, Any]]:
    """Load supplier material groups from CSV and build a lookup dictionary.

    Reads supplier group data and creates a mapping from normalized
    supplier names to their metadata (original name, BPID, and material groups).
    Handles suppliers that appear multiple times with different material groups by aggregating them.

    Args:
        source: DataFrame or path to the supplier material groups data.

    Returns:
        Dictionary mapping normalized supplier names to:
        {
            "original_name": str,  # Canonical supplier name from CSV
            "bpid": str,  # Business Partner ID
            "material_groups": List[str]  # List of all material groups for this supplier
        }

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the CSV is missing required columns.
    """
    try:
        df = _coerce_df(source, csv_sep=",", csv_encoding="utf-8", csv_header="infer")
        df = _canonicalize_columns(
            df,
            SUPPLIER_COLUMN_ALIASES,
            required=(SUPPLIER_NAME_COLUMN, BUSINESS_PARTNER_ID_COLUMN, MATERIAL_GROUP_COLUMN),
            dataset_name="Supplier material groups",
        )
    except FileNotFoundError:
        warnings.warn(f"Supplier material groups file not found: {source}. Supplier filtering will be disabled.")
        return {}
    except ValueError as exc:
        warnings.warn(f"{exc}. Supplier filtering will be disabled.")
        return {}

    # Clean and normalize the data
    df = df.dropna(subset=[SUPPLIER_NAME_COLUMN, BUSINESS_PARTNER_ID_COLUMN, MATERIAL_GROUP_COLUMN])
    df[SUPPLIER_NAME_COLUMN] = df[SUPPLIER_NAME_COLUMN].astype(str).str.strip()
    df[BUSINESS_PARTNER_ID_COLUMN] = df[BUSINESS_PARTNER_ID_COLUMN].astype(str).str.strip()
    df[MATERIAL_GROUP_COLUMN] = df[MATERIAL_GROUP_COLUMN].astype(str).str.strip()

    # Filter out invalid entries
    df = df[
        (df[SUPPLIER_NAME_COLUMN] != "")
        & (df[BUSINESS_PARTNER_ID_COLUMN] != "")
        & (df[MATERIAL_GROUP_COLUMN] != "")
        & (df[BUSINESS_PARTNER_ID_COLUMN] != "#")  # Exclude special cases like "Not assigned"
    ]

    if df.empty:
        warnings.warn("Supplier material groups CSV contains no valid entries. Supplier filtering will be disabled.")
        return {}

    # Build lookup dictionary by aggregating material groups per supplier
    supplier_lookup = {}
    for supplier_name in df[SUPPLIER_NAME_COLUMN].unique():
        supplier_rows = df[df[SUPPLIER_NAME_COLUMN] == supplier_name]
        # Use the first BPID if multiple exist (should be consistent, but handle edge cases)
        bpid = supplier_rows[BUSINESS_PARTNER_ID_COLUMN].iloc[0]
        # Collect all unique material groups for this supplier
        material_groups = supplier_rows[MATERIAL_GROUP_COLUMN].unique().tolist()

        # Store with normalized key for fuzzy matching
        normalized_key = supplier_name.lower().strip()
        supplier_lookup[normalized_key] = {
            "original_name": supplier_name,
            "bpid": bpid,
            "material_groups": material_groups,
        }

    return supplier_lookup


def _match_vendor_to_supplier(
    vendor_name: Optional[str],
    supplier_lookup: Dict[str, Dict[str, Any]],
    threshold: float = 70.0,
) -> Optional[Dict[str, Any]]:
    """Fuzzy match a vendor name to a supplier in the lookup dictionary.

    Uses a multi-stage matching strategy with RapidFuzz to handle various edge cases:
    1. High-confidence exact/ratio matching (85%+)
    2. Token-based matching for abbreviations and word order (70%+)
    3. Ambiguity check to avoid false positives when multiple suppliers match similarly

    Args:
        vendor_name: The vendor name extracted from the document (can be None or empty).
        supplier_lookup: Dictionary of suppliers from _load_supplier_material_groups().
        threshold: Minimum fuzzy match score (0-100) to consider a match (default: 70.0).

    Returns:
        Dictionary with match details if a match is found:
        {
            "matched_supplier_name": str,  # The canonical supplier name
            "bpid": str,  # Business Partner ID
            "material_groups": List[str],  # Material groups for this supplier
            "match_score": float,  # Fuzzy match score (0-100)
            "match_method": str  # Matching method used: "exact", "ratio", or "token_sort"
        }
        Returns None if no match is found or vendor_name is invalid.
    """
    if not vendor_name or not isinstance(vendor_name, str) or not vendor_name.strip():
        return None

    if not supplier_lookup:
        return None

    vendor_name_clean = vendor_name.strip()
    vendor_name_lower = vendor_name_clean.lower()
    supplier_names = list(supplier_lookup.keys())

    # Stage 1: Try exact match first (fastest)
    if vendor_name_lower in supplier_lookup:
        supplier_data = supplier_lookup[vendor_name_lower]
        return {
            "matched_supplier_name": supplier_data["original_name"],
            "bpid": supplier_data["bpid"],
            "material_groups": supplier_data["material_groups"],
            "match_score": 100.0,
            "match_method": "exact",
        }

    # Stage 2: Try high-confidence ratio match (85%+)
    # This catches very similar names with minor differences
    high_conf_match = process.extractOne(
        vendor_name_lower,
        supplier_names,
        scorer=fuzz.ratio,
        score_cutoff=85.0,
    )

    if high_conf_match:
        matched_name, score, _ = high_conf_match
        supplier_data = supplier_lookup[matched_name]
        return {
            "matched_supplier_name": supplier_data["original_name"],
            "bpid": supplier_data["bpid"],
            "material_groups": supplier_data["material_groups"],
            "match_score": score,
            "match_method": "ratio",
        }

    # Stage 3: Try token-based matching for abbreviations and word order
    # This handles cases like "Acme Industrial Controls AB" vs "Acme Ind. Ctrl AB".
    # Get top 3 candidates to check for ambiguity
    token_matches = process.extract(
        vendor_name_lower,
        supplier_names,
        scorer=fuzz.token_sort_ratio,
        limit=3,
    )

    if not token_matches or token_matches[0][1] < threshold:
        return None

    best_match_name, best_score, _ = token_matches[0]

    # Check for ambiguity: if multiple candidates are very close in score,
    # we need to be more careful to avoid false positives
    if len(token_matches) > 1:
        second_score = token_matches[1][1]
        score_gap = best_score - second_score

        # If the gap is less than 5 points and both are above threshold,
        # it's ambiguous - use additional checks
        if score_gap < 5.0 and second_score >= threshold:
            # Additional disambiguation: prefer longer match (more specific)
            # or exact substring match
            best_supplier = supplier_lookup[best_match_name]["original_name"]
            second_supplier = supplier_lookup[token_matches[1][0]]["original_name"]

            # Check if vendor name contains one supplier name as substring
            vendor_lower = vendor_name_clean.lower()
            best_in_vendor = best_supplier.lower() in vendor_lower
            second_in_vendor = second_supplier.lower() in vendor_lower

            if best_in_vendor and not second_in_vendor:
                # Best match is substring, second is not - use best
                pass
            elif second_in_vendor and not best_in_vendor:
                # Second match is substring, best is not - use second
                best_match_name = token_matches[1][0]
                best_score = second_score
            elif len(best_supplier) > len(second_supplier) and abs(len(best_supplier) - len(vendor_name_clean)) < abs(len(second_supplier) - len(vendor_name_clean)):
                # Best match is closer in length to vendor name
                pass
            else:
                # Still ambiguous - be conservative and don't match
                warnings.warn(
                    f"Ambiguous supplier match for '{vendor_name}': "
                    f"'{best_supplier}' ({best_score:.1f}) vs '{second_supplier}' ({second_score:.1f}). "
                    f"Skipping match to avoid false positive."
                )
                return None

    supplier_data = supplier_lookup[best_match_name]
    return {
        "matched_supplier_name": supplier_data["original_name"],
        "bpid": supplier_data["bpid"],
        "material_groups": supplier_data["material_groups"],
        "match_score": best_score,
        "match_method": "token_sort",
    }


def _filter_catalog_by_material_groups(
    catalog: pd.DataFrame,
    material_groups: List[str],
    code_column: str,
) -> pd.DataFrame:
    """Filter a commodity code catalog to only include specified material groups.

    Args:
        catalog: The full commodity code catalog DataFrame.
        material_groups: List of material group codes to filter by.
        code_column: The name of the column containing commodity codes.

    Returns:
        Filtered DataFrame containing only rows where the code is in material_groups.
        Returns the original catalog if filtering would result in an empty DataFrame.
    """
    if not material_groups or code_column not in catalog.columns:
        return catalog

    # Filter catalog to only include codes in the material groups
    filtered = catalog[catalog[code_column].isin(material_groups)]

    # Only return filtered catalog if it's not empty (safety check)
    if len(filtered) == 0:
        warnings.warn(
            f"Filtering by material groups {material_groups} resulted in empty catalog. "
            "Using full catalog instead."
        )
        return catalog

    return filtered


def _select_sheet(xls: pd.ExcelFile, sheet_name_exact: Optional[str], sheet_hint: str) -> str:
    """Select a sheet from an Excel file by exact name or by searching for a hint.

    First attempts an exact match (case-insensitive) if sheet_name_exact is provided.
    Otherwise, searches for a sheet containing the hint fragment.

    Args:
        xls: An open ExcelFile object.
        sheet_name_exact: Optional exact sheet name to match (case-insensitive).
        sheet_hint: Fallback hint to search for in sheet names.

    Returns:
        The name of the selected sheet.

    Raises:
        ValueError: If no matching sheet is found.
    """
    if sheet_name_exact:
        for name in xls.sheet_names:
            if str(name).lower() == sheet_name_exact.lower():
                return name
    sheet = _find_sheet_like(xls, sheet_hint)
    if sheet:
        return sheet
    raise ValueError(f"Cannot find sheet '{sheet_name_exact or sheet_hint}'. Available: {xls.sheet_names}")


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Normalize a matrix by dividing each row by its L2 norm.

    Rows with zero norm are left unchanged (divided by 1.0 to avoid division by zero).

    Args:
        matrix: 2D numpy array where each row is a vector to normalize.

    Returns:
        Normalized matrix with unit-length rows (L2 norm = 1.0 for each row).
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix between two sets of normalized vectors.

    Assumes input matrices are already normalized. Computes dot product between
    each row of 'a' and each row of 'b', resulting in a similarity matrix.

    Args:
        a: First matrix of normalized vectors (shape: [n_a, dim]).
        b: Second matrix of normalized vectors (shape: [n_b, dim]).

    Returns:
        Similarity matrix of shape [n_a, n_b] where each element [i, j]
        is the cosine similarity between a[i] and b[j].
    """
    return a @ b.T


class _ProxyEmbeddings:
    """Proxy client for generating text embeddings via Gen AI Hub or hash-based fallback.

    Attempts to use SAP Gen AI Hub proxy client for OpenAI embeddings.
    Falls back to hash-based embeddings if the proxy client is not available.
    """

    def __init__(self, model_name: str) -> None:
        """Initialize the embedding proxy client.

        Args:
            model_name: Name of the embedding model to use (e.g., "text-embedding-3-large").
        """
        self.model_name = model_name
        self.enabled = False
        self._backend = None
        try:  # pragma: no cover - requires SAP environment
            from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
            from gen_ai_hub.proxy.langchain.openai import OpenAIEmbeddings

            proxy_client = get_proxy_client("gen-ai-hub")
            self._backend = OpenAIEmbeddings(proxy_model_name=self.model_name, proxy_client=proxy_client)
            self.enabled = True
        except Exception as exc:
            warnings.warn(
                "Gen AI Hub proxy embedding client not available. "
                f"Reason: {exc}. Falling back to offline hash embeddings."
            )

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of text strings.

        Uses the Gen AI Hub proxy client if available, otherwise falls back to
        hash-based embeddings.

        Args:
            texts: List of text strings to embed.

        Returns:
            Numpy array of shape [len(texts), embedding_dim] containing the embeddings.
        """
        if not self.enabled or self._backend is None:
            return _cheap_hash_embeddings(texts, dim=384)
        vectors = self._backend.embed_documents(texts)
        return np.array(vectors, dtype="float32")


def _cheap_hash_embeddings(texts: List[str], dim: int = 384) -> np.ndarray:
    """Generate hash-based embeddings as a fallback when real embeddings are unavailable.

    Creates deterministic embeddings by hashing each text string with SHA-256
    and converting the hash bytes to a normalized float vector.

    Args:
        texts: List of text strings to embed.
        dim: Desired embedding dimension (default: 384).

    Returns:
        Numpy array of shape [len(texts), dim] containing hash-based embeddings.
    """
    import hashlib

    matrix = np.zeros((len(texts), dim), dtype="float32")
    iterator: Iterable[str] = texts if tqdm is None else tqdm(texts, desc="Embedding (fallback)", unit="row")
    for idx, text in enumerate(iterator):
        digest = hashlib.sha256(str(text).encode("utf-8")).digest()
        repeats = math.ceil(dim / len(digest))
        buffer = (digest * repeats)[:dim]
        matrix[idx, :] = np.frombuffer(buffer, dtype=np.uint8) / 255.0
    return matrix


def _topk_indices(sim_matrix: np.ndarray, k: int) -> np.ndarray:
    """Find the indices of the top-k highest similarity scores for each row.

    For each row in the similarity matrix, returns the column indices corresponding
    to the k highest similarity values, sorted in descending order.

    Args:
        sim_matrix: 2D similarity matrix of shape [n_rows, n_cols].
        k: Number of top indices to return per row.

    Returns:
        Array of shape [n_rows, k] containing the column indices of top-k similarities
        for each row, sorted in descending order of similarity.
    """
    if k >= sim_matrix.shape[1]:
        return np.argsort(-sim_matrix, axis=1)[:, :k]
    partition = np.argpartition(sim_matrix, -k, axis=1)[:, -k:]
    rows = np.arange(sim_matrix.shape[0])[:, None]
    sims_subset = sim_matrix[rows, partition]
    order = np.argsort(-sims_subset, axis=1)
    return partition[rows, order]


def _build_catalog_row_map(catalog: pd.DataFrame, code_column: str) -> Dict[str, Dict[str, str]]:
    mapping: Dict[str, Dict[str, str]] = {}
    for _, row in catalog.iterrows():
        raw_code = str(row[code_column]).strip()
        if not raw_code:
            continue
        normalized_code = raw_code.upper()
        serialized: Dict[str, str] = {}
        for column, value in row.items():
            if pd.isna(value):
                continue
            text = str(value).strip()
            if not text:
                continue
            if len(text) > LLM_FIELD_CHAR_LIMIT:
                text = text[:LLM_FIELD_CHAR_LIMIT] + "…"
            serialized[str(column)] = text
        mapping[normalized_code] = serialized
    return mapping


def _codes_csv_to_candidate_objs(codes_csv: str, code_map: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
    codes = [code.strip() for code in (codes_csv or "").split(",") if code.strip()]
    candidates: List[Dict[str, Any]] = []
    for code in codes:
        normalized = code.upper()
        candidates.append({"code": normalized, "catalog_row": code_map.get(normalized, {})})
    return candidates


LLM_SYSTEM_PROMPT = """You are an expert procurement category classifier.\n\n""" \
    "Your task is to match purchase items to the correct Commodity Code from this company's procurement taxonomy.\n\n" \
    "🏗️ TAXONOMY STRUCTURE:\nThe taxonomy has a hierarchical structure:\n" \
    "1. Procurement Group (highest level - e.g., \"DIGITAL\", \"PROPERTY & ENVIRONMENTAL SERVICES\")\n" \
    "2. Description - high level (category - e.g., \"DIGITAL HARDWARE\", \"FACILITY MANAGEMENT\")\n" \
    "3. Code (unique identifier - e.g., CF01, GR01)\n" \
    "4. Description - low level (detailed name - e.g., \"CF01 CORPORATE DIGITAL WORKPLACE\")\n" \
    "5. Detailed description. Keywords (specific items/keywords)\n\n" \
    "🎯 MATCHING GUIDELINES:\n" \
    "1. Prioritize semantic fit: match the item's purpose/function, not just word overlap.\n" \
    "2. Use the hierarchy (Procurement Group → High Level → Keywords).\n" \
    "3. Focus on the Keywords field for specific fits.\n" \
    "4. Consider business context (office vs. IT vs. manufacturing vs. services).\n\n" \
    "🔍 PROCESS:\n" \
    "1. Read the item description.\n" \
    "2. For each candidate, check if procurement group, high level, and keywords align.\n" \
    "3. Select the BEST match or respond \"UNSURE\" if none fit.\n" \
    "4. Explain reasoning briefly.\n\n" \
    "⚠️ If no candidate fits, return \"UNSURE\" with low confidence. Do not force matches.\n\n" \
    "📤 OUTPUT JSON ONLY (strict schema): {\"suggested_code\": CODE_or_UNSURE, \"confidence\": 0-1 decimal, \"reason\": text}.\n" \
    "Always provide `confidence` as a numeric value between 0 and 1 (e.g., 0.72).\n"


class _LlmVerifier:
    def __init__(self, model_name: str, reasoning_effort: str = LLM_REASONING_EFFORT) -> None:
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self._chat = None
        self.enabled = False
        try:  # pragma: no cover - requires SAP environment
            from gen_ai_hub.proxy.native.openai import chat

            self._chat = chat
            self.enabled = True
        except Exception as exc:
            warnings.warn(
                "Gen AI Hub native chat client not available for verification. "
                f"Reason: {exc}. Falling back to heuristic suggestions."
            )

    def verify(self, context_text: str, candidates: List[Dict[str, Any]], mode_label: str) -> Dict[str, Any]:
        if not candidates:
            return {"suggested_code": "UNSURE", "confidence": 0.0, "reason": "No candidates provided."}

        if self.enabled and self._chat is not None:
            formatted_candidates = []
            for idx, candidate in enumerate(candidates, 1):
                catalog_row = candidate.get("catalog_row") or {}
                formatted_candidates.append(
                    {
                        "rank": idx,
                        "code": candidate.get("code"),
                        "procurement_group": catalog_row.get(PROCUREMENT_GROUP_COLUMN, "N/A"),
                        "high_level": catalog_row.get(DESCRIPTION_HIGH_LEVEL_COLUMN, "N/A"),
                        "low_level": catalog_row.get(DESCRIPTION_LOW_LEVEL_COLUMN, "N/A"),
                        "keywords": catalog_row.get(KEYWORDS_COLUMN, "N/A"),
                    }
                )

            user_payload = {
                "task": f"Match this item to the best commodity code ({mode_label} mode)",
                "item_description": context_text,
                "available_candidates": formatted_candidates,
                "instructions": "Analyze each candidate and select the best match. Use concise reasoning.",
            }

            try:
                call_kwargs: Dict[str, Any] = {
                    "messages": [
                        {"role": "system", "content": LLM_SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
                    ],
                    "model": self.model_name,
                }
                if self.reasoning_effort and self.model_name.startswith("gpt-5"):
                    call_kwargs["reasoning_effort"] = self.reasoning_effort

                response = self._chat.completions.create(**call_kwargs)

                choices = getattr(response, "choices", None)
                if choices is None and isinstance(response, dict):
                    choices = response.get("choices")
                if choices:
                    first_choice = choices[0]
                    if isinstance(first_choice, dict):
                        message = first_choice.get("message") or {}
                        content_obj = message.get("content") or first_choice.get("content") or ""
                    else:
                        message = getattr(first_choice, "message", None)
                        content_obj = getattr(message, "content", None) or getattr(first_choice, "content", "")
                else:
                    content_obj = getattr(response, "content", None)
                    if content_obj is None and isinstance(response, dict):
                        content_obj = response.get("content")
                    if content_obj is None:
                        content_obj = response

                content = _stringify_chat_content(content_obj)
                match = re.search(r"\{[\s\S]*\}", content)
                if match:
                    data = json.loads(match.group(0))
                    proposed = str(data.get("suggested_code", "")).strip().upper() or "UNSURE"
                    valid_codes = {str(cand["code"]).upper() for cand in candidates}
                    if proposed not in valid_codes and proposed != "UNSURE":
                        proposed = "UNSURE"
                    confidence = _coerce_confidence(data.get("confidence", 0.0))
                    reason = str(data.get("reason", "")).strip()
                    return {
                        "suggested_code": proposed,
                        "confidence": max(0.0, min(1.0, confidence)),
                        "reason": reason or "LLM provided no explanation",
                    }
            except Exception as exc:
                warnings.warn(f"LLM verification failed ({mode_label}): {exc}")

        fallback_code = candidates[0].get("code", "UNSURE")
        fallback_reason = "Fallback heuristic (LLM unavailable)"
        return {
            "suggested_code": fallback_code,
            "confidence": 0.55 if fallback_code != "UNSURE" else 0.0,
            "reason": fallback_reason,
        }


def _run_llm_verification(
    output_df: pd.DataFrame,
    *,
    catalog: pd.DataFrame,
    code_column: str,
    llm_model: str,
    min_confidence: float,
) -> None:
    if "Codes_Full_Top5" not in output_df.columns or "Codes_Desc_Top5" not in output_df.columns:
        warnings.warn("LLM verification skipped: missing Top-5 code columns.")
        return

    print("🤖 Running LLM verification (improved prompt)...")
    code_map = _build_catalog_row_map(catalog, code_column)
    verifier = _LlmVerifier(llm_model)

    suggestions_full: List[str] = []
    conf_full: List[float] = []
    reason_full: List[str] = []
    suggestions_desc: List[str] = []
    conf_desc: List[float] = []
    reason_desc: List[str] = []

    iterator: Iterable[int]
    if tqdm is not None:
        iterator = tqdm(range(len(output_df)), desc="🔮 LLM suggestions", unit="row")
    else:
        iterator = range(len(output_df))

    for idx in iterator:
        row = output_df.iloc[idx]
        cand_full = _codes_csv_to_candidate_objs(str(row.get("Codes_Full_Top5", "")), code_map)
        cand_desc = _codes_csv_to_candidate_objs(str(row.get("Codes_Desc_Top5", "")), code_map)

        res_full = verifier.verify(str(row.get("_full_text", ""))[:4000], cand_full, "FULL")
        res_desc = verifier.verify(str(row.get("_desc_text", ""))[:4000], cand_desc, "DESC")

        suggestions_full.append(res_full["suggested_code"])
        conf_full.append(res_full["confidence"])
        reason_full.append(res_full["reason"])
        suggestions_desc.append(res_desc["suggested_code"])
        conf_desc.append(res_desc["confidence"])
        reason_desc.append(res_desc["reason"])

    output_df["LLM_Suggestion_Full"] = suggestions_full
    output_df["LLM_Confidence_Full"] = conf_full
    output_df["LLM_Reason_Full"] = reason_full
    output_df["LLM_Suggestion_Desc"] = suggestions_desc
    output_df["LLM_Confidence_Desc"] = conf_desc
    output_df["LLM_Reason_Desc"] = reason_desc
    output_df["Block_By_LLM_Full"] = output_df["LLM_Confidence_Full"].fillna(0.0) < float(min_confidence)
    output_df["Block_By_LLM_Desc"] = output_df["LLM_Confidence_Desc"].fillna(0.0) < float(min_confidence)

    unsure_full = int((output_df["LLM_Suggestion_Full"] == "UNSURE").sum())
    unsure_desc = int((output_df["LLM_Suggestion_Desc"] == "UNSURE").sum())
    print(
        f"   • UNSURE counts — Full: {unsure_full}/{len(output_df)}, Desc: {unsure_desc}/{len(output_df)}"
    )
    print(
        f"   • Mean confidence — Full: {output_df['LLM_Confidence_Full'].mean():.3f}, "
        f"Desc: {output_df['LLM_Confidence_Desc'].mean():.3f}"
    )


def run_community_code_matching(
    line_items: DataFrameOrPath,
    *,
    community_codes_path: DataFrameOrPath = DEFAULT_COMMUNITY_CATALOG_PATH,
    unspsc_context_path: Optional[DataFrameOrPath] = DEFAULT_UNSPSC_CONTEXT_PATH,
    supplier_groups_path: Optional[DataFrameOrPath] = DEFAULT_SUPPLIER_GROUPS_PATH,
    output_path: Union[str, os.PathLike],
    headers: Optional[DataFrameOrPath] = None,
    product_structure_sheet_hint: str = "Product category Structure",
    product_structure_sheet_name: Optional[str] = None,
    code_column_hint: Optional[str] = None,
    embedding_model: str = EMBEDDING_MODEL,
    llm_verify: bool = False,
    llm_model: Optional[str] = None,
    llm_min_confidence: float = 0.6,
    enable_supplier_filtering: bool = True,
    supplier_match_threshold: float = 70.0,
    retry_confidence_threshold: float = 0.45,
    csv_sep: str = ",",
    csv_encoding: str = "utf-8",
    csv_header: Union[int, str, None] = "infer",
    top_k_codes: int = TOP_K_CODES,
    show_preview: bool = False,
    export_columns: Optional[Sequence[str]] = None,
) -> tuple[Path, pd.DataFrame]:
    """Match line items to community commodity codes using embedding-based similarity.

    This is the main function that performs commodity code matching. It:
    1. Loads line items (and optionally headers) from files or DataFrames
    2. Loads community codes from an Excel catalog
    3. Optionally loads supplier material groups for pre-filtering
    4. Matches vendors to suppliers and filters catalog by historical material groups
    5. Enriches codes with UNSPSC context descriptions if available
    6. Generates embeddings for line items and catalog entries
    7. Computes cosine similarity to find best-matching codes
    8. Optionally runs LLM verification for top matches
    9. Retries low-confidence matches without supplier filtering
    10. Outputs results to an Excel file with matched codes and similarity scores

    Args:
        line_items: DataFrame or path to line items file (CSV/Excel).
        community_codes_path: Path to Excel file containing community commodity codes.
        unspsc_context_path: Optional path to Excel file with UNSPSC context descriptions.
        supplier_groups_path: Optional path to CSV with supplier→material group mappings.
        output_path: Path where the output Excel file will be written.
        headers: Optional DataFrame or path to header data to merge with line items.
        product_structure_sheet_hint: Hint for finding the product structure sheet in Excel.
        product_structure_sheet_name: Exact name of the product structure sheet (optional).
        code_column_hint: Optional hint for the code column name in the catalog.
        embedding_model: Name of the embedding model to use.
        llm_verify: Whether to run LLM verification on top matches (default: False).
        llm_model: Name of the LLM model to use for verification (default: from env).
        llm_min_confidence: Minimum confidence threshold for LLM verification (default: 0.6).
        enable_supplier_filtering: Whether to enable supplier-based catalog pre-filtering (default: True).
        supplier_match_threshold: Fuzzy match threshold for vendor→supplier matching (0-100, default: 70).
        retry_confidence_threshold: Retry threshold for low confidence matches (default: 0.45).
        csv_sep: Separator for CSV files.
        csv_encoding: Encoding for CSV files.
        csv_header: Row number to use as header for CSV files.
        top_k_codes: Number of top matching codes to include per line item.
        show_preview: Whether to display a preview of results (requires Jupyter environment).
        export_columns: Optional ordered subset of columns to include in the Excel output.

    Returns:
        A tuple of (output_path, enriched_df) where output_path is the Path to the
        output Excel file and enriched_df is the DataFrame with matched codes.

    Raises:
        FileNotFoundError: If the community codes file is not found.
        ValueError: If no code column is detected in the catalog.
    """
    print("Embedding configuration:")
    print(f"   • Embedding model: {embedding_model}")
    print(f"   • Top-K: {top_k_codes}")
    if llm_verify:
        print(f"   • LLM verification model: {llm_model or LLM_MODEL_NAME} (min confidence {llm_min_confidence})")
    if enable_supplier_filtering and supplier_groups_path is not None:
        print(f"   • Supplier filtering: enabled (threshold {supplier_match_threshold}%, retry < {retry_confidence_threshold})")

    li = _coerce_df(line_items, csv_sep, csv_encoding, csv_header)
    print(f"📥 Line items loaded: {len(li)} rows, {len(li.columns)} cols")

    # Load supplier material groups and match vendors
    supplier_lookup: Dict[str, Dict[str, Any]] = {}
    if enable_supplier_filtering and supplier_groups_path is not None:
        try:
            supplier_source_label = (
                "in-memory reference table"
                if isinstance(supplier_groups_path, pd.DataFrame)
                else Path(supplier_groups_path).expanduser().name
            )
            print(f"📥 Loading supplier material groups from {supplier_source_label} ...")
            supplier_lookup = _load_supplier_material_groups(supplier_groups_path)
            print(f"   • Loaded {len(supplier_lookup)} unique suppliers")
        except FileNotFoundError:
            warnings.warn(f"Supplier groups file not found: {supplier_groups_path}. Supplier filtering disabled.")
            enable_supplier_filtering = False
        except ValueError as exc:
            warnings.warn(f"{exc}. Supplier filtering disabled.")
            enable_supplier_filtering = False

    if headers is not None:
        hd = _coerce_df(headers, csv_sep, csv_encoding, csv_header)
        join_keys = _find_join_keys(li, hd)
        if not join_keys:
            warnings.warn("No shared keys found between line items and headers; skipping merge.")
        else:
            left_on, right_on = join_keys[0]
            print(f"🔗 Joining on LineItems.{left_on} ⇐ Headers.{right_on}")
            li = li.merge(hd, how="left", left_on=left_on, right_on=right_on, suffixes=("", "_hdr"))
            print(f"   • After merge: {len(li)} rows, {len(li.columns)} cols")

    print("🧱 Building text representations ...")
    desc_col = _pick_description_column(li)
    if not desc_col:
        warnings.warn("No description-like column found; falling back to concatenated text.")
    li = li.copy()
    li["_full_text"] = li.apply(_concat_kv_text, axis=1)
    # Gather any columns that store semantic usage summaries (including item_usageSummary).
    usage_columns = [col for col in li.columns if re.search(r"usage\s*summary", str(col), re.IGNORECASE)]

    def _build_desc_text(row: pd.Series) -> str:
        """Combine the short description with usage summaries for richer embeddings."""
        parts: List[str] = []
        if desc_col and desc_col in row.index:
            value = row.get(desc_col)
            if pd.notna(value):
                text = str(value).strip()
                if text:
                    parts.append(text)
        for column in usage_columns:
            value = row.get(column)
            if pd.notna(value):
                text = str(value).strip()
                if text and text not in parts:
                    parts.append(text)
        if parts:
            return " | ".join(parts)
        fallback = row.get("_full_text", "")
        return str(fallback) if pd.notna(fallback) else ""

    li["_desc_text"] = li.apply(_build_desc_text, axis=1)

    print("📥 Loading community codes ...")
    catalog = _load_catalog_df(
        community_codes_path,
        product_structure_sheet_name=product_structure_sheet_name,
        product_structure_sheet_hint=product_structure_sheet_hint,
    )
    print(f"   • Catalog loaded: {len(catalog)} rows, {len(catalog.columns)} cols")

    code_col = _detect_code_column(catalog, code_column_hint)
    if code_col is None:
        raise ValueError(f"No code-like column detected. Columns: {list(catalog.columns)}")

    context_map = _load_unspsc_context_map(unspsc_context_path)
    if context_map:
        normalized_code_series = catalog[code_col].astype(str).str.strip().str.upper()
        context_series = normalized_code_series.map(context_map)
        catalog[UNSPSC_CONTEXT_COLUMN] = context_series
        matched = int(context_series.notna().sum())
        if isinstance(unspsc_context_path, pd.DataFrame):
            context_source = "in-memory reference table"
        elif unspsc_context_path:
            context_source = Path(unspsc_context_path).name
        else:
            context_source = "context data"
        print(f"   • UNSPSC context merged for {matched} codes from {context_source}")
    elif unspsc_context_path is not None:
        print("   • UNSPSC context file provided but no matches were added.")

    taxonomy_fields = list(TAXONOMY_FIELDS) + [UNSPSC_CONTEXT_COLUMN]
    taxonomy_map: Dict[str, Optional[str]] = {
        logical: logical if logical in catalog.columns else None for logical in taxonomy_fields
    }

    # Match vendors to suppliers and add metadata columns
    li["Original_Vendor_Name"] = None
    li["Business_Partner_ID"] = None
    li["Supplier_Match_Score"] = None
    li["Supplier_Match_Method"] = None
    vendor_matches: Dict[int, Optional[Dict[str, Any]]] = {}

    if enable_supplier_filtering and supplier_lookup:
        # Check if Vendor column exists (from vendor annotation)
        vendor_col = None
        for col_candidate in ["Vendor", "vendor", "vendorName", "vendor_name"]:
            if col_candidate in li.columns:
                vendor_col = col_candidate
                break

        if vendor_col:
            print(f"🔍 Matching vendors to suppliers (using column '{vendor_col}') ...")
            matched_count = 0
            match_method_counts = {"exact": 0, "ratio": 0, "token_sort": 0}

            # CRITICAL: Capture current vendor values BEFORE any replacements
            # This preserves the pre-match vendor names for Original_Vendor_Name column
            pre_match_vendors = li[vendor_col].copy()

            for idx, row in li.iterrows():
                vendor_name = row.get(vendor_col)
                supplier_match = _match_vendor_to_supplier(
                    vendor_name,
                    supplier_lookup,
                    threshold=supplier_match_threshold
                )
                vendor_matches[idx] = supplier_match

                if supplier_match:
                    matched_count += 1
                    match_method = supplier_match.get("match_method", "unknown")
                    if match_method in match_method_counts:
                        match_method_counts[match_method] += 1

                    # Store original vendor name from pre-match capture
                    li.at[idx, "Original_Vendor_Name"] = pre_match_vendors.loc[idx]
                    # Replace vendor name with canonical supplier name
                    li.at[idx, vendor_col] = supplier_match["matched_supplier_name"]
                    # Add Business Partner ID
                    li.at[idx, "Business_Partner_ID"] = supplier_match["bpid"]
                    # Add match score
                    li.at[idx, "Supplier_Match_Score"] = supplier_match["match_score"]
                    # Add match method
                    li.at[idx, "Supplier_Match_Method"] = match_method

            print(f"   • Matched {matched_count}/{len(li)} line items to suppliers")
            if matched_count > 0:
                methods_summary = ", ".join([f"{method}: {count}" for method, count in match_method_counts.items() if count > 0])
                print(f"   • Match methods: {methods_summary}")
        else:
            warnings.warn("No Vendor column found in line items. Supplier filtering will be disabled.")
            enable_supplier_filtering = False

    print("🧠 Computing embeddings ...")
    catalog["_cc_text"] = catalog.apply(_concat_kv_text, axis=1)
    embedder = _ProxyEmbeddings(embedding_model)
    li_full_vecs = embedder.embed(li["_full_text"].tolist())
    li_desc_vecs = embedder.embed(li["_desc_text"].tolist())
    cc_vecs = embedder.embed(catalog["_cc_text"].tolist())

    print("Normalizing vectors ...")
    li_full_vecs = _normalize_matrix(li_full_vecs)
    li_desc_vecs = _normalize_matrix(li_desc_vecs)
    cc_vecs = _normalize_matrix(cc_vecs)

    print("Computing cosine similarities ...")
    sim_full = _cosine_similarity(li_full_vecs, cc_vecs)
    sim_desc = _cosine_similarity(li_desc_vecs, cc_vecs)

    # Apply supplier-based filtering by masking similarity scores
    filtered_items_mask = np.zeros(len(li), dtype=bool)
    if enable_supplier_filtering and vendor_matches:
        print("🔍 Applying supplier-based catalog filtering ...")
        code_values_array = catalog[code_col].astype(str).values
        filtered_count = 0

        for idx, supplier_match in vendor_matches.items():
            if supplier_match and supplier_match["material_groups"]:
                # Create mask for catalog entries matching this supplier's material groups
                catalog_mask = np.isin(code_values_array, supplier_match["material_groups"])

                if catalog_mask.any():
                    # Mask out non-matching catalog entries by setting similarity to -2
                    # (lower than minimum cosine similarity of -1)
                    sim_desc[idx, ~catalog_mask] = -2.0
                    filtered_items_mask[idx] = True
                    filtered_count += 1

        if filtered_count > 0:
            print(f"   • Applied filtering to {filtered_count} line items based on supplier material groups")

    top_full_idx = np.argmax(sim_full, axis=1)
    top_desc_idx = np.argmax(sim_desc, axis=1)
    top_full_sim = sim_full[np.arange(sim_full.shape[0]), top_full_idx]
    top_desc_sim = sim_desc[np.arange(sim_desc.shape[0]), top_desc_idx]

    k = max(1, int(top_k_codes))
    topk_full_idx = _topk_indices(sim_full, k)
    topk_desc_idx = _topk_indices(sim_desc, k)
    code_values = catalog[code_col].astype(str).tolist()

    if tqdm is not None:
        iterator = tqdm(range(sim_full.shape[0]), desc="Assembling TOP-K codes", unit="row")
    else:
        iterator = range(sim_full.shape[0])
    codes_full_topk = [",".join(code_values[j] for j in topk_full_idx[i]) for i in iterator]
    codes_desc_topk = [",".join(code_values[j] for j in topk_desc_idx[i]) for i in range(sim_desc.shape[0])]

    print("🧱 Assembling final dataframe ...")
    output_df = li.copy()

    # Diagnostic: Check for missing critical line item fields
    expected_line_item_fields = ["description", "netAmount", "quantity", "unitPrice", "materialNumber", "itemNumber", "usageSummary"]
    missing_fields = [f for f in expected_line_item_fields if f not in output_df.columns]
    if missing_fields:
        warnings.warn(
            f"Line items DataFrame is missing expected fields: {missing_fields}. "
            f"These fields may not have been extracted from the PDFs. "
            f"Available columns: {list(output_df.columns)}"
        )

    # Diagnostic: Check for rows with all NaN values in critical fields
    present_critical_fields = [f for f in expected_line_item_fields if f in output_df.columns]
    if present_critical_fields:
        all_nan_mask = output_df[present_critical_fields].isna().all(axis=1)
        if all_nan_mask.any():
            nan_count = int(all_nan_mask.sum())
            affected_files = []
            if "file" in output_df.columns:
                affected_files = output_df.loc[all_nan_mask, "file"].unique().tolist()
            warnings.warn(
                f"Found {nan_count} rows with ALL critical fields ({present_critical_fields}) empty. "
                f"Affected files: {affected_files if affected_files else 'unknown (no file column)'}. "
                f"This may indicate incomplete PDF extraction."
            )

    output_df["CosSim_Full"] = top_full_sim
    output_df["CosSim_Desc"] = top_desc_sim
    output_df["Code_Full"] = [code_values[idx] for idx in top_full_idx]
    output_df["Code_Desc"] = [code_values[idx] for idx in top_desc_idx]
    output_df["Codes_Full_Top5"] = codes_full_topk
    output_df["Codes_Desc_Top5"] = codes_desc_topk

    def _catalog_value(index: int, logical_name: str) -> Any:
        column = taxonomy_map.get(logical_name)
        if column is None or column not in catalog.columns:
            return None
        return catalog.iloc[index][column]

    for logical_name in [DESCRIPTION_LOW_LEVEL_COLUMN, DESCRIPTION_HIGH_LEVEL_COLUMN, PROCUREMENT_GROUP_COLUMN, UNSPSC_CONTEXT_COLUMN]:
        output_df[f"{logical_name}_Full"] = [_catalog_value(idx, logical_name) for idx in top_full_idx]
        output_df[f"{logical_name}_Desc"] = [_catalog_value(idx, logical_name) for idx in top_desc_idx]

    output_df["MatchSource_Full"] = "Full"
    output_df["MatchSource_Desc"] = "Desc"

    if llm_verify:
        _run_llm_verification(
            output_df,
            catalog=catalog,
            code_column=code_col,
            llm_model=llm_model or LLM_MODEL_NAME,
            min_confidence=llm_min_confidence,
        )

        # Retry low-confidence filtered items without supplier filtering
        if enable_supplier_filtering and "LLM_Confidence_Desc" in output_df.columns:
            retry_mask = (
                filtered_items_mask
                & (output_df["LLM_Confidence_Desc"].fillna(0.0) < retry_confidence_threshold)
            )
            retry_count = int(retry_mask.sum())

            if retry_count > 0:
                print(f"🔄 Retrying {retry_count} low-confidence filtered items without supplier filtering ...")

                # Re-compute DESC similarity for retry items using full catalog
                retry_indices = np.where(retry_mask)[0]
                retry_desc_vecs = li_desc_vecs[retry_indices]

                # Compute similarity against full catalog (no filtering)
                retry_sim_desc = _cosine_similarity(retry_desc_vecs, cc_vecs)

                # Find top matches
                retry_top_desc_idx = np.argmax(retry_sim_desc, axis=1)
                retry_top_desc_sim = retry_sim_desc[np.arange(len(retry_indices)), retry_top_desc_idx]
                retry_topk_desc_idx = _topk_indices(retry_sim_desc, k)

                # Update output DataFrame for retry items
                for i, orig_idx in enumerate(retry_indices):
                    output_df.at[orig_idx, "CosSim_Desc"] = retry_top_desc_sim[i]
                    output_df.at[orig_idx, "Code_Desc"] = code_values[retry_top_desc_idx[i]]
                    output_df.at[orig_idx, "Codes_Desc_Top5"] = ",".join(
                        code_values[j] for j in retry_topk_desc_idx[i]
                    )

                    # Update taxonomy fields
                    for logical_name in [DESCRIPTION_LOW_LEVEL_COLUMN, DESCRIPTION_HIGH_LEVEL_COLUMN, PROCUREMENT_GROUP_COLUMN, UNSPSC_CONTEXT_COLUMN]:
                        col_name = f"{logical_name}_Desc"
                        if col_name in output_df.columns:
                            output_df.at[orig_idx, col_name] = _catalog_value(retry_top_desc_idx[i], logical_name)

                    # Mark as retried
                    output_df.at[orig_idx, "MatchSource_Desc"] = "Desc_Unfiltered"

                # Re-run LLM verification for retry items
                print("   • Re-running LLM verification for retried items ...")
                retry_df = output_df.loc[retry_mask].copy()
                _run_llm_verification(
                    retry_df,
                    catalog=catalog,
                    code_column=code_col,
                    llm_model=llm_model or LLM_MODEL_NAME,
                    min_confidence=llm_min_confidence,
                )

                # Update main DataFrame with retry results
                llm_cols = ["LLM_Confidence_Desc", "LLM_Suggestion_Desc", "LLM_Reason_Desc", "Block_By_LLM_Desc"]
                for col in llm_cols:
                    if col in retry_df.columns:
                        output_df.loc[retry_mask, col] = retry_df[col]

                print(f"   • Retry complete. Avg confidence improved for retried items.")

    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing results to {output_path}")
    export_df = output_df
    if export_columns:
        selected_cols = [col for col in export_columns if col in output_df.columns]
        missing_export_cols = [col for col in export_columns if col not in output_df.columns]

        if missing_export_cols:
            warnings.warn(
                f"Requested export columns not found in output DataFrame: {missing_export_cols}. "
                f"These columns will be omitted from the export. "
                f"This may indicate missing fields from PDF extraction or processing."
            )

        if selected_cols:
            export_df = output_df[selected_cols].copy()
            print(f"   • Exporting {len(selected_cols)}/{len(export_columns)} requested columns")
        else:
            warnings.warn("No requested export columns were found; writing full output.")

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        export_df.to_excel(writer, index=False, sheet_name="LineItems+Codes")

    print("Commodity code matching complete.")
    return output_path, export_df


__all__ = [
    "run_community_code_matching",
]
