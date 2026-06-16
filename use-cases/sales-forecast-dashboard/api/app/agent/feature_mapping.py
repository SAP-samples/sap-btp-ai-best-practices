"""
Feature mapping and metadata for the forecasting agent.

Provides:
1. Natural language aliases for Model A features
2. Feature metadata (bounds, categories, descriptions)
3. Helper functions to resolve feature names and parse modifications

This enables users to refer to features using natural language like
"white glove" instead of "pct_white_glove_roll_mean_4".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class FeatureMetadata:
    """
    Metadata for a single Model A feature.

    Attributes
    ----------
    name : str
        Technical feature name (e.g., "pct_white_glove_roll_mean_4")
    category : str
        Category for grouping (financing, staffing, product_mix, etc.)
    description : str
        Human-readable description
    min_value : float
        Minimum valid value
    max_value : float
        Maximum valid value
    default_value : float
        Default/typical value
    is_bm_only : bool
        True if feature only applies to B&M channel
    value_type : str
        Type of value: "percentage", "count", "distance", "boolean"
    aliases : List[str]
        Natural language aliases for this feature
    """

    name: str
    category: str
    description: str
    min_value: float
    max_value: float
    default_value: float
    is_bm_only: bool = False
    value_type: str = "percentage"
    aliases: List[str] = None

    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []


# -----------------------------------------------------------------------------
# Feature Definitions
# Organized by category matching Agent_plan.md
# -----------------------------------------------------------------------------

FEATURE_METADATA: Dict[str, FeatureMetadata] = {
    # -------------------------------------------------------------------------
    # Financing Levers (3)
    # -------------------------------------------------------------------------
    "pct_primary_financing_roll_mean_4": FeatureMetadata(
        name="pct_primary_financing_roll_mean_4",
        category="financing",
        description="4-week rolling mean of primary financing rate (0-60 month options)",
        min_value=0.0,
        max_value=100.0,
        default_value=35.0,
        value_type="percentage",
        aliases=[
            "primary financing",
            "financing",
            "primary finance",
            "finance rate",
            "0-60 financing",
        ],
    ),
    "pct_secondary_financing_roll_mean_4": FeatureMetadata(
        name="pct_secondary_financing_roll_mean_4",
        category="financing",
        description="4-week rolling mean of secondary financing rate (promotional/deferred)",
        min_value=0.0,
        max_value=100.0,
        default_value=20.0,
        value_type="percentage",
        aliases=[
            "secondary financing",
            "promotional financing",
            "deferred financing",
        ],
    ),
    "pct_tertiary_financing_roll_mean_4": FeatureMetadata(
        name="pct_tertiary_financing_roll_mean_4",
        category="financing",
        description="4-week rolling mean of tertiary financing rate (lease-to-own)",
        min_value=0.0,
        max_value=100.0,
        default_value=5.0,
        value_type="percentage",
        aliases=[
            "tertiary financing",
            "lease to own",
            "lto financing",
        ],
    ),
    # -------------------------------------------------------------------------
    # Staffing Levers (2) - B&M Only
    # -------------------------------------------------------------------------
    "staffing_unique_associates_roll_mean_4": FeatureMetadata(
        name="staffing_unique_associates_roll_mean_4",
        category="staffing",
        description="4-week rolling mean of unique associates working in store",
        min_value=0.0,
        max_value=50.0,
        default_value=12.0,
        is_bm_only=True,
        value_type="count",
        aliases=[
            "unique associates",
            "associates",
            "staff count",
            "headcount",
            "employees",
        ],
    ),
    "staffing_hours_roll_mean_4": FeatureMetadata(
        name="staffing_hours_roll_mean_4",
        category="staffing",
        description="4-week rolling mean of total staffing hours",
        min_value=0.0,
        max_value=2000.0,
        default_value=400.0,
        is_bm_only=True,
        value_type="count",
        aliases=[
            "staffing hours",
            "labor hours",
            "hours",
            "staff hours",
        ],
    ),
    # -------------------------------------------------------------------------
    # Product Mix Levers (4)
    # -------------------------------------------------------------------------
    "pct_omni_channel_roll_mean_4": FeatureMetadata(
        name="pct_omni_channel_roll_mean_4",
        category="product_mix",
        description="4-week rolling mean of omni-channel order percentage",
        min_value=0.0,
        max_value=100.0,
        default_value=40.0,
        value_type="percentage",
        aliases=[
            "omni channel",
            "omnichannel",
            "omni",
            "bopis",
            "buy online pickup in store",
        ],
    ),
    "pct_value_product_roll_mean_4": FeatureMetadata(
        name="pct_value_product_roll_mean_4",
        category="product_mix",
        description="4-week rolling mean of value-tier product percentage",
        min_value=0.0,
        max_value=100.0,
        default_value=30.0,
        value_type="percentage",
        aliases=[
            "value product",
            "value tier",
            "value mix",
            "budget product",
        ],
    ),
    "pct_premium_product_roll_mean_4": FeatureMetadata(
        name="pct_premium_product_roll_mean_4",
        category="product_mix",
        description="4-week rolling mean of premium-tier product percentage",
        min_value=0.0,
        max_value=100.0,
        default_value=25.0,
        value_type="percentage",
        aliases=[
            "premium product",
            "premium tier",
            "premium mix",
            "high-end product",
        ],
    ),
    "pct_white_glove_roll_mean_4": FeatureMetadata(
        name="pct_white_glove_roll_mean_4",
        category="product_mix",
        description="4-week rolling mean of white glove delivery percentage",
        min_value=0.0,
        max_value=100.0,
        default_value=15.0,
        value_type="percentage",
        aliases=[
            "white glove",
            "white glove delivery",
            "premium delivery",
            "full service delivery",
        ],
    ),
    # -------------------------------------------------------------------------
    # Market Signals / Awareness (2)
    # -------------------------------------------------------------------------
    "brand_awareness_dma_roll_mean_4": FeatureMetadata(
        name="brand_awareness_dma_roll_mean_4",
        category="awareness",
        description="4-week rolling mean of brand awareness score in DMA (0-100)",
        min_value=0.0,
        max_value=100.0,
        default_value=50.0,
        value_type="percentage",
        aliases=[
            "brand awareness",
            "awareness",
            "brand recognition",
            "market awareness",
        ],
    ),
    "brand_consideration_dma_roll_mean_4": FeatureMetadata(
        name="brand_consideration_dma_roll_mean_4",
        category="awareness",
        description="4-week rolling mean of brand consideration score in DMA (0-100)",
        min_value=0.0,
        max_value=100.0,
        default_value=40.0,
        value_type="percentage",
        aliases=[
            "brand consideration",
            "consideration",
            "purchase intent",
        ],
    ),
    # -------------------------------------------------------------------------
    # Cannibalization (4) - B&M Only
    # -------------------------------------------------------------------------
    "cannibalization_pressure": FeatureMetadata(
        name="cannibalization_pressure",
        category="cannibalization",
        description="Aggregate cannibalization pressure from nearby new store openings",
        min_value=0.0,
        max_value=10.0,
        default_value=0.0,
        is_bm_only=True,
        value_type="score",
        aliases=[
            "cannibalization",
            "cannibalization pressure",
            "cannibal pressure",
            "network pressure",
        ],
    ),
    "min_dist_new_store_km": FeatureMetadata(
        name="min_dist_new_store_km",
        category="cannibalization",
        description="Distance to nearest new store opened in last 52 weeks (km)",
        min_value=0.0,
        max_value=1000.0,
        default_value=100.0,
        is_bm_only=True,
        value_type="distance",
        aliases=[
            "nearest new store",
            "min distance new store",
            "closest new store",
        ],
    ),
    "num_new_stores_within_10mi_last_52wk": FeatureMetadata(
        name="num_new_stores_within_10mi_last_52wk",
        category="cannibalization",
        description="Count of new stores opened within 10 miles in last 52 weeks",
        min_value=0.0,
        max_value=10.0,
        default_value=0.0,
        is_bm_only=True,
        value_type="count",
        aliases=[
            "new stores 10mi",
            "new stores within 10 miles",
        ],
    ),
    "num_new_stores_within_20mi_last_52wk": FeatureMetadata(
        name="num_new_stores_within_20mi_last_52wk",
        category="cannibalization",
        description="Count of new stores opened within 20 miles in last 52 weeks",
        min_value=0.0,
        max_value=20.0,
        default_value=0.0,
        is_bm_only=True,
        value_type="count",
        aliases=[
            "new stores 20mi",
            "new stores within 20 miles",
        ],
    ),
    # -------------------------------------------------------------------------
    # Store DNA (4)
    # -------------------------------------------------------------------------
    "weeks_since_open": FeatureMetadata(
        name="weeks_since_open",
        category="store_dna",
        description="Weeks since store opening (uncapped)",
        min_value=0.0,
        max_value=2000.0,
        default_value=200.0,
        value_type="count",
        aliases=[
            "weeks since open",
            "store age",
            "weeks open",
            "age",
        ],
    ),
    "weeks_since_open_capped_13": FeatureMetadata(
        name="weeks_since_open_capped_13",
        category="store_dna",
        description="Weeks since store opening, capped at 13 (new store ramp)",
        min_value=0.0,
        max_value=13.0,
        default_value=13.0,
        value_type="count",
        aliases=[
            "weeks since open capped 13",
            "new store ramp 13",
        ],
    ),
    "weeks_since_open_capped_52": FeatureMetadata(
        name="weeks_since_open_capped_52",
        category="store_dna",
        description="Weeks since store opening, capped at 52 (maturation)",
        min_value=0.0,
        max_value=52.0,
        default_value=52.0,
        value_type="count",
        aliases=[
            "weeks since open capped 52",
            "new store maturation",
        ],
    ),
    "merchandising_sf": FeatureMetadata(
        name="merchandising_sf",
        category="store_dna",
        description="Merchandising square footage",
        min_value=0.0,
        max_value=100000.0,
        default_value=25000.0,
        is_bm_only=True,
        value_type="count",
        aliases=[
            "merchandising sf",
            "square footage",
            "store size",
            "sqft",
        ],
    ),
    # -------------------------------------------------------------------------
    # Categorical Store DNA (3) - Boolean flags
    # -------------------------------------------------------------------------
    "is_outlet": FeatureMetadata(
        name="is_outlet",
        category="store_dna",
        description="Whether store is an outlet location",
        min_value=0.0,
        max_value=1.0,
        default_value=0.0,
        value_type="boolean",
        aliases=[
            "outlet",
            "is outlet",
            "outlet store",
        ],
    ),
    "is_comp_store": FeatureMetadata(
        name="is_comp_store",
        category="store_dna",
        description="Whether store is a comparable store (>52 weeks old)",
        min_value=0.0,
        max_value=1.0,
        default_value=1.0,
        value_type="boolean",
        aliases=[
            "comp store",
            "is comp store",
            "comparable store",
        ],
    ),
    "is_new_store": FeatureMetadata(
        name="is_new_store",
        category="store_dna",
        description="Whether store opened in last 52 weeks",
        min_value=0.0,
        max_value=1.0,
        default_value=0.0,
        value_type="boolean",
        aliases=[
            "new store",
            "is new store",
            "new location",
        ],
    ),
}


# Human-readable display names for reports and user-facing output
FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    "pct_primary_financing_roll_mean_4": "Primary Financing %",
    "pct_secondary_financing_roll_mean_4": "Secondary Financing %",
    "pct_tertiary_financing_roll_mean_4": "Tertiary Financing %",
    "staffing_unique_associates_roll_mean_4": "Unique Associates",
    "staffing_hours_roll_mean_4": "Staffing Hours",
    "pct_omni_channel_roll_mean_4": "Omni-Channel %",
    "pct_value_product_roll_mean_4": "Value Product %",
    "pct_premium_product_roll_mean_4": "Premium Product %",
    "pct_white_glove_roll_mean_4": "White Glove %",
    "brand_awareness_dma_roll_mean_4": "Brand Awareness",
    "brand_consideration_dma_roll_mean_4": "Brand Consideration",
    "cannibalization_pressure": "Cannibalization Pressure",
    "min_dist_new_store_km": "Distance to Nearest New Store",
    "num_new_stores_within_10mi_last_52wk": "New Stores (10mi)",
    "num_new_stores_within_20mi_last_52wk": "New Stores (20mi)",
    "weeks_since_open": "Store Age (Weeks)",
    "weeks_since_open_capped_13": "Store Age (Capped 13w)",
    "weeks_since_open_capped_52": "Store Age (Capped 52w)",
    "merchandising_sf": "Store Size (sq ft)",
    "is_outlet": "Outlet Store",
    "is_comp_store": "Comp Store",
    "is_new_store": "New Store",
}


def get_display_name(feature_name: str) -> str:
    """
    Get human-readable display name for a feature.

    Parameters
    ----------
    feature_name : str
        Technical feature name (e.g., "staffing_hours_roll_mean_4")

    Returns
    -------
    str
        Human-readable display name (e.g., "Staffing Hours"),
        or the original name if no mapping exists
    """
    return FEATURE_DISPLAY_NAMES.get(feature_name, feature_name)


def get_all_display_names() -> Dict[str, str]:
    """
    Get all feature display name mappings.

    Useful for including in LLM prompts to ensure consistent naming.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping technical names to display names
    """
    return FEATURE_DISPLAY_NAMES.copy()


# Build reverse lookup from aliases to feature names
_ALIAS_TO_FEATURE: Dict[str, str] = {}
for feature_name, metadata in FEATURE_METADATA.items():
    for alias in metadata.aliases:
        _ALIAS_TO_FEATURE[alias.lower()] = feature_name
    # Also add the feature name itself
    _ALIAS_TO_FEATURE[feature_name.lower()] = feature_name


# Feature categories for grouping
FEATURE_CATEGORIES = {
    "financing": [
        "pct_primary_financing_roll_mean_4",
        "pct_secondary_financing_roll_mean_4",
        "pct_tertiary_financing_roll_mean_4",
    ],
    "staffing": [
        "staffing_unique_associates_roll_mean_4",
        "staffing_hours_roll_mean_4",
    ],
    "product_mix": [
        "pct_omni_channel_roll_mean_4",
        "pct_value_product_roll_mean_4",
        "pct_premium_product_roll_mean_4",
        "pct_white_glove_roll_mean_4",
    ],
    "awareness": [
        "brand_awareness_dma_roll_mean_4",
        "brand_consideration_dma_roll_mean_4",
    ],
    "cannibalization": [
        "cannibalization_pressure",
        "min_dist_new_store_km",
        "num_new_stores_within_10mi_last_52wk",
        "num_new_stores_within_20mi_last_52wk",
    ],
    "store_dna": [
        "weeks_since_open",
        "weeks_since_open_capped_13",
        "weeks_since_open_capped_52",
        "merchandising_sf",
        "is_outlet",
        "is_comp_store",
        "is_new_store",
    ],
}


# -----------------------------------------------------------------------------
# Feature Resolution Functions
# -----------------------------------------------------------------------------


def resolve_feature_name(user_input: str) -> Optional[str]:
    """
    Resolve a user input to a canonical feature name.

    Supports:
    - Exact feature names
    - Natural language aliases
    - Case-insensitive matching
    - Fuzzy matching on common variations

    Parameters
    ----------
    user_input : str
        User-provided feature name or alias

    Returns
    -------
    Optional[str]
        Canonical feature name, or None if not found

    Examples
    --------
    >>> resolve_feature_name("white glove")
    'pct_white_glove_roll_mean_4'
    >>> resolve_feature_name("pct_primary_financing_roll_mean_4")
    'pct_primary_financing_roll_mean_4'
    >>> resolve_feature_name("brand awareness")
    'brand_awareness_dma_roll_mean_4'
    """
    normalized = user_input.strip().lower()

    # Direct lookup
    if normalized in _ALIAS_TO_FEATURE:
        return _ALIAS_TO_FEATURE[normalized]

    # Try with underscores replaced by spaces
    normalized_spaces = normalized.replace("_", " ")
    if normalized_spaces in _ALIAS_TO_FEATURE:
        return _ALIAS_TO_FEATURE[normalized_spaces]

    # Try partial match on aliases
    for alias, feature in _ALIAS_TO_FEATURE.items():
        if normalized in alias or alias in normalized:
            return feature

    return None


def get_feature_metadata(feature_name: str) -> Optional[FeatureMetadata]:
    """
    Get metadata for a feature.

    Parameters
    ----------
    feature_name : str
        Feature name (canonical or alias)

    Returns
    -------
    Optional[FeatureMetadata]
        Feature metadata, or None if not found
    """
    resolved = resolve_feature_name(feature_name)
    if resolved:
        return FEATURE_METADATA.get(resolved)
    return None


def get_features_by_category(category: str) -> List[str]:
    """
    Get all feature names in a category.

    Parameters
    ----------
    category : str
        Category name (financing, staffing, product_mix, awareness, cannibalization, store_dna)

    Returns
    -------
    List[str]
        List of feature names in the category
    """
    return FEATURE_CATEGORIES.get(category.lower(), [])


def get_modifiable_features(channel: str = "B&M") -> List[str]:
    """
    Get list of features that can be modified for a channel.

    Parameters
    ----------
    channel : str
        Channel: "B&M" or "WEB"

    Returns
    -------
    List[str]
        List of modifiable feature names
    """
    features = []
    for name, meta in FEATURE_METADATA.items():
        if channel == "WEB" and meta.is_bm_only:
            continue
        features.append(name)
    return features


def is_bm_only_feature(feature_name: str) -> bool:
    """
    Check if a feature is B&M-only.

    Parameters
    ----------
    feature_name : str
        Feature name

    Returns
    -------
    bool
        True if B&M-only, False otherwise
    """
    meta = get_feature_metadata(feature_name)
    return meta.is_bm_only if meta else False


# -----------------------------------------------------------------------------
# Modification Parsing
# -----------------------------------------------------------------------------


@dataclass
class ParsedModification:
    """
    Parsed modification instruction.

    Attributes
    ----------
    operation : str
        Operation type: "set", "increase", "decrease", "multiply"
    value : float
        Numeric value for the operation
    is_percentage : bool
        Whether value is a percentage (for increase/decrease)
    """

    operation: str
    value: float
    is_percentage: bool = False


def parse_modification(modification_str: str) -> ParsedModification:
    """
    Parse a modification string into operation and value.

    Supports formats:
    - "set to 0.5" / "set to 50%"
    - "increase by 10%" / "increase by 5"
    - "decrease by 20%"
    - "multiply by 1.5"
    - "+10%" / "-5%"
    - "50" (implicit set)

    Parameters
    ----------
    modification_str : str
        User-provided modification instruction

    Returns
    -------
    ParsedModification
        Parsed operation and value

    Raises
    ------
    ValueError
        If modification string cannot be parsed

    Examples
    --------
    >>> parse_modification("set to 0.5")
    ParsedModification(operation='set', value=0.5, is_percentage=False)
    >>> parse_modification("increase by 10%")
    ParsedModification(operation='increase', value=10.0, is_percentage=True)
    >>> parse_modification("+20%")
    ParsedModification(operation='increase', value=20.0, is_percentage=True)
    """
    text = modification_str.strip().lower()

    # Pattern: "set to X" or "set X"
    match = re.match(r"set\s+(?:to\s+)?(-?[\d.]+)\s*(%)?", text)
    if match:
        value = float(match.group(1))
        is_pct = match.group(2) == "%"
        return ParsedModification(operation="set", value=value, is_percentage=is_pct)

    # Pattern: "increase by X%" or "increase by X"
    match = re.match(r"increase\s+(?:by\s+)?(-?[\d.]+)\s*(%)?", text)
    if match:
        value = float(match.group(1))
        is_pct = match.group(2) == "%"
        return ParsedModification(operation="increase", value=value, is_percentage=is_pct)

    # Pattern: "decrease by X%" or "decrease by X"
    match = re.match(r"decrease\s+(?:by\s+)?(-?[\d.]+)\s*(%)?", text)
    if match:
        value = float(match.group(1))
        is_pct = match.group(2) == "%"
        return ParsedModification(operation="decrease", value=value, is_percentage=is_pct)

    # Pattern: "multiply by X"
    match = re.match(r"multiply\s+(?:by\s+)?(-?[\d.]+)", text)
    if match:
        value = float(match.group(1))
        return ParsedModification(operation="multiply", value=value, is_percentage=False)

    # Pattern: "+X%" or "-X%"
    match = re.match(r"([+-])?\s*([\d.]+)\s*(%)", text)
    if match:
        sign = match.group(1) or "+"
        value = float(match.group(2))
        if sign == "-":
            return ParsedModification(operation="decrease", value=value, is_percentage=True)
        return ParsedModification(operation="increase", value=value, is_percentage=True)

    # Pattern: just a number (implicit set)
    match = re.match(r"(-?[\d.]+)\s*(%)?$", text)
    if match:
        value = float(match.group(1))
        is_pct = match.group(2) == "%"
        return ParsedModification(operation="set", value=value, is_percentage=is_pct)

    raise ValueError(
        f"Cannot parse modification: '{modification_str}'. "
        "Expected format: 'set to X', 'increase by X%', 'decrease by X', '+X%', etc."
    )


def apply_modification(
    current_value: float,
    modification: ParsedModification,
    metadata: Optional[FeatureMetadata] = None,
) -> Tuple[float, Optional[str]]:
    """
    Apply a parsed modification to a current value with clamping.

    Parameters
    ----------
    current_value : float
        Current value of the feature
    modification : ParsedModification
        Parsed modification to apply
    metadata : Optional[FeatureMetadata]
        Feature metadata for bounds clamping

    Returns
    -------
    Tuple[float, Optional[str]]
        Tuple of (new_value, warning_message).
        warning_message is None if no clamping occurred.
    """
    op = modification.operation
    val = modification.value

    if op == "set":
        new_value = val
    elif op == "increase":
        if modification.is_percentage:
            new_value = current_value * (1 + val / 100)
        else:
            new_value = current_value + val
    elif op == "decrease":
        if modification.is_percentage:
            new_value = current_value * (1 - val / 100)
        else:
            new_value = current_value - val
    elif op == "multiply":
        new_value = current_value * val
    else:
        raise ValueError(f"Unknown operation: {op}")

    # Bounds clamping (instead of raising errors)
    warning = None
    if metadata:
        if new_value < metadata.min_value:
            warning = (
                f"Value clamped from {new_value:.2f} to minimum {metadata.min_value} "
                f"for feature '{metadata.name}'"
            )
            new_value = metadata.min_value
        elif new_value > metadata.max_value:
            warning = (
                f"Value clamped from {new_value:.2f} to maximum {metadata.max_value} "
                f"for feature '{metadata.name}'"
            )
            new_value = metadata.max_value

    return new_value, warning


__all__ = [
    "FeatureMetadata",
    "FEATURE_METADATA",
    "FEATURE_CATEGORIES",
    "FEATURE_DISPLAY_NAMES",
    "resolve_feature_name",
    "get_feature_metadata",
    "get_features_by_category",
    "get_modifiable_features",
    "is_bm_only_feature",
    "get_display_name",
    "get_all_display_names",
    "ParsedModification",
    "parse_modification",
    "apply_modification",
]
