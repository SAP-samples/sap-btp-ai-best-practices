"""
Hierarchy Service — builds and caches the SAP organizational hierarchy.

Hierarchy (authoritative from SAP master data):
  Segment
    -> Profit Centers  (v_profitcenter.Segment)
      -> Cost Centers  (enriched dataset: PC <-> CC transactional relationship)
                       (v_costcenter.Segment as secondary confirmation)

Usage:
  hierarchy = HierarchyService()
  hierarchy.build(pc_table, cc_table, enriched_df)

  pcs = hierarchy.get_profit_centers_for_segment("Corporate")
  ccs = hierarchy.get_cost_centers_for_segment("Corporate")
"""

import logging
from collections import defaultdict
from typing import Dict, FrozenSet, List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)

_PC_COL     = "Profit Center"
_PC_SEG_COL = "Segment"
_CC_COL     = "Cost Center"
_CC_SEG_COL = "Segment"


class HierarchyService:
    """
    Builds and caches the SAP organizational hierarchy from master data tables.

    Hierarchy: Segment -> Profit Centers -> Cost Centers

    Data sources:
      v_profitcenter : Profit Center -> Segment  (authoritative)
      v_costcenter   : Cost Center  -> Segment   (secondary confirmation)
      enriched_df    : Profit Center <-> Cost Center (transactional relationship)
    """

    def __init__(self) -> None:
        self._seg_to_pcs: Dict[str, FrozenSet[str]] = {}
        self._pc_to_ccs:  Dict[str, FrozenSet[str]] = {}
        self._seg_to_ccs: Dict[str, FrozenSet[str]] = {}
        self._pc_to_seg:  Dict[str, str] = {}
        self._cc_to_seg:  Dict[str, str] = {}
        self._built = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        pc_table: Optional[pd.DataFrame],
        cc_table: Optional[pd.DataFrame],
        enriched_df: Optional[pd.DataFrame],
    ) -> None:
        """
        Build hierarchy from master data tables.

        Args:
            pc_table:    v_profitcenter DataFrame
            cc_table:    v_costcenter DataFrame
            enriched_df: enriched transactional DataFrame (for PC<->CC relationship)
        """
        seg_to_pcs_raw: Dict[str, Set[str]] = defaultdict(set)
        pc_to_ccs_raw:  Dict[str, Set[str]] = defaultdict(set)

        # STEP 1: PC -> Segment from v_profitcenter (authoritative)
        if pc_table is not None and not pc_table.empty:
            if _PC_COL in pc_table.columns and _PC_SEG_COL in pc_table.columns:
                for _, row in pc_table[[_PC_COL, _PC_SEG_COL]].dropna().iterrows():
                    pc  = str(row[_PC_COL]).strip()
                    seg = str(row[_PC_SEG_COL]).strip()
                    if pc and seg and seg.lower() not in ("nan", "none", ""):
                        self._pc_to_seg[pc] = seg
                        seg_to_pcs_raw[seg].add(pc)
                logger.info(
                    "HierarchyService: v_profitcenter -> %d PCs across %d segments: %s",
                    len(self._pc_to_seg),
                    len(seg_to_pcs_raw),
                    sorted(seg_to_pcs_raw.keys()),
                )
            else:
                logger.warning(
                    "HierarchyService: v_profitcenter missing '%s' or '%s'. Available: %s",
                    _PC_COL, _PC_SEG_COL, list(pc_table.columns),
                )
        else:
            logger.warning("HierarchyService: v_profitcenter table is empty or None.")

        # STEP 2: CC -> Segment from v_costcenter (secondary confirmation)
        if cc_table is not None and not cc_table.empty:
            if _CC_COL in cc_table.columns and _CC_SEG_COL in cc_table.columns:
                for _, row in cc_table[[_CC_COL, _CC_SEG_COL]].dropna().iterrows():
                    cc  = str(row[_CC_COL]).strip()
                    seg = str(row[_CC_SEG_COL]).strip()
                    if cc and seg and seg.lower() not in ("nan", "none", ""):
                        self._cc_to_seg[cc] = seg
                logger.info(
                    "HierarchyService: v_costcenter -> %d CCs with segment mapping",
                    len(self._cc_to_seg),
                )
            else:
                logger.warning(
                    "HierarchyService: v_costcenter missing '%s' or '%s'. Available: %s",
                    _CC_COL, _CC_SEG_COL, list(cc_table.columns),
                )
        else:
            logger.warning("HierarchyService: v_costcenter table is empty or None.")

        # STEP 3: PC <-> CC from enriched dataset (transactional relationship)
        if enriched_df is not None and not enriched_df.empty:
            if _PC_COL in enriched_df.columns and _CC_COL in enriched_df.columns:
                pc_cc_pairs = (
                    enriched_df[[_PC_COL, _CC_COL]]
                    .dropna()
                    .drop_duplicates()
                )
                for _, row in pc_cc_pairs.iterrows():
                    pc = str(row[_PC_COL]).strip()
                    cc = str(row[_CC_COL]).strip()
                    if pc and cc:
                        pc_to_ccs_raw[pc].add(cc)
                total_pc_cc = sum(len(v) for v in pc_to_ccs_raw.values())
                logger.info(
                    "HierarchyService: enriched dataset -> %d PC->CC relationships "
                    "across %d profit centers",
                    total_pc_cc, len(pc_to_ccs_raw),
                )
            else:
                logger.warning(
                    "HierarchyService: enriched_df missing '%s' or '%s' for PC<->CC mapping.",
                    _PC_COL, _CC_COL,
                )

        # STEP 4: Derive Segment -> CCs
        # Primary: via seg->pcs->ccs chain
        # Secondary: direct cc->seg from v_costcenter
        seg_to_ccs_raw: Dict[str, Set[str]] = defaultdict(set)

        for seg, pcs in seg_to_pcs_raw.items():
            for pc in pcs:
                seg_to_ccs_raw[seg].update(pc_to_ccs_raw.get(pc, set()))

        # Add CCs directly mapped to a segment via v_costcenter
        for cc, seg in self._cc_to_seg.items():
            seg_to_ccs_raw[seg].add(cc)

        # Freeze all sets
        self._seg_to_pcs = {k: frozenset(v) for k, v in seg_to_pcs_raw.items()}
        self._pc_to_ccs  = {k: frozenset(v) for k, v in pc_to_ccs_raw.items()}
        self._seg_to_ccs = {k: frozenset(v) for k, v in seg_to_ccs_raw.items()}

        self._built = True
        logger.info(
            "HierarchyService: built successfully — "
            "%d segments, %d profit centers, %d cost centers",
            len(self._seg_to_pcs),
            len(self._pc_to_seg),
            len(self._cc_to_seg),
        )

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    @property
    def is_built(self) -> bool:
        return self._built

    def get_profit_centers_for_segment(self, segment: str) -> FrozenSet[str]:
        """Return all Profit Center IDs belonging to the given Segment."""
        return self._seg_to_pcs.get(segment.strip(), frozenset())

    def get_cost_centers_for_segment(self, segment: str) -> FrozenSet[str]:
        """Return all Cost Center IDs belonging to the given Segment."""
        return self._seg_to_ccs.get(segment.strip(), frozenset())

    def get_cost_centers_for_profit_center(self, profit_center: str) -> FrozenSet[str]:
        """Return all Cost Center IDs belonging to the given Profit Center."""
        return self._pc_to_ccs.get(profit_center.strip(), frozenset())

    def get_segment_for_profit_center(self, profit_center: str) -> Optional[str]:
        """Return the Segment for a given Profit Center (from v_profitcenter)."""
        return self._pc_to_seg.get(profit_center.strip())

    def get_segment_for_cost_center(self, cost_center: str) -> Optional[str]:
        """Return the Segment for a given Cost Center (from v_costcenter)."""
        return self._cc_to_seg.get(cost_center.strip())

    def get_all_segments(self) -> List[str]:
        """Return all known segments (sorted)."""
        return sorted(self._seg_to_pcs.keys())

    def get_hierarchy_stats(self, segment: Optional[str] = None) -> dict:
        """
        Return hierarchy statistics for logging/debugging.
        If segment is provided, returns stats scoped to that segment.
        """
        if segment:
            pcs = self.get_profit_centers_for_segment(segment)
            ccs = self.get_cost_centers_for_segment(segment)
            return {
                "segment":                segment,
                "matched_profit_centers": len(pcs),
                "matched_cost_centers":   len(ccs),
                "profit_center_ids":      sorted(pcs)[:20],
                "cost_center_ids":        sorted(ccs)[:20],
            }
        return {
            "total_segments":       len(self._seg_to_pcs),
            "total_profit_centers": len(self._pc_to_seg),
            "total_cost_centers":   len(self._cc_to_seg),
            "segments":             self.get_all_segments(),
        }

    def apply_segment_filter(
        self,
        df: pd.DataFrame,
        segment: str,
        context: str = "",
    ) -> pd.DataFrame:
        """
        Apply a segment filter to a DataFrame using the hierarchy.

        Filtering logic (OR of three conditions):
          1. Profit Center is in the segment's PC set  (authoritative)
          2. Cost Center is in the segment's CC set    (secondary)
          3. Segment column == segment                 (fallback for rows without PC/CC)

        Logs:
          - matched_profit_centers
          - matched_cost_centers
          - filtered_transaction_rows
          - excluded_cost_centers_outside_segment

        Args:
            df:      DataFrame to filter (enriched dataset)
            segment: Segment name to filter by
            context: Logging context label

        Returns:
            Filtered DataFrame
        """
        before = len(df)
        seg_pcs = self.get_profit_centers_for_segment(segment)
        seg_ccs = self.get_cost_centers_for_segment(segment)

        # Collect all CCs present before filtering (for exclusion logging)
        all_ccs_before: Set[str] = set()
        if _CC_COL in df.columns:
            all_ccs_before = set(df[_CC_COL].dropna().astype(str).str.strip().unique())

        # Build inclusion mask
        mask = pd.Series(False, index=df.index)

        if seg_pcs and _PC_COL in df.columns:
            mask |= df[_PC_COL].astype(str).str.strip().isin(seg_pcs)

        if seg_ccs and _CC_COL in df.columns:
            mask |= df[_CC_COL].astype(str).str.strip().isin(seg_ccs)

        # Fallback: use Segment column directly (handles rows without PC/CC)
        if _PC_SEG_COL in df.columns:
            mask |= df[_PC_SEG_COL].astype(str).str.strip() == segment.strip()

        df_filtered = df[mask].copy()

        # Compute post-filter stats
        matched_pcs = (
            df_filtered[_PC_COL].dropna().astype(str).str.strip().nunique()
            if _PC_COL in df_filtered.columns else 0
        )
        matched_ccs = (
            df_filtered[_CC_COL].dropna().astype(str).str.strip().nunique()
            if _CC_COL in df_filtered.columns else 0
        )

        # Excluded CCs (present before filter, absent after)
        all_ccs_after: Set[str] = set()
        if _CC_COL in df_filtered.columns:
            all_ccs_after = set(
                df_filtered[_CC_COL].dropna().astype(str).str.strip().unique()
            )
        excluded_ccs = all_ccs_before - all_ccs_after

        logger.info(
            "hierarchy_service[%s]: segment='%s' filter applied — "
            "matched_profit_centers=%d, matched_cost_centers=%d, "
            "filtered_transaction_rows=%d (from %d), "
            "excluded_cost_centers_outside_segment=%d",
            context, segment,
            matched_pcs, matched_ccs,
            len(df_filtered), before,
            len(excluded_ccs),
        )

        if excluded_ccs:
            logger.info(
                "hierarchy_service[%s]: excluded CCs outside segment '%s': %s",
                context, segment,
                sorted(excluded_ccs)[:30],
            )

        return df_filtered