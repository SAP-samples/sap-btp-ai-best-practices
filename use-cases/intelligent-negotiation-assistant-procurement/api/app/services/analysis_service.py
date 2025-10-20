from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import os
import json
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.services.kg_service import KGService
from app.core.structured_llm_analyzers.structured_cost_analyzer import (
    analyze_costs_structured,
)
from app.core.structured_llm_analyzers.structured_risk_analyzer import (
    analyze_risks_structured,
)
from app.core.structured_llm_analyzers.structured_parts_analyzer import (
    analyze_parts_structured,
)
from app.core.structured_llm_analyzers.structured_homepage import (
    analyze_homepage_structured,
)
from app.core.structured_llm_analyzers.structured_tqdcs_analyzer import (
    analyze_tqdcs_structured,
)
from app.core.structured_llm_analyzers.structured_comparator import (
    compare_suppliers_structured,
)


class AnalysisService:
    """Service providing wrappers for analyzer functions with supplier id support."""

    def __init__(self) -> None:
        self.kg_service = KGService()
        self.cache_dir = Path(os.getenv("CACHE_DIR", "app/data/cache/analyses")).resolve()
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Best-effort in read-only environments
            pass

    def _slug(self, text: Optional[str]) -> str:
        t = (text or "unknown").lower()
        return "".join(c if c.isalnum() or c == '_' else '_' for c in t)

    def _mtime_key(self, path: str) -> str:
        try:
            return str(int(os.path.getmtime(path)))
        except Exception:
            return "0"

    def _cache_file(self, analysis: str, kg_path: str, supplier_name: Optional[str], extra: Optional[str] = None) -> Path:
        name = self._slug(supplier_name)
        mkey = self._mtime_key(kg_path)
        suffix = f"_{extra}" if extra else ""
        fname = f"{analysis}_{name}_{mkey}{suffix}.json"
        return self.cache_dir / fname

    def _read_cache(self, file_path: Path) -> Optional[Dict[str, Any]]:
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            return None
        return None

    def _write_cache(self, file_path: Path, data: Dict[str, Any]) -> None:
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            # Ignore caching errors
            pass

    def _dated_cache_file(self, analysis: str, supplier_name: Optional[str], extra: Optional[str] = None) -> Path:
        from datetime import datetime
        datestr = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = self._slug(supplier_name)
        suffix = f"_{extra}" if extra else ""
        fname = f"{analysis}_{name}_{datestr}{suffix}.json"
        return self.cache_dir / fname

    def _find_latest_cache(self, analysis: str, supplier_name: Optional[str], extra: Optional[str] = None) -> Optional[Path]:
        name = self._slug(supplier_name)
        try:
            pattern = f"{analysis}_{name}_*{('_' + extra) if extra else ''}.json"
            matches = sorted(self.cache_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            return matches[0] if matches else None
        except Exception:
            return None

    def _has_any_cache(self, analysis: str, supplier_name: Optional[str], extra: Optional[str] = None) -> bool:
        name = self._slug(supplier_name)
        try:
            pattern = f"{analysis}_{name}_*{('_' + extra) if extra else ''}.json"
            for _ in self.cache_dir.glob(pattern):
                return True
            return False
        except Exception:
            return False

    def _slug_candidates(self, supplier_name: Optional[str], supplier_id: Optional[str]) -> list[str]:
        candidates = []
        try:
            if supplier_name:
                candidates.append(self._slug(supplier_name))
            if supplier_id:
                candidates.append(self._slug(supplier_id))
            # Fallback for older runs that may have written 'unknown'
            candidates.append("unknown")
        except Exception:
            candidates = [self._slug(supplier_name or ""), self._slug(supplier_id or ""), "unknown"]
        # Deduplicate while preserving order
        seen = set()
        uniq: list[str] = []
        for c in candidates:
            if c and c not in seen:
                uniq.append(c)
                seen.add(c)
        return uniq

    def _has_any_cache_by_slugs(self, analysis: str, slugs: list[str], extra: Optional[str] = None) -> bool:
        try:
            for slug in slugs:
                pattern = f"{analysis}_{slug}_*{('_' + extra) if extra else ''}.json"
                for _ in self.cache_dir.glob(pattern):
                    return True
            return False
        except Exception:
            return False

    def _find_latest_cache_by_slugs(self, analysis: str, slugs: list[str], extra: Optional[str] = None) -> Optional[Path]:
        try:
            matches: list[Path] = []
            for slug in slugs:
                pattern = f"{analysis}_{slug}_*{('_' + extra) if extra else ''}.json"
                matches.extend(list(self.cache_dir.glob(pattern)))
            if not matches:
                return None
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return matches[0]
        except Exception:
            return None

    def _find_latest_comparison(self, supplier1_name: Optional[str], supplier2_name: Optional[str], supplier1_id: Optional[str], supplier2_id: Optional[str]) -> Optional[Path]:
        slugs1 = self._slug_candidates(supplier1_name, supplier1_id)
        slugs2 = self._slug_candidates(supplier2_name, supplier2_id)
        try:
            matches: list[Path] = []
            for a in slugs1:
                for b in slugs2:
                    pattern1 = f"comparison_{a}_vs_{b}_*.json"
                    pattern2 = f"comparison_{b}_vs_{a}_*.json"
                    matches.extend(list(self.cache_dir.glob(pattern1)))
                    matches.extend(list(self.cache_dir.glob(pattern2)))
            if not matches:
                return None
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return matches[0]
        except Exception:
            return None

    def _resolve_kg_path(self, supplier_id: Optional[str], kg_path: Optional[str]) -> Tuple[str, Optional[str]]:
        """Return an absolute kg path and supplier name if id provided."""
        if kg_path:
            return kg_path, None
        if supplier_id:
            path = str(self.kg_service.get_static_supplier_path(supplier_id))
            # Prefer a stable, human-friendly name for slugging when available
            display_name = self.kg_service.get_supplier_display_name(supplier_id)
            if display_name:
                return path, display_name
            if supplier_id == self.kg_service.supplier1_id:
                return path, self.kg_service.supplier1_name
            if supplier_id == self.kg_service.supplier2_id:
                return path, self.kg_service.supplier2_name
            return path, supplier_id
        raise ValueError("Either supplier_id or kg_path must be provided")

    def run_cost(self, supplier_id: Optional[str] = None, kg_path: Optional[str] = None, model: Optional[str] = None, force_refresh: bool = False) -> Dict[str, Any]:
        path, name = self._resolve_kg_path(supplier_id, kg_path)
        latest_fp = self._find_latest_cache('cost', name)
        if latest_fp and not force_refresh:
            cached = self._read_cache(latest_fp)
            if cached:
                return cached
        result = analyze_costs_structured(kg_json_path=path, supplier_name=name, model_name=model)
        self._write_cache(self._dated_cache_file('cost', name), result)
        return result

    def run_risk(self, supplier_id: Optional[str] = None, kg_path: Optional[str] = None, model: Optional[str] = None, force_refresh: bool = False) -> Dict[str, Any]:
        path, name = self._resolve_kg_path(supplier_id, kg_path)
        latest_fp = self._find_latest_cache('risk', name)
        if latest_fp and not force_refresh:
            cached = self._read_cache(latest_fp)
            if cached:
                return cached
        result = analyze_risks_structured(kg_json_path=path, supplier_name=name, model_name=model)
        self._write_cache(self._dated_cache_file('risk', name), result)
        return result

    def run_parts(
        self,
        supplier_id: Optional[str] = None,
        kg_path: Optional[str] = None,
        model: Optional[str] = None,
        core_part_categories: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        path, name = self._resolve_kg_path(supplier_id, kg_path)
        extra = None
        if core_part_categories:
            try:
                cats_key = ",".join(sorted([str(c).strip().upper() for c in core_part_categories]))
                extra = hashlib.md5(cats_key.encode()).hexdigest()[:6]
            except Exception:
                extra = None
        latest_fp = self._find_latest_cache('parts', name, extra)
        if latest_fp:
            cached = self._read_cache(latest_fp)
            if cached:
                return cached
        result = analyze_parts_structured(
            kg_json_path=path, supplier_name=name, model_name=model, core_categories=core_part_categories
        )
        self._write_cache(self._dated_cache_file('parts', name, extra), result)
        return result

    def run_homepage(self, supplier_id: Optional[str] = None, kg_path: Optional[str] = None, model: Optional[str] = None, force_refresh: bool = False) -> Dict[str, Any]:
        path, name = self._resolve_kg_path(supplier_id, kg_path)
        latest_fp = self._find_latest_cache('homepage', name)
        if latest_fp and not force_refresh:
            cached = self._read_cache(latest_fp)
            if cached:
                return cached
        result = analyze_homepage_structured(kg_json_path=path, supplier_name=name, model_name=model)
        self._write_cache(self._dated_cache_file('homepage', name), result)
        return result

    def run_tqdcs(
        self,
        supplier_id: Optional[str] = None,
        kg_path: Optional[str] = None,
        model: Optional[str] = None,
        prior_analyses: Optional[Dict[str, Any]] = None,
        weights: Optional[Dict[str, float]] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        path, name = self._resolve_kg_path(supplier_id, kg_path)
        latest_fp = self._find_latest_cache('tqdcs', name)
        if latest_fp and not force_refresh:
            cached = self._read_cache(latest_fp)
            if cached:
                return cached
        result = analyze_tqdcs_structured(
            kg_json_path=path,
            supplier_name=name,
            weights=weights,
            prior_analyses=prior_analyses,
            model_name=model,
        )
        self._write_cache(self._dated_cache_file('tqdcs', name), result)
        return result

    def run_compare(
        self,
        supplier1_name: str,
        supplier2_name: str,
        supplier1_analyses: Optional[Dict[str, Any]] = None,
        supplier2_analyses: Optional[Dict[str, Any]] = None,
        tqdcs_weights: Optional[Dict[str, float]] = None,
        generate_metrics: bool = True,
        generate_strengths_weaknesses: bool = True,
        generate_recommendation_and_split: bool = True,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = compare_suppliers_structured(
            supplier1_name=supplier1_name,
            supplier2_name=supplier2_name,
            supplier1_analyses=supplier1_analyses,
            supplier2_analyses=supplier2_analyses,
            tqdcs_weights=tqdcs_weights,
            generate_metrics=generate_metrics,
            generate_strengths_weaknesses=generate_strengths_weaknesses,
            generate_recommendation_and_split=generate_recommendation_and_split,
            model_name=model,
        )
        # Always write a new comparison file; include both names and timestamp
        name_combo = f"{self._slug(supplier1_name)}_vs_{self._slug(supplier2_name)}"
        dated = self._dated_cache_file(f"comparison_{name_combo}", None)
        self._write_cache(dated, result)
        return result

    # Cache introspection and ensure orchestration
    def get_cache_status(
        self,
        supplier1_id: str,
        supplier2_id: str,
        core_part_categories: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        path1, name1 = self._resolve_kg_path(supplier1_id, None)
        path2, name2 = self._resolve_kg_path(supplier2_id, None)
        extra: Optional[str] = None
        if core_part_categories:
            try:
                cats_key = ",".join(sorted([str(c).strip().upper() for c in core_part_categories]))
                extra = hashlib.md5(cats_key.encode()).hexdigest()[:6]
            except Exception:
                extra = None

        def _status_for(sid: str, nm: Optional[str]) -> Dict[str, bool]:
            slugs = self._slug_candidates(nm, sid)
            return {
                "has_cost": self._has_any_cache_by_slugs('cost', slugs),
                "has_risk": self._has_any_cache_by_slugs('risk', slugs),
                "has_parts": self._has_any_cache_by_slugs('parts', slugs, extra),
                "has_homepage": self._has_any_cache_by_slugs('homepage', slugs),
                "has_tqdcs": self._has_any_cache_by_slugs('tqdcs', slugs),
            }

        has_comp = self._find_latest_comparison(name1, name2, supplier1_id, supplier2_id) is not None

        return {
            "supplier1": {"id": supplier1_id, "name": name1, **_status_for(supplier1_id, name1)},
            "supplier2": {"id": supplier2_id, "name": name2, **_status_for(supplier2_id, name2)},
            "has_comparison": has_comp,
        }

    def ensure_complete_or_compare(
        self,
        supplier1_id: str,
        supplier2_id: str,
        model: Optional[str] = None,
        comparator_model: Optional[str] = None,
        core_part_categories: Optional[list[str]] = None,
        tqdcs_weights: Optional[Dict[str, float]] = None,
        force_refresh: bool = False,
        generate_metrics: bool = True,
        generate_strengths_weaknesses: bool = True,
        generate_recommendation_and_split: bool = True,
    ) -> Dict[str, Any]:
        # Granular ensure: reuse what exists, compute only what's missing or when forced
        status = self.get_cache_status(supplier1_id, supplier2_id, core_part_categories)
        path1, name1 = self._resolve_kg_path(supplier1_id, None)
        path2, name2 = self._resolve_kg_path(supplier2_id, None)

        extra = None
        if core_part_categories:
            try:
                cats_key = ",".join(sorted([str(c).strip().upper() for c in core_part_categories]))
                extra = hashlib.md5(cats_key.encode()).hexdigest()[:6]
            except Exception:
                extra = None

        def _load_latest_for(analysis: str, name: Optional[str], extra_key: Optional[str] = None) -> Dict[str, Any]:
            fp = self._find_latest_cache(analysis, name, extra_key)
            return self._read_cache(fp) or {}

        def _build_for_supplier(supplier_id: str, name: Optional[str]) -> Dict[str, Any]:
            # cost
            if not force_refresh and status[f"supplier{1 if supplier_id==supplier1_id else 2}"]["has_cost"]:
                cost_data = _load_latest_for('cost', name)
            else:
                cost_data = self.run_cost(supplier_id, None, model, force_refresh)

            # risk
            if not force_refresh and status[f"supplier{1 if supplier_id==supplier1_id else 2}"]["has_risk"]:
                risk_data = _load_latest_for('risk', name)
            else:
                risk_data = self.run_risk(supplier_id, None, model, force_refresh)

            # parts (depends on categories)
            if not force_refresh and status[f"supplier{1 if supplier_id==supplier1_id else 2}"]["has_parts"]:
                parts_data = _load_latest_for('parts', name, extra)
            else:
                parts_data = self.run_parts(supplier_id, None, model, core_part_categories)

            # homepage
            if not force_refresh and status[f"supplier{1 if supplier_id==supplier1_id else 2}"]["has_homepage"]:
                homepage_data = _load_latest_for('homepage', name)
            else:
                homepage_data = self.run_homepage(supplier_id, None, model, force_refresh)

            # tqdcs (computed after prior analyses). Cache not keyed by weights.
            if not force_refresh and status[f"supplier{1 if supplier_id==supplier1_id else 2}"]["has_tqdcs"]:
                tqdcs_data = _load_latest_for('tqdcs', name)
            else:
                prior = {"parts": parts_data, "cost": cost_data, "risk": risk_data}
                tqdcs_data = self.run_tqdcs(supplier_id, None, model, prior_analyses=prior, weights=None, force_refresh=force_refresh)

            return {
                "supplier_name": name,
                "kg_path": self.kg_service.get_static_supplier_path(supplier_id),
                "cost": cost_data,
                "risks": risk_data,
                "parts": parts_data,
                "homepage": homepage_data,
                "tqdcs": tqdcs_data,
            }

        s1 = _build_for_supplier(supplier1_id, name1)
        s2 = _build_for_supplier(supplier2_id, name2)

        # If no weights override and cached comparison exists, reuse it
        comparison_fp = None
        if not force_refresh and (tqdcs_weights is None or (isinstance(tqdcs_weights, dict) and not any(tqdcs_weights.values()))):
            comparison_fp = self._find_latest_comparison(name1, name2, supplier1_id, supplier2_id)

        if comparison_fp:
            comparison = self._read_cache(comparison_fp) or {}
        else:
            comparison = self.run_compare(
                supplier1_name=name1,
                supplier2_name=name2,
                supplier1_analyses=s1,
                supplier2_analyses=s2,
                tqdcs_weights=tqdcs_weights,
                generate_metrics=generate_metrics,
                generate_strengths_weaknesses=generate_strengths_weaknesses,
                generate_recommendation_and_split=generate_recommendation_and_split,
                model=comparator_model or model,
            )

        used_cache = (
            status["supplier1"]["has_cost"]
            and status["supplier1"]["has_risk"]
            and status["supplier1"]["has_parts"]
            and status["supplier1"]["has_homepage"]
            and status["supplier1"]["has_tqdcs"]
            and status["supplier2"]["has_cost"]
            and status["supplier2"]["has_risk"]
            and status["supplier2"]["has_parts"]
            and status["supplier2"]["has_homepage"]
            and status["supplier2"]["has_tqdcs"]
            and not force_refresh
        )

        return {"supplier1": s1, "supplier2": s2, "comparison": comparison, "used_cache": used_cache}

    # Orchestrators
    def run_supplier_full(
        self,
        supplier_id: Optional[str] = None,
        kg_path: Optional[str] = None,
        model: Optional[str] = None,
        core_part_categories: Optional[list[str]] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        path, name = self._resolve_kg_path(supplier_id, kg_path)

        results: Dict[str, Any] = {"supplier_name": name, "kg_path": path}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                'cost': executor.submit(self.run_cost, supplier_id, None, model, force_refresh),
                'risk': executor.submit(self.run_risk, supplier_id, None, model, force_refresh),
                'parts': executor.submit(self.run_parts, supplier_id, None, model, core_part_categories),
                'homepage': executor.submit(self.run_homepage, supplier_id, None, model, force_refresh),
            }
            for f in as_completed(list(futures.values())):
                pass  # wait for all
        results['cost'] = futures['cost'].result()
        results['risks'] = futures['risk'].result()
        results['parts'] = futures['parts'].result()
        results['homepage'] = futures['homepage'].result()

        # TQDCS after prior analyses
        prior = {"parts": results['parts'], "cost": results['cost'], "risk": results['risks']}
        results['tqdcs'] = self.run_tqdcs(supplier_id, None, model, prior_analyses=prior, force_refresh=force_refresh)
        return results

    def run_complete(
        self,
        supplier1_id: Optional[str],
        supplier2_id: Optional[str],
        supplier1_name: Optional[str] = None,
        supplier2_name: Optional[str] = None,
        model: Optional[str] = None,
        comparator_model: Optional[str] = None,
        core_part_categories: Optional[list[str]] = None,
        tqdcs_weights: Optional[Dict[str, float]] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            fut1 = executor.submit(self.run_supplier_full, supplier1_id, None, model, core_part_categories, force_refresh)
            fut2 = executor.submit(self.run_supplier_full, supplier2_id, None, model, core_part_categories, force_refresh)
            supplier1_data = fut1.result()
            supplier2_data = fut2.result()
        comparison = self.run_compare(
            supplier1_name=supplier1_data['supplier_name'],
            supplier2_name=supplier2_data['supplier_name'],
            supplier1_analyses=supplier1_data,
            supplier2_analyses=supplier2_data,
            tqdcs_weights=tqdcs_weights,
            model=comparator_model or model,
        )
        return {"supplier1": supplier1_data, "supplier2": supplier2_data, "comparison": comparison}
