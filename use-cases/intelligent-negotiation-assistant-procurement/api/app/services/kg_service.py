from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import image-based pipeline and unification/export utilities
from app.core.kg_creation.image_kg_pipeline import create_image_based_kg
from app.core.kg_creation.unification.kg_unifier import KGUnifier
from app.core.kg_creation.serialization.graph_exporter import GraphExporter


@dataclass
class SupplierInfo:
    """Represents a static supplier KG reference.

    Attributes:
        supplier_id: Stable identifier (e.g., "supplier1").
        name: Human-friendly supplier name.
        filename: Filename of the KG JSON within the supplier directory.
    """

    supplier_id: str
    name: str
    filename: str


class KGService:
    """Service for Knowledge Graph operations used by API routers.

    Provides:
    - Static KG listing/loading from packaged data directory
    - Future: wrappers for KG creation and unification
    """

    def __init__(self) -> None:
        # Resolve settings from environment with sensible defaults
        self.supplier_dir = Path(os.getenv("SUPPLIER_KG_DIR", "app/data/kg/suppliers")).resolve()
        self.supplier1_id = os.getenv("SUPPLIER1_ID", "supplier1")
        self.supplier2_id = os.getenv("SUPPLIER2_ID", "supplier2")
        self.supplier1_file = os.getenv("SUPPLIER1_FILE", "supplier1.json")
        self.supplier2_file = os.getenv("SUPPLIER2_FILE", "supplier2.json")
        self.supplier1_name = os.getenv("SUPPLIER1_NAME", "SupplierA")
        self.supplier2_name = os.getenv("SUPPLIER2_NAME", "SupplierB")
        # Ensure base directory exists for dynamic runs
        self.supplier_dir.mkdir(parents=True, exist_ok=True)
        self._supplier_index_path = self.supplier_dir / "index.json"

    # ============
    # Index helpers
    # ============

    def _load_supplier_index(self) -> Dict[str, Any]:
        """Load supplier index JSON, returning an empty skeleton on failure."""
        default: Dict[str, Any] = {"suppliers": []}
        if not self._supplier_index_path.exists():
            return default
        try:
            with open(self._supplier_index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("suppliers"), list):
                return data
        except Exception:
            pass
        return default

    def _save_supplier_index(self, data: Dict[str, Any]) -> None:
        """Persist supplier index JSON safely, avoiding partial writes."""
        tmp_path = self._supplier_index_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        tmp_path.replace(self._supplier_index_path)

    def _update_supplier_index(self, new_suppliers: List[Dict[str, Any]]) -> None:
        """Merge new supplier metadata into the index file."""
        if not new_suppliers:
            return

        index_data = self._load_supplier_index()
        existing = {
            str(entry.get("id")): entry
            for entry in index_data.get("suppliers", [])
            if isinstance(entry, dict) and entry.get("id")
        }
        now_iso = datetime.now(timezone.utc).isoformat()

        for supplier in new_suppliers:
            supplier_id = str(supplier.get("id", "")).strip()
            if not supplier_id:
                continue
            merged = dict(existing.get(supplier_id, {}))
            merged.update(supplier)
            if not merged.get("created_at"):
                merged["created_at"] = now_iso
            merged["updated_at"] = now_iso
            existing[supplier_id] = merged

        index_data["suppliers"] = list(existing.values())
        self._save_supplier_index(index_data)

    def list_static_suppliers(self) -> Dict[str, List[Dict[str, str]]]:
        """Return the list of statically packaged supplier KGs.

        Returns:
            A dict with key "suppliers" containing supplier entries.
        """
        suppliers: List[Dict[str, str]] = []
        # If an index.json exists, prefer it
        index_path = self.supplier_dir / "index.json"
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and isinstance(data.get("suppliers"), list):
                    # Ensure minimal schema
                    for s in data["suppliers"]:
                        sid = str(s.get("id", "")).strip()
                        name = str(s.get("name", "")).strip() or sid
                        filename = str(s.get("filename", "")).strip() or f"{sid}.json"
                        if sid:
                            suppliers.append({"id": sid, "name": name, "filename": filename})
                    return {"suppliers": suppliers}
            except Exception:
                # Fall through to env defaults
                pass

        # Fallback: build from environment defaults
        suppliers.append({
            "id": self.supplier1_id,
            "name": self.supplier1_name,
            "filename": self.supplier1_file,
        })
        suppliers.append({
            "id": self.supplier2_id,
            "name": self.supplier2_name,
            "filename": self.supplier2_file,
        })
        return {"suppliers": suppliers}

    def get_supplier_display_name(self, supplier_id: str) -> Optional[str]:
        """Return a human-friendly supplier name for a given id if known.

        Looks into suppliers/index.json first, then falls back to env defaults
        for SUPPLIER1_ID and SUPPLIER2_ID. Returns None if unknown.
        """
        try:
            index_path = self.supplier_dir / "index.json"
            if index_path.exists():
                with open(index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for s in (data.get("suppliers") or []):
                    if str(s.get("id")) == supplier_id:
                        nm = str(s.get("name") or "").strip()
                        if nm:
                            return nm
        except Exception:
            pass
        try:
            if supplier_id == self.supplier1_id:
                return self.supplier1_name
            if supplier_id == self.supplier2_id:
                return self.supplier2_name
        except Exception:
            pass
        return None

    def get_static_supplier_path(self, supplier_id: str) -> Path:
        """Resolve on-disk JSON path for a given supplier id.

        Raises FileNotFoundError if not found.
        """
        # Helper to remap absolute paths from different machines to local supplier_dir
        def _coerce_local(path_str: Optional[str]) -> Optional[Path]:
            if not path_str:
                return None
            try:
                p = Path(path_str)
                if p.exists():
                    return p
                parts = list(p.parts)
                # If path contains a 'suppliers' segment, rebuild locally after it
                if "suppliers" in parts:
                    sidx = parts.index("suppliers")
                    rel = Path(*parts[sidx + 1:])
                    return self.supplier_dir / rel
            except Exception:
                return None
            return None

        def _find_unified_json_in_dir(dir_path: Optional[Path]) -> Optional[Path]:
            try:
                if not dir_path or not dir_path.exists() or not dir_path.is_dir():
                    return None
                candidates = sorted(dir_path.glob("unified_kg_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                if candidates:
                    return candidates[0]
                # fallback to any per-file *_kg.json
                per_file = sorted(dir_path.glob("*_kg.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                if per_file:
                    return per_file[0]
            except Exception:
                return None
            return None

        # Try index.json map first
        index_path = self.supplier_dir / "index.json"
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                suppliers_list = list(data.get("suppliers") or [])
                for s in suppliers_list:
                    if str(s.get("id")) == supplier_id:
                        # 1) Direct filename in suppliers dir
                        filename = str(s.get("filename") or "").strip()
                        if filename:
                            path = self.supplier_dir / filename
                            if path.exists():
                                return path
                        # 2) kgs_dir/root_dir unified files
                        kgs_dir = _coerce_local(s.get("kgs_dir"))
                        root_dir = _coerce_local(s.get("root_dir"))
                        path = _find_unified_json_in_dir(kgs_dir)
                        if path:
                            return path
                        if root_dir:
                            path = _find_unified_json_in_dir(root_dir / "kgs") or _find_unified_json_in_dir(root_dir)
                            if path:
                                return path
                        # 3) If no direct mapping and multiple unified files exist in root, map by index order
                        try:
                            unified_in_root = sorted(self.supplier_dir.glob("unified_kg_*.json"), key=lambda p: p.stat().st_mtime)
                            if len(unified_in_root) >= 2:
                                idx = next((i for i, it in enumerate(suppliers_list) if str(it.get("id")) == supplier_id), 0)
                                idx = 0 if idx < 0 else (1 if idx >= 2 else idx)
                                return unified_in_root[idx]
                        except Exception:
                            pass
            except Exception:
                pass

        # Fallback: environment defaults
        if supplier_id == self.supplier1_id:
            path = self.supplier_dir / self.supplier1_file
        elif supplier_id == self.supplier2_id:
            path = self.supplier_dir / self.supplier2_file
        else:
            # Generic fallback to {id}.json
            path = self.supplier_dir / f"{supplier_id}.json"
        if not path.exists():
            # Try old demo files as a last resort
            try:
                old_dir = self.supplier_dir / "old"
                sid_lower = supplier_id.lower()
                if ("supplier_a" in sid_lower or sid_lower.endswith("1")) and (old_dir / "supplier1.json").exists():
                    return old_dir / "supplier1.json"
                if ("supplier_b" in sid_lower or sid_lower.endswith("2")) and (old_dir / "supplier2.json").exists():
                    return old_dir / "supplier2.json"
            except Exception:
                pass
            raise FileNotFoundError(f"Static KG not found for supplier '{supplier_id}' at {path}")
        return path

    def load_static_supplier_kg(self, supplier_id: str) -> Dict[str, Any]:
        """Load KG JSON for a given static supplier id.

        Returns the parsed JSON content.
        """
        path = self.get_static_supplier_path(supplier_id)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # =====================
    # Dynamic KG generation
    # =====================

    def _slugify(self, name: str) -> str:
        """Create a filesystem-safe slug for a supplier name.

        Converts to lowercase, replaces non-alphanumeric with underscores,
        and trims repeated underscores.
        """
        base = name.strip().lower()
        base = re.sub(r"[^a-z0-9]+", "_", base)
        base = re.sub(r"_+", "_", base).strip("_")
        return base or "supplier"

    def _prepare_supplier_run_dir(self, supplier_name: str) -> Dict[str, str]:
        """Create a unique run directory for a supplier and return paths.

        Structure:
        - app/data/kg/suppliers/<slug>-<timestamp>/
            - pdfs/   (uploaded PDFs)
            - kgs/    (per-file KG outputs)
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        slug = self._slugify(supplier_name)
        supplier_id = f"{slug}-{timestamp}"

        root_dir = self.supplier_dir / supplier_id
        pdfs_dir = root_dir / "pdfs"
        kgs_dir = root_dir / "kgs"
        pdfs_dir.mkdir(parents=True, exist_ok=True)
        kgs_dir.mkdir(parents=True, exist_ok=True)

        return {
            "supplier_id": supplier_id,
            "root_dir": str(root_dir),
            "pdfs_dir": str(pdfs_dir),
            "kgs_dir": str(kgs_dir),
        }

    def _save_uploads(self, upload_files: List[Any], dest_dir: str) -> List[str]:
        """Save uploaded files to destination directory and return saved paths.

        This function streams file content to disk to avoid loading
        entire files into memory.
        """
        saved_paths: List[str] = []
        for uf in upload_files or []:
            # Derive a safe filename
            original_name = getattr(uf, "filename", "document.pdf") or "document.pdf"
            safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", original_name)
            dest_path = Path(dest_dir) / safe_name
            # Stream copy to disk
            with open(dest_path, "wb") as out_f:
                shutil.copyfileobj(uf.file, out_f)
            saved_paths.append(str(dest_path))
        return saved_paths

    def _extract_image_based(self, pdf_path: str, output_dir: str) -> Dict[str, Any]:
        """Run image-based KG extraction for a single PDF.

        Returns dictionary with 'output_files' including json and graphml.
        """
        # Use default model suitable for vision; allow env override if set
        llm_model = os.getenv("KG_IMAGE_MODEL", "gemini-2.5-pro")
        result = create_image_based_kg(
            file_path=pdf_path,
            output_dir=output_dir,
            llm_model=llm_model,
            parallel_processing=True,
        )
        return result

    def _create_supplier_run(self, supplier_name: str, supplier_files: Optional[List[Any]]) -> Dict[str, Any]:
        """Create directories, store uploads, and run KG extraction for a supplier."""
        dirs = self._prepare_supplier_run_dir(supplier_name)
        saved_files = self._save_uploads(supplier_files or [], dirs["pdfs_dir"])

        kg_outputs: List[Dict[str, str]] = []
        for src in saved_files:
            result = self._extract_image_based(src, dirs["kgs_dir"])
            kg_outputs.append(result.get("output_files", {}))

        return {
            "id": dirs["supplier_id"],
            "name": supplier_name,
            "root_dir": dirs["root_dir"],
            "pdfs_dir": dirs["pdfs_dir"],
            "kgs_dir": dirs["kgs_dir"],
            "pdfs": saved_files,
            "kg_outputs": kg_outputs,
        }

    def create_from_uploads(self, suppliers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create KGs for one or more suppliers from uploaded files using image-mode.

        Saves files and per-file KG outputs under app/data/kg/suppliers/.
        Returns identifiers and paths needed for later unification.

        Args:
            suppliers: List of dicts with 'name' and 'files' keys.
        """
        if not suppliers:
            raise ValueError("At least one supplier must be provided for KG creation.")

        created_suppliers: List[Dict[str, Any]] = []
        for supplier in suppliers:
            if not supplier:
                continue
            name = str(supplier.get("name", "")).strip()
            files = supplier.get("files")
            if not name:
                raise ValueError("Each supplier entry must include a non-empty name.")
            created_suppliers.append(self._create_supplier_run(name, files))

        if not created_suppliers:
            raise ValueError("No valid suppliers supplied for KG creation.")

        response: Dict[str, Any] = {"suppliers": created_suppliers}
        # Backward compatible keys
        if created_suppliers:
            response["supplier1"] = created_suppliers[0]
        if len(created_suppliers) > 1:
            response["supplier2"] = created_suppliers[1]

        # Persist metadata so /kg/static/list reflects newly created runs
        index_entries: List[Dict[str, Any]] = []
        for supplier in created_suppliers:
            index_entries.append({
                "id": supplier["id"],
                "name": supplier["name"],
                "root_dir": supplier["root_dir"],
                "kgs_dir": supplier["kgs_dir"],
                "source": "dynamic",
            })
        self._update_supplier_index(index_entries)

        return response

    def unify_supplier(self, supplier_id: str) -> Dict[str, Any]:
        """Unify all per-file KGs for a given supplier run directory.

        Looks for JSON files ending with *_kg.json under the supplier's kgs_dir.
        Writes unified JSON and GraphML in the same directory.
        """
        supplier_path = Path(self.supplier_dir) / supplier_id
        kgs_dir = supplier_path / "kgs"
        if not kgs_dir.exists():
            raise FileNotFoundError(f"KGs directory not found for supplier '{supplier_id}': {kgs_dir}")

        # Collect per-file KG JSONs (exclude previously unified outputs)
        json_files = sorted([
            str(p) for p in kgs_dir.glob("*_kg.json")
            if p.is_file()
        ])
        if not json_files:
            raise FileNotFoundError(f"No per-file KG JSONs found for supplier '{supplier_id}' in {kgs_dir}")

        # Unify graphs
        unifier = KGUnifier(conflict_resolution=os.getenv("KG_UNIFY_STRATEGY", "last_wins"))
        unified_kg = unifier.unify_from_json_files(json_files, unified_name=supplier_id)

        # Export unified graphs with timestamp
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        exporter = GraphExporter(str(kgs_dir))
        unified_json = exporter.export_json(unified_kg, f"unified_kg_{timestamp}.json")
        unified_graphml = exporter.export_graphml(unified_kg, f"unified_kg_{timestamp}.graphml.gz")

        return {
            "supplier_id": supplier_id,
            "unified_json": unified_json,
            "unified_graphml": unified_graphml,
            "statistics": unifier.get_statistics(),
        }

    def unify_two_suppliers(self, supplier1_id: str, supplier2_id: str) -> Dict[str, Any]:
        """Unify KGs for two suppliers and return output paths for each."""
        s1 = self.unify_supplier(supplier1_id)
        s2 = self.unify_supplier(supplier2_id)
        return {"supplier1": s1, "supplier2": s2}
