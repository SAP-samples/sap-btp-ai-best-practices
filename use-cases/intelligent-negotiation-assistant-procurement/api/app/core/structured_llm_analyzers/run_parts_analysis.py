"""
Run Parts Analysis - CLI helper to test structured_parts_analyzer.

Purpose:
- Given a unified Knowledge Graph JSON path, generate a dashboard-compatible
  parts analysis JSON in the template dashboard cache directory.

Notes:
- The output filename follows the dashboard cache convention:
  parts_analysis_{supplier_name}_{YYYY_MM_DD}_{hash8}.json
- Uses the upgraded analyzer with deterministic canonicalization enabled by default.
"""
import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

# Local import of analyzer (module ensures required sys.path for llm factory)
from structured_parts_analyzer import analyze_parts_structured


def _clean_supplier_name(name: str) -> str:
    """Normalize supplier name for filenames (lowercase, underscores, no dots)."""
    if not name:
        return "unknown_supplier"
    return name.replace(" ", "_").replace(".", "").lower()


def _compute_cache_name(kg_path: str, supplier_name: str) -> str:
    """
    Build a cache filename consistent with the dashboard loader patterns.
    Includes current date and an 8-char hash of (path, supplier, mtime).
    """
    clean_supplier = _clean_supplier_name(supplier_name)
    current_date = datetime.now().strftime("%Y_%m_%d")
    try:
        mtime = os.path.getmtime(kg_path)
    except Exception:
        mtime = 0
    hash_string = f"{kg_path}|{supplier_name}|{mtime}"
    hash_suffix = hashlib.md5(hash_string.encode()).hexdigest()[:8]
    return f"parts_analysis_{clean_supplier}_{current_date}_{hash_suffix}.json"


def _default_cache_dir(script_path: Path) -> Path:
    """
    Resolve the template dashboard cache directory relative to this script:
    resources/kg_creation/prototype/template_dashboard/cache/analyses
    """
    # From structured_llm_analyzers/ → up to prototype/
    prototype_dir = script_path.parent.parent
    cache_dir = prototype_dir / "template_dashboard" / "cache" / "analyses"
    return cache_dir


def main():
    parser = argparse.ArgumentParser(description="Run parts analysis for a KG JSON and emit dashboard-compatible output")
    parser.add_argument("kg_path", help="Path to the unified KG JSON file")
    parser.add_argument("--supplier", default="Unknown Supplier", help="Supplier name for labeling the analysis")
    parser.add_argument("--output", default=None, help="Optional explicit output JSON path")
    parser.add_argument("--no-canonicalize", action="store_true", help="Disable deterministic canonicalization")
    parser.add_argument("--no-filter", action="store_true", help="Disable part-focused filtering")
    args = parser.parse_args()

    kg_path = args.kg_path
    supplier_name = args.supplier

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        script_path = Path(__file__).resolve()
        cache_dir = _default_cache_dir(script_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_filename = _compute_cache_name(kg_path, supplier_name)
        output_path = cache_dir / output_filename

    # Run analysis
    result = analyze_parts_structured(
        kg_json_path=kg_path,
        supplier_name=supplier_name,
        save_to_file=True,
        output_path=str(output_path),
        use_part_filter=not args.no_filter,
        canonicalize=not args.no_canonicalize,
    )

    # Print a brief summary for convenience
    summary = {
        "output_file": str(output_path),
        "supplier": result.get("supplier_name"),
        "total_parts": result.get("parts_summary", {}).get("total_parts_count"),
        "canonical_parts": len(result.get("canonical_parts", [])) if isinstance(result.get("canonical_parts"), list) else 0,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


