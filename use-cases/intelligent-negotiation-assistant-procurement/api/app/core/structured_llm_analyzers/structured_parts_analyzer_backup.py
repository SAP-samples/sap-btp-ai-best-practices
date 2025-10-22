"""
Backup of Structured Parts Analyzer - original implementation

This file was created to preserve the current behavior before introducing
canonicalization and alignment per the engineering spec.
"""
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# Add parent directories for imports
current_file = Path(__file__).resolve()
kg_creation_dir = current_file.parent.parent.parent.parent
sys.path.append(str(kg_creation_dir))

from llm.factory import create_llm  # noqa: E402
from .business_context import format_prompt_with_business_context  # noqa: E402


def filter_part_related_data(kg_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter KG to only include part nodes and their direct connections.
    """
    nodes = kg_data.get('nodes', [])
    relationships = kg_data.get('relationships', [])

    part_node_ids = {n['id'] for n in nodes if n.get('id', '').startswith('part:')}
    if not part_node_ids:
        return kg_data

    part_relationships = []
    connected_node_ids = set()
    for rel in relationships:
        source = rel.get('source')
        target = rel.get('target')
        if source in part_node_ids or target in part_node_ids:
            part_relationships.append(rel)
            connected_node_ids.add(source)
            connected_node_ids.add(target)

    relevant_nodes = [n for n in nodes if n['id'] in connected_node_ids]
    return {
        'nodes': relevant_nodes,
        'relationships': part_relationships,
        'metadata': kg_data.get('metadata', {}),
        'export_metadata': kg_data.get('export_metadata', {})
    }


def analyze_parts_structured(
    kg_json_path: str,
    supplier_name: Optional[str] = None,
    save_to_file: bool = False,
    output_path: Optional[str] = None,
    use_part_filter: bool = True
) -> Dict[str, Any]:
    """
    Original analyzer implementation (backup).
    """
    with open(kg_json_path, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)

    if use_part_filter:
        kg_data = filter_part_related_data(kg_data)

    base_prompt = "Analyze the knowledge graph and extract comprehensive parts information.\n\n" + \
        json.dumps(kg_data, indent=2)

    prompt = format_prompt_with_business_context(base_prompt)
    llm = create_llm(model_name="gemini-2.5-pro", temperature=0.0)
    try:
        response = llm.invoke(prompt)
        content = getattr(response, 'content', str(response))
        if isinstance(content, list):
            content = content[0] if content else ""
        content = str(content).strip()
        if content.startswith('```json'):
            content = content[7:]
        elif content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        result = json.loads(content)
    except Exception:
        result = {
            "parts": [],
            "part_families": [],
            "supplier_capabilities": {},
            "parts_summary": {
                "total_parts_count": 0,
                "categories": [],
                "price_range": {"min": 0, "max": 0, "currency": "EUR"},
                "key_certifications": [],
                "production_readiness": "Unable to assess"
            }
        }

    result['supplier_name'] = supplier_name or "Unknown Supplier"
    if save_to_file and output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def main():
    import argparse
    from datetime import datetime
    parser = argparse.ArgumentParser(description='Analyze parts from KG (backup)')
    parser.add_argument('kg_path', help='Path to KG JSON file')
    parser.add_argument('--supplier', help='Supplier name', default=None)
    parser.add_argument('--output', help='Output JSON file path', default=None)
    args = parser.parse_args()
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        supplier_suffix = f"_{args.supplier.replace(' ', '_')}" if args.supplier else ""
        args.output = f"output/llm_analysis/parts_analysis_{timestamp}{supplier_suffix}.json"
    result = analyze_parts_structured(
        kg_json_path=args.kg_path,
        supplier_name=args.supplier,
        save_to_file=True,
        output_path=args.output
    )
    print(json.dumps({
        "supplier": result.get('supplier_name'),
        "total_parts": result.get('parts_summary', {}).get('total_parts_count', 0)
    }, indent=2))


if __name__ == "__main__":
    main()

