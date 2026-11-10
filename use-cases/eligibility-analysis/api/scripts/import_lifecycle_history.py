"""Ingest, compare, or explicitly activate HANA lifecycle history.

The same import is available over HTTP at POST /api/workspace/lifecycle/datasets.

Examples (repository root):
  PYTHONPATH=api .venv/bin/python api/scripts/import_lifecycle_history.py --input <path-to-history.xlsx> --dataset-id reference-runtime-20260907
  PYTHONPATH=api .venv/bin/python api/scripts/import_lifecycle_history.py --input <path-to-history.xlsx> --dataset-id invoice-lifetime-reference-v1 --fixed-reference
  PYTHONPATH=api .venv/bin/python api/scripts/import_lifecycle_history.py --activate invoice-lifetime-reference-v1
  PYTHONPATH=api .venv/bin/python api/scripts/import_lifecycle_history.py --compare reference-runtime-20260907 reference-evaluation-20260907
  PYTHONPATH=api .venv/bin/python api/scripts/import_lifecycle_history.py --activate reference-runtime-20260907
"""
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from app.services.database.backend import get_backend
from app.services.lifecycle.store import LifecycleStore
from app.services.lifecycle.importer import import_workbook, compare_datasets


def main():
    """Run one explicit HANA import/compare/activation operation with progress."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input')
    group.add_argument('--compare',nargs=2)
    group.add_argument('--activate')
    parser.add_argument('--dataset-id')
    parser.add_argument('--fixed-reference',action='store_true',help='Label an imported dataset as a reusable inference reference')
    parser.add_argument('--release-event',choices=['reconciliation_file_date'],default='reconciliation_file_date')
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1]/'.env')
    backend = get_backend()
    if not backend.is_hana: parser.error('Configure HANA credentials; lifecycle history is not stored locally')
    with tqdm(total=2,desc='Lifecycle history',unit='stage') as progress:
        store = LifecycleStore(backend); progress.update(1)
        if args.input:
            if not args.dataset_id: parser.error('--input requires --dataset-id')
            result = import_workbook(store,args.input,args.dataset_id,reference_purpose='fixed_reference' if args.fixed_reference else 'chronological_history')
        elif args.compare: result = compare_datasets(store,*args.compare)
        else:
            store.activate(args.activate); result = store.active_dataset()
        progress.update(1)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
