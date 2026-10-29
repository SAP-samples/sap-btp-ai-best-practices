"""Ingest lifecycle workbooks using a fixed reconciliation-file target definition."""
import hashlib
import io
import json
from pathlib import Path
import pandas as pd
import numpy as np
from .context import invoice_key


def normalize_history(frame):
    """Return valid unique observed lifetimes and metadata describing all exclusions."""
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    for key in ('Company Code','Customer','Invoice Reference'):
        if key not in frame: frame[key] = None
        frame[key] = frame[key].map(lambda value: str(int(value)) if isinstance(value,float) and np.isfinite(value) and value.is_integer() else value)
    required = ['Summary File Date (UTC)','Reconciliation File Date (UTC)']
    if any(key not in frame for key in required):
        raise ValueError('Lifecycle history requires Summary File Date and Reconciliation File Date columns; target fallback is disabled')
    frame['credit_start'] = pd.to_datetime(frame[required[0]],errors='coerce',utc=True)
    frame['credit_release'] = pd.to_datetime(frame[required[1]],errors='coerce',utc=True)
    frame['credit_duration_days'] = (frame['credit_release']-frame['credit_start']).dt.total_seconds()/86400
    frame['outcome_known_at'] = frame['credit_release']
    frame['invoice_key'] = frame.apply(invoice_key,axis=1)
    valid = frame['credit_duration_days'].gt(0) & np.isfinite(frame['credit_duration_days'])
    missing_ids = frame['invoice_key'].isna()
    # Repeated business identities are ambiguous: remove every occurrence rather than
    # guessing which lifecycle belongs to a new query or treating copies as examples.
    ambiguous = frame['invoice_key'].duplicated(keep=False) & ~missing_ids
    clean = frame[valid & ~missing_ids & ~ambiguous].copy()
    fields=['Company Code','Customer','Invoice Reference','Document Number','Fiscal Year',
            'PROGRAMA','Funding Currency','Currency','ORIGINAL CURRENCY','Purchase Price','Amount',
            'Issuance date','ISSUANCE DATE','Due Date','Summary File Date (UTC)',
            'Reconciliation File Date (UTC)','credit_start','credit_release','credit_duration_days',
            'outcome_known_at','invoice_key']
    records = json.loads(clean[[key for key in fields if key in clean]].to_json(orient='records',date_format='iso'))
    metadata = dict(raw_rows=len(frame),valid_rows=len(clean),invalid_outcome_rows=int((~valid).sum()),
                    missing_identity_rows=int(missing_ids.sum()),ambiguous_identity_rows=int(ambiguous.sum()),
                    target_event='summary_file_date_to_reconciliation_file_date',
                    outcome_availability='Reconciliation-file timestamp proxy; actual ingestion availability is unknown',
                    credit_start_min=str(clean['credit_start'].min()),credit_start_max=str(clean['credit_start'].max()),
                    duration_median_days=float(clean['credit_duration_days'].median()) if len(clean) else None,
                    feature_coverage={key:round(float(clean[key].notna().mean()),4) if len(clean) else 0 for key in
                                      ('Company Code','Customer','Invoice Reference','Purchase Price','Amount','Due Date') if key in clean})
    return records,metadata


def import_workbook_bytes(store,content,source_name,dataset_id,sheet_name=0,reference_purpose="chronological_history"):
    """Normalize one uploaded history workbook and persist it as an immutable HANA dataset.

    Args:
        store: LifecycleStore receiving the normalized rows.
        content: Raw .xlsx bytes (API upload or file read by the CLI).
        source_name: Original file name, kept as provenance only.
        dataset_id: Version identifier; re-importing identical bytes is idempotent.
        sheet_name: Worksheet to read (first sheet by default).
        reference_purpose: 'fixed_reference' for inference context, otherwise
            'chronological_history' for evaluation.

    Returns:
        Saved dataset metadata (row counts, hashes, coverage, date range).
    """
    with pd.ExcelFile(io.BytesIO(content)) as workbook:
        frame = pd.read_excel(workbook,sheet_name=sheet_name)
    rows,metadata = normalize_history(frame)
    metadata.update(reference_purpose=reference_purpose,source_name=source_name,source_hash=hashlib.sha256(content).hexdigest())
    return store.import_dataset(dataset_id,rows,metadata)


def import_workbook(store,input_path,dataset_id,sheet_name=0,reference_purpose="chronological_history"):
    """Read one local workbook (CLI path) and delegate to import_workbook_bytes."""
    path = Path(input_path)
    return import_workbook_bytes(store,path.read_bytes(),path.name,dataset_id,sheet_name,reference_purpose)


def compare_datasets(store,left,right):
    """Return source metadata and canonical identity overlap without exposing invoice rows."""
    a,b = store.get_dataset(left),store.get_dataset(right)
    keys_a,keys_b = {row['invoice_key'] for row in store.records(left)}, {row['invoice_key'] for row in store.records(right)}
    return {'left':a,'right':b,'shared_invoice_keys':len(keys_a&keys_b),
            'left_only':len(keys_a-keys_b),'right_only':len(keys_b-keys_a),
            'same_raw_hash':a.get('source_hash') == b.get('source_hash'),
            'same_normalized_content':a.get('content_hash') == b.get('content_hash')}
