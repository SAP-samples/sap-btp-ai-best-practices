"""Versioned HANA lifecycle records; no runtime dependency on local history workbooks."""
import hashlib
import json
from datetime import datetime, timezone
from .schema import ensure_lifecycle_schema
from .context import admissible_history
from ..workspace.schema import decode_json, transaction
from ...models.workspace import RevisionConflict
import pandas as pd


class LifecycleStore:
    """Persist immutable normalized datasets and a separately activated context version."""

    def __init__(self, backend, db_path=None):
        """Initialize HANA, or an explicitly supplied isolated test backend/path."""
        self.backend,self.db_path = backend,db_path
        ensure_lifecycle_schema(backend,db_path)

    def import_dataset(self,dataset_id,rows,metadata):
        """Atomically import a nonempty dataset; a version cannot refer to changed content."""
        if not rows or not dataset_id or len(dataset_id)>128:
            raise ValueError('A dataset requires a valid ID and positive observed lifetimes')
        content_hash = hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
        metadata = {**metadata,'dataset_id':dataset_id,'row_count':len(rows),'content_hash':content_hash,
                    'imported_at':datetime.now(timezone.utc).isoformat()}
        with transaction(self.backend,self.db_path) as cursor:
            cursor.execute('SELECT metadata FROM RECEIVABLES_LIFECYCLE_DATASETS WHERE dataset_id = ?', (dataset_id,))
            if saved := cursor.fetchone():
                previous = decode_json(saved[0])
                if previous['content_hash'] != content_hash or previous.get('source_hash') != metadata.get('source_hash'):
                    raise RevisionConflict('Dataset version already identifies different content')
                return previous
            cursor.execute('INSERT INTO RECEIVABLES_LIFECYCLE_DATASETS (dataset_id,metadata) VALUES (?, ?)',(dataset_id,json.dumps(metadata)))
            entries = [(hashlib.sha256(f'{dataset_id}:{index}'.encode()).hexdigest(),dataset_id,row['invoice_key'],json.dumps(row))
                       for index,row in enumerate(rows)]
            for start in range(0,len(entries),500):
                cursor.executemany('INSERT INTO RECEIVABLES_LIFECYCLE_ROWS (record_id,dataset_id,invoice_key,payload) VALUES (?, ?, ?, ?)',entries[start:start+500])
        return metadata

    def get_dataset(self,dataset_id):
        """Return metadata for one named immutable version or raise LookupError."""
        with transaction(self.backend,self.db_path) as cursor:
            cursor.execute('SELECT metadata FROM RECEIVABLES_LIFECYCLE_DATASETS WHERE dataset_id = ?', (dataset_id,))
            row = cursor.fetchone()
            if row is None: raise LookupError('Lifecycle dataset not found')
            return decode_json(row[0])

    def activate(self,dataset_id):
        """Select one reviewed nonempty version explicitly for future preparations."""
        metadata = self.get_dataset(dataset_id)
        if not metadata['row_count']: raise ValueError('Cannot activate an empty dataset')
        with transaction(self.backend,self.db_path) as cursor:
            cursor.execute('UPDATE RECEIVABLES_LIFECYCLE_ACTIVE SET dataset_id = ? WHERE slot = 1',(dataset_id,))
            if cursor.rowcount == 0:
                cursor.execute('INSERT INTO RECEIVABLES_LIFECYCLE_ACTIVE (slot,dataset_id) VALUES (1, ?)',(dataset_id,))

    def active_dataset(self):
        """Resolve the explicitly activated version; never silently fall back to a file."""
        with transaction(self.backend,self.db_path) as cursor:
            cursor.execute('SELECT dataset_id FROM RECEIVABLES_LIFECYCLE_ACTIVE WHERE slot = 1')
            row = cursor.fetchone()
            if row is None: raise LookupError('No lifecycle history dataset is active')
            dataset_id = row[0]
        return self.get_dataset(dataset_id)

    def list_datasets(self):
        """Return every dataset's metadata, newest import first, flagging the active one."""
        with transaction(self.backend,self.db_path) as cursor:
            cursor.execute('SELECT dataset_id FROM RECEIVABLES_LIFECYCLE_ACTIVE WHERE slot = 1')
            row = cursor.fetchone()
            active = row[0] if row else None
            cursor.execute('SELECT metadata FROM RECEIVABLES_LIFECYCLE_DATASETS')
            items = [decode_json(item[0]) for item in cursor.fetchall()]
        items.sort(key=lambda item: item.get('imported_at',''),reverse=True)
        return [{**item,'active':item['dataset_id'] == active} for item in items]

    def records(self,dataset_id):
        """Load normalized context rows for a verified saved dataset version."""
        self.get_dataset(dataset_id)
        with transaction(self.backend,self.db_path) as cursor:
            return self._payloads(cursor,dataset_id)

    def rows(self,dataset_id,limit,offset):
        """Return one page of normalized rows plus the dataset's total row count."""
        metadata = self.get_dataset(dataset_id)
        with transaction(self.backend,self.db_path) as cursor:
            items = self._payloads(cursor,dataset_id,f' LIMIT {int(limit)} OFFSET {int(offset)}')
        return {'dataset_id':dataset_id,'total':metadata['row_count'],'limit':int(limit),'offset':int(offset),'items':items}

    def _payloads(self,cursor,dataset_id,page_sql=''):
        """Read row payloads in record order; page_sql is a validated integer LIMIT/OFFSET clause."""
        if not self.backend.is_hana:
            cursor.execute('SELECT payload FROM RECEIVABLES_LIFECYCLE_ROWS WHERE dataset_id = ? ORDER BY record_id'+page_sql,(dataset_id,))
            return [decode_json(row[0]) for row in cursor.fetchall()]
        # Reuse the workspace's bounded scalar-read pattern to avoid one WAN
        # LOB fetch per example; oversized records are still retrieved in full.
        cursor.execute('SELECT TO_NVARCHAR(SUBSTRING(payload,1,5000)), LENGTH(payload), record_id '
                       'FROM RECEIVABLES_LIFECYCLE_ROWS WHERE dataset_id = ? ORDER BY record_id'+page_sql,(dataset_id,))
        records = cursor.fetchall()
        result = []
        for text,length,record_id in records:
            if length > 5000:
                cursor.execute('SELECT payload FROM RECEIVABLES_LIFECYCLE_ROWS WHERE record_id = ?',(record_id,))
                text = cursor.fetchone()[0]
            result.append(decode_json(text))
        return result

    def context(self,prediction_at,excluded_invoice_keys,dataset_id=None,context_policy="chronological"):
        """Return fixed-reference or chronological examples, always excluding query identities.

        Fixed-reference inference deliberately ignores scenario dates. Chronological
        evaluation remains the default and must never inherit the demo policy.
        """
        if context_policy not in ('chronological','fixed_reference'):
            raise ValueError('Unknown historical context policy')
        metadata = self.get_dataset(dataset_id) if dataset_id else self.active_dataset()
        rows = pd.DataFrame(self.records(metadata['dataset_id']))
        if context_policy == 'fixed_reference':
            # The imported reference already contains only positive, unique observed
            # lifetimes. Exclude any queried business identities from these examples.
            admitted = rows.loc[~rows['invoice_key'].isin(excluded_invoice_keys)].copy()
        else:
            admitted = admissible_history(rows,prediction_at,excluded_invoice_keys)
        return admitted,dict(dataset_id=metadata['dataset_id'],content_hash=metadata['content_hash'],
            source_name=metadata.get('source_name'),context_policy=context_policy,
            scenario_as_of=str(prediction_at),
            effective_as_of=str(prediction_at) if context_policy == 'chronological' else None,
            target_event=metadata.get('target_event'),admitted_rows=len(admitted),
            excluded_rows=len(rows)-len(admitted),availability=metadata.get('outcome_availability'))
