"""HANA experiments and resumable row predictions for lifetime validation."""
import hashlib
import json
from datetime import datetime,timezone
from ..workspace.schema import transaction,decode_json


class EvaluationStore:
    """Persist immutable experiment configuration and independently resumable predictions."""

    def __init__(self,backend,db_path=None):
        """Initialize experiment tables automatically in HANA or explicit test SQLite."""
        self.backend,self.db_path=backend,db_path
        with backend.get_connection(db_path) as conn:
            cursor=backend.cursor(conn);text='NCLOB' if backend.is_hana else 'TEXT'
            for name,ddl in [
                ('RECEIVABLES_LIFETIME_EXPERIMENTS',f'CREATE TABLE RECEIVABLES_LIFETIME_EXPERIMENTS (experiment_id NVARCHAR(64) PRIMARY KEY, metadata {text}, metrics {text})'),
                ('RECEIVABLES_LIFETIME_PREDICTIONS',f'CREATE TABLE RECEIVABLES_LIFETIME_PREDICTIONS (experiment_id NVARCHAR(64), row_id NVARCHAR(128), payload {text}, PRIMARY KEY (experiment_id,row_id))')]:
                if not backend.table_exists(conn,name):cursor.execute(ddl)
            backend.commit(conn)

    def save_experiment(self,metadata):
        """Content-address experiment inputs so unchanged runs can safely resume."""
        identity=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()
        with transaction(self.backend,self.db_path) as cursor:
            cursor.execute('SELECT metadata FROM RECEIVABLES_LIFETIME_EXPERIMENTS WHERE experiment_id = ?',(identity,))
            if not cursor.fetchone():
                cursor.execute('INSERT INTO RECEIVABLES_LIFETIME_EXPERIMENTS (experiment_id,metadata) VALUES (?, ?)',
                    (identity,json.dumps({**metadata,'created_at':datetime.now(timezone.utc).isoformat()})))
        return identity

    def load(self,experiment_id):
        """Read a known experiment or reject an invalid resume identity."""
        with transaction(self.backend,self.db_path) as cursor:
            cursor.execute('SELECT metadata,metrics FROM RECEIVABLES_LIFETIME_EXPERIMENTS WHERE experiment_id = ?',(experiment_id,))
            row=cursor.fetchone()
            if row is None:raise LookupError('Experiment not found')
            return dict(metadata=decode_json(row[0]),metrics=decode_json(row[1]) if row[1] else None)

    def predictions(self,experiment_id):
        """Return persisted measured rows; unavailable calls are represented explicitly."""
        self.load(experiment_id)
        with transaction(self.backend,self.db_path) as cursor:
            cursor.execute('SELECT payload FROM RECEIVABLES_LIFETIME_PREDICTIONS WHERE experiment_id = ?',(experiment_id,))
            return [decode_json(row[0]) for row in cursor.fetchall()]

    def save_predictions(self,experiment_id,rows):
        """Commit a completed batch without overwriting earlier observed responses."""
        with transaction(self.backend,self.db_path) as cursor:
            for row in rows:
                cursor.execute('SELECT row_id FROM RECEIVABLES_LIFETIME_PREDICTIONS WHERE experiment_id = ? AND row_id = ?',(experiment_id,row['row_id']))
                if cursor.fetchone():continue
                cursor.execute('INSERT INTO RECEIVABLES_LIFETIME_PREDICTIONS (experiment_id,row_id,payload) VALUES (?, ?, ?)',(experiment_id,row['row_id'],json.dumps(row)))

    def save_metrics(self,experiment_id,metrics):
        """Save reproducible aggregate metrics after individual responses are durable."""
        with transaction(self.backend,self.db_path) as cursor:
            cursor.execute('UPDATE RECEIVABLES_LIFETIME_EXPERIMENTS SET metrics = ? WHERE experiment_id = ?',(json.dumps(metrics),experiment_id))
