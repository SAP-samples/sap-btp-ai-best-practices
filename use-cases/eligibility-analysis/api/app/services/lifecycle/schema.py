"""HANA lifecycle dataset, row, and activation contracts."""
from ..workspace.schema import ensure_schema

LIFECYCLE_TABLES = {
    'RECEIVABLES_LIFECYCLE_DATASETS': {'dataset_id':'NVARCHAR(128) PRIMARY KEY','metadata':'NCLOB NOT NULL'},
    'RECEIVABLES_LIFECYCLE_ROWS': {'record_id':'NVARCHAR(64) PRIMARY KEY','dataset_id':'NVARCHAR(128) NOT NULL',
                                'invoice_key':'NVARCHAR(64) NOT NULL','payload':'NCLOB NOT NULL'},
    'RECEIVABLES_LIFECYCLE_ACTIVE': {'slot':'INTEGER PRIMARY KEY','dataset_id':'NVARCHAR(128) NOT NULL'},
}


def ensure_lifecycle_schema(backend, db_path=None):
    """Initialize and validate lifecycle tables using the existing HANA dialect guard."""
    ensure_schema(backend,db_path,LIFECYCLE_TABLES)
