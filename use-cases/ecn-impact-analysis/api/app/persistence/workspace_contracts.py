"""Small HANA contracts for explicit generation batches and snapshot-bound reviews."""

from app.persistence.table_contracts import ForeignKeyContract, TableContract, _col


def workspace_contracts(names):
    """Return additive tables after their existing ECN/job dependencies."""
    return (
        TableContract(
            key="generation_batch",
            name=names.generation_batch,
            columns=(
                _col("BATCH_ID", "NVARCHAR", length=36),
                _col("SELECTION_JSON", "NCLOB"),
                _col("CREATED_AT", "TIMESTAMP"),
                _col("CANCELLED", "TINYINT"),
            ),
            primary_key=("BATCH_ID",),
            unique_constraints=(),
            foreign_keys=(),
        ),
        TableContract(
            key="batch_job",
            name=names.batch_job,
            columns=(
                _col("JOB_ID", "NVARCHAR", length=36),
                _col("BATCH_ID", "NVARCHAR", length=36),
            ),
            primary_key=("JOB_ID",),
            unique_constraints=(),
            foreign_keys=(
                ForeignKeyContract(("JOB_ID",), "analysis_job", ("JOB_ID",)),
                ForeignKeyContract(("BATCH_ID",), "generation_batch", ("BATCH_ID",)),
            ),
        ),
        TableContract(
            key="review",
            name=names.review,
            columns=(
                _col("REVIEW_ID", "NVARCHAR", length=36),
                _col("ECN_ROW_ID", "NVARCHAR", length=36),
                _col("EVIDENCE_HASH", "NVARCHAR", length=64),
                _col("SNAPSHOT_VERSION", "NVARCHAR", length=40),
                _col("REVIEWER", "NVARCHAR", length=200),
                _col("ACTOR_ID", "NVARCHAR", length=64),
                _col("OUTCOME", "NVARCHAR", length=40),
                _col("COMMENTS", "NCLOB"),
                _col("CREATED_AT", "TIMESTAMP"),
            ),
            primary_key=("REVIEW_ID",),
            unique_constraints=(),
            foreign_keys=(ForeignKeyContract(("ECN_ROW_ID",), "ecn", ("ECN_ROW_ID",)),),
        ),
    )
