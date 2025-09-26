"""
Utility helpers to store and search text embeddings in SAP HANA native vector store.

This module provides minimal wrappers for:
- establishing a connection using hana_ml.ConnectionContext
- creating a temporary table with REAL_VECTOR column
- inserting rows (text, optional metadata, vector)
- running vector search using COSINE_SIMILARITY or L2DISTANCE

Environment variables expected:
- HANA_ADDRESS, HANA_PORT, HANA_USER, HANA_PASSWORD

These functions are kept synchronous to match FastAPI's default sync route function
in this project.
"""

import json
import math
import os
import uuid
from typing import Iterable, List, Optional, Sequence, Tuple

from hana_ml import ConnectionContext  # type: ignore


def connect() -> ConnectionContext:
    """Create a ConnectionContext from environment variables.

    Returns a live connection; caller is responsible to close via cc.connection.close().
    """

    cc = ConnectionContext(
        address=os.environ.get("HANA_ADDRESS"),
        port=os.environ.get("HANA_PORT"),
        user=os.environ.get("HANA_USER"),
        password=os.environ.get("HANA_PASSWORD"),
        encrypt=True,
    )
    return cc


def create_temp_table(cc: ConnectionContext, table_name: Optional[str] = None) -> str:
    """Create a temporary table to store text, metadata and vectors.

    Returns the created table name. Table schema:
      MY_TEXT NCLOB,
      MY_METADATA NCLOB,
      MY_VECTOR REAL_VECTOR
    """

    name = table_name or f"VECTORS_{uuid.uuid4().hex[:12].upper()}"
    cursor = cc.connection.cursor()
    try:
        cursor.execute(
            f"""
            CREATE LOCAL TEMPORARY COLUMN TABLE #{name} (
                MY_TEXT NCLOB,
                MY_METADATA NCLOB,
                MY_VECTOR REAL_VECTOR
            )
            """
        )
        return f"#{name}"
    finally:
        cursor.close()


def drop_table(cc: ConnectionContext, table_name: str) -> None:
    """Drop a (temporary) table if it exists."""

    cursor = cc.connection.cursor()
    try:
        try:
            cursor.execute(f"DROP TABLE {table_name}")
        except Exception:
            # Best-effort; ignore if not exists
            pass
    finally:
        cursor.close()


def insert_rows(
    cc: ConnectionContext,
    table_name: str,
    rows: Sequence[Tuple[str, Optional[dict], Sequence[float]]],
    batch_size: int = 200,
) -> None:
    """Insert rows into HANA vector table.

    Each row is (text, metadata_dict_or_None, vector_floats). Vectors are passed
    as JSON array converted with TO_REAL_VECTOR in SQL.
    """

    cursor = cc.connection.cursor()
    try:
        sql_insert = f"""
            INSERT INTO {table_name}
            (MY_TEXT, MY_METADATA, MY_VECTOR)
            VALUES (?, ?, TO_REAL_VECTOR(?))
        """

        total_batches = math.ceil(len(rows) / batch_size) if rows else 0
        for i in range(total_batches):
            batch = rows[i * batch_size : (i + 1) * batch_size]
            payload = [
                (
                    text,
                    json.dumps(metadata) if metadata is not None else None,
                    json.dumps(list(vector)),
                )
                for (text, metadata, vector) in batch
            ]
            cursor.executemany(sql_insert, payload)
            cc.connection.commit()
    finally:
        cursor.close()


def search_top_k(
    cc: ConnectionContext,
    table_name: str,
    query_vector: Sequence[float],
    k: int = 5,
    metric: str = "COSINE_SIMILARITY",
) -> List[Tuple[str, Optional[str]]]:
    """Return top-k rows by similarity.

    metric can be COSINE_SIMILARITY or L2DISTANCE. For COSINE, results are DESC.
    Returns list of (MY_TEXT, MY_METADATA_JSON) as strings.
    """

    cursor = cc.connection.cursor()
    try:
        sort_order = "DESC" if metric != "L2DISTANCE" else "ASC"
        qv = json.dumps(list(query_vector))
        sql_query = f"""
            SELECT TOP {int(k)} MY_TEXT, MY_METADATA
            FROM {table_name}
            ORDER BY {metric}(MY_VECTOR, TO_REAL_VECTOR('{qv}')) {sort_order}
        """
        cursor.execute(sql_query)
        return [(r[0], r[1]) for r in cursor.fetchall()]
    finally:
        cursor.close()


