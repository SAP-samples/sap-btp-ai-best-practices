"""Bound database-driver buffers while preserving the caller's transaction."""

from itertools import islice


def insert_chunks(connection, statement, rows, *, chunk_size=500):
    """Execute lazy bound row mappings in small chunks in the caller's open transaction.

    Inputs are a SQLAlchemy connection, parameterized insert statement and iterable
    of row dictionaries. Returns no value. Exceptions propagate so the caller rolls
    back the entire publication. Large ECN NCLOB rows should use chunk_size=1 because
    HANA executemany buffers can scale with the widest payload times the batch size.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    source = iter(rows)
    while batch := list(islice(source, chunk_size)):
        connection.execute(statement, batch)
