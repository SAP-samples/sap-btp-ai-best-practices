"""HANA bulk imports must use driver batching rather than one remote call per row."""
import unittest
from unittest.mock import Mock
from app.services.database.backend import DictCursor


class DatabaseBatchTests(unittest.TestCase):
    """Exercise the DB adapter's network-call boundary for multi-row inserts."""

    def test_batch_is_sent_to_driver_once(self):
        """A thousand inputs should not cause a thousand network round trips."""
        raw=Mock(description=None)
        wrapped=DictCursor(raw,Mock())
        values=[(index,'value') for index in range(1000)]
        wrapped.executemany('INSERT INTO EXAMPLE (ID, VALUE) VALUES (?, ?)',values)
        raw.executemany.assert_called_once_with('INSERT INTO EXAMPLE (ID, VALUE) VALUES (?, ?)',values)
        raw.execute.assert_not_called()
