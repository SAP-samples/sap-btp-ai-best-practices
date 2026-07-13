from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.utils import hana as hana_module
from app.utils.hana import HANAConnection


class _FakeHANAConnection:
    def __init__(self) -> None:
        self.commit_calls = 0
        self.rollback_calls = 0
        self.close_calls = 0
        self.autocommit_values: list[bool] = []

    def setautocommit(self, value: bool) -> None:
        """Capture requested driver autocommit values for connection tests."""
        self.autocommit_values.append(bool(value))

    def commit(self) -> None:
        self.commit_calls += 1

    def rollback(self) -> None:
        self.rollback_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def _configured_connection(monkeypatch: pytest.MonkeyPatch, fake_connection: _FakeHANAConnection) -> HANAConnection:
    connection = HANAConnection()
    connection.address = "hana.example.invalid"
    connection.port = "30015"
    connection.user = "pytest"
    connection.password = "secret"

    fake_dbapi = SimpleNamespace(connect=lambda **kwargs: fake_connection)
    monkeypatch.setattr(hana_module.HANAConnection, "_require_dbapi", lambda self: fake_dbapi)
    return connection


def test_hana_connection_disables_driver_autocommit(monkeypatch: pytest.MonkeyPatch) -> None:
    """HANA connections should use explicit transaction commits, not driver autocommit."""
    fake_connection = _FakeHANAConnection()
    connection = _configured_connection(monkeypatch, fake_connection)

    connection.connect()

    assert fake_connection.autocommit_values == [False]


def test_hana_connection_nested_transaction_commits_only_once(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_connection = _FakeHANAConnection()
    connection = _configured_connection(monkeypatch, fake_connection)

    with connection.transaction():
        with connection.transaction():
            pass

    assert fake_connection.commit_calls == 1
    assert fake_connection.rollback_calls == 0
    assert connection._transaction_depth == 0


def test_hana_connection_nested_transaction_rolls_back_and_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_connection = _FakeHANAConnection()
    connection = _configured_connection(monkeypatch, fake_connection)

    with pytest.raises(RuntimeError):
        with connection.transaction():
            with connection.transaction():
                raise RuntimeError("boom")

    with connection.transaction():
        pass

    assert fake_connection.commit_calls == 1
    assert fake_connection.rollback_calls == 1
    assert connection._transaction_depth == 0
