from __future__ import annotations

import sys
import types

import pytest

from app.nbo import hana as hana_runtime
from app.nbo.data_loader import DataStore
from app.nbo.hana import HanaConfigurationError, HanaSettings


def test_hana_settings_requires_all_fields(monkeypatch) -> None:
    for env_name in (
        "hana_address",
        "hana_port",
        "hana_user",
        "hana_password",
        "hana_encrypt",
    ):
        monkeypatch.delenv(env_name, raising=False)

    with pytest.raises(HanaConfigurationError) as exc_info:
        HanaSettings.from_env()

    assert "hana_address" in str(exc_info.value)
    assert "hana_password" in str(exc_info.value)


def test_hana_settings_parses_valid_env(monkeypatch) -> None:
    monkeypatch.setenv("hana_address", "hana.example.internal")
    monkeypatch.setenv("hana_port", "39015")
    monkeypatch.setenv("hana_user", "COA_USER")
    monkeypatch.setenv("hana_password", "top-secret")
    monkeypatch.setenv("hana_encrypt", "true")

    settings = HanaSettings.from_env()

    assert settings.address == "hana.example.internal"
    assert settings.port == 39015
    assert settings.user == "COA_USER"
    assert settings.password == "top-secret"
    assert settings.encrypt is True


def test_hana_connect_assembles_expected_driver_kwargs(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_connect(**kwargs):
        captured.update(kwargs)
        return object()

    fake_dbapi = types.SimpleNamespace(connect=fake_connect)
    fake_module = types.ModuleType("hdbcli")
    fake_module.dbapi = fake_dbapi
    monkeypatch.setitem(sys.modules, "hdbcli", fake_module)

    settings = HanaSettings(
        address="hana.local",
        port=39017,
        user="COA_USER",
        password="secret",
        encrypt=False,
    )

    hana_runtime.connect(settings)

    assert captured == {
        "address": "hana.local",
        "port": 39017,
        "user": "COA_USER",
        "password": "secret",
        "encrypt": False,
    }


def test_datastore_surfaces_clear_hana_configuration_errors(monkeypatch) -> None:
    def fail():
        raise HanaConfigurationError(
            "Missing required HANA configuration. Set these environment variables: hana_address"
        )

    monkeypatch.setattr(hana_runtime, "load_runtime_datasets", fail)

    with pytest.raises(HanaConfigurationError) as exc_info:
        DataStore()

    assert "Missing required HANA configuration" in str(exc_info.value)
    assert "hana_address" in str(exc_info.value)
