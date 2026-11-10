"""Tests for RPT-1 deployment discovery by foundation-model name.

Run from the repository root:
    cd api && PYTHONPATH=. ../.venv/bin/python -m unittest tests.test_rpt1_model_resolution
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import unittest
from pathlib import Path


def _load_module():
    """Load api/rpt1/rpt1_client.py as a fresh module so its cache starts empty."""
    module_path = Path(__file__).resolve().parents[1] / "rpt1" / "rpt1_client.py"
    spec = importlib.util.spec_from_file_location("rpt1_client_resolution_tests", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Response:
    """Minimal requests.Response stand-in."""

    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200
        self.ok = True
        self.text = json.dumps(payload)

    def json(self):
        """Return the fake JSON body."""
        return self._payload


class _Session:
    """Fake session routing AI Core list endpoints to canned resources."""

    def __init__(self, configurations, deployments):
        self.configurations = configurations
        self.deployments = deployments
        self.calls = []

    def request(self, method, url, **kwargs):
        """Serve GET /lm/configurations and GET /lm/deployments."""
        self.calls.append((method, url, kwargs.get("params")))
        if url.endswith("/v2/lm/configurations"):
            return _Response({"resources": self.configurations})
        if url.endswith("/v2/lm/deployments"):
            return _Response({"resources": self.deployments})
        raise AssertionError(f"Unexpected request {method} {url}")


def _configuration(config_id, name, version="1", as_mapping=False):
    """Build an AI Core configuration with model parameter bindings."""
    bindings = {"modelName": name, "modelVersion": version}
    if not as_mapping:
        bindings = [{"key": key, "value": value} for key, value in bindings.items()]
    return {"id": config_id, "parameterBindings": bindings}


def _deployment(deployment_id, config_id, status="RUNNING", url=True):
    """Build an AI Core deployment record."""
    record = {"id": deployment_id, "configurationId": config_id, "status": status}
    if url:
        record["deploymentUrl"] = f"https://inference.example.invalid/{deployment_id}/"
    return record


class TestRPT1ModelResolution(unittest.TestCase):
    """Deployment discovery must be exact and never guess between candidates."""

    def setUp(self):
        """Load a fresh module per test so the process cache does not leak."""
        self.module = _load_module()

    def _client(self, session, version=None):
        """Create a client with a preset token so no OAuth call happens."""
        client = self.module.RPT1Client(
            aicore_base_url="https://api.example.invalid/v2",
            aicore_auth_url="https://auth.example.invalid",
            client_id="client",
            client_secret="secret",
            model_name="sap-rpt-1-small",
            model_version=version,
            session=session,
            max_retries=0,
        )
        client._access_token = "token"
        client._access_token_expiry = time.time() + 3600
        return client

    def test_name_only_match_returns_running_deployment(self):
        """Without a version any configuration of the named model qualifies."""
        session = _Session(
            [_configuration("c1", "sap-rpt-1-small"), _configuration("c2", "sap-rpt-1-large")],
            [_deployment("d1", "c1"), _deployment("d2", "c2"), _deployment("d3", "c1", status="STOPPED")],
        )
        client = self._client(session)
        self.assertEqual(client._resolve_deployment_url(), "https://inference.example.invalid/d1")
        self.assertEqual(client.deployment_id, "d1")
        config_call = session.calls[0]
        self.assertEqual(config_call[2]["scenarioId"], "foundation-models")
        self.assertEqual(config_call[2]["executableIds"], "aicore-sap")

    def test_version_filter_uses_mapping_bindings(self):
        """A configured version excludes other versions; mapping bindings are accepted."""
        session = _Session(
            [_configuration("c1", "sap-rpt-1-small", "1", True), _configuration("c2", "sap-rpt-1-small", "2", True)],
            [_deployment("d1", "c1"), _deployment("d2", "c2")],
        )
        client = self._client(session, version="2")
        self.assertEqual(client._resolve_deployment_url(), "https://inference.example.invalid/d2")

    def test_no_match_raises(self):
        """A model without a running deployment fails explicitly."""
        session = _Session([_configuration("c1", "sap-rpt-1-large")], [_deployment("d1", "c1")])
        with self.assertRaisesRegex(self.module.RPT1RequestError, "No AI Core configuration"):
            self._client(session)._resolve_deployment_url()
        session = _Session([_configuration("c1", "sap-rpt-1-small")], [_deployment("d1", "c1", status="STOPPED")])
        with self.assertRaisesRegex(self.module.RPT1RequestError, "No RUNNING"):
            self._client(session)._resolve_deployment_url()

    def test_multiple_running_matches_raise(self):
        """Two running deployments of the same model are ambiguous."""
        session = _Session([_configuration("c1", "sap-rpt-1-small")], [_deployment("d1", "c1"), _deployment("d2", "c1")])
        with self.assertRaisesRegex(self.module.RPT1RequestError, "Several RUNNING"):
            self._client(session)._resolve_deployment_url()

    def test_missing_deployment_url_uses_inference_path(self):
        """Deployments without deploymentUrl fall back to the inference route."""
        session = _Session([_configuration("c1", "sap-rpt-1-small")], [_deployment("d9", "c1", url=False)])
        self.assertEqual(
            self._client(session)._resolve_deployment_url(),
            "https://api.example.invalid/v2/inference/deployments/d9",
        )

    def test_cache_is_shared_across_clients(self):
        """A second client in the same process reuses the resolved deployment."""
        session = _Session([_configuration("c1", "sap-rpt-1-small")], [_deployment("d1", "c1")])
        self._client(session)._resolve_deployment_url()
        calls_after_first = len(session.calls)
        second = self._client(session)
        self.assertEqual(second._resolve_deployment_url(), "https://inference.example.invalid/d1")
        self.assertEqual(second.deployment_id, "d1")
        self.assertEqual(len(session.calls), calls_after_first)

    def test_model_name_is_required(self):
        """An empty model name is a configuration error, not a silent fallback."""
        with self.assertRaises(self.module.RPT1ValidationError):
            self._client(_Session([], []), version=None).__class__(
                aicore_base_url="https://api.example.invalid",
                aicore_auth_url="https://auth.example.invalid",
                client_id="client",
                client_secret="secret",
                model_name="",
            )


if __name__ == "__main__":
    unittest.main()
