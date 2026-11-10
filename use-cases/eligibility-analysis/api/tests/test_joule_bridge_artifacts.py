"""Bridge-only schema and public discovery boundary checks."""
import unittest
from pathlib import Path
from app.a2a.bridge_validation import validate_bridge
from app.a2a.public_url import public_agent_endpoint


class BridgeTests(unittest.TestCase):
    """Verify local references without weakening transport authentication."""

    def test_bridge_round_trip_contract(self):
        """The committed bridge forwards and saves context without interpretation mode."""
        self.assertEqual(validate_bridge(Path(__file__).resolve().parents[2]),[])

    def test_public_routes_and_localhost_boundary(self):
        """CF discovery never silently advertises a local development endpoint."""
        self.assertEqual(public_agent_endpoint({'AGENT_PUBLIC_URL':'https://agent.example/api/a2a/'}),'https://agent.example/api/a2a')
        self.assertEqual(public_agent_endpoint({'VCAP_APPLICATION':'{"application_uris":["agent.example"]}'}),'https://agent.example/api/a2a')
        self.assertEqual(public_agent_endpoint({'A2A_ENDPOINT_URL':'https://agent.example/custom'}),'https://agent.example/custom')
        with self.assertRaises(ValueError):public_agent_endpoint({'APP_ENV':'production'})
        with self.assertRaises(ValueError):public_agent_endpoint({'VCAP_APPLICATION':'{}','API_BASE_URL':'http://localhost:8000'})
