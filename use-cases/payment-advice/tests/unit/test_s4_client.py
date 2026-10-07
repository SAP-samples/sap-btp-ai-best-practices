"""Offline checks for S/4 connectivity mode selection, BTP context and CSRF writes."""
import json
import os
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from app.s4.client import S4Client, S4Config, S4ConfigError, S4HTTPError, load_s4_config, odata_error_message
from app.s4.connectivity import S4BtpConnectivityClient, load_s4_btp_connectivity_config_from_env


class FakeResponse:
    """Minimal stand-in for requests.Response."""
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self.text = payload if isinstance(payload, str) else json.dumps(payload) if payload is not None else ""
        self.headers = headers or {}
        self.reason = "OK" if status_code < 400 else "Bad Request"

    def json(self):
        return json.loads(self.text)


class FakeSession:
    """Records requests and replays queued responses in order."""
    def __init__(self, responses):
        self.responses, self.requests, self.headers, self.auth = list(responses), [], {}, None

    def request(self, method, url, **kwargs):
        self.requests.append({"method": method, "url": url, **kwargs})
        return self.responses.pop(0)


BTP_ENV = {
    "S4_DESTINATION_NAME": "S4_HTTP", "DESTINATION_SERVICE_URI": "https://destination.example",
    "DESTINATION_TOKEN_BASE_URL": "https://tenant.auth.example", "DESTINATION_CLIENT_ID": "dest-id",
    "DESTINATION_CLIENT_SECRET": "dest-secret", "CONNECTIVITY_PROXY_HOST": "connectivityproxy.internal.example",
    "CONNECTIVITY_PROXY_PORT": "20003", "CONNECTIVITY_TOKEN_BASE_URL": "https://tenant.auth.example",
    "CONNECTIVITY_CLIENT_ID": "conn-id", "CONNECTIVITY_CLIENT_SECRET": "conn-secret", "S4_CLIENT": "550",
}
DIRECT_ENV = {"S4_BASE_URL": "https://s4.example:44301/", "S4_USERNAME": "user", "S4_PASSWORD": "pw", "S4_CLIENT": "550"}
NO_ENV_FILE = os.devnull


class ModeSelectionTests(unittest.TestCase):
    """`auto` must pick btp only inside Cloud Foundry and direct on a laptop."""
    def test_auto_prefers_direct_locally_and_btp_in_cloud_foundry(self):
        with patch.dict(os.environ, {**BTP_ENV, **DIRECT_ENV}, clear=True):
            self.assertEqual(load_s4_config(NO_ENV_FILE).mode, "direct")
        with patch.dict(os.environ, {**BTP_ENV, **DIRECT_ENV, "VCAP_APPLICATION": "{}"}, clear=True):
            self.assertEqual(load_s4_config(NO_ENV_FILE).mode, "btp")

    def test_direct_mode_reports_missing_variables(self):
        with patch.dict(os.environ, {"S4_CONNECTIVITY_MODE": "direct", "S4_BASE_URL": "https://s4"}, clear=True):
            with self.assertRaisesRegex(S4ConfigError, "S4_PASSWORD, S4_USERNAME"):
                load_s4_config(NO_ENV_FILE)

    def test_explicit_btp_requires_complete_service_keys(self):
        env = {**BTP_ENV, "S4_CONNECTIVITY_MODE": "btp"}
        del env["CONNECTIVITY_CLIENT_SECRET"]
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(S4ConfigError):
                load_s4_config(NO_ENV_FILE)


class BtpContextTests(unittest.TestCase):
    """Destination auth header and Connectivity proxy are merged into one context."""
    def test_runtime_context_combines_destination_auth_and_proxy(self):
        session = FakeSession([
            FakeResponse(200, {"access_token": "dest-token", "expires_in": 3600}),
            FakeResponse(200, {"destinationConfiguration": {"URL": "http://s4-virtual:44301", "ProxyType": "OnPremise",
                                                            "Authentication": "BasicAuthentication", "sap-client": "550"},
                               "authTokens": [{"type": "Basic", "value": "abc", "expires_in": 3600,
                                               "http_header": {"key": "Authorization", "value": "Basic abc"}}]}),
            FakeResponse(200, {"access_token": "conn-token", "expires_in": 3600}),
        ])
        with patch.dict(os.environ, BTP_ENV, clear=True):
            config = load_s4_btp_connectivity_config_from_env()
        now = datetime(2026, 9, 28, tzinfo=timezone.utc)
        context = S4BtpConnectivityClient(config, session=session, now=lambda: now).resolve_runtime_context()
        self.assertEqual(context.base_url, "http://s4-virtual:44301")
        self.assertEqual(context.client, "550")
        self.assertEqual(context.headers["Authorization"], "Basic abc")
        self.assertEqual(context.headers["Proxy-Authorization"], "Bearer conn-token")
        self.assertEqual(context.proxies["https"], "http://connectivityproxy.internal.example:20003")


class ClientTests(unittest.TestCase):
    """OData paths, sap-client, CSRF round trip and error details."""
    def client(self, responses):
        session = FakeSession(responses)
        config = S4Config(base_url="https://s4.example", client="550", verify=True, username="u", password="p")
        return S4Client(config, session=session), session

    def test_post_fetches_csrf_token_and_echoes_it(self):
        client, session = self.client([
            FakeResponse(200, "", {"X-CSRF-Token": "tok-1"}),
            FakeResponse(201, {"d": {"PaymentAdvice": "0400000001"}}),
        ])
        body = client.post_json("API_PAYMENT_ADVICE_SRV", "/A_PaymentAdvice", {"CompanyCode": "CA01"})
        self.assertEqual(body["d"]["PaymentAdvice"], "0400000001")
        fetch, post = session.requests
        self.assertEqual(fetch["headers"]["X-CSRF-Token"], "Fetch")
        self.assertEqual(fetch["url"], "https://s4.example/sap/opu/odata/sap/API_PAYMENT_ADVICE_SRV/")
        self.assertEqual(post["headers"]["X-CSRF-Token"], "tok-1")
        self.assertEqual(post["params"]["sap-client"], "550")
        self.assertEqual(post["json"], {"CompanyCode": "CA01"})

    def test_http_error_exposes_odata_message(self):
        error_body = {"error": {"message": {"value": "Company code XX01 does not exist"},
                                "innererror": {"errordetails": [{"message": "Check entry"}]}}}
        client, _ = self.client([FakeResponse(400, error_body)])
        with self.assertRaises(S4HTTPError) as caught:
            client.get_json("API_PAYMENT_ADVICE_SRV", "/A_PaymentAdvice")
        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(caught.exception.detail, "Company code XX01 does not exist; Check entry")

    def test_s4_verify_false_reaches_every_request(self):
        with patch.dict(os.environ, {**DIRECT_ENV, "S4_CONNECTIVITY_MODE": "direct", "S4_VERIFY": "false"}, clear=True):
            config = load_s4_config(NO_ENV_FILE)
        session = FakeSession([FakeResponse(200, {"d": {"results": []}})])
        S4Client(config, session=session).get_json("API_COMPANYCODE_SRV", "/A_CompanyCode")
        self.assertIs(session.requests[0]["verify"], False)
        self.assertEqual(session.requests[0]["url"], "https://s4.example:44301/sap/opu/odata/sap/API_COMPANYCODE_SRV/A_CompanyCode")

    def test_error_message_parser_tolerates_non_json(self):
        self.assertIsNone(odata_error_message("<html>502</html>"))


if __name__ == "__main__":
    unittest.main()
