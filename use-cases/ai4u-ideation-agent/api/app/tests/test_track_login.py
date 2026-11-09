"""Integration tests for POST /api/auth/track-login and CDC service writes.

Tests the full flow: endpoint auth → CDC search → CDC read → CDC write,
with mocked HTTP calls to the CDC API.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.auth import UserIdentity
from app.services.cdc import (
    _create_account_and_set_data,
    _extract_email_domain,
    _get_account_data,
    _is_public_email_domain,
    _search_account_by_email,
    _update_account,
    store_user_login_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_XSUAA_USER = UserIdentity(
    user_id="test-xsuaa-user",
    email="test.user@sap.com",
    given_name="Test",
    family_name="User",
    user_uuid="abc-123",
    auth_provider="xsuaa",
)

_CDC_USER = UserIdentity(
    user_id="cdc-uid-456",
    email="external@example.com",
    given_name="External",
    family_name="User",
    user_uuid="cdc-uid-456",
    auth_provider="cdc",
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_auth_xsuaa():
    """Override get_current_user to return an XSUAA user."""
    from app.auth import get_current_user

    async def _override():
        return _XSUAA_USER

    app.dependency_overrides[get_current_user] = _override
    yield _XSUAA_USER
    app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
def mock_auth_cdc():
    """Override get_current_user to return a CDC user."""
    from app.auth import get_current_user

    async def _override():
        return _CDC_USER

    app.dependency_overrides[get_current_user] = _override
    yield _CDC_USER
    app.dependency_overrides.pop(get_current_user, None)


def _cdc_env():
    """Patch environment with CDC credentials."""
    return patch.dict("os.environ", {
        "CDC_API_KEY": "test-api-key",
        "CDC_USER_KEY": "test-user-key",
        "CDC_SECRET_KEY": "test-secret",
        "CDC_DATA_CENTER": "us1.gigya.com",
    })


def _mock_cdc_response(error_code=0, results=None, data=None, reg_token=None, **extra):
    """Create a mock requests.Response for CDC API calls."""
    body = {"errorCode": error_code, "errorMessage": "" if error_code == 0 else "mock error"}
    if results is not None:
        body["results"] = results
    if data is not None:
        body["data"] = data
    if reg_token is not None:
        body["regToken"] = reg_token
    body.update(extra)

    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = body
    return resp


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------


class TestTrackLoginEndpoint:
    """Test POST /api/auth/track-login."""

    def test_returns_401_without_auth(self, client):
        response = client.post("/api/auth/track-login")
        assert response.status_code == 401

    def test_returns_200_for_xsuaa_user(self, client, mock_auth_xsuaa):
        with _cdc_env(), patch("app.services.cdc.requests.post") as mock_post:
            # Search returns existing account
            mock_post.side_effect = [
                _mock_cdc_response(results=[{"UID": "existing-uid"}]),  # search
                _mock_cdc_response(data={"ideation_agent_first_login": "2026-01-01T00:00:00Z"}),  # getAccountInfo
                _mock_cdc_response(),  # setAccountInfo
            ]

            response = client.post("/api/auth/track-login")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_returns_200_for_cdc_user(self, client, mock_auth_cdc):
        with _cdc_env(), patch("app.services.cdc.requests.post") as mock_post:
            mock_post.side_effect = [
                _mock_cdc_response(results=[{"UID": "cdc-uid-456"}]),  # search
                _mock_cdc_response(data={}),  # getAccountInfo (no first_login yet)
                _mock_cdc_response(),  # setAccountInfo
            ]

            response = client.post("/api/auth/track-login")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_returns_200_even_when_cdc_fails(self, client, mock_auth_xsuaa):
        """Endpoint is non-blocking: always 200 even if CDC write throws."""
        with _cdc_env(), patch("app.services.cdc.requests.post") as mock_post:
            mock_post.side_effect = Exception("CDC is down")

            response = client.post("/api/auth/track-login")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_returns_200_when_cdc_credentials_missing(self, client, mock_auth_xsuaa):
        """Graceful failure when admin creds not configured."""
        with patch.dict("os.environ", {"CDC_API_KEY": "", "CDC_USER_KEY": "", "CDC_SECRET_KEY": ""}):
            response = client.post("/api/auth/track-login")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# CDC service unit tests
# ---------------------------------------------------------------------------


class TestSearchAccountByEmail:

    def test_returns_uid_when_found(self):
        with _cdc_env(), patch("app.services.cdc.requests.post") as mock_post:
            mock_post.return_value = _mock_cdc_response(results=[{"UID": "found-uid"}])

            uid = _search_account_by_email("test@example.com")

        assert uid == "found-uid"
        call_data = mock_post.call_args[1]["data"]
        assert "profile.email = 'test@example.com'" in call_data["query"]

    def test_returns_none_when_not_found(self):
        with _cdc_env(), patch("app.services.cdc.requests.post") as mock_post:
            mock_post.return_value = _mock_cdc_response(results=[])

            uid = _search_account_by_email("missing@example.com")

        assert uid is None

    def test_returns_none_on_cdc_error(self):
        with _cdc_env(), patch("app.services.cdc.requests.post") as mock_post:
            mock_post.return_value = _mock_cdc_response(error_code=500)

            uid = _search_account_by_email("error@example.com")

        assert uid is None

    def test_escapes_single_quotes_in_email(self):
        with _cdc_env(), patch("app.services.cdc.requests.post") as mock_post:
            mock_post.return_value = _mock_cdc_response(results=[])

            _search_account_by_email("o'malley@example.com")

        call_data = mock_post.call_args[1]["data"]
        assert "o''malley" in call_data["query"]


class TestGetAccountData:

    def test_returns_data_fields(self):
        with _cdc_env(), patch("app.services.cdc.requests.post") as mock_post:
            mock_post.return_value = _mock_cdc_response(
                data={"ideation_agent_first_login": "2026-01-01T00:00:00Z", "other_field": "value"}
            )

            data = _get_account_data("some-uid")

        assert data["ideation_agent_first_login"] == "2026-01-01T00:00:00Z"
        assert data["other_field"] == "value"

    def test_returns_empty_dict_on_error(self):
        with _cdc_env(), patch("app.services.cdc.requests.post") as mock_post:
            mock_post.return_value = _mock_cdc_response(error_code=403)

            data = _get_account_data("bad-uid")

        assert data == {}


class TestUpdateAccount:

    def test_sends_data_and_preferences(self):
        with _cdc_env(), patch("app.services.cdc.requests.post") as mock_post:
            mock_post.return_value = _mock_cdc_response()

            _update_account(
                "uid-123",
                data={"ideation_agent_last_login": "2026-07-01T12:00:00Z"},
                preferences={"privacy.ideation_agent": {"isConsentGranted": True}},
            )

        call_data = mock_post.call_args[1]["data"]
        assert call_data["UID"] == "uid-123"
        assert "ideation_agent_last_login" in call_data["data"]
        assert "privacy.ideation_agent" in call_data["preferences"]

    def test_raises_on_cdc_error(self):
        with _cdc_env(), patch("app.services.cdc.requests.post") as mock_post:
            mock_post.return_value = _mock_cdc_response(error_code=400)

            with pytest.raises(ValueError, match="setAccountInfo failed"):
                _update_account("uid-123", data={"x": "y"}, preferences={})


class TestCreateAccountAndSetData:

    def test_two_step_flow(self):
        with _cdc_env(), patch("app.services.cdc.requests.post") as mock_post:
            mock_post.side_effect = [
                _mock_cdc_response(regToken="reg-token-abc"),  # initRegistration
                _mock_cdc_response(UID="new-uid"),  # register
            ]

            _create_account_and_set_data(
                "new@example.com",
                data={"ideation_agent_first_login": "2026-07-01T12:00:00Z"},
                preferences={"terms.ideation_agent": {"isConsentGranted": True}},
            )

        # Verify initRegistration call
        init_call = mock_post.call_args_list[0]
        init_data = init_call[1]["data"]
        assert "isLite" not in init_data  # Full account, not lite

        # Verify register call
        reg_call = mock_post.call_args_list[1]
        reg_data = reg_call[1]["data"]
        assert reg_data["email"] == "new@example.com"
        assert "password" in reg_data  # Random password included
        assert reg_data["finalizeRegistration"] == "true"
        assert "ideation_agent_first_login" in reg_data["data"]

    def test_raises_when_no_reg_token(self):
        with _cdc_env(), patch("app.services.cdc.requests.post") as mock_post:
            mock_post.return_value = _mock_cdc_response(error_code=500)

            with pytest.raises(ValueError, match="initRegistration failed"):
                _create_account_and_set_data("fail@example.com", data={}, preferences={})

    def test_raises_when_register_fails(self):
        with _cdc_env(), patch("app.services.cdc.requests.post") as mock_post:
            mock_post.side_effect = [
                _mock_cdc_response(regToken="reg-token-abc"),  # initRegistration OK
                _mock_cdc_response(error_code=400),  # register fails
            ]

            with pytest.raises(ValueError, match="accounts.register failed"):
                _create_account_and_set_data("fail@example.com", data={}, preferences={})


class TestStoreUserLoginData:
    """Integration tests for the orchestrator function."""

    def test_existing_account_with_first_login_already_set(self):
        """Should only update last_login, not overwrite first_login."""
        with _cdc_env(), patch("app.services.cdc.requests.post") as mock_post:
            mock_post.side_effect = [
                _mock_cdc_response(results=[{"UID": "uid-existing"}]),  # search
                _mock_cdc_response(data={"ideation_agent_first_login": "2026-01-01T00:00:00Z"}),  # getAccountInfo
                _mock_cdc_response(),  # setAccountInfo
            ]

            store_user_login_data("returning@example.com")

        # The setAccountInfo call (3rd) should NOT include first_login
        set_call = mock_post.call_args_list[2]
        written_data = json.loads(set_call[1]["data"]["data"])
        assert "ideation_agent_last_login" in written_data
        assert "ideation_agent_first_login" not in written_data

    def test_existing_account_without_first_login(self):
        """Should write both first_login and last_login."""
        with _cdc_env(), patch("app.services.cdc.requests.post") as mock_post:
            mock_post.side_effect = [
                _mock_cdc_response(results=[{"UID": "uid-existing"}]),  # search
                _mock_cdc_response(data={}),  # getAccountInfo — no first_login
                _mock_cdc_response(),  # setAccountInfo
            ]

            store_user_login_data("first-time-existing@example.com")

        set_call = mock_post.call_args_list[2]
        written_data = json.loads(set_call[1]["data"]["data"])
        assert "ideation_agent_first_login" in written_data
        assert "ideation_agent_last_login" in written_data

    def test_new_account_creation(self):
        """No account found → create full account with all data."""
        with _cdc_env(), patch("app.services.cdc.requests.post") as mock_post:
            mock_post.side_effect = [
                _mock_cdc_response(results=[]),  # search — not found
                _mock_cdc_response(regToken="new-reg-token"),  # initRegistration
                _mock_cdc_response(UID="brand-new-uid"),  # register
            ]

            store_user_login_data("brand-new@example.com")

        # initRegistration (2nd call)
        init_call = mock_post.call_args_list[1]
        assert "accounts.initRegistration" in init_call[0][0] or "initRegistration" in str(init_call)

        # register (3rd call)
        reg_call = mock_post.call_args_list[2]
        reg_data = reg_call[1]["data"]
        assert reg_data["email"] == "brand-new@example.com"
        written_data = json.loads(reg_data["data"])
        assert "ideation_agent_first_login" in written_data
        assert "ideation_agent_last_login" in written_data

    def test_preferences_always_written(self):
        """Consent preferences are sent on every call."""
        with _cdc_env(), patch("app.services.cdc.requests.post") as mock_post:
            mock_post.side_effect = [
                _mock_cdc_response(results=[{"UID": "uid-1"}]),
                _mock_cdc_response(data={"ideation_agent_first_login": "2026-01-01T00:00:00Z"}),
                _mock_cdc_response(),
            ]

            store_user_login_data("user@example.com")

        set_call = mock_post.call_args_list[2]
        written_prefs = json.loads(set_call[1]["data"]["preferences"])
        assert written_prefs["privacy.ideation_agent"]["isConsentGranted"] is True
        assert written_prefs["terms.ideation_agent"]["isConsentGranted"] is True


# ---------------------------------------------------------------------------
# Company domain helper tests
# ---------------------------------------------------------------------------


class TestExtractEmailDomain:

    def test_extracts_domain(self):
        assert _extract_email_domain("user@sap.com") == "sap.com"

    def test_lowercases_domain(self):
        assert _extract_email_domain("User@SAP.COM") == "sap.com"

    def test_strips_whitespace(self):
        assert _extract_email_domain("  user@sap.com  ") == "sap.com"

    def test_returns_none_for_no_at(self):
        assert _extract_email_domain("no-at-sign") is None

    def test_returns_none_for_empty(self):
        assert _extract_email_domain("") is None

    def test_returns_none_for_trailing_at(self):
        assert _extract_email_domain("user@") is None


class TestIsPublicEmailDomain:

    def test_gmail_is_public(self):
        assert _is_public_email_domain("gmail.com") is True

    def test_hotmail_is_public(self):
        assert _is_public_email_domain("hotmail.com") is True

    def test_outlook_is_public(self):
        assert _is_public_email_domain("outlook.com") is True

    def test_corporate_is_not_public(self):
        assert _is_public_email_domain("sap.com") is False

    def test_case_insensitive(self):
        assert _is_public_email_domain("Gmail.COM") is True

    def test_env_override(self):
        with patch.dict("os.environ", {"CDC_PUBLIC_EMAIL_DOMAINS": "custom.io,other.org"}):
            assert _is_public_email_domain("custom.io") is True
            assert _is_public_email_domain("gmail.com") is False  # not in override


class TestCompanyDomainInStoreUserLoginData:

    def test_corporate_email_sets_company_domain(self):
        """SAP email → company_domain = 'sap.com'."""
        with _cdc_env(), patch("app.services.cdc.requests.post") as mock_post:
            mock_post.side_effect = [
                _mock_cdc_response(results=[{"UID": "uid-sap"}]),
                _mock_cdc_response(data={"ideation_agent_first_login": "2026-01-01T00:00:00Z"}),
                _mock_cdc_response(),
            ]

            store_user_login_data("test.user@sap.com")

        set_call = mock_post.call_args_list[2]
        written_data = json.loads(set_call[1]["data"]["data"])
        assert written_data["company_domain"] == "sap.com"

    def test_public_email_sets_empty_company_domain(self):
        """Gmail email → company_domain = ''."""
        with _cdc_env(), patch("app.services.cdc.requests.post") as mock_post:
            mock_post.side_effect = [
                _mock_cdc_response(results=[{"UID": "uid-gmail"}]),
                _mock_cdc_response(data={"ideation_agent_first_login": "2026-01-01T00:00:00Z"}),
                _mock_cdc_response(),
            ]

            store_user_login_data("user@gmail.com")

        set_call = mock_post.call_args_list[2]
        written_data = json.loads(set_call[1]["data"]["data"])
        assert written_data["company_domain"] == ""

    def test_new_account_includes_company_domain(self):
        """New account creation also includes company_domain."""
        with _cdc_env(), patch("app.services.cdc.requests.post") as mock_post:
            mock_post.side_effect = [
                _mock_cdc_response(results=[]),  # search — not found
                _mock_cdc_response(regToken="reg-token"),  # initRegistration
                _mock_cdc_response(UID="new-uid"),  # register
            ]

            store_user_login_data("new-user@acme.corp")

        reg_call = mock_post.call_args_list[2]
        written_data = json.loads(reg_call[1]["data"]["data"])
        assert written_data["company_domain"] == "acme.corp"

    def test_new_account_public_email_empty_domain(self):
        """New account with public email → company_domain = ''."""
        with _cdc_env(), patch("app.services.cdc.requests.post") as mock_post:
            mock_post.side_effect = [
                _mock_cdc_response(results=[]),
                _mock_cdc_response(regToken="reg-token"),
                _mock_cdc_response(UID="new-uid"),
            ]

            store_user_login_data("user@protonmail.com")

        reg_call = mock_post.call_args_list[2]
        written_data = json.loads(reg_call[1]["data"]["data"])
        assert written_data["company_domain"] == ""
