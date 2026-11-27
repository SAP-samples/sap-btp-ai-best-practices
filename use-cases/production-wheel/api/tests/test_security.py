"""User authentication contract for the workspace API (app/security.py).

The tests sign real JWTs with a throwaway RSA key and bind a fake XSUAA service
through VCAP_SERVICES whose ``verificationkey`` is the matching public key. The
token is therefore validated end to end by the real ``sap-xssec`` library; no
validator is mocked. The app under test only mounts ``GET /api/me``, which uses
the same ``require_user`` dependency as every other protected router.

Run with:  cd api && ../.venv/bin/python -m pytest tests/test_security.py -q
"""

import json
import time

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.user_session import router
from app.security import load_xsuaa_credentials

CLIENT_ID = "sb-test-app!t1"
XSAPPNAME = "test-app!t1"


def make_key():
    """Create an RSA private key and its PEM public key for signing test tokens."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_pem = private_key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()
    return private_key, public_pem


SIGNING_KEY, PUBLIC_PEM = make_key()
OTHER_KEY, _ = make_key()


def token(scopes=(f"{XSAPPNAME}.User",), key=SIGNING_KEY, expires_in=600):
    """Sign a user token shaped like an XSUAA authorization-code token."""
    now = int(time.time())
    claims = {
        "cid": CLIENT_ID,
        "zid": "zone-1",
        "aud": [CLIENT_ID],
        "scope": list(scopes),
        "grant_type": "authorization_code",
        "user_name": "jane.doe@example.com",
        "email": "jane.doe@example.com",
        "given_name": "Jane",
        "family_name": "Doe",
        "iat": now,
        "exp": now + expires_in,
    }
    return jwt.encode(claims, key, algorithm="RS256")


@pytest.fixture
def client(monkeypatch):
    """App with GET /api/me and a bound fake XSUAA service in xsuaa mode."""
    vcap = {"xsuaa": [{"credentials": {
        "clientid": CLIENT_ID,
        "clientsecret": "not-used-by-offline-validation",
        "xsappname": XSAPPNAME,
        "url": "https://test.authentication.example.com",
        "uaadomain": "authentication.example.com",
        "verificationkey": PUBLIC_PEM,
    }}]}
    monkeypatch.setenv("VCAP_SERVICES", json.dumps(vcap))
    monkeypatch.delenv("AUTH_MODE", raising=False)
    monkeypatch.delenv("XSAPPNAME", raising=False)
    monkeypatch.setenv("APP_ENV", "production")
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_valid_token_with_role_returns_user(client):
    """A signed, unexpired token carrying $XSAPPNAME.User is admitted."""
    response = client.get("/api/me", headers={"Authorization": f"Bearer {token()}"})
    assert response.status_code == 200
    assert response.json() == {
        "user_id": "jane.doe@example.com",
        "display_name": "Jane Doe",
        "email": "jane.doe@example.com",
        "sign_in_required": True,
    }


def test_missing_token_requires_sign_in(client):
    """No Authorization header means the caller is not signed in."""
    response = client.get("/api/me")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_token_without_role_is_forbidden(client):
    """A valid user without the role collection gets 403, not 401."""
    response = client.get("/api/me", headers={"Authorization": f"Bearer {token(scopes=('openid',))}"})
    assert response.status_code == 403


def test_token_signed_by_another_key_is_rejected(client):
    """A forged token fails signature validation."""
    response = client.get("/api/me", headers={"Authorization": f"Bearer {token(key=OTHER_KEY)}"})
    assert response.status_code == 401


def test_expired_token_is_rejected(client):
    """An expired session token must be renewed through the approuter."""
    response = client.get("/api/me", headers={"Authorization": f"Bearer {token(expires_in=-60)}"})
    assert response.status_code == 401


def test_missing_binding_fails_closed(client, monkeypatch):
    """Without an XSUAA binding nothing is served, even with a token."""
    monkeypatch.delenv("VCAP_SERVICES")
    response = client.get("/api/me", headers={"Authorization": f"Bearer {token()}"})
    assert response.status_code == 500


def test_local_mode_returns_fixed_developer(client, monkeypatch):
    """AUTH_MODE=local outside production admits the fixed developer user."""
    monkeypatch.setenv("AUTH_MODE", "local")
    monkeypatch.setenv("APP_ENV", "development")
    response = client.get("/api/me")
    assert response.status_code == 200
    assert response.json()["user_id"] == "local-developer"
    assert response.json()["sign_in_required"] is False


def test_local_mode_is_refused_in_production(client, monkeypatch):
    """A leaked AUTH_MODE=local can never open a production deployment."""
    monkeypatch.setenv("AUTH_MODE", "local")
    response = client.get("/api/me")
    assert response.status_code == 500


def test_public_mode_admits_anyone_in_production(client, monkeypatch):
    """AUTH_MODE=public (showcase deployment) needs no binding and no token."""
    monkeypatch.setenv("AUTH_MODE", "public")
    monkeypatch.delenv("VCAP_SERVICES")
    response = client.get("/api/me")
    assert response.status_code == 200
    assert response.json() == {
        "user_id": "public",
        "display_name": "Public access",
        "email": None,
        "sign_in_required": False,
    }


def test_unknown_mode_is_refused(client, monkeypatch):
    """A typo in AUTH_MODE refuses requests instead of weakening security."""
    monkeypatch.setenv("AUTH_MODE", "none")
    response = client.get("/api/me", headers={"Authorization": f"Bearer {token()}"})
    assert response.status_code == 500


def test_credentials_found_by_tag(monkeypatch):
    """A user-provided service tagged 'xsuaa' is accepted as the binding."""
    vcap = {"user-provided": [{"tags": ["xsuaa"], "credentials": {"clientid": "x"}}]}
    monkeypatch.setenv("VCAP_SERVICES", json.dumps(vcap))
    assert load_xsuaa_credentials() == {"clientid": "x"}
