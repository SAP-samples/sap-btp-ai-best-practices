"""Shared pytest fixtures for backend API tests."""

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

os.environ["API_KEY"] = "test-api-key"

from app.main import app


@pytest.fixture(autouse=True)
def test_api_key() -> None:
    """Maintain a stable API key for every backend test.

    Inputs:
        None. The fixture is applied automatically by pytest.

    Outputs:
        None. The ``API_KEY`` environment variable remains set to
        ``test-api-key`` for the duration of the test process.
    """
    os.environ["API_KEY"] = "test-api-key"


@pytest.fixture
def api_client() -> TestClient:
    """Create a FastAPI test client for exercising public API endpoints.

    Inputs:
        None. The fixture uses the application instance imported from
        ``app.main``.

    Outputs:
        TestClient: A synchronous client configured for the backend FastAPI
        application.
    """
    return TestClient(app)


@pytest.fixture
def repo_root() -> Path:
    """Return the repository root path for tests that need fixture files.

    Inputs:
        None. The path is derived from this file's location.

    Outputs:
        Path: Absolute path to the repository root directory.
    """
    return Path(__file__).resolve().parents[2]
