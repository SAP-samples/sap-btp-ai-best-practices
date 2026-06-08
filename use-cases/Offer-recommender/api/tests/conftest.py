from __future__ import annotations

import sys
import os
from functools import lru_cache
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
API_ROOT = REPO_ROOT / "api"
TESTS_ROOT = API_ROOT / "tests"

for candidate in (str(REPO_ROOT), str(API_ROOT), str(TESTS_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("API_KEY", "test-api-key")
for _aicore_name in (
    "AICORE_AUTH_URL",
    "AICORE_CLIENT_ID",
    "AICORE_CLIENT_SECRET",
    "AICORE_BASE_URL",
    "AICORE_RESOURCE_GROUP",
):
    os.environ.setdefault(_aicore_name, "")

from app.nbo import hana as hana_runtime
from synthetic_runtime import synthetic_runtime_datasets


@lru_cache(maxsize=1)
def _seed_runtime_datasets():
    """Return cached anonymized runtime DataFrames for tests."""
    return synthetic_runtime_datasets()


def _patched_load_runtime_datasets(settings=None):
    """Return test runtime data through the same API as the HANA loader."""
    del settings
    return {
        dataset_name: frame.copy(deep=True)
        for dataset_name, frame in _seed_runtime_datasets().items()
    }


hana_runtime.load_runtime_datasets = _patched_load_runtime_datasets


@pytest.fixture(autouse=True)
def _reset_saved_customer_answers_between_tests():
    """Keep process-local saved customer answers from leaking between tests."""
    from app.services.saved_answers import saved_answer_service

    saved_answer_service.reset_all()
    yield
    saved_answer_service.reset_all()
