"""Run unit tests with an explicitly isolated SQLite backend, never ambient HANA.

Examples:
  PYTHONPATH=api .venv/bin/python api/scripts/run_unit_tests.py
  PYTHONPATH=api .venv/bin/python api/scripts/run_unit_tests.py --pattern 'test_workspace*.py'

Live HANA/model/browser smoke checks are separate explicit operations.
"""
import argparse
import unittest
from unittest.mock import patch
from pathlib import Path
from app.services.database import backend


def main():
    """Pin the test backend before discovery imports any credential-loading modules."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pattern',default='test*.py');args=parser.parse_args()
    backend._backend_instance=backend.DatabaseBackend(backend.BackendType.SQLITE)
    suite=unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),pattern=args.pattern)
    with patch('app.a2a.common.make_llm',side_effect=RuntimeError('External model calls are disabled in unit tests')):
        result=unittest.TextTestRunner(verbosity=1).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__=='__main__':main()
