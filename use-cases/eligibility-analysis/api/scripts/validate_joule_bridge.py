"""Validate local Joule A2A bridge artifacts without deploying.

Example: PYTHONPATH=api .venv/bin/python api/scripts/validate_joule_bridge.py --root .
"""
import argparse
from app.a2a.bridge_validation import validate_bridge


def main():
    """Report concrete bridge errors and exit nonzero when a contract fails."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default='.')
    errors=validate_bridge(parser.parse_args().root)
    if errors:
        for error in errors:print('ERROR: '+error)
        raise SystemExit(1)
    print('Joule bridge contracts validated; no deployment or live tenant call performed.')


if __name__=='__main__':main()
