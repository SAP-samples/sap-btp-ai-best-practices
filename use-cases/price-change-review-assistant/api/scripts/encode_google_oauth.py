"""Encode local Gmail OAuth JSON files for api/.env or Cloud Foundry vars.

Examples:
    python api/scripts/encode_google_oauth.py \
      --credentials google_tmp/credentials.json \
      --token google_tmp/token.json

    python api/scripts/encode_google_oauth.py \
      --credentials google_tmp/credentials.json \
      --token google_tmp/token.json \
      --format dotenv >> api/.env
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path


def encode_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode Gmail OAuth JSON files as base64.")
    parser.add_argument("--credentials", required=True, type=Path)
    parser.add_argument("--token", required=True, type=Path)
    parser.add_argument("--format", choices=["shell", "dotenv"], default="shell")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    credentials = encode_file(args.credentials)
    token = encode_file(args.token)
    quote = "" if args.format == "dotenv" else "'"
    print(f"GOOGLE_CREDENTIALS_JSON_B64={quote}{credentials}{quote}")
    print(f"GOOGLE_TOKEN_JSON_B64={quote}{token}{quote}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
