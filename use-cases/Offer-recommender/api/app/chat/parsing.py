from __future__ import annotations

import re

from app.services.recommendations import find_matching_account


def extract_account_number(text: str, all_accounts: list[str]) -> str | None:
    candidates = re.findall(r"[A-Za-z0-9]{3,}", text or "")
    for candidate in candidates:
        matched = find_matching_account(candidate, all_accounts)
        if matched:
            return matched
    return None
