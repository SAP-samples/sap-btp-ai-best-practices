from __future__ import annotations

from app.chat.parsing import extract_account_number


def test_extract_account_number_from_free_text_sentence() -> None:
    all_accounts = ["103", "104", "100000"]

    result = extract_account_number(
        "my billing account number is 104",
        all_accounts,
    )

    assert result == "104"


def test_extract_account_number_returns_none_when_no_known_account_found() -> None:
    all_accounts = ["103", "104", "100000"]

    result = extract_account_number(
        "the account number is 999999",
        all_accounts,
    )

    assert result is None

