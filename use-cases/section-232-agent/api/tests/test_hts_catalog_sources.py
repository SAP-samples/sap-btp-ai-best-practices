from __future__ import annotations

import pytest

from app.services.metal_composition.hts_catalog_sources import (
    INVALID_FILENAME_MESSAGE,
    normalize_hts_catalog_filename,
)


def _chapter_csv_with_leading_blank_codes(*, first_code_row: int, code: str) -> str:
    rows = [
        "HTS Number,Indent,Description,Unit of Quantity,General Rate of Duty,Special Rate of Duty,Column 2 Rate of Duty,Quota Quantity,Additional Duties",
    ]
    for row_number in range(2, first_code_row):
        rows.append(f'"","{row_number - 1}","Heading row {row_number}","","","","","",""')
    rows.append(f'"{code}","0","Sample description","","","","","",""')
    return "\n".join(rows) + "\n"


def test_normalize_hts_catalog_filename_infers_chapter_from_third_row_code():
    content_text = _chapter_csv_with_leading_blank_codes(first_code_row=3, code="2801")

    normalized = normalize_hts_catalog_filename("htsdata.csv", content_text=content_text)

    assert normalized == "chapter28.csv"


def test_normalize_hts_catalog_filename_infers_chapter_from_fourth_row_code():
    content_text = _chapter_csv_with_leading_blank_codes(first_code_row=4, code="3901")

    normalized = normalize_hts_catalog_filename("htsdata.csv", content_text=content_text)

    assert normalized == "chapter39.csv"


def test_normalize_hts_catalog_filename_rejects_files_without_code_in_first_four_rows():
    content_text = _chapter_csv_with_leading_blank_codes(first_code_row=5, code="4901")

    with pytest.raises(ValueError, match=INVALID_FILENAME_MESSAGE):
        normalize_hts_catalog_filename("htsdata.csv", content_text=content_text)
