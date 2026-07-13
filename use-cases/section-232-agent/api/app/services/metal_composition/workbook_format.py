"""Supported workbook format helpers for Material Master uploads."""

from __future__ import annotations

from pathlib import Path

SUPPORTED_MATERIAL_MASTER_WORKBOOK_EXTENSIONS = (".xlsb", ".xlsx")
MATERIAL_MASTER_WORKBOOK_READ_ENGINES = {
    ".xlsb": "pyxlsb",
    ".xlsx": "openpyxl",
}


def workbook_suffix(path_or_name: str | Path) -> str:
    """Return the lowercase file extension for a workbook path or filename.

    Args:
        path_or_name: User-supplied workbook path or uploaded filename.

    Returns:
        A lowercase suffix such as ``.xlsb`` or ``.xlsx``. Empty and missing
        values return an empty string.
    """

    return Path(str(path_or_name or "").strip()).suffix.lower()


def is_supported_material_master_workbook(path_or_name: str | Path) -> bool:
    """Determine whether a path uses a supported Material Master workbook format.

    Args:
        path_or_name: User-supplied workbook path or uploaded filename.

    Returns:
        True when the suffix is one of the supported Excel workbook formats.
    """

    return workbook_suffix(path_or_name) in SUPPORTED_MATERIAL_MASTER_WORKBOOK_EXTENSIONS


def supported_material_master_workbook_description() -> str:
    """Return human-readable copy describing accepted Material Master workbook formats.

    Returns:
        A compact extension list suitable for validation messages.
    """

    return " or ".join(SUPPORTED_MATERIAL_MASTER_WORKBOOK_EXTENSIONS)


def pandas_engine_for_material_master_workbook(path_or_name: str | Path) -> str:
    """Return the pandas Excel reader engine for a Material Master workbook path.

    Args:
        path_or_name: User-supplied workbook path or uploaded filename.

    Returns:
        The pandas engine name required to read the workbook format.

    Raises:
        ValueError: If the workbook suffix is not supported.
    """

    suffix = workbook_suffix(path_or_name)
    try:
        return MATERIAL_MASTER_WORKBOOK_READ_ENGINES[suffix]
    except KeyError as exc:
        raise ValueError(
            "Unsupported Material Master workbook format. "
            f"Expected {supported_material_master_workbook_description()}."
        ) from exc
