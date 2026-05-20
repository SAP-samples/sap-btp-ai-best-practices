"""Supported workbook format helpers for GCC Tracker uploads."""

from __future__ import annotations

from pathlib import Path

SUPPORTED_GCC_TRACKER_WORKBOOK_EXTENSIONS = (".xlsb", ".xlsx")
GCC_TRACKER_WORKBOOK_READ_ENGINES = {
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


def is_supported_gcc_tracker_workbook(path_or_name: str | Path) -> bool:
    """Determine whether a path uses a supported GCC Tracker workbook format.

    Args:
        path_or_name: User-supplied workbook path or uploaded filename.

    Returns:
        True when the suffix is one of the supported Excel workbook formats.
    """

    return workbook_suffix(path_or_name) in SUPPORTED_GCC_TRACKER_WORKBOOK_EXTENSIONS


def supported_gcc_tracker_workbook_description() -> str:
    """Return human-readable copy describing accepted GCC Tracker workbook formats.

    Returns:
        A compact extension list suitable for validation messages.
    """

    return " or ".join(SUPPORTED_GCC_TRACKER_WORKBOOK_EXTENSIONS)


def pandas_engine_for_gcc_tracker_workbook(path_or_name: str | Path) -> str:
    """Return the pandas Excel reader engine for a GCC Tracker workbook path.

    Args:
        path_or_name: User-supplied workbook path or uploaded filename.

    Returns:
        The pandas engine name required to read the workbook format.

    Raises:
        ValueError: If the workbook suffix is not supported.
    """

    suffix = workbook_suffix(path_or_name)
    try:
        return GCC_TRACKER_WORKBOOK_READ_ENGINES[suffix]
    except KeyError as exc:
        raise ValueError(
            "Unsupported GCC Tracker workbook format. "
            f"Expected {supported_gcc_tracker_workbook_description()}."
        ) from exc
