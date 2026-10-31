"""Auditable production-wheel grouping optimizer prototype."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("production-wheel-prototype")
except PackageNotFoundError:  # pragma: no cover - editable source before install
    __version__ = "0.1.0"

__all__ = ["__version__"]
