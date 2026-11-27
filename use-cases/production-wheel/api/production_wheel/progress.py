"""Small dependency-free progress reporting for long CLI operations."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import TextIO


@dataclass(slots=True)
class ProgressBar:
    """Render a compact progress bar.

    Args:
        total: Expected number of completed units.
        label: Short operation label shown in the terminal.
        stream: Text stream used for progress messages.

    The class writes display-only progress and returns no value.
    """

    total: int
    label: str
    stream: TextIO = sys.stderr
    width: int = 24
    _last_percent: int = field(default=-1, init=False)

    def update(self, completed: int) -> None:
        """Render progress when the displayed percentage changes.

        Args:
            completed: Number of completed units.

        Returns:
            None.
        """

        denominator = max(self.total, 1)
        ratio = min(max(completed / denominator, 0.0), 1.0)
        percent = int(ratio * 100)
        display = 100 if percent == 100 else percent // 5 * 5
        if display == self._last_percent:
            return
        self._last_percent = display
        filled = int(display / 100 * self.width)
        bar = "#" * filled + "-" * (self.width - filled)
        self.stream.write(f"\r{self.label:<24} [{bar}] {display:3d}%")
        self.stream.flush()

    def finish(self) -> None:
        """Complete the bar and terminate its terminal line."""

        self.update(self.total)
        self.stream.write("\n")
        self.stream.flush()
