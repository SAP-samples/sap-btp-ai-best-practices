"""
Base Configuration Dataclasses.

Provides common configuration fields shared across model and training configs.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class BaseConfig:
    """Base configuration with common fields for all configs."""

    name: str = "default"
    channel: Optional[str] = None  # 'B&M', 'WEB', or None for all
    output_dir: Path = field(default_factory=lambda: Path("output"))
    random_seed: int = 42

    def __post_init__(self):
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        # Validate channel
        if self.channel is not None and self.channel not in ("B&M", "WEB"):
            raise ValueError(f"Invalid channel '{self.channel}'. Must be 'B&M', 'WEB', or None.")


@dataclass
class BiasCorrection:
    """Configuration for log-normal bias correction.

    When predicting in log-space and converting back via exp(),
    the mean is biased. Correction applies: exp(mu + sigma^2/2)
    """

    correct_bm: bool = False
    correct_web: bool = False
    correct_web_sales: bool = False
    correct_web_aov: bool = False

    def should_correct_bm(self) -> bool:
        """Whether to apply bias correction to B&M predictions."""
        return self.correct_bm

    def should_correct_web_sales(self) -> bool:
        """Whether to apply bias correction to WEB Sales predictions."""
        return self.correct_web or self.correct_web_sales

    def should_correct_web_aov(self) -> bool:
        """Whether to apply bias correction to WEB AOV predictions."""
        return self.correct_web or self.correct_web_aov
