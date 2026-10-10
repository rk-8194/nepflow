"""Typed settings for the information-entropy selector."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InformationEntropyConfig:
    """Numerical controls for the deterministic finite-pool objective."""

    background_mass: float = 1.0e-12
    kernel_scale: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.background_mass)) or self.background_mass <= 0.0:
            raise ValueError("background_mass must be finite and positive")
        if not math.isfinite(float(self.kernel_scale)) or self.kernel_scale <= 0.0:
            raise ValueError("kernel_scale must be finite and positive")
