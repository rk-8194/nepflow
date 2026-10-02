"""Typed interfaces shared by configurational generator implementations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class ConfigurationalGenerator(Protocol):
    """Generate base structures for one requested composition."""

    def generate(
        self,
        composition: Mapping[str, float],
        crystal_structures: Sequence[str],
        target_n_atoms: int = 250,
    ) -> list[Any]: ...
