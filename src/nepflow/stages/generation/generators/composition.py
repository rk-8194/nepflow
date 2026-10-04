"""Typed unary, binary, and ternary composition-grid generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Mapping

from nepflow.config.models import CompositionConfig
from nepflow.stages.generation.validation import validate_composition_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CompositionGrid:
    """Generate the Phase 2 composition grid from typed values."""

    elements: tuple[str, ...] = field(default_factory=tuple)
    step: float = 0.1
    include_pure: bool = True
    include_binaries: bool = True
    include_ternaries: bool = True

    def __init__(
        self,
        elements: tuple[str, ...] | list[str],
        step: float = 0.1,
        include_pure: bool = True,
        include_binaries: bool = True,
        include_ternaries: bool = True,
    ) -> None:
        object.__setattr__(self, "elements", tuple(sorted(str(item) for item in elements)))
        object.__setattr__(self, "step", float(step))
        object.__setattr__(self, "include_pure", bool(include_pure))
        object.__setattr__(self, "include_binaries", bool(include_binaries))
        object.__setattr__(self, "include_ternaries", bool(include_ternaries))
        validate_composition_config(
            CompositionConfig(
                elements=self.elements,
                composition_step=self.step,
                include_pure_elements=self.include_pure,
                include_binaries=self.include_binaries,
                include_ternaries=self.include_ternaries,
            )
        )

    @classmethod
    def from_config(cls, config: CompositionConfig) -> "CompositionGrid":
        """Build a grid without converting typed config back to a raw dict."""

        validate_composition_config(config)
        return cls(
            config.elements,
            step=config.composition_step,
            include_pure=config.include_pure_elements,
            include_binaries=config.include_binaries,
            include_ternaries=config.include_ternaries,
        )

    def generate(self) -> list[dict[str, float]]:
        compositions: list[dict[str, float]] = []
        if self.include_pure:
            compositions.extend(self._generate_pure_compositions())
        if self.include_binaries and len(self.elements) >= 2:
            compositions.extend(self._generate_binary_compositions())
        if self.include_ternaries and len(self.elements) >= 3:
            compositions.extend(self._generate_ternary_compositions())
        logger.info("Generated %d compositions", len(compositions))
        return compositions

    def _generate_pure_compositions(self) -> list[dict[str, float]]:
        return [{element: 1.0} for element in self.elements]

    def _generate_binary_compositions(self) -> list[dict[str, float]]:
        compositions: list[dict[str, float]] = []
        for first, second in combinations(self.elements, 2):
            for fraction_second in self._interior_fractions():
                compositions.append(
                    {
                        first: round(1.0 - fraction_second, 10),
                        second: round(fraction_second, 10),
                    }
                )
        return compositions

    def _generate_ternary_compositions(self) -> list[dict[str, float]]:
        compositions: list[dict[str, float]] = []
        for first, second, third in combinations(self.elements, 3):
            for fraction_second in self._interior_fractions():
                for fraction_third in self._interior_fractions():
                    fraction_first = round(1.0 - fraction_second - fraction_third, 10)
                    if fraction_first > 0.0:
                        compositions.append(
                            {
                                first: fraction_first,
                                second: round(fraction_second, 10),
                                third: round(fraction_third, 10),
                            }
                        )
        return compositions

    def _interior_fractions(self) -> list[float]:
        n_steps = round(1.0 / self.step)
        return [round(index * self.step, 10) for index in range(1, n_steps)]

    @staticmethod
    def format_composition(composition: Mapping[str, float]) -> str:
        parts = sorted(composition, key=lambda element: -composition[element])
        labels: list[str] = []
        for element in parts:
            fraction = composition[element]
            labels.append(element if fraction == 1.0 else f"{element}{fraction:.2g}")
        return "-".join(labels)
