"""
Composition sampling on the N-element simplex.

Generates systematic composition grids for unary, binary, and ternary
(and higher) systems at a configurable atomic-fraction step size.
"""

import logging
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger("nepflow.composition")


class CompositionGrid:
    """Generate compositions on the element simplex."""

    def __init__(
        self,
        elements: List[str],
        step: float = 0.1,
        include_pure: bool = True,
        include_binaries: bool = True,
        include_ternaries: bool = True,
    ):
        self.elements = sorted(elements)
        self.step = step
        self.include_pure = include_pure
        self.include_binaries = include_binaries
        self.include_ternaries = include_ternaries

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def generate(self) -> List[Dict[str, float]]:
        """Return all compositions as ``{element: fraction}`` dicts.

        Fractions sum to 1.0 within floating-point tolerance.
        """
        compositions: List[Dict[str, float]] = []

        if self.include_pure:
            compositions.extend(self._pure())

        if self.include_binaries and len(self.elements) >= 2:
            compositions.extend(self._binaries())

        if self.include_ternaries and len(self.elements) >= 3:
            compositions.extend(self._ternaries())

        logger.info(
            f"Generated {len(compositions)} compositions "
            f"({len(self.elements)} elements, step={self.step})"
        )
        return compositions

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _pure(self) -> List[Dict[str, float]]:
        """Pure element compositions."""
        return [{el: 1.0} for el in self.elements]

    def _binaries(self) -> List[Dict[str, float]]:
        """Binary compositions excluding pure endpoints (already in _pure)."""
        comps: List[Dict[str, float]] = []
        fractions = self._interior_fractions()

        for a, b in combinations(self.elements, 2):
            for x_b in fractions:
                comps.append({a: round(1.0 - x_b, 10), b: round(x_b, 10)})

        return comps

    def _ternaries(self) -> List[Dict[str, float]]:
        """Ternary compositions on triangular grid, excluding edges."""
        comps: List[Dict[str, float]] = []
        fractions = self._interior_fractions()

        for a, b, c in combinations(self.elements, 3):
            for x_b in fractions:
                for x_c in fractions:
                    x_a = round(1.0 - x_b - x_c, 10)
                    if x_a > 0.0:
                        comps.append({a: x_a, b: round(x_b, 10), c: round(x_c, 10)})

        return comps

    def _interior_fractions(self) -> List[float]:
        """Fractions from step to (1 - step), excluding 0 and 1."""
        n_steps = round(1.0 / self.step)
        return [round(i * self.step, 10) for i in range(1, n_steps)]

    # ------------------------------------------------------------------
    # utility
    # ------------------------------------------------------------------

    @staticmethod
    def format_composition(comp: Dict[str, float]) -> str:
        """Human-readable label, e.g. 'W0.5-Cr0.3-C0.2'."""
        parts = []
        for el in sorted(comp, key=lambda e: -comp[e]):
            frac = comp[el]
            if frac == 1.0:
                parts.append(el)
            else:
                parts.append(f"{el}{frac:.2g}")
        return "-".join(parts)
