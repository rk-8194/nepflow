"""Normal, coupled-normal, and tensor-shear strain perturbations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .models import PerturbationSettings

Annotate = Callable[..., Any]


def normal_strain_matrix(amplitude: float, axis: int) -> np.ndarray:
    """Return a diagonal deformation matrix for normal strain.

    ``amplitude`` is dimensionless and ``axis`` is zero-based.  The returned
    ``(3, 3)`` matrix has ``1 + amplitude`` on the selected diagonal entry;
    callers apply it to the cell as ``matrix @ cell``.
    """

    matrix = np.eye(3)
    matrix[axis, axis] += amplitude
    return matrix


def coupled_strain_matrix(amplitude: float, axis_a: int, axis_b: int) -> np.ndarray:
    """Return a coupled normal deformation matrix with unit determinant.

    ``amplitude`` is dimensionless and must avoid ``1 - amplitude**2 == 0``;
    the two selected axes receive opposite normal strains and the remaining
    axis compensates the volume.  The result is a dimensionless ``(3, 3)``
    matrix.
    """

    matrix = np.eye(3)
    axis_c = ({0, 1, 2} - {axis_a, axis_b}).pop()
    matrix[axis_a, axis_a] = 1.0 + amplitude
    matrix[axis_b, axis_b] = 1.0 - amplitude
    matrix[axis_c, axis_c] = 1.0 / (1.0 - amplitude * amplitude)
    return matrix


def shear_strain_matrix(amplitude: float, axis_a: int, axis_b: int) -> np.ndarray:
    """Return a symmetric tensor-shear deformation matrix.

    ``amplitude`` is the tensor shear component, not engineering shear
    ``gamma``; both off-diagonal entries are set to that value.  The result is
    dimensionless with shape ``(3, 3)``.
    """

    matrix = np.eye(3)
    matrix[axis_a, axis_b] = amplitude
    matrix[axis_b, axis_a] = amplitude
    return matrix


def elastic_stress_set(
    supercell: Any,
    base: Any,
    settings: PerturbationSettings,
    annotate: Annotate,
    *,
    slot_start: int = 0,
    slot_stop: int | None = None,
) -> list[Any]:
    """Generate enabled strain modes from a periodic cell.

    Each output cell is ``matrix @ supercell.cell`` and positions are scaled
    with the cell.  Strain amplitudes are dimensionless, modes preserve the
    declared tensor-shear convention, and provenance is attached through
    ``annotate``.  The input structures are not mutated.
    """

    if not settings.elastic_stress_enabled:
        return []
    modes = (
        ("normal_xx", normal_strain_matrix, (0,)),
        ("normal_yy", normal_strain_matrix, (1,)),
        ("normal_zz", normal_strain_matrix, (2,)),
        ("coupled_xy", coupled_strain_matrix, (0, 1)),
        ("coupled_xz", coupled_strain_matrix, (0, 2)),
        ("coupled_yz", coupled_strain_matrix, (1, 2)),
        ("shear_xy", shear_strain_matrix, (0, 1)),
        ("shear_xz", shear_strain_matrix, (0, 2)),
        ("shear_yz", shear_strain_matrix, (1, 2)),
    )

    output: list[Any] = []
    slot = 0
    end = slot_stop
    for amplitude in settings.elastic_strain_amplitudes:
        value = float(amplitude)
        if value == 0.0:
            continue
        for mode, matrix_factory, axes in modes:
            if slot < slot_start:
                slot += 1
                continue
            if end is not None and slot >= end:
                return output
            matrix = matrix_factory(value, *axes)
            strained = supercell.copy()
            strained.set_cell(matrix @ strained.cell[:], scale_atoms=True)
            annotate(
                strained,
                base,
                "elastic_stress",
                parameters={
                    "elastic_mode": mode,
                    "strain_amplitude": value,
                    "strain_matrix": (matrix - np.eye(3)).reshape(-1).tolist(),
                },
                operation_id=f"elastic:{mode}:{value}",
            )
            output.append(strained)
            slot += 1
    return output


__all__ = [
    "coupled_strain_matrix",
    "elastic_stress_set",
    "normal_strain_matrix",
    "shear_strain_matrix",
]
