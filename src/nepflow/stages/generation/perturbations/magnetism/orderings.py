"""Deterministic version-one collinear magnetic ordering enumeration."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence

import numpy as np

from .topology import ParentTopology


def half_grid_propagation_vectors() -> tuple[tuple[float, float, float], ...]:
    """Return the supported half-grid q vectors in deterministic order."""

    vectors = [
        (first, second, third) for first, second, third in itertools.product((0.0, 0.5), repeat=3)
    ]
    return tuple(sorted(vectors, key=lambda q: (sum(value != 0.0 for value in q), q)))


def is_commensurate(
    propagation_vector: Sequence[float],
    transformation: Sequence[Sequence[float]],
    *,
    tolerance: float = 1.0e-8,
) -> bool:
    """Return whether ``S.T @ q`` is an integer vector."""

    q = np.asarray(tuple(float(value) for value in propagation_vector), dtype=float)
    matrix = np.asarray(transformation, dtype=float)
    if matrix.shape == (3,):
        matrix = np.diag(matrix)
    if q.shape != (3,) or matrix.shape != (3, 3):
        raise ValueError("propagation vector and transformation must have dimensions 3 and 3x3")
    if not math.isfinite(float(tolerance)) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    translated = matrix.T @ q
    return bool(np.all(np.abs(translated - np.rint(translated)) <= tolerance))


def enumerate_orbit_phases(
    orbit_indices: Iterable[int] | int,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate orbit signs with the first orbit fixed positive."""

    if isinstance(orbit_indices, int):
        orbits = tuple(range(orbit_indices))
    else:
        orbits = tuple(sorted(set(int(value) for value in orbit_indices)))
    if any(value < 0 for value in orbits):
        raise ValueError("orbit indices must be non-negative")
    if not orbits:
        return ((),)
    phases = tuple((1, *tail) for tail in itertools.product((-1, 1), repeat=len(orbits) - 1))
    return tuple(sorted(phases, key=lambda values: (sum(value < 0 for value in values), values)))


def translation_phase_sign(
    q: Sequence[float],
    fractional_position: Sequence[float],
    *,
    tolerance: float = 1.0e-8,
) -> int | None:
    """Return a real half-grid phase sign, or ``None`` for a complex phase."""

    q_array = np.asarray(tuple(float(value) for value in q), dtype=float)
    position = np.asarray(tuple(float(value) for value in fractional_position), dtype=float)
    phase = 2.0 * np.pi * float(np.dot(q_array, position))
    imaginary = math.sin(phase)
    real = math.cos(phase)
    if abs(imaginary) > tolerance or abs(real) <= tolerance:
        return None
    return 1 if real > 0.0 else -1


def signs_for_mode(
    topology: ParentTopology,
    q: Sequence[float],
    orbit_phases: Sequence[int],
    *,
    magnetic_mask: Sequence[bool],
    phase_tolerance: float,
) -> tuple[tuple[int, ...] | None, str | None]:
    """Calculate signs for mapped magnetic atoms using parent topology."""

    mask = np.asarray(magnetic_mask, dtype=bool)
    if mask.shape != topology.mapped.shape:
        raise ValueError("magnetic mask must match topology length")
    orbit_values = tuple(sorted(set(int(value) for value in topology.orbit_indices[mask])))
    if len(orbit_values) != len(orbit_phases):
        raise ValueError("orbit phase count does not match magnetic orbit count")
    phase_by_orbit = dict(zip(orbit_values, (int(value) for value in orbit_phases)))
    signs: list[int] = []
    for index in range(len(mask)):
        if not mask[index]:
            signs.append(0)
            continue
        phase = translation_phase_sign(
            q,
            topology.unwrapped_fractional[index],
            tolerance=phase_tolerance,
        )
        if phase is None:
            return None, f"parent site {index} has a non-real propagation phase"
        signs.append(phase_by_orbit[int(topology.orbit_indices[index])] * phase)
    return tuple(signs), None


__all__ = [
    "enumerate_orbit_phases",
    "half_grid_propagation_vectors",
    "is_commensurate",
    "signs_for_mode",
    "translation_phase_sign",
]
