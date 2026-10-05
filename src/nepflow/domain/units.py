"""Explicit units and tensor conventions used by domain records."""

from __future__ import annotations

from typing import Any

import numpy as np

ENERGY_UNIT_EV = "eV"
FORCE_UNIT_EV_PER_ANGSTROM = "eV/Angstrom"
STRESS_UNIT_EV_PER_ANGSTROM3 = "eV/Angstrom^3"
STRESS_UNIT_KBAR = "kB"
STRESS_UNIT_GPA = "GPa"
VIRIAL_UNIT_EV = "eV"
VIRIAL_CONVENTION_POSITIVE_COMPRESSION = "positive_compression"
VIRIAL_TENSOR_CONVENTION_CARTESIAN_3X3 = "cartesian_3x3"
CARTESIAN_3X3_COMPONENT_ORDER = ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz")
VOIGT_COMPONENT_ORDER = ("xx", "yy", "zz", "yz", "xz", "xy")
KBAR_PER_EV_PER_ANGSTROM3 = 1602.17663
GPA_PER_EV_PER_ANGSTROM3 = 160.21766208


def stress_kbar_to_ev_per_angstrom3(stress_kbar: Any) -> np.ndarray:
    """Convert a 3x3 VASP stress tensor from kB to eV/Angstrom^3.

    VASP's Cartesian matrix is kept in its native sign convention.  No
    stress-to-virial sign change is performed here; callers that need the
    NEPFlow positive-compression virial must use :func:`virial_from_stress`.
    Matrix components are ordered ``(xx, xy, xz; yx, yy, yz; zx, zy, zz)``.
    """

    stress_tensor = require_tensor_shape(stress_kbar, (3, 3), name="stress_kbar")
    return stress_tensor / KBAR_PER_EV_PER_ANGSTROM3


def stress_ev_per_angstrom3_to_gpa(stress_ev_per_angstrom3: Any) -> np.ndarray:
    """Convert a 3x3 stress tensor from eV/Angstrom^3 to GPa.

    The input and output retain the same Cartesian component order and sign;
    this is a unit conversion only, not a virial convention conversion.
    """

    stress_tensor = require_tensor_shape(
        stress_ev_per_angstrom3,
        (3, 3),
        name="stress_ev_per_angstrom3",
    )
    return stress_tensor * GPA_PER_EV_PER_ANGSTROM3


def virial_from_stress(stress_ev_per_angstrom3: Any, volume_angstrom3: float) -> np.ndarray:
    """Convert stress to a positive-compression Cartesian virial tensor.

    ``stress_ev_per_angstrom3`` is a 3x3 Cartesian stress tensor in
    eV/Angstrom^3 using the backend-native sign.  ``volume_angstrom3`` is the
    positive cell volume in Angstrom^3.  The returned 3x3 tensor is in eV and
    uses ``virial = -stress * volume``; the minus sign is the sole convention
    change at this boundary.
    """

    stress_tensor = require_tensor_shape(
        stress_ev_per_angstrom3,
        (3, 3),
        name="stress_ev_per_angstrom3",
    )
    return -stress_tensor * _require_positive_volume(volume_angstrom3)


def stress_from_virial(virial_ev: Any, volume_angstrom3: float) -> np.ndarray:
    """Convert a positive-compression Cartesian virial back to stress.

    ``virial_ev`` is a 3x3 tensor in eV and ``volume_angstrom3`` is in
    Angstrom^3.  The returned tensor is in eV/Angstrom^3 with the same
    Cartesian ordering as the input and applies ``stress = -virial / volume``.
    """

    virial_tensor = require_tensor_shape(virial_ev, (3, 3), name="virial_ev")
    return -virial_tensor / _require_positive_volume(volume_angstrom3)


def _require_positive_volume(volume_angstrom3: float) -> float:
    volume = float(volume_angstrom3)
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError("volume_angstrom3 must be finite and positive")
    return volume


def require_tensor_shape(value: Any, shape: tuple[int, ...], *, name: str) -> np.ndarray:
    """Return a finite-ness-neutral float tensor with the required shape."""

    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array
