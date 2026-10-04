"""Explicit units and tensor conventions used by domain records."""

from __future__ import annotations

from typing import Any

import numpy as np

ENERGY_UNIT_EV = "eV"
FORCE_UNIT_EV_PER_ANGSTROM = "eV/Angstrom"
STRESS_UNIT_EV_PER_ANGSTROM3 = "eV/Angstrom^3"
STRESS_UNIT_KBAR = "kB"
VIRIAL_UNIT_EV = "eV"
VIRIAL_CONVENTION_POSITIVE_COMPRESSION = "positive_compression"
VIRIAL_TENSOR_CONVENTION_CARTESIAN_3X3 = "cartesian_3x3"
CARTESIAN_3X3_COMPONENT_ORDER = ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz")
VOIGT_COMPONENT_ORDER = ("xx", "yy", "zz", "yz", "xz", "xy")
KBAR_PER_EV_PER_ANGSTROM3 = 1602.17663


def stress_kbar_to_ev_per_angstrom3(stress_kbar: Any) -> np.ndarray:
    """Convert VASP kB stress to eV/Angstrom^3 explicitly."""

    return np.asarray(stress_kbar, dtype=float) / KBAR_PER_EV_PER_ANGSTROM3


def virial_from_stress(stress_ev_per_a3: Any, volume_a3: float) -> np.ndarray:
    """Convert ASE/VASP stress to the positive-compression virial convention."""

    return -np.asarray(stress_ev_per_a3, dtype=float) * float(volume_a3)


def stress_from_virial(virial_ev: Any, volume_a3: float) -> np.ndarray:
    """Convert a positive-compression virial back to stress."""

    return -np.asarray(virial_ev, dtype=float) / float(volume_a3)


def require_tensor_shape(value: Any, shape: tuple[int, ...], *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array
