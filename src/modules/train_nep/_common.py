"""Shared constants and utilities for the train_nep sub-stages."""

import logging
from numbers import Real
from typing import Mapping

import numpy as np

logger = logging.getLogger("nepflow.train_nep")

VASP_COMPLETION_MARKERS = ["General timing", "Voluntary context switches"]

# Try to import tqdm for progress bars; graceful fallback if not available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    def tqdm(iterable, *args, **kwargs):
        """Fallback: return iterable as-is if tqdm is not available."""
        return iterable


def is_completed(outcar_text: str) -> bool:
    """Check if OUTCAR indicates a successfully completed VASP run."""
    tail = outcar_text[-2000:]
    return any(marker in tail for marker in VASP_COMPLETION_MARKERS)


class StructureValidationError(ValueError):
    """Raised when a parsed training structure is not label-complete."""


def validate_structure(structure: Mapping[str, object]) -> bool:
    """Validate parsed structure labels and raise a machine-readable error."""
    energy = structure.get("energy")
    if not isinstance(energy, Real) or not np.isfinite(float(energy)):
        raise StructureValidationError("missing_or_nonfinite_energy")

    species = structure.get("species")
    if not isinstance(species, (list, tuple)) or not species:
        raise StructureValidationError("missing_species")

    try:
        forces = np.asarray(structure["forces"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise StructureValidationError("missing_or_invalid_forces") from exc
    if forces.shape != (len(species), 3):
        raise StructureValidationError("forces_shape_mismatch")
    if not np.all(np.isfinite(forces)):
        raise StructureValidationError("nonfinite_forces")

    try:
        lattice = np.asarray(structure["lattice"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise StructureValidationError("missing_or_invalid_lattice") from exc
    if lattice.shape != (3, 3):
        raise StructureValidationError("lattice_shape_mismatch")
    if not np.all(np.isfinite(lattice)):
        raise StructureValidationError("nonfinite_lattice")

    virial = structure.get("virial")
    if virial is not None:
        try:
            virial_array = np.asarray(virial, dtype=float)
        except (TypeError, ValueError) as exc:
            raise StructureValidationError("invalid_virial") from exc
        if virial_array.shape != (3, 3):
            raise StructureValidationError("virial_shape_mismatch")
        if not np.all(np.isfinite(virial_array)):
            raise StructureValidationError("nonfinite_virial")

    return True
