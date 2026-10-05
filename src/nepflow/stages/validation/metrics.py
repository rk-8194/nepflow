"""Pure pairing and metric calculations for validation predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from nepflow.domain.units import (
    VIRIAL_CONVENTION_POSITIVE_COMPRESSION,
    VIRIAL_TENSOR_CONVENTION_CARTESIAN_3X3,
)
from nepflow.errors import ValidationError
from nepflow.mlip.simulation import StaticPrediction

from .protocols import ValidationCaseSpec


def _finite(values: Any, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValidationError(f"{label} must contain finite values")
    return result


def _mae_rmse(values: Any, label: str) -> tuple[float, float]:
    errors = _finite(values, label).reshape(-1)
    return float(np.mean(np.abs(errors))), float(np.sqrt(np.mean(errors**2)))


@dataclass(frozen=True, slots=True)
class PairedValidationCase:
    """One prediction paired to authoritative reference labels.

    Energies are in eV, forces are Cartesian eV/Angstrom with shape
    ``(n_atoms, 3)``, and optional virials are positive-compression Cartesian
    ``(3, 3)`` tensors in eV.  ``reference_indices`` maps prediction order to
    the original reference atom order; repeated-cell predictions may contain
    each reference atom ``repeat_count`` times.
    """

    case_id: str
    structure_id: str
    dft_energy_ev: float
    ml_energy_ev: float
    dft_forces_ev_per_angstrom: np.ndarray
    ml_forces_ev_per_angstrom: np.ndarray
    reference_indices: tuple[int, ...]
    repeat_count: int
    dft_virial_ev: np.ndarray | None = None
    ml_virial_ev: np.ndarray | None = None
    virial_convention: str = VIRIAL_CONVENTION_POSITIVE_COMPRESSION
    virial_tensor_convention: str = VIRIAL_TENSOR_CONVENTION_CARTESIAN_3X3

    def __post_init__(self) -> None:
        try:
            dft_energy = float(self.dft_energy_ev)
            ml_energy = float(self.ml_energy_ev)
        except (TypeError, ValueError) as exc:
            raise ValidationError("paired validation energies must be numeric") from exc
        if not np.isfinite(dft_energy) or not np.isfinite(ml_energy):
            raise ValidationError("paired validation energies must be finite")
        object.__setattr__(self, "dft_energy_ev", dft_energy)
        object.__setattr__(self, "ml_energy_ev", ml_energy)

        dft_forces = _finite(self.dft_forces_ev_per_angstrom, "DFT forces")
        ml_forces = _finite(self.ml_forces_ev_per_angstrom, "ML forces")
        if dft_forces.ndim != 2 or dft_forces.shape[1] != 3:
            raise ValidationError("paired DFT forces must have shape (n_atoms, 3)")
        if ml_forces.shape != (len(self.reference_indices), 3):
            raise ValidationError("paired ML forces do not match atom provenance")
        if any(index < 0 or index >= len(dft_forces) for index in self.reference_indices):
            raise ValidationError("paired atom provenance contains an invalid reference index")
        if self.repeat_count < 1:
            raise ValidationError("paired validation repeat_count must be positive")
        for name, value in (
            ("dft_forces_ev_per_angstrom", dft_forces),
            ("ml_forces_ev_per_angstrom", ml_forces),
        ):
            frozen = np.array(value, copy=True)
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)
        object.__setattr__(self, "reference_indices", tuple(int(i) for i in self.reference_indices))

        if self.virial_convention != VIRIAL_CONVENTION_POSITIVE_COMPRESSION:
            raise ValidationError(
                "paired validation virial must use the positive-compression convention"
            )
        if self.virial_tensor_convention != VIRIAL_TENSOR_CONVENTION_CARTESIAN_3X3:
            raise ValidationError("paired validation virial must use Cartesian 3x3 tensor ordering")

        if (self.dft_virial_ev is None) != (self.ml_virial_ev is None):
            raise ValidationError("paired validation virials must be present as a complete pair")
        if self.dft_virial_ev is not None and self.ml_virial_ev is not None:
            dft_virial = _finite(self.dft_virial_ev, "DFT virial")
            ml_virial = _finite(self.ml_virial_ev, "ML virial")
            if dft_virial.shape != (3, 3) or ml_virial.shape != (3, 3):
                raise ValidationError("paired validation virials must have shape (3, 3)")
            for name, value in (("dft_virial_ev", dft_virial), ("ml_virial_ev", ml_virial)):
                frozen = np.array(value, copy=True)
                frozen.setflags(write=False)
                object.__setattr__(self, name, frozen)


@dataclass(frozen=True, slots=True)
class ValidationMetrics:
    """Aggregate finite DFT-vs-ML metrics.

    Energy and virial fields are in eV; force fields are in eV/Angstrom.
    Component metrics use Cartesian ``x, y, z`` values, while magnitude
    metrics use the Euclidean norm per atom.  Virial metrics compare
    positive-compression Cartesian tensors after repeat normalization.
    """

    energy_mae: float
    energy_rmse: float
    force_component_mae: float
    force_component_rmse: float
    force_magnitude_mae: float
    force_magnitude_rmse: float
    virial_mae: float | None = None
    virial_rmse: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        """Return JSON-shaped metric names and numeric values."""
        return {
            "energy_mae": self.energy_mae,
            "energy_rmse": self.energy_rmse,
            "force_component_mae": self.force_component_mae,
            "force_component_rmse": self.force_component_rmse,
            "force_magnitude_mae": self.force_magnitude_mae,
            "force_magnitude_rmse": self.force_magnitude_rmse,
            "virial_mae": self.virial_mae,
            "virial_rmse": self.virial_rmse,
        }


def pair_prediction(
    case: ValidationCaseSpec,
    prediction: StaticPrediction,
) -> PairedValidationCase:
    """Pair one canonical prediction with its case reference labels.

    The backend supplies output-order provenance through ``atom_mapping``;
    this function performs no file I/O and never guesses atom order.

    The prediction must contain ``n_atoms * prod(replicates)`` atoms.  Virials,
    when requested by the reference, must use positive-compression Cartesian
    ``(3, 3)`` tensors in eV.  ``ValidationError`` is raised for identity,
    shape, provenance, or convention mismatches.
    """

    expected_count = case.atom_count * int(np.prod(case.replicates))
    if prediction.atom_count != expected_count:
        raise ValidationError(
            f"prediction atom count {prediction.atom_count} does not match case {expected_count}"
        )
    if prediction.structure.structure_id != case.structure_id:
        raise ValidationError("prediction structure identity does not match validation case")
    if prediction.model_run.model_run_id != case.model_run_id:
        raise ValidationError("prediction model identity does not match validation case")
    if prediction.atom_mapping is None:
        raise ValidationError("prediction is missing atom provenance mapping")
    if len(prediction.atom_mapping) != expected_count:
        raise ValidationError("prediction atom provenance does not match atom count")
    reference_indices = tuple(int(index) for index in prediction.atom_mapping)
    expected_repeats = int(np.prod(case.replicates))
    counts = {index: reference_indices.count(index) for index in range(case.atom_count)}
    if any(count != expected_repeats for count in counts.values()):
        raise ValidationError("prediction atom provenance does not cover each reference atom")
    if case.reference.virial_ev is not None:
        if prediction.virial_convention != VIRIAL_CONVENTION_POSITIVE_COMPRESSION:
            raise ValidationError(
                "prediction virial uses a convention incompatible with the DFT reference"
            )
        if prediction.virial_tensor_convention != VIRIAL_TENSOR_CONVENTION_CARTESIAN_3X3:
            raise ValidationError(
                "prediction virial tensor ordering is incompatible with the DFT reference"
            )

    return PairedValidationCase(
        case_id=case.case_id,
        structure_id=case.structure_id,
        dft_energy_ev=case.reference.energy_ev,
        ml_energy_ev=prediction.energy_ev,
        dft_forces_ev_per_angstrom=case.reference.forces_ev_per_angstrom,
        ml_forces_ev_per_angstrom=prediction.forces_ev_per_angstrom,
        reference_indices=reference_indices,
        repeat_count=expected_repeats,
        dft_virial_ev=case.reference.virial_ev,
        # A backend may provide an optional virial even when the authoritative
        # dataset did not request one; it must not turn an otherwise valid
        # non-virial comparison into a pairing error.
        ml_virial_ev=(prediction.virial_ev if case.reference.virial_ev is not None else None),
        virial_convention=case.reference.virial_convention,
        virial_tensor_convention=case.reference.virial_tensor_convention,
    )


def calculate_metrics(paired: Sequence[PairedValidationCase]) -> ValidationMetrics:
    """Calculate aggregate metrics from already paired scientific values.

    Energies are normalized per atom before error calculation; forces are
    compared in prediction order mapped back to reference atoms.  Virials are
    normalized by the cell repeat count but retain the positive-compression
    Cartesian convention.  No plotting or persistence occurs.

    Raises:
        ValidationError: If no cases are supplied or virial availability is
            inconsistent across cases.
    """

    records = tuple(paired)
    if not records:
        raise ValidationError("cannot calculate validation metrics without paired cases")
    energy_errors = np.asarray(
        [
            item.ml_energy_ev / (len(item.dft_forces_ev_per_angstrom) * item.repeat_count)
            - item.dft_energy_ev / len(item.dft_forces_ev_per_angstrom)
            for item in records
        ],
        dtype=float,
    )
    force_errors = np.concatenate(
        [
            item.ml_forces_ev_per_angstrom
            - item.dft_forces_ev_per_angstrom[list(item.reference_indices)]
            for item in records
        ],
        axis=0,
    )
    force_magnitude_errors = np.concatenate(
        [
            np.linalg.norm(item.ml_forces_ev_per_angstrom, axis=1)
            - np.linalg.norm(item.dft_forces_ev_per_angstrom[list(item.reference_indices)], axis=1)
            for item in records
        ]
    )
    energy_mae, energy_rmse = _mae_rmse(energy_errors, "energy errors")
    force_component_mae, force_component_rmse = _mae_rmse(force_errors, "force errors")
    force_magnitude_mae, force_magnitude_rmse = _mae_rmse(
        force_magnitude_errors, "force magnitude errors"
    )

    virial_values = [item for item in records if item.dft_virial_ev is not None]
    virial_mae = virial_rmse = None
    if virial_values:
        if len(virial_values) != len(records):
            raise ValidationError("virial labels must be present for every paired case")
        virial_pairs: list[tuple[np.ndarray, np.ndarray, int]] = []
        for item in virial_values:
            ml_virial = item.ml_virial_ev
            dft_virial = item.dft_virial_ev
            if ml_virial is None or dft_virial is None:
                raise ValidationError("virial labels must be present for every paired case")
            virial_pairs.append((ml_virial, dft_virial, item.repeat_count))
        # Both boundaries carry positive-compression Cartesian tensors in eV;
        # only replication normalization is needed before direct comparison.
        virial_errors = np.concatenate(
            [ml / repeat_count - dft for ml, dft, repeat_count in virial_pairs],
            axis=0,
        )
        virial_mae, virial_rmse = _mae_rmse(virial_errors, "virial errors")

    return ValidationMetrics(
        energy_mae=energy_mae,
        energy_rmse=energy_rmse,
        force_component_mae=force_component_mae,
        force_component_rmse=force_component_rmse,
        force_magnitude_mae=force_magnitude_mae,
        force_magnitude_rmse=force_magnitude_rmse,
        virial_mae=virial_mae,
        virial_rmse=virial_rmse,
    )


# Descriptive aliases for callers that prefer explicit verbs.
pair_prediction_to_reference = pair_prediction
calculate_validation_metrics = calculate_metrics


__all__ = [
    "PairedValidationCase",
    "ValidationMetrics",
    "calculate_metrics",
    "calculate_validation_metrics",
    "pair_prediction",
    "pair_prediction_to_reference",
]
