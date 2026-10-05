"""Preparation of identity-bound validation cases for GPUMD."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nepflow.errors import ValidationError
from nepflow.mlip.gpumd import GpumdBackend
from nepflow.mlip.nep.artifacts import parse_nep_cutoff_angstrom

from .protocols import ValidationCaseSpec, ValidationPreparation
from .resolution import ResolvedModelDataset, resolve_model_dataset


def cell_perpendicular_heights_angstrom(cell: Any) -> np.ndarray:
    """Return triclinic cell-plane heights in Angstroms.

    ``cell`` is interpreted as three row vectors with shape ``(3, 3)`` in
    Angstroms.  For each lattice direction, the height is
    ``abs(det(cell)) / |cross(other two vectors)|``; this is the periodic
    thickness normal to that cell face, not the norm of a lattice vector.
    The cell must be finite and non-degenerate.  Returns a finite ``(3,)``
    array in the same axis order as the input, or raises ``ValidationError``.
    """

    vectors = np.asarray(cell, dtype=float)
    if vectors.shape != (3, 3) or not np.isfinite(vectors).all():
        raise ValidationError("validation cell must be a finite 3x3 matrix")
    volume = abs(float(np.linalg.det(vectors)))
    faces = np.array(
        [
            np.linalg.norm(np.cross(vectors[1], vectors[2])),
            np.linalg.norm(np.cross(vectors[2], vectors[0])),
            np.linalg.norm(np.cross(vectors[0], vectors[1])),
        ],
        dtype=float,
    )
    if (
        not np.isfinite(volume)
        or volume <= 0
        or not np.all(np.isfinite(faces))
        or np.any(faces <= 0)
    ):
        raise ValidationError("validation cell must be non-degenerate")
    heights = volume / faces
    if not np.all(np.isfinite(heights)) or np.any(heights <= 0):
        raise ValidationError("validation cell heights must be positive and finite")
    return heights


def calculate_cell_replicates_for_cutoff(
    cell: Any,
    cutoff_angstrom: float,
) -> tuple[int, int, int]:
    """Choose repeats that make every periodic thickness exceed ``2*cutoff``.

    ``cutoff_angstrom`` is the interaction cutoff in Angstroms.  The returned
    tuple follows the cell-axis order and is the smallest deterministic set of
    positive integer repeats satisfying ``repeat[i] * height[i] > 2 * cutoff``.
    Perpendicular heights are used because a triclinic lattice vector norm
    does not measure the shortest periodic separation across a face.  Raises
    ``ValidationError`` for a non-positive/non-finite cutoff or degenerate cell.
    """

    cutoff = float(cutoff_angstrom)
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValidationError("validation cutoff must be finite and positive")
    heights = cell_perpendicular_heights_angstrom(cell)
    required = 2.0 * cutoff
    repeats = tuple(max(1, int(np.floor(required / height)) + 1) for height in heights)
    while any(repeat * height <= required for repeat, height in zip(repeats, heights)):
        repeats = tuple(
            repeat + 1 if repeat * height <= required else repeat
            for repeat, height in zip(repeats, heights)
        )
    result = tuple(int(value) for value in repeats)
    if any(repeat * height <= required for repeat, height in zip(result, heights)):
        raise ValidationError("validation replication does not satisfy cutoff thickness")
    return result[0], result[1], result[2]


def prepare_validation_cases(
    project_dir: Path,
    model_run_id: str,
    *,
    dataset_id: str | None = None,
    state_store: Any | None = None,
    gpumd_potential_dir: Path | None = None,
    backend: GpumdBackend | None = None,
) -> ValidationPreparation:
    """Resolve identities and create deterministic GPUMD validation cases.

    The model run and dataset identities are resolved from authoritative state;
    each test reference receives a stable case ordinal, an input/output path,
    and repeats sufficient for the NEP cutoff.  This function creates the
    case directories and backend inputs as an idempotent preparation side
    effect, but does not submit or execute jobs.  Raises the resolver or
    backend's typed validation errors when identities, model artifacts, or
    cell geometry are invalid.
    """

    resolved: ResolvedModelDataset = resolve_model_dataset(
        project_dir,
        model_run_id,
        dataset_id=dataset_id,
        state_store=state_store,
    )
    backend = backend or GpumdBackend()
    # Directory names are operational paths, not identities.  Keep them
    # bounded for Windows/HPC path limits; the complete IDs remain persisted
    # in ``ValidationPreparation`` and every case schema.
    default_potential_dir = (
        Path(project_dir)
        / "gpumd"
        / "validation"
        / f"{resolved.dataset_id[-16:]}-{resolved.model_run_id[-16:]}"
    )
    potential_dir = Path(gpumd_potential_dir or default_potential_dir)
    cutoff = parse_nep_cutoff_angstrom(resolved.model_path)

    cases: list[ValidationCaseSpec] = []
    references = resolved.test_references()
    for ordinal, reference in enumerate(references):
        case_dir = potential_dir / "validation" / f"case_{ordinal:04d}"
        replicates = calculate_cell_replicates_for_cutoff(
            reference.cell_angstrom,
            cutoff,
        )
        case = ValidationCaseSpec.create(
            ordinal=ordinal,
            model_run_id=resolved.model_run_id,
            dataset_id=resolved.dataset_id,
            reference=reference,
            input_path=case_dir / "model.xyz",
            working_directory=case_dir,
            output_path=case_dir / "out.xyz",
            replicates=replicates,
            virial_requested=reference.virial_ev is not None,
        )
        backend.prepare_inputs(
            case.static_prediction_request(resolved.model_run),
            replicates=replicates,
        )
        cases.append(case)

    return ValidationPreparation(
        model_run_id=resolved.model_run_id,
        dataset_id=resolved.dataset_id,
        model_path=resolved.model_path,
        dataset_path=resolved.dataset_path,
        cases=tuple(cases),
    )


__all__ = [
    "calculate_cell_replicates_for_cutoff",
    "cell_perpendicular_heights_angstrom",
    "prepare_validation_cases",
]
