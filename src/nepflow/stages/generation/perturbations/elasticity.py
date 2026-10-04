"""Reusable elastic-strain conversions and stiffness-tensor fitting.

The command-line elastic report is an operator client.  This module owns the
scientific representation and fitting rules so reports and tests use one
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from nepflow.dft.vasp.outputs import parse_stress_from_outcar

VOIGT_LABELS = ("xx", "yy", "zz", "yz", "xz", "xy")


@dataclass
class ElasticRecord:
    """One strain/stress observation used by a tensor fit."""

    group_key: str
    source_label: str
    structure_id: str
    reference_structure_id: str
    formula: str | None
    material_id: str | None
    structure_name: str | None
    elastic_mode: str
    strain_amplitude: float
    strain_voigt: np.ndarray
    stress_voigt_gpa: np.ndarray
    outcar_path: Path


def strain_matrix_to_voigt(strain_matrix: Iterable[float]) -> np.ndarray:
    matrix = np.array(list(strain_matrix), dtype=float).reshape(3, 3)
    symmetric = 0.5 * (matrix + matrix.T)
    return np.array(
        [
            symmetric[0, 0],
            symmetric[1, 1],
            symmetric[2, 2],
            2.0 * symmetric[1, 2],
            2.0 * symmetric[0, 2],
            2.0 * symmetric[0, 1],
        ],
        dtype=float,
    )


def voigt_to_tensor(voigt: Iterable[float]) -> np.ndarray:
    e1, e2, e3, e4, e5, e6 = [float(value) for value in voigt]
    return np.array(
        [[e1, 0.5 * e6, 0.5 * e5], [0.5 * e6, e2, 0.5 * e4], [0.5 * e5, 0.5 * e4, e3]],
        dtype=float,
    )


def cell_strain_voigt(reference_cell, strained_cell) -> np.ndarray:
    reference = np.array(reference_cell, dtype=float)
    strained = np.array(strained_cell, dtype=float)
    deformation_gradient = np.linalg.inv(reference) @ strained
    strain = 0.5 * (deformation_gradient + deformation_gradient.T) - np.eye(3)
    return strain_matrix_to_voigt(strain.reshape(-1))


def deformation_matrix_from_metadata(strain_matrix: Iterable[float]) -> np.ndarray:
    return np.eye(3) + np.array(list(strain_matrix), dtype=float).reshape(3, 3)


def reference_cell_from_metadata(strained_cell, strain_matrix: Iterable[float]) -> np.ndarray:
    strained = np.array(strained_cell, dtype=float)
    return np.linalg.inv(deformation_matrix_from_metadata(strain_matrix)) @ strained


def stress_matrix_to_voigt(stress_gpa: np.ndarray) -> np.ndarray:
    return np.array(
        [
            stress_gpa[0, 0],
            stress_gpa[1, 1],
            stress_gpa[2, 2],
            stress_gpa[1, 2],
            stress_gpa[0, 2],
            stress_gpa[0, 1],
        ],
        dtype=float,
    )


def tensor_to_strain_voigt(strain_tensor: np.ndarray) -> np.ndarray:
    return strain_matrix_to_voigt(np.asarray(strain_tensor).reshape(-1))


def rotate_tensor(tensor: np.ndarray, rotation_rows: np.ndarray | None) -> np.ndarray:
    if rotation_rows is None:
        return tensor
    return rotation_rows @ tensor @ rotation_rows.T


def conventional_rotation_rows(
    reference_cell: np.ndarray,
    structure_name: str | None,
) -> np.ndarray | None:
    name = (structure_name or "").lower()
    cell = np.array(reference_cell, dtype=float)
    if cell.shape != (3, 3):
        return None
    a1, a2, a3 = cell[0], cell[1], cell[2]
    if name == "bcc":
        cubic = np.array([a2 + a3, a1 + a3, a1 + a2], dtype=float)
    elif name == "fcc":
        cubic = np.array([-a1 + a2 + a3, a1 - a2 + a3, a1 + a2 - a3], dtype=float)
    else:
        return None
    norms = np.linalg.norm(cubic, axis=1)
    if np.any(norms <= 0.0):
        return None
    return cubic / norms[:, None]


def parse_stress_tensor_gpa(outcar_path: Path) -> np.ndarray | None:
    """Read VASP stress through the canonical output parser and convert to GPa."""

    stress = parse_stress_from_outcar(Path(outcar_path))
    if stress is None:
        return None
    return np.asarray(stress, dtype=float) * 160.21766208


def fit_elastic_tensor(
    records: list[ElasticRecord],
) -> tuple[np.ndarray, np.ndarray, str, np.ndarray]:
    structure_name = (records[0].structure_name or "").lower()
    if structure_name in {"bcc", "fcc"}:
        return fit_cubic_tensor(records)
    if structure_name == "hcp":
        return fit_hexagonal_tensor(records)
    return fit_general_tensor(records)


def fit_general_tensor(
    records: list[ElasticRecord],
) -> tuple[np.ndarray, np.ndarray, str, np.ndarray]:
    strains = np.array([record.strain_voigt for record in records], dtype=float)
    stresses = np.array([record.stress_voigt_gpa for record in records], dtype=float)
    if strains.ndim != 2 or strains.shape[1] != 6:
        raise ValueError("Expected 6-component Voigt strain vectors")
    if len(records) < 6:
        raise ValueError("At least 6 elastic-stress records are required")
    if np.linalg.matrix_rank(strains) < 6:
        raise ValueError("Elastic strain set is rank-deficient")

    tensor = np.zeros((6, 6), dtype=float)
    offsets = np.zeros(6, dtype=float)
    design = np.column_stack([np.ones(len(records)), strains])
    for row in range(6):
        coeffs, *_ = np.linalg.lstsq(design, stresses[:, row], rcond=None)
        offsets[row] = coeffs[0]
        tensor[row, :] = coeffs[1:]
    return 0.5 * (tensor + tensor.T), tensor, "general", offsets


def paired_strain_stress_differences(
    records: list[ElasticRecord],
) -> list[tuple[np.ndarray, np.ndarray]]:
    buckets: dict[tuple[str, float], dict[str, list[ElasticRecord]]] = {}
    for record in records:
        key = (record.elastic_mode, round(abs(record.strain_amplitude), 12))
        bucket = buckets.setdefault(key, {"positive": [], "negative": []})
        if record.strain_amplitude > 0.0:
            bucket["positive"].append(record)
        elif record.strain_amplitude < 0.0:
            bucket["negative"].append(record)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for key in sorted(buckets):
        bucket = buckets[key]
        positives = sorted(bucket["positive"], key=lambda item: str(item.outcar_path))
        negatives = sorted(bucket["negative"], key=lambda item: str(item.outcar_path))
        for positive, negative in zip(positives, negatives):
            pairs.append(
                (
                    positive.strain_voigt - negative.strain_voigt,
                    positive.stress_voigt_gpa - negative.stress_voigt_gpa,
                )
            )
    return pairs


def _cubic_tensor(c11: float, c12: float, c44: float) -> np.ndarray:
    return np.array(
        [
            [c11, c12, c12, 0.0, 0.0, 0.0],
            [c12, c11, c12, 0.0, 0.0, 0.0],
            [c12, c12, c11, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, c44, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, c44, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, c44],
        ],
        dtype=float,
    )


def cubic_offsets_from_tensor(records: list[ElasticRecord], tensor: np.ndarray) -> np.ndarray:
    strains = np.array([record.strain_voigt for record in records], dtype=float)
    stresses = np.array([record.stress_voigt_gpa for record in records], dtype=float)
    residuals = stresses - strains @ tensor.T
    sigma0_normal = float(np.mean(residuals[:, :3]))
    sigma0_shear = float(np.mean(residuals[:, 3:]))
    return np.array([sigma0_normal] * 3 + [sigma0_shear] * 3, dtype=float)


def fit_cubic_tensor(
    records: list[ElasticRecord],
) -> tuple[np.ndarray, np.ndarray, str, np.ndarray]:
    paired_differences = paired_strain_stress_differences(records)
    if paired_differences:
        design_rows: list[list[float]] = []
        targets: list[float] = []
        for strain_delta, stress_delta in paired_differences:
            e1, e2, e3, e4, e5, e6 = strain_delta
            s1, s2, s3, s4, s5, s6 = stress_delta
            design_rows.extend(
                [
                    [e1, e2 + e3, 0.0],
                    [e2, e1 + e3, 0.0],
                    [e3, e1 + e2, 0.0],
                    [0.0, 0.0, e4],
                    [0.0, 0.0, e5],
                    [0.0, 0.0, e6],
                ]
            )
            targets.extend([s1, s2, s3, s4, s5, s6])
        design = np.array(design_rows, dtype=float)
        if design.size and np.linalg.matrix_rank(design) >= 3:
            coeffs, *_ = np.linalg.lstsq(design, np.array(targets), rcond=None)
            tensor = _cubic_tensor(*coeffs.tolist())
            return tensor, tensor.copy(), "cubic", cubic_offsets_from_tensor(records, tensor)

    design_rows = []
    targets = []
    for record in records:
        e1, e2, e3, e4, e5, e6 = record.strain_voigt
        s1, s2, s3, s4, s5, s6 = record.stress_voigt_gpa
        design_rows.extend(
            [
                [1.0, 0.0, e1, e2 + e3, 0.0],
                [1.0, 0.0, e2, e1 + e3, 0.0],
                [1.0, 0.0, e3, e1 + e2, 0.0],
                [0.0, 1.0, 0.0, 0.0, e4],
                [0.0, 1.0, 0.0, 0.0, e5],
                [0.0, 1.0, 0.0, 0.0, e6],
            ]
        )
        targets.extend([s1, s2, s3, s4, s5, s6])
    coeffs, *_ = np.linalg.lstsq(np.array(design_rows), np.array(targets), rcond=None)
    sigma0_normal, sigma0_shear, c11, c12, c44 = coeffs.tolist()
    tensor = _cubic_tensor(c11, c12, c44)
    offsets = np.array([sigma0_normal] * 3 + [sigma0_shear] * 3, dtype=float)
    return tensor, tensor.copy(), "cubic", offsets


def hexagonal_offsets_from_tensor(records: list[ElasticRecord], tensor: np.ndarray) -> np.ndarray:
    strains = np.array([record.strain_voigt for record in records], dtype=float)
    stresses = np.array([record.stress_voigt_gpa for record in records], dtype=float)
    residuals = stresses - strains @ tensor.T
    sigma0_xx = float(np.mean(residuals[:, :2]))
    sigma0_zz = float(np.mean(residuals[:, 2]))
    sigma0_shear = float(np.mean(residuals[:, 3:]))
    return np.array([sigma0_xx, sigma0_xx, sigma0_zz, sigma0_shear, sigma0_shear, sigma0_shear])


def _hexagonal_tensor(c11: float, c12: float, c13: float, c33: float, c44: float) -> np.ndarray:
    c66 = 0.5 * (c11 - c12)
    return np.array(
        [
            [c11, c12, c13, 0.0, 0.0, 0.0],
            [c12, c11, c13, 0.0, 0.0, 0.0],
            [c13, c13, c33, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, c44, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, c44, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, c66],
        ],
        dtype=float,
    )


def fit_hexagonal_tensor(
    records: list[ElasticRecord],
) -> tuple[np.ndarray, np.ndarray, str, np.ndarray]:
    paired_differences = paired_strain_stress_differences(records)
    if paired_differences:
        design_rows: list[list[float]] = []
        targets: list[float] = []
        for strain_delta, stress_delta in paired_differences:
            e1, e2, e3, e4, e5, e6 = strain_delta
            s1, s2, s3, s4, s5, s6 = stress_delta
            design_rows.extend(
                [
                    [e1, e2, e3, 0.0, 0.0],
                    [e2, e1, e3, 0.0, 0.0],
                    [0.0, 0.0, e1 + e2, e3, 0.0],
                    [0.0, 0.0, 0.0, 0.0, e4],
                    [0.0, 0.0, 0.0, 0.0, e5],
                    [0.5 * e6, -0.5 * e6, 0.0, 0.0, 0.0],
                ]
            )
            targets.extend([s1, s2, s3, s4, s5, s6])
        design = np.array(design_rows, dtype=float)
        if design.size and np.linalg.matrix_rank(design) >= 5:
            coeffs, *_ = np.linalg.lstsq(design, np.array(targets), rcond=None)
            tensor = _hexagonal_tensor(*coeffs.tolist())
            return (
                tensor,
                tensor.copy(),
                "hexagonal",
                hexagonal_offsets_from_tensor(records, tensor),
            )

    design_rows = []
    targets = []
    for record in records:
        e1, e2, e3, e4, e5, e6 = record.strain_voigt
        s1, s2, s3, s4, s5, s6 = record.stress_voigt_gpa
        design_rows.extend(
            [
                [1.0, 0.0, 0.0, e1, e2, e3, 0.0, 0.0],
                [1.0, 0.0, 0.0, e2, e1, e3, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, e1 + e2, e3, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, e4],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, e5],
                [0.0, 0.0, 1.0, 0.5 * e6, -0.5 * e6, 0.0, 0.0, 0.0],
            ]
        )
        targets.extend([s1, s2, s3, s4, s5, s6])
    coeffs, *_ = np.linalg.lstsq(np.array(design_rows), np.array(targets), rcond=None)
    sigma0_xx, sigma0_zz, sigma0_shear, c11, c12, c13, c33, c44 = coeffs.tolist()
    tensor = _hexagonal_tensor(c11, c12, c13, c33, c44)
    offsets = np.array([sigma0_xx, sigma0_xx, sigma0_zz, sigma0_shear, sigma0_shear, sigma0_shear])
    return tensor, tensor.copy(), "hexagonal", offsets


def summarise_group(
    records: list[ElasticRecord],
    tensor_gpa: np.ndarray,
    raw_tensor_gpa: np.ndarray,
    fit_model: str,
    stress_offset_gpa: np.ndarray,
) -> dict:
    strains = np.array([record.strain_voigt for record in records], dtype=float)
    stresses = np.array([record.stress_voigt_gpa for record in records], dtype=float)
    predicted = strains @ tensor_gpa.T + stress_offset_gpa
    rmse = float(np.sqrt(np.mean(np.square(stresses - predicted))))
    eigenvalues = np.linalg.eigvalsh(tensor_gpa)
    max_asymmetry = float(np.max(np.abs(raw_tensor_gpa - raw_tensor_gpa.T)))
    mode_counts: dict[str, int] = {}
    for record in records:
        mode_counts[record.elastic_mode] = mode_counts.get(record.elastic_mode, 0) + 1
    warnings: list[str] = []
    if rmse > 10.0:
        warnings.append("high_rmse")
    if float(np.min(eigenvalues)) <= 0.0:
        warnings.append("non_positive_definite")
    if max_asymmetry > 5.0:
        warnings.append("large_asymmetry_before_symmetrization")
    named_constants: dict[str, float] = {}
    if fit_model == "cubic":
        named_constants = {
            "C11": float(tensor_gpa[0, 0]),
            "C12": float(tensor_gpa[0, 1]),
            "C44": float(tensor_gpa[3, 3]),
        }
    elif fit_model == "hexagonal":
        named_constants = {
            "C11": float(tensor_gpa[0, 0]),
            "C12": float(tensor_gpa[0, 1]),
            "C13": float(tensor_gpa[0, 2]),
            "C33": float(tensor_gpa[2, 2]),
            "C44": float(tensor_gpa[3, 3]),
            "C66": float(tensor_gpa[5, 5]),
        }
    return {
        "group_key": records[0].group_key,
        "source_label": records[0].source_label,
        "structure_id": records[0].structure_id,
        "formula": records[0].formula,
        "material_id": records[0].material_id,
        "structure_name": records[0].structure_name,
        "fit_model": fit_model,
        "num_records": len(records),
        "modes": mode_counts,
        "tensor_gpa": tensor_gpa.tolist(),
        "stress_offset_gpa": stress_offset_gpa.tolist(),
        "fit_rmse_gpa": rmse,
        "min_eigenvalue_gpa": float(np.min(eigenvalues)),
        "max_asymmetry_gpa": max_asymmetry,
        "named_constants_gpa": named_constants,
        "warnings": warnings,
    }


__all__ = [
    "ElasticRecord",
    "VOIGT_LABELS",
    "cell_strain_voigt",
    "conventional_rotation_rows",
    "cubic_offsets_from_tensor",
    "deformation_matrix_from_metadata",
    "fit_cubic_tensor",
    "fit_elastic_tensor",
    "fit_general_tensor",
    "fit_hexagonal_tensor",
    "hexagonal_offsets_from_tensor",
    "paired_strain_stress_differences",
    "parse_stress_tensor_gpa",
    "reference_cell_from_metadata",
    "rotate_tensor",
    "strain_matrix_to_voigt",
    "stress_matrix_to_voigt",
    "summarise_group",
    "tensor_to_strain_voigt",
    "voigt_to_tensor",
]
