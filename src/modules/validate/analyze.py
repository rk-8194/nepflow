"""Post-validation analysis: compare DFT labels with genuine GPUMD output."""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from ase.atoms import Atoms
from ase.io import read as ase_read

from nepflow.domain.models import ModelRunRecord
from nepflow.mlip.gpumd import GpumdBackend
from nepflow.stages.validation.protocols import ValidationCaseSpec

logger = logging.getLogger("nepflow.validate")

_VIRIAL_COMPONENTS = ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz")


def _finite_scalar(value: object, label: str) -> float:
    """Return one finite numeric value or raise a precise validation error."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not np.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _finite_force_array(value: object, atoms_count: int, label: str) -> np.ndarray:
    """Validate a per-atom Cartesian force array in eV/Angstrom."""
    if value is None:
        raise ValueError(f"{label} is missing; GPUMD validation requires forces")
    forces = np.asarray(value, dtype=float)
    if forces.shape != (atoms_count, 3):
        raise ValueError(
            f"{label} must have shape ({atoms_count}, 3), got {forces.shape}"
        )
    if not np.isfinite(forces).all():
        raise ValueError(f"{label} contains non-finite values")
    return forces


def _tensor_from_value(value: object, label: str) -> np.ndarray:
    """Normalize a row-major 3x3 tensor and reject malformed values."""
    if value is None:
        raise ValueError(f"{label} is missing")
    tensor = np.asarray(value, dtype=float)
    if tensor.shape == (9,):
        tensor = tensor.reshape(3, 3)
    if tensor.shape != (3, 3):
        raise ValueError(f"{label} must have shape (3, 3), got {tensor.shape}")
    if not np.isfinite(tensor).all():
        raise ValueError(f"{label} contains non-finite values")
    return tensor


def _stress_to_virial(atoms: Atoms, stress: object) -> np.ndarray:
    """Convert stress in eV/Angstrom^3 to positive-compression virial in eV."""
    stress_array = np.asarray(stress, dtype=float)
    if stress_array.shape == (6,):
        xx, yy, zz, yz, xz, xy = stress_array
        stress_array = np.array(
            [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=float
        )
    stress_array = _tensor_from_value(stress_array, "stress")
    volume = float(atoms.get_volume())
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("stress-to-virial conversion requires a positive cell volume")
    return -stress_array * volume


def _extract_energy(atoms: Atoms, label: str) -> float:
    """Read the calculator-backed total energy from an extended-XYZ frame."""
    try:
        energy = atoms.get_potential_energy()
    except Exception as exc:
        raise ValueError(f"{label} is missing a readable total energy") from exc
    return _finite_scalar(energy, f"{label} energy")


def _extract_forces(atoms: Atoms, label: str) -> np.ndarray:
    """Read the force array emitted by DFT or GPUMD."""
    force_value = atoms.arrays.get("force")
    if force_value is None:
        force_value = atoms.arrays.get("forces")
    return _finite_force_array(force_value, len(atoms), f"{label} forces")


def _extract_virial(atoms: Atoms, label: str, required: bool) -> Optional[np.ndarray]:
    """Read a virial tensor, converting an explicit stress field when needed."""
    if "virial" in atoms.info:
        return _tensor_from_value(atoms.info["virial"], f"{label} virial")
    if "stress" in atoms.info:
        return _stress_to_virial(atoms, atoms.info["stress"])
    if required:
        raise ValueError(f"{label} is missing required virial/stress output")
    return None


def _composition(atoms: Atoms) -> str:
    counts: Dict[str, int] = {}
    for symbol in atoms.get_chemical_symbols():
        counts[symbol] = counts.get(symbol, 0) + 1
    return " ".join(f"{symbol}{counts[symbol]}" for symbol in sorted(counts))


def parse_dft_properties(test_xyz_path: Path) -> Dict[int, Dict]:
    """Parse required DFT energy/force labels and optional virial labels.

    Energies are eV, forces are eV/Angstrom, and virials are eV in row-major
    order using the positive-compression convention.  Stress labels are
    converted with ``virial = -stress * volume`` when supplied instead.
    """
    if not test_xyz_path.exists():
        raise FileNotFoundError(f"test.xyz not found: {test_xyz_path}")
    try:
        frames = ase_read(str(test_xyz_path), index=":", format="extxyz")
    except Exception as exc:
        raise ValueError(f"Could not parse test.xyz: {exc}") from exc
    if isinstance(frames, Atoms):
        frames = [frames]
    if not frames:
        raise ValueError(f"test.xyz contains no structures: {test_xyz_path}")

    dft_data: Dict[int, Dict] = {}
    for idx, atoms in enumerate(frames):
        dft_data[idx] = {
            "atoms_count": len(atoms),
            "energy": _extract_energy(atoms, f"DFT structure {idx}"),
            "forces": _extract_forces(atoms, f"DFT structure {idx}"),
            "virial": _extract_virial(atoms, f"DFT structure {idx}", required=False),
            "species": list(atoms.get_chemical_symbols()),
            "positions": np.asarray(atoms.positions, dtype=float),
            "cell": np.asarray(atoms.cell, dtype=float),
            "pbc": np.asarray(atoms.pbc, dtype=bool),
            "composition": _composition(atoms),
            "perturbation_family": str(
                atoms.info.get("perturbation_type", atoms.info.get("configuration_family", ""))
            ),
        }

    logger.info("Parsed DFT data for %d structures", len(dft_data))
    return dft_data


def _metrics(errors: np.ndarray) -> Tuple[float, float]:
    """Return MAE and RMSE for a non-empty finite error array."""
    values = np.asarray(errors, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Cannot calculate metrics from empty or non-finite errors")
    return float(np.mean(np.abs(values))), float(np.sqrt(np.mean(values**2)))


def generate_comparison_csv(
    validation_root: Path,
    test_xyz_path: Path | None,
    output_csv_path: Path,
    *,
    cases: Sequence[ValidationCaseSpec] | None = None,
    model: ModelRunRecord | None = None,
) -> None:
    """Generate a complete DFT-vs-model report from paired predictions.

    ``cases`` and ``model`` are required so the report cannot invent model
    identities or reconstruct geometry from a legacy ``test.xyz`` path.
    """
    if cases is None or model is None:
        raise ValueError(
            "Validation analysis requires authoritative typed cases and model record"
        )
    typed_cases = tuple(cases)
    case_by_index = {case.ordinal: case for case in typed_cases}
    dft_data = {
        case.ordinal: {
            "atoms_count": case.reference.atom_count,
            "energy": case.reference.energy_ev,
            "forces": np.asarray(case.reference.forces_ev_per_angstrom, dtype=float),
            "virial": (
                None
                if case.reference.virial_ev is None
                else np.asarray(case.reference.virial_ev, dtype=float)
            ),
            "species": list(case.reference.species),
            "positions": np.asarray(case.reference.positions_angstrom, dtype=float),
            "cell": np.asarray(case.reference.cell_angstrom, dtype=float),
            "pbc": np.asarray(case.reference.pbc, dtype=bool),
            "composition": case.reference.metadata.get("composition", "")
            if case.reference.metadata
            else "",
            "perturbation_family": case.reference.metadata.get("perturbation_family", "")
            if case.reference.metadata
            else "",
        }
        for case in typed_cases
    }
    if not dft_data:
        raise ValueError("No DFT structures are available for validation")

    virial_required = any(record.get("virial") is not None for record in dft_data.values())
    if virial_required and any(record.get("virial") is None for record in dft_data.values()):
        raise ValueError("Validation requires virial labels for every DFT structure")

    paired = []
    for struct_idx in sorted(dft_data):
        dft = dft_data[struct_idx]
        case = case_by_index.get(struct_idx)
        if case is None:
            raise ValueError(f"Validation case ordinal {struct_idx} is missing")
        struct_dir = case.working_directory
        out_xyz_path = struct_dir / "out.xyz"
        factors = case.replicates
        request = case.static_prediction_request(model)
        prediction = GpumdBackend().parse_prediction(request, out_xyz_path)
        ml_count = prediction.atom_count
        ml_positions = np.asarray(prediction.positions_angstrom, dtype=float)
        ml = {
            "energy": prediction.energy_ev,
            "forces": prediction.forces_ev_per_angstrom,
            "virial": prediction.virial_ev,
            "species": list(prediction.species or ()),
            "cell": np.asarray(prediction.cell_angstrom, dtype=float),
            "pbc": np.asarray(prediction.pbc, dtype=bool),
        }
        if prediction.atom_mapping is None:
            raise ValueError("Canonical GPUMD prediction is missing atom provenance mapping")
        reference_indices = np.asarray(prediction.atom_mapping, dtype=int)
        repeat_count = int(np.prod(factors))

        ml_forces = _finite_force_array(
            ml.get("forces"), ml_count, f"struct_{struct_idx:04d} GPUMD forces"
        )
        ml_virial = ml.get("virial")
        if virial_required and ml_virial is None:
            raise ValueError(f"struct_{struct_idx:04d}: required GPUMD virial is missing")
        paired.append(
            {
                "struct_idx": struct_idx,
                "dft": dft,
                "ml_energy": _finite_scalar(
                    ml.get("energy"), f"struct_{struct_idx:04d} GPUMD energy"
                ),
                "ml_forces": ml_forces,
                "ml_virial": ml_virial,
                "reference_indices": reference_indices,
                "repeat_count": repeat_count,
            }
        )

    energy_errors = np.array(
        [
            item["ml_energy"] / item["dft"]["atoms_count"]
            / item["repeat_count"]
            - item["dft"]["energy"] / item["dft"]["atoms_count"]
            for item in paired
        ]
    )
    energy_mae, energy_rmse = _metrics(energy_errors)
    force_errors = np.concatenate(
        [
            item["ml_forces"] - item["dft"]["forces"][item["reference_indices"]]
            for item in paired
        ],
        axis=0,
    )
    force_component_mae, force_component_rmse = _metrics(force_errors.reshape(-1))
    force_magnitude_errors = np.concatenate(
        [
            np.linalg.norm(item["ml_forces"], axis=1)
            - np.linalg.norm(item["dft"]["forces"][item["reference_indices"]], axis=1)
            for item in paired
        ]
    )
    force_magnitude_mae, force_magnitude_rmse = _metrics(force_magnitude_errors)

    virial_mae = virial_rmse = None
    if virial_required:
        virial_errors = np.concatenate(
            [
                (
                    item["ml_virial"] / item["repeat_count"]
                    - item["dft"]["virial"]
                ).reshape(-1)
                for item in paired
            ]
        )
        virial_mae, virial_rmse = _metrics(virial_errors)

    rows = []
    for item in paired:
        struct_idx = item["struct_idx"]
        dft = item["dft"]
        dft_forces = np.asarray(dft["forces"], dtype=float)[item["reference_indices"]]
        ml_forces = item["ml_forces"]
        dft_energy_per_atom = dft["energy"] / dft["atoms_count"]
        ml_energy_per_atom = item["ml_energy"] / item["repeat_count"] / dft["atoms_count"]
        for atom_idx in range(len(ml_forces)):
            row = {
                "struct_id": struct_idx,
                "atom_id": atom_idx,
                "reference_atom_id": item["reference_indices"][atom_idx],
                "species": dft["species"][item["reference_indices"][atom_idx]],
                "composition": dft.get("composition", ""),
                "perturbation_family": dft.get("perturbation_family", ""),
                "energy_unit": "eV",
                "force_unit": "eV/Angstrom",
                "energy_per_atom_dft": dft_energy_per_atom,
                "energy_per_atom_ml": ml_energy_per_atom,
                "energy_error_per_atom": ml_energy_per_atom - dft_energy_per_atom,
                "energy_mae": energy_mae,
                "energy_rmse": energy_rmse,
                "force_x_dft": dft_forces[atom_idx, 0],
                "force_y_dft": dft_forces[atom_idx, 1],
                "force_z_dft": dft_forces[atom_idx, 2],
                "force_x_ml": ml_forces[atom_idx, 0],
                "force_y_ml": ml_forces[atom_idx, 1],
                "force_z_ml": ml_forces[atom_idx, 2],
                "force_x_error": ml_forces[atom_idx, 0] - dft_forces[atom_idx, 0],
                "force_y_error": ml_forces[atom_idx, 1] - dft_forces[atom_idx, 1],
                "force_z_error": ml_forces[atom_idx, 2] - dft_forces[atom_idx, 2],
                "force_magnitude_dft": np.linalg.norm(dft_forces[atom_idx]),
                "force_magnitude_ml": np.linalg.norm(ml_forces[atom_idx]),
                "force_magnitude_error": np.linalg.norm(ml_forces[atom_idx])
                - np.linalg.norm(dft_forces[atom_idx]),
                "force_component_mae": force_component_mae,
                "force_component_rmse": force_component_rmse,
                "force_magnitude_mae": force_magnitude_mae,
                "force_magnitude_rmse": force_magnitude_rmse,
            }
            if virial_required:
                row["virial_unit"] = "eV"
                row["virial_convention"] = "positive_compression"
                for component, dft_value, ml_value in zip(
                    _VIRIAL_COMPONENTS,
                    dft["virial"].reshape(-1),
                    (item["ml_virial"] / item["repeat_count"]).reshape(-1),
                ):
                    row[f"virial_{component}_dft"] = dft_value
                    row[f"virial_{component}_ml"] = ml_value
                    row[f"virial_{component}_error"] = ml_value - dft_value
                row["virial_mae"] = virial_mae
                row["virial_rmse"] = virial_rmse
            rows.append(row)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("CSV generated: %d paired atoms from %d structures", len(rows), len(paired))


def _read_csv_rows(csv_path: Path) -> List[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Comparison CSV not found: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise ValueError(f"Comparison CSV contains no paired rows: {csv_path}")
    return rows


def _paired_column_values(
    rows: List[dict], dft_key: str, ml_key: str
) -> Tuple[List[float], List[float]]:
    if not all(dft_key in row and ml_key in row for row in rows):
        raise ValueError(f"Comparison CSV is missing paired columns {dft_key!r}/{ml_key!r}")
    dft_values = []
    ml_values = []
    for index, row in enumerate(rows):
        try:
            dft_value = _finite_scalar(row[dft_key], f"CSV row {index} {dft_key}")
            ml_value = _finite_scalar(row[ml_key], f"CSV row {index} {ml_key}")
        except ValueError as exc:
            raise ValueError(
                f"Comparison CSV row {index} lacks a complete DFT/ML pair for {dft_key}/{ml_key}"
            ) from exc
        dft_values.append(dft_value)
        ml_values.append(ml_value)
    return dft_values, ml_values


def plot_comparison_results(
    csv_path: Path,
    output_dir: Path,
    dataset_name: str,
    potential_name: str,
) -> None:
    """Generate parity plots only from complete paired DFT/ML values."""
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Plotting validation results requires matplotlib") from exc

    rows = _read_csv_rows(csv_path)
    energy_dft, energy_ml = _paired_column_values(
        rows, "energy_per_atom_dft", "energy_per_atom_ml"
    )
    force_dft, force_ml = _paired_column_values(
        rows, "force_magnitude_dft", "force_magnitude_ml"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_num = dataset_name.split("_")[-1]
    potential_num = potential_name.split("_")[-1]
    _create_scatter_plot(
        energy_dft,
        energy_ml,
        output_dir / f"dataset_{dataset_num}_potential_{potential_num}_energy.png",
        "DFT Energy (eV/atom)",
        "ML Energy (eV/atom)",
    )
    _create_scatter_plot(
        force_dft,
        force_ml,
        output_dir / f"dataset_{dataset_num}_potential_{potential_num}_force.png",
        "DFT Force (eV/Angstrom)",
        "ML Force (eV/Angstrom)",
    )

    if all(
        f"virial_{component}_dft" in rows[0] and f"virial_{component}_ml" in rows[0]
        for component in _VIRIAL_COMPONENTS
    ):
        virial_dft = []
        virial_ml = []
        for component in _VIRIAL_COMPONENTS:
            dft_values, ml_values = _paired_column_values(
                rows, f"virial_{component}_dft", f"virial_{component}_ml"
            )
            virial_dft.extend(dft_values)
            virial_ml.extend(ml_values)
        _create_scatter_plot(
            virial_dft,
            virial_ml,
            output_dir / f"dataset_{dataset_num}_potential_{potential_num}_virial.png",
            "DFT Virial (eV)",
            "ML Virial (eV)",
        )


def _create_scatter_plot(
    dft_values: List[float],
    ml_values: List[float],
    output_path: Path,
    xlabel: str,
    ylabel: str,
) -> None:
    """Create one parity plot from finite, paired values."""
    if len(dft_values) != len(ml_values) or not dft_values:
        raise ValueError(f"No complete paired data available for plot: {output_path}")
    dft_vals = np.asarray(dft_values, dtype=float)
    ml_vals = np.asarray(ml_values, dtype=float)
    if not np.isfinite(dft_vals).all() or not np.isfinite(ml_vals).all():
        raise ValueError(f"Plot data contains non-finite values: {output_path}")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    try:
        min_val = min(float(dft_vals.min()), float(ml_vals.min()))
        max_val = max(float(dft_vals.max()), float(ml_vals.max()))
        ax.scatter(dft_vals, ml_vals, alpha=0.5, s=20)
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            alpha=0.5,
            label="Perfect prediction",
        )

        if len(dft_vals) >= 2 and not np.allclose(dft_vals, dft_vals[0]):
            from scipy import stats

            slope, intercept, r_value, _, _ = stats.linregress(dft_vals, ml_vals)
            x_line = np.array([min_val, max_val])
            ax.plot(
                x_line,
                slope * x_line + intercept,
                "b-",
                alpha=0.7,
                label=f"Linear fit (R²={r_value**2:.3f})",
            )

        rmse = float(np.sqrt(np.mean((ml_vals - dft_vals) ** 2)))
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{xlabel} vs {ylabel}\nRMSE={rmse:.4f}", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
    finally:
        plt.close(fig)
