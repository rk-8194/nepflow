"""Fit elastic tensors from selected single-element elastic strain records.

This utility streams the selected ``train.xyz`` / ``test.xyz`` files that feed
the ``run_vasp`` stage, filters single-element ``elastic_stress`` structures,
resolves their corresponding VASP job folders, extracts final stress tensors
from completed ``OUTCAR`` files, and fits 6x6 elastic stiffness tensors in GPa.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


VOIGT_LABELS = ("xx", "yy", "zz", "yz", "xz", "xy")
DEFAULT_DATASETS = ("train", "test")
PROGRESS_EVERY_FRAMES = 500
PREVIEW_LIMIT = 8
MATCHED_PREVIEW_LIMIT = 5
VASP_COMPLETION_MARKERS = ("General timing", "Voluntary context switches")
STRUCTURE_HASH_VERSION = "structure-v1"
STRESS_PATTERN = re.compile(
    r"STRESS\s+in cartesian coordinates \(kB\)\n"
    r"\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)\n"
    r"\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)\n"
    r"\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)"
)
FLOAT_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?")


@dataclass
class LocalJobRecord:
    dataset: str
    struct_dir: Path
    identity: tuple[str, str, str]
    status: str
    reused_from: Path | None
    outcar_path: Path | None
    outcar_complete: bool


@dataclass
class ElasticRecord:
    group_key: str
    source_label: str
    structure_hash: str
    reference_hash: str
    formula: str | None
    material_id: str | None
    structure_name: str | None
    elastic_mode: str
    strain_amplitude: float
    strain_voigt: np.ndarray
    stress_voigt_gpa: np.ndarray
    outcar_path: Path


@dataclass
class UnresolvedCandidate:
    dataset: str
    structure_index: int
    group_key: str
    source_label: str
    elastic_mode: str
    strain_amplitude: float
    struct_dir: Path
    status: str
    outcar_path: Path | None
    outcar_complete: bool
    registry_path: Path | None
    registry_complete: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit elastic tensors (GPa) from selected elastic-stress structures."
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        required=True,
        help="Project directory, for example projects/project_wcryzr.",
    )
    parser.add_argument(
        "--structures",
        type=Path,
        help="Optional single XYZ file to stream instead of selected train/test files.",
    )
    parser.add_argument(
        "--dataset",
        choices=("train", "test", "all"),
        default="all",
        help="Which selected datasets / VASP job subdirectories to search.",
    )
    parser.add_argument(
        "--min-records",
        type=int,
        default=6,
        help="Minimum elastic-stress records required before fitting a tensor.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional JSON file for machine-readable fitted tensors.",
    )
    return parser.parse_args()


def get_dataset_names(dataset_arg: str) -> tuple[str, ...]:
    if dataset_arg == "all":
        return DEFAULT_DATASETS
    return (dataset_arg,)


def default_selected_dir(project_dir: Path) -> Path:
    return project_dir / "structures" / "selected"


def iter_structures(structures_path: Path):
    from ase.io import iread

    yield from iread(str(structures_path), index=":")


def is_single_element_structure(atoms) -> bool:
    composition = dict(getattr(atoms, "info", {}) or {}).get("composition")
    if isinstance(composition, dict):
        active = [key for key, value in composition.items() if float(value) > 0.0]
        return len(active) == 1
    return len(set(atoms.get_chemical_symbols())) == 1


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
        [
            [e1, 0.5 * e6, 0.5 * e5],
            [0.5 * e6, e2, 0.5 * e4],
            [0.5 * e5, 0.5 * e4, e3],
        ],
        dtype=float,
    )


def cell_strain_voigt(reference_cell, strained_cell) -> np.ndarray:
    reference = np.array(reference_cell, dtype=float)
    strained = np.array(strained_cell, dtype=float)
    # ASE stores cell vectors as rows and NEPFlow generates strained cells as
    # C_strained = M @ C_reference. In this row-vector convention, the
    # Cartesian deformation gradient is F = inv(C_reference) @ C_strained.
    deformation_gradient = np.linalg.inv(reference) @ strained
    strain = 0.5 * (deformation_gradient + deformation_gradient.T) - np.eye(3)
    return np.array(
        [
            strain[0, 0],
            strain[1, 1],
            strain[2, 2],
            2.0 * strain[1, 2],
            2.0 * strain[0, 2],
            2.0 * strain[0, 1],
        ],
        dtype=float,
    )


def deformation_matrix_from_metadata(strain_matrix: Iterable[float]) -> np.ndarray:
    return np.eye(3) + np.array(list(strain_matrix), dtype=float).reshape(3, 3)


def reference_cell_from_metadata(strained_cell, strain_matrix: Iterable[float]) -> np.ndarray:
    strained = np.array(strained_cell, dtype=float)
    deformation = deformation_matrix_from_metadata(strain_matrix)
    return np.linalg.inv(deformation) @ strained


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
    return np.array(
        [
            strain_tensor[0, 0],
            strain_tensor[1, 1],
            strain_tensor[2, 2],
            2.0 * strain_tensor[1, 2],
            2.0 * strain_tensor[0, 2],
            2.0 * strain_tensor[0, 1],
        ],
        dtype=float,
    )


def rotate_tensor(tensor: np.ndarray, rotation_rows: np.ndarray | None) -> np.ndarray:
    if rotation_rows is None:
        return tensor
    return rotation_rows @ tensor @ rotation_rows.T


def conventional_rotation_rows(reference_cell: np.ndarray, structure_name: str | None) -> np.ndarray | None:
    name = (structure_name or "").lower()
    cell = np.array(reference_cell, dtype=float)
    if cell.shape != (3, 3):
        return None

    a1, a2, a3 = cell[0], cell[1], cell[2]
    if name == "bcc":
        cubic = np.array(
            [
                a2 + a3,
                a1 + a3,
                a1 + a2,
            ],
            dtype=float,
        )
    elif name == "fcc":
        cubic = np.array(
            [
                -a1 + a2 + a3,
                a1 - a2 + a3,
                a1 + a2 - a3,
            ],
            dtype=float,
        )
    else:
        return None

    norms = np.linalg.norm(cubic, axis=1)
    if np.any(norms <= 0.0):
        return None
    return cubic / norms[:, None]


def parse_stress_tensor_gpa(outcar_path: Path) -> np.ndarray | None:
    # First try ASE's OUTCAR reader, which handles more VASP variants than a
    # handwritten regex.
    try:
        from ase.io import read as ase_read

        atoms = ase_read(str(outcar_path), format="vasp-out")
        stress = atoms.get_stress(voigt=False)
        # ASE returns stress in eV/Ang^3. Convert directly to GPa.
        return np.array(stress, dtype=float) * 160.21766208
    except Exception:
        pass

    try:
        text = outcar_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    matches = list(STRESS_PATTERN.finditer(text))
    if not matches:
        # Fallback for alternate VASP formatting: find a STRESS header and read
        # the next three numeric rows even if spacing differs from the canonical block.
        lines = text.splitlines()
        last_matrix: np.ndarray | None = None
        for idx, line in enumerate(lines):
            if "STRESS" not in line.upper() or "KB" not in line.upper():
                continue
            rows: list[list[float]] = []
            for probe in lines[idx + 1: idx + 8]:
                numbers = [float(value) for value in FLOAT_PATTERN.findall(probe)]
                if len(numbers) >= 3:
                    rows.append(numbers[:3])
                if len(rows) == 3:
                    break
            if len(rows) == 3:
                last_matrix = np.array(rows, dtype=float)
        if last_matrix is not None:
            return last_matrix * 0.1

        # Another common VASP layout prints six stress components on a single line.
        for line in reversed(lines):
            upper = line.upper()
            if "KB" not in upper or "STRESS" not in upper:
                continue
            numbers = [float(value) for value in FLOAT_PATTERN.findall(line)]
            if len(numbers) >= 6:
                xx, yy, zz, xy, yz, zx = numbers[-6:]
                return np.array(
                    [
                        [xx, xy, zx],
                        [xy, yy, yz],
                        [zx, yz, zz],
                    ],
                    dtype=float,
                ) * 0.1
        return None

    values = [float(matches[-1].group(i + 1)) for i in range(9)]
    return np.array(values, dtype=float).reshape(3, 3) * 0.1


def build_group_key(atoms) -> str:
    metadata = dict(getattr(atoms, "info", {}) or {})
    for key in ("seed_id", "material_id", "source"):
        value = metadata.get(key)
        if value:
            return str(value)
    symbols = "-".join(sorted(set(atoms.get_chemical_symbols())))
    return f"{symbols}:{metadata.get('structure_name', 'unknown')}"


def build_reference_hash(atoms, reference_cell: np.ndarray) -> str:
    reference_atoms = atoms.copy()
    reference_atoms.set_cell(reference_cell, scale_atoms=True)
    return hash_structure(reference_atoms)


def build_source_label(metadata: dict) -> str:
    for key in ("seed_id", "material_id", "source", "formula"):
        value = metadata.get(key)
        if value:
            return str(value)
    return "unknown"


def read_json_dict(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def get_nepflow_root(project_dir: Path) -> Path:
    if project_dir.name.startswith("project_") and project_dir.parent.name == "projects":
        return project_dir.parent.parent
    return project_dir


def read_completed_registry(nepflow_root: Path) -> dict:
    path = nepflow_root / ".vasp_completed_jobs.json"
    if not path.exists():
        return {"jobs": {}}
    return read_json_dict(path) or {"jobs": {}}


def strip_resource_incar_params(incar_text: str) -> str:
    kept = []
    for line in incar_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            kept.append(line.rstrip())
            continue
        if re.match(r"^(NCORE|KPAR)\s*=", stripped, re.IGNORECASE):
            continue
        kept.append(line.rstrip())
    return "\n".join(kept).rstrip() + "\n"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_incar_text(incar_text: str) -> str:
    return sha256_bytes(strip_resource_incar_params(incar_text).encode("utf-8"))


def hash_potcar_bytes(potcar_bytes: bytes) -> str:
    return sha256_bytes(potcar_bytes)


def format_vector(values) -> str:
    return " ".join(f"{float(values[i]):.14f}" for i in range(3))


def pbc_values(atoms) -> list[bool]:
    pbc = getattr(atoms, "pbc", (True, True, True))
    if hasattr(pbc, "tolist"):
        pbc = pbc.tolist()
    if isinstance(pbc, bool):
        return [pbc, pbc, pbc]
    values = list(pbc)
    if len(values) == 0:
        return [True, True, True]
    if len(values) == 1:
        return [bool(values[0])] * 3
    return [bool(value) for value in values[:3]]


def canonical_structure_text(atoms) -> str:
    symbols = list(atoms.get_chemical_symbols())
    unique_elements = sorted(set(symbols))
    scaled_positions = atoms.get_scaled_positions()
    cell = atoms.get_cell()
    pbc = pbc_values(atoms)

    lines = [STRUCTURE_HASH_VERSION]
    lines.append("pbc " + " ".join("1" if value else "0" for value in pbc))
    lines.append("cell")
    for row in cell:
        lines.append(format_vector(row))
    lines.append("atoms")
    for elem in unique_elements:
        indices = [i for i, symbol in enumerate(symbols) if symbol == elem]
        for idx in indices:
            lines.append(f"{elem} {format_vector(scaled_positions[idx])}")
    return "\n".join(lines) + "\n"


def hash_structure(atoms) -> str:
    return sha256_bytes(canonical_structure_text(atoms).encode("utf-8"))


def outcar_is_complete(outcar_path: Path) -> bool:
    if not outcar_path.exists():
        return False
    try:
        with open(outcar_path, "r", encoding="utf-8", errors="replace") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - 50_000))
            tail = handle.read()
        return any(marker in tail for marker in VASP_COMPLETION_MARKERS)
    except OSError:
        return False


def read_status(struct_dir: Path) -> dict:
    status_file = struct_dir / ".vasp_status"
    if status_file.exists():
        try:
            return json.loads(status_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"status": "pending", "retry_level": 0}


def inject_incar_defaults(incar_text: str, config: ConfigParser) -> str:
    lines = incar_text.rstrip("\n")
    has_kspacing = bool(re.search(r"^\s*KSPACING\s*=", incar_text, re.MULTILINE | re.IGNORECASE))
    has_kgamma = bool(re.search(r"^\s*KGAMMA\s*=", incar_text, re.MULTILINE | re.IGNORECASE))

    additions = []
    if not has_kspacing:
        additions.append(f"KSPACING = {config.get('vasp', 'kspacing', fallback='0.30')}")
    if not has_kgamma:
        additions.append(f"KGAMMA = {config.get('vasp', 'kgamma', fallback='.TRUE.')}")

    if additions:
        lines += "\n\n# --- Injected by nepflow (not in user template) ---\n"
        lines += "\n".join(additions) + "\n"
    else:
        lines += "\n"
    return lines


def parse_incar_settings(incar_text: str) -> dict[str, str]:
    settings: dict[str, str] = {}
    for raw_line in incar_text.splitlines():
        stripped = raw_line.split("#", 1)[0].split("!", 1)[0].strip()
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        settings[key.strip().upper()] = value.strip()
    return settings


def incar_elastic_warnings(settings: dict[str, str]) -> list[str]:
    warnings: list[str] = []

    nsw_text = settings.get("NSW")
    if nsw_text is not None:
        try:
            if int(float(nsw_text)) > 0:
                warnings.append(f"NSW={nsw_text} allows ionic steps during elastic jobs")
        except ValueError:
            warnings.append(f"NSW={nsw_text} could not be parsed")

    ibrion_text = settings.get("IBRION")
    if ibrion_text is not None:
        try:
            if int(float(ibrion_text)) != -1:
                warnings.append(f"IBRION={ibrion_text} is not a strict static setting")
        except ValueError:
            warnings.append(f"IBRION={ibrion_text} could not be parsed")

    isif_text = settings.get("ISIF")
    if isif_text is not None:
        try:
            isif_value = int(float(isif_text))
            if isif_value >= 3:
                warnings.append(f"ISIF={isif_text} allows cell relaxation during elastic jobs")
            elif isif_value == 2:
                warnings.append(f"ISIF={isif_text} computes stress but still permits ionic relaxation if NSW>0")
        except ValueError:
            warnings.append(f"ISIF={isif_text} could not be parsed")

    if "ISYM" in settings and settings["ISYM"] != "0":
        warnings.append(f"ISYM={settings['ISYM']} may symmetrize stresses in strained cells")

    return warnings


def load_input_context(project_dir: Path) -> dict:
    vasp_config_dir = project_dir / "config" / "vasp"
    incar_template = vasp_config_dir / "INCAR"
    if not incar_template.exists():
        raise FileNotFoundError(
            f"Could not build VASP input context under {project_dir / 'config' / 'vasp'}"
        )

    config = ConfigParser()
    for candidate in (
        project_dir / "config" / "project.config",
        project_dir / "config" / f"{project_dir.name.removeprefix('project_')}.ini",
    ):
        if candidate.exists():
            config.read(candidate)
            break

    incar_text = inject_incar_defaults(incar_template.read_text(encoding="utf-8"), config)
    incar_settings = parse_incar_settings(incar_text)
    potcar_data = {
        potcar_path.name.split("_", 1)[1]: potcar_path.read_bytes()
        for potcar_path in vasp_config_dir.glob("POTCAR_*")
    }
    return {
        "incar_hash": hash_incar_text(incar_text),
        "incar_settings": incar_settings,
        "incar_warnings": incar_elastic_warnings(incar_settings),
        "potcar_data": potcar_data,
        "registry": read_completed_registry(get_nepflow_root(project_dir)),
    }


def structure_identity_tuple(atoms, input_context: dict) -> tuple[str, str, str]:
    structure_hash = hash_structure(atoms)
    potcar_data = input_context["potcar_data"]
    struct_elements = sorted(set(atoms.get_chemical_symbols()))
    missing = [elem for elem in struct_elements if elem not in potcar_data]
    if missing:
        raise FileNotFoundError(f"missing POTCAR files for: {', '.join(missing)}")

    potcar_hash = hash_potcar_bytes(b"".join(potcar_data[elem] for elem in struct_elements))
    return (structure_hash, input_context["incar_hash"], potcar_hash)


def resolve_registry_outcar(identity: tuple[str, str, str], input_context: dict) -> Path | None:
    structure_hash, incar_hash, potcar_hash = identity
    entry = (
        input_context.get("registry", {})
        .get("jobs", {})
        .get(incar_hash, {})
        .get(potcar_hash, {})
        .get(structure_hash)
    )
    if not isinstance(entry, dict):
        return None
    job_path = entry.get("job_path")
    if not job_path:
        return None
    outcar = Path(job_path) / "OUTCAR"
    return outcar if outcar_is_complete(outcar) else None


def resolve_local_job(project_dir: Path, dataset: str, structure_index: int) -> LocalJobRecord:
    struct_dir = project_dir / "vasp" / "jobs" / dataset / f"struct_{structure_index:04d}"
    identity = read_json_dict(struct_dir / ".vasp_identity")
    status = read_status(struct_dir)
    reused_from_raw = status.get("reused_from")
    reused_from = Path(reused_from_raw) if reused_from_raw else None

    direct_outcar = struct_dir / "OUTCAR"
    reused_outcar = reused_from / "OUTCAR" if reused_from is not None else None

    outcar_path: Path | None = None
    outcar_complete = False
    if reused_outcar is not None and outcar_is_complete(reused_outcar):
        outcar_path = reused_outcar
        outcar_complete = True
    elif outcar_is_complete(direct_outcar):
        outcar_path = direct_outcar
        outcar_complete = True
    elif reused_outcar is not None and reused_outcar.exists():
        outcar_path = reused_outcar
    elif direct_outcar.exists():
        outcar_path = direct_outcar

    return LocalJobRecord(
        dataset=dataset,
        struct_dir=struct_dir,
        identity=(
            str(identity.get("structure_hash", "")),
            str(identity.get("incar_hash", "")),
            str(identity.get("potcar_hash", "")),
        ),
        status=str(status.get("status", "pending")),
        reused_from=reused_from,
        outcar_path=outcar_path,
        outcar_complete=outcar_complete,
    )


def choose_completed_outcar(
    identity: tuple[str, str, str],
    local_job: LocalJobRecord,
    input_context: dict,
) -> Path | None:
    if local_job.outcar_path is not None and (
        local_job.outcar_complete or local_job.status in {"completed", "reused"}
    ):
        return local_job.outcar_path
    return resolve_registry_outcar(identity, input_context)


def fit_elastic_tensor(records: list[ElasticRecord]) -> tuple[np.ndarray, np.ndarray, str]:
    structure_name = (records[0].structure_name or "").lower()
    if structure_name in {"bcc", "fcc"}:
        return fit_cubic_tensor(records)
    if structure_name == "hcp":
        return fit_hexagonal_tensor(records)
    return fit_general_tensor(records)


def fit_general_tensor(records: list[ElasticRecord]) -> tuple[np.ndarray, np.ndarray, str]:
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


def paired_strain_stress_differences(records: list[ElasticRecord]) -> list[tuple[np.ndarray, np.ndarray]]:
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
            strain_delta = positive.strain_voigt - negative.strain_voigt
            stress_delta = positive.stress_voigt_gpa - negative.stress_voigt_gpa
            pairs.append((strain_delta, stress_delta))
    return pairs


def cubic_offsets_from_tensor(records: list[ElasticRecord], tensor: np.ndarray) -> np.ndarray:
    strains = np.array([record.strain_voigt for record in records], dtype=float)
    stresses = np.array([record.stress_voigt_gpa for record in records], dtype=float)
    residuals = stresses - strains @ tensor.T
    sigma0_normal = float(np.mean(residuals[:, :3]))
    sigma0_shear = float(np.mean(residuals[:, 3:]))
    return np.array(
        [
            sigma0_normal,
            sigma0_normal,
            sigma0_normal,
            sigma0_shear,
            sigma0_shear,
            sigma0_shear,
        ],
        dtype=float,
    )


def hexagonal_offsets_from_tensor(records: list[ElasticRecord], tensor: np.ndarray) -> np.ndarray:
    strains = np.array([record.strain_voigt for record in records], dtype=float)
    stresses = np.array([record.stress_voigt_gpa for record in records], dtype=float)
    residuals = stresses - strains @ tensor.T
    sigma0_xx = float(np.mean(residuals[:, :2]))
    sigma0_zz = float(np.mean(residuals[:, 2]))
    sigma0_shear = float(np.mean(residuals[:, 3:]))
    return np.array(
        [
            sigma0_xx,
            sigma0_xx,
            sigma0_zz,
            sigma0_shear,
            sigma0_shear,
            sigma0_shear,
        ],
        dtype=float,
    )


def fit_cubic_tensor(records: list[ElasticRecord]) -> tuple[np.ndarray, np.ndarray, str, np.ndarray]:
    paired_differences = paired_strain_stress_differences(records)
    if paired_differences:
        design_rows: list[list[float]] = []
        targets: list[float] = []
        for strain_delta, stress_delta in paired_differences:
            e1, e2, e3, e4, e5, e6 = strain_delta
            s1, s2, s3, s4, s5, s6 = stress_delta
            design_rows.extend([
                [e1, e2 + e3, 0.0],
                [e2, e1 + e3, 0.0],
                [e3, e1 + e2, 0.0],
                [0.0, 0.0, e4],
                [0.0, 0.0, e5],
                [0.0, 0.0, e6],
            ])
            targets.extend([s1, s2, s3, s4, s5, s6])

        design = np.array(design_rows, dtype=float)
        target = np.array(targets, dtype=float)
        if design.size and np.linalg.matrix_rank(design) >= 3:
            coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
            c11, c12, c44 = coeffs.tolist()
            tensor = np.array(
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
            offsets = cubic_offsets_from_tensor(records, tensor)
            return tensor, tensor.copy(), "cubic", offsets

    design_rows: list[list[float]] = []
    targets: list[float] = []
    for record in records:
        e1, e2, e3, e4, e5, e6 = record.strain_voigt
        s1, s2, s3, s4, s5, s6 = record.stress_voigt_gpa
        design_rows.extend([
            [1.0, 0.0, e1, e2 + e3, 0.0],
            [1.0, 0.0, e2, e1 + e3, 0.0],
            [1.0, 0.0, e3, e1 + e2, 0.0],
            [0.0, 1.0, 0.0, 0.0, e4],
            [0.0, 1.0, 0.0, 0.0, e5],
            [0.0, 1.0, 0.0, 0.0, e6],
        ])
        targets.extend([s1, s2, s3, s4, s5, s6])

    design = np.array(design_rows, dtype=float)
    target = np.array(targets, dtype=float)
    coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
    sigma0_normal, sigma0_shear, c11, c12, c44 = coeffs.tolist()
    tensor = np.array(
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
    offsets = np.array(
        [
            sigma0_normal,
            sigma0_normal,
            sigma0_normal,
            sigma0_shear,
            sigma0_shear,
            sigma0_shear,
        ],
        dtype=float,
    )
    return tensor, tensor.copy(), "cubic", offsets


def fit_hexagonal_tensor(records: list[ElasticRecord]) -> tuple[np.ndarray, np.ndarray, str, np.ndarray]:
    paired_differences = paired_strain_stress_differences(records)
    if paired_differences:
        design_rows: list[list[float]] = []
        targets: list[float] = []
        for strain_delta, stress_delta in paired_differences:
            e1, e2, e3, e4, e5, e6 = strain_delta
            s1, s2, s3, s4, s5, s6 = stress_delta
            design_rows.extend([
                [e1, e2, e3, 0.0, 0.0],
                [e2, e1, e3, 0.0, 0.0],
                [0.0, 0.0, e1 + e2, e3, 0.0],
                [0.0, 0.0, 0.0, 0.0, e4],
                [0.0, 0.0, 0.0, 0.0, e5],
                [0.5 * e6, -0.5 * e6, 0.0, 0.0, 0.0],
            ])
            targets.extend([s1, s2, s3, s4, s5, s6])

        design = np.array(design_rows, dtype=float)
        target = np.array(targets, dtype=float)
        if design.size and np.linalg.matrix_rank(design) >= 5:
            coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
            c11, c12, c13, c33, c44 = coeffs.tolist()
            c66 = 0.5 * (c11 - c12)
            tensor = np.array(
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
            offsets = hexagonal_offsets_from_tensor(records, tensor)
            return tensor, tensor.copy(), "hexagonal", offsets

    design_rows: list[list[float]] = []
    targets: list[float] = []
    for record in records:
        e1, e2, e3, e4, e5, e6 = record.strain_voigt
        s1, s2, s3, s4, s5, s6 = record.stress_voigt_gpa
        design_rows.extend([
            [1.0, 0.0, 0.0, e1, e2, e3, 0.0, 0.0],
            [1.0, 0.0, 0.0, e2, e1, e3, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, e1 + e2, e3, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, e4],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, e5],
            [0.0, 0.0, 1.0, 0.5 * e6, -0.5 * e6, 0.0, 0.0, 0.0],
        ])
        targets.extend([s1, s2, s3, s4, s5, s6])

    design = np.array(design_rows, dtype=float)
    target = np.array(targets, dtype=float)
    coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
    sigma0_xx, sigma0_zz, sigma0_shear, c11, c12, c13, c33, c44 = coeffs.tolist()
    c66 = 0.5 * (c11 - c12)
    tensor = np.array(
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
    offsets = np.array(
        [
            sigma0_xx,
            sigma0_xx,
            sigma0_zz,
            sigma0_shear,
            sigma0_shear,
            sigma0_shear,
        ],
        dtype=float,
    )
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
        "structure_hash": records[0].structure_hash,
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


def format_tensor(tensor_gpa: np.ndarray) -> str:
    header = "          " + "".join(f"{label:>12s}" for label in VOIGT_LABELS)
    rows = [header]
    for idx, label in enumerate(VOIGT_LABELS):
        values = "".join(f"{tensor_gpa[idx, col]:12.3f}" for col in range(6))
        rows.append(f"{label:>8s}{values}")
    return "\n".join(rows)


def format_named_constants(result: dict) -> str | None:
    named = result.get("named_constants_gpa") or {}
    if not named:
        return None
    ordered_keys = (
        ("C11", "C12", "C44")
        if result.get("fit_model") == "cubic"
        else ("C11", "C12", "C13", "C33", "C44", "C66")
    )
    parts = [
        f"{key}={named[key]:.3f}"
        for key in ordered_keys
        if key in named
    ]
    return "  constants_gpa=" + " ".join(parts)


def format_stress_offset(result: dict) -> str:
    offset = result.get("stress_offset_gpa") or [0.0] * 6
    labels = ("sxx", "syy", "szz", "syz", "sxz", "sxy")
    return "  stress_offset_gpa=" + " ".join(
        f"{label}={float(value):.3f}" for label, value in zip(labels, offset)
    )


def debug_status_lines(structure_sources: list[Path], unresolved: list[UnresolvedCandidate]) -> list[str]:
    lines = [
        (
            "Status: "
            f"structures={','.join(str(path) for path in structure_sources)} "
            f"unresolved_elastic_candidates={len(unresolved)}"
        )
    ]
    for candidate in unresolved[:PREVIEW_LIMIT]:
        lines.append(
            "  "
            f"dataset={candidate.dataset} index={candidate.structure_index} "
            f"source={candidate.source_label} "
            f"mode={candidate.elastic_mode} amp={candidate.strain_amplitude} "
            f"struct_dir={candidate.struct_dir} "
            f"status={candidate.status} "
            f"outcar_path={candidate.outcar_path} "
            f"outcar_complete={candidate.outcar_complete} "
            f"registry_complete={candidate.registry_complete} "
            f"registry_path={candidate.registry_path}"
        )
    return lines


def stream_elastic_records(
    project_dir: Path,
    structures_override: Path | None,
    datasets: tuple[str, ...],
) -> tuple[list[ElasticRecord], dict[str, int], list[UnresolvedCandidate], list[Path]]:
    print(f"Building VASP input context from {project_dir}")
    input_context = load_input_context(project_dir)
    incar_warnings = input_context.get("incar_warnings", [])
    incar_settings = input_context.get("incar_settings", {})
    if incar_settings:
        summary_keys = ("NSW", "IBRION", "ISIF", "ISYM")
        summary = " ".join(
            f"{key}={incar_settings[key]}" for key in summary_keys if key in incar_settings
        )
        if summary:
            print(f"VASP elastic run settings: {summary}")
    for warning in incar_warnings:
        print(f"  warning: {warning}")

    selected_dir = default_selected_dir(project_dir)
    if structures_override is not None:
        structure_sources = [structures_override]
        stream_plan = [(datasets[0], structures_override)]
    else:
        structure_sources = [selected_dir / f"{dataset}.xyz" for dataset in datasets]
        stream_plan = [(dataset, selected_dir / f"{dataset}.xyz") for dataset in datasets]

    print(f"Streaming structures from {', '.join(str(path) for path in structure_sources)}")

    stats = {
        "frames_seen": 0,
        "elastic_candidates": 0,
        "single_element_candidates": 0,
        "matched_completed_records": 0,
        "missing_metadata": 0,
        "identity_errors": 0,
        "unresolved_outcar": 0,
        "unparsed_stress": 0,
        "identity_mismatches": 0,
        "reference_reconstruction_failures": 0,
    }
    records: list[ElasticRecord] = []
    unresolved: list[UnresolvedCandidate] = []

    for dataset, structure_path in stream_plan:
        if not structure_path.exists():
            print(f"  structure file missing for dataset={dataset}: {structure_path}")
            continue

        print(f"Processing dataset={dataset} file={structure_path}")
        for structure_index, atoms in enumerate(iter_structures(structure_path)):
            stats["frames_seen"] += 1
            if stats["frames_seen"] % PROGRESS_EVERY_FRAMES == 0:
                print(
                    "  progress: "
                    f"frames={stats['frames_seen']} "
                    f"elastic={stats['elastic_candidates']} "
                    f"single_element={stats['single_element_candidates']} "
                    f"matched={stats['matched_completed_records']} "
                    f"unresolved={stats['unresolved_outcar']} "
                    f"stress_parse_failures={stats['unparsed_stress']}"
                )

            metadata = dict(getattr(atoms, "info", {}) or {})
            if str(metadata.get("perturbation_type", "")) != "elastic_stress":
                continue
            stats["elastic_candidates"] += 1

            if not is_single_element_structure(atoms):
                continue
            stats["single_element_candidates"] += 1

            strain_matrix = metadata.get("strain_matrix")
            elastic_mode = metadata.get("elastic_mode")
            strain_amplitude = metadata.get("strain_amplitude")
            if strain_matrix is None or elastic_mode is None or strain_amplitude is None:
                stats["missing_metadata"] += 1
                if stats["missing_metadata"] <= PREVIEW_LIMIT:
                    print(
                        "  skipping elastic candidate with missing metadata: "
                        f"dataset={dataset} index={structure_index} "
                        f"source={build_source_label(metadata)}"
                    )
                continue

            try:
                identity = structure_identity_tuple(atoms, input_context)
            except FileNotFoundError as exc:
                stats["identity_errors"] += 1
                if stats["identity_errors"] <= PREVIEW_LIMIT:
                    print(
                        "  failed to build structure identity: "
                        f"dataset={dataset} index={structure_index} "
                        f"source={build_source_label(metadata)} error={exc}"
                    )
                continue

            local_job = resolve_local_job(project_dir, dataset, structure_index)
            if all(local_job.identity) and local_job.identity != identity:
                stats["identity_mismatches"] += 1
                if stats["identity_mismatches"] <= PREVIEW_LIMIT:
                    print(
                        "  local job identity mismatch, ignoring direct job mapping: "
                        f"dataset={dataset} index={structure_index} "
                        f"source={build_source_label(metadata)} "
                        f"expected={identity[0][:12]} got={local_job.identity[0][:12]}"
                    )
                local_job = LocalJobRecord(
                    dataset=local_job.dataset,
                    struct_dir=local_job.struct_dir,
                    identity=local_job.identity,
                    status=local_job.status,
                    reused_from=local_job.reused_from,
                    outcar_path=None,
                    outcar_complete=False,
                )
            outcar_path = choose_completed_outcar(identity, local_job, input_context)
            if outcar_path is None:
                stats["unresolved_outcar"] += 1
                registry_path = resolve_registry_outcar(identity, input_context)
                unresolved.append(
                    UnresolvedCandidate(
                        dataset=dataset,
                        structure_index=structure_index,
                        group_key=build_group_key(atoms),
                        source_label=build_source_label(metadata),
                        elastic_mode=str(elastic_mode),
                        strain_amplitude=float(strain_amplitude),
                        struct_dir=local_job.struct_dir,
                        status=local_job.status,
                        outcar_path=local_job.outcar_path,
                        outcar_complete=local_job.outcar_complete,
                        registry_path=registry_path,
                        registry_complete=registry_path is not None,
                    )
                )
                if stats["unresolved_outcar"] <= PREVIEW_LIMIT:
                    print(
                        "  unresolved elastic candidate: "
                        f"dataset={dataset} index={structure_index} "
                        f"source={build_source_label(metadata)} "
                        f"mode={elastic_mode} amp={strain_amplitude} "
                        f"status={local_job.status} "
                        f"outcar_complete={local_job.outcar_complete} "
                        f"registry_complete={registry_path is not None}"
                    )
                continue

            stress_tensor_gpa = parse_stress_tensor_gpa(outcar_path)
            if stress_tensor_gpa is None:
                stats["unparsed_stress"] += 1
                unresolved.append(
                    UnresolvedCandidate(
                        dataset=dataset,
                        structure_index=structure_index,
                        group_key=build_group_key(atoms),
                        source_label=build_source_label(metadata),
                        elastic_mode=str(elastic_mode),
                        strain_amplitude=float(strain_amplitude),
                        struct_dir=local_job.struct_dir,
                        status=local_job.status,
                        outcar_path=outcar_path,
                        outcar_complete=False,
                        registry_path=outcar_path,
                        registry_complete=False,
                    )
                )
                if stats["unparsed_stress"] <= PREVIEW_LIMIT:
                    print(
                        "  matched OUTCAR but failed to parse stress: "
                        f"dataset={dataset} index={structure_index} "
                        f"source={build_source_label(metadata)} "
                        f"mode={elastic_mode} amp={strain_amplitude} "
                        f"outcar={outcar_path}"
                )
                continue

            try:
                reference_cell = reference_cell_from_metadata(atoms.get_cell(), strain_matrix)
                strain_voigt = cell_strain_voigt(reference_cell, atoms.get_cell())
                reference_hash = build_reference_hash(atoms, reference_cell)
            except np.linalg.LinAlgError:
                stats["reference_reconstruction_failures"] += 1
                if stats["reference_reconstruction_failures"] <= PREVIEW_LIMIT:
                    print(
                        "  failed to reconstruct reference cell, falling back to stored strain metadata: "
                        f"dataset={dataset} index={structure_index} "
                        f"source={build_source_label(metadata)}"
                    )
                reference_cell = np.array(atoms.get_cell(), dtype=float)
                strain_voigt = strain_matrix_to_voigt(strain_matrix)
                reference_hash = identity[0]

            structure_name = metadata.get("structure_name")
            rotation_rows = conventional_rotation_rows(reference_cell, structure_name)
            if rotation_rows is not None:
                strain_voigt = tensor_to_strain_voigt(
                    rotate_tensor(voigt_to_tensor(strain_voigt), rotation_rows)
                )
                stress_voigt = stress_matrix_to_voigt(
                    rotate_tensor(stress_tensor_gpa, rotation_rows)
                )
            else:
                stress_voigt = stress_matrix_to_voigt(stress_tensor_gpa)

            records.append(
                ElasticRecord(
                    group_key=build_source_label(metadata),
                    source_label=build_source_label(metadata),
                    structure_hash=identity[0],
                    reference_hash=reference_hash,
                    formula=metadata.get("formula"),
                    material_id=metadata.get("material_id"),
                    structure_name=structure_name,
                    elastic_mode=str(elastic_mode),
                    strain_amplitude=float(strain_amplitude),
                    strain_voigt=strain_voigt,
                    stress_voigt_gpa=stress_voigt,
                    outcar_path=outcar_path,
                )
            )
            stats["matched_completed_records"] += 1
            if stats["matched_completed_records"] <= MATCHED_PREVIEW_LIMIT:
                print(
                    "  matched elastic record: "
                    f"dataset={dataset} index={structure_index} "
                    f"source={build_source_label(metadata)} "
                    f"mode={elastic_mode} amp={strain_amplitude} "
                    f"outcar={outcar_path}"
                )

    return records, stats, unresolved, structure_sources


def main() -> int:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    structures_override = args.structures.resolve() if args.structures is not None else None
    if structures_override is not None and not structures_override.exists():
        print(f"Structure file not found: {structures_override}", file=sys.stderr)
        return 1

    datasets = get_dataset_names(args.dataset)
    records, stats, unresolved, structure_sources = stream_elastic_records(
        project_dir=project_dir,
        structures_override=structures_override,
        datasets=datasets,
    )

    print(
        "Scanned "
        f"{stats['frames_seen']} frames; "
        f"elastic candidates={stats['elastic_candidates']}; "
        f"single-element elastic candidates={stats['single_element_candidates']}; "
        f"matched completed records={stats['matched_completed_records']}; "
        f"missing_metadata={stats['missing_metadata']}; "
        f"identity_errors={stats['identity_errors']}; "
        f"unresolved_outcar={stats['unresolved_outcar']}; "
        f"stress_parse_failures={stats['unparsed_stress']}; "
        f"identity_mismatches={stats['identity_mismatches']}; "
        f"reference_reconstruction_failures={stats['reference_reconstruction_failures']}"
    )

    if not records:
        for line in debug_status_lines(structure_sources, unresolved):
            print(line)
        print("No single-element elastic-stress records with completed OUTCAR files were found.")
        return 1

    grouped: dict[str, list[ElasticRecord]] = {}
    for record in records:
        grouped.setdefault(record.group_key, []).append(record)

    results: list[dict] = []
    for group_key in sorted(grouped):
        group_records = grouped[group_key]
        print(
            f"Fitting group {group_key}: records={len(group_records)} "
            f"unique_modes={len({record.elastic_mode for record in group_records})}"
        )
        if len(group_records) < args.min_records:
            print(
                f"  skipping group {group_key}: "
                f"records={len(group_records)} is below min_records={args.min_records}"
            )
            continue
        try:
            tensor_gpa, raw_tensor_gpa, fit_model, stress_offset_gpa = fit_elastic_tensor(group_records)
        except ValueError as exc:
            print(f"  fit failed for group {group_key}: {exc}")
            continue
        results.append(
            summarise_group(
                group_records,
                tensor_gpa,
                raw_tensor_gpa,
                fit_model,
                stress_offset_gpa,
            )
        )

    if not results:
        print(
            "Elastic records were found, but no group had enough independent "
            "completed data to fit a full tensor."
        )
        return 1

    for result in results:
        print(f"[{result['group_key']}]")
        print(
            f"  source={result['source_label']} "
            f"formula={result['formula']} "
            f"structure_name={result['structure_name']} "
            f"material_id={result['material_id']} "
            f"fit_model={result['fit_model']} "
            f"records={result['num_records']} "
            f"fit_rmse_gpa={result['fit_rmse_gpa']:.6f} "
            f"min_eigenvalue_gpa={result['min_eigenvalue_gpa']:.6f} "
            f"max_asymmetry_gpa={result['max_asymmetry_gpa']:.6f}"
        )
        named_line = format_named_constants(result)
        if named_line is not None:
            print(named_line)
        print(format_stress_offset(result))
        if result["warnings"]:
            print(f"  warnings={','.join(result['warnings'])}")
        tensor = np.array(result["tensor_gpa"], dtype=float)
        print(format_tensor(tensor))
        print()

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
