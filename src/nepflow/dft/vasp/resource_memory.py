"""Legacy ``.vasp_memory`` resource learning for VASP executions.

The file format is intentionally kept separate from StateStore performance
events.  StateStore contains provenance; this module owns the shared legacy
resource predictor and its append-only CSV sink.
"""

from __future__ import annotations

import csv
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read as ase_read

from nepflow.config.models import NepflowConfig
from nepflow.errors import StateError, VaspError

CSV_COLUMNS = (
    "n_atoms",
    "n_kpoints_irr",
    "n_electrons",
    "nodes",
    "gpus",
    "ncore",
    "kpar",
    "avg_loop_time",
    "oom",
)
VASP_MEMORY_COLUMNS = CSV_COLUMNS


@dataclass(frozen=True, slots=True)
class VaspMemoryObservation:
    """One legacy memory observation with explicit internal GPU semantics."""

    n_atoms: int
    n_kpoints_irr: int
    n_electrons: int
    nodes: int
    total_gpus: int
    ncore: int
    kpar: int
    avg_loop_time: float
    oom: int = 0

    @property
    def gpus(self) -> int:
        """Return the legacy on-disk GPU count (total GPUs/ranks)."""

        return self.total_gpus

    @property
    def gpus_per_node(self) -> int:
        """Return a conservative per-node interpretation of legacy ``gpus``."""

        return max(1, math.ceil(self.total_gpus / max(1, self.nodes)))


@dataclass(frozen=True, slots=True)
class VaspResourcePrediction:
    """Typed initial execution parameters selected from memory observations."""

    ncore: int
    kpar: int
    nodes: int
    gpus_per_node: int
    mpi_ranks: int
    source: str


def resolve_memory_path(project_root: Path) -> Path:
    """Resolve the shared legacy memory file without searching arbitrary parents."""

    root = Path(project_root).resolve()
    if root.parent.name == "projects" and root.name.startswith("project_"):
        return root.parent.parent / ".vasp_memory"
    return root / ".vasp_memory"


def _parse_int(value: Any, *, path: Path, row: int, field: str) -> int:
    try:
        # Reject values such as ``1.0``: the legacy schema is integer-valued.
        if isinstance(value, str) and not re.fullmatch(r"[+-]?\d+", value.strip()):
            raise ValueError
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise StateError(f"Malformed .vasp_memory {path} row {row}: {field} is not an integer") from exc
    return result


def _parse_float(value: Any, *, path: Path, row: int, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise StateError(f"Malformed .vasp_memory {path} row {row}: {field} is not numeric") from exc
    if not math.isfinite(result):
        raise StateError(f"Malformed .vasp_memory {path} row {row}: {field} is not finite")
    return result


def load_vasp_memory(path: Path) -> tuple[VaspMemoryObservation, ...]:
    """Read legacy memory observations, preserving OOM rows.

    Missing and genuinely empty files are valid no-memory states.  Once a
    header or row exists, malformed required data is an explicit state error.
    """

    path = Path(path)
    if path.is_dir():
        path = resolve_memory_path(path)
    if not path.exists():
        return ()
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            if not handle.read(1):
                return ()
            handle.seek(0)
            reader = csv.DictReader(handle)
            header = tuple(reader.fieldnames or ())
            supported = set(CSV_COLUMNS)
            missing = {
                "n_atoms",
                "n_electrons",
                "nodes",
                "gpus",
                "ncore",
                "kpar",
                "avg_loop_time",
            } - set(header)
            if missing or not set(header).issubset(supported) or len(set(header)) != len(header):
                raise StateError(
                    f"Malformed .vasp_memory {path} header: expected legacy columns; "
                    f"missing={sorted(missing)}"
                )
            observations: list[VaspMemoryObservation] = []
            for row_number, row in enumerate(reader, start=2):
                if row is None or not any(str(value or "").strip() for value in row.values()):
                    continue
                if None in row:
                    raise StateError(f"Malformed .vasp_memory {path} row {row_number}: too many columns")
                required = ("n_atoms", "n_electrons", "nodes", "gpus", "ncore", "kpar")
                values = {
                    field: _parse_int(row.get(field), path=path, row=row_number, field=field)
                    for field in required
                }
                n_kpoints = (
                    _parse_int(row.get("n_kpoints_irr"), path=path, row=row_number, field="n_kpoints_irr")
                    if "n_kpoints_irr" in header
                    else 0
                )
                oom = (
                    _parse_int(row.get("oom"), path=path, row=row_number, field="oom")
                    if "oom" in header
                    else 0
                )
                if oom not in (0, 1):
                    raise StateError(f"Malformed .vasp_memory {path} row {row_number}: oom must be 0 or 1")
                for field in ("n_atoms", "n_electrons", "nodes", "gpus", "ncore", "kpar"):
                    if values[field] < 1:
                        raise StateError(
                            f"Malformed .vasp_memory {path} row {row_number}: {field} must be positive"
                        )
                if n_kpoints < 0:
                    raise StateError(
                        f"Malformed .vasp_memory {path} row {row_number}: n_kpoints_irr must not be negative"
                    )
                avg = _parse_float(
                    row.get("avg_loop_time"), path=path, row=row_number, field="avg_loop_time"
                )
                observations.append(
                    VaspMemoryObservation(
                        n_atoms=values["n_atoms"],
                        n_kpoints_irr=n_kpoints,
                        n_electrons=values["n_electrons"],
                        nodes=values["nodes"],
                        total_gpus=values["gpus"],
                        ncore=values["ncore"],
                        kpar=values["kpar"],
                        avg_loop_time=avg,
                        oom=oom,
                    )
                )
            return tuple(observations)
    except UnicodeError as exc:
        raise StateError(f"Could not read .vasp_memory {path}: {exc}") from exc
    except OSError as exc:
        raise StateError(f"Could not read .vasp_memory {path}: {exc}") from exc


@contextmanager
def _append_lock(path: Path):
    """Lock a small sidecar while appending a single CSV row."""

    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def append_vasp_memory(path: Path, observation: VaspMemoryObservation) -> None:
    """Append one observation using the exact legacy header and serialized rows."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _append_lock(path):
        needs_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            if needs_header:
                writer.writerow(CSV_COLUMNS)
            writer.writerow(
                (
                    observation.n_atoms,
                    observation.n_kpoints_irr,
                    observation.n_electrons,
                    observation.nodes,
                    observation.total_gpus,
                    observation.ncore,
                    observation.kpar,
                    f"{observation.avg_loop_time:.4f}",
                    observation.oom,
                )
            handle.flush()
            os.fsync(handle.fileno())


def _powers_of_two(maximum: int, *, minimum: int = 1) -> tuple[int, ...]:
    return tuple(value for value in (2**index for index in range(0, 32)) if minimum <= value <= maximum)


def _snap_nearest(value: int, candidates: Iterable[int]) -> int:
    values = tuple(sorted(set(candidates)))
    if not values:
        return max(1, value)
    return min(values, key=lambda candidate: (abs(candidate - value), candidate))


def _valid_ncores(config: NepflowConfig) -> tuple[int, ...]:
    values = tuple(
        value
        for value in _powers_of_two(config.hpc.cores_per_node, minimum=2)
        if config.hpc.cores_per_node % value == 0
    )
    return values or (config.hpc.cores_per_node,)


def _valid_gpu_counts(config: NepflowConfig) -> tuple[int, ...]:
    return tuple(value for value in (1, 2, 4, 8) if value <= config.hpc.gpus_per_node) or (1,)


def default_vasp_resources(
    config: NepflowConfig,
    *,
    source: str = "default",
) -> VaspResourcePrediction:
    """Return the explicitly permitted conservative no-confidence default."""

    gpus = 1
    return VaspResourcePrediction(
        ncore=max(2, config.hpc.cores_per_node // max(1, config.hpc.gpus_per_node)),
        kpar=1,
        nodes=1,
        gpus_per_node=gpus,
        mpi_ranks=1,
        source=source,
    )


def predict_vasp_params(
    observations: Sequence[VaspMemoryObservation] | int,
    n_electrons: int | None = None,
    memory_root: Path | None = None,
    config: NepflowConfig | None = None,
    n_kpoints_irr: int | None = None,
    *,
    n_atoms: int | None = None,
) -> VaspResourcePrediction | None:
    """Predict execution resources using the legacy nearest-neighbour policy.

    The typed form accepts observations as the first argument.  For callers
    migrating the old ``main`` helper, the compatible form is
    ``(n_atoms, n_electrons, repository_root, config, n_kpoints_irr=None)``.
    """

    if isinstance(observations, int):
        if n_atoms is not None or n_electrons is None or memory_root is None or config is None:
            raise TypeError("legacy predict_vasp_params requires atoms, electrons, root, and config")
        n_atoms_value = observations
        observations = load_vasp_memory(resolve_memory_path(memory_root))
    else:
        if n_atoms is None or n_electrons is None or config is None:
            raise TypeError("typed predict_vasp_params requires observations, features, and config")
        n_atoms_value = n_atoms
    n_kpoints_irr = 0 if n_kpoints_irr is None else n_kpoints_irr

    successful = tuple(item for item in observations if item.oom != 1)
    if len(successful) < 3:
        return None
    use_kpoints = n_kpoints_irr > 0 and any(item.n_kpoints_irr > 0 for item in successful)
    query = np.asarray(
        [n_atoms_value, n_electrons, n_kpoints_irr]
        if use_kpoints
        else [n_atoms_value, n_electrons],
        dtype=float,
    )
    matrix = np.asarray(
        [
            [item.n_atoms, item.n_electrons, item.n_kpoints_irr]
            if use_kpoints
            else [item.n_atoms, item.n_electrons]
            for item in successful
        ],
        dtype=float,
    )
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std == 0] = 1.0
    distances = np.linalg.norm((matrix - mean) / std - (query - mean) / std, axis=1)
    minimum = float(distances.min())
    nearest = [item for item, distance in zip(successful, distances) if distance <= minimum + 1e-6]
    # Legacy ``gpus`` is total GPUs.  Rank rows by the smallest represented
    # total GPU requirement, then by their measured loop time.
    gpu_tier = min(max(1, item.total_gpus) for item in nearest)
    chosen = min((item for item in nearest if max(1, item.total_gpus) == gpu_tier), key=lambda item: item.avg_loop_time)
    nodes = _snap_nearest(chosen.nodes, _powers_of_two(max(1, config.hpc.max_nodes)))
    per_node = _snap_nearest(
        max(1, math.ceil(chosen.total_gpus / max(1, chosen.nodes))),
        _valid_gpu_counts(config),
    )
    ncore = _snap_nearest(chosen.ncore, _valid_ncores(config))
    return VaspResourcePrediction(
        ncore=ncore,
        kpar=max(1, int(chosen.kpar)),
        nodes=nodes,
        gpus_per_node=per_node,
        mpi_ranks=nodes * per_node,
        source="vasp_memory",
    )


def parse_zval(source: str | Path) -> float:
    """Parse a VASP ``ZVAL`` value from POTCAR text or a POTCAR path."""

    source_path = Path(source) if isinstance(source, Path) else None
    if source_path is None and isinstance(source, str) and "\n" not in source:
        try:
            candidate = Path(source)
            source_path = candidate if candidate.is_file() else None
        except OSError:
            source_path = None
    text = (
        source_path.read_text(encoding="utf-8", errors="replace")
        if source_path is not None
        else str(source)
    )
    match = re.search(r"\bZVAL\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))", text, re.IGNORECASE)
    if match is None:
        raise VaspError("POTCAR does not contain a numeric ZVAL")
    value = float(match.group(1))
    if not math.isfinite(value):
        raise VaspError("POTCAR contains a non-finite ZVAL")
    return value


def estimate_n_electrons(poscar_path: Path, project_root: Path | Mapping[str, float]) -> int:
    """Estimate valence electrons from prepared POSCAR composition and POTCAR_* files."""

    try:
        atoms = ase_read(str(poscar_path))
    except Exception as exc:
        raise VaspError(f"Could not read prepared POSCAR {poscar_path}") from exc
    if isinstance(project_root, Mapping):
        counts = {
            element: atoms.get_chemical_symbols().count(element)
            for element in set(atoms.get_chemical_symbols())
        }
        missing = sorted(element for element in counts if element not in project_root)
        if missing:
            raise VaspError(f"No POTCAR ZVAL available for {', '.join(missing)}")
        return int(round(sum(count * float(project_root[element]) for element, count in counts.items())))
    potcar_dir = Path(project_root) / "config" / "vasp"
    values: dict[str, float] = {}
    for path in potcar_dir.glob("POTCAR_*"):
        element = path.name.split("_", 1)[1]
        try:
            values[element] = parse_zval(path)
        except (OSError, UnicodeError, VaspError):
            continue
    counts = {element: atoms.get_chemical_symbols().count(element) for element in set(atoms.get_chemical_symbols())}
    missing = sorted(element for element in counts if element not in values)
    if missing:
        raise VaspError(f"No POTCAR ZVAL available for {', '.join(missing)} in {potcar_dir}")
    return int(round(sum(count * values[element] for element, count in counts.items())))


def _incar_value(text: str, key: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(key)}\s*=\s*([^!#\n]+)", text, re.IGNORECASE | re.MULTILINE)
    return None if match is None else match.group(1).strip()


def estimate_kpoints_irr(
    poscar_path: Path,
    project_root: Path | None = None,
    *,
    config: NepflowConfig | None = None,
    incar_path: Path | None = None,
) -> int:
    """Estimate irreducible k-points with the prepared VASP mesh semantics."""

    try:
        atoms = ase_read(str(poscar_path))
    except Exception as exc:
        raise VaspError(f"Could not read prepared POSCAR {poscar_path}") from exc
    if incar_path is None:
        incar_path = (Path(project_root) / "config" / "vasp" / "INCAR") if project_root else None
    text = ""
    if incar_path is not None and Path(incar_path).is_file():
        text = Path(incar_path).read_text(encoding="utf-8", errors="replace")
    kspacing = float(_incar_value(text, "KSPACING") or (config.vasp.kspacing if config else 0.30))
    if kspacing <= 0:
        raise VaspError("KSPACING must be positive")
    kgamma_value = (_incar_value(text, "KGAMMA") or (".TRUE." if config is None or config.vasp.kgamma else ".FALSE.")).upper()
    kgamma = kgamma_value in {".TRUE.", "TRUE", "T", "1", "YES"}
    cell = np.asarray(atoms.cell.array, dtype=float)
    if abs(float(np.linalg.det(cell))) <= 1e-12:
        raise VaspError(f"Prepared POSCAR has no valid cell: {poscar_path}")
    reciprocal = np.linalg.inv(cell).T
    mesh = tuple(max(1, int(math.ceil(2 * math.pi * np.linalg.norm(vector) / kspacing))) for vector in reciprocal)
    try:
        import spglib

        mapping, _grid = spglib.get_ir_reciprocal_mesh(
            mesh,
            (cell, atoms.get_scaled_positions(), atoms.get_atomic_numbers()),
            is_shift=(0, 0, 0) if kgamma else (1, 1, 1),
        )
    except Exception as exc:
        raise VaspError(f"Could not derive irreducible k-points for {poscar_path}") from exc
    return len(set(int(value) for value in mapping))


def extract_resource_features(
    working_directory: Path,
    project_root: Path,
    *,
    config: NepflowConfig,
) -> tuple[int, int, int]:
    """Return ``(atoms, valence electrons, irreducible k-points)`` for a prepared run."""

    poscar = Path(working_directory) / "POSCAR"
    atoms = len(ase_read(str(poscar)))
    electrons = estimate_n_electrons(poscar, project_root)
    kpoints = estimate_kpoints_irr(
        poscar,
        project_root,
        config=config,
        incar_path=Path(working_directory) / "INCAR",
    )
    return atoms, electrons, kpoints


def read_incar_resource_parameters(incar_path: Path) -> tuple[int, int]:
    """Read the actual launcher-controlled NCORE/KPAR values from an INCAR."""

    try:
        text = Path(incar_path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise VaspError(f"Could not read INCAR {incar_path}") from exc
    values: dict[str, int] = {}
    for key in ("NCORE", "KPAR"):
        raw = _incar_value(text, key)
        if raw is not None:
            try:
                values[key] = int(raw.split()[0])
            except (TypeError, ValueError) as exc:
                raise VaspError(f"Malformed {key} in INCAR {incar_path}") from exc
    return max(1, values.get("NCORE", 1)), max(1, values.get("KPAR", 1))


__all__ = [
    "CSV_COLUMNS",
    "VASP_MEMORY_COLUMNS",
    "VaspMemoryObservation",
    "VaspResourcePrediction",
    "append_vasp_memory",
    "default_vasp_resources",
    "estimate_kpoints_irr",
    "estimate_n_electrons",
    "extract_resource_features",
    "load_vasp_memory",
    "parse_zval",
    "predict_vasp_params",
    "read_incar_resource_parameters",
    "resolve_memory_path",
]
