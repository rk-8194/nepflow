#!/usr/bin/env python3
"""
Backfill nepflow's completed VASP job registry from existing project jobs.

Scans <nepflow_root>/projects for completed VASP job folders that have not yet
been stored in .vasp_completed_jobs.json. For each completed job, it hashes:
  - the canonical POSCAR structure
  - the job POTCAR bytes
  - the job INCAR with launcher resource params (NCORE/KPAR) removed

It also writes .vasp_identity into job folders that do not already have it,
unless --dry-run is used.

Usage:
    python utilities/populate_vasp_completed_registry.py
    python utilities/populate_vasp_completed_registry.py --dry-run
    python utilities/populate_vasp_completed_registry.py --projects-dir /path/to/projects
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

VASP_COMPLETION_MARKERS = ["General timing", "Voluntary context switches"]
VASP_REGISTRY_VERSION = 1
STRUCTURE_HASH_VERSION = "structure-v1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str | None:
    try:
        return sha256_bytes(path.read_bytes())
    except OSError:
        return None


def outcar_is_complete(outcar_path: Path) -> bool:
    if not outcar_path.exists():
        return False
    try:
        with open(outcar_path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 50_000))
            tail = f.read()
        return any(marker in tail for marker in VASP_COMPLETION_MARKERS)
    except OSError:
        return False


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


def hash_incar_file(incar_path: Path) -> str:
    return sha256_bytes(
        strip_resource_incar_params(
            incar_path.read_text(encoding="utf-8", errors="replace")
        ).encode("utf-8")
    )


def canonical_poscar_text(poscar: dict[str, Any]) -> str:
    symbols = poscar["symbols"]
    unique_elements = sorted(set(symbols))

    sorted_indices = []
    counts = []
    for elem in unique_elements:
        indices = [i for i, symbol in enumerate(symbols) if symbol == elem]
        sorted_indices.extend(indices)
        counts.append(len(indices))

    cell = poscar["cell"]
    frac_positions = poscar["scaled_positions"]
    lines = [STRUCTURE_HASH_VERSION]
    lines.append("pbc 1 1 1")
    lines.append("cell")
    for row in cell:
        lines.append(_format_vector(row))
    lines.append("atoms")
    for idx in sorted_indices:
        p = frac_positions[idx]
        lines.append(f"{symbols[idx]} {_format_vector(p)}")
    return "\n".join(lines) + "\n"


def hash_poscar_structure(poscar_path: Path) -> str:
    poscar = parse_poscar(poscar_path)
    return sha256_bytes(canonical_poscar_text(poscar).encode("utf-8"))


def parse_poscar(poscar_path: Path) -> dict[str, Any]:
    """Parse the VASP5 POSCAR subset written by nepflow."""
    lines = [
        line.strip()
        for line in poscar_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    if len(lines) < 8:
        raise ValueError("POSCAR has too few lines")

    comment = lines[0]
    scale = float(lines[1].split()[0])
    if scale <= 0:
        raise ValueError("negative POSCAR scale factors are not supported")

    cell = []
    for line in lines[2:5]:
        values = [float(value) * scale for value in line.split()[:3]]
        if len(values) != 3:
            raise ValueError("invalid lattice row")
        cell.append(values)

    elements = lines[5].split()
    if not elements or all(_is_number(token) for token in elements):
        raise ValueError("VASP4 POSCAR without element symbols is not supported")
    counts = [int(value) for value in lines[6].split()]
    if len(elements) != len(counts):
        raise ValueError("element/count length mismatch")

    coord_line_idx = 7
    if lines[coord_line_idx].lower().startswith("s"):
        coord_line_idx += 1
    coord_mode = lines[coord_line_idx].lower()
    coord_start = coord_line_idx + 1
    n_atoms = sum(counts)
    coord_lines = lines[coord_start:coord_start + n_atoms]
    if len(coord_lines) != n_atoms:
        raise ValueError("atom coordinate count mismatch")

    symbols = []
    for elem, count in zip(elements, counts):
        symbols.extend([elem] * count)

    scaled_positions = []
    inv_cell = _invert_3x3(cell)
    for line in coord_lines:
        values = [float(value) for value in line.split()[:3]]
        if coord_mode.startswith(("d", "f")):
            scaled_positions.append(values)
        elif coord_mode.startswith(("c", "k")):
            cart = [value * scale for value in values]
            scaled_positions.append(_row_matmul(cart, inv_cell))
        else:
            raise ValueError(f"unknown coordinate mode: {lines[coord_line_idx]}")

    return {
        "comment": comment,
        "symbols": symbols,
        "cell": cell,
        "scaled_positions": scaled_positions,
    }


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _format_vector(values: list[float]) -> str:
    return " ".join(f"{float(values[i]):.14f}" for i in range(3))


def _row_matmul(row: list[float], matrix: list[list[float]]) -> list[float]:
    return [
        row[0] * matrix[0][col] + row[1] * matrix[1][col] + row[2] * matrix[2][col]
        for col in range(3)
    ]


def _invert_3x3(matrix: list[list[float]]) -> list[list[float]]:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    det = (
        a * (e * i - f * h)
        - b * (d * i - f * g)
        + c * (d * h - e * g)
    )
    if abs(det) < 1e-15:
        raise ValueError("singular lattice")
    return [
        [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
        [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
        [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
    ]


def read_registry(registry_path: Path) -> dict:
    if not registry_path.exists():
        return {"version": VASP_REGISTRY_VERSION, "jobs": {}}
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        sys.exit(f"Error: could not read {registry_path}: {e}")
    if not isinstance(data, dict):
        sys.exit(f"Error: registry is not a JSON object: {registry_path}")
    data.setdefault("version", VASP_REGISTRY_VERSION)
    data.setdefault("jobs", {})
    return data


def write_registry(registry_path: Path, registry: dict) -> None:
    tmp_path = registry_path.with_suffix(registry_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(registry_path)


def registry_contains(registry: dict, incar_hash: str, potcar_hash: str, structure_hash: str) -> bool:
    return (
        registry.get("jobs", {})
        .get(incar_hash, {})
        .get(potcar_hash, {})
        .get(structure_hash)
        is not None
    )


def upsert_registry(registry: dict, incar_hash: str, potcar_hash: str, structure_hash: str, entry: dict) -> None:
    jobs = registry.setdefault("jobs", {})
    jobs.setdefault(incar_hash, {}).setdefault(potcar_hash, {})[structure_hash] = entry


def parse_job_context(struct_dir: Path, projects_dir: Path) -> dict:
    project_dir = next((parent for parent in struct_dir.parents if parent.parent == projects_dir), None)
    project_name = project_dir.name.removeprefix("project_") if project_dir else ""
    dataset = struct_dir.parent.name if struct_dir.parent.name != "jobs" else ""
    selected_index = None
    match = re.match(r"struct_(\d+)$", struct_dir.name)
    if match:
        selected_index = int(match.group(1))
    return {
        "project_name": project_name,
        "dataset": dataset,
        "selected_index": selected_index,
    }


def build_entry(
    struct_dir: Path,
    projects_dir: Path,
    verbose: bool,
) -> tuple[str, str, str, dict] | None:
    poscar = struct_dir / "POSCAR"
    incar = struct_dir / "INCAR"
    potcar = struct_dir / "POTCAR"
    outcar = struct_dir / "OUTCAR"

    missing = [name for name, path in (("POSCAR", poscar), ("INCAR", incar), ("POTCAR", potcar)) if not path.exists()]
    if missing:
        if verbose:
            print(f"SKIP {struct_dir}: missing {', '.join(missing)}")
        return None
    if not outcar_is_complete(outcar):
        if verbose:
            print(f"SKIP {struct_dir}: OUTCAR is absent or incomplete")
        return None

    try:
        structure_hash = hash_poscar_structure(poscar)
        incar_hash = hash_incar_file(incar)
        potcar_hash = sha256_bytes(potcar.read_bytes())
    except Exception as e:
        if verbose:
            print(f"SKIP {struct_dir}: could not hash inputs: {e}")
        return None

    context = parse_job_context(struct_dir, projects_dir)
    entry = {
        "job_path": str(struct_dir.resolve()),
        "project_name": context["project_name"],
        "dataset": context["dataset"],
        "selected_index": context["selected_index"],
        "completed_at": datetime.now().isoformat(),
        "backfilled_at": datetime.now().isoformat(),
        "structure_hash": structure_hash,
        "incar_hash": incar_hash,
        "potcar_hash": potcar_hash,
        "outcar_hash": file_sha256(outcar),
        "vasprun_hash": file_sha256(struct_dir / "vasprun.xml"),
    }
    return incar_hash, potcar_hash, structure_hash, entry


def write_identity(struct_dir: Path, entry: dict, dry_run: bool) -> bool:
    identity_path = struct_dir / ".vasp_identity"
    if identity_path.exists():
        return False
    identity = {
        "project_name": entry.get("project_name", ""),
        "dataset": entry.get("dataset", ""),
        "selected_index": entry.get("selected_index"),
        "source_xyz": "",
        "structure_hash": entry["structure_hash"],
        "incar_hash": entry["incar_hash"],
        "potcar_hash": entry["potcar_hash"],
        "backfilled": True,
    }
    if not dry_run:
        identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True), encoding="utf-8")
    return True


def main() -> None:
    nepflow_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Backfill .vasp_completed_jobs.json from completed VASP jobs under projects/."
    )
    parser.add_argument(
        "--projects-dir",
        type=Path,
        default=nepflow_root / "projects",
        help="Projects directory to scan (default: <nepflow_root>/projects).",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=nepflow_root / ".vasp_completed_jobs.json",
        help="Completed-job registry path (default: <nepflow_root>/.vasp_completed_jobs.json).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be added without writing registry or identity files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print skipped incomplete/unhashable job folders as well as additions.",
    )
    args = parser.parse_args()

    projects_dir = args.projects_dir.resolve()
    registry_path = args.registry.resolve()
    if not projects_dir.is_dir():
        sys.exit(f"Error: projects directory not found: {projects_dir}")

    registry = read_registry(registry_path)
    struct_dirs = sorted(
        path for path in projects_dir.glob("project_*/vasp/jobs/*/struct_*")
        if path.is_dir()
    )

    scanned = completed = added = already = identities = 0
    print(f"Scanning {len(struct_dirs)} VASP job folder(s) under {projects_dir}")

    for struct_dir in struct_dirs:
        scanned += 1
        built = build_entry(struct_dir, projects_dir, args.verbose)
        if built is None:
            continue

        completed += 1
        incar_hash, potcar_hash, structure_hash, entry = built
        identity_written = write_identity(struct_dir, entry, args.dry_run)
        if identity_written:
            identities += 1

        if registry_contains(registry, incar_hash, potcar_hash, structure_hash):
            already += 1
            print(f"EXISTS {struct_dir}")
            continue

        added += 1
        print(f"ADD    {struct_dir}")
        if not args.dry_run:
            upsert_registry(registry, incar_hash, potcar_hash, structure_hash, entry)

    if not args.dry_run:
        write_registry(registry_path, registry)

    mode = "would write" if args.dry_run else "wrote"
    print("")
    print(f"Scanned        : {scanned}")
    print(f"Completed      : {completed}")
    print(f"Already stored : {already}")
    print(f"New entries    : {added}")
    print(f"Identities {mode}: {identities}")
    print(f"Registry       : {registry_path}")


if __name__ == "__main__":
    main()
