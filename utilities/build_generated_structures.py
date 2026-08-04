#!/usr/bin/env python3
"""
Build or extend a nepflow ``generated_structures.xyz`` file from manual inputs.

This utility is designed for repeated use across many folders. It can:
  - scan one or more files / directories for structures
  - append them into a single extxyz output file
  - add nepflow-friendly metadata such as composition, seed_id, and source
  - skip duplicate physical structures by default
  - continue seed numbering across multiple runs

Typical usage:
    python utilities/build_generated_structures.py ^
        --input folder_a folder_b folder_c ^
        --output projects/project_demo/structures/generated/generated_structures.xyz

    python utilities/build_generated_structures.py ^
        --input seed_01 seed_02 other_batch ^
        --output projects/project_demo/structures/generated/generated_structures.xyz ^
        --seed-output projects/project_demo/structures/seeds/base_structures.xyz ^
        --seed-list seed_files.txt ^
        --recursive ^
        --configurational-type manual_seed ^
        --perturbation-type manual_import

Notes:
  - The output format is extxyz, suitable for nepflow's select stage.
  - ``composition`` is stored as a dict. If not provided explicitly, it is
    inferred from the chemical symbols in each structure.
  - ``actual_composition`` is always inferred from the actual atom counts.
  - ``seed-list`` accepts one structure file path per line.
  - ``elastic-strain-list`` accepts one folder path per line.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

from ase import Atoms
from ase.io import read as ase_read, write as ase_write

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.structure_identity import annotate_structure_hash, hash_structure


DEFAULT_PATTERNS = [
    "POSCAR",
    "CONTCAR",
    "*.vasp",
    "*.cif",
    "*.xyz",
    "*.extxyz",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create or extend a nepflow generated_structures.xyz file from "
            "manually supplied structure files."
        )
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input files or directories to scan.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to generated_structures.xyz output file.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=None,
        help=(
            "Glob pattern for files inside input directories. "
            f"Defaults: {', '.join(DEFAULT_PATTERNS)}"
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan input directories.",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Allow duplicate physical structures to be appended.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file instead of appending to it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be imported without writing output.",
    )
    parser.add_argument(
        "--seed-output",
        type=Path,
        default=None,
        help=(
            "Optional path to structures/seeds/base_structures.xyz. Seed structures "
            "selected by --seed-list are also written there."
        ),
    )
    parser.add_argument(
        "--seed-list",
        type=Path,
        default=None,
        help=(
            "Text file listing structure file paths that should be treated as seeds, "
            "one file path per line."
        ),
    )
    parser.add_argument(
        "--elastic-strain-list",
        type=Path,
        default=None,
        help=(
            "Text file listing folder paths whose imported structures should be "
            "tagged as elastic strain / elastic_stress, one folder path per line."
        ),
    )
    parser.add_argument(
        "--composition-json",
        type=str,
        default=None,
        help=(
            "Optional target composition dict to apply to every imported structure, "
            'for example: \'{"W": 0.5, "Cr": 0.5}\''
        ),
    )
    parser.add_argument(
        "--configurational-type",
        type=str,
        default="manual_import",
        help="configurational_type metadata value. Default: manual_import",
    )
    parser.add_argument(
        "--perturbation-type",
        type=str,
        default="manual_import",
        help="perturbation_type metadata value. Default: manual_import",
    )
    parser.add_argument(
        "--source-prefix",
        type=str,
        default="manual",
        help="Prefix used when generating source metadata. Default: manual",
    )
    parser.add_argument(
        "--seed-prefix",
        type=str,
        default="seed_",
        help="Prefix for generated seed_id values. Default: seed_",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=None,
        help=(
            "Starting integer for seed_id numbering. If omitted, numbering continues "
            "from the existing output file when possible."
        ),
    )
    parser.add_argument(
        "--preserve-existing-seed-ids",
        action="store_true",
        help="Keep seed_id values already present in input extxyz structures.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=None,
        help=(
            "Extra metadata key=value to apply to every structure. "
            "Repeat as needed."
        ),
    )
    return parser.parse_args()


def detect_format(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name in {"poscar", "contcar"} or suffix == ".vasp":
        return "vasp"
    if suffix == ".cif":
        return "cif"
    if suffix in {".xyz", ".extxyz"}:
        return "extxyz"
    raise ValueError(f"Unsupported structure file type: {path}")


def read_structures(path: Path) -> list[Atoms]:
    fmt = detect_format(path)
    structures = ase_read(str(path), index=":", format=fmt)
    if isinstance(structures, list):
        return structures
    return [structures]


def parse_tags(raw_tags: list[str] | None) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for item in raw_tags or []:
        if "=" not in item:
            raise ValueError(f"Invalid --tag '{item}'. Expected key=value.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --tag '{item}'. Empty key.")
        parsed[key] = coerce_scalar(value.strip())
    return parsed


def coerce_scalar(value: str) -> object:
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def normalized_composition_from_atoms(atoms: Atoms) -> dict[str, float]:
    counts: dict[str, int] = {}
    for symbol in atoms.get_chemical_symbols():
        counts[symbol] = counts.get(symbol, 0) + 1
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {
        element: count / total
        for element, count in sorted(counts.items())
    }


def normalize_composition_map(raw: dict[str, object]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for element, value in raw.items():
        fraction = float(value)
        if fraction > 0.0:
            normalized[str(element)] = fraction
    total = sum(normalized.values())
    if total <= 0.0:
        raise ValueError("composition-json must contain a positive total fraction")
    return {
        element: fraction / total
        for element, fraction in sorted(normalized.items())
    }


def seed_number(seed_id: object, prefix: str) -> int | None:
    if not isinstance(seed_id, str):
        return None
    if not seed_id.startswith(prefix):
        return None
    suffix = seed_id[len(prefix):]
    if not suffix.isdigit():
        return None
    return int(suffix)


def next_seed_index(existing: Iterable[Atoms], seed_prefix: str, fallback: int) -> int:
    max_seen = fallback - 1
    for atoms in existing:
        number = seed_number(atoms.info.get("seed_id"), seed_prefix)
        if number is not None:
            max_seen = max(max_seen, number)
    return max_seen + 1


def load_existing_output(output_path: Path) -> list[Atoms]:
    if not output_path.exists():
        return []
    structures = ase_read(str(output_path), index=":", format="extxyz")
    if isinstance(structures, list):
        return structures
    return [structures]


def load_seed_file_list(seed_list_path: Path) -> set[Path]:
    return load_path_list(seed_list_path)


def load_path_list(list_path: Path) -> set[Path]:
    base_dir = list_path.resolve().parent
    selected: set[Path] = set()
    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        candidate = Path(line)
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        else:
            candidate = candidate.resolve()
        selected.add(candidate)
    return selected


def load_folder_list(list_path: Path) -> set[Path]:
    base_dir = list_path.resolve().parent
    selected: set[Path] = set()
    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        candidate = Path(line)
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        else:
            candidate = candidate.resolve()
        selected.add(candidate)
    return selected


def merge_unique_by_structure_hash(
    existing: list[Atoms],
    incoming: list[Atoms],
) -> tuple[list[Atoms], int]:
    merged = list(existing)
    seen = {hash_structure(atoms) for atoms in existing}
    skipped = 0
    for atoms in incoming:
        structure_hash = hash_structure(atoms)
        if structure_hash in seen:
            skipped += 1
            continue
        merged.append(atoms)
        seen.add(structure_hash)
    return merged, skipped


def collect_input_files(
    inputs: list[str],
    patterns: list[str],
    recursive: bool,
    output_path: Path,
) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()

    for raw in inputs:
        path = Path(raw).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input path not found: {path}")

        if path.is_file():
            if path != output_path.resolve() and path not in seen:
                files.append(path)
                seen.add(path)
            continue

        matches: list[Path] = []
        for pattern in patterns:
            iterator = path.rglob(pattern) if recursive else path.glob(pattern)
            matches.extend(p for p in iterator if p.is_file())

        matches = sorted({match.resolve() for match in matches if match.resolve() != output_path.resolve()})
        for match in matches:
            if match not in seen:
                files.append(match)
                seen.add(match)

    return files


def main() -> int:
    args = parse_args()

    patterns = args.pattern or DEFAULT_PATTERNS
    output_path = args.output.resolve()
    seed_output_path = args.seed_output.resolve() if args.seed_output is not None else None
    user_tags = parse_tags(args.tag)

    seed_list_paths: set[Path] = set()
    if args.seed_list is not None:
        seed_list_path = args.seed_list.resolve()
        seed_list_paths = load_seed_file_list(seed_list_path)
        if seed_output_path is None:
            raise ValueError("--seed-list requires --seed-output")
    elastic_strain_dirs: set[Path] = set()
    if args.elastic_strain_list is not None:
        elastic_strain_dirs = load_folder_list(args.elastic_strain_list.resolve())

    target_composition = None
    if args.composition_json is not None:
        parsed = json.loads(args.composition_json)
        if not isinstance(parsed, dict):
            raise ValueError("--composition-json must decode to a JSON object")
        target_composition = normalize_composition_map(parsed)

    existing_structures: list[Atoms] = []
    existing_hashes: set[str] = set()
    if output_path.exists() and not args.overwrite:
        existing_structures = load_existing_output(output_path)
        existing_hashes = {hash_structure(atoms) for atoms in existing_structures}

    seed_index = (
        args.seed_start
        if args.seed_start is not None
        else next_seed_index(existing_structures, args.seed_prefix, fallback=0)
    )

    input_files = collect_input_files(
        inputs=args.input,
        patterns=patterns,
        recursive=args.recursive,
        output_path=output_path,
    )
    if not input_files:
        print("No input structure files found.")
        return 1

    imported: list[Atoms] = []
    imported_seed_structures: list[Atoms] = []
    skipped_duplicates = 0
    skipped_errors = 0

    for structure_file in input_files:
        try:
            structures = read_structures(structure_file)
        except Exception as exc:
            print(f"Skipping unreadable file: {structure_file} ({exc})")
            skipped_errors += 1
            continue

        for frame_index, atoms in enumerate(structures):
            structure_hash = hash_structure(atoms)
            if not args.allow_duplicates and structure_hash in existing_hashes:
                skipped_duplicates += 1
                continue

            atoms = atoms.copy()
            atoms.info = dict(getattr(atoms, "info", {}) or {})

            actual_composition = normalized_composition_from_atoms(atoms)
            is_elastic_strain_structure = structure_file.parent in elastic_strain_dirs
            atoms.info["actual_composition"] = actual_composition
            atoms.info["composition"] = target_composition or actual_composition
            atoms.info["configurational_type"] = args.configurational_type
            atoms.info["perturbation_type"] = (
                "elastic_stress"
                if is_elastic_strain_structure
                else args.perturbation_type
            )
            atoms.info["source"] = (
                f"{args.source_prefix}:{structure_file.parent.name}/{structure_file.name}:{frame_index}"
            )
            atoms.info["import_file"] = str(structure_file)
            atoms.info["import_frame"] = frame_index
            atoms.info["is_seed_structure"] = structure_file in seed_list_paths
            atoms.info["is_elastic_strain_structure"] = is_elastic_strain_structure

            for key, value in user_tags.items():
                atoms.info[key] = value

            if not (
                args.preserve_existing_seed_ids and atoms.info.get("seed_id") is not None
            ):
                atoms.info["seed_id"] = f"{args.seed_prefix}{seed_index:06d}"
                seed_index += 1

            annotate_structure_hash(atoms, overwrite=True)
            imported.append(atoms)
            if structure_file in seed_list_paths:
                imported_seed_structures.append(atoms.copy())
            existing_hashes.add(structure_hash)

    print(f"Matched files: {len(input_files)}")
    print(f"Imported structures: {len(imported)}")
    print(f"Imported seed structures: {len(imported_seed_structures)}")
    print(
        "Imported elastic strain structures: "
        f"{sum(1 for atoms in imported if atoms.info.get('is_elastic_strain_structure'))}"
    )
    print(f"Skipped duplicates: {skipped_duplicates}")
    print(f"Skipped unreadable files: {skipped_errors}")

    if not imported:
        print("No new structures to write.")
        return 0

    if args.dry_run:
        print("Dry run only; output file not written.")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_append = output_path.exists() and not args.overwrite
    ase_write(str(output_path), imported, format="extxyz", append=write_append)
    print(f"Wrote {len(imported)} structures to {output_path}")

    if seed_output_path is not None and imported_seed_structures:
        seed_output_path.parent.mkdir(parents=True, exist_ok=True)
        seed_existing = [] if args.overwrite else load_existing_output(seed_output_path)
        seeds_to_write = [atoms.copy() for atoms in imported_seed_structures]
        for atoms in seeds_to_write:
            atoms.info = dict(getattr(atoms, "info", {}) or {})
            atoms.info["source"] = f"seed:{atoms.info.get('source', 'manual')}"
        merged_seeds, skipped_seed_duplicates = merge_unique_by_structure_hash(
            seed_existing,
            seeds_to_write,
        )
        ase_write(str(seed_output_path), merged_seeds, format="extxyz")
        print(
            f"Wrote {len(merged_seeds)} total seed structures to {seed_output_path} "
            f"({skipped_seed_duplicates} duplicates skipped while merging)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
