"""Prepare sub-stage: parse OUTCAR files and write XYZ datasets."""

from configparser import ConfigParser
from pathlib import Path
from typing import Generator, List

from ase.atoms import Atoms
from ase.io import read as ase_read

import numpy as np

from ._common import HAS_TQDM, logger, parse_virial_from_outcar, tqdm, validate_structure


def prepare_dataset(
    dataset_path: Path,
    ase_structures: List[Atoms],
    is_train: bool,
    project_dir: Path,
    train_virial: bool = False,
    debug: bool = False,
) -> int:
    """Parse OUTCAR files and write XYZ dataset in streaming mode.
    
    Combines parsing and writing for memory efficiency with large datasets.
    
    Args:
        dataset_path: Output path for train.xyz or test.xyz
        ase_structures: List of ASE Atoms objects from selected structures
        is_train: Whether this is train (True) or test (False) dataset
        project_dir: Project root directory
        train_virial: Whether to include virial tensor
        debug: Whether to use synthetic data if OUTCAR missing
        
    Returns:
        Number of structures successfully written
    """
    dataset_type = "train" if is_train else "test"
    logger.info(f"Processing {len(ase_structures)} {dataset_type} structures")

    # Create generator for parsing (doesn't load all into memory)
    structure_generator = _parse_structures(
        ase_structures, is_train=is_train, project_dir=project_dir, debug=debug
    )

    # Wrap generator with progress bar if tqdm available
    if HAS_TQDM:
        structure_generator = tqdm(
            structure_generator,
            total=len(ase_structures),
            desc=f"Writing {dataset_type}.xyz",
            unit=" structures",
        )

    # Write XYZ file, streaming structures one at a time
    count = _write_xyz_file(dataset_path, structure_generator, include_virial=train_virial)

    logger.info(f"Wrote {count}/{len(ase_structures)} {dataset_type} structures to {dataset_path.name}")

    return count


def _parse_structures(
    ase_structures: List[Atoms],
    is_train: bool,
    project_dir: Path,
    debug: bool = False,
) -> Generator[dict, None, None]:
    """Generator: parse structures from ASE Atoms objects and find corresponding OUTCAR files.
    
    Streams results one at a time to avoid memory buildup with large datasets.
    """
    dataset_type = "train" if is_train else "test"
    vasp_jobs_path = project_dir / "vasp" / "jobs" / dataset_type

    if not vasp_jobs_path.exists():
        logger.warning(f"VASP jobs path not found: {vasp_jobs_path}")
        if not debug:
            logger.info("Cannot find OUTCAR files. Skipping this dataset.")
            return
        # In debug mode, yield synthetic data
        for struct_idx, atoms in enumerate(ase_structures):
            logger.debug(f"  [{dataset_type}] Generating synthetic structure {struct_idx}")
            yield _extract_from_atoms(atoms)
        return

    # Enumerate struct_XXXX folders in order
    struct_folders = sorted(
        [d for d in vasp_jobs_path.iterdir() if d.is_dir() and d.name.startswith("struct_")],
        key=lambda d: int(d.name.split("_")[1]),
    )

    logger.debug(f"Found {len(struct_folders)} struct folders in {vasp_jobs_path}")

    for struct_idx, atoms in enumerate(ase_structures):
        if struct_idx >= len(struct_folders):
            logger.warning(
                f"[{dataset_type}] More structures ({len(ase_structures)}) than struct folders ({len(struct_folders)}). Stopping."
            )
            break

        struct_folder = struct_folders[struct_idx]
        outcar_path = struct_folder / "OUTCAR"

        if not outcar_path.exists():
            logger.warning(f"[{dataset_type}] Skipping struct_{struct_idx:04d}: OUTCAR not found")
            continue

        try:
            logger.debug(f"[{dataset_type}] Parsing struct_{struct_idx:04d}")
            result = _parse_outcar(outcar_path, atoms)
            if result:
                yield result
            else:
                logger.warning(f"[{dataset_type}] Skipping struct_{struct_idx:04d}: parse failed")
        except Exception as e:
            logger.warning(f"[{dataset_type}] Skipping struct_{struct_idx:04d}: {e}")


def _parse_outcar(outcar_path: Path, ase_atoms: Atoms) -> dict | None:
    """Parse OUTCAR file using ASE with fallback to manual parsing."""
    try:
        # Try to read with ASE
        atoms = ase_read(str(outcar_path))

        result = {
            "energy": atoms.get_potential_energy(),
            "forces": atoms.get_forces(),
            "positions": atoms.get_positions(),
            "lattice": atoms.get_cell().array,
            "species": atoms.get_chemical_symbols(),
            "pbc": atoms.pbc.tolist(),
            "virial": None,
        }

        # Try to extract virial from atoms.info
        if "virial" in atoms.info:
            result["virial"] = atoms.info["virial"]
        else:
            # Try manual parsing from OUTCAR
            try:
                virial = parse_virial_from_outcar(outcar_path, atoms.get_volume())
                if virial is not None:
                    result["virial"] = virial
            except Exception as e:
                logger.debug(f"Could not extract virial from {outcar_path}: {e}")

        # Validate
        if not validate_structure(result):
            logger.warning(f"Validation failed for {outcar_path}")
            return None

        return result

    except Exception as e:
        logger.warning(f"ASE failed to parse {outcar_path}: {e}. Trying fallback.")
        # Fallback: use provided atoms if ASE fails
        return _extract_from_atoms(ase_atoms)


def _extract_from_atoms(atoms: Atoms) -> dict:
    """Extract structure data from ASE Atoms object."""
    result = {
        "energy": atoms.get_potential_energy() if "energy" in atoms.info or hasattr(atoms, "calc") else -1.0,
        "forces": atoms.get_forces() if "forces" in atoms.arrays else np.zeros((len(atoms), 3)),
        "positions": atoms.get_positions(),
        "lattice": atoms.get_cell().array,
        "species": atoms.get_chemical_symbols(),
        "pbc": atoms.pbc.tolist(),
        "virial": atoms.info.get("virial", None),
    }
    return result


def _write_xyz_file(
    output_path: Path,
    structures: Generator[dict, None, None],
    include_virial: bool = False,
) -> int:
    """Write extended XYZ file with NEP format, processing structures from generator.
    
    Streams structures one at a time to avoid memory buildup.
    Output format matches GPUMD NEP training requirements:
    - Line 1: number of atoms
    - Line 2: energy, pbc, Lattice, Properties (and optional virial)
    - Lines 3+: species, position (x, y, z), force (fx, fy, fz)
    
    Returns:
        Number of structures written
    """
    count = 0
    with open(output_path, "w") as f:
        for structure in structures:
            n_atoms = len(structure["species"])
            f.write(f"{n_atoms}\n")

            # Build header in correct order for NEP
            energy = structure["energy"]
            pbc = structure["pbc"]
            lattice = structure["lattice"]

            # Format PBC vector: "T T T" or "T T F" etc
            pbc_str = " ".join("T" if p else "F" for p in pbc)

            # Format Lattice as flattened 3x3: "ax ay az bx by bz cx cy cz"
            lattice_str = " ".join(f"{v:.10f}" for v in lattice.flat)

            # Properties descriptor: species, position, force
            properties_str = "species:S:1:pos:R:3:force:R:3"

            # Build header line in NEP format
            header = (
                f"energy={energy:.10f} "
                f'pbc="{pbc_str}" '
                f'Lattice="{lattice_str}" '
                f"Properties={properties_str}"
            )

            # Add virial if requested and available
            if include_virial and structure["virial"] is not None:
                virial_str = " ".join(f"{v:.10f}" for v in structure["virial"].flat)
                header += f' virial="{virial_str}"'

            f.write(header + "\n")

            # Write atoms: species followed by position (3 floats) then force (3 floats)
            for species, pos, force in zip(
                structure["species"],
                structure["positions"],
                structure["forces"],
            ):
                f.write(
                    f"{species:3s} {pos[0]:15.10f} {pos[1]:15.10f} {pos[2]:15.10f} "
                    f"{force[0]:15.10f} {force[1]:15.10f} {force[2]:15.10f}\n"
                )

            count += 1

    return count
