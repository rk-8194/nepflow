"""Prepare validation structures and datasets for GPUMD simulations."""

import logging
import shutil
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import numpy as np
from ase.io import read as ase_read
from ase.atoms import Atoms

from nepflow.mlip.nep.artifacts import (
    NepArtifactError,
    find_model_run_manifest,
    validate_model_run_manifest,
)
from nepflow.io.hashing import sha256_file

logger = logging.getLogger("nepflow.validate")


def _validated_model_run(project_dir: Path, model_run_id: str) -> dict:
    """Load and validate the manifest for one explicitly requested run."""
    manifest_path = find_model_run_manifest(project_dir, model_run_id)
    try:
        manifest = validate_model_run_manifest(
            manifest_path,
            expected_model_run_id=model_run_id,
        )
    except (FileNotFoundError, NepArtifactError) as exc:
        raise RuntimeError(f"Invalid model-run manifest for {model_run_id}") from exc
    artifact_path = Path(str(manifest["potential_artifact_path"])).resolve()
    canonical_dir = (project_dir / "nep" / "potentials").resolve()
    if artifact_path.parent.parent != canonical_dir:
        raise RuntimeError(
            f"Model artifact is outside canonical NEP storage: {artifact_path}"
        )
    return manifest


def find_model_run_and_dataset(project_dir: Path, model_run_id: str) -> Tuple[Path, Path]:
    """Resolve one completed model run through its persisted manifest."""
    manifest = _validated_model_run(project_dir, model_run_id)

    potential_artifact = Path(str(manifest["potential_artifact_path"])).resolve()
    potential_path = potential_artifact.parent
    canonical_dir = (project_dir / "nep" / "potentials").resolve()
    try:
        if potential_path.parent != canonical_dir:
            raise NepArtifactError(
                f"Model artifact is outside canonical NEP storage: {potential_artifact}"
            )
    except OSError as exc:
        raise RuntimeError(f"Could not resolve model artifact path: {potential_artifact}") from exc

    dataset_path = Path(str(manifest["dataset_path"])).resolve()
    logger.info(
        "Resolved model_run_id=%s to potential=%s and dataset_id=%s",
        model_run_id,
        potential_path,
        manifest["dataset_id"],
    )
    return potential_path, dataset_path


def find_latest_potential_and_dataset(
    project_dir: Path,
    model_run_id: str | None = None,
) -> Tuple[Path, Path]:
    """Compatibility wrapper requiring an explicit model-run identity.

    The historical name is retained for callers during the Phase 2 migration;
    it never performs latest-directory discovery.
    """
    if not model_run_id:
        raise ValueError("model_run_id is required; latest model discovery is disabled")
    return find_model_run_and_dataset(project_dir, model_run_id)


def finalize_nep_potential(
    project_dir: Path,
    model_run_id: str | None = None,
) -> Tuple[Path, Path]:
    """Finalize trained potential by moving it to gpumd structure.
    
    Copies nep.txt from nep/potentials/<run>/ to gpumd/dataset_XXXX/<run>/
    
    Args:
        project_dir: Project root directory
        
    Returns:
        Tuple of (gpumd_potential_path, dataset_folder_name)
        
    Raises:
        FileNotFoundError: If no potential found
    """
    gpumd_dir = project_dir / "gpumd"
    
    if not model_run_id:
        raise ValueError("model_run_id is required to finalize a potential")
    manifest = _validated_model_run(project_dir, model_run_id)
    source_artifact = Path(str(manifest["potential_artifact_path"])).resolve()
    potential_src = source_artifact.parent
    dataset_path = Path(str(manifest["dataset_path"])).resolve()
    
    # Extract folder names
    potential_name = potential_src.name  # potential_XXXX
    dataset_name = dataset_path.name      # dataset_XXXX
    
    # Create destination structure: gpumd/dataset_XXXX/potential_XXXX/
    gpumd_dataset_dir = gpumd_dir / dataset_name
    gpumd_potential_dir = gpumd_dataset_dir / potential_name
    gpumd_potential_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy exactly the artifact path validated from the manifest.
    src_nep = source_artifact
    dst_nep = gpumd_potential_dir / "nep.txt"
    
    if src_nep.exists() and sha256_file(
        src_nep,
        required=True,
        error_type=NepArtifactError,
    ) != manifest["potential_artifact_sha256"]:
        raise NepArtifactError(
            f"Manifest-bound artifact changed after validation: {src_nep}"
        )
    if src_nep.exists() and not dst_nep.exists():
        logger.info(f"Moving nep.txt to {gpumd_potential_dir}")
        shutil.copy2(src_nep, dst_nep)
        logger.debug(f"Copied nep.txt: {src_nep} → {dst_nep}")
    elif dst_nep.exists():
        if sha256_file(
            dst_nep,
            required=True,
            error_type=NepArtifactError,
        ) != manifest["potential_artifact_sha256"]:
            raise NepArtifactError(
                f"Existing finalized artifact does not match manifest: {dst_nep}"
            )
        logger.info(f"nep.txt already exists at {dst_nep}")
    else:
        raise FileNotFoundError(f"nep.txt not found at {src_nep}")
    
    logger.info(f"NEP potential finalized at: {gpumd_potential_dir}")
    
    return gpumd_potential_dir, dataset_name


def parse_cutoff_from_nep(nep_path: Path) -> float:
    """Extract potential cutoff radius from nep.txt file.
    
    The cutoff is on line 3 (index 2), space-separated format:
    cutoff 6 5 112 60
    
    Args:
        nep_path: Path to nep.txt file
        
    Returns:
        Cutoff radius in Angstroms
        
    Raises:
        ValueError: If cutoff cannot be parsed
    """
    try:
        lines = nep_path.read_text().strip().split("\n")
        if len(lines) < 3:
            raise ValueError("nep.txt too short")
        
        cutoff_line = lines[2]  # Line 3 (0-indexed)
        
        # Parse: "cutoff 6 5 112 60"
        parts = cutoff_line.split()
        if len(parts) < 2 or parts[0] != "cutoff":
            raise ValueError(f"Unexpected cutoff line format: {cutoff_line}")
        
        cutoff = float(parts[1])
        logger.debug(f"Parsed cutoff from nep.txt: {cutoff} Å")
        return cutoff
    except (OSError, IndexError, ValueError) as e:
        logger.error(f"Failed to parse cutoff from {nep_path}: {e}")
        raise ValueError(f"Could not parse cutoff from nep.txt: {e}") from e


def parse_lattice_from_xyz(atoms: Atoms) -> np.ndarray:
    """Extract lattice vectors from ASE Atoms object.
    
    Args:
        atoms: ASE Atoms object
        
    Returns:
        3x3 lattice matrix (rows are lattice vectors)
    """
    lattice = atoms.get_cell()
    return np.array(lattice)


def _cell_perpendicular_heights_angstrom(cell: np.ndarray) -> np.ndarray:
    """Return perpendicular lattice-plane heights for row-wise cell vectors.

    For cell rows ``a``, ``b``, and ``c``, the heights are ``V / |b x c|``,
    ``V / |c x a|``, and ``V / |a x b|``, respectively, where ``V`` is the
    absolute cell volume.

    Raises:
        ValueError: If the cell is not a finite, non-degenerate 3x3 matrix.
    """
    try:
        vectors = np.asarray(cell, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid cell: could not convert cell to floats: {exc}") from exc

    if vectors.shape != (3, 3):
        raise ValueError(
            f"Invalid cell shape {vectors.shape}; expected a 3x3 cell matrix"
        )
    if not np.all(np.isfinite(vectors)):
        raise ValueError("Invalid cell: all cell entries must be finite")

    with np.errstate(over="ignore", invalid="ignore"):
        volume = abs(float(np.linalg.det(vectors)))
        face_areas = np.array(
            [
                np.linalg.norm(np.cross(vectors[1], vectors[2])),
                np.linalg.norm(np.cross(vectors[2], vectors[0])),
                np.linalg.norm(np.cross(vectors[0], vectors[1])),
            ],
            dtype=float,
        )

    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError(f"Invalid cell volume: {volume!r}")
    if not np.all(np.isfinite(face_areas)) or np.any(face_areas <= 0.0):
        raise ValueError(f"Invalid cell opposite-face area: {face_areas!r}")

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        heights = volume / face_areas
    if not np.all(np.isfinite(heights)) or np.any(heights <= 0.0):
        raise ValueError(f"Invalid cell height: {heights!r}")
    return heights


def calculate_required_replicates(
    model_xyz_path: Path, nep_path: Path
) -> Tuple[int, int, int]:
    """Calculate required replicates per direction to satisfy model thickness constraint.
    
    Constraint: min_thickness_in_direction > 2 * cutoff
    
    Args:
        model_xyz_path: Path to model.xyz file
        nep_path: Path to nep.txt file
        
    Returns:
        Tuple of (nx, ny, nz) replicate counts
        
    Raises:
        ValueError: If parsing fails or constraint cannot be satisfied
    """
    cutoff = parse_cutoff_from_nep(nep_path)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError(
            f"Invalid cutoff {cutoff!r}: cutoff must be finite and strictly positive"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        min_required_thickness = 2.0 * cutoff
    if not np.isfinite(min_required_thickness):
        raise ValueError(
            f"Invalid cutoff {cutoff!r}: required height is not finite"
        )
    
    try:
        atoms = ase_read(str(model_xyz_path))
    except Exception as e:
        logger.error(f"Failed to read model.xyz: {e}")
        raise ValueError(f"Could not read model.xyz: {e}") from e
    
    thicknesses = _cell_perpendicular_heights_angstrom(atoms.get_cell())
    
    logger.debug(f"Cutoff: {cutoff} Å, min required thickness: {min_required_thickness} Å")
    logger.debug(f"Model thicknesses: {thicknesses}")
    
    replicates = []
    for i, thickness in enumerate(thicknesses):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            ratio = min_required_thickness / thickness
        if not np.isfinite(ratio):
            raise ValueError(
                f"Could not calculate repeat count for cell height {thickness!r} "
                f"and cutoff {cutoff!r}"
            )

        # Start above the quotient, then verify the strict inequality below.
        n_replicate = max(1, int(np.floor(ratio)) + 1)
        while n_replicate * thickness <= min_required_thickness:
            n_replicate += 1
        replicates.append(n_replicate)
        logger.debug(
            f"  Direction {i}: thickness={thickness:.3f} Å, "
            f"replicates={n_replicate} → effective thickness={thickness * n_replicate:.3f} Å"
        )
    
    nx, ny, nz = replicates
    
    # Verify the strict constraint after the guarded estimate.
    for i, (n, thickness) in enumerate(zip(replicates, thicknesses)):
        effective_thickness = thickness * n
        if effective_thickness <= min_required_thickness:
            raise ValueError(
                f"Direction {i}: thickness constraint may not be fully satisfied "
                f"({effective_thickness:.3f} Å ≈ {min_required_thickness:.3f} Å)"
            )
    
    logger.info(f"Calculated replicates: replicate {nx} {ny} {nz}")
    return nx, ny, nz


def parse_test_xyz(test_xyz_path: Path) -> List[Dict]:
    """Parse test.xyz file and extract structures with properties.
    
    Args:
        test_xyz_path: Path to test.xyz file
        
    Returns:
        List of dicts with keys: 'atoms', 'energy', 'forces', 'stress', 'index'
        
    Raises:
        FileNotFoundError: If test.xyz not found
        ValueError: If parsing fails
    """
    if not test_xyz_path.exists():
        raise FileNotFoundError(f"test.xyz not found: {test_xyz_path}")
    
    try:
        # Read all structures from XYZ file
        structures = []
        atoms_list = ase_read(str(test_xyz_path), index=":")
        
        # Handle single structure vs multiple
        if isinstance(atoms_list, Atoms):
            atoms_list = [atoms_list]
        
        for idx, atoms in enumerate(atoms_list):
            try:
                energy = float(atoms.get_potential_energy())
            except Exception:
                energy = None
            force_value = atoms.arrays.get("force")
            if force_value is None:
                force_value = atoms.arrays.get("forces")
            struct_data = {
                "atoms": atoms,
                "index": idx,
                "energy": energy,
                "forces": force_value,
                "stress": atoms.info.get("stress", None),
            }
            structures.append(struct_data)
        
        logger.info(f"Parsed {len(structures)} structures from test.xyz")
        return structures
    
    except Exception as e:
        logger.error(f"Failed to parse test.xyz: {e}")
        raise ValueError(f"Could not parse test.xyz: {e}") from e


def create_model_xyz_from_structure(atoms: Atoms) -> str:
    """Create model.xyz content from ASE Atoms object (without energy/forces).
    
    Args:
        atoms: ASE Atoms object
        
    Returns:
        XYZ file content as string
    """
    from ase.io import write
    from io import StringIO
    
    # GPUMD consumes extended XYZ so the reference cell and PBC survive into
    # its model evaluation.  Build a label-free copy: DFT energy/force/virial
    # fields must never be copied into the model input as predictions.
    model_atoms = Atoms(
        symbols=atoms.get_chemical_symbols(),
        positions=np.asarray(atoms.positions, dtype=float),
        cell=np.asarray(atoms.cell, dtype=float),
        pbc=np.asarray(atoms.pbc, dtype=bool),
    )
    output = StringIO()
    write(output, model_atoms, format="extxyz", write_info=False, write_results=False)
    return output.getvalue()


def prepare_validation_structures(
    dataset_path: Path,
    gpumd_potential_dir: Path,
    project_dir: Path,
    config_gpumd_dir: Path,
) -> Dict:
    """Prepare validation structure folders for GPUMD simulations.
    
    Creates folder structure:
    gpumd/dataset_XXXX/potential_YYYY/validation/struct_0000/
                                                    struct_0001/
                                                    ...
    
    Each struct folder contains:
    - run.in (modified replicate line)
    - nep.txt (copy of potential)
    - model.xyz (structure to validate)
    
    Args:
        dataset_path: Path to nep/datasets/dataset_XXXX/
        gpumd_potential_dir: Path to gpumd/dataset_XXXX/potential_YYYY/
        project_dir: Project root directory
        config_gpumd_dir: Path to config/gpumd/ folder
        
    Returns:
        Dict with validation preparation state (for launcher to use)
    """
    # Read test.xyz structures
    test_xyz_path = dataset_path / "test.xyz"
    structures = parse_test_xyz(test_xyz_path)
    
    # Create validation root
    validation_root = gpumd_potential_dir / "validation"
    validation_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Creating validation directory: {validation_root}")
    
    # Get cutoff and base replicate values for later
    nep_path = gpumd_potential_dir / "nep.txt"
    cutoff = parse_cutoff_from_nep(nep_path)
    
    # Read template run.in_validate
    template_run_in = config_gpumd_dir / "run.in_validate"
    if not template_run_in.exists():
        raise FileNotFoundError(f"Template run.in_validate not found: {template_run_in}")
    
    struct_folders = []
    
    for struct_idx, struct_data in enumerate(structures):
        struct_name = f"struct_{struct_idx:04d}"
        struct_dir = validation_root / struct_name
        struct_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"Preparing {struct_name}")
        
        # Extract atoms
        atoms = struct_data["atoms"]
        
        # Create model.xyz
        model_xyz_path = struct_dir / "model.xyz"
        model_xyz_content = create_model_xyz_from_structure(atoms)
        model_xyz_path.write_text(model_xyz_content)
        logger.debug(f"  Created model.xyz ({len(atoms)} atoms)")
        
        # Calculate required replicates. Geometry failures must stop
        # preparation rather than producing an unsafe default protocol.
        nx, ny, nz = calculate_required_replicates(model_xyz_path, nep_path)
        
        # Generate the authoritative static NEP protocol.  The template is
        # intentionally not inherited: arbitrary GPUMD commands could alter
        # the physical configuration or select a different model artifact.
        dump_command = "dump_xyz 1 out.xyz precision double force potential virial"
        setup_lines = [
            f"replicate {nx} {ny} {nz}",
            "potential nep.txt",
            "ensemble nve",
            "time_step 0",
            dump_command,
            "run 1",
        ]
        run_in_content = "\n".join(setup_lines) + "\n"
        
        run_in_path = struct_dir / "run.in"
        run_in_path.write_text(run_in_content)
        logger.debug(f"  Created run.in (replicate {nx} {ny} {nz})")
        
        # Copy nep.txt
        dst_nep = struct_dir / "nep.txt"
        shutil.copy2(nep_path, dst_nep)
        logger.debug(f"  Copied nep.txt")
        
        struct_folders.append({
            "name": struct_name,
            "path": str(struct_dir),
            "atoms_count": len(atoms),
            "replicates": [nx, ny, nz],
        })
    
    logger.info(f"Prepared {len(struct_folders)} validation structures")
    
    preparation_state = {
        "validation_root": str(validation_root),
        "dataset_path": str(dataset_path),
        "potential_path": str(gpumd_potential_dir),
        "nep_path": str(nep_path),
        "cutoff": cutoff,
        "struct_count": len(struct_folders),
        "struct_folders": struct_folders,
    }
    
    return preparation_state
