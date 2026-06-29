"""Prepare validation structures and datasets for GPUMD simulations."""

import logging
import re
import shutil
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import numpy as np
from ase.io import read as ase_read
from ase.atoms import Atoms

logger = logging.getLogger("nepflow.validate")


def find_latest_potential_and_dataset(project_dir: Path) -> Tuple[Path, Path]:
    """Find the latest trained NEP potential and its associated dataset.
    
    Args:
        project_dir: Project root directory
        
    Returns:
        Tuple of (potential_path, dataset_path)
        
    Raises:
        FileNotFoundError: If no potential or dataset found
    """
    nep_dir = project_dir / "nep"
    
    # Find latest potential folder (highest potential_XXXX number)
    potential_runs = sorted(nep_dir.glob("runs/potential_*"))
    if not potential_runs:
        raise FileNotFoundError("No potential folders found in nep/runs/")
    
    latest_potential = potential_runs[-1]
    nep_txt = latest_potential / "nep.txt"
    
    if not nep_txt.exists():
        raise FileNotFoundError(f"nep.txt not found in {latest_potential}")
    
    logger.info(f"Found latest potential: {latest_potential.name}")
    
    # Find latest dataset folder (highest dataset_XXXX number)
    datasets = sorted(nep_dir.glob("datasets/dataset_*"))
    if not datasets:
        raise FileNotFoundError("No dataset folders found in nep/datasets/")
    
    latest_dataset = datasets[-1]
    logger.info(f"Associated dataset: {latest_dataset.name}")
    
    return latest_potential, latest_dataset


def finalize_nep_potential(project_dir: Path) -> Tuple[Path, Path]:
    """Finalize trained potential by moving it to gpumd structure.
    
    Moves nep.txt from nep/runs/potential_XXXX/ to gpumd/dataset_XXXX/potential_XXXX/
    
    Args:
        project_dir: Project root directory
        
    Returns:
        Tuple of (gpumd_potential_path, dataset_folder_name)
        
    Raises:
        FileNotFoundError: If no potential found
    """
    nep_dir = project_dir / "nep"
    gpumd_dir = project_dir / "gpumd"
    
    # Find latest potential and dataset
    potential_src, dataset_path = find_latest_potential_and_dataset(project_dir)
    
    # Extract folder names
    potential_name = potential_src.name  # potential_XXXX
    dataset_name = dataset_path.name      # dataset_XXXX
    
    # Create destination structure: gpumd/dataset_XXXX/potential_XXXX/
    gpumd_dataset_dir = gpumd_dir / dataset_name
    gpumd_potential_dir = gpumd_dataset_dir / potential_name
    gpumd_potential_dir.mkdir(parents=True, exist_ok=True)
    
    # Move nep.txt to destination
    src_nep = potential_src / "nep.txt"
    dst_nep = gpumd_potential_dir / "nep.txt"
    
    if src_nep.exists() and not dst_nep.exists():
        logger.info(f"Moving nep.txt to {gpumd_potential_dir}")
        shutil.copy2(src_nep, dst_nep)
        logger.debug(f"Copied nep.txt: {src_nep} → {dst_nep}")
    elif dst_nep.exists():
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
    min_required_thickness = 2 * cutoff
    
    try:
        atoms = ase_read(str(model_xyz_path))
    except Exception as e:
        logger.error(f"Failed to read model.xyz: {e}")
        raise ValueError(f"Could not read model.xyz: {e}") from e
    
    cell = atoms.get_cell()
    
    # Calculate thickness in each direction as the projection of the cell on its normal
    # For a cubic/orthorhombic cell, this is simply the diagonal element
    # For arbitrary cells, compute the perpendicular distance
    thicknesses = []
    for i in range(3):
        # Normal direction is the i-th Cartesian axis
        # Compute perpendicular distance as |cell_vector_dot_normal| / |normal|
        height = abs(cell[i, i]) if np.abs(cell[i, i]) > 1e-10 else np.linalg.norm(cell[i, :])
        thicknesses.append(height)
    
    logger.debug(f"Cutoff: {cutoff} Å, min required thickness: {min_required_thickness} Å")
    logger.debug(f"Model thicknesses: {thicknesses}")
    
    replicates = []
    for i, thickness in enumerate(thicknesses):
        # Calculate minimum replicates needed
        n_replicate = int(np.ceil(min_required_thickness / thickness))
        # Ensure at least 1
        n_replicate = max(1, n_replicate)
        replicates.append(n_replicate)
        logger.debug(
            f"  Direction {i}: thickness={thickness:.3f} Å, "
            f"replicates={n_replicate} → effective thickness={thickness * n_replicate:.3f} Å"
        )
    
    nx, ny, nz = replicates
    
    # Verify constraint
    for i, (n, thickness) in enumerate(zip(replicates, thicknesses)):
        effective_thickness = thickness * n
        if effective_thickness <= min_required_thickness:
            logger.warning(
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
            struct_data = {
                "atoms": atoms,
                "index": idx,
                "energy": atoms.get_potential_energy() if "energy" in atoms.info else None,
                "forces": atoms.get_forces() if hasattr(atoms, "arrays") and "forces" in atoms.arrays else None,
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
    
    output = StringIO()
    write(output, atoms, format="xyz")
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
    
    template_content = template_run_in.read_text()
    logger.debug(f"Loaded template from {template_run_in}")
    
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
        
        # Calculate required replicates
        try:
            nx, ny, nz = calculate_required_replicates(model_xyz_path, nep_path)
        except ValueError as e:
            logger.warning(f"  Failed to calculate replicates for {struct_name}: {e}, using defaults")
            nx, ny, nz = 2, 2, 2
        
        # Create run.in with updated replicate line
        run_in_content = template_content
        # Replace "replicate 2 2 2" with calculated values
        run_in_content = re.sub(
            r"replicate\s+\d+\s+\d+\s+\d+",
            f"replicate {nx} {ny} {nz}",
            run_in_content,
        )
        
        run_in_path = struct_dir / "run.in"
        run_in_path.write_text(run_in_content)
        logger.debug(f"  Created run.in (replicate {nx} {ny} {nz})")
        
        # Copy nep.txt
        dst_nep = struct_dir / "nep.txt"
        shutil.copy2(nep_path, dst_nep)
        logger.debug(f"  Copied nep.txt")
        
        struct_folders.append({
            "name": struct_name,
            "path": struct_dir,
            "atoms_count": len(atoms),
            "replicates": (nx, ny, nz),
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
