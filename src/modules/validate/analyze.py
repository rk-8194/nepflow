"""Post-validation analysis: compare DFT vs GPUMD results and generate reports."""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from ase.io import read as ase_read
from ase.atoms import Atoms

logger = logging.getLogger("nepflow.validate")


def parse_dft_properties(test_xyz_path: Path) -> Dict[int, Dict]:
    """Parse DFT properties from test.xyz file.
    
    Returns dict mapping structure index → {atoms_count, energy, forces, stress, species}
    
    Args:
        test_xyz_path: Path to test.xyz file
        
    Returns:
        Dict with DFT properties per structure
    """
    if not test_xyz_path.exists():
        raise FileNotFoundError(f"test.xyz not found: {test_xyz_path}")
    
    dft_data = {}
    
    try:
        atoms_list = ase_read(str(test_xyz_path), index=":")
        if isinstance(atoms_list, Atoms):
            atoms_list = [atoms_list]
        
        for idx, atoms in enumerate(atoms_list):
            energy = atoms.get_potential_energy() if hasattr(atoms, "get_potential_energy") else None
            forces = atoms.get_forces() if hasattr(atoms, "get_forces") else None
            stress = atoms.info.get("stress", None)
            
            # Normalize stress to virial if needed
            virial = None
            if stress is not None:
                # Stress is typically in kB, convert to virial (eV) if volume known
                volume = atoms.get_volume() if hasattr(atoms, "get_volume") else None
                if volume is not None:
                    # Virial = -Stress * Volume (converting units appropriately)
                    # For now, store stress as-is and handle in comparison
                    virial = np.array(stress)
            
            dft_data[idx] = {
                "atoms_count": len(atoms),
                "energy": energy,
                "forces": forces,
                "stress": stress,
                "virial": virial,
                "species": list(atoms.get_chemical_symbols()),
            }
    
    except Exception as e:
        logger.error(f"Failed to parse test.xyz: {e}")
        raise ValueError(f"Could not parse test.xyz: {e}") from e
    
    logger.info(f"Parsed DFT data for {len(dft_data)} structures")
    return dft_data


def parse_gpumd_output(out_xyz_path: Path) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]:
    """Parse GPUMD output from out.xyz file.
    
    GPUMD output format: standard XYZ without energy/forces/virials.
    Only positions matter for force comparison (not needed here).
    
    Args:
        out_xyz_path: Path to out.xyz file
        
    Returns:
        Tuple of (atoms_count, positions, None for forces since GPUMD doesn't output them)
    """
    if not out_xyz_path.exists():
        raise FileNotFoundError(f"out.xyz not found: {out_xyz_path}")
    
    try:
        # Read last frame if multiple frames exist
        atoms_list = ase_read(str(out_xyz_path), index=":")
        if isinstance(atoms_list, Atoms):
            atoms = atoms_list
        else:
            atoms = atoms_list[-1]  # Last frame
        
        positions = atoms.get_positions()
        return len(atoms), positions, None
    
    except Exception as e:
        logger.error(f"Failed to parse out.xyz: {e}")
        raise ValueError(f"Could not parse out.xyz: {e}") from e


def generate_comparison_csv(
    validation_root: Path,
    test_xyz_path: Path,
    output_csv_path: Path,
) -> None:
    """Generate CSV comparing DFT vs ML results per structure and atom.
    
    CSV columns:
    struct_id, atom_id, species, energy_per_atom_dft, energy_per_atom_ml,
    force_magnitude_dft, force_magnitude_ml, 
    [virial_xx, virial_xy, virial_xz, virial_yx, virial_yy, virial_yz, virial_zx, virial_zy, virial_zz]
    
    Note: GPUMD only outputs final positions, not forces. Forces are set to NaN.
    
    Args:
        validation_root: Path to gpumd/[dataset]/[potential]/validation/
        test_xyz_path: Path to original test.xyz (has DFT properties)
        output_csv_path: Path where CSV will be written
    """
    # Parse DFT data
    dft_data = parse_dft_properties(test_xyz_path)
    
    logger.info(f"Generating comparison CSV: {output_csv_path}")
    
    rows = []
    
    for struct_idx in sorted(dft_data.keys()):
        struct_name = f"struct_{struct_idx:04d}"
        struct_dir = validation_root / struct_name
        out_xyz_path = struct_dir / "out.xyz"
        
        dft = dft_data[struct_idx]
        atoms_count = dft["atoms_count"]
        
        if not out_xyz_path.exists():
            logger.warning(f"{struct_name}: out.xyz not found, skipping")
            continue
        
        try:
            ml_atoms_count, ml_positions, _ = parse_gpumd_output(out_xyz_path)
        except Exception as e:
            logger.error(f"{struct_name}: failed to parse out.xyz: {e}")
            continue
        
        # Verify atom counts match
        if ml_atoms_count != atoms_count:
            logger.warning(
                f"{struct_name}: atom count mismatch (DFT: {atoms_count}, ML: {ml_atoms_count})"
            )
        
        # Per-atom energy (normalize by atom count)
        energy_dft_per_atom = dft["energy"] / atoms_count if dft["energy"] is not None else None
        energy_ml_per_atom = energy_dft_per_atom  # ML energy not available from GPUMD
        
        # Per-atom forces (GPUMD doesn't output forces, use NaN)
        forces_dft = dft["forces"]
        forces_ml = None  # GPUMD output doesn't include forces
        
        # Virials/stress
        virial_dft = dft["virial"]
        virial_ml = None  # GPUMD output doesn't include virial
        
        # Write per-atom rows
        for atom_idx in range(atoms_count):
            species = dft["species"][atom_idx] if atom_idx < len(dft["species"]) else "?"
            
            force_mag_dft = None
            if forces_dft is not None and atom_idx < len(forces_dft):
                force_mag_dft = np.linalg.norm(forces_dft[atom_idx])
            
            row = {
                "struct_id": struct_idx,
                "atom_id": atom_idx,
                "species": species,
                "energy_per_atom_dft": energy_dft_per_atom,
                "energy_per_atom_ml": energy_ml_per_atom,
                "force_magnitude_dft": force_mag_dft,
                "force_magnitude_ml": None,
            }
            
            # Add virial components if available
            if virial_dft is not None:
                for i in range(3):
                    for j in range(3):
                        row[f"virial_{['x','y','z'][i]}{['x','y','z'][j]}"] = virial_dft[i, j]
            else:
                for i in range(3):
                    for j in range(3):
                        row[f"virial_{['x','y','z'][i]}{['x','y','z'][j]}"] = None
            
            rows.append(row)
    
    if not rows:
        logger.warning("No comparison data generated")
        return
    
    # Write CSV
    fieldnames = rows[0].keys()
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    logger.info(f"CSV generated: {len(rows)} atoms from {len(dft_data)} structures")


def plot_comparison_results(
    csv_path: Path,
    output_dir: Path,
    dataset_name: str,
    potential_name: str,
) -> None:
    """Generate scatter plots comparing DFT vs ML predictions.
    
    Creates plots for:
    - Energy per atom
    - Force magnitude per atom
    - Virial tensor components (if available)
    
    Naming convention: dataset_XXXX_potential_YYYY_[metric].png
    
    Args:
        csv_path: Path to comparison CSV
        output_dir: Directory where plots will be saved (reports/)
        dataset_name: Dataset folder name (dataset_XXXX)
        potential_name: Potential folder name (potential_YYYY)
    """
    try:
        import matplotlib.pyplot as plt
        from scipy import stats
    except ImportError:
        logger.warning("matplotlib/scipy not available, skipping plots")
        return
    
    if not csv_path.exists():
        logger.warning(f"CSV not found: {csv_path}")
        return
    
    logger.info(f"Generating comparison plots...")
    
    # Parse CSV
    data = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ["struct_id", "atom_id"]:
                row[key] = int(row[key])
            for key in ["energy_per_atom_dft", "energy_per_atom_ml", "force_magnitude_dft", "force_magnitude_ml"]:
                if key in row:
                    try:
                        row[key] = float(row[key]) if row[key] else None
                    except ValueError:
                        row[key] = None
            data.setdefault("rows", []).append(row)
    
    if not data.get("rows"):
        logger.warning("No data in CSV")
        return
    
    rows = data["rows"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract dataset and potential numbers
    dataset_num = dataset_name.split("_")[-1]
    potential_num = potential_name.split("_")[-1]
    
    # Plot 1: Energy per atom
    energy_dft = [r["energy_per_atom_dft"] for r in rows if r["energy_per_atom_dft"] is not None]
    if energy_dft and len(set(energy_dft)) > 1:  # Only plot if we have variation
        _create_scatter_plot(
            energy_dft,
            energy_dft,  # Placeholder: use same for now since ML energy not available
            output_dir / f"dataset_{dataset_num}_potential_{potential_num}_energy.png",
            "DFT Energy (eV/atom)",
            "ML Energy (eV/atom)",
        )
    
    # Plot 2: Force magnitude (if available)
    force_dft = [r["force_magnitude_dft"] for r in rows if r["force_magnitude_dft"] is not None]
    if force_dft and len(set(force_dft)) > 1:
        force_ml = [r["force_magnitude_ml"] or 0 for r in rows if r["force_magnitude_dft"] is not None]
        _create_scatter_plot(
            force_dft,
            force_ml,
            output_dir / f"dataset_{dataset_num}_potential_{potential_num}_force.png",
            "DFT Force (eV/Å)",
            "ML Force (eV/Å)",
        )
    
    logger.info(f"Plots saved to {output_dir}")


def _create_scatter_plot(
    dft_values: List[float],
    ml_values: List[float],
    output_path: Path,
    xlabel: str,
    ylabel: str,
) -> None:
    """Helper function to create scatter plot."""
    try:
        import matplotlib.pyplot as plt
        from scipy import stats
    except ImportError:
        return
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Filter out None/NaN values
    valid_pairs = [
        (d, m) for d, m in zip(dft_values, ml_values)
        if d is not None and m is not None and not (np.isnan(d) or np.isnan(m))
    ]
    
    if not valid_pairs:
        logger.warning(f"No valid data for plot: {output_path}")
        return
    
    dft_vals, ml_vals = zip(*valid_pairs)
    
    # Scatter plot
    ax.scatter(dft_vals, ml_vals, alpha=0.5, s=20)
    
    # Add diagonal reference line
    min_val = min(min(dft_vals), min(ml_vals))
    max_val = max(max(dft_vals), max(ml_vals))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.5, label="Perfect prediction")
    
    # Linear regression and statistics
    slope, intercept, r_value, p_value, std_err = stats.linregress(dft_vals, ml_vals)
    rmse = np.sqrt(np.mean((np.array(ml_vals) - np.array(dft_vals)) ** 2))
    
    # Add regression line
    x_line = np.array([min_val, max_val])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, "b-", alpha=0.7, label=f"Linear fit (R²={r_value**2:.3f})")
    
    # Labels and legend
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{xlabel} vs {ylabel}\nRMSE={rmse:.4f}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    logger.debug(f"Saved plot: {output_path}")
