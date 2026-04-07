#!/usr/bin/env python3
"""
Script to generate isotropically expanded structures.

This script reads a POSCAR file from the root directory, isotropically expands
the volume by different scaling factors, and creates subdirectories with the
new POSCAR files. It also copies INCAR, POTCAR, and KPOINTS files from the
root directory to each subdirectory.
"""

import os
import shutil
from pathlib import Path
from ase.io import read, write
from ase.atoms import Atoms
import numpy as np


def main():
    """
    Main function to generate isotropically expanded structures.
    
    Creates folders named struct_0000 to struct_9999 (or fewer if specified)
    with isotropically scaled atomic structures.
    """
    
    # Configuration
    root_dir = Path(".")
    all_dir = root_dir / "all"
    poscar_file = root_dir / "POSCAR"
    files_to_copy = ["INCAR", "POTCAR", "KPOINTS"]
    
    # Check if POSCAR exists
    if not poscar_file.exists():
        raise FileNotFoundError(f"POSCAR file not found at {poscar_file}")
    
    # Create the all directory if it doesn't exist
    all_dir.mkdir(exist_ok=True)
    
    # Read the original structure
    atoms = read(str(poscar_file))
    
    # Define scaling factors for volume expansion
    # You can modify this list to control which volume scalings are generated
    # Example: scale factors from 0.95 to 1.05 with 0.01 step
    scale_factors = np.linspace(0.8, 1.2, 11)  # 21 structures
    
    # Generate structures for each scale factor
    for i, scale_factor in enumerate(scale_factors):
        # Create folder name
        folder_name = f"struct_{i:04d}"
        struct_dir = all_dir / folder_name
        
        # Create the directory
        struct_dir.mkdir(exist_ok=True)
        
        # Scale the structure isotropically
        scaled_atoms = atoms.copy()
        # Scale volume by scaling the cell uniformly
        scaled_atoms.set_cell(atoms.cell * (scale_factor ** (1/3)), scale_atoms=True)
        
        # Write the new POSCAR file
        poscar_path = struct_dir / "POSCAR"
        write(str(poscar_path), scaled_atoms, format="vasp")
        
        # Copy supporting files
        for file_to_copy in files_to_copy:
            src_file = root_dir / file_to_copy
            dst_file = struct_dir / file_to_copy
            
            if src_file.exists():
                shutil.copy2(str(src_file), str(dst_file))
                print(f"Created {folder_name}: scale={scale_factor:.4f}, copied {file_to_copy}")
            else:
                print(f"Warning: {file_to_copy} not found in root directory for {folder_name}")
    
    print(f"\nSuccessfully created {len(scale_factors)} structures in {all_dir}")
    print(f"Scale factors range from {scale_factors[0]:.4f} to {scale_factors[-1]:.4f}")


if __name__ == "__main__":
    main()
