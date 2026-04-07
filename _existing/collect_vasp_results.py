#!/usr/bin/env python3
"""
Extract results from completed VASP calculations and write to extended XYZ format.
Collects energies and forces from successful VASP runs using POSCAR for structure.

Usage:
    python collect_vasp_results.py

Output:
    vasp_results.xyz - Extended XYZ file with all successful structures
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

def parse_vasprun_xml(vasprun_path):
    """Parse vasprun.xml file and extract energy and forces only."""
    try:
        tree = ET.parse(vasprun_path)
        root = tree.getroot()
        
        # Get final energy (eV) - the LAST e_fr_energy in entire file
        energy = None
        all_energies = root.findall(".//i[@name='e_fr_energy']")
        if all_energies:
            # Last energy in the file is the converged one
            energy = float(all_energies[-1].text)
        
        # Get forces (eV/Å) - from last varray in file
        forces = None
        all_forces = root.findall(".//varray[@name='forces']")
        if all_forces:
            forces_elem = all_forces[-1]
            forces = []
            for v in forces_elem.findall('v'):
                force = [float(x) for x in v.text.split()]
                forces.append(force)
            forces = np.array(forces) if forces else None
        
        # Get stress tensor (kB) - from last varray in file
        stress = None
        all_stress = root.findall(".//varray[@name='stress']")
        if all_stress:
            stress_elem = all_stress[-1]
            stress = []
            for v in stress_elem.findall('v'):
                stress_row = [float(x) for x in v.text.split()]
                stress.append(stress_row)
            stress = np.array(stress) if stress else None
        
        return {
            'energy': energy,
            'forces': forces,
            'stress': stress,
            'lattice': None,      # Get from POSCAR
            'positions': None,    # Get from POSCAR
            'species': None       # Get from POSCAR
        }
    
    except Exception as e:
        print(f"Error parsing {vasprun_path}: {e}")
        return None

def read_poscar(poscar_path):
    """Read POSCAR file to get structure information."""
    try:
        with open(poscar_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        comment = lines[0].strip()
        scale = float(lines[1])
        
        # Lattice vectors
        lattice = []
        for i in range(2, 5):
            lattice.append([float(x) * scale for x in lines[i].split()])
        lattice = np.array(lattice)
        
        # Element types and counts
        elements = lines[5].split()
        counts = [int(x) for x in lines[6].split()]
        
        # Generate species list
        species = []
        for elem, count in zip(elements, counts):
            species.extend([elem] * count)
        
        # Coordinate type
        coord_type = lines[7].strip().lower()
        
        # Positions
        positions = []
        start_line = 8
        for i in range(start_line, start_line + sum(counts)):
            pos = [float(x) for x in lines[i].split()[:3]]
            if coord_type.startswith('d'):  # Direct coordinates
                # Convert to Cartesian
                pos = np.dot(pos, lattice)
            positions.append(pos)
        
        return {
            'lattice': lattice,
            'positions': np.array(positions),
            'species': species
        }
    
    except Exception as e:
        print(f"Error reading {poscar_path}: {e}")
        return None

def write_xyz_structure(f, struct_data):
    """Write a single structure to XYZ file."""
    positions = struct_data['positions']
    species = struct_data['species']
    lattice = struct_data['lattice']
    energy = struct_data['energy']
    forces = struct_data['forces']
    
    n_atoms = len(positions)
    
    # Write number of atoms
    f.write(f"{n_atoms}\n")
    
    # Write properties line in exact format as original file
    lattice_str = ' '.join([f'{x:.3f}' for row in lattice for x in row])
    
    # Properties line must match original format exactly
    properties_line = f'Lattice="{lattice_str}" Properties=species:S:1:pos:R:3:force:R:3 '
    properties_line += f'Config_type=neptrainkit energy={energy:.6f} pbc="T T T"\n'
    f.write(properties_line)
    
    # Write atomic data
    for spec, pos, force in zip(species, positions, forces):
        f.write(f'{spec:8s} {pos[0]:15.8f} {pos[1]:15.8f} {pos[2]:15.8f} ')
        f.write(f'{force[0]:15.8f} {force[1]:15.8f} {force[2]:15.8f}\n')

def check_outcar_completion(outcar_path):
    """Check if OUTCAR shows successful completion."""
    try:
        with open(outcar_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for error messages first
        error_keywords = ['ERROR', 'REFUSE TO CONTINUE', 'I GIVE UP', 'VERY BAD NEWS', 'EXITING']
        for keyword in error_keywords:
            if keyword in content:
                return False
        
        # Check for completion markers
        completion_markers = ['General timing', 'Voluntary context switches']
        for marker in completion_markers:
            if marker in content:
                return True
        
        return False
    
    except Exception as e:
        print(f"Error checking {outcar_path}: {e}")
        return False

def main():
    """Main collection routine."""
    base_dir = Path('.')
    output_file = 'vasp_results.xyz'
    
    successful_structures = []
    failed_structures = []
    
    print("=" * 72)
    print("VASP RESULTS COLLECTION")
    print("=" * 72)
    print(f"Scanning for structures...")
    print()
    
    # Auto-detect all struct_XXXX directories
    struct_dirs = sorted([d for d in base_dir.glob('struct_*') if d.is_dir()])
    
    if not struct_dirs:
        print("ERROR: No struct_* directories found")
        return
    
    print(f"Found {len(struct_dirs)} structures")
    print()
    
    # Iterate through all detected structures
    for struct_dir in struct_dirs:
        
        struct_name = struct_dir.name
        # Extract numeric part for struct_num
        try:
            struct_num = int(struct_name.replace('struct_', ''))
        except ValueError:
            continue
        
        if not struct_dir.exists():
            continue
        vasprun_path = struct_dir / 'vasprun.xml'
        poscar_path = struct_dir / 'POSCAR'
        outcar_path = struct_dir / 'OUTCAR'
        
        # Check OUTCAR completion first
        if outcar_path.exists() and not check_outcar_completion(outcar_path):
            print(f"struct_{struct_num}: FAILED (VASP error detected)")
            failed_structures.append(struct_num)
            continue
        
        if vasprun_path.exists():
            vasp_data = parse_vasprun_xml(vasprun_path)
        else:
            print(f"struct_{struct_num}: No vasprun.xml found, skipping")
            failed_structures.append(struct_num)
            continue
        
        if vasp_data is None:
            print(f"struct_{struct_num}: Failed to parse vasprun.xml")
            failed_structures.append(struct_num)
            continue
        
        # ALWAYS get structure from POSCAR (lattice, positions, species)
        poscar_data = read_poscar(poscar_path)
        if poscar_data is None:
            print(f"struct_{struct_num}: Failed to read POSCAR")
            failed_structures.append(struct_num)
            continue
        
        # Merge: VASP energy/forces + POSCAR structure
        vasp_data['lattice'] = poscar_data['lattice']
        vasp_data['positions'] = poscar_data['positions']
        vasp_data['species'] = poscar_data['species']
        
        # Validate data
        if (vasp_data['energy'] is None or vasp_data['forces'] is None or 
            vasp_data['lattice'] is None or vasp_data['positions'] is None or
            vasp_data['species'] is None):
            print(f"struct_{struct_num}: Incomplete data, skipping")
            failed_structures.append(struct_num)
            continue
        
        # Add structure number for tracking
        vasp_data['struct_num'] = struct_num
        successful_structures.append(vasp_data)
        print(f"struct_{struct_num}: SUCCESS (E={vasp_data['energy']:.6f} eV)")
    
    print("")
    print("=" * 72)
    print("COLLECTION SUMMARY")
    print("=" * 72)
    print(f"Successful structures: {len(successful_structures)}")
    print(f"Failed structures: {len(failed_structures)}")
    print(f"Total processed: {len(successful_structures) + len(failed_structures)}")
    print("=" * 72)
    
    if successful_structures:
        print(f"Writing {len(successful_structures)} structures to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for struct_data in successful_structures:
                write_xyz_structure(f, struct_data)
        
        print(f"✓ Successfully wrote {len(successful_structures)} structures to {output_file}")
        
        # Print energy range
        energies = [s['energy'] for s in successful_structures]
        print(f"Energy range: {min(energies):.6f} to {max(energies):.6f} eV")
    else:
        print("No successful structures found!")
    
    if failed_structures:
        print(f"Failed structures: {', '.join([f'struct_{i:04d}' for i in failed_structures])}")

if __name__ == "__main__":
    main()
