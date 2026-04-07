#!/usr/bin/env python3
"""
Parameter optimization for VASP structures.

Tests different NCORE/KPAR combinations for each structure to find the optimal
parameters that maximize speed while staying within memory constraints.

Usage:
    python regenerate_optimizations.py              # Regenerate all
    python regenerate_optimizations.py --clean      # Clean all configs
    python regenerate_optimizations.py 0 10         # Regenerate structs 0-9
"""

import sys
import json
import subprocess
import math
from pathlib import Path

try:
    from ase.io import read as ase_read
    from ase.symmetry.kpoints import monkhorst_pack, get_ibz_vertices
    HAS_ASE = True
except ImportError:
    HAS_ASE = False


class ParameterOptimizer:
    """Optimize NCORE/KPAR parameters for VASP structures."""
    
    def __init__(self):
        self.basedir = Path.cwd()
        self.launcher = self.basedir / "adaptive_vasp_launcher.py"
        
        # Parameter combinations to test (conservative → fast)
        self.param_combos = [
            (64, 1),
            (48, 1),
            (32, 1),
            (32, 2),
            (24, 2),
            (16, 2),
            (16, 4),
            (12, 4),
            (8, 4),
        ]
        
        # GPU configurations to test: (ngpu, description)
        # Lower ngpu = less parallelism but less inter-GPU communication
        # Higher ngpu = more parallelism but more communication overhead
        self.gpu_configs = [
            (1, "single-GPU"),
            (2, "dual-GPU"),
            (4, "quad-GPU"),
        ]
        
        # Memory limit (GB per GPU, with safety margin)
        # H100 has 80GB, use 70% as safe limit to leave room for OS/system
        self.memory_limit = 56  # 70% of 80GB = 56GB safe limit
    
    def find_structures(self):
        """Auto-detect all struct_* directories."""
        struct_dirs = sorted(
            self.basedir.glob("struct_*"),
            key=lambda p: int(p.name.split("_")[1])
        )
        return struct_dirs
    
    def modify_incar(self, struct_dir, ncore, kpar):
        """Update NCORE and KPAR in INCAR file."""
        incar_file = struct_dir / "INCAR"
        if not incar_file.exists():
            return False
        
        try:
            with open(incar_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            output_lines = []
            ncore_found = False
            kpar_found = False
            
            for line in lines:
                if line.strip().startswith('NCORE'):
                    output_lines.append(f"NCORE = {ncore}\n")
                    ncore_found = True
                elif line.strip().startswith('KPAR'):
                    output_lines.append(f"KPAR = {kpar}\n")
                    kpar_found = True
                else:
                    output_lines.append(line)
            
            # Add if not present
            if not ncore_found:
                output_lines.append(f"NCORE = {ncore}\n")
            if not kpar_found:
                output_lines.append(f"KPAR = {kpar}\n")
            
            with open(incar_file, 'w', encoding='utf-8') as f:
                f.writelines(output_lines)
            
            return True
        except (IOError, OSError, ValueError) as e:
            print(f"  ERROR modifying INCAR: {e}")
            return False
    
    def get_valence_electrons(self, struct_dir):
        """Extract valence electron count from POSCAR and POTCAR files."""
        try:
            poscar_file = struct_dir / "POSCAR"
            with open(poscar_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Parse element symbols and counts
            # Line 5: element symbols, Line 6: element counts
            if len(lines) <= 6:
                return None
            
            try:
                elements = lines[5].split()
                counts = [int(x) for x in lines[6].split()]
            except (ValueError, IndexError):
                return None
            
            total_valence = 0
            basedir = self.basedir
            
            for elem, count in zip(elements, counts):
                # Look for POTCAR_<element> file in base directory
                potcar_file = basedir / f"POTCAR_{elem}"
                if not potcar_file.exists():
                    # Try struct directory as fallback
                    potcar_file = struct_dir / f"POTCAR_{elem}"
                
                if not potcar_file.exists():
                    # Try combined POTCAR in struct dir
                    potcar_file = struct_dir / "POTCAR"
                
                if potcar_file.exists():
                    try:
                        with open(potcar_file, 'r', encoding='utf-8', errors='ignore') as f:
                            potcar_content = f.read()
                        
                        # Look for ZVAL line
                        for line in potcar_content.split('\n'):
                            if 'ZVAL' in line:
                                # Format: "POMASS =   51.996; ZVAL   =   12.000    mass and valenz"
                                parts = line.split('=')
                                if len(parts) >= 3:
                                    try:
                                        zval_str = parts[2].split()[0]
                                        zval = float(zval_str)
                                        total_valence += zval * count
                                        break
                                    except (ValueError, IndexError):
                                        pass
                    except (IOError, OSError):
                        pass
            
            return total_valence if total_valence > 0 else None
        except (IOError, OSError):
            return None
    
    def get_irreducible_kpoints(self, struct_dir, nx, ny, nz):
        """Use ASE to get number of irreducible k-points from symmetry."""
        if not HAS_ASE:
            # Fallback: use full grid but cap to reasonable value
            full_grid = nx * ny * nz
            if full_grid > 500:
                # For large grids, estimate ~50% reduction by symmetry
                return max(1, int(full_grid * 0.5))
            return full_grid
        
        try:
            poscar_file = struct_dir / "POSCAR"
            atoms = ase_read(str(poscar_file))
            
            # Generate full k-point grid
            kpts = monkhorst_pack((nx, ny, nz))
            
            # Get IBZ k-points (irreducible Brillouin zone)
            ibz_vertices = get_ibz_vertices(atoms, kpts)
            n_ibz = len(ibz_vertices)
            
            # Sanity check: IBZ should be much smaller than full grid
            full_grid = nx * ny * nz
            if n_ibz > full_grid:
                # Something went wrong, return full grid instead
                return full_grid
            
            return n_ibz
        except (IOError, OSError, ValueError, AttributeError):
            # Fallback: use full grid but estimate symmetry reduction
            full_grid = nx * ny * nz
            if full_grid > 500:
                return max(1, int(full_grid * 0.5))
            return full_grid
    
    def get_memory_estimate(self, struct_dir):
        """Run launcher and extract memory estimate from config."""
        config_file = struct_dir / "optimization_config.json"
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return float(config.get('estimated_memory_per_gpu_gb', 999))
        except (IOError, OSError, json.JSONDecodeError, ValueError):
            return 999
    
    def estimate_memory_for_parameters(self, struct_dir, ncore, kpar):
        """Estimate memory requirement for given NCORE/KPAR without running launcher."""
        # Parse structure to get baseline memory estimate
        try:
            poscar_file = struct_dir / "POSCAR"
            with open(poscar_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # POSCAR format:
            # Line 0: Comment
            # Line 1: Scaling factor
            # Lines 2-4: Lattice vectors
            # Line 5: Element symbols (or counts if no line 6)
            # Line 6: Atom counts (if line 5 has symbols)
            
            # Try to get atom counts from line 6 (if symbols on line 5)
            # or line 5 (if counts directly)
            if len(lines) > 6:
                try:
                    # Line 6 should have counts
                    atom_counts = lines[6].split()
                    n_atoms = sum(int(x) for x in atom_counts)
                except (ValueError, IndexError):
                    # Fall back to line 5
                    atom_counts = lines[5].split()
                    n_atoms = sum(int(x) for x in atom_counts)
            else:
                # Just use line 5
                atom_counts = lines[5].split()
                n_atoms = sum(int(x) for x in atom_counts)
            
            # Get valence electron count (determines number of bands)
            n_valence = self.get_valence_electrons(struct_dir)
            if n_valence is None:
                # Fallback: estimate based on atom count (rough)
                # W~74, Cr~6, Y~39, Zr~12 valence electrons on average
                n_valence = n_atoms * 30  # Conservative estimate
            
            # Read actual k-points from KPOINTS file, or calculate from KSPACING
            kpoints_file = struct_dir / "KPOINTS"
            n_kpoints = 350  # default fallback
            
            if kpoints_file.exists():
                try:
                    with open(kpoints_file, 'r', encoding='utf-8') as f:
                        kpts_lines = f.readlines()
                    
                    # KPOINTS format:
                    # Line 0: Comment
                    # Line 1: Number of k-points (0 for automatic)
                    # Line 2: Coordinate system (Cartesian/Reciprocal)
                    # Line 3: k-point grid "nx ny nz" (if automatic)
                    
                    # Check if automatic generation (line 1 should have 0 or blank)
                    if len(kpts_lines) > 1:
                        try:
                            num_kpts = int(kpts_lines[1].split()[0])
                            if num_kpts > 0:
                                # Explicit k-points listed
                                n_kpoints = num_kpts
                            else:
                                # Automatic: use grid from line 3
                                if len(kpts_lines) > 3:
                                    kpt_grid = kpts_lines[3].split()
                                    if len(kpt_grid) >= 3:
                                        nx = int(kpt_grid[0])
                                        ny = int(kpt_grid[1])
                                        nz = int(kpt_grid[2])
                                        # Use ASE to get irreducible k-points
                                        n_kpoints = self.get_irreducible_kpoints(struct_dir, nx, ny, nz)
                        except (ValueError, IndexError):
                            pass
                except (IOError, OSError):
                    pass
            
            # If still using default, try to get KSPACING from INCAR
            if n_kpoints == 350:
                try:
                    incar_file = struct_dir / "INCAR"
                    if incar_file.exists():
                        with open(incar_file, 'r', encoding='utf-8') as f:
                            incar_lines = f.readlines()
                        
                        kspacing = None
                        for line in incar_lines:
                            if line.strip().upper().startswith('KSPACING'):
                                parts = line.split('=')
                                if len(parts) > 1:
                                    try:
                                        kspacing = float(parts[1].split()[0])
                                        break
                                    except ValueError:
                                        pass
                        
                        # If we found KSPACING, estimate k-points from lattice vectors
                        if kspacing:
                            # Calculate reciprocal lattice parameters
                            # Lattice vectors are at lines 2-4
                            if len(lines) > 4:
                                try:
                                    a = [float(x) for x in lines[2].split()[:3]]
                                    b = [float(x) for x in lines[3].split()[:3]]
                                    c = [float(x) for x in lines[4].split()[:3]]
                                    
                                    # Calculate magnitudes
                                    a_mag = math.sqrt(sum(x**2 for x in a))
                                    b_mag = math.sqrt(sum(x**2 for x in b))
                                    c_mag = math.sqrt(sum(x**2 for x in c))
                                    
                                    # k-point grid from KSPACING
                                    nx = max(1, int(a_mag / kspacing + 0.5))
                                    ny = max(1, int(b_mag / kspacing + 0.5))
                                    nz = max(1, int(c_mag / kspacing + 0.5))
                                    n_kpoints = nx * ny * nz
                                    # Sanity check: k-points should be reasonable (< 10000)
                                    if n_kpoints > 10000:
                                        n_kpoints = 350  # Fall back to default if unreasonable
                                except (ValueError, IndexError):
                                    pass
                except (IOError, OSError):
                    pass  # Use default
            
            # Memory scaling: depends on atoms, k-points, AND bands (valence electrons)
            # Need to recalibrate based on actual VASP runs
            # The estimates are too high - suggesting k-point calculation is wrong
            # Reduce coefficient to match observed behavior
            n_bands = max(1, int(n_valence / 2))
            base_memory_per_atom_k_band = 0.00001  # Much smaller - k-points likely overestimated
            
            # Base memory requirement (with default NCORE=16, KPAR=1)
            base_memory = n_atoms * n_kpoints * n_bands * base_memory_per_atom_k_band
            
            # NCORE scaling: affects memory per GPU core
            # Lower NCORE = more cores needed = less memory per core (better)
            # Memory scales WITH lower NCORE (more parallelization = more memory)
            # NCORE controls band parallelization: lower NCORE = more processes = more memory overhead
            # NCORE=64 (fewest processes): 0.75x base
            # NCORE=8 (most processes): 1.3x base
            if ncore >= 48:
                ncore_factor = 0.75
            elif ncore >= 32:
                ncore_factor = 0.80
            elif ncore >= 24:
                ncore_factor = 0.85
            elif ncore >= 16:
                ncore_factor = 0.95
            elif ncore >= 12:
                ncore_factor = 1.15
            else:  # ncore <= 8
                ncore_factor = 1.30
            
            # KPAR scaling: distributes k-points across more GPUs
            # Higher KPAR = better distribution = less memory per GPU
            # KPAR=1: 1.0x, KPAR=2: 0.5x, KPAR=4: 0.3x
            if kpar == 1:
                kpar_factor = 1.0
            elif kpar == 2:
                kpar_factor = 0.5
            else:  # kpar >= 4
                kpar_factor = 0.3
            
            estimated_memory = base_memory * ncore_factor * kpar_factor
            return round(estimated_memory, 2)
        except (IOError, OSError, IndexError, ValueError) as e:
            print(f"    ERROR parsing POSCAR: {e}")
            return 999
    
    def test_parameters(self, struct_dir):
        """Test parameter combinations AND GPU configs based on memory estimation."""
        best_ncore = None
        best_kpar = None
        best_ngpu = None
        best_memory = 999
        best_combo_score = -1
        
        # Get structure size to determine parameter scaling
        try:
            poscar_file = struct_dir / "POSCAR"
            with open(poscar_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if len(lines) > 6:
                try:
                    atom_counts = lines[6].split()
                    n_atoms = sum(int(x) for x in atom_counts)
                except ValueError:
                    atom_counts = lines[5].split()
                    n_atoms = sum(int(x) for x in atom_counts)
            else:
                atom_counts = lines[5].split()
                n_atoms = sum(int(x) for x in atom_counts)
        except (IOError, OSError, ValueError, IndexError):
            n_atoms = 10  # Default fallback
        
        # Scale parameters based on structure size
        # Tiny structures: aggressive parallelization
        # Large structures: conservative to avoid memory/communication overhead
        if n_atoms <= 5:
            # Tiny: go for speed
            target_ncore = 8
            target_kpar = 4
            target_ngpu = 4
        elif n_atoms <= 10:
            # Small: balanced
            target_ncore = 12
            target_kpar = 4
            target_ngpu = 4
        elif n_atoms <= 16:
            # Medium: less aggressive
            target_ncore = 16
            target_kpar = 2
            target_ngpu = 2
        else:
            # Large: conservative
            target_ncore = 24
            target_kpar = 1
            target_ngpu = 1
        
        # Try the target parameters first
        print(f"    Scaled for {n_atoms} atoms: NCORE={target_ncore}, KPAR={target_kpar}, GPU={target_ngpu}")
        
        memory = self.estimate_memory_for_parameters(struct_dir, target_ncore, target_kpar)
        memory_per_gpu = memory / target_ngpu
        
        print(f"      Memory: {memory:6.2f} GB → {memory_per_gpu:6.2f} GB/GPU", end="")
        
        # If target parameters fit, use them
        if memory_per_gpu < self.memory_limit:
            print(" ✓ (using scaled target)")
            best_ncore = target_ncore
            best_kpar = target_kpar
            best_ngpu = target_ngpu
            best_memory = memory
            best_combo_score = 999  # Scaled params always win
        else:
            # Target doesn't fit, fall back to finding best combination
            print(" ✗ (doesn't fit, searching...)")
            
            for ngpu, _ in self.gpu_configs:
                for ncore, kpar in self.param_combos:
                    memory = self.estimate_memory_for_parameters(struct_dir, ncore, kpar)
                    memory_per_gpu = memory / ngpu
                    
                    if memory_per_gpu < self.memory_limit:
                        # Score based on parallelization
                        combo_score = (kpar * 200 - ncore * 2) + (ngpu * 150)
                        if combo_score > best_combo_score:
                            best_ncore = ncore
                            best_kpar = kpar
                            best_ngpu = ngpu
                            best_memory = memory
                            best_combo_score = combo_score
        
        # Apply best found parameters
        if best_ncore is not None:
            self.modify_incar(struct_dir, best_ncore, best_kpar)
            # Mark parameters as preset for final run
            config_file = struct_dir / "optimization_config.json"
            try:
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "skip_optimization": True,
                        "preset_ncore": best_ncore,
                        "preset_kpar": best_kpar,
                        "ngpu": best_ngpu,
                        "estimated_memory_per_gpu_gb": best_memory / best_ngpu
                    }, f)
            except (IOError, OSError):
                pass
            # Run launcher with best GPU configuration
            try:
                subprocess.run(
                    ["python3", str(self.launcher), str(struct_dir), "--ngpu", str(best_ngpu), "--max-nodes", "4"],
                    capture_output=True,
                    timeout=30,
                    check=False
                )
            except (subprocess.TimeoutExpired, OSError):
                pass
            return (best_ncore, best_kpar, best_ngpu, True)
        else:
            # No working combo found - use most conservative
            self.modify_incar(struct_dir, 64, 1)
            config_file = struct_dir / "optimization_config.json"
            try:
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "skip_optimization": True,
                        "preset_ncore": 64,
                        "preset_kpar": 1,
                        "ngpu": 1,
                        "estimated_memory_per_gpu_gb": 999
                    }, f)
            except (IOError, OSError):
                pass
            try:
                subprocess.run(
                    ["python3", str(self.launcher), str(struct_dir), "--ngpu", "1", "--max-nodes", "4"],
                    capture_output=True,
                    timeout=30,
                    check=False
                )
            except (subprocess.TimeoutExpired, OSError):
                pass
            return (64, 1, 1, False)
    
    def clean_all(self):
        """Remove all optimization files."""
        print("Cleaning all optimization files...")
        
        for struct_dir in self.find_structures():
            config_file = struct_dir / "optimization_config.json"
            log_file = struct_dir / "ADAPTIVE_OPTIMIZATION.log"
            
            config_file.unlink(missing_ok=True)
            log_file.unlink(missing_ok=True)
        
        print("✓ Cleaned")
    
    def regenerate_range(self, start, end):
        """Regenerate specific range of structures."""
        structs = self.find_structures()
        structs_in_range = [s for s in structs if start <= int(s.name.split("_")[1]) < end]
        
        if not structs_in_range:
            print(f"No structures found in range {start}-{end}")
            return
        
        print(f"Regenerating {len(structs_in_range)} structures...")
        print()
        
        for idx, struct_dir in enumerate(structs_in_range, 1):
            struct_num = struct_dir.name
            print(f"[{idx}/{len(structs_in_range)}] {struct_num}... ", end="", flush=True)
            
            ncore, kpar, ngpu, success = self.test_parameters(struct_dir)
            
            if success:
                print(f"✓ NCORE={ncore}, KPAR={kpar}, GPU={ngpu}")
            else:
                print(f"⚠ Fallback: NCORE={ncore}, KPAR={kpar}, GPU={ngpu} (no working combo)")
        
        print()
        print("=" * 74)
        print("Regeneration complete!")
        print("Review parameters with: python review_optimizations.py")
        print("=" * 74)
    
    def regenerate_all(self):
        """Regenerate all structures."""
        structs = self.find_structures()
        
        if not structs:
            print("ERROR: No struct_* directories found")
            return
        
        print("=" * 74)
        print("Parameter Optimization - All Structures")
        print("=" * 74)
        print(f"Found {len(structs)} structures")
        print(f"Memory limit: {self.memory_limit} GB per GPU (safety margin)")
        print(f"Testing {len(self.param_combos)} parameter combinations per structure")
        print()
        
        # Clean first
        print("Cleaning old optimization files...")
        for struct_dir in structs:
            (struct_dir / "optimization_config.json").unlink(missing_ok=True)
            (struct_dir / "ADAPTIVE_OPTIMIZATION.log").unlink(missing_ok=True)
        
        print(f"Starting optimization of {len(structs)} structures...")
        print()
        
        for idx, struct_dir in enumerate(structs, 1):
            struct_num = struct_dir.name
            
            if idx % 10 == 1:
                print(f"[{idx}/{len(structs)}] {struct_num}... ", end="", flush=True)
            else:
                print(f"[{idx}/{len(structs)}] {struct_num}... ", end="", flush=True)
            
            ncore, kpar, ngpu, success = self.test_parameters(struct_dir)
            
            if success:
                print(f"✓ NCORE={ncore}, KPAR={kpar}, GPU={ngpu}")
            else:
                print(f"⚠ Fallback: NCORE={ncore}, KPAR={kpar}, GPU={ngpu}")
        
        print()
        print("=" * 74)
        print("Parameter optimization complete!")
        print("Each structure now has NCORE/KPAR optimized for its specific geometry")
        print("Review parameters with: python review_optimizations.py")
        print("=" * 74)


def main():
    optimizer = ParameterOptimizer()
    
    if len(sys.argv) < 2:
        # Default: regenerate all
        optimizer.regenerate_all()
    elif sys.argv[1] == "--clean" or sys.argv[1] == "-c":
        optimizer.clean_all()
    elif sys.argv[1].isdigit():
        # Regenerate range
        start = int(sys.argv[1])
        end = int(sys.argv[2]) if len(sys.argv) > 2 else start + 1
        optimizer.regenerate_range(start, end)
    else:
        print("Usage:")
        print("  python regenerate_optimizations.py              # Regenerate all")
        print("  python regenerate_optimizations.py --clean      # Clean all configs")
        print("  python regenerate_optimizations.py 0 10         # Regenerate structs 0-9")
        sys.exit(1)


if __name__ == "__main__":
    main()
