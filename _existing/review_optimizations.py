#!/usr/bin/env python3
"""
Review optimized VASP parameters across all structures.

Shows distribution of NCORE, KPAR, and GPU configurations.
Identifies memory constraints and parallelization efficiency.

Usage:
    python review_optimizations.py              # Review all structures
    python review_optimizations.py 0 50         # Review structs 0-49
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

try:
    from ase.io import read as ase_read
    from ase.symmetry.kpoints import monkhorst_pack, get_ibz_vertices
    HAS_ASE = True
except ImportError:
    HAS_ASE = False


class ParameterReviewer:
    """Review optimized parameters across structures."""
    
    def __init__(self):
        self.workspace = Path.cwd()
    
    def get_irreducible_kpoints(self, struct_dir):
        """Get number of irreducible k-points from KPOINTS file or ASE."""
        kpoints_file = struct_dir / "KPOINTS"
        
        if not kpoints_file.exists():
            return None
        
        try:
            with open(kpoints_file, 'r', encoding='utf-8') as f:
                kpts_lines = f.readlines()
            
            # Read the k-point grid
            if len(kpts_lines) > 3:
                kpt_grid = kpts_lines[3].split()
                if len(kpt_grid) >= 3:
                    nx = int(kpt_grid[0])
                    ny = int(kpt_grid[1])
                    nz = int(kpt_grid[2])
                    
                    # Use ASE if available to get irreducible k-points
                    if HAS_ASE:
                        try:
                            poscar_file = struct_dir / "POSCAR"
                            atoms = ase_read(str(poscar_file))
                            kpts = monkhorst_pack((nx, ny, nz))
                            ibz_vertices = get_ibz_vertices(atoms, kpts)
                            return len(ibz_vertices)
                        except (IOError, OSError, ValueError, AttributeError):
                            pass
                    
                    # Fallback: estimate ~50% reduction
                    return int((nx * ny * nz) * 0.5)
        except (IOError, OSError, ValueError, IndexError):
            pass
        
        return None
    
    def get_atom_count(self, struct_dir):
        """Get number of atoms from POSCAR."""
        poscar_file = struct_dir / "POSCAR"
        
        if not poscar_file.exists():
            return None
        
        try:
            with open(poscar_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # POSCAR format: line 5 or 6 has atom counts
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
            
            return n_atoms
        except (IOError, OSError, ValueError, IndexError):
            pass
        
        return None
    
    def calculate_parallelization_score(self, ncore, kpar, ngpu):
        """Calculate parallelization effectiveness score.
        
        Higher score = better parallelization:
        - Lower NCORE: band parallelism (8 < 16 < 32 < 64)
        - Higher KPAR: k-point parallelism (4 > 2 > 1)
        - More GPUs: distributed computation (4 > 2 > 1)
        """
        # Band parallelism: reciprocal of NCORE (lower is better)
        band_score = 1000.0 / ncore if ncore > 0 else 0
        
        # K-point parallelism: direct (higher is better)
        kpar_score = kpar * 100
        
        # GPU parallelism: direct (more is better)
        gpu_score = ngpu * 50
        
        total_score = band_score + kpar_score + gpu_score
        return round(total_score, 1)

        
    def get_incar_params(self, struct_dir):
        """Read NCORE and KPAR directly from INCAR file."""
        incar_file = struct_dir / "INCAR"
        
        if not incar_file.exists():
            return None, None
        
        try:
            with open(incar_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            ncore = None
            kpar = None
            
            for line in lines:
                if line.strip().upper().startswith('NCORE'):
                    parts = line.split('=')
                    if len(parts) > 1:
                        try:
                            ncore = int(parts[1].split()[0])
                        except ValueError:
                            pass
                elif line.strip().upper().startswith('KPAR'):
                    parts = line.split('=')
                    if len(parts) > 1:
                        try:
                            kpar = int(parts[1].split()[0])
                        except ValueError:
                            pass
            
            return ncore, kpar
        except (IOError, OSError):
            return None, None
    
    def find_structures(self):
        """Find all struct_* directories."""
        structs = sorted([d for d in self.workspace.glob("struct_*") if d.is_dir()])
        return structs
    
    def review_all(self):
        """Review all structures."""
        structs = self.find_structures()
        
        if not structs:
            print("ERROR: No struct_* directories found")
            return
        
        print("=" * 80)
        print("Parameter Optimization Review - All Structures")
        print("=" * 80)
        print()
        
        # Collect statistics
        ncore_dist = defaultdict(int)
        kpar_dist = defaultdict(int)
        ngpu_dist = defaultdict(int)
        memory_dist = defaultdict(list)
        
        configs = []
        
        for struct_dir in structs:
            config_file = struct_dir / "optimization_config.json"
            
            if not config_file.exists():
                continue
            
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            except (IOError, OSError, json.JSONDecodeError):
                continue
            
            struct_num = struct_dir.name
            ncore = config.get('preset_ncore', 'N/A')
            kpar = config.get('preset_kpar', 'N/A')
            ngpu = config.get('ngpu', 'N/A')
            mem_per_gpu = config.get('estimated_memory_per_gpu_gb', 999)
            
            # If not in config, try reading from INCAR
            if ncore == 'N/A' or kpar == 'N/A':
                incar_ncore, incar_kpar = self.get_incar_params(struct_dir)
                if incar_ncore is not None:
                    ncore = incar_ncore
                if incar_kpar is not None:
                    kpar = incar_kpar
            
            # Get additional info
            n_atoms = self.get_atom_count(struct_dir)
            n_kpoints = self.get_irreducible_kpoints(struct_dir)
            
            # Calculate parallelization score if we have valid parameters
            para_score = None
            if isinstance(ncore, int) and isinstance(kpar, int) and isinstance(ngpu, int):
                para_score = self.calculate_parallelization_score(ncore, kpar, ngpu)
            
            configs.append({
                'struct': struct_num,
                'ncore': ncore,
                'kpar': kpar,
                'ngpu': ngpu,
                'memory_per_gpu': mem_per_gpu,
                'n_atoms': n_atoms,
                'n_kpoints': n_kpoints,
                'para_score': para_score
            })
            
            if isinstance(ncore, int):
                ncore_dist[ncore] += 1
            if isinstance(kpar, int):
                kpar_dist[kpar] += 1
            if isinstance(ngpu, int):
                ngpu_dist[ngpu] += 1
                if mem_per_gpu != 999 and mem_per_gpu is not None:
                    memory_dist[ngpu].append(mem_per_gpu)
        
        if not configs:
            print("No optimization configs found. Run: python regenerate_optimizations.py")
            return
        
        # Display statistics
        print(f"Reviewed {len(configs)} optimized structures\n")
        
        print("NCORE Distribution:")
        print("-" * 40)
        for ncore in sorted(ncore_dist.keys()):
            count = ncore_dist[ncore]
            pct = 100 * count / len(configs)
            bar = "█" * (count // 2)
            ncore_str = str(ncore) if isinstance(ncore, int) else ncore
            print(f"  NCORE={ncore_str:>2}: {count:3d} structs ({pct:5.1f}%) {bar}")
        print()
        
        print("KPAR Distribution:")
        print("-" * 40)
        for kpar in sorted(kpar_dist.keys()):
            count = kpar_dist[kpar]
            pct = 100 * count / len(configs)
            bar = "█" * (count // 2)
            kpar_str = str(kpar) if isinstance(kpar, int) else kpar
            print(f"  KPAR={kpar_str}: {count:3d} structs ({pct:5.1f}%) {bar}")
        print()
        
        print("GPU Distribution:")
        print("-" * 40)
        for ngpu in sorted(ngpu_dist.keys()):
            count = ngpu_dist[ngpu]
            pct = 100 * count / len(configs)
            mem_list = [m for m in memory_dist.get(ngpu, []) if m is not None and m != 999]
            avg_mem = sum(mem_list) / len(mem_list) if mem_list else 0
            max_mem = max(mem_list) if mem_list else 0
            bar = "█" * (count // 2)
            print(f"  GPU={ngpu}: {count:3d} structs ({pct:5.1f}%) {bar}")
            if mem_list:
                print(f"           Avg memory/GPU: {avg_mem:6.2f} GB, Max: {max_mem:6.2f} GB")
        print()
        
        # Find potential bottlenecks
        high_memory = [c for c in configs if c['memory_per_gpu'] is not None and c['memory_per_gpu'] != 999 and c['memory_per_gpu'] > 50]
        if high_memory:
            print("⚠  HIGH MEMORY USAGE (>50 GB/GPU):")
            print("-" * 40)
            for config in sorted(high_memory, key=lambda x: x['memory_per_gpu'], reverse=True)[:10]:
                print(f"  {config['struct']}: {config['memory_per_gpu']:.2f} GB/GPU "
                      f"(NCORE={config['ncore']}, KPAR={config['kpar']}, GPU={config['ngpu']})")
            print()
        
        # Show parameter combinations
        print("Parameter Combinations:")
        print("-" * 40)
        combos = defaultdict(int)
        for config in configs:
            key = (config['ncore'], config['kpar'], config['ngpu'])
            combos[key] += 1
        
        for (ncore, kpar, ngpu), count in sorted(combos.items(), key=lambda x: x[1], reverse=True):
            pct = 100 * count / len(configs)
            ncore_str = str(ncore) if isinstance(ncore, int) else ncore
            kpar_str = str(kpar) if isinstance(kpar, int) else kpar
            ngpu_str = str(ngpu) if isinstance(ngpu, int) else ngpu
            print(f"  NCORE={ncore_str:>2}, KPAR={kpar_str}, GPU={ngpu_str}: {count:3d} structs ({pct:5.1f}%)")
        print()
        
        # Summary statistics
        print("Summary:")
        print("-" * 40)
        all_mems = [c['memory_per_gpu'] for c in configs if c['memory_per_gpu'] is not None and c['memory_per_gpu'] != 999]
        if all_mems:
            print(f"  Min memory/GPU:  {min(all_mems):6.2f} GB")
            print(f"  Avg memory/GPU:  {sum(all_mems)/len(all_mems):6.2f} GB")
            print(f"  Max memory/GPU:  {max(all_mems):6.2f} GB")
            print(f"  Median memory/GPU: {sorted(all_mems)[len(all_mems)//2]:6.2f} GB")
        print()
        
        # Parallelization statistics
        print("Parallelization Statistics:")
        print("-" * 40)
        valid_para_scores = [c['para_score'] for c in configs if c['para_score'] is not None]
        if valid_para_scores:
            avg_para = sum(valid_para_scores) / len(valid_para_scores)
            min_para = min(valid_para_scores)
            max_para = max(valid_para_scores)
            print(f"  Parallelization Score Range: {min_para:.1f} - {max_para:.1f}")
            print(f"  Average Parallelization Score: {avg_para:.1f}")
        
        # Show k-point statistics
        kpoint_counts = [c['n_kpoints'] for c in configs if c['n_kpoints'] is not None]
        atom_counts = [c['n_atoms'] for c in configs if c['n_atoms'] is not None]
        if kpoint_counts:
            print(f"  K-point range: {min(kpoint_counts)} - {max(kpoint_counts)}")
            print(f"  Atom count range: {min(atom_counts)} - {max(atom_counts)}")
        print()
        
        print("=" * 80)
    
    def review_range(self, start, end):
        """Review specific range of structures."""
        structs = self.find_structures()
        
        if not structs:
            print("ERROR: No struct_* directories found")
            return
        
        # Filter to range
        structs_in_range = [s for s in structs if start <= int(s.name.split('_')[1]) < end]
        
        if not structs_in_range:
            print(f"No structures found in range {start}-{end}")
            return
        
        print("=" * 80)
        print(f"Parameter Optimization Review - Structs {start}-{end-1}")
        print("=" * 80)
        print()
        
        configs = []
        ncore_dist = defaultdict(int)
        kpar_dist = defaultdict(int)
        ngpu_dist = defaultdict(int)
        
        for struct_dir in structs_in_range:
            config_file = struct_dir / "optimization_config.json"
            
            if not config_file.exists():
                continue
            
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            except (IOError, OSError, json.JSONDecodeError):
                continue
            
            struct_num = struct_dir.name
            ncore = config.get('preset_ncore', 'N/A')
            kpar = config.get('preset_kpar', 'N/A')
            ngpu = config.get('ngpu', 'N/A')
            mem_per_gpu = config.get('estimated_memory_per_gpu_gb', 999)
            
            # If not in config, try reading from INCAR
            if ncore == 'N/A' or kpar == 'N/A':
                incar_ncore, incar_kpar = self.get_incar_params(struct_dir)
                if incar_ncore is not None:
                    ncore = incar_ncore
                if incar_kpar is not None:
                    kpar = incar_kpar
            
            # Get additional info
            n_atoms = self.get_atom_count(struct_dir)
            n_kpoints = self.get_irreducible_kpoints(struct_dir)
            
            # Calculate parallelization score
            para_score = None
            if isinstance(ncore, int) and isinstance(kpar, int) and isinstance(ngpu, int):
                para_score = self.calculate_parallelization_score(ncore, kpar, ngpu)
            
            configs.append({
                'struct': struct_num,
                'ncore': ncore,
                'kpar': kpar,
                'ngpu': ngpu,
                'memory_per_gpu': mem_per_gpu,
                'n_atoms': n_atoms,
                'n_kpoints': n_kpoints,
                'para_score': para_score
            })
            
            if isinstance(ncore, int):
                ncore_dist[ncore] += 1
            if isinstance(kpar, int):
                kpar_dist[kpar] += 1
            if isinstance(ngpu, int):
                ngpu_dist[ngpu] += 1
        
        if not configs:
            print("No optimization configs found in this range.")
            return
        
        # Detailed listing
        print(f"Reviewed {len(configs)} structures\n")
        print("Structure Details:")
        print("-" * 100)
        print(f"{'Struct':<12} {'Atoms':<6} {'K-pts':<8} {'NCORE':<8} {'KPAR':<6} {'GPU':<5} {'Para.Score':<12} {'Mem/GPU':<10}")
        print("-" * 100)
        
        for config in sorted(configs, key=lambda x: x['struct']):
            mem_str = f"{config['memory_per_gpu']:.2f} GB" if config['memory_per_gpu'] != 999 else "N/A"
            para_str = f"{config['para_score']:.1f}" if config['para_score'] is not None else "N/A"
            atoms_str = str(config['n_atoms']) if config['n_atoms'] is not None else "N/A"
            kpts_str = str(config['n_kpoints']) if config['n_kpoints'] is not None else "N/A"
            print(f"{config['struct']:<12} {atoms_str:<6} {kpts_str:<8} {config['ncore']:<8} {config['kpar']:<6} "
                  f"{config['ngpu']:<5} {para_str:<12} {mem_str:<10}")
        
        print()
        print("=" * 80)


def main():
    """Main entry point."""
    reviewer = ParameterReviewer()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print(__doc__)
            return
        
        try:
            start = int(sys.argv[1])
            end = int(sys.argv[2]) if len(sys.argv) > 2 else start + 1
            reviewer.review_range(start, end)
        except ValueError:
            print(f"Error: Invalid arguments. Use: python {sys.argv[0]} [start] [end]")
            sys.exit(1)
    else:
        reviewer.review_all()


if __name__ == "__main__":
    main()
