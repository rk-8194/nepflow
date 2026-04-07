#!/usr/bin/env python3
"""
Analyze LOOP times from VASP OUTCAR files.
Extracts average electronic step time for each structure.
"""

import re
from pathlib import Path
from collections import defaultdict

def extract_loop_times(outcar_path):
    """
    Extract all LOOP times from OUTCAR file.
    Returns list of loop times in seconds (real time).
    """
    loop_times = []
    try:
        with open(outcar_path, 'r') as f:
            for line in f:
                # Match lines like: "      LOOP:  cpu time      1.3783: real time      1.3935"
                match = re.search(r'LOOP:\s+cpu time\s+[\d.]+:\s+real time\s+([\d.]+)', line)
                if match:
                    loop_times.append(float(match.group(1)))
    except (FileNotFoundError, IOError):
        return None
    
    return loop_times if loop_times else None

def main():
    """Scan all struct_XXXX directories and report average LOOP times."""
    base_dir = Path('.')
    struct_dirs = sorted(base_dir.glob('struct_????'))
    
    results = []
    
    for struct_dir in struct_dirs:
        outcar_file = struct_dir / 'OUTCAR'
        
        if not outcar_file.exists():
            continue
        
        loop_times = extract_loop_times(outcar_file)
        
        if loop_times:
            avg_time = sum(loop_times) / len(loop_times)
            total_time = sum(loop_times)
            n_loops = len(loop_times)
            
            results.append({
                'struct': struct_dir.name,
                'n_loops': n_loops,
                'avg_time': avg_time,
                'total_time': total_time,
                'min_time': min(loop_times),
                'max_time': max(loop_times)
            })
    
    # Print results sorted by structure
    print(f"\n{'Structure':<12} {'Loops':<8} {'Avg Time':<12} {'Total Time':<14} {'Min':<10} {'Max':<10}")
    print("=" * 80)
    
    for result in results:
        print(f"{result['struct']:<12} {result['n_loops']:<8} "
              f"{result['avg_time']:<12.4f} {result['total_time']:<14.2f} "
              f"{result['min_time']:<10.4f} {result['max_time']:<10.4f}")
    
    # Summary statistics
    if results:
        avg_times = [r['avg_time'] for r in results]
        print("\n" + "=" * 80)
        print(f"Statistics across {len(results)} completed structures:")
        print(f"  Fastest avg LOOP time: {min(avg_times):.4f} s ({results[avg_times.index(min(avg_times))]['struct']})")
        print(f"  Slowest avg LOOP time: {max(avg_times):.4f} s ({results[avg_times.index(max(avg_times))]['struct']})")
        print(f"  Overall average: {sum(avg_times) / len(avg_times):.4f} s")
        print(f"  Median average: {sorted(avg_times)[len(avg_times)//2]:.4f} s")

if __name__ == '__main__':
    main()
