#!/usr/bin/env python3
"""
Recursively search for OUTCAR files and populate nepflow's .vasp_memory CSV.

Extracts VASP simulation parameters and average electronic loop times from
completed OUTCAR files, writing results to <nepflow_root>/.vasp_memory.
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Optional

VASP_COMPLETION_MARKERS = ["General timing", "Voluntary context switches"]

CSV_HEADER = [
    "n_atoms", "n_kpoints_irr", "n_electrons",
    "nodes", "gpus", "ncore", "kpar", "avg_loop_time", "oom",
]


def is_completed(outcar_text: str) -> bool:
    """Check if OUTCAR indicates a successfully completed VASP run."""
    tail = outcar_text[-2000:]
    return any(marker in tail for marker in VASP_COMPLETION_MARKERS)


def parse_outcar(outcar_path: Path, gpus_per_node: int) -> Optional[dict]:
    """Extract performance data from an OUTCAR and its sibling POSCAR/INCAR."""
    struct_dir = outcar_path.parent
    poscar = struct_dir / "POSCAR"
    incar = struct_dir / "INCAR"

    if not poscar.exists() or not incar.exists():
        missing = [f for f, p in (("POSCAR", poscar), ("INCAR", incar)) if not p.exists()]
        print(f"  SKIP {outcar_path.parent}: missing {', '.join(missing)}")
        return None

    try:
        outcar_text = outcar_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"  SKIP {outcar_path.parent}: cannot read OUTCAR: {e}")
        return None

    if not is_completed(outcar_text):
        print(f"  SKIP {outcar_path.parent}: OUTCAR not completed")
        return None

    try:
        # n_atoms from POSCAR line 7
        poscar_lines = poscar.read_text(encoding="utf-8").splitlines()
        n_atoms = sum(int(x) for x in poscar_lines[6].split())

        # NCORE / KPAR from INCAR
        incar_text = incar.read_text(encoding="utf-8")
        ncore = kpar = 0
        for line in incar_text.splitlines():
            stripped = line.strip().upper()
            if stripped.startswith("NCORE"):
                ncore = int(line.split("=")[1].split("#")[0].strip())
            elif stripped.startswith("KPAR"):
                kpar = int(line.split("=")[1].split("#")[0].strip())

        # Parse MPI ranks from OUTCAR to determine nodes/gpus
        total_ranks = 0
        m_ranks = re.search(r"running on\s+(\d+)\s+total cores", outcar_text)
        if m_ranks:
            total_ranks = int(m_ranks.group(1))
        nodes = max(1, total_ranks // gpus_per_node) if total_ranks > 0 else 1
        gpus = total_ranks if total_ranks > 0 else gpus_per_node

        # LOOP times
        loop_times = [
            float(m.group(1))
            for m in re.finditer(
                r"LOOP:\s+cpu time\s+[\d.]+:\s+real time\s+([\d.]+)", outcar_text
            )
        ]
        if not loop_times:
            print(f"  SKIP {outcar_path.parent}: no LOOP times found")
            return None
        avg_loop = sum(loop_times) / len(loop_times)

        # Irreducible k-points
        n_kpoints_irr = 0
        m = re.search(r"Found\s+(\d+)\s+irreducible k-points", outcar_text)
        if m:
            n_kpoints_irr = int(m.group(1))

        # Number of electrons
        n_electrons = 0
        m = re.search(r"NELECT\s*=\s*([\d.]+)", outcar_text)
        if m:
            n_electrons = int(float(m.group(1)))

        result = {
            "n_atoms": n_atoms,
            "n_kpoints_irr": n_kpoints_irr,
            "n_electrons": n_electrons,
            "nodes": nodes,
            "gpus": gpus,
            "ncore": ncore,
            "kpar": kpar,
            "avg_loop_time": f"{avg_loop:.4f}",
            "oom": 0,
        }
        print(f"    OK {outcar_path.parent.name}: "
              f"atoms={n_atoms} kpts={n_kpoints_irr} nel={n_electrons} "
              f"ncore={ncore} kpar={kpar} avg_loop={avg_loop:.4f}s")
        return result

    except (ValueError, IndexError, OSError) as e:
        print(f"  Warning: could not parse {struct_dir.name}: {e}", file=sys.stderr)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Populate .vasp_memory from existing OUTCAR files."
    )
    parser.add_argument(
        "search_dir",
        type=Path,
        help="Root directory to recursively search for OUTCAR files.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <nepflow>/.vasp_memory (next to this script).",
    )
    parser.add_argument(
        "--gpus-per-node", type=int, default=4,
        help="GPUs per node on target HPC (default: 4).",
    )
    args = parser.parse_args()

    search_dir: Path = args.search_dir.resolve()
    nepflow_root = Path(__file__).resolve().parent.parent
    output_path: Path = (args.output or nepflow_root / ".vasp_memory").resolve()

    if not search_dir.is_dir():
        sys.exit(f"Error: {search_dir} is not a directory.")

    outcars = sorted(search_dir.rglob("OUTCAR"))
    print(f"Found {len(outcars)} OUTCAR file(s) under {search_dir}")

    rows: list[dict] = []
    for outcar in outcars:
        result = parse_outcar(outcar, args.gpus_per_node)
        if result is not None:
            rows.append(result)

    if not rows:
        print("No completed VASP runs found.")
        return

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for row in rows:
            writer.writerow([row[col] for col in CSV_HEADER])

    print(f"Wrote {len(rows)} entries to {output_path}")


if __name__ == "__main__":
    main()
