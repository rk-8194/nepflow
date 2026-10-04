#!/usr/bin/env python3
"""
Recursively search for OUTCAR files and populate nepflow's .vasp_memory CSV.

Extracts VASP simulation parameters and average electronic loop times from
completed OUTCAR files, writing results to <nepflow_root>/.vasp_memory.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

from nepflow.dft.vasp.outputs import (
    parse_memory_record,
)

CSV_HEADER = [
    "n_atoms",
    "n_kpoints_irr",
    "n_electrons",
    "nodes",
    "gpus",
    "ncore",
    "kpar",
    "avg_loop_time",
    "oom",
]


def parse_outcar(outcar_path: Path, gpus_per_node: int) -> Optional[dict]:
    """Extract a legacy benchmark row using the canonical VASP parser."""
    result = parse_memory_record(outcar_path, gpus_per_node)
    if result is None:
        print(f"  SKIP {outcar_path.parent}: incomplete or absent OUTCAR")
        return None
    print(
        f"    OK {outcar_path.parent.name}: "
        f"atoms={result['n_atoms']} kpts={result['n_kpoints_irr']} "
        f"nel={result['n_electrons']} ncore={result['ncore']} "
        f"kpar={result['kpar']} avg_loop={result['avg_loop_time']}s"
    )
    return result


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
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <nepflow>/.vasp_memory (next to this script).",
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=4,
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
