#!/usr/bin/env python3
"""
Identify structures in an XYZ/extxyz file that are not suitable for nepflow.

This is mainly intended for debugging manually assembled generated-structure
files where ASE/extxyz parsing fails with messages such as:

    Lattice must have 9 values

The script scans the file frame-by-frame, reports the exact structure index and
line range, and validates each structure with ASE independently so one bad frame
does not hide the rest of the file.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

try:
    from ase.io import read as ase_read
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "ASE is required for this utility. Install it in the active environment."
    ) from exc


LATTICE_RE = re.compile(r'Lattice="([^"]*)"')


@dataclass
class FrameRecord:
    index: int
    start_line: int
    end_line: int
    atom_count: int
    header: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find invalid structures in an XYZ/extxyz file and report which frame "
            "breaks ASE/nepflow parsing."
        )
    )
    parser.add_argument("xyz_file", type=Path, help="Path to the XYZ/extxyz file.")
    parser.add_argument(
        "--show-valid",
        action="store_true",
        help="Also print a short line for frames that pass validation.",
    )
    return parser.parse_args()


def read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig").splitlines()


def split_xyz_frames(lines: list[str]) -> list[FrameRecord]:
    frames: list[FrameRecord] = []
    cursor = 0
    frame_index = 0

    while cursor < len(lines):
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor >= len(lines):
            break

        count_line_number = cursor + 1
        raw_count = lines[cursor].strip()
        try:
            atom_count = int(raw_count)
        except ValueError as exc:
            raise ValueError(
                f"Frame {frame_index}: line {count_line_number} is not a valid "
                f"atom-count line: {raw_count!r}"
            ) from exc

        header_index = cursor + 1
        if header_index >= len(lines):
            raise ValueError(
                f"Frame {frame_index}: missing comment/header line after "
                f"atom count on line {count_line_number}"
            )

        end_index = cursor + atom_count + 1
        if end_index >= len(lines):
            raise ValueError(
                f"Frame {frame_index}: expected {atom_count} atom lines after "
                f"line {count_line_number}, but file ended early"
            )

        frame_lines = lines[cursor : end_index + 1]
        frames.append(
            FrameRecord(
                index=frame_index,
                start_line=count_line_number,
                end_line=end_index + 1,
                atom_count=atom_count,
                header=lines[header_index],
                text="\n".join(frame_lines) + "\n",
            )
        )
        frame_index += 1
        cursor = end_index + 1

    return frames


def inspect_lattice(header: str) -> str | None:
    match = LATTICE_RE.search(header)
    if match is None:
        return "missing Lattice field"

    values = [value for value in match.group(1).split() if value]
    if len(values) != 9:
        return f"Lattice has {len(values)} values instead of 9"

    try:
        [float(value) for value in values]
    except ValueError as exc:
        return f"Lattice contains a non-numeric value: {exc}"

    return None


def validate_frame(frame: FrameRecord) -> list[str]:
    problems: list[str] = []

    lattice_problem = inspect_lattice(frame.header)
    if lattice_problem is not None:
        problems.append(lattice_problem)

    try:
        atoms = ase_read(StringIO(frame.text), format="extxyz")
    except Exception as exc:  # pragma: no cover - depends on ASE internals
        problems.append(f"ASE extxyz parse failed: {exc}")
        return problems

    cell = atoms.get_cell().array
    flat_cell = [component for row in cell for component in row]
    if len(flat_cell) != 9:
        problems.append(f"Parsed cell flattened to {len(flat_cell)} values instead of 9")

    return problems


def main() -> int:
    args = parse_args()
    xyz_path = args.xyz_file.resolve()
    if not xyz_path.exists():
        print(f"File not found: {xyz_path}", file=sys.stderr)
        return 2

    try:
        lines = read_lines(xyz_path)
        frames = split_xyz_frames(lines)
    except Exception as exc:
        print(f"Failed to split XYZ file: {exc}", file=sys.stderr)
        return 2

    if not frames:
        print(f"No structures found in {xyz_path}")
        return 1

    invalid_found = False
    print(f"Scanned {len(frames)} structures in {xyz_path}")

    for frame in frames:
        problems = validate_frame(frame)
        if problems:
            invalid_found = True
            print("")
            print(
                f"Structure {frame.index} is not suitable "
                f"(lines {frame.start_line}-{frame.end_line}, atoms={frame.atom_count})"
            )
            print(f"Header: {frame.header}")
            for problem in problems:
                print(f"  - {problem}")
        elif args.show_valid:
            print(
                f"Structure {frame.index} OK "
                f"(lines {frame.start_line}-{frame.end_line}, atoms={frame.atom_count})"
            )

    if invalid_found:
        return 1

    print("All structures passed basic ASE/extxyz and lattice validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
