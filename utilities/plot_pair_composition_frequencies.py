#!/usr/bin/env python3
"""
Compute per-structure compositions from a selected train.xyz file and plot
pairwise composition frequencies.

For each unique unordered element pair A-B found in the dataset, this script
creates a scatter plot where:
  - x = fraction of element B in the structure
  - y = number of matching structures with that B fraction
  - color = number of distinct elements in the structure

Outputs:
  - composition_summary.csv
  - pair_frequency_summary.csv
  - one PNG per unique unordered element pair

Usage:
    python utilities/plot_pair_composition_frequencies.py --project PATH
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from ase.io import read as ase_read

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("plot_pair_composition_frequencies")


@dataclass(frozen=True)
class StructureComposition:
    structure_index: int
    formula: str
    total_atoms: int
    unique_elements: tuple[str, ...]
    element_counts: dict[str, int]
    element_fractions: dict[str, float]


@dataclass(frozen=True)
class PairFrequencyPoint:
    pair: tuple[str, str]
    b_fraction: float
    frequency: int
    distinct_elements: int


def _load_structures(train_xyz: Path) -> list:
    structures = ase_read(str(train_xyz), index=":", format="extxyz")
    if isinstance(structures, list):
        return structures
    return [structures]


def _composition_from_atoms(atoms, structure_index: int) -> StructureComposition:
    counts = Counter(atoms.get_chemical_symbols())
    total_atoms = sum(counts.values())
    if total_atoms == 0:
        raise ValueError(f"Structure {structure_index} contains no atoms")

    ordered_elements = tuple(sorted(counts))
    fractions = {
        element: counts[element] / total_atoms
        for element in ordered_elements
    }
    return StructureComposition(
        structure_index=structure_index,
        formula=atoms.get_chemical_formula(),
        total_atoms=total_atoms,
        unique_elements=ordered_elements,
        element_counts=dict(counts),
        element_fractions=fractions,
    )


def _fraction_label(value: float) -> str:
    fraction = Fraction(value).limit_denominator()
    if math.isclose(float(fraction), value, rel_tol=0.0, abs_tol=1e-12):
        return f"{fraction.numerator}/{fraction.denominator}"
    return f"{value:.6f}"


def _collect_pair_points(
    compositions: list[StructureComposition],
) -> dict[tuple[str, str], list[PairFrequencyPoint]]:
    pair_fraction_counts: dict[tuple[str, str], Counter[float]] = defaultdict(Counter)
    pair_fraction_distinct: dict[tuple[str, str], dict[float, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )

    for composition in compositions:
        distinct_count = len(composition.unique_elements)
        for i, element_a in enumerate(composition.unique_elements):
            for element_b in composition.unique_elements[i + 1:]:
                pair = (element_a, element_b)
                b_fraction = composition.element_fractions[element_b]
                pair_fraction_counts[pair][b_fraction] += 1
                pair_fraction_distinct[pair][b_fraction].add(distinct_count)

    pair_points: dict[tuple[str, str], list[PairFrequencyPoint]] = {}
    for pair, counts in pair_fraction_counts.items():
        points: list[PairFrequencyPoint] = []
        for b_fraction, frequency in sorted(counts.items(), key=lambda item: item[0]):
            for distinct_elements in sorted(pair_fraction_distinct[pair][b_fraction]):
                points.append(
                    PairFrequencyPoint(
                        pair=pair,
                        b_fraction=b_fraction,
                        frequency=frequency,
                        distinct_elements=distinct_elements,
                    )
                )
        pair_points[pair] = points
    return pair_points


def _write_composition_summary(
    compositions: list[StructureComposition],
    output_path: Path,
) -> None:
    all_elements = sorted({
        element
        for composition in compositions
        for element in composition.unique_elements
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "structure_index",
            "formula",
            "total_atoms",
            "distinct_elements",
            "elements",
        ]
        for element in all_elements:
            fieldnames.append(f"count_{element}")
            fieldnames.append(f"fraction_{element}")

        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for composition in compositions:
            row = {
                "structure_index": composition.structure_index,
                "formula": composition.formula,
                "total_atoms": composition.total_atoms,
                "distinct_elements": len(composition.unique_elements),
                "elements": ",".join(composition.unique_elements),
            }
            for element in all_elements:
                row[f"count_{element}"] = composition.element_counts.get(element, 0)
                row[f"fraction_{element}"] = f"{composition.element_fractions.get(element, 0.0):.12f}"
            writer.writerow(row)


def _write_pair_summary(
    pair_points: dict[tuple[str, str], list[PairFrequencyPoint]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pair",
                "element_a",
                "element_b",
                "b_fraction",
                "b_fraction_label",
                "frequency",
                "distinct_elements",
            ],
        )
        writer.writeheader()
        for pair, points in sorted(pair_points.items()):
            for point in points:
                writer.writerow(
                    {
                        "pair": f"{pair[0]}-{pair[1]}",
                        "element_a": pair[0],
                        "element_b": pair[1],
                        "b_fraction": f"{point.b_fraction:.12f}",
                        "b_fraction_label": _fraction_label(point.b_fraction),
                        "frequency": point.frequency,
                        "distinct_elements": point.distinct_elements,
                    }
                )


def _plot_pair(
    pair: tuple[str, str],
    points: list[PairFrequencyPoint],
    output_path: Path,
    title_prefix: str,
    show: bool,
) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm

    xs = [point.b_fraction for point in points]
    ys = [point.frequency for point in points]
    colors = [point.distinct_elements for point in points]
    unique_distinct_counts = sorted(set(colors))
    boundaries = [count - 0.5 for count in unique_distinct_counts]
    boundaries.append(unique_distinct_counts[-1] + 0.5)
    cmap = plt.get_cmap("viridis", len(unique_distinct_counts))
    norm = BoundaryNorm(boundaries, cmap.N)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    scatter = ax.scatter(
        xs,
        ys,
        c=colors,
        cmap=cmap,
        norm=norm,
        s=36,
        edgecolors="black",
        linewidths=0.4,
        alpha=0.9,
    )

    ax.set_title(f"{title_prefix}: {pair[0]}-{pair[1]}")
    ax.set_xlabel(f"Fraction of {pair[1]}")
    ax.set_ylabel("Structure frequency")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([i / 10 for i in range(11)])

    colorbar = fig.colorbar(scatter, ax=ax, ticks=unique_distinct_counts)
    colorbar.set_label("Number of distinct elements in structure")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-structure compositions from structures/selected/train.xyz "
            "and plot unordered pair composition frequencies."
        )
    )
    parser.add_argument(
        "--project",
        type=Path,
        required=True,
        help="Path to the nepflow project directory",
    )
    parser.add_argument(
        "--train-xyz",
        type=Path,
        default=None,
        help="Path to the selected training XYZ file (default: PROJECT/structures/selected/train.xyz)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for CSV summaries and plots (default: PROJECT/reports/pair_composition_frequencies)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively after saving them",
    )
    args = parser.parse_args()

    project_dir = args.project.resolve()
    if not project_dir.exists():
        logger.error(f"Project directory not found: {project_dir}")
        sys.exit(1)

    train_xyz = args.train_xyz or (project_dir / "structures" / "selected" / "train.xyz")
    if not train_xyz.exists():
        logger.error(f"Training XYZ not found: {train_xyz}")
        sys.exit(1)

    output_dir = args.output_dir or (project_dir / "reports" / "pair_composition_frequencies")
    plots_dir = output_dir / "plots"

    logger.info(f"Project directory: {project_dir}")
    logger.info(f"Reading structures from: {train_xyz}")

    structures = _load_structures(train_xyz)
    if not structures:
        logger.error("No structures found in the training XYZ")
        sys.exit(1)

    compositions = [
        _composition_from_atoms(atoms, structure_index=index)
        for index, atoms in enumerate(structures)
    ]
    logger.info(f"Loaded {len(compositions)} structures")

    pair_points = _collect_pair_points(compositions)
    if not pair_points:
        logger.error("No structures containing at least one unordered element pair were found")
        sys.exit(1)

    composition_csv = output_dir / "composition_summary.csv"
    pair_csv = output_dir / "pair_frequency_summary.csv"
    _write_composition_summary(compositions, composition_csv)
    _write_pair_summary(pair_points, pair_csv)

    logger.info(f"Wrote composition summary: {composition_csv}")
    logger.info(f"Wrote pair summary: {pair_csv}")

    for pair, points in sorted(pair_points.items()):
        plot_path = plots_dir / f"{pair[0]}_{pair[1]}.png"
        _plot_pair(
            pair,
            points,
            plot_path,
            title_prefix="Pairwise composition frequency",
            show=args.show,
        )
        logger.info(f"Saved plot: {plot_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
