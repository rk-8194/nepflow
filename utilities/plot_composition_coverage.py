#!/usr/bin/env python3
"""
Analyze composition-space coverage for selected training structures.

This utility replaces pair-frequency scatter plots with coverage-oriented
projections and metrics:
  - binary subsets use binned 1D histograms
  - ternary subsets use simplex density plots
  - higher-order structures contribute through binary / ternary projections

Outputs:
  - composition_summary.csv
  - coverage_summary.csv and/or coverage_summary.json
  - binary and ternary projection plots

Usage:
    python utilities/plot_composition_coverage.py --project PATH
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("plot_composition_coverage")


SQRT3_OVER_2 = math.sqrt(3.0) / 2.0


@dataclass(frozen=True)
class StructureComposition:
    structure_index: int
    formula: str
    total_atoms: int
    unique_elements: tuple[str, ...]
    element_counts: dict[str, int]
    element_fractions: dict[str, float]


@dataclass(frozen=True)
class BinaryProjection:
    subset: tuple[str, str]
    structure_index: int
    normalized_fraction_b: float


@dataclass(frozen=True)
class TernaryProjection:
    subset: tuple[str, str, str]
    structure_index: int
    barycentric: tuple[float, float, float]


@dataclass(frozen=True)
class CoverageSummary:
    subset_label: str
    subset_size: int
    dimensions: int
    structure_count: int
    total_bins: int
    occupied_bins: int
    occupied_bin_fraction: float
    normalized_entropy: float
    max_bin_fraction: float
    gini: float
    nn_distance_mean: float | None
    nn_distance_p95: float | None


def _load_structures(train_xyz: Path) -> list:
    try:
        from ase.io import read as ase_read
    except ImportError as exc:
        raise ImportError(
            "ASE is required to read extxyz files. Install it in the active environment "
            "before running plot_composition_coverage.py."
        ) from exc

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


def _normalize_subset_fractions(
    composition: StructureComposition,
    subset: tuple[str, ...],
) -> tuple[float, ...]:
    subset_total = sum(composition.element_fractions[element] for element in subset)
    if subset_total <= 0:
        raise ValueError(f"Subset {subset} has zero total mass in structure {composition.structure_index}")
    return tuple(composition.element_fractions[element] / subset_total for element in subset)


def _collect_binary_projections(
    compositions: list[StructureComposition],
) -> dict[tuple[str, str], list[BinaryProjection]]:
    projections: dict[tuple[str, str], list[BinaryProjection]] = defaultdict(list)
    for composition in compositions:
        for subset in combinations(composition.unique_elements, 2):
            normalized = _normalize_subset_fractions(composition, subset)
            projections[subset].append(
                BinaryProjection(
                    subset=subset,
                    structure_index=composition.structure_index,
                    normalized_fraction_b=normalized[1],
                )
            )
    return dict(projections)


def _collect_ternary_projections(
    compositions: list[StructureComposition],
) -> dict[tuple[str, str, str], list[TernaryProjection]]:
    projections: dict[tuple[str, str, str], list[TernaryProjection]] = defaultdict(list)
    for composition in compositions:
        for subset in combinations(composition.unique_elements, 3):
            projections[subset].append(
                TernaryProjection(
                    subset=subset,
                    structure_index=composition.structure_index,
                    barycentric=_normalize_subset_fractions(composition, subset),
                )
            )
    return dict(projections)


def _binary_bin_index(value: float, bins: int) -> int:
    clamped = min(max(value, 0.0), 1.0)
    if math.isclose(clamped, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return bins - 1
    return min(int(clamped * bins), bins - 1)


def _largest_remainder_integer_partition(values: tuple[float, ...], total: int) -> tuple[int, ...]:
    scaled = [value * total for value in values]
    floors = [math.floor(value) for value in scaled]
    remainder = total - sum(floors)
    if remainder > 0:
        ranked = sorted(
            enumerate(scaled),
            key=lambda item: (item[1] - floors[item[0]], -item[0]),
            reverse=True,
        )
        for idx, _ in ranked[:remainder]:
            floors[idx] += 1
    return tuple(int(value) for value in floors)


def _ternary_bin_index(
    barycentric: tuple[float, float, float],
    resolution: int,
) -> tuple[int, int, int]:
    counts = _largest_remainder_integer_partition(barycentric, resolution)
    return counts


def _ternary_bin_center(
    index: tuple[int, int, int],
    resolution: int,
) -> tuple[float, float, float]:
    return tuple(value / resolution for value in index)


def _barycentric_to_cartesian(barycentric: tuple[float, float, float]) -> tuple[float, float]:
    a, b, c = barycentric
    _ = a
    x = b + 0.5 * c
    y = c * SQRT3_OVER_2
    return x, y


def _occupied_bin_fraction(counts: list[int], total_bins: int) -> tuple[int, float]:
    occupied = sum(1 for count in counts if count > 0)
    fraction = occupied / total_bins if total_bins > 0 else 0.0
    return occupied, fraction


def _normalized_entropy(counts: list[int], total_bins: int) -> float:
    total = sum(counts)
    if total <= 0 or total_bins <= 1:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy / math.log(total_bins)


def _max_bin_fraction(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    return max(counts) / total


def _gini(values: list[int]) -> float:
    if not values:
        return 0.0
    total = sum(values)
    if total <= 0:
        return 0.0
    sorted_values = sorted(values)
    n = len(sorted_values)
    weighted_sum = sum((2 * idx - n - 1) * value for idx, value in enumerate(sorted_values, start=1))
    return weighted_sum / (n * total)


def _nearest_neighbor_distances(points: list[tuple[float, ...]]) -> list[float]:
    if len(points) < 2:
        return []
    distances: list[float] = []
    for idx, point in enumerate(points):
        best = math.inf
        for other_idx, other in enumerate(points):
            if idx == other_idx:
                continue
            squared = sum((a - b) ** 2 for a, b in zip(point, other))
            best = min(best, math.sqrt(squared))
        distances.append(best)
    return distances


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = 0.95 * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summarize_binary_subset(
    subset: tuple[str, str],
    projections: list[BinaryProjection],
    bins: int,
) -> tuple[CoverageSummary, list[int]]:
    bin_counts = [0] * bins
    coordinates: list[tuple[float]] = []
    for projection in projections:
        bin_idx = _binary_bin_index(projection.normalized_fraction_b, bins)
        bin_counts[bin_idx] += 1
        coordinates.append((projection.normalized_fraction_b,))

    occupied_bins, occupied_fraction = _occupied_bin_fraction(bin_counts, bins)
    nn_distances = _nearest_neighbor_distances(coordinates)
    summary = CoverageSummary(
        subset_label="-".join(subset),
        subset_size=2,
        dimensions=1,
        structure_count=len(projections),
        total_bins=bins,
        occupied_bins=occupied_bins,
        occupied_bin_fraction=occupied_fraction,
        normalized_entropy=_normalized_entropy(bin_counts, bins),
        max_bin_fraction=_max_bin_fraction(bin_counts),
        gini=_gini(bin_counts),
        nn_distance_mean=statistics.fmean(nn_distances) if nn_distances else None,
        nn_distance_p95=_p95(nn_distances),
    )
    return summary, bin_counts


def _summarize_ternary_subset(
    subset: tuple[str, str, str],
    projections: list[TernaryProjection],
    resolution: int,
) -> tuple[CoverageSummary, dict[tuple[int, int, int], int]]:
    bin_counts: dict[tuple[int, int, int], int] = {}
    coordinates: list[tuple[float, float]] = []
    for projection in projections:
        bin_idx = _ternary_bin_index(projection.barycentric, resolution)
        bin_counts[bin_idx] = bin_counts.get(bin_idx, 0) + 1
        coordinates.append(_barycentric_to_cartesian(projection.barycentric))

    total_bins = (resolution + 1) * (resolution + 2) // 2
    dense_counts = list(bin_counts.values()) + [0] * (total_bins - len(bin_counts))
    occupied_bins, occupied_fraction = _occupied_bin_fraction(dense_counts, total_bins)
    nn_distances = _nearest_neighbor_distances(coordinates)
    summary = CoverageSummary(
        subset_label="-".join(subset),
        subset_size=3,
        dimensions=2,
        structure_count=len(projections),
        total_bins=total_bins,
        occupied_bins=occupied_bins,
        occupied_bin_fraction=occupied_fraction,
        normalized_entropy=_normalized_entropy(dense_counts, total_bins),
        max_bin_fraction=_max_bin_fraction(dense_counts),
        gini=_gini(dense_counts),
        nn_distance_mean=statistics.fmean(nn_distances) if nn_distances else None,
        nn_distance_p95=_p95(nn_distances),
    )
    return summary, bin_counts


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


def _write_summary_csv(
    summaries: Iterable[CoverageSummary],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subset_label",
        "subset_size",
        "dimensions",
        "structure_count",
        "total_bins",
        "occupied_bins",
        "occupied_bin_fraction",
        "normalized_entropy",
        "max_bin_fraction",
        "gini",
        "nn_distance_mean",
        "nn_distance_p95",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({
                key: (
                    f"{value:.12f}" if isinstance(value, float) else value
                )
                for key, value in asdict(summary).items()
            })


def _write_summary_json(
    summaries: Iterable[CoverageSummary],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(summary) for summary in summaries]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _plot_binary_subset(
    subset: tuple[str, str],
    bin_counts: list[int],
    output_path: Path,
    show: bool,
) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm

    max_count = max(bin_counts) if bin_counts else 0
    boundaries = list(range(max_count + 2))
    cmap = plt.get_cmap("viridis", max(max_count + 1, 1))
    norm = BoundaryNorm(boundaries, cmap.N)
    xs = [(idx + 0.5) / len(bin_counts) for idx in range(len(bin_counts))]
    colors = [count for count in bin_counts]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    bars = ax.bar(
        xs,
        bin_counts,
        width=0.9 / len(bin_counts),
        color=[cmap(norm(count)) for count in colors],
        edgecolor="black",
        linewidth=0.25,
    )
    _ = bars

    ax.set_title(f"Binary composition coverage: {subset[0]}-{subset[1]}")
    ax.set_xlabel(f"Normalized fraction of {subset[1]} within {subset[0]}-{subset[1]}")
    ax.set_ylabel("Structures in bin")
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([i / 10 for i in range(11)])

    heat_ax = ax.inset_axes([0.0, -0.18, 1.0, 0.08], transform=ax.transAxes)
    heat_ax.imshow([bin_counts], aspect="auto", cmap=cmap, norm=norm, extent=[0.0, 1.0, 0.0, 1.0])
    heat_ax.set_yticks([])
    heat_ax.set_xticks([i / 10 for i in range(11)])
    heat_ax.set_xlabel("Occupancy strip")

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm),
        ax=ax,
        pad=0.02,
    )
    colorbar.set_label("Structures in bin")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _plot_ternary_subset(
    subset: tuple[str, str, str],
    bin_counts: dict[tuple[int, int, int], int],
    resolution: int,
    output_path: Path,
    show: bool,
) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm

    if not bin_counts:
        return

    centers = [_ternary_bin_center(index, resolution) for index in bin_counts]
    xs, ys = zip(*[_barycentric_to_cartesian(center) for center in centers])
    counts = [bin_counts[index] for index in bin_counts]

    max_count = max(counts)
    boundaries = list(range(max_count + 2))
    cmap = plt.get_cmap("viridis", max(max_count + 1, 1))
    norm = BoundaryNorm(boundaries, cmap.N)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=160)
    marker_area = max(1000 / max(resolution, 1), 24)
    scatter = ax.scatter(
        xs,
        ys,
        c=counts,
        cmap=cmap,
        norm=norm,
        s=marker_area,
        marker="h",
        edgecolors="none",
    )

    triangle_x = [0.0, 1.0, 0.5, 0.0]
    triangle_y = [0.0, 0.0, SQRT3_OVER_2, 0.0]
    ax.plot(triangle_x, triangle_y, color="black", linewidth=1.2)
    ax.text(-0.03, -0.03, subset[0], ha="right", va="top")
    ax.text(1.03, -0.03, subset[1], ha="left", va="top")
    ax.text(0.5, SQRT3_OVER_2 + 0.03, subset[2], ha="center", va="bottom")
    ax.set_title(f"Ternary composition coverage: {'-'.join(subset)}")
    ax.set_aspect("equal")
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, SQRT3_OVER_2 + 0.1)
    ax.axis("off")

    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("Structures in bin")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze binary / ternary composition-space coverage for selected structures."
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
        help="Output directory (default: PROJECT/reports/composition_coverage)",
    )
    parser.add_argument(
        "--binary-bins",
        type=int,
        default=20,
        help="Number of bins for each binary projection (default: 20)",
    )
    parser.add_argument(
        "--ternary-resolution",
        type=int,
        default=18,
        help="Simplex grid resolution for ternary projections (default: 18)",
    )
    parser.add_argument(
        "--include-subsets",
        choices=("binary", "ternary", "both"),
        default="both",
        help="Which subset projections to generate (default: both)",
    )
    parser.add_argument(
        "--summary-format",
        choices=("csv", "json", "both"),
        default="both",
        help="Summary output format (default: both)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively after saving them",
    )
    args = parser.parse_args()
    if args.binary_bins <= 0:
        parser.error("--binary-bins must be positive")
    if args.ternary_resolution <= 0:
        parser.error("--ternary-resolution must be positive")
    return args


def main() -> None:
    args = _parse_args()

    project_dir = args.project.resolve()
    if not project_dir.exists():
        logger.error(f"Project directory not found: {project_dir}")
        sys.exit(1)

    train_xyz = args.train_xyz or (project_dir / "structures" / "selected" / "train.xyz")
    if not train_xyz.exists():
        logger.error(f"Training XYZ not found: {train_xyz}")
        sys.exit(1)

    output_dir = args.output_dir or (project_dir / "reports" / "composition_coverage")
    binary_plot_dir = output_dir / "binary"
    ternary_plot_dir = output_dir / "ternary"

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

    _write_composition_summary(compositions, output_dir / "composition_summary.csv")

    summaries: list[CoverageSummary] = []

    if args.include_subsets in {"binary", "both"}:
        binary_projections = _collect_binary_projections(compositions)
        for subset, projections in sorted(binary_projections.items()):
            summary, bin_counts = _summarize_binary_subset(
                subset,
                projections,
                args.binary_bins,
            )
            summaries.append(summary)
            _plot_binary_subset(
                subset,
                bin_counts,
                binary_plot_dir / f"{subset[0]}_{subset[1]}.png",
                args.show,
            )

    if args.include_subsets in {"ternary", "both"}:
        ternary_projections = _collect_ternary_projections(compositions)
        for subset, projections in sorted(ternary_projections.items()):
            summary, bin_counts = _summarize_ternary_subset(
                subset,
                projections,
                args.ternary_resolution,
            )
            summaries.append(summary)
            _plot_ternary_subset(
                subset,
                bin_counts,
                args.ternary_resolution,
                ternary_plot_dir / f"{subset[0]}_{subset[1]}_{subset[2]}.png",
                args.show,
            )

    if not summaries:
        logger.error("No binary or ternary subset projections were found in the dataset")
        sys.exit(1)

    summaries.sort(key=lambda item: (item.subset_size, item.subset_label))
    if args.summary_format in {"csv", "both"}:
        _write_summary_csv(summaries, output_dir / "coverage_summary.csv")
    if args.summary_format in {"json", "both"}:
        _write_summary_json(summaries, output_dir / "coverage_summary.json")

    logger.info(f"Wrote coverage summary to {output_dir}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
