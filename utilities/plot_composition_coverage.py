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
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from nepflow.stages.selection.sampling import (
    SQRT3_OVER_2,
    BinaryProjection,
    CoverageSummary,
    StructureComposition,
    TernaryProjection,
    barycentric_to_cartesian,
    collect_binary_projections,
    collect_ternary_projections,
    composition_from_atoms,
    summarize_binary_subset,
    summarize_ternary_subset,
    ternary_bin_center,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("plot_composition_coverage")


def _load_structures(train_xyz: Path) -> list:
    from ase.io import read as ase_read

    structures = ase_read(str(train_xyz), index=":", format="extxyz")
    return structures if isinstance(structures, list) else [structures]


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

    centers = [ternary_bin_center(index, resolution) for index in bin_counts]
    xs, ys = zip(*[barycentric_to_cartesian(center) for center in centers])
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
        composition_from_atoms(atoms, structure_index=index)
        for index, atoms in enumerate(structures)
    ]
    logger.info(f"Loaded {len(compositions)} structures")

    _write_composition_summary(compositions, output_dir / "composition_summary.csv")

    summaries: list[CoverageSummary] = []

    if args.include_subsets in {"binary", "both"}:
        binary_projections = collect_binary_projections(compositions)
        for subset, projections in sorted(binary_projections.items()):
            summary, bin_counts = summarize_binary_subset(
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
        ternary_projections = collect_ternary_projections(compositions)
        for subset, projections in sorted(ternary_projections.items()):
            summary, bin_counts = summarize_ternary_subset(
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
