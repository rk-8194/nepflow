#!/usr/bin/env python3
"""
Plot nearest-neighbor distance trends against dataset size.

Input format:
    --point N,MEAN,P95,P99

Or:
    --input file.csv

CSV/TSV input must contain columns named:
    structures, mean, p95, p99

Example:
    python plot_nn_distance_trends.py \
        --point 910,0.005992,0.020548,0.030592 \
        --point 1779,0.003580,0.011589,0.017068 \
        --output reports/nn_distance_trends.png
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class SummaryPoint:
    structures: int
    mean: float
    p95: float
    p99: float


def _parse_point(text: str) -> SummaryPoint:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "Points must be formatted as N,MEAN,P95,P99"
        )
    try:
        return SummaryPoint(
            structures=int(parts[0]),
            mean=float(parts[1]),
            p95=float(parts[2]),
            p99=float(parts[3]),
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _load_points_from_file(path: Path) -> list[SummaryPoint]:
    with path.open(newline="", encoding="utf-8") as handle:
        sample = handle.read(2048)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        reader = csv.DictReader(handle, dialect=dialect)
        required = {"structures", "mean", "p95", "p99"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Input file must contain columns: {', '.join(sorted(required))}"
            )
        points: list[SummaryPoint] = []
        for row in reader:
            points.append(
                SummaryPoint(
                    structures=int(row["structures"]),
                    mean=float(row["mean"]),
                    p95=float(row["p95"]),
                    p99=float(row["p99"]),
                )
            )
    return points


def _sort_points(points: Iterable[SummaryPoint]) -> list[SummaryPoint]:
    return sorted(points, key=lambda p: p.structures)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot nearest-neighbor distance statistics against dataset size."
    )
    parser.add_argument(
        "--point",
        action="append",
        default=[],
        metavar="N,MEAN,P95,P99",
        help="Add one summary point. May be repeated.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="CSV/TSV file with columns structures, mean, p95, p99.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports") / "nn_distance_trends.png",
        help="Output image path (default: reports/nn_distance_trends.png)",
    )
    parser.add_argument(
        "--title",
        default="NN Distance Coverage Trends",
        help="Plot title",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively after saving it.",
    )
    args = parser.parse_args()

    points: list[SummaryPoint] = []
    if args.input:
        points.extend(_load_points_from_file(args.input))
    points.extend(_parse_point(text) for text in args.point)

    if not points:
        parser.error("Provide at least one --point or an --input file")

    points = _sort_points(points)
    xs = [p.structures for p in points]
    mean = [p.mean for p in points]
    p95 = [p.p95 for p in points]
    p99 = [p.p99 for p in points]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    ax.plot(xs, mean, marker="o", linewidth=2.2, label="Mean")
    ax.plot(xs, p95, marker="o", linewidth=2.2, label="P95")
    ax.plot(xs, p99, marker="o", linewidth=2.2, label="P99")

    for label, values in (("Mean", mean), ("P95", p95), ("P99", p99)):
        for x, y in zip(xs, values):
            ax.annotate(
                f"{y:.4f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
                alpha=0.8,
            )

    ax.set_title(args.title)
    ax.set_xlabel("Number of structures in dataset")
    ax.set_ylabel("Nearest-neighbor distance")
    ax.legend(frameon=True)
    ax.margins(x=0.05)
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
