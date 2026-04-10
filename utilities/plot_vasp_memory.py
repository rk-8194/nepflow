#!/usr/bin/env python3
"""
Generate benchmark summary plots from nepflow's .vasp_memory CSV.

Produces three plots saved to <output_dir>/:
  heatmap_<N>atoms.png  — NCORE × KPAR loop-time heatmap per (n_atoms, gpus)
  scaling.png           — best loop time vs n_atoms per GPU count
  oom_map.png           — OK vs OOM counts per system size and GPU count

Usage:
    python plot_vasp_memory.py [--memory PATH] [--output DIR]

Defaults:
    --memory  <nepflow_root>/.vasp_memory   (auto-detected from script location)
    --output  <memory_dir_parent>/plots/
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def load_rows(csv_path: Path) -> list[dict]:
    """Read .vasp_memory CSV and return list of row dicts."""
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            rows.append({
                "n_atoms":        int(r["n_atoms"]),
                "n_kpoints_irr":  int(r["n_kpoints_irr"]),
                "n_electrons":    int(r["n_electrons"]),
                "nodes":          int(r["nodes"]),
                "gpus":           int(r["gpus"]),
                "ncore":          int(r["ncore"]),
                "kpar":           int(r["kpar"]),
                "avg_loop_time":  float(r["avg_loop_time"]),
                "oom":            int(r["oom"]),
            })
    return rows


def plot_heatmaps(rows: list[dict], plot_dir: Path) -> None:
    """Plot 1: NCORE × KPAR heatmap per (n_atoms, gpus)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    all_atoms  = sorted({r["n_atoms"] for r in rows})
    all_ncores = sorted({r["ncore"]   for r in rows})
    all_kpars  = sorted({r["kpar"]    for r in rows})

    for n_atoms in all_atoms:
        atom_rows  = [r for r in rows if r["n_atoms"] == n_atoms]
        gpu_counts = sorted({r["gpus"] for r in atom_rows})

        fig, axes = plt.subplots(
            1, len(gpu_counts),
            figsize=(5 * len(gpu_counts), 4),
            squeeze=False,
        )
        fig.suptitle(f"VASP Loop Time — {n_atoms} atoms", fontsize=14)

        nc_idx = {v: i for i, v in enumerate(all_ncores)}
        kp_idx = {v: i for i, v in enumerate(all_kpars)}

        for col, gpus in enumerate(gpu_counts):
            ax     = axes[0, col]
            subset = [r for r in atom_rows if r["gpus"] == gpus]

            grid     = np.full((len(all_ncores), len(all_kpars)), np.nan)
            oom_mask = np.zeros_like(grid, dtype=bool)

            for r in subset:
                ni = nc_idx.get(r["ncore"])
                ki = kp_idx.get(r["kpar"])
                if ni is None or ki is None:
                    continue
                if r["oom"] == 1:
                    oom_mask[ni, ki] = True
                else:
                    grid[ni, ki] = r["avg_loop_time"]

            valid = grid[~np.isnan(grid)]
            if len(valid) > 0:
                im = ax.imshow(
                    grid, aspect="auto", origin="lower",
                    norm=LogNorm(
                        vmin=max(valid.min(), 1e-3),
                        vmax=valid.max(),
                    ),
                    cmap="viridis_r",
                )
                fig.colorbar(im, ax=ax, label="avg loop (s)")

                for ni in range(len(all_ncores)):
                    for ki in range(len(all_kpars)):
                        if oom_mask[ni, ki]:
                            ax.text(ki, ni, "OOM", ha="center", va="center",
                                    color="red", fontweight="bold", fontsize=8)
                        elif not np.isnan(grid[ni, ki]):
                            ax.text(ki, ni, f"{grid[ni, ki]:.1f}",
                                    ha="center", va="center",
                                    color="white", fontsize=7)
            else:
                ax.imshow(np.zeros_like(grid), aspect="auto", origin="lower",
                          cmap="Greys", vmin=0, vmax=1)
                for ni in range(len(all_ncores)):
                    for ki in range(len(all_kpars)):
                        if oom_mask[ni, ki]:
                            ax.text(ki, ni, "OOM", ha="center", va="center",
                                    color="red", fontweight="bold", fontsize=8)

            ax.set_xticks(range(len(all_kpars)))
            ax.set_xticklabels(all_kpars)
            ax.set_yticks(range(len(all_ncores)))
            ax.set_yticklabels(all_ncores)
            ax.set_xlabel("KPAR")
            ax.set_ylabel("NCORE")
            ax.set_title(f"GPU={gpus}")

        fig.tight_layout()
        path = plot_dir / f"heatmap_{n_atoms}atoms.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved {path}")


def plot_scaling(rows: list[dict], plot_dir: Path) -> None:
    """Plot 2: best loop time vs n_electrons per GPU count."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    completed = [r for r in rows if r["oom"] == 0]
    all_gpus  = sorted({r["gpus"] for r in rows})

    all_ne = sorted({r["n_electrons"] for r in rows})

    fig, ax = plt.subplots(figsize=(10, 5))

    series: dict[int, tuple[list, list, list]] = {}
    for gpus in all_gpus:
        x_vals, y_vals, labels = [], [], []
        for ne in all_ne:
            sub = [r for r in completed if r["gpus"] == gpus and r["n_electrons"] == ne]
            if not sub:
                continue
            best = min(sub, key=lambda r: r["avg_loop_time"])
            x_vals.append(ne)
            y_vals.append(best["avg_loop_time"])
            labels.append(f"NC={best['ncore']}, KP={best['kpar']}")
        if x_vals:
            series[gpus] = (x_vals, y_vals, labels)
            ax.plot(x_vals, y_vals, "o-", label=f"GPU={gpus}", markersize=6)

    # Annotate only first and last point per series
    for gpus, (x_vals, y_vals, labels) in series.items():
        for i in {0, len(x_vals) - 1}:
            ax.annotate(
                labels[i], (x_vals[i], y_vals[i]),
                textcoords="offset points",
                xytext=(6, 6), fontsize=7, color="gray",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, ec="none"),
            )

    ax.set_xscale("log")
    ax.set_xlabel("N$_\\mathrm{electrons}$")
    ax.set_ylabel("Best avg loop time (s)")
    ax.set_title("VASP Scaling — Best Parameters per GPU Count")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    path = plot_dir / "scaling.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_oom_map(rows: list[dict], plot_dir: Path) -> None:
    """Plot 3: stacked bar of OK vs OOM counts per system size and GPU count."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_atoms = sorted({r["n_atoms"] for r in rows})
    all_gpus  = sorted({r["gpus"]   for r in rows})

    fig, ax = plt.subplots(figsize=(7, 4))
    bar_width = 0.25
    x_pos = np.arange(len(all_atoms))

    for i, gpus in enumerate(all_gpus):
        ok_counts, oom_counts = [], []
        for n_atoms in all_atoms:
            sub = [r for r in rows if r["gpus"] == gpus and r["n_atoms"] == n_atoms]
            ok_counts.append(sum(1 for r in sub if r["oom"] == 0))
            oom_counts.append(sum(1 for r in sub if r["oom"] == 1))

        offset = (i - len(all_gpus) / 2 + 0.5) * bar_width
        ax.bar(x_pos + offset, ok_counts, bar_width,
               label=f"GPU={gpus} OK", color=f"C{i}", alpha=0.8)
        ax.bar(x_pos + offset, oom_counts, bar_width, bottom=ok_counts,
               color=f"C{i}", alpha=0.3, hatch="//",
               label=f"GPU={gpus} OOM")

    ax.set_xlabel("Number of atoms")
    ax.set_ylabel("Number of benchmarks")
    ax.set_title("Benchmark Outcomes — OK vs OOM")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_atoms)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = plot_dir / "oom_map.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def main() -> None:
    # Auto-detect .vasp_memory relative to this script's location
    # (utilities/ lives one level below the nepflow root)
    script_dir   = Path(__file__).resolve().parent
    nepflow_root = script_dir.parent
    default_csv  = nepflow_root / ".vasp_memory"

    parser = argparse.ArgumentParser(
        description="Plot benchmark data from nepflow's .vasp_memory CSV."
    )
    parser.add_argument(
        "--memory", type=Path, default=default_csv,
        metavar="PATH",
        help=f"Path to .vasp_memory CSV (default: {default_csv})",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        metavar="DIR",
        help="Output directory for plots (default: <memory_dir>/plots/)",
    )
    args = parser.parse_args()

    csv_path: Path = args.memory
    if not csv_path.exists():
        print(f"ERROR: .vasp_memory not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    plot_dir: Path = args.output if args.output else csv_path.parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("ERROR: matplotlib is required. Install with: pip install matplotlib",
              file=sys.stderr)
        sys.exit(1)

    print(f"Reading {csv_path}")
    rows = load_rows(csv_path)
    print(f"  {len(rows)} rows loaded "
          f"({sum(1 for r in rows if r['oom'] == 0)} timed, "
          f"{sum(1 for r in rows if r['oom'] == 1)} OOM)")

    print(f"Writing plots to {plot_dir}/")
    plot_heatmaps(rows, plot_dir)
    plot_scaling(rows, plot_dir)
    plot_oom_map(rows, plot_dir)

    print("Done.")


if __name__ == "__main__":
    main()
