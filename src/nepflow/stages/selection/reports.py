"""Selection diagnostics and descriptor-space reporting."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from nepflow.io.atomic import atomic_write_text
from nepflow.io.json import dumps

from .algorithms.information_entropy.diagnostics import EntropyScientificDiagnostics

logger = logging.getLogger(__name__)


def plot_descriptor_space(
    representations: np.ndarray,
    train_indices: list[int],
    test_indices: list[int],
    output_path: Path,
    *,
    descriptor_label: str = "descriptor",
) -> None:
    """Write the accepted PCA train/test/unselected descriptor plot."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(representations)

    train_mask = np.zeros(len(representations), dtype=bool)
    train_mask[train_indices] = True
    test_mask = np.zeros(len(representations), dtype=bool)
    test_mask[test_indices] = True
    unselected = ~(train_mask | test_mask)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        coords_2d[unselected, 0],
        coords_2d[unselected, 1],
        s=4,
        alpha=0.2,
        c="gray",
        label=f"Unselected ({unselected.sum()})",
    )
    ax.scatter(
        coords_2d[train_mask, 0],
        coords_2d[train_mask, 1],
        s=10,
        alpha=0.7,
        c="tab:red",
        label=f"Train ({train_mask.sum()})",
    )
    ax.scatter(
        coords_2d[test_mask, 0],
        coords_2d[test_mask, 1],
        s=10,
        alpha=0.7,
        c="tab:blue",
        label=f"Test ({test_mask.sum()})",
    )
    variance = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({variance[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({variance[1] * 100:.1f}%)")
    ax.set_title(f"{descriptor_label} - Train/Test Selection")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info("  Saved plot to %s", output_path)


def diagnostics_manifest(
    diagnostics: EntropyScientificDiagnostics | Mapping[str, Any],
) -> dict[str, Any]:
    """Return a validated copy of structured diagnostics for reporting."""

    if isinstance(diagnostics, EntropyScientificDiagnostics):
        return diagnostics.to_manifest()
    if not isinstance(diagnostics, Mapping):
        raise TypeError("diagnostics report input must be a scientific diagnostics record")
    return EntropyScientificDiagnostics.from_manifest(diagnostics).to_manifest()


def render_diagnostics_markdown(
    diagnostics: EntropyScientificDiagnostics | Mapping[str, Any],
) -> str:
    """Render a human-readable report without changing scientific state."""

    manifest = diagnostics_manifest(diagnostics)
    dimensions = manifest["dimensions"]
    objective = manifest["objective"]
    graph = manifest["graph"]
    bandwidth = manifest["bandwidth"]
    coverage = manifest["coverage"]
    lines = [
        "# Information-entropy selection diagnostics",
        "",
        f"Scientific fingerprint: `{manifest['scientific_fingerprint']}`",
        "",
        "## Finite pool",
        "",
        (
            f"- Candidates M={dimensions['candidate_count_M']}; atomic rows "
            f"N={dimensions['atomic_row_count']}; descriptor dimensions "
            f"d={dimensions['descriptor_dimension_before']} and "
            f"d'={dimensions['descriptor_dimension_after']}."
        ),
        f"- Representation fingerprint: `{manifest['representation']['fingerprint']}`.",
        f"- Magnetic mode: `{manifest['representation']['magnetic_mode']}`.",
        "",
        "## Bandwidth and graph",
        "",
        (
            f"- Selected bandwidth: k={bandwidth['calibration']['selected']['k']}, "
            f"c={bandwidth['calibration']['selected']['c']}."
        ),
        f"- Graph edges E={graph['edge_count']} (E/N={graph['edge_per_row']:.6g}).",
        f"- Candidate contribution entries Q={graph['candidate_entry_count']}.",
        f"- Graph CSR array bytes={graph['memory']['csr_array_bytes']}; candidate CSR array bytes="
        f"{graph['memory']['candidate_csr_array_bytes']}; indexed workspace bytes="
        f"{graph['memory']['indexed_workspace_bytes']}; measured peak bytes="
        f"{graph['memory']['measured_peak_memory_bytes']}.",
        "",
        "## Final objective",
        "",
        f"- F(A)={objective['objective_F']:.17g}.",
        f"- Cross entropy={objective['cross_entropy_nats']:.17g} nats.",
        f"- Shannon entropy={objective['shannon_entropy_nats']:.17g} nats.",
        f"- Forward KL={objective['forward_kl_nats']:.17g} nats.",
        f"- Target mass={objective['target_mass']:.17g}; q mass={objective['q_mass']:.17g}.",
        "",
        "## Atomic coverage",
        "",
    ]
    for population, value in coverage.items():
        summary = value["summary"]
        lines.append(
            f"- {population}: rows={summary['count']}, mean={summary['mean']}, "
            f"q50={summary['q50']}, q95={summary['q95']}, q99={summary['q99']}, "
            f"weighting={summary['weighting']}."
        )
    return "\n".join(lines) + "\n"


def write_diagnostics_report(
    diagnostics: EntropyScientificDiagnostics | Mapping[str, Any],
    output_path: Path,
) -> None:
    """Atomically publish a presentation-only Markdown diagnostics report."""

    atomic_write_text(output_path, render_diagnostics_markdown(diagnostics), encoding="utf-8")


def render_diagnostics_json(
    diagnostics: EntropyScientificDiagnostics | Mapping[str, Any],
) -> str:
    """Return strict JSON for machine consumers after validation."""

    return dumps(diagnostics_manifest(diagnostics), indent=2)


__all__ = [
    "diagnostics_manifest",
    "plot_descriptor_space",
    "render_diagnostics_json",
    "render_diagnostics_markdown",
    "write_diagnostics_report",
]
