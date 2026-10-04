"""Selection diagnostics and descriptor-space reporting."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def plot_descriptor_space(
    representations: np.ndarray,
    train_indices: list[int],
    test_indices: list[int],
    output_path: Path,
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
    ax.set_title("NEP Descriptor Space - Train/Test Selection")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info("  Saved plot to %s", output_path)


__all__ = ["plot_descriptor_space"]
