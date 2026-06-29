"""
Structure selection stage - farthest-point sampling in NEP descriptor space.

Pipeline:
  1. Load generated structures from structures/generated/
  2. Compute NEP descriptors using foundation model (NEP89) via NepTrainKit
  3. FPS to select training set (target_train_count)
  4. FPS on remainder to select test set (target_test_count), ensuring
     minimum separation from the training set in descriptor space
  5. Plot descriptor-space visualization (PCA 2D): training=red, test=blue
  6. Save training and test sets
"""

import logging
import time
from configparser import ConfigParser
from pathlib import Path

import numpy as np
from ase.io import read as ase_read, write as ase_write
from NepTrainKit.core.structure import Structure

from common.FPS import cross_distance_stats, fps_target_count
from common.descriptors import descriptor_cache_path, load_or_compute_descriptors
from ..base import Stage

logger = logging.getLogger("nepflow.select")


class SelectStage(Stage):
    """Select representative subset from candidates via FPS in descriptor space."""

    def run(self) -> None:
        logger.info("Running structure selection")

        if self.debug:
            return self._run_debug()

        config, settings = self.load_config()
        prepared = self.prepare()
        if prepared is None:
            return

        result = self.execute(config, settings, prepared)
        self.finalize(prepared, result)

    # ==================================================================
    # stage lifecycle
    # ==================================================================

    def load_config(self) -> tuple[ConfigParser, dict]:
        """Load config and selection-stage settings."""
        config_path = self._find_config_file()
        config = self._load_config()
        logger.debug(f"Loaded config from {config_path}")
        settings = {
            "descriptor_type": config.get(
                "selection", "descriptor_type", fallback="structure"
            ),
            "batch_size": config.getint("selection", "batch_size", fallback=500),
            "target_train": config.getint(
                "selection", "target_train_count", fallback=1000
            ),
            "target_test": config.getint(
                "selection", "target_test_count", fallback=200
            ),
            "tolerance": config.getint("selection", "target_tolerance", fallback=50),
            "max_iterations": config.getint(
                "selection", "max_search_iterations", fallback=30
            ),
            "test_pool_factor": config.getfloat(
                "selection", "test_pool_factor", fallback=0.5
            ),
            "mean_descriptor": config.get(
                "selection", "descriptor_type", fallback="structure"
            )
            == "structure",
            "nep_model_file": config.get(
                "selection", "nep_model_file", fallback="nep89.txt"
            ),
        }
        return config, settings

    def prepare(self) -> dict | None:
        """Load the generated structures needed for selection."""
        logger.info("")
        logger.info("Step 1: Loading generated structures")

        generated_path = (
            self.project_dir / "structures" / "generated" / "generated_structures.xyz"
        )
        if not generated_path.exists():
            raise FileNotFoundError(
                f"No generated structures found at {generated_path}\n"
                f"Run the 'generate' stage first."
            )

        t0 = time.perf_counter()
        structures = Structure.read_multiple(str(generated_path))
        ase_structures = ase_read(str(generated_path), index=":", format="extxyz")
        elapsed = time.perf_counter() - t0
        logger.info(f"  Loaded {len(structures)} candidate structures ({elapsed:.1f}s)")

        if len(structures) == 0:
            logger.warning("No structures to select from - aborting")
            return None

        return {
            "generated_path": generated_path,
            "structures": structures,
            "ase_structures": ase_structures,
        }

    def execute(self, config: ConfigParser, settings: dict, prepared: dict) -> dict:
        """Compute descriptors and select train/test indices."""
        logger.info("")
        logger.info("Step 2: Computing NEP descriptors")
        descriptors = load_or_compute_descriptors(
            self.project_dir,
            prepared["structures"],
            mean_descriptor=settings["mean_descriptor"],
            batch_size=settings["batch_size"],
            nep_model_file=settings["nep_model_file"],
        )
        logger.info(f"  Descriptor shape: {descriptors.shape}")
        logger.info(f"  Descriptor type: {settings['descriptor_type']}")

        train_indices, train_min_dist = self._select_training_set(
            descriptors,
            prepared["structures"],
            settings,
        )
        test_selection = self._select_test_set(
            descriptors,
            prepared["structures"],
            train_indices,
            settings,
        )

        return {
            "descriptors": descriptors,
            "train_indices": train_indices,
            "train_min_dist": train_min_dist,
            **test_selection,
        }

    def finalize(self, prepared: dict, result: dict) -> None:
        """Persist the selection outputs and report summary."""
        self._plot_results(result["descriptors"], result["train_indices"], result["test_indices"])
        self._save_selected_structures(
            prepared["ase_structures"],
            result["train_indices"],
            result["test_indices"],
        )

        total = len(prepared["structures"])
        train_indices = result["train_indices"]
        test_indices = result["test_indices"]
        logger.info("")
        logger.info(f"Selection complete from {total} candidates:")
        logger.info(
            f"  Training: {len(train_indices)} structures "
            f"(FPS min_distance={result['train_min_dist']:.6f})"
        )
        logger.info(
            f"  Test:     {len(test_indices)} structures "
            f"(FPS min_distance={result['test_min_dist']:.6f})"
        )
        logger.info(
            "  Train<->test NN distance - min: %.6f, mean: %.6f",
            result["min_train_test_dist"],
            result["mean_train_test_dist"],
        )
        logger.info(
            "  Total selected: %d (%.1f%%)",
            len(train_indices) + len(test_indices),
            100 * (len(train_indices) + len(test_indices)) / total,
        )
        logger.info(f"  Plot: {self.project_dir / 'reports' / 'descriptor_space.png'}")

    # ==================================================================
    # selection steps
    # ==================================================================

    def _select_training_set(
        self,
        descriptors: np.ndarray,
        structures: list[Structure],
        settings: dict,
    ) -> tuple[list[int], float]:
        logger.info("")
        logger.info("Step 3: Selecting training set via FPS")

        target_train = settings["target_train"]
        if target_train >= len(structures):
            logger.info(
                "  target_train_count (%d) >= total (%d), selecting all for training",
                target_train,
                len(structures),
            )
            train_indices = list(range(len(structures)))
            train_min_dist = 0.0
        else:
            train_indices, train_min_dist = self._fps_target_count(
                descriptors, structures, settings, target_train, label="train"
            )

        logger.info(
            f"  Training set: {len(train_indices)} structures "
            f"(min_distance={train_min_dist:.6f})"
        )
        return train_indices, train_min_dist

    def _select_test_set(
        self,
        descriptors: np.ndarray,
        structures: list[Structure],
        train_indices: list[int],
        settings: dict,
    ) -> dict:
        logger.info("")
        logger.info("Step 4: Selecting test set via FPS (from remaining structures)")

        train_set = set(train_indices)
        remaining_mask = np.array([i not in train_set for i in range(len(structures))])
        remaining_indices = np.where(remaining_mask)[0]
        logger.info(
            f"  {len(remaining_indices)} candidates remaining after training selection"
        )

        target_test = settings["target_test"]
        if len(remaining_indices) == 0:
            logger.warning("  No structures remain for test set")
            return {
                "test_indices": [],
                "test_min_dist": 0.0,
                "min_train_test_dist": float("inf"),
                "mean_train_test_dist": float("inf"),
            }

        if target_test >= len(remaining_indices):
            logger.info(
                "  target_test_count (%d) >= remaining (%d), using all remaining for test",
                target_test,
                len(remaining_indices),
            )
            test_indices = remaining_indices.tolist()
            min_dist, mean_dist = self._cross_distance_stats(
                descriptors, train_indices, test_indices
            )
            return {
                "test_indices": test_indices,
                "test_min_dist": 0.0,
                "min_train_test_dist": min_dist,
                "mean_train_test_dist": mean_dist,
            }

        from scipy.spatial.distance import cdist

        d_to_train = cdist(descriptors[remaining_indices], descriptors[train_indices]).min(
            axis=1
        )
        pool_size = max(
            target_test,
            int(len(remaining_indices) * settings["test_pool_factor"]),
        )
        top_k = np.argsort(d_to_train)[::-1][:pool_size]
        pool_indices = remaining_indices[top_k]
        logger.info(
            "  Pre-filtered to %d candidates farthest from training "
            "(top %.0f%%, min d_train in pool: %.6f, max: %.6f)",
            len(pool_indices),
            settings["test_pool_factor"] * 100,
            d_to_train[top_k[-1]],
            d_to_train[top_k[0]],
        )

        pool_descriptors = descriptors[pool_indices]
        test_local, test_min_dist = self._fps_target_count(
            pool_descriptors,
            [structures[i] for i in pool_indices],
            settings,
            target_test,
            label="test",
        )
        test_indices = [int(pool_indices[i]) for i in test_local]
        min_dist, mean_dist = self._cross_distance_stats(
            descriptors,
            train_indices,
            test_indices,
        )
        logger.info(
            f"  Test set: {len(test_indices)} structures (min_distance={test_min_dist:.6f})"
        )
        logger.info(
            "  Train<->test nearest-neighbour distance - min: %.6f, mean: %.6f",
            min_dist,
            mean_dist,
        )
        return {
            "test_indices": test_indices,
            "test_min_dist": test_min_dist,
            "min_train_test_dist": min_dist,
            "mean_train_test_dist": mean_dist,
        }

    def _plot_results(
        self,
        descriptors: np.ndarray,
        train_indices: list[int],
        test_indices: list[int],
    ) -> None:
        logger.info("")
        logger.info("Step 5: Plotting descriptor space")
        reports_dir = self.project_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        self._plot_descriptor_space(
            descriptors,
            train_indices,
            test_indices,
            reports_dir / "descriptor_space.png",
        )

    def _save_selected_structures(
        self,
        ase_structures: list,
        train_indices: list[int],
        test_indices: list[int],
    ) -> None:
        logger.info("")
        logger.info("Step 6: Saving selected structures")
        selected_dir = self.project_dir / "structures" / "selected"
        selected_dir.mkdir(parents=True, exist_ok=True)

        train_path = selected_dir / "train.xyz"
        ase_write(
            str(train_path),
            [ase_structures[i] for i in train_indices],
            format="extxyz",
        )
        logger.info(f"  Training set saved to {train_path}")

        test_path = selected_dir / "test.xyz"
        ase_write(
            str(test_path),
            [ase_structures[i] for i in test_indices],
            format="extxyz",
        )
        logger.info(f"  Test set saved to {test_path}")

    # ==================================================================
    # debug / algorithms
    # ==================================================================

    def _run_debug(self) -> None:
        """Select structures via random split (no NepCalculator/FPS)."""
        generated_path = (
            self.project_dir / "structures" / "generated" / "generated_structures.xyz"
        )
        if not generated_path.exists():
            raise FileNotFoundError(
                f"No generated structures found at {generated_path}\n"
                f"Run the 'generate' stage first."
            )

        ase_structures = ase_read(str(generated_path), index=":", format="extxyz")
        n = len(ase_structures)
        logger.info(f"[DEBUG] Loaded {n} structures from {generated_path}")

        rng = np.random.RandomState(42)
        descriptors = rng.randn(n, 10).astype(np.float64)
        descriptor_cache = descriptor_cache_path(self.project_dir)
        descriptor_cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(descriptor_cache, descriptors)
        logger.info(f"[DEBUG] Saved random descriptors ({descriptors.shape}) to {descriptor_cache}")

        indices = rng.permutation(n)
        n_train = max(1, int(round(0.7 * n)))
        train_indices = sorted(indices[:n_train].tolist())
        test_indices = sorted(indices[n_train:].tolist())
        logger.info(f"[DEBUG] Random split: {len(train_indices)} train, {len(test_indices)} test")

        self._save_selected_structures(ase_structures, train_indices, test_indices)
        logger.info("[DEBUG] Structure selection complete")

    def _fps_target_count(
        self,
        descriptors: np.ndarray,
        structures: list[Structure],
        settings: dict,
        target: int,
        label: str = "",
    ) -> tuple[list[int], float]:
        return fps_target_count(
            descriptors,
            structures,
            settings["mean_descriptor"],
            target,
            settings["tolerance"],
            settings["max_iterations"],
            label=label,
        )

    @staticmethod
    def _cross_distance_stats(
        descriptors: np.ndarray,
        indices_a: list[int],
        indices_b: list[int],
    ) -> tuple[float, float]:
        return cross_distance_stats(descriptors, indices_a, indices_b)

    @staticmethod
    def _plot_descriptor_space(
        descriptors: np.ndarray,
        train_indices: list[int],
        test_indices: list[int],
        output_path: Path,
    ) -> None:
        """PCA 2D scatter plot: training (red), test (blue), unselected (gray)."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(descriptors)

        train_mask = np.zeros(len(descriptors), dtype=bool)
        train_mask[train_indices] = True
        test_mask = np.zeros(len(descriptors), dtype=bool)
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
        var = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
        ax.set_title("NEP Descriptor Space - Train/Test Selection")
        ax.legend()
        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150)
        plt.close(fig)
        logger.info(f"  Saved plot to {output_path}")
