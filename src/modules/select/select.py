"""
Structure selection stage — farthest-point sampling in NEP descriptor space.

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
from NepTrainKit.core.calculator import NepCalculator
from NepTrainKit.core.structure import Structure
from NepTrainKit.core.io import farthest_point_sampling

from ..base import Stage

logger = logging.getLogger("nepflow.select")


class SelectStage(Stage):
    """Select representative subset from candidates via FPS in descriptor space."""

    def run(self) -> None:
        logger.info("Running structure selection")

        # === DEBUG: random split without NepCalculator/FPS ===
        if self.debug:
            return self._run_debug()

        config_path = self._find_config_file()
        config = ConfigParser()
        config.read(config_path)
        logger.debug(f"Loaded config from {config_path}")

        # ----------------------------------------------------------
        # Step 1: Load generated structures
        # ----------------------------------------------------------
        logger.info("")
        logger.info("Step 1: Loading generated structures")
        generated_path = self.project_dir / "structures" / "generated" / "generated_structures.xyz"
        if not generated_path.exists():
            raise FileNotFoundError(
                f"No generated structures found at {generated_path}\n"
                f"Run the 'generate' stage first."
            )

        t0 = time.perf_counter()
        structures = Structure.read_multiple(str(generated_path))
        # Also load with ASE for lossless extxyz round-tripping (NepTrainKit's
        # writer corrupts quoted JSON values in info fields).
        ase_structures = ase_read(str(generated_path), index=":", format="extxyz")
        t_load = time.perf_counter() - t0
        logger.info(f"  Loaded {len(structures)} candidate structures ({t_load:.1f}s)")

        if len(structures) == 0:
            logger.warning("No structures to select from — aborting")
            return

        # ----------------------------------------------------------
        # Step 2: Initialize NEP calculator and compute descriptors
        # ----------------------------------------------------------
        logger.info("")
        logger.info("Step 2: Computing NEP descriptors")

        descriptor_type = config.get("selection", "descriptor_type", fallback="structure")
        mean_descriptor = descriptor_type == "structure"
        batch_size = config.getint("selection", "batch_size", fallback=500)

        descriptor_cache = self.project_dir / "nep" / "datasets" / "descriptors.npy"
        if descriptor_cache.exists():
            descriptors = np.load(descriptor_cache)
            logger.info(f"  Loaded cached descriptors from {descriptor_cache}")
            if descriptors.shape[0] != len(structures):
                logger.warning(
                    f"  Cache mismatch: {descriptors.shape[0]} descriptors vs {len(structures)} structures — recomputing"
                )
                descriptors = None
        else:
            descriptors = None

        if descriptors is None:
            nep_model_file = config.get("selection", "nep_model_file", fallback="nep89.txt")
            nep_model_path = self.project_dir / "config" / "nep" / nep_model_file
            if not nep_model_path.exists():
                raise FileNotFoundError(
                    f"NEP model not found at {nep_model_path}\n"
                    f"Download NEP89 from:\n"
                    f"  https://github.com/brucefan1983/GPUMD/tree/master/potentials/nep/nep89_20250409\n"
                    f"Place the model file as: {nep_model_path}"
                )

            calc = NepCalculator(str(nep_model_path))
            logger.info(f"  Loaded NEP model: {nep_model_file}")

            descriptors = self._compute_descriptors_batched(
                calc, structures, mean_descriptor, batch_size
            )
            descriptor_cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(descriptor_cache, descriptors)
            logger.info(f"  Saved descriptors to {descriptor_cache}")

        logger.info(f"  Descriptor shape: {descriptors.shape}")
        logger.info(f"  Descriptor type: {descriptor_type}")

        # ----------------------------------------------------------
        # Step 3: FPS — training set
        # ----------------------------------------------------------
        logger.info("")
        logger.info("Step 3: Selecting training set via FPS")

        target_train = config.getint("selection", "target_train_count", fallback=1000)
        target_test = config.getint("selection", "target_test_count", fallback=200)
        tolerance = config.getint("selection", "target_tolerance", fallback=50)
        max_iterations = config.getint("selection", "max_search_iterations", fallback=30)

        if target_train >= len(structures):
            logger.info(f"  target_train_count ({target_train}) >= total ({len(structures)}), selecting all for training")
            train_indices = list(range(len(structures)))
            train_min_dist = 0.0
        else:
            train_indices, train_min_dist = self._fps_target_count(
                descriptors, structures, mean_descriptor,
                target_train, tolerance, max_iterations, label="train",
            )

        logger.info(f"  Training set: {len(train_indices)} structures (min_distance={train_min_dist:.6f})")

        # ----------------------------------------------------------
        # Step 4: FPS — test set from remaining structures
        #   Pre-filter to candidates farthest from the training set
        #   so the test set targets under-represented regions.
        # ----------------------------------------------------------
        logger.info("")
        logger.info("Step 4: Selecting test set via FPS (from remaining structures)")

        train_set = set(train_indices)
        remaining_mask = np.array([i not in train_set for i in range(len(structures))])
        remaining_indices = np.where(remaining_mask)[0]
        logger.info(f"  {len(remaining_indices)} candidates remaining after training selection")

        if len(remaining_indices) == 0:
            logger.warning("  No structures remain for test set")
            test_indices = []
            test_min_dist = 0.0
            min_train_test_dist = float("inf")
            mean_train_test_dist = float("inf")
        elif target_test >= len(remaining_indices):
            logger.info(f"  target_test_count ({target_test}) >= remaining ({len(remaining_indices)}), using all remaining for test")
            test_indices = remaining_indices.tolist()
            test_min_dist = 0.0
            min_train_test_dist, mean_train_test_dist = self._cross_distance_stats(descriptors, train_indices, test_indices)
        else:
            # Rank remaining candidates by distance to nearest training point
            # and keep only the farthest half — the pool for test FPS.
            from scipy.spatial.distance import cdist
            d_to_train = cdist(
                descriptors[remaining_indices], descriptors[train_indices]
            ).min(axis=1)

            test_pool_factor = config.getfloat("selection", "test_pool_factor", fallback=0.5)
            pool_size = max(target_test, int(len(remaining_indices) * test_pool_factor))
            top_k = np.argsort(d_to_train)[::-1][:pool_size]
            pool_indices = remaining_indices[top_k]
            logger.info(
                f"  Pre-filtered to {len(pool_indices)} candidates farthest from training "
                f"(top {test_pool_factor*100:.0f}%, min d_train in pool: {d_to_train[top_k[-1]]:.6f}, "
                f"max: {d_to_train[top_k[0]]:.6f})"
            )

            pool_descriptors = descriptors[pool_indices]
            test_local, test_min_dist = self._fps_target_count(
                pool_descriptors, [structures[i] for i in pool_indices],
                mean_descriptor, target_test, tolerance, max_iterations, label="test",
            )
            test_indices = [int(pool_indices[i]) for i in test_local]
            min_train_test_dist, mean_train_test_dist = self._cross_distance_stats(descriptors, train_indices, test_indices)

        logger.info(f"  Test set: {len(test_indices)} structures (min_distance={test_min_dist:.6f})")
        logger.info(f"  Train↔test nearest-neighbour distance — min: {min_train_test_dist:.6f}, mean: {mean_train_test_dist:.6f}")

        # ----------------------------------------------------------
        # Step 5: Plot descriptor space
        # ----------------------------------------------------------
        logger.info("")
        logger.info("Step 5: Plotting descriptor space")

        reports_dir = self.project_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        self._plot_descriptor_space(
            descriptors, train_indices, test_indices,
            reports_dir / "descriptor_space.png",
        )

        # ----------------------------------------------------------
        # Step 6: Save training and test sets
        # ----------------------------------------------------------
        logger.info("")
        logger.info("Step 6: Saving selected structures")

        selected_dir = self.project_dir / "structures" / "selected"
        selected_dir.mkdir(parents=True, exist_ok=True)

        train_path = selected_dir / "train.xyz"
        ase_write(str(train_path), [ase_structures[i] for i in train_indices], format="extxyz")
        logger.info(f"  Training set saved to {train_path}")

        test_path = selected_dir / "test.xyz"
        ase_write(str(test_path), [ase_structures[i] for i in test_indices], format="extxyz")
        logger.info(f"  Test set saved to {test_path}")

        # ----------------------------------------------------------
        # Summary
        # ----------------------------------------------------------
        logger.info("")
        logger.info(f"Selection complete from {len(structures)} candidates:")
        logger.info(f"  Training: {len(train_indices)} structures (FPS min_distance={train_min_dist:.6f})")
        logger.info(f"  Test:     {len(test_indices)} structures (FPS min_distance={test_min_dist:.6f})")
        logger.info(f"  Train↔test NN distance — min: {min_train_test_dist:.6f}, mean: {mean_train_test_dist:.6f}")
        logger.info(f"  Total selected: {len(train_indices) + len(test_indices)} ({100 * (len(train_indices) + len(test_indices)) / len(structures):.1f}%)")
        logger.info(f"  Plot: {reports_dir / 'descriptor_space.png'}")

    # ==================================================================
    # helpers
    # ==================================================================

    def _find_config_file(self) -> Path:
        project_config = self.project_dir / "config" / "project.config"
        if project_config.exists():
            return project_config
        if self.config_file.exists():
            return self.config_file
        raise FileNotFoundError(
            f"Config file not found. Tried:\n"
            f"  - {project_config}\n"
            f"  - {self.config_file}"
        )

    def _run_debug(self) -> None:
        """Select structures via random split (no NepCalculator/FPS)."""
        generated_path = self.project_dir / "structures" / "generated" / "generated_structures.xyz"
        if not generated_path.exists():
            raise FileNotFoundError(
                f"No generated structures found at {generated_path}\n"
                f"Run the 'generate' stage first."
            )

        ase_structures = ase_read(str(generated_path), index=":", format="extxyz")
        n = len(ase_structures)
        logger.info(f"[DEBUG] Loaded {n} structures from {generated_path}")

        # Fake descriptors (random vectors, shape [N, 10])
        rng = np.random.RandomState(42)
        descriptors = rng.randn(n, 10).astype(np.float64)
        descriptor_cache = self.project_dir / "nep" / "datasets" / "descriptors.npy"
        descriptor_cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(descriptor_cache, descriptors)
        logger.info(f"[DEBUG] Saved random descriptors ({descriptors.shape}) to {descriptor_cache}")

        # Random 70/30 train/test split
        indices = rng.permutation(n)
        n_train = max(1, int(round(0.7 * n)))
        train_indices = sorted(indices[:n_train].tolist())
        test_indices = sorted(indices[n_train:].tolist())
        logger.info(f"[DEBUG] Random split: {len(train_indices)} train, {len(test_indices)} test")

        # Save
        selected_dir = self.project_dir / "structures" / "selected"
        selected_dir.mkdir(parents=True, exist_ok=True)

        train_path = selected_dir / "train.xyz"
        ase_write(str(train_path), [ase_structures[i] for i in train_indices], format="extxyz")
        logger.info(f"[DEBUG] Training set saved to {train_path}")

        test_path = selected_dir / "test.xyz"
        ase_write(str(test_path), [ase_structures[i] for i in test_indices], format="extxyz")
        logger.info(f"[DEBUG] Test set saved to {test_path}")

        logger.info("[DEBUG] Structure selection complete")

    def _fps_target_count(
        self,
        descriptors: np.ndarray,
        structures: list[Structure],
        mean_descriptor: bool,
        target: int,
        tolerance: int,
        max_iterations: int,
        label: str = "",
    ) -> tuple[list[int], float]:
        """Binary search on min_distance to hit target structure count."""
        prefix = f"[{label}] " if label else ""
        lo, hi = 0.0, None

        # Find an upper bound: keep doubling until we get fewer than target
        dist = 0.01
        while True:
            count = self._fps_count(descriptors, structures, mean_descriptor, dist)
            logger.info(f"  {prefix}Probing min_distance={dist:.6f} → {count} structures")
            if count <= target:
                hi = dist
                break
            dist *= 2

        # If even dist=0 gives <= target, just use dist=0
        if hi == 0.0:
            indices = self._fps_run(descriptors, structures, mean_descriptor, 0.0)
            return indices, 0.0

        best_dist = hi
        best_indices = self._fps_run(descriptors, structures, mean_descriptor, hi)

        for i in range(max_iterations):
            mid = (lo + hi) / 2
            count = self._fps_count(descriptors, structures, mean_descriptor, mid)
            logger.info(f"  {prefix}Iteration {i+1}: min_distance={mid:.6f} → {count} structures (target={target}±{tolerance})")

            if abs(count - target) <= tolerance:
                best_dist = mid
                best_indices = self._fps_run(descriptors, structures, mean_descriptor, mid)
                break

            if count > target:
                lo = mid
            else:
                hi = mid
                best_dist = mid
                best_indices = self._fps_run(descriptors, structures, mean_descriptor, mid)
        else:
            # Exhausted iterations — use best hi (closest from above)
            logger.info(f"  {prefix}Search did not converge within tolerance; using min_distance={best_dist:.6f}")

        return best_indices, best_dist

    def _fps_run(
        self,
        descriptors: np.ndarray,
        structures: list[Structure],
        mean_descriptor: bool,
        min_dist: float,
    ) -> list[int]:
        """Run FPS and return frame indices."""
        selected_indices = farthest_point_sampling(
            descriptors, n_samples=len(structures), min_dist=min_dist,
        )
        if not mean_descriptor:
            atoms_per_frame = [s.num_atoms for s in structures]
            comesfrom = []
            for i, n in enumerate(atoms_per_frame):
                comesfrom.extend([i] * n)
            return sorted(set(comesfrom[i] for i in selected_indices))
        return sorted(selected_indices)

    def _fps_count(
        self,
        descriptors: np.ndarray,
        structures: list[Structure],
        mean_descriptor: bool,
        min_dist: float,
    ) -> int:
        """Run FPS and return count of selected frame indices."""
        return len(self._fps_run(descriptors, structures, mean_descriptor, min_dist))

    @staticmethod
    def _cross_distance_stats(
        descriptors: np.ndarray,
        indices_a: list[int],
        indices_b: list[int],
    ) -> tuple[float, float]:
        """Return (min, mean) nearest-neighbour distance from B to A.

        For each point in B, find the distance to the closest point in A,
        then return the minimum and mean of those nearest-neighbour distances.
        """
        from scipy.spatial.distance import cdist
        if not indices_a or not indices_b:
            return float("inf"), float("inf")
        # shape (len_b, len_a)
        d = cdist(descriptors[indices_b], descriptors[indices_a])
        nn_dists = d.min(axis=1)  # nearest training neighbour per test point
        return float(nn_dists.min()), float(nn_dists.mean())

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
            coords_2d[unselected, 0], coords_2d[unselected, 1],
            s=4, alpha=0.2, c="gray", label=f"Unselected ({unselected.sum()})",
        )
        ax.scatter(
            coords_2d[train_mask, 0], coords_2d[train_mask, 1],
            s=10, alpha=0.7, c="tab:red", label=f"Train ({train_mask.sum()})",
        )
        ax.scatter(
            coords_2d[test_mask, 0], coords_2d[test_mask, 1],
            s=10, alpha=0.7, c="tab:blue", label=f"Test ({test_mask.sum()})",
        )
        var = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
        ax.set_title("NEP Descriptor Space — Train/Test Selection")
        ax.legend()
        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150)
        plt.close(fig)
        logger.info(f"  Saved plot to {output_path}")

    @staticmethod
    def _compute_descriptors_batched(
        calc: NepCalculator,
        structures: list[Structure],
        mean_descriptor: bool,
        batch_size: int,
    ) -> np.ndarray:
        """Compute descriptors in batches to avoid OOM."""
        all_descriptors = []
        n = len(structures)
        t0 = time.perf_counter()
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = structures[start:end]
            desc = calc.get_structures_descriptor(batch, mean_descriptor=mean_descriptor)
            all_descriptors.append(desc)
            elapsed = time.perf_counter() - t0
            rate = end / elapsed if elapsed > 0 else 0
            eta = (n - end) / rate if rate > 0 else 0
            logger.info(f"  Batch {end}/{n} ({100*end/n:.0f}%) — {elapsed:.1f}s elapsed, ~{eta:.0f}s remaining")
        return np.concatenate(all_descriptors, axis=0)
