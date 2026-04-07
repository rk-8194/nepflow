"""
Structure selection stage — farthest-point sampling in NEP descriptor space.

Pipeline:
  1. Load generated structures from structures/generated/
  2. Compute NEP descriptors using foundation model (NEP89) via NepTrainKit
  3. Run farthest-point sampling (FPS) to select target number of structures
  4. Plot descriptor-space visualization (PCA 2D)
  5. Save selected structures to structures/selected/
"""

import logging
import time
from configparser import ConfigParser
from pathlib import Path

import numpy as np
from NepTrainKit.core.calculator import NepCalculator
from NepTrainKit.core.structure import Structure
from NepTrainKit.core.io import farthest_point_sampling

from ..base import Stage

logger = logging.getLogger("nepflow.select")


class SelectStage(Stage):
    """Select representative subset from candidates via FPS in descriptor space."""

    def run(self) -> None:
        logger.info("Running structure selection")

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
        # Step 3: Farthest-point sampling with target count
        # ----------------------------------------------------------
        logger.info("")
        logger.info("Step 3: Running farthest-point sampling")

        target_count = config.getint("selection", "target_count", fallback=1000)
        tolerance = config.getint("selection", "target_tolerance", fallback=50)
        max_iterations = config.getint("selection", "max_search_iterations", fallback=30)

        if target_count >= len(structures):
            logger.info(f"  target_count ({target_count}) >= total structures ({len(structures)}), selecting all")
            frame_indices = list(range(len(structures)))
            final_min_distance = 0.0
        else:
            frame_indices, final_min_distance = self._fps_target_count(
                descriptors, structures, mean_descriptor,
                target_count, tolerance, max_iterations,
            )

        logger.info(f"  Selected {len(frame_indices)} / {len(structures)} structures")
        logger.info(f"  Final min_distance: {final_min_distance:.6f}")

        # ----------------------------------------------------------
        # Step 4: Plot descriptor space
        # ----------------------------------------------------------
        logger.info("")
        logger.info("Step 4: Plotting descriptor space")

        reports_dir = self.project_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        self._plot_descriptor_space(
            descriptors, frame_indices, mean_descriptor,
            reports_dir / "descriptor_space.png",
        )

        # ----------------------------------------------------------
        # Step 5: Save selected structures
        # ----------------------------------------------------------
        logger.info("")
        logger.info("Step 5: Saving selected structures")

        selected_structures = [structures[i] for i in frame_indices]

        selected_dir = self.project_dir / "structures" / "selected"
        selected_dir.mkdir(parents=True, exist_ok=True)
        output_path = selected_dir / "selected_structures.xyz"
        with open(output_path, "w") as f:
            for s in selected_structures:
                s.write(f)
        logger.info(f"  Saved to {output_path}")

        # ----------------------------------------------------------
        # Summary
        # ----------------------------------------------------------
        logger.info("")
        logger.info(f"Selection complete: {len(selected_structures)} structures selected from {len(structures)} candidates")
        logger.info(f"  Reduction: {100 * (1 - len(selected_structures) / len(structures)):.1f}%")
        logger.info(f"  min_distance used: {final_min_distance:.6f}")
        logger.info(f"  Plot saved to: {reports_dir / 'descriptor_space.png'}")

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

    def _fps_target_count(
        self,
        descriptors: np.ndarray,
        structures: list[Structure],
        mean_descriptor: bool,
        target: int,
        tolerance: int,
        max_iterations: int,
    ) -> tuple[list[int], float]:
        """Binary search on min_distance to hit target structure count."""
        n_total = len(structures)
        lo, hi = 0.0, None

        # Find an upper bound: keep doubling until we get fewer than target
        dist = 0.01
        while True:
            count = self._fps_count(descriptors, structures, mean_descriptor, dist)
            logger.info(f"  Probing min_distance={dist:.6f} → {count} structures")
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
            logger.info(f"  Iteration {i+1}: min_distance={mid:.6f} → {count} structures (target={target}±{tolerance})")

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
            logger.info(f"  Search did not converge within tolerance; using min_distance={best_dist:.6f}")

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
    def _plot_descriptor_space(
        descriptors: np.ndarray,
        selected_indices: list[int],
        mean_descriptor: bool,
        output_path: Path,
    ) -> None:
        """PCA 2D scatter plot of all vs selected structures in descriptor space."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(descriptors)

        mask = np.zeros(len(descriptors), dtype=bool)
        mask[selected_indices] = True

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(
            coords_2d[~mask, 0], coords_2d[~mask, 1],
            s=6, alpha=0.3, c="gray", label="Unselected",
        )
        ax.scatter(
            coords_2d[mask, 0], coords_2d[mask, 1],
            s=12, alpha=0.7, c="tab:red", label=f"Selected ({mask.sum()})",
        )
        var = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
        ax.set_title("NEP Descriptor Space — FPS Selection")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
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
