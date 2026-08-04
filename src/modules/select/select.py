"""
Structure selection stage - farthest-point sampling in NEP descriptor space.

Pipeline:
  1. Load generated structures from structures/generated/
  2. Compute NEP descriptors using foundation model (NEP89) via NepTrainKit
  3. Optionally include seed / elastic stress structures as fixed training anchors
  4. FPS to select the remaining training set up to target_train_count
  5. FPS on remainder to select test set (target_test_count), ensuring
     minimum separation from the training set in descriptor space
  6. Plot descriptor-space visualization (PCA 2D): training=red, test=blue
  7. Save training and test sets
"""

import logging
import math
import time
from collections import Counter
from configparser import ConfigParser
from itertools import combinations
from pathlib import Path

import numpy as np
from ase.io import read as ase_read, write as ase_write
from NepTrainKit.core.structure import Structure

from common.FPS import cross_distance_stats, fps_target_count
from common.descriptors import descriptor_cache_path, load_or_compute_descriptors
from ..base import Stage

logger = logging.getLogger("nepflow.select")


COMPOSITION_AWARE_BINARY_BINS = 20
COMPOSITION_AWARE_TERNARY_RESOLUTION = 18
COMPOSITION_AWARE_NOVELTY_FLOOR_FRACTION = 0.90


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
            "include_seed_structures": config.getboolean(
                "selection", "include_seed_structures", fallback=False
            ),
            "include_single_element_elastic_stress_structures": config.getboolean(
                "selection",
                "include_single_element_elastic_stress_structures",
                fallback=False,
            ),
            "include_elastic_stress_structures": config.getboolean(
                "selection", "include_elastic_stress_structures", fallback=False
            ),
            "composition_aware_fps": config.getboolean(
                "selection", "composition_aware_fps", fallback=False
            ),
            "composition_aware_fps_frontier_fraction": config.getfloat(
                "selection",
                "composition_aware_fps_frontier_fraction",
                fallback=0.10,
            ),
            "composition_aware_fps_ternary_weight": config.getfloat(
                "selection",
                "composition_aware_fps_ternary_weight",
                fallback=1.0,
            ),
            "composition_aware_fps_adaptive_retries": config.getint(
                "selection",
                "composition_aware_fps_adaptive_retries",
                fallback=4,
            ),
            "composition_aware_fps_descriptor_floor_fraction": config.getfloat(
                "selection",
                "composition_aware_fps_descriptor_floor_fraction",
                fallback=0.95,
            ),
            "nep_model_file": config.get(
                "selection", "nep_model_file", fallback="nep89.txt"
            ),
        }
        if not 0.0 < settings["composition_aware_fps_frontier_fraction"] <= 1.0:
            raise ValueError(
                "selection.composition_aware_fps_frontier_fraction must be in (0, 1]"
            )
        if settings["composition_aware_fps_ternary_weight"] < 0.0:
            raise ValueError(
                "selection.composition_aware_fps_ternary_weight must be >= 0"
            )
        if settings["composition_aware_fps_adaptive_retries"] < 1:
            raise ValueError(
                "selection.composition_aware_fps_adaptive_retries must be >= 1"
            )
        if not 0.0 < settings["composition_aware_fps_descriptor_floor_fraction"] <= 1.0:
            raise ValueError(
                "selection.composition_aware_fps_descriptor_floor_fraction must be in (0, 1]"
            )
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
        if not isinstance(ase_structures, list):
            ase_structures = [ase_structures]
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

        seed_indices = []
        if settings["include_seed_structures"]:
            seed_indices = self._load_seed_indices(prepared["ase_structures"])
            logger.info(f"  Seed anchors enabled: {len(seed_indices)} structures")

        single_element_elastic_indices = []
        if settings["include_single_element_elastic_stress_structures"]:
            single_element_elastic_indices = (
                self._load_single_element_elastic_stress_indices(
                    prepared["ase_structures"]
                )
            )
            logger.info(
                "  Single-element elastic stress anchors enabled: "
                f"{len(single_element_elastic_indices)} structures"
            )

        elastic_indices = []
        if settings["include_elastic_stress_structures"]:
            elastic_indices = self._load_elastic_stress_indices(prepared["ase_structures"])
            logger.info(
                f"  Elastic stress anchors enabled: {len(elastic_indices)} structures"
            )

        anchor_indices = sorted(
            set(seed_indices + single_element_elastic_indices + elastic_indices)
        )
        train_indices, train_min_dist = self._select_training_set(
            descriptors,
            prepared["structures"],
            settings,
            ase_structures=prepared["ase_structures"],
            seed_indices=seed_indices,
            single_element_elastic_indices=single_element_elastic_indices,
            elastic_indices=elastic_indices,
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
            "train_seed_count": len(seed_indices),
            "train_single_element_elastic_count": len(single_element_elastic_indices),
            "train_elastic_count": len(elastic_indices),
            "train_anchor_count": len(anchor_indices),
            "train_fps_count": len(train_indices) - len(anchor_indices),
            "seed_indices": seed_indices,
            "single_element_elastic_indices": single_element_elastic_indices,
            "elastic_indices": elastic_indices,
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
            f"({result['train_seed_count']} seed anchors, "
            f"{result.get('train_single_element_elastic_count', 0)} "
            f"single-element elastic anchors, "
            f"{result.get('train_elastic_count', 0)} elastic anchors, "
            f"{result.get('train_anchor_count', len(train_indices))} unique anchors, "
            f"{result['train_fps_count']} FPS-selected; "
            f"FPS min_distance={result['train_min_dist']:.6f})"
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
        ase_structures: list | None = None,
        seed_indices: list[int] | None = None,
        single_element_elastic_indices: list[int] | None = None,
        elastic_indices: list[int] | None = None,
    ) -> tuple[list[int], float]:
        logger.info("")
        logger.info("Step 3: Selecting training set via FPS")

        seed_indices = sorted(set(seed_indices or []))
        single_element_elastic_indices = sorted(
            set(single_element_elastic_indices or [])
        )
        elastic_indices = sorted(set(elastic_indices or []))
        anchor_indices = sorted(
            set(seed_indices + single_element_elastic_indices + elastic_indices)
        )
        target_train = settings["target_train"]
        if len(anchor_indices) > target_train:
            raise ValueError(
                "Preselected anchor count exceeds target_train_count: "
                "seed anchors=%d, single-element elastic stress anchors=%d, "
                "all elastic stress anchors=%d, "
                "unique anchors=%d, target_train_count=%d"
                % (
                    len(seed_indices),
                    len(single_element_elastic_indices),
                    len(elastic_indices),
                    len(anchor_indices),
                    target_train,
                )
            )

        if seed_indices:
            logger.info(
                "  Preselected %d seed structures as training anchors",
                len(seed_indices),
            )
        if single_element_elastic_indices:
            logger.info(
                "  Preselected %d single-element elastic stress structures as training anchors",
                len(single_element_elastic_indices),
            )
        if elastic_indices:
            logger.info(
                "  Preselected %d elastic stress structures as training anchors",
                len(elastic_indices),
            )
        if settings.get("composition_aware_fps"):
            logger.info("  Composition-aware FPS enabled for training selection")

        if target_train >= len(structures):
            logger.info(
                "  target_train_count (%d) >= total (%d), selecting all for training",
                target_train,
                len(structures),
            )
            train_indices = list(range(len(structures)))
            train_min_dist = 0.0
        else:
            anchor_set = set(anchor_indices)
            remaining_indices = [i for i in range(len(structures)) if i not in anchor_set]
            remaining_target = target_train - len(anchor_indices)
            if remaining_target == 0:
                train_indices = anchor_indices
                train_min_dist = 0.0
            elif settings.get("composition_aware_fps"):
                train_indices, train_min_dist = self._select_training_set_composition_aware(
                    descriptors,
                    structures,
                    settings,
                    ase_structures or [],
                    anchor_indices,
                    remaining_indices,
                    remaining_target,
                )
            else:
                remaining_descriptors = descriptors[remaining_indices]
                remaining_structures = [structures[i] for i in remaining_indices]
                fps_indices, train_min_dist = self._fps_target_count(
                    remaining_descriptors,
                    remaining_structures,
                    settings,
                    remaining_target,
                    label="train",
                )
                fps_indices = [int(remaining_indices[i]) for i in fps_indices]
                train_indices = sorted(anchor_indices + fps_indices)

        logger.info(
            f"  Training set: {len(train_indices)} structures "
            f"(min_distance={train_min_dist:.6f})"
        )
        return train_indices, train_min_dist

    def _select_training_set_composition_aware(
        self,
        descriptors: np.ndarray,
        structures: list[Structure],
        settings: dict,
        ase_structures: list,
        anchor_indices: list[int],
        remaining_indices: list[int],
        remaining_target: int,
    ) -> tuple[list[int], float]:
        if not ase_structures:
            logger.warning(
                "  Composition-aware FPS requested, but ASE structures were not provided; "
                "falling back to descriptor-only FPS"
            )
            fps_indices, train_min_dist = self._select_training_set_plain_fps(
                descriptors,
                structures,
                settings,
                remaining_indices,
                remaining_target,
            )
            return sorted(anchor_indices + fps_indices), train_min_dist

        candidate_bins = {
            idx: self._composition_projection_bins(ase_structures[idx])
            for idx in range(len(ase_structures))
        }
        total_binary_bins = len({
            bin_key
            for idx in remaining_indices + anchor_indices
            for bin_key in candidate_bins[idx]["binary"]
        })
        total_ternary_bins = len({
            bin_key
            for idx in remaining_indices + anchor_indices
            for bin_key in candidate_bins[idx]["ternary"]
        })
        if total_binary_bins == 0 and total_ternary_bins == 0:
            logger.info(
                "  No binary or ternary composition projections found; "
                "falling back to descriptor-only FPS"
            )
            fps_indices, train_min_dist = self._select_training_set_plain_fps(
                descriptors,
                structures,
                settings,
                remaining_indices,
                remaining_target,
            )
            return sorted(anchor_indices + fps_indices), train_min_dist

        selected_indices = list(anchor_indices)
        binary_counts: Counter = Counter()
        ternary_counts: Counter = Counter()
        for idx in anchor_indices:
            binary_counts.update(candidate_bins[idx]["binary"])
            ternary_counts.update(candidate_bins[idx]["ternary"])

        logger.info(
            "  Composition-aware fill: %d remaining slots, %d binary bins, %d ternary bins",
            remaining_target,
            total_binary_bins,
            total_ternary_bins,
        )
        logger.info(
            "  Anchor composition footprint: %d occupied binary bins, %d occupied ternary bins",
            len(binary_counts),
            len(ternary_counts),
        )

        attempt_schedule = self._composition_aware_attempt_schedule(
            settings["composition_aware_fps_frontier_fraction"],
            settings["composition_aware_fps_ternary_weight"],
            settings["composition_aware_fps_adaptive_retries"],
        )
        attempt_results: list[dict] = []
        for attempt_number, (frontier_fraction, ternary_weight) in enumerate(
            attempt_schedule,
            start=1,
        ):
            logger.info(
                "  Attempt %d/%d: frontier_fraction=%.3f, ternary_weight=%.3f",
                attempt_number,
                len(attempt_schedule),
                frontier_fraction,
                ternary_weight,
            )
            attempt_result = self._run_composition_aware_attempt(
                descriptors,
                candidate_bins,
                anchor_indices,
                remaining_indices,
                remaining_target,
                frontier_fraction,
                ternary_weight,
            )
            attempt_result["attempt_number"] = attempt_number
            attempt_results.append(attempt_result)
            logger.info(
                "    Coverage: binary occ=%.3f, binary entropy=%.3f, ternary occ=%.3f, ternary entropy=%.3f",
                attempt_result["binary_occupied_bin_fraction"],
                attempt_result["binary_normalized_entropy"],
                attempt_result["ternary_occupied_bin_fraction"],
                attempt_result["ternary_normalized_entropy"],
            )
            logger.info(
                "    Descriptor quality: min_dist=%.6f, mean_nn=%.6f",
                attempt_result["train_min_dist"],
                attempt_result["train_mean_nn_dist"],
            )

        best_attempt = self._pick_best_composition_aware_attempt(
            attempt_results,
            settings["composition_aware_fps_descriptor_floor_fraction"],
        )
        logger.info(
            "  Selected attempt %d with frontier_fraction=%.3f, ternary_weight=%.3f",
            best_attempt["attempt_number"],
            best_attempt["frontier_fraction"],
            best_attempt["ternary_weight"],
        )
        logger.info(
            "  Composition-aware selection filled %d structures; touched %d/%d binary bins and %d/%d ternary bins",
            len(best_attempt["selected_indices"]) - len(anchor_indices),
            best_attempt["occupied_binary_bins"],
            total_binary_bins,
            best_attempt["occupied_ternary_bins"],
            total_ternary_bins,
        )
        return best_attempt["selected_indices"], best_attempt["train_min_dist"]

    def _select_training_set_plain_fps(
        self,
        descriptors: np.ndarray,
        structures: list[Structure],
        settings: dict,
        remaining_indices: list[int],
        remaining_target: int,
    ) -> tuple[list[int], float]:
        remaining_descriptors = descriptors[remaining_indices]
        remaining_structures = [structures[i] for i in remaining_indices]
        fps_indices, train_min_dist = self._fps_target_count(
            remaining_descriptors,
            remaining_structures,
            settings,
            remaining_target,
            label="train",
        )
        fps_indices = [int(remaining_indices[i]) for i in fps_indices]
        return fps_indices, train_min_dist

    @staticmethod
    def _extract_composition_fractions(atoms) -> dict[str, float]:
        composition = atoms.info.get("composition")
        if isinstance(composition, dict):
            fractions: dict[str, float] = {}
            for element, value in composition.items():
                try:
                    fraction = float(value)
                except (TypeError, ValueError):
                    continue
                if fraction > 0.0:
                    fractions[str(element)] = fraction
            if fractions:
                total = sum(fractions.values())
                if total > 0.0:
                    return {
                        element: fraction / total
                        for element, fraction in sorted(fractions.items())
                    }

        counts = Counter(atoms.get_chemical_symbols())
        total_atoms = sum(counts.values())
        if total_atoms <= 0:
            return {}
        return {
            element: count / total_atoms
            for element, count in sorted(counts.items())
        }

    def _composition_projection_bins(self, atoms) -> dict[str, list[tuple]]:
        fractions = self._extract_composition_fractions(atoms)
        active_elements = tuple(sorted(
            element for element, fraction in fractions.items() if fraction > 0.0
        ))
        binary_bins: list[tuple] = []
        ternary_bins: list[tuple] = []

        for subset in combinations(active_elements, 2):
            normalized = self._normalize_subset_fractions(fractions, subset)
            binary_bins.append(
                subset + (self._binary_bin_index(normalized[1], COMPOSITION_AWARE_BINARY_BINS),)
            )

        for subset in combinations(active_elements, 3):
            normalized = self._normalize_subset_fractions(fractions, subset)
            ternary_bins.append(
                subset + self._ternary_bin_index(
                    normalized,
                    COMPOSITION_AWARE_TERNARY_RESOLUTION,
                )
            )

        return {"binary": binary_bins, "ternary": ternary_bins}

    @staticmethod
    def _normalize_subset_fractions(
        fractions: dict[str, float],
        subset: tuple[str, ...],
    ) -> tuple[float, ...]:
        subset_total = sum(fractions[element] for element in subset)
        if subset_total <= 0.0:
            return tuple(0.0 for _ in subset)
        return tuple(fractions[element] / subset_total for element in subset)

    @staticmethod
    def _binary_bin_index(value: float, bins: int) -> int:
        clamped = min(max(value, 0.0), 1.0)
        if math.isclose(clamped, 1.0, rel_tol=0.0, abs_tol=1e-12):
            return bins - 1
        return min(int(clamped * bins), bins - 1)

    @staticmethod
    def _largest_remainder_integer_partition(
        values: tuple[float, ...],
        total: int,
    ) -> tuple[int, ...]:
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

    @classmethod
    def _ternary_bin_index(
        cls,
        barycentric: tuple[float, float, float],
        resolution: int,
    ) -> tuple[int, int, int]:
        return cls._largest_remainder_integer_partition(barycentric, resolution)

    @staticmethod
    def _composition_sparsity_reward(
        candidate_bins: dict[str, list[tuple]],
        binary_counts: Counter,
        ternary_counts: Counter,
        binary_subset_totals: Counter,
        ternary_subset_totals: Counter,
        ternary_weight: float,
    ) -> float:
        weighted_rewards = 0.0
        total_weight = 0.0
        for bin_key in candidate_bins["binary"]:
            subset = bin_key[:2]
            weighted_rewards += SelectStage._bin_deficit_reward(
                binary_counts[bin_key],
                binary_subset_totals[subset],
                COMPOSITION_AWARE_BINARY_BINS,
            )
            total_weight += 1.0
        for bin_key in candidate_bins["ternary"]:
            if ternary_weight <= 0.0:
                continue
            subset = bin_key[:3]
            weighted_rewards += ternary_weight * SelectStage._bin_deficit_reward(
                ternary_counts[bin_key],
                ternary_subset_totals[subset],
                SelectStage._ternary_bin_count(COMPOSITION_AWARE_TERNARY_RESOLUTION),
            )
            total_weight += ternary_weight
        if total_weight <= 0.0:
            return 0.0
        return weighted_rewards / total_weight

    @staticmethod
    def _bin_deficit_reward(
        bin_count: int,
        subset_total_count: int,
        total_bins: int,
    ) -> float:
        if total_bins <= 0:
            return 0.0
        target_level = max(1, int(math.floor(subset_total_count / total_bins)) + 1)
        deficit = max(0, target_level - bin_count)
        if deficit <= 0:
            return 0.0
        return deficit / target_level

    @staticmethod
    def _min_max_normalize(values: dict[int, float]) -> dict[int, float]:
        if not values:
            return {}
        low = min(values.values())
        high = max(values.values())
        if math.isclose(low, high, rel_tol=0.0, abs_tol=1e-12):
            return {key: 0.0 for key in values}
        scale = high - low
        return {key: (value - low) / scale for key, value in values.items()}

    def _nearest_descriptor_distance(
        self,
        descriptors: np.ndarray,
        index: int,
        selected_indices: list[int],
    ) -> float:
        if not selected_indices:
            return 0.0
        best = math.inf
        for other_index in selected_indices:
            best = min(best, self._descriptor_distance(descriptors, index, other_index))
        return float(best)

    def _initialize_nearest_distances(
        self,
        descriptors: np.ndarray,
        candidate_indices: list[int],
        selected_indices: list[int],
    ) -> dict[int, float]:
        if not selected_indices:
            return {idx: 0.0 for idx in candidate_indices}
        try:
            from scipy.spatial.distance import cdist

            candidate_matrix = descriptors[candidate_indices]
            selected_matrix = descriptors[selected_indices]
            nearest = cdist(candidate_matrix, selected_matrix).min(axis=1)
            return {
                idx: float(distance)
                for idx, distance in zip(candidate_indices, nearest)
            }
        except Exception:
            return {
                idx: self._nearest_descriptor_distance(descriptors, idx, selected_indices)
                for idx in candidate_indices
            }

    def _update_nearest_distances(
        self,
        descriptors: np.ndarray,
        candidate_indices: list[int],
        new_selected_index: int,
        current_nearest: dict[int, float],
    ) -> None:
        if not candidate_indices:
            return
        try:
            from scipy.spatial.distance import cdist

            candidate_matrix = descriptors[candidate_indices]
            selected_matrix = descriptors[[new_selected_index]]
            distance_matrix = cdist(candidate_matrix, selected_matrix)
            distances = self._flatten_single_column_distances(distance_matrix)
            for idx, distance in zip(candidate_indices, distances):
                current_nearest[idx] = min(current_nearest[idx], float(distance))
        except Exception:
            for idx in candidate_indices:
                distance = self._descriptor_distance(descriptors, idx, new_selected_index)
                current_nearest[idx] = min(current_nearest[idx], distance)

    @staticmethod
    def _flatten_single_column_distances(distance_matrix) -> list[float]:
        if hasattr(distance_matrix, "ravel"):
            return [float(value) for value in distance_matrix.ravel()]

        flattened: list[float] = []
        for row in distance_matrix:
            if isinstance(row, (list, tuple)):
                if not row:
                    continue
                flattened.append(float(row[0]))
            else:
                flattened.append(float(row))
        return flattened

    def _selected_min_distance(
        self,
        descriptors: np.ndarray,
        selected_indices: list[int],
    ) -> float:
        if len(selected_indices) < 2:
            return 0.0
        best = math.inf
        for pos, idx in enumerate(selected_indices):
            for other in selected_indices[pos + 1:]:
                best = min(best, self._descriptor_distance(descriptors, idx, other))
        return float(best if best < math.inf else 0.0)

    def _selected_positive_min_distance(
        self,
        descriptors: np.ndarray,
        selected_indices: list[int],
    ) -> float:
        if len(selected_indices) < 2:
            return 0.0
        best = math.inf
        for pos, idx in enumerate(selected_indices):
            for other in selected_indices[pos + 1:]:
                distance = self._descriptor_distance(descriptors, idx, other)
                if distance > 1e-12:
                    best = min(best, distance)
        return float(best if best < math.inf else 0.0)

    @staticmethod
    def _descriptor_distance(
        descriptors: np.ndarray,
        idx_a: int,
        idx_b: int,
    ) -> float:
        vector_a = descriptors[idx_a]
        vector_b = descriptors[idx_b]
        return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(vector_a, vector_b)))

    @staticmethod
    def _composition_aware_attempt_schedule(
        frontier_fraction: float,
        ternary_weight: float,
        adaptive_retries: int,
    ) -> list[tuple[float, float]]:
        attempts = [
            (frontier_fraction, ternary_weight),
            (max(0.05, 0.5 * frontier_fraction), ternary_weight),
            (min(0.25, 2.0 * frontier_fraction), ternary_weight),
            (frontier_fraction, 2.0 * ternary_weight),
        ]
        return attempts[:max(1, adaptive_retries)]

    def _run_composition_aware_attempt(
        self,
        descriptors: np.ndarray,
        candidate_bins: dict[int, dict[str, list[tuple]]],
        anchor_indices: list[int],
        remaining_indices: list[int],
        remaining_target: int,
        frontier_fraction: float,
        ternary_weight: float,
    ) -> dict:
        selected_indices = list(anchor_indices)
        remaining_pool = list(remaining_indices)
        binary_counts: Counter = Counter()
        ternary_counts: Counter = Counter()
        binary_subset_totals: Counter = Counter()
        ternary_subset_totals: Counter = Counter()
        for idx in anchor_indices:
            binary_counts.update(candidate_bins[idx]["binary"])
            ternary_counts.update(candidate_bins[idx]["ternary"])
            binary_subset_totals.update(
                bin_key[:2] for bin_key in candidate_bins[idx]["binary"]
            )
            ternary_subset_totals.update(
                bin_key[:3] for bin_key in candidate_bins[idx]["ternary"]
            )

        logger.info(
            "    Initializing descriptor distances for %d candidates against %d anchors",
            len(remaining_pool),
            len(selected_indices),
        )
        current_nearest = self._initialize_nearest_distances(
            descriptors,
            remaining_pool,
            selected_indices,
        )
        logger.info("    Initial descriptor-distance initialization complete")

        for iteration in range(remaining_target):
            if not remaining_pool:
                break

            descriptor_ranked = sorted(
                remaining_pool,
                key=lambda idx: (-current_nearest[idx], idx),
            )
            frontier_size = max(1, int(math.ceil(frontier_fraction * len(remaining_pool))))
            frontier = descriptor_ranked[:frontier_size]
            positive_frontier = [
                idx for idx in frontier if current_nearest[idx] > 1e-12
            ]
            if positive_frontier:
                frontier = positive_frontier
            best_frontier_novelty = max(current_nearest[idx] for idx in frontier)
            if best_frontier_novelty > 1e-12:
                novelty_floor = (
                    COMPOSITION_AWARE_NOVELTY_FLOOR_FRACTION * best_frontier_novelty
                )
                novelty_frontier = [
                    idx for idx in frontier if current_nearest[idx] >= novelty_floor
                ]
                if novelty_frontier:
                    frontier = novelty_frontier
            composition_reward = {
                idx: self._composition_sparsity_reward(
                    candidate_bins[idx],
                    binary_counts,
                    ternary_counts,
                    binary_subset_totals,
                    ternary_subset_totals,
                    ternary_weight,
                )
                for idx in frontier
            }
            best_idx = min(
                frontier,
                key=lambda idx: (-composition_reward[idx], -current_nearest[idx], idx),
            )

            selected_indices.append(best_idx)
            binary_counts.update(candidate_bins[best_idx]["binary"])
            ternary_counts.update(candidate_bins[best_idx]["ternary"])
            binary_subset_totals.update(
                bin_key[:2] for bin_key in candidate_bins[best_idx]["binary"]
            )
            ternary_subset_totals.update(
                bin_key[:3] for bin_key in candidate_bins[best_idx]["ternary"]
            )
            remaining_pool.remove(best_idx)
            current_nearest.pop(best_idx, None)
            self._update_nearest_distances(
                descriptors,
                remaining_pool,
                best_idx,
                current_nearest,
            )

            if (iteration + 1) % 50 == 0 or iteration + 1 == remaining_target:
                logger.info(
                    "    Attempt progress: %d/%d selected (remaining candidates: %d)",
                    iteration + 1,
                    remaining_target,
                    len(remaining_pool),
                )

        selected_indices = sorted(selected_indices)
        coverage = self._composition_coverage_metrics(selected_indices, candidate_bins)
        coverage.update(
            {
                "selected_indices": selected_indices,
                "train_min_dist": self._selected_min_distance(
                    descriptors,
                    selected_indices,
                ),
                "train_positive_min_dist": self._selected_positive_min_distance(
                    descriptors,
                    selected_indices,
                ),
                "train_mean_nn_dist": self._selected_mean_nearest_distance(
                    descriptors,
                    selected_indices,
                ),
                "frontier_fraction": frontier_fraction,
                "ternary_weight": ternary_weight,
                "occupied_binary_bins": len(binary_counts),
                "occupied_ternary_bins": len(ternary_counts),
            }
        )
        return coverage

    def _composition_coverage_metrics(
        self,
        selected_indices: list[int],
        candidate_bins: dict[int, dict[str, list[tuple]]],
    ) -> dict[str, float]:
        binary_subset_counts: dict[tuple[str, str], Counter] = {}
        ternary_subset_counts: dict[tuple[str, str, str], Counter] = {}

        for idx in selected_indices:
            for bin_key in candidate_bins[idx]["binary"]:
                subset = bin_key[:2]
                binary_subset_counts.setdefault(subset, Counter())[bin_key] += 1
            for bin_key in candidate_bins[idx]["ternary"]:
                subset = bin_key[:3]
                ternary_subset_counts.setdefault(subset, Counter())[bin_key] += 1

        binary_occupied = [
            len(counts) / COMPOSITION_AWARE_BINARY_BINS
            for counts in binary_subset_counts.values()
        ]
        binary_entropy = [
            self._normalized_entropy(list(counts.values()), COMPOSITION_AWARE_BINARY_BINS)
            for counts in binary_subset_counts.values()
        ]
        ternary_total_bins = self._ternary_bin_count(COMPOSITION_AWARE_TERNARY_RESOLUTION)
        ternary_occupied = [
            len(counts) / ternary_total_bins
            for counts in ternary_subset_counts.values()
        ]
        ternary_entropy = [
            self._normalized_entropy(list(counts.values()), ternary_total_bins)
            for counts in ternary_subset_counts.values()
        ]

        return {
            "binary_occupied_bin_fraction": self._mean(binary_occupied),
            "binary_normalized_entropy": self._mean(binary_entropy),
            "ternary_occupied_bin_fraction": self._mean(ternary_occupied),
            "ternary_normalized_entropy": self._mean(ternary_entropy),
        }

    @staticmethod
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

    @staticmethod
    def _ternary_bin_count(resolution: int) -> int:
        return (resolution + 1) * (resolution + 2) // 2

    @staticmethod
    def _mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _selected_mean_nearest_distance(
        self,
        descriptors: np.ndarray,
        selected_indices: list[int],
    ) -> float:
        if len(selected_indices) < 2:
            return 0.0
        nearest_distances: list[float] = []
        for pos, idx in enumerate(selected_indices):
            best = math.inf
            for other_pos, other_idx in enumerate(selected_indices):
                if pos == other_pos:
                    continue
                best = min(best, self._descriptor_distance(descriptors, idx, other_idx))
            nearest_distances.append(float(best if best < math.inf else 0.0))
        return self._mean(nearest_distances)

    @staticmethod
    def _composition_aware_attempt_score(attempt_result: dict) -> float:
        return (
            0.35 * attempt_result["binary_occupied_bin_fraction"]
            + 0.20 * attempt_result["binary_normalized_entropy"]
            + 0.30 * attempt_result["ternary_occupied_bin_fraction"]
            + 0.15 * attempt_result["ternary_normalized_entropy"]
        )

    def _pick_best_composition_aware_attempt(
        self,
        attempt_results: list[dict],
        descriptor_floor_fraction: float,
    ) -> dict:
        baseline = attempt_results[0]
        baseline_score = self._composition_aware_attempt_score(baseline)
        best_attempt = baseline
        best_score = baseline_score
        baseline_min_reference = max(
            baseline["train_positive_min_dist"],
            baseline["train_min_dist"],
        )
        min_distance_floor = baseline_min_reference * descriptor_floor_fraction
        mean_nn_floor = baseline["train_mean_nn_dist"] * descriptor_floor_fraction
        logger.info(
            "  Baseline descriptor floors: positive_min_dist>=%.6f, mean_nn>=%.6f",
            min_distance_floor,
            mean_nn_floor,
        )
        logger.info(
            "    Attempt %d accepted as baseline (score=%.3f)",
            baseline["attempt_number"],
            baseline_score,
        )

        for attempt in attempt_results[1:]:
            attempt_score = self._composition_aware_attempt_score(attempt)
            attempt_min_reference = max(
                attempt["train_positive_min_dist"],
                attempt["train_min_dist"],
            )
            eligible = (
                attempt_min_reference >= min_distance_floor
                and attempt["train_mean_nn_dist"] >= mean_nn_floor
            )
            logger.info(
                "    Attempt %d %s descriptor floors (score=%.3f)",
                attempt["attempt_number"],
                "passed" if eligible else "failed",
                attempt_score,
            )
            if not eligible:
                continue

            better = False
            if attempt_score > best_score + 1e-12:
                better = True
            elif math.isclose(attempt_score, best_score, rel_tol=0.0, abs_tol=1e-12):
                if attempt["train_min_dist"] > best_attempt["train_min_dist"] + 1e-12:
                    better = True
                elif math.isclose(
                    attempt["train_min_dist"],
                    best_attempt["train_min_dist"],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    if (
                        attempt["train_mean_nn_dist"]
                        > best_attempt["train_mean_nn_dist"] + 1e-12
                    ):
                        better = True
                    elif math.isclose(
                        attempt["train_mean_nn_dist"],
                        best_attempt["train_mean_nn_dist"],
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        better = (
                            attempt["attempt_number"] < best_attempt["attempt_number"]
                        )
            if better:
                best_attempt = attempt
                best_score = attempt_score
                logger.info(
                    "    Attempt %d became current best",
                    attempt["attempt_number"],
                )

        return best_attempt

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
    def _seed_id(atoms) -> str | None:
        seed_id = atoms.info.get("seed_id")
        if seed_id is None:
            return None
        return str(seed_id)

    @staticmethod
    def _is_elastic_stress(atoms) -> bool:
        return str(atoms.info.get("perturbation_type", "")) == "elastic_stress"

    @staticmethod
    def _is_single_element_structure(atoms) -> bool:
        composition = atoms.info.get("composition")
        if isinstance(composition, dict):
            try:
                positive = [
                    element for element, fraction in composition.items()
                    if float(fraction) > 0.0
                ]
            except (TypeError, ValueError):
                positive = []
            if positive:
                return len(positive) == 1

        try:
            return len(set(atoms.get_chemical_symbols())) == 1
        except Exception:
            return False

    def _is_single_element_elastic_stress(self, atoms) -> bool:
        return self._is_elastic_stress(atoms) and self._is_single_element_structure(atoms)

    def _load_single_element_elastic_stress_indices(self, reference_ase: list) -> list[int]:
        logger.info("")
        logger.info("Step 2c: Loading single-element elastic stress structures")
        elastic_indices = [
            i for i, atoms in enumerate(reference_ase)
            if self._is_single_element_elastic_stress(atoms)
        ]
        if not elastic_indices:
            raise ValueError(
                "Single-element elastic stress structure inclusion enabled, but no "
                "generated structures with perturbation_type=elastic_stress and "
                "single-element composition were found. Run the generate stage with "
                "elastic_stress_enabled=true and check unary seed generation."
            )
        logger.info(
            f"  Matched {len(elastic_indices)} single-element elastic stress structures in generated candidates"
        )
        return elastic_indices

    def _load_elastic_stress_indices(self, reference_ase: list) -> list[int]:
        logger.info("")
        logger.info("Step 2d: Loading elastic stress structures")
        elastic_indices = [
            i for i, atoms in enumerate(reference_ase)
            if self._is_elastic_stress(atoms)
        ]
        if not elastic_indices:
            raise ValueError(
                "Elastic stress structure inclusion enabled, but no generated "
                "structures with perturbation_type=elastic_stress were found. "
                "Run the generate stage with elastic_stress_enabled=true."
            )
        logger.info(
            f"  Matched {len(elastic_indices)} elastic stress structures in generated candidates"
        )
        return elastic_indices

    def _load_seed_indices(self, reference_ase: list) -> list[int]:
        seeds_path = self.project_dir / "structures" / "seeds" / "base_structures.xyz"
        if not seeds_path.exists():
            raise FileNotFoundError(
                f"Seed structures requested, but file not found: {seeds_path}"
            )

        logger.info("")
        logger.info("Step 2b: Loading seed structures")
        seed_ase = ase_read(str(seeds_path), index=":", format="extxyz")
        if not isinstance(seed_ase, list):
            seed_ase = [seed_ase]
        if len(seed_ase) == 0:
            raise ValueError(
                f"Seed inclusion enabled, but no structures were found in {seeds_path}"
            )

        ref_index: dict[str, int] = {}
        missing_generated_metadata = 0
        for i, atoms in enumerate(reference_ase):
            seed_id = self._seed_id(atoms)
            if seed_id is None:
                missing_generated_metadata += 1
                continue
            ref_index[seed_id] = i

        if missing_generated_metadata:
            raise ValueError(
                "Generated structures are missing seed_id metadata. "
                "Regenerate the project with the updated generator before enabling seed inclusion."
            )

        seed_indices: list[int] = []
        missing_seed_metadata = 0
        for atom in seed_ase:
            seed_id = self._seed_id(atom)
            if seed_id is None:
                missing_seed_metadata += 1
                continue
            if seed_id in ref_index:
                seed_indices.append(ref_index[seed_id])

        if missing_seed_metadata:
            raise ValueError(
                "Seed structures are missing seed_id metadata. "
                "Regenerate the project with the updated generator before enabling seed inclusion."
            )

        missing = len(seed_ase) - len(seed_indices)
        if missing:
            raise ValueError(
                f"Seed inclusion enabled, but {missing}/{len(seed_ase)} seed structures "
                f"did not match any generated structure by seed_id"
            )

        seed_indices = sorted(set(seed_indices))
        logger.info(
            f"  Matched {len(seed_indices)}/{len(seed_ase)} seed structures against generated structures"
        )
        return seed_indices

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
