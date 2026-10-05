import unittest
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("NepTrainKit")

from nepflow.stages.selection import sampling as SAMPLING  # noqa: E402


class StructureStub:
    def __init__(self, num_atoms=1):
        self.num_atoms = num_atoms


class FPSTests(unittest.TestCase):
    def test_select_farthest_points_returns_sorted_frame_indices_for_mean_descriptors(self) -> None:
        descriptors = np.ones((3, 2))
        structures = [StructureStub(), StructureStub(), StructureStub()]

        with patch(
            "NepTrainKit.core.io.farthest_point_sampling",
            return_value=[2, 0, 1],
        ) as sampler:
            result = SAMPLING.select_farthest_points(descriptors, structures, True, 0.1)

        sampler.assert_called_once_with(descriptors, n_samples=3, min_dist=0.1)
        self.assertEqual(result, [0, 1, 2])

    def test_select_farthest_points_maps_atomic_rows_back_to_frames(self) -> None:
        descriptors = np.ones((5, 2))
        structures = [StructureStub(2), StructureStub(3)]

        with patch(
            "NepTrainKit.core.io.farthest_point_sampling",
            return_value=[0, 1, 2, 4],
        ) as sampler:
            result = SAMPLING.select_farthest_points(descriptors, structures, False, 0.2)

        sampler.assert_called_once_with(descriptors, n_samples=2, min_dist=0.2)
        self.assertEqual(result, [0, 1])

    def test_target_count_converges_on_first_iteration_when_already_within_tolerance(self) -> None:
        descriptors = np.array([[0.0, 0.0], [1.0, 0.0]])
        structures = [StructureStub(), StructureStub()]

        indices, best_dist = SAMPLING.select_farthest_points_for_target(
            descriptors,
            structures,
            True,
            target=2,
            tolerance=0,
            max_iterations=3,
            label="train",
        )

        self.assertEqual(indices, [0, 1])
        self.assertAlmostEqual(best_dist, 0.005, places=12)

    def test_target_count_runs_binary_search_until_within_tolerance(self) -> None:
        descriptors = np.arange(5, dtype=float).reshape(5, 1)
        structures = [StructureStub() for _ in range(5)]

        indices, best_dist = SAMPLING.select_farthest_points_for_target(
            descriptors,
            structures,
            True,
            target=3,
            tolerance=0,
            max_iterations=12,
            label="train",
        )

        self.assertEqual(len(indices), 3)
        self.assertEqual(indices, sorted(set(indices)))
        self.assertGreater(best_dist, 0.0)

    def test_cross_distance_handles_empty_inputs(self) -> None:
        descriptors = np.ones((2, 2))

        result = SAMPLING.calculate_cross_distance_stats(descriptors, [], [0])

        self.assertEqual(result, (float("inf"), float("inf")))

    def test_cross_distance_returns_min_and_mean(self) -> None:
        descriptors = np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ]
        )

        min_dist, mean_dist = SAMPLING.calculate_cross_distance_stats(descriptors, [0], [1, 2])

        self.assertAlmostEqual(min_dist, 2.0, places=12)
        self.assertAlmostEqual(mean_dist, 2.5, places=12)


if __name__ == "__main__":
    unittest.main()
