import importlib
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np


def _import_fps_module():
    """Import FPS, mocking only the optional NepTrainKit sampling boundary if absent."""
    try:
        return importlib.import_module("common.FPS")
    except ModuleNotFoundError as exc:
        if not str(exc.name).startswith("NepTrainKit"):
            raise

        io_module = types.ModuleType("NepTrainKit.core.io")
        io_module.farthest_point_sampling = lambda descriptors, n_samples, min_dist: list(
            range(min(len(descriptors), n_samples))
        )
        core = types.ModuleType("NepTrainKit.core")
        core.__path__ = []
        package = types.ModuleType("NepTrainKit")
        package.__path__ = []
        modules = {
            "NepTrainKit": package,
            "NepTrainKit.core": core,
            "NepTrainKit.core.io": io_module,
        }
        with patch.dict(sys.modules, modules, clear=False):
            return importlib.import_module("common.FPS")


FPS = _import_fps_module()


class StructureStub:
    def __init__(self, num_atoms=1):
        self.num_atoms = num_atoms


class FPSTests(unittest.TestCase):
    def test_fps_run_returns_sorted_frame_indices_for_mean_descriptors(self) -> None:
        descriptors = np.ones((3, 2))
        structures = [StructureStub(), StructureStub(), StructureStub()]

        result = FPS.fps_run(descriptors, structures, True, 0.1)

        self.assertEqual(result, [0, 1, 2])

    def test_fps_run_maps_atomic_descriptor_indices_back_to_frames(self) -> None:
        descriptors = np.ones((5, 2))
        structures = [StructureStub(2), StructureStub(3)]

        with patch.object(FPS, "farthest_point_sampling", return_value=[0, 1, 2, 4]):
            result = FPS.fps_run(descriptors, structures, False, 0.2)

        self.assertEqual(result, [0, 1])

    def test_fps_count_uses_fps_run_length(self) -> None:
        descriptors = np.ones((4, 2))
        structures = [StructureStub(), StructureStub()]

        with patch.object(FPS, "fps_run", return_value=[0, 2, 3]):
            count = FPS.fps_count(descriptors, structures, True, 0.1)

        self.assertEqual(count, 3)

    def test_fps_target_count_converges_on_first_iteration_when_already_within_tolerance(self) -> None:
        descriptors = np.ones((2, 2))
        structures = [StructureStub(), StructureStub()]

        with patch.object(FPS, "fps_count", return_value=2):
            with patch.object(FPS, "fps_run", return_value=[0, 1]) as fps_run:
                indices, best_dist = FPS.fps_target_count(
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
        self.assertEqual(
            fps_run.call_args_list[-1].args,
            (descriptors, structures, True, 0.005),
        )

    def test_fps_target_count_runs_binary_search_until_within_tolerance(self) -> None:
        descriptors = np.ones((5, 2))
        structures = [StructureStub() for _ in range(5)]
        count_values = [5, 4, 3, 3]

        with patch.object(FPS, "fps_count", side_effect=count_values):
            with patch.object(FPS, "fps_run", return_value=[0, 1, 2]) as fps_run:
                indices, best_dist = FPS.fps_target_count(
                    descriptors,
                    structures,
                    True,
                    target=3,
                    tolerance=0,
                    max_iterations=3,
                    label="train",
                )

        self.assertEqual(indices, [0, 1, 2])
        self.assertAlmostEqual(best_dist, 0.02, places=12)
        self.assertGreaterEqual(fps_run.call_count, 2)

    def test_cross_distance_stats_handles_empty_inputs(self) -> None:
        descriptors = np.ones((2, 2))

        result = FPS.cross_distance_stats(descriptors, [], [0])

        self.assertEqual(result, (float("inf"), float("inf")))

    def test_cross_distance_stats_returns_min_and_mean(self) -> None:
        descriptors = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ])

        min_dist, mean_dist = FPS.cross_distance_stats(descriptors, [0], [1, 2])

        self.assertAlmostEqual(min_dist, 2.0, places=12)
        self.assertAlmostEqual(mean_dist, 2.5, places=12)


if __name__ == "__main__":
    unittest.main()
