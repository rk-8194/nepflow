import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


class FakeArray:
    def __init__(self, data):
        self.data = data

    @property
    def shape(self):
        if not self.data:
            return (0,)
        first = self.data[0]
        if isinstance(first, list):
            return (len(self.data), len(first))
        return (len(self.data),)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if isinstance(item, list):
            return FakeArray([self.data[i] for i in item])
        return self.data[item]


def install_numpy_and_scipy_stubs():
    numpy_module = types.ModuleType("numpy")
    numpy_module.ndarray = FakeArray
    numpy_module.ones = lambda shape: FakeArray(
        [[1 for _ in range(shape[1])] for _ in range(shape[0])]
    )
    numpy_module.array = lambda data: FakeArray(data)
    sys.modules["numpy"] = numpy_module

    scipy_module = types.ModuleType("scipy")
    scipy_module.__path__ = []
    sys.modules["scipy"] = scipy_module
    scipy_spatial = types.ModuleType("scipy.spatial")
    scipy_spatial.__path__ = []
    sys.modules["scipy.spatial"] = scipy_spatial
    scipy_distance = types.ModuleType("scipy.spatial.distance")

    def cdist(a, b):
        rows = []
        for row_a in a.data:
            rows.append(
                [
                    sum((xa - xb) ** 2 for xa, xb in zip(row_a, row_b)) ** 0.5
                    for row_b in b.data
                ]
            )
        return FakeDistance(rows)

    scipy_distance.cdist = cdist
    sys.modules["scipy.spatial.distance"] = scipy_distance


class FakeDistance:
    def __init__(self, data):
        self.data = data

    def min(self, axis):
        if axis != 1:
            raise ValueError("Only axis=1 is supported in test stub")
        return FakeVector([min(row) for row in self.data])


class FakeVector(list):
    def min(self):
        return min(list(self))

    def mean(self):
        values = list(self)
        return sum(values) / len(values)


class FakeStructure:
    def __init__(self, num_atoms=1):
        self.num_atoms = num_atoms


def install_test_stubs():
    install_numpy_and_scipy_stubs()
    package_names = [
        "common",
        "NepTrainKit",
        "NepTrainKit.core",
    ]
    for name in package_names:
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module

    nep_io = types.ModuleType("NepTrainKit.core.io")
    nep_io.farthest_point_sampling = lambda descriptors, n_samples, min_dist: list(
        range(min(len(descriptors), n_samples))
    )
    sys.modules["NepTrainKit.core.io"] = nep_io


def load_fps_module():
    install_test_stubs()
    sys.modules.pop("common.FPS", None)
    spec = importlib.util.spec_from_file_location(
        "common.FPS",
        SRC / "common" / "FPS.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["common.FPS"] = module
    sys.modules["common"].FPS = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


FPS = load_fps_module()
NP = sys.modules["numpy"]


class FPSTests(unittest.TestCase):
    def test_fps_run_returns_sorted_frame_indices_for_mean_descriptors(self) -> None:
        descriptors = NP.ones((3, 2))
        structures = [FakeStructure(), FakeStructure(), FakeStructure()]

        result = FPS.fps_run(descriptors, structures, True, 0.1)

        self.assertEqual(result, [0, 1, 2])

    def test_fps_run_maps_atomic_descriptor_indices_back_to_frames(self) -> None:
        descriptors = NP.ones((5, 2))
        structures = [FakeStructure(2), FakeStructure(3)]

        with patch.object(FPS, "farthest_point_sampling", return_value=[0, 1, 2, 4]):
            result = FPS.fps_run(descriptors, structures, False, 0.2)

        self.assertEqual(result, [0, 1])

    def test_fps_count_uses_fps_run_length(self) -> None:
        descriptors = NP.ones((4, 2))
        structures = [FakeStructure(), FakeStructure()]

        with patch.object(FPS, "fps_run", return_value=[0, 2, 3]):
            count = FPS.fps_count(descriptors, structures, True, 0.1)

        self.assertEqual(count, 3)

    def test_fps_target_count_converges_on_first_iteration_when_already_within_tolerance(self) -> None:
        descriptors = NP.ones((2, 2))
        structures = [FakeStructure(), FakeStructure()]

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
        self.assertAlmostEqual(best_dist, 0.005)
        self.assertEqual(
            fps_run.call_args_list[-1].args,
            (descriptors, structures, True, 0.005),
        )

    def test_fps_target_count_runs_binary_search_until_within_tolerance(self) -> None:
        descriptors = NP.ones((5, 2))
        structures = [FakeStructure() for _ in range(5)]
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
        self.assertAlmostEqual(best_dist, 0.02)
        self.assertGreaterEqual(fps_run.call_count, 2)

    def test_cross_distance_stats_handles_empty_inputs(self) -> None:
        descriptors = NP.ones((2, 2))

        result = FPS.cross_distance_stats(descriptors, [], [0])

        self.assertEqual(result, (float("inf"), float("inf")))

    def test_cross_distance_stats_returns_min_and_mean(self) -> None:
        descriptors = NP.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ]
        )

        min_dist, mean_dist = FPS.cross_distance_stats(descriptors, [0], [1, 2])

        self.assertEqual(min_dist, 2.0)
        self.assertEqual(mean_dist, 2.5)


if __name__ == "__main__":
    unittest.main()
