import importlib.util
import json
import sys
import tempfile
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


class FakeCalculator:
    def __init__(self, path):
        self.path = path
        self.batch_sizes = []
        self.mean_descriptor_flags = []

    def get_structures_descriptor(self, batch, mean_descriptor=True):
        self.batch_sizes.append(len(batch))
        self.mean_descriptor_flags.append(mean_descriptor)
        width = 3 if mean_descriptor else 2
        return sys.modules["numpy"].ones((len(batch), width))


def install_numpy_stub():
    numpy_module = types.ModuleType("numpy")
    numpy_module.ndarray = FakeArray
    numpy_module.ones = lambda shape, dtype=None: FakeArray(
        [[1 for _ in range(shape[1])] for _ in range(shape[0])]
    )
    numpy_module.concatenate = lambda arrays, axis=0: FakeArray(
        [row for array in arrays for row in array.data]
    )
    numpy_module.save = lambda path, arr: Path(path).write_text(
        json.dumps(arr.data),
        encoding="utf-8",
    )
    numpy_module.load = lambda path: FakeArray(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )
    sys.modules["numpy"] = numpy_module


def install_test_stubs():
    install_numpy_stub()
    for name in ["common", "NepTrainKit", "NepTrainKit.core"]:
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module

    nep_calc = types.ModuleType("NepTrainKit.core.calculator")
    nep_calc.NepCalculator = FakeCalculator
    sys.modules["NepTrainKit.core.calculator"] = nep_calc


def load_descriptors_module():
    install_test_stubs()
    sys.modules.pop("common.descriptors", None)
    spec = importlib.util.spec_from_file_location(
        "common.descriptors",
        SRC / "common" / "descriptors.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["common.descriptors"] = module
    sys.modules["common"].descriptors = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


DESCRIPTORS = load_descriptors_module()
NP = sys.modules["numpy"]


class DescriptorTests(unittest.TestCase):
    def test_descriptor_cache_path_uses_project_nep_dataset_folder(self) -> None:
        project_dir = Path("demo_project")

        result = DESCRIPTORS.descriptor_cache_path(project_dir)

        self.assertEqual(result, project_dir / "nep" / "datasets" / "descriptors.npy")

    def test_compute_descriptors_batched_splits_batches(self) -> None:
        calc = FakeCalculator("model.txt")
        structures = ["s0", "s1", "s2"]

        descriptors = DESCRIPTORS.compute_descriptors_batched(
            calc,
            structures,
            mean_descriptor=True,
            batch_size=2,
        )

        self.assertEqual(calc.batch_sizes, [2, 1])
        self.assertEqual(descriptors.shape, (3, 3))
        self.assertEqual(calc.mean_descriptor_flags, [True, True])

    def test_compute_descriptors_batched_forwards_mean_descriptor_flag(self) -> None:
        calc = FakeCalculator("model.txt")

        descriptors = DESCRIPTORS.compute_descriptors_batched(
            calc,
            ["s0", "s1"],
            mean_descriptor=False,
            batch_size=1,
        )

        self.assertEqual(calc.mean_descriptor_flags, [False, False])
        self.assertEqual(descriptors.shape, (2, 2))

    def test_load_or_compute_descriptors_uses_cache_when_shape_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            cache_path = project_dir / "nep" / "datasets"
            cache_path.mkdir(parents=True, exist_ok=True)
            NP.save(cache_path / "descriptors.npy", NP.ones((3, 4)))

            with patch.object(DESCRIPTORS, "NepCalculator") as calc_cls:
                descriptors = DESCRIPTORS.load_or_compute_descriptors(
                    project_dir,
                    ["a", "b", "c"],
                    mean_descriptor=True,
                    batch_size=2,
                    nep_model_file="nep89.txt",
                )

            calc_cls.assert_not_called()
            self.assertEqual(descriptors.shape, (3, 4))

    def test_load_or_compute_descriptors_computes_and_saves_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            model_dir = project_dir / "config" / "nep"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "nep89.txt").write_text("stub", encoding="utf-8")

            descriptors = DESCRIPTORS.load_or_compute_descriptors(
                project_dir,
                ["a", "b", "c"],
                mean_descriptor=False,
                batch_size=2,
                nep_model_file="nep89.txt",
            )

            self.assertEqual(descriptors.shape, (3, 2))
            self.assertTrue(
                (project_dir / "nep" / "datasets" / "descriptors.npy").exists()
            )

    def test_load_or_compute_descriptors_recomputes_when_cache_shape_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            cache_path = project_dir / "nep" / "datasets"
            cache_path.mkdir(parents=True, exist_ok=True)
            NP.save(cache_path / "descriptors.npy", NP.ones((2, 9)))
            model_dir = project_dir / "config" / "nep"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "nep89.txt").write_text("stub", encoding="utf-8")

            with patch.object(DESCRIPTORS, "compute_descriptors_batched") as compute:
                compute.return_value = NP.ones((3, 2))
                descriptors = DESCRIPTORS.load_or_compute_descriptors(
                    project_dir,
                    ["a", "b", "c"],
                    mean_descriptor=False,
                    batch_size=2,
                    nep_model_file="nep89.txt",
                )

            compute.assert_called_once()
            self.assertEqual(descriptors.shape, (3, 2))

    def test_load_or_compute_descriptors_raises_when_model_file_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)

            with self.assertRaisesRegex(FileNotFoundError, "NEP model not found"):
                DESCRIPTORS.load_or_compute_descriptors(
                    project_dir,
                    ["a"],
                    mean_descriptor=True,
                    batch_size=2,
                    nep_model_file="nep89.txt",
                )


if __name__ == "__main__":
    unittest.main()
