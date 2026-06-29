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
        if not isinstance(self.data, list):
            return ()
        if not self.data:
            return (0,)
        first = self.data[0]
        if isinstance(first, list):
            return (len(self.data), len(first))
        return (len(self.data),)

    def astype(self, _dtype):
        return self

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if isinstance(item, list):
            return FakeArray([self.data[i] for i in item])
        return self.data[item]


class FakeRandomState:
    def __init__(self, seed):
        self.seed = seed

    def randn(self, rows, cols):
        return FakeArray([[0.0 for _ in range(cols)] for _ in range(rows)])

    def permutation(self, n):
        return FakePermutation(range(n))


class FakePermutation(list):
    def __getitem__(self, item):
        result = super().__getitem__(item)
        if isinstance(item, slice):
            return FakePermutation(result)
        return result

    def tolist(self):
        return list(self)


def install_numpy_stub():
    numpy_module = types.ModuleType("numpy")
    numpy_module.float64 = float
    numpy_module.ndarray = FakeArray
    numpy_module.ones = lambda shape, dtype=None: FakeArray(
        [[1 for _ in range(shape[1])] for _ in range(shape[0])]
    )
    numpy_module.save = lambda path, arr: Path(path).write_text(
        json.dumps(arr.data), encoding="utf-8"
    )
    numpy_module.load = lambda path: FakeArray(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )
    numpy_module.random = types.SimpleNamespace(RandomState=FakeRandomState)
    sys.modules["numpy"] = numpy_module


class FakeStructure:
    def __init__(self, num_atoms=1):
        self.num_atoms = num_atoms


class FakeStructureClass:
    @staticmethod
    def read_multiple(path):
        return [FakeStructure(), FakeStructure(), FakeStructure()]


class FakeCalculator:
    def __init__(self, path):
        self.path = path

    def get_structures_descriptor(self, batch, mean_descriptor=True):
        width = 3 if mean_descriptor else 2
        return sys.modules["numpy"].ones((len(batch), width))


def fake_fps(descriptors, n_samples, min_dist):
    return list(range(min(len(descriptors), n_samples)))


def install_test_stubs():
    install_numpy_stub()
    package_names = [
        "testpkg",
        "testpkg.modules",
        "testpkg.modules.select",
        "common",
    ]
    for name in package_names:
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module

    ase_module = types.ModuleType("ase")
    sys.modules["ase"] = ase_module

    ase_io = types.ModuleType("ase.io")
    ase_io.read = lambda *args, **kwargs: ["ase0", "ase1", "ase2"]
    ase_io.write = lambda *args, **kwargs: None
    sys.modules["ase.io"] = ase_io

    nep_pkg = types.ModuleType("NepTrainKit")
    nep_pkg.__path__ = []
    sys.modules["NepTrainKit"] = nep_pkg
    nep_core = types.ModuleType("NepTrainKit.core")
    nep_core.__path__ = []
    sys.modules["NepTrainKit.core"] = nep_core

    nep_calc = types.ModuleType("NepTrainKit.core.calculator")
    nep_calc.NepCalculator = FakeCalculator
    sys.modules["NepTrainKit.core.calculator"] = nep_calc

    nep_struct = types.ModuleType("NepTrainKit.core.structure")
    nep_struct.Structure = FakeStructureClass
    sys.modules["NepTrainKit.core.structure"] = nep_struct

    nep_io = types.ModuleType("NepTrainKit.core.io")
    nep_io.farthest_point_sampling = fake_fps
    sys.modules["NepTrainKit.core.io"] = nep_io

    descriptor_spec = importlib.util.spec_from_file_location(
        "common.descriptors",
        SRC / "common" / "descriptors.py",
    )
    descriptor_module = importlib.util.module_from_spec(descriptor_spec)
    sys.modules["common.descriptors"] = descriptor_module
    sys.modules["common"].descriptors = descriptor_module
    assert descriptor_spec.loader is not None
    descriptor_spec.loader.exec_module(descriptor_module)

    fps_module = types.ModuleType("common.FPS")
    fps_module.fps_target_count = (
        lambda descriptors, structures, mean_descriptor, target, tolerance, max_iterations, label="":
        (list(range(min(target, len(structures)))), 0.0)
    )
    fps_module.cross_distance_stats = (
        lambda descriptors, a, b:
        (float("inf"), float("inf")) if not a or not b else (0.0, 0.0)
    )
    sys.modules["common.FPS"] = fps_module
    sys.modules["common"].FPS = fps_module

    base_spec = importlib.util.spec_from_file_location(
        "testpkg.modules.base",
        SRC / "modules" / "base.py",
    )
    base_module = importlib.util.module_from_spec(base_spec)
    sys.modules["testpkg.modules.base"] = base_module
    assert base_spec.loader is not None
    base_spec.loader.exec_module(base_module)


def load_select_stage():
    install_test_stubs()
    spec = importlib.util.spec_from_file_location(
        "testpkg.modules.select.select",
        SRC / "modules" / "select" / "select.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["testpkg.modules.select.select"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.SelectStage


SelectStage = load_select_stage()
NP = sys.modules["numpy"]
DESCRIPTORS = sys.modules["common.descriptors"]


def write_project_config(project_dir: Path) -> None:
    config_dir = project_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "project.config").write_text(
        "\n".join(
            [
                "[selection]",
                "descriptor_type=structure",
                "batch_size=2",
                "target_train_count=2",
                "target_test_count=1",
                "target_tolerance=1",
                "max_search_iterations=4",
                "test_pool_factor=1.0",
                "nep_model_file=nep89.txt",
            ]
        ),
        encoding="utf-8",
    )


class SelectStageTests(unittest.TestCase):
    def create_stage(self, project_dir: Path) -> SelectStage:
        return SelectStage(
            project_name="demo",
            config_file=project_dir / "config" / "demo.yaml",
            state_file=project_dir / "state.db",
            project_dir=project_dir,
            debug=False,
        )

    def test_prepare_raises_when_generated_structures_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            stage = self.create_stage(project_dir)

            with self.assertRaisesRegex(FileNotFoundError, "Run the 'generate' stage first."):
                stage.prepare()

    def test_prepare_returns_none_for_empty_structures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            generated = project_dir / "structures" / "generated"
            generated.mkdir(parents=True, exist_ok=True)
            (generated / "generated_structures.xyz").write_text("stub", encoding="utf-8")
            stage = self.create_stage(project_dir)

            with patch("testpkg.modules.select.select.Structure.read_multiple", return_value=[]):
                with patch("testpkg.modules.select.select.ase_read", return_value=[]):
                    prepared = stage.prepare()

            self.assertIsNone(prepared)

    def test_run_executes_and_finalizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(project_dir)
            stage = self.create_stage(project_dir)
            prepared = {"structures": [FakeStructure()], "ase_structures": ["ase0"]}
            result = {
                "descriptors": NP.ones((1, 3)),
                "train_indices": [0],
                "train_min_dist": 0.0,
                "test_indices": [],
                "test_min_dist": 0.0,
                "min_train_test_dist": float("inf"),
                "mean_train_test_dist": float("inf"),
            }

            with patch.object(stage, "prepare", return_value=prepared) as prepare:
                with patch.object(stage, "execute", return_value=result) as execute:
                    with patch.object(stage, "finalize") as finalize:
                        stage.run()

            prepare.assert_called_once()
            execute.assert_called_once()
            finalize.assert_called_once_with(prepared, result)

    def test_load_or_compute_descriptors_uses_cache_when_shape_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(project_dir)
            stage = self.create_stage(project_dir)
            cache_path = project_dir / "nep" / "datasets"
            cache_path.mkdir(parents=True, exist_ok=True)
            NP.save(cache_path / "descriptors.npy", NP.ones((3, 4)))
            structures = [FakeStructure(), FakeStructure(), FakeStructure()]

            with patch.object(DESCRIPTORS, "NepCalculator") as calc_cls:
                descriptors = DESCRIPTORS.load_or_compute_descriptors(
                    project_dir,
                    structures,
                    mean_descriptor=True,
                    batch_size=2,
                    nep_model_file="nep89.txt",
                )

            calc_cls.assert_not_called()
            self.assertEqual(descriptors.shape, (3, 4))

    def test_execute_uses_shared_descriptor_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(project_dir)
            stage = self.create_stage(project_dir)
            config, settings = stage.load_config()
            prepared = {
                "structures": [FakeStructure(), FakeStructure()],
                "ase_structures": ["ase0", "ase1"],
            }
            descriptors = NP.ones((2, 3))

            with patch(
                "testpkg.modules.select.select.load_or_compute_descriptors",
                return_value=descriptors,
            ) as loader:
                with patch.object(
                    stage,
                    "_select_training_set",
                    return_value=([0], 0.1),
                ):
                    with patch.object(
                        stage,
                        "_select_test_set",
                        return_value={
                            "test_indices": [1],
                            "test_min_dist": 0.2,
                            "min_train_test_dist": 0.3,
                            "mean_train_test_dist": 0.4,
                        },
                    ):
                        result = stage.execute(config, settings, prepared)

            loader.assert_called_once_with(
                project_dir,
                prepared["structures"],
                mean_descriptor=True,
                batch_size=2,
                nep_model_file="nep89.txt",
            )
            self.assertEqual(result["descriptors"], descriptors)

    def test_debug_run_saves_split_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            generated = project_dir / "structures" / "generated"
            generated.mkdir(parents=True, exist_ok=True)
            (generated / "generated_structures.xyz").write_text("stub", encoding="utf-8")
            stage = SelectStage(
                project_name="demo",
                config_file=project_dir / "config" / "demo.yaml",
                state_file=project_dir / "state.db",
                project_dir=project_dir,
                debug=True,
            )

            with patch.object(stage, "_save_selected_structures") as save_selected:
                stage.run()

            save_selected.assert_called_once()


if __name__ == "__main__":
    unittest.main()
