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


class FakeDistance:
    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)


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


class FakePositions:
    def __init__(self, coords):
        self.coords = coords

    def round(self, _ndigits):
        return self

    def tobytes(self):
        return repr(self.coords).encode("utf-8")


class FakeAtoms:
    def __init__(self, formula, coords):
        self._formula = formula
        self._positions = FakePositions(coords)
        self.info = {}

    def get_positions(self):
        return self._positions

    def get_chemical_formula(self):
        return self._formula

    def get_chemical_symbols(self):
        return list(self._formula)


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


def write_project_config(
    project_dir: Path,
    *,
    include_seed_structures: bool = False,
    include_single_element_elastic_stress_structures: bool = False,
    include_elastic_stress_structures: bool = False,
    composition_aware_fps: bool = False,
    target_train_count: int = 2,
    composition_aware_fps_frontier_fraction: float = 0.10,
    composition_aware_fps_ternary_weight: float = 1.0,
    composition_aware_fps_adaptive_retries: int = 4,
    composition_aware_fps_descriptor_floor_fraction: float = 0.95,
) -> None:
    config_dir = project_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    seed_line = f"include_seed_structures={'true' if include_seed_structures else 'false'}"
    single_elastic_line = (
        "include_single_element_elastic_stress_structures="
        f"{'true' if include_single_element_elastic_stress_structures else 'false'}"
    )
    elastic_line = (
        "include_elastic_stress_structures="
        f"{'true' if include_elastic_stress_structures else 'false'}"
    )
    composition_aware_line = (
        "composition_aware_fps="
        f"{'true' if composition_aware_fps else 'false'}"
    )
    (config_dir / "project.config").write_text(
        "\n".join(
            [
                "[selection]",
                "descriptor_type=structure",
                "batch_size=2",
                f"target_train_count={target_train_count}",
                "target_test_count=1",
                "target_tolerance=1",
                "max_search_iterations=4",
                "test_pool_factor=1.0",
                "nep_model_file=nep89.txt",
                seed_line,
                single_elastic_line,
                elastic_line,
                composition_aware_line,
                f"composition_aware_fps_frontier_fraction={composition_aware_fps_frontier_fraction}",
                f"composition_aware_fps_ternary_weight={composition_aware_fps_ternary_weight}",
                f"composition_aware_fps_adaptive_retries={composition_aware_fps_adaptive_retries}",
                "composition_aware_fps_descriptor_floor_fraction="
                f"{composition_aware_fps_descriptor_floor_fraction}",
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

    def test_execute_includes_seed_structures_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(project_dir, include_seed_structures=True, target_train_count=3)
            seed_file = project_dir / "structures" / "seeds" / "base_structures.xyz"
            seed_file.parent.mkdir(parents=True, exist_ok=True)
            seed_file.write_text("stub", encoding="utf-8")
            stage = self.create_stage(project_dir)
            config, settings = stage.load_config()
            generated_atoms = [
                FakeAtoms("A", [[0.0, 0.0, 0.0]]),
                FakeAtoms("B", [[1.0, 0.0, 0.0]]),
                FakeAtoms("C", [[2.0, 0.0, 0.0]]),
                FakeAtoms("D", [[3.0, 0.0, 0.0]]),
            ]
            seed_atoms = [
                FakeAtoms("A", [[0.0, 0.0, 0.0]]),
                FakeAtoms("C", [[2.0, 0.0, 0.0]]),
            ]
            generated_atoms[0].info["seed_id"] = "seed_000000"
            generated_atoms[1].info["seed_id"] = "seed_000001"
            generated_atoms[2].info["seed_id"] = "seed_000002"
            generated_atoms[3].info["seed_id"] = "seed_000003"
            seed_atoms[0].info["seed_id"] = "seed_000000"
            seed_atoms[1].info["seed_id"] = "seed_000002"
            prepared = {
                "structures": [FakeStructure() for _ in range(4)],
                "ase_structures": generated_atoms,
            }
            descriptors = NP.ones((4, 3))

            def fake_ase_read(path, *args, **kwargs):
                if "base_structures.xyz" in str(path):
                    return seed_atoms
                return generated_atoms

            with patch(
                "testpkg.modules.select.select.load_or_compute_descriptors",
                return_value=descriptors,
            ):
                with patch("testpkg.modules.select.select.ase_read", side_effect=fake_ase_read):
                    with patch.object(
                        stage,
                        "_select_test_set",
                        return_value={
                            "test_indices": [],
                            "test_min_dist": 0.0,
                            "min_train_test_dist": float("inf"),
                            "mean_train_test_dist": float("inf"),
                        },
                    ):
                        result = stage.execute(config, settings, prepared)

            self.assertEqual(result["seed_indices"], [0, 2])
            self.assertEqual(result["train_seed_count"], 2)
            self.assertEqual(result["train_fps_count"], 1)
            self.assertEqual(result["train_indices"], [0, 1, 2])

    def test_execute_skips_seed_loading_when_option_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(project_dir, include_seed_structures=False)
            stage = self.create_stage(project_dir)
            config, settings = stage.load_config()
            prepared = {
                "structures": [FakeStructure(), FakeStructure()],
                "ase_structures": [
                    FakeAtoms("A", [[0.0, 0.0, 0.0]]),
                    FakeAtoms("B", [[1.0, 0.0, 0.0]]),
                ],
            }
            descriptors = NP.ones((2, 3))

            with patch(
                "testpkg.modules.select.select.load_or_compute_descriptors",
                return_value=descriptors,
            ):
                with patch.object(stage, "_load_seed_indices") as load_seeds:
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

            load_seeds.assert_not_called()
            self.assertEqual(result["train_indices"], [0])
            self.assertEqual(result["train_seed_count"], 0)
            self.assertEqual(result["train_fps_count"], 1)

    def test_execute_skips_elastic_loading_when_option_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(project_dir, include_elastic_stress_structures=False)
            stage = self.create_stage(project_dir)
            config, settings = stage.load_config()
            prepared = {
                "structures": [FakeStructure(), FakeStructure()],
                "ase_structures": [
                    FakeAtoms("A", [[0.0, 0.0, 0.0]]),
                    FakeAtoms("B", [[1.0, 0.0, 0.0]]),
                ],
            }
            descriptors = NP.ones((2, 3))

            with patch(
                "testpkg.modules.select.select.load_or_compute_descriptors",
                return_value=descriptors,
            ):
                with patch.object(stage, "_load_elastic_stress_indices") as load_elastic:
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

            load_elastic.assert_not_called()
            self.assertEqual(result["train_elastic_count"], 0)
            self.assertEqual(result["train_fps_count"], 1)

    def test_load_config_defaults_single_element_elastic_inclusion_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(project_dir)
            stage = self.create_stage(project_dir)

            _, settings = stage.load_config()

            self.assertFalse(settings["include_single_element_elastic_stress_structures"])

    def test_load_config_defaults_composition_aware_fps_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(project_dir)
            stage = self.create_stage(project_dir)

            _, settings = stage.load_config()

            self.assertFalse(settings["composition_aware_fps"])

    def test_load_config_reads_composition_aware_fps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(project_dir, composition_aware_fps=True)
            stage = self.create_stage(project_dir)

            _, settings = stage.load_config()

            self.assertTrue(settings["composition_aware_fps"])

    def test_load_config_reads_composition_aware_fps_retry_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(
                project_dir,
                composition_aware_fps_frontier_fraction=0.2,
                composition_aware_fps_ternary_weight=1.5,
                composition_aware_fps_adaptive_retries=3,
                composition_aware_fps_descriptor_floor_fraction=0.9,
            )
            stage = self.create_stage(project_dir)

            _, settings = stage.load_config()

            self.assertEqual(settings["composition_aware_fps_frontier_fraction"], 0.2)
            self.assertEqual(settings["composition_aware_fps_ternary_weight"], 1.5)
            self.assertEqual(settings["composition_aware_fps_adaptive_retries"], 3)
            self.assertEqual(
                settings["composition_aware_fps_descriptor_floor_fraction"],
                0.9,
            )

    def test_load_config_rejects_invalid_composition_aware_retry_settings(self) -> None:
        invalid_cases = [
            (
                {"composition_aware_fps_frontier_fraction": 0.0},
                "frontier_fraction must be in",
            ),
            (
                {"composition_aware_fps_ternary_weight": -1.0},
                "ternary_weight must be >=",
            ),
            (
                {"composition_aware_fps_adaptive_retries": 0},
                "adaptive_retries must be >=",
            ),
            (
                {"composition_aware_fps_descriptor_floor_fraction": 1.5},
                "descriptor_floor_fraction must be in",
            ),
        ]
        for kwargs, message in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with tempfile.TemporaryDirectory() as tmp:
                    project_dir = Path(tmp)
                    write_project_config(project_dir, **kwargs)
                    stage = self.create_stage(project_dir)

                    with self.assertRaisesRegex(ValueError, message):
                        stage.load_config()

    def test_execute_skips_single_element_elastic_loading_when_option_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(
                project_dir,
                include_single_element_elastic_stress_structures=False,
            )
            stage = self.create_stage(project_dir)
            config, settings = stage.load_config()
            prepared = {
                "structures": [FakeStructure(), FakeStructure()],
                "ase_structures": [
                    FakeAtoms("A", [[0.0, 0.0, 0.0]]),
                    FakeAtoms("B", [[1.0, 0.0, 0.0]]),
                ],
            }
            descriptors = NP.ones((2, 3))

            with patch(
                "testpkg.modules.select.select.load_or_compute_descriptors",
                return_value=descriptors,
            ):
                with patch.object(
                    stage,
                    "_load_single_element_elastic_stress_indices",
                ) as load_single_elastic:
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

            load_single_elastic.assert_not_called()
            self.assertEqual(result["train_single_element_elastic_count"], 0)
            self.assertEqual(result["train_fps_count"], 1)

    def test_execute_includes_only_single_element_elastic_stress_structures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(
                project_dir,
                include_single_element_elastic_stress_structures=True,
                target_train_count=4,
            )
            stage = self.create_stage(project_dir)
            config, settings = stage.load_config()
            generated_atoms = [
                FakeAtoms("A", [[0.0, 0.0, 0.0]]),
                FakeAtoms("B", [[1.0, 0.0, 0.0]]),
                FakeAtoms("C", [[2.0, 0.0, 0.0]]),
                FakeAtoms("D", [[3.0, 0.0, 0.0]]),
                FakeAtoms("E", [[4.0, 0.0, 0.0]]),
            ]
            generated_atoms[1].info.update({
                "perturbation_type": "elastic_stress",
                "composition": {"W": 1.0},
            })
            generated_atoms[2].info.update({
                "perturbation_type": "elastic_stress",
                "composition": {"W": 0.5, "Cr": 0.5},
            })
            generated_atoms[3].info.update({
                "perturbation_type": "elastic_stress",
                "composition": {"Y": 1.0, "Zr": 0.0},
            })
            generated_atoms[4].info["composition"] = {"W": 1.0}
            prepared = {
                "structures": [FakeStructure() for _ in range(5)],
                "ase_structures": generated_atoms,
            }
            descriptors = NP.ones((5, 3))

            with patch(
                "testpkg.modules.select.select.load_or_compute_descriptors",
                return_value=descriptors,
            ):
                with patch.object(
                    stage,
                    "_select_test_set",
                    return_value={
                        "test_indices": [],
                        "test_min_dist": 0.0,
                        "min_train_test_dist": float("inf"),
                        "mean_train_test_dist": float("inf"),
                    },
                ):
                    result = stage.execute(config, settings, prepared)

            self.assertEqual(result["single_element_elastic_indices"], [1, 3])
            self.assertEqual(result["train_single_element_elastic_count"], 2)
            self.assertEqual(result["train_elastic_count"], 0)
            self.assertEqual(result["train_anchor_count"], 2)
            self.assertEqual(result["train_fps_count"], 2)
            self.assertEqual(result["train_indices"], [0, 1, 2, 3])

    def test_single_element_elastic_detection_falls_back_to_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(
                project_dir,
                include_single_element_elastic_stress_structures=True,
                target_train_count=3,
            )
            stage = self.create_stage(project_dir)
            config, settings = stage.load_config()
            generated_atoms = [
                FakeAtoms("A", [[0.0, 0.0, 0.0]]),
                FakeAtoms("AA", [[1.0, 0.0, 0.0]]),
                FakeAtoms("AB", [[2.0, 0.0, 0.0]]),
            ]
            generated_atoms[1].info.update({
                "perturbation_type": "elastic_stress",
                "composition": "not-a-dict",
            })
            generated_atoms[2].info["perturbation_type"] = "elastic_stress"
            prepared = {
                "structures": [FakeStructure() for _ in range(3)],
                "ase_structures": generated_atoms,
            }
            descriptors = NP.ones((3, 3))

            with patch(
                "testpkg.modules.select.select.load_or_compute_descriptors",
                return_value=descriptors,
            ):
                with patch.object(
                    stage,
                    "_select_test_set",
                    return_value={
                        "test_indices": [],
                        "test_min_dist": 0.0,
                        "min_train_test_dist": float("inf"),
                        "mean_train_test_dist": float("inf"),
                    },
                ):
                    result = stage.execute(config, settings, prepared)

            self.assertEqual(result["single_element_elastic_indices"], [1])
            self.assertEqual(result["train_single_element_elastic_count"], 1)

    def test_single_element_elastic_loading_errors_when_none_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            stage = self.create_stage(project_dir)
            atoms = [
                FakeAtoms("A", [[0.0, 0.0, 0.0]]),
                FakeAtoms("AB", [[1.0, 0.0, 0.0]]),
            ]
            atoms[1].info.update({
                "perturbation_type": "elastic_stress",
                "composition": {"W": 0.5, "Cr": 0.5},
            })

            with self.assertRaisesRegex(
                ValueError,
                "elastic_stress_enabled=true.*unary seed generation",
            ):
                stage._load_single_element_elastic_stress_indices(atoms)

    def test_execute_includes_elastic_stress_structures_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(
                project_dir,
                include_elastic_stress_structures=True,
                target_train_count=3,
            )
            stage = self.create_stage(project_dir)
            config, settings = stage.load_config()
            generated_atoms = [
                FakeAtoms("A", [[0.0, 0.0, 0.0]]),
                FakeAtoms("B", [[1.0, 0.0, 0.0]]),
                FakeAtoms("C", [[2.0, 0.0, 0.0]]),
                FakeAtoms("D", [[3.0, 0.0, 0.0]]),
            ]
            generated_atoms[1].info["perturbation_type"] = "elastic_stress"
            generated_atoms[3].info["perturbation_type"] = "elastic_stress"
            prepared = {
                "structures": [FakeStructure() for _ in range(4)],
                "ase_structures": generated_atoms,
            }
            descriptors = NP.ones((4, 3))

            with patch(
                "testpkg.modules.select.select.load_or_compute_descriptors",
                return_value=descriptors,
            ):
                with patch.object(
                    stage,
                    "_select_test_set",
                    return_value={
                        "test_indices": [],
                        "test_min_dist": 0.0,
                        "min_train_test_dist": float("inf"),
                        "mean_train_test_dist": float("inf"),
                    },
                ):
                    result = stage.execute(config, settings, prepared)

            self.assertEqual(result["elastic_indices"], [1, 3])
            self.assertEqual(result["train_elastic_count"], 2)
            self.assertEqual(result["train_fps_count"], 1)
            self.assertEqual(result["train_indices"], [0, 1, 3])

    def test_select_training_set_deduplicates_seed_and_elastic_anchors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(project_dir, target_train_count=3)
            stage = self.create_stage(project_dir)
            _, settings = stage.load_config()
            settings["target_train"] = 3
            descriptors = NP.ones((4, 3))
            structures = [FakeStructure() for _ in range(4)]

            train_indices, _ = stage._select_training_set(
                descriptors,
                structures,
                settings,
                seed_indices=[0, 1],
                single_element_elastic_indices=[1, 2],
                elastic_indices=[1, 2],
            )

            self.assertEqual(train_indices, [0, 1, 2])

    def test_select_training_set_errors_when_anchors_exceed_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(project_dir, target_train_count=2)
            stage = self.create_stage(project_dir)
            _, settings = stage.load_config()
            settings["target_train"] = 2
            descriptors = NP.ones((4, 3))
            structures = [FakeStructure() for _ in range(4)]

            with self.assertRaisesRegex(ValueError, "elastic stress anchors=2"):
                stage._select_training_set(
                    descriptors,
                    structures,
                    settings,
                    seed_indices=[0, 1],
                    single_element_elastic_indices=[1],
                    elastic_indices=[2, 3],
                )

    def test_select_training_set_error_reports_all_anchor_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(project_dir, target_train_count=2)
            stage = self.create_stage(project_dir)
            _, settings = stage.load_config()
            settings["target_train"] = 2
            descriptors = NP.ones((5, 3))
            structures = [FakeStructure() for _ in range(5)]

            with self.assertRaisesRegex(
                ValueError,
                "seed anchors=1, single-element elastic stress anchors=2, "
                "all elastic stress anchors=2, unique anchors=5, "
                "target_train_count=2",
            ):
                stage._select_training_set(
                    descriptors,
                    structures,
                    settings,
                    seed_indices=[0],
                    single_element_elastic_indices=[1, 2],
                    elastic_indices=[3, 4],
                )

    def test_select_training_set_errors_when_seeds_exceed_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(project_dir, include_seed_structures=True, target_train_count=1)
            stage = self.create_stage(project_dir)
            _, settings = stage.load_config()
            settings["target_train"] = 1
            descriptors = NP.ones((2, 3))
            structures = [FakeStructure(), FakeStructure()]

            with self.assertRaisesRegex(ValueError, "exceeds target_train_count"):
                stage._select_training_set(
                    descriptors,
                    structures,
                    settings,
                    seed_indices=[0, 1],
                )

    def test_composition_projection_bins_share_binary_bin_between_binary_and_ternary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.create_stage(Path(tmp))
            binary_atoms = FakeAtoms("AB", [[0.0, 0.0, 0.0]])
            binary_atoms.info["composition"] = {"W": 0.75, "Y": 0.25}
            ternary_atoms = FakeAtoms("ABC", [[1.0, 0.0, 0.0]])
            ternary_atoms.info["composition"] = {"Cr": 0.25, "W": 0.5625, "Y": 0.1875}

            binary_bins = stage._composition_projection_bins(binary_atoms)
            ternary_bins = stage._composition_projection_bins(ternary_atoms)

            self.assertIn(("W", "Y", 5), binary_bins["binary"])
            self.assertIn(("W", "Y", 5), ternary_bins["binary"])

    def test_composition_projection_bins_include_quaternary_ternary_subsets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.create_stage(Path(tmp))
            atoms = FakeAtoms("ABCD", [[0.0, 0.0, 0.0]])
            atoms.info["composition"] = {"Cr": 0.2, "W": 0.4, "Y": 0.2, "Zr": 0.2}

            projection_bins = stage._composition_projection_bins(atoms)
            ternary_subsets = {entry[:3] for entry in projection_bins["ternary"]}

            self.assertEqual(
                ternary_subsets,
                {
                    ("Cr", "W", "Y"),
                    ("Cr", "W", "Zr"),
                    ("Cr", "Y", "Zr"),
                    ("W", "Y", "Zr"),
                },
            )

    def test_composition_aware_training_selection_preserves_anchors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(project_dir, composition_aware_fps=True, target_train_count=2)
            stage = self.create_stage(project_dir)
            _, settings = stage.load_config()
            descriptors = FakeArray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
            structures = [FakeStructure() for _ in range(3)]
            ase_structures = [FakeAtoms("A", [[0.0, 0.0, 0.0]]) for _ in range(3)]
            ase_structures[0].info["composition"] = {"W": 1.0}
            ase_structures[1].info["composition"] = {"W": 0.5, "Y": 0.5}
            ase_structures[2].info["composition"] = {"W": 0.75, "Y": 0.25}

            train_indices, _ = stage._select_training_set(
                descriptors,
                structures,
                settings,
                ase_structures=ase_structures,
                seed_indices=[1],
            )

            self.assertIn(1, train_indices)
            self.assertEqual(len(train_indices), 2)

    def test_composition_aware_training_selection_skips_fill_when_anchors_meet_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(project_dir, composition_aware_fps=True, target_train_count=2)
            stage = self.create_stage(project_dir)
            _, settings = stage.load_config()
            descriptors = FakeArray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
            structures = [FakeStructure() for _ in range(3)]
            ase_structures = [FakeAtoms("A", [[0.0, 0.0, 0.0]]) for _ in range(3)]

            with patch.object(stage, "_select_training_set_composition_aware") as selector:
                train_indices, train_min_dist = stage._select_training_set(
                    descriptors,
                    structures,
                    settings,
                    ase_structures=ase_structures,
                    seed_indices=[0],
                    elastic_indices=[2],
                )

            selector.assert_not_called()
            self.assertEqual(train_indices, [0, 2])
            self.assertEqual(train_min_dist, 0.0)

    def test_composition_aware_training_selection_prefers_sparse_composition_bin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(
                project_dir,
                composition_aware_fps=True,
                target_train_count=2,
                composition_aware_fps_frontier_fraction=1.0,
            )
            stage = self.create_stage(project_dir)
            _, settings = stage.load_config()
            descriptors = FakeArray([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
            structures = [FakeStructure() for _ in range(3)]
            ase_structures = [
                FakeAtoms("AB", [[0.0, 0.0, 0.0]]),
                FakeAtoms("AB", [[1.0, 0.0, 0.0]]),
                FakeAtoms("AB", [[2.0, 0.0, 0.0]]),
            ]
            ase_structures[0].info["composition"] = {"W": 0.5, "Y": 0.5}
            ase_structures[1].info["composition"] = {"W": 0.5, "Y": 0.5}
            ase_structures[2].info["composition"] = {"W": 0.75, "Y": 0.25}

            train_indices, _ = stage._select_training_set(
                descriptors,
                structures,
                settings,
                ase_structures=ase_structures,
            )

            self.assertEqual(train_indices, [0, 2])

    def test_composition_aware_training_selection_uses_lower_index_when_scores_tie(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(project_dir, composition_aware_fps=True, target_train_count=1)
            stage = self.create_stage(project_dir)
            _, settings = stage.load_config()
            descriptors = FakeArray([[0.0, 0.0], [0.0, 0.0]])
            structures = [FakeStructure() for _ in range(2)]
            ase_structures = [FakeAtoms("AB", [[0.0, 0.0, 0.0]]), FakeAtoms("AB", [[1.0, 0.0, 0.0]])]
            for atoms in ase_structures:
                atoms.info["composition"] = {"W": 0.5, "Y": 0.5}

            train_indices, _ = stage._select_training_set(
                descriptors,
                structures,
                settings,
                ase_structures=ase_structures,
            )

            self.assertEqual(train_indices, [0])

    def test_composition_aware_training_selection_falls_back_for_unary_only_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(project_dir, composition_aware_fps=True, target_train_count=2)
            stage = self.create_stage(project_dir)
            _, settings = stage.load_config()
            descriptors = FakeArray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
            structures = [FakeStructure() for _ in range(3)]
            ase_structures = [FakeAtoms("A", [[0.0, 0.0, 0.0]]) for _ in range(3)]
            for atoms in ase_structures:
                atoms.info["composition"] = {"W": 1.0}

            with patch.object(stage, "_fps_target_count", return_value=([0, 1], 0.25)) as fps:
                train_indices, train_min_dist = stage._select_training_set(
                    descriptors,
                    structures,
                    settings,
                    ase_structures=ase_structures,
                )

            fps.assert_called_once()
            self.assertEqual(train_indices, [0, 1])
            self.assertEqual(train_min_dist, 0.25)

    def test_composition_aware_fallback_preserves_anchor_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(project_dir, composition_aware_fps=True, target_train_count=2)
            stage = self.create_stage(project_dir)
            _, settings = stage.load_config()
            descriptors = FakeArray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
            structures = [FakeStructure() for _ in range(3)]
            ase_structures = [FakeAtoms("A", [[0.0, 0.0, 0.0]]) for _ in range(3)]
            for atoms in ase_structures:
                atoms.info["composition"] = {"W": 1.0}

            train_indices, train_min_dist = stage._select_training_set(
                descriptors,
                structures,
                settings,
                ase_structures=ase_structures,
                seed_indices=[2],
            )

            self.assertEqual(train_indices, [0, 2])
            self.assertEqual(train_min_dist, 0.0)

    def test_composition_aware_attempt_schedule_matches_retry_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.create_stage(Path(tmp))

            self.assertEqual(
                stage._composition_aware_attempt_schedule(0.1, 1.0, 1),
                [(0.1, 1.0)],
            )
            self.assertEqual(
                stage._composition_aware_attempt_schedule(0.1, 1.0, 2),
                [(0.1, 1.0), (0.05, 1.0)],
            )
            self.assertEqual(
                stage._composition_aware_attempt_schedule(0.1, 1.0, 3),
                [(0.1, 1.0), (0.05, 1.0), (0.2, 1.0)],
            )
            self.assertEqual(
                stage._composition_aware_attempt_schedule(0.1, 1.0, 4),
                [(0.1, 1.0), (0.05, 1.0), (0.2, 1.0), (0.1, 2.0)],
            )

    def test_composition_coverage_metrics_return_expected_subset_means(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.create_stage(Path(tmp))
            candidate_bins = {
                0: {
                    "binary": [("W", "Y", 10)],
                    "ternary": [("Cr", "W", "Y", 6, 6, 6)],
                },
                1: {
                    "binary": [("W", "Y", 10)],
                    "ternary": [("Cr", "W", "Y", 6, 6, 6)],
                },
                2: {
                    "binary": [("W", "Y", 5)],
                    "ternary": [("Cr", "W", "Y", 9, 6, 3)],
                },
            }

            metrics = stage._composition_coverage_metrics([0, 1, 2], candidate_bins)

            self.assertEqual(metrics["binary_occupied_bin_fraction"], 2 / 20)
            self.assertAlmostEqual(
                metrics["binary_normalized_entropy"],
                stage._normalized_entropy([2, 1], 20),
            )
            self.assertEqual(
                metrics["ternary_occupied_bin_fraction"],
                2 / stage._ternary_bin_count(18),
            )
            self.assertAlmostEqual(
                metrics["ternary_normalized_entropy"],
                stage._normalized_entropy([2, 1], stage._ternary_bin_count(18)),
            )

    def test_selected_mean_nearest_distance_matches_expected_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.create_stage(Path(tmp))
            descriptors = FakeArray([[0.0], [1.0], [3.0]])

            mean_nn = stage._selected_mean_nearest_distance(descriptors, [0, 1, 2])

            self.assertAlmostEqual(mean_nn, (1.0 + 1.0 + 2.0) / 3.0)

    def test_selected_positive_min_distance_ignores_exact_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.create_stage(Path(tmp))
            descriptors = FakeArray([[0.0], [0.0], [2.0]])

            positive_min = stage._selected_positive_min_distance(descriptors, [0, 1, 2])

            self.assertEqual(positive_min, 2.0)

    def test_flatten_single_column_distances_accepts_iterable_distance_stub(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.create_stage(Path(tmp))

            distances = stage._flatten_single_column_distances(
                FakeDistance([[1.5], [2.5], [3.5]])
            )

            self.assertEqual(distances, [1.5, 2.5, 3.5])

    def test_bin_deficit_reward_prefers_underfilled_bins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.create_stage(Path(tmp))

            underfilled = stage._bin_deficit_reward(0, 10, 20)
            balanced = stage._bin_deficit_reward(1, 20, 20)
            overfilled = stage._bin_deficit_reward(2, 20, 20)

            self.assertGreater(underfilled, balanced)
            self.assertGreater(balanced, 0.0)
            self.assertEqual(overfilled, 0.0)

    def test_pick_best_composition_aware_attempt_rejects_below_floor_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.create_stage(Path(tmp))
            baseline = {
                "attempt_number": 1,
                "frontier_fraction": 0.1,
                "ternary_weight": 1.0,
                "train_min_dist": 10.0,
                "train_positive_min_dist": 10.0,
                "train_mean_nn_dist": 8.0,
                "binary_occupied_bin_fraction": 0.10,
                "binary_normalized_entropy": 0.10,
                "ternary_occupied_bin_fraction": 0.10,
                "ternary_normalized_entropy": 0.10,
            }
            retry = {
                "attempt_number": 2,
                "frontier_fraction": 0.2,
                "ternary_weight": 1.0,
                "train_min_dist": 9.4,
                "train_positive_min_dist": 9.4,
                "train_mean_nn_dist": 7.7,
                "binary_occupied_bin_fraction": 0.90,
                "binary_normalized_entropy": 0.90,
                "ternary_occupied_bin_fraction": 0.90,
                "ternary_normalized_entropy": 0.90,
            }

            best = stage._pick_best_composition_aware_attempt([baseline, retry], 0.95)

            self.assertEqual(best["attempt_number"], 1)

    def test_pick_best_composition_aware_attempt_accepts_higher_score_above_floor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.create_stage(Path(tmp))
            baseline = {
                "attempt_number": 1,
                "frontier_fraction": 0.1,
                "ternary_weight": 1.0,
                "train_min_dist": 10.0,
                "train_positive_min_dist": 10.0,
                "train_mean_nn_dist": 8.0,
                "binary_occupied_bin_fraction": 0.10,
                "binary_normalized_entropy": 0.10,
                "ternary_occupied_bin_fraction": 0.10,
                "ternary_normalized_entropy": 0.10,
            }
            retry = {
                "attempt_number": 2,
                "frontier_fraction": 0.2,
                "ternary_weight": 1.0,
                "train_min_dist": 9.6,
                "train_positive_min_dist": 9.6,
                "train_mean_nn_dist": 7.8,
                "binary_occupied_bin_fraction": 0.40,
                "binary_normalized_entropy": 0.40,
                "ternary_occupied_bin_fraction": 0.40,
                "ternary_normalized_entropy": 0.40,
            }

            best = stage._pick_best_composition_aware_attempt([baseline, retry], 0.95)

            self.assertEqual(best["attempt_number"], 2)

    def test_pick_best_composition_aware_attempt_breaks_ties_by_descriptor_quality(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.create_stage(Path(tmp))
            baseline = {
                "attempt_number": 1,
                "frontier_fraction": 0.1,
                "ternary_weight": 1.0,
                "train_min_dist": 10.0,
                "train_positive_min_dist": 10.0,
                "train_mean_nn_dist": 8.0,
                "binary_occupied_bin_fraction": 0.30,
                "binary_normalized_entropy": 0.30,
                "ternary_occupied_bin_fraction": 0.30,
                "ternary_normalized_entropy": 0.30,
            }
            retry = {
                "attempt_number": 2,
                "frontier_fraction": 0.05,
                "ternary_weight": 1.0,
                "train_min_dist": 10.2,
                "train_positive_min_dist": 10.2,
                "train_mean_nn_dist": 8.0,
                "binary_occupied_bin_fraction": 0.30,
                "binary_normalized_entropy": 0.30,
                "ternary_occupied_bin_fraction": 0.30,
                "ternary_normalized_entropy": 0.30,
            }

            best = stage._pick_best_composition_aware_attempt([baseline, retry], 0.95)

            self.assertEqual(best["attempt_number"], 2)

    def test_pick_best_composition_aware_attempt_uses_positive_min_floor_when_raw_min_is_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.create_stage(Path(tmp))
            baseline = {
                "attempt_number": 1,
                "frontier_fraction": 0.1,
                "ternary_weight": 1.0,
                "train_min_dist": 0.0,
                "train_positive_min_dist": 4.0,
                "train_mean_nn_dist": 8.0,
                "binary_occupied_bin_fraction": 0.10,
                "binary_normalized_entropy": 0.10,
                "ternary_occupied_bin_fraction": 0.10,
                "ternary_normalized_entropy": 0.10,
            }
            retry = {
                "attempt_number": 2,
                "frontier_fraction": 0.2,
                "ternary_weight": 1.0,
                "train_min_dist": 0.0,
                "train_positive_min_dist": 3.0,
                "train_mean_nn_dist": 8.0,
                "binary_occupied_bin_fraction": 0.90,
                "binary_normalized_entropy": 0.90,
                "ternary_occupied_bin_fraction": 0.90,
                "ternary_normalized_entropy": 0.90,
            }

            best = stage._pick_best_composition_aware_attempt([baseline, retry], 0.95)

            self.assertEqual(best["attempt_number"], 1)

    def test_composition_aware_attempt_respects_descriptor_frontier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.create_stage(Path(tmp))
            descriptors = FakeArray([[0.0], [10.0], [8.0]])
            candidate_bins = {
                0: {"binary": [("W", "Y", 10)], "ternary": []},
                1: {"binary": [("W", "Y", 10)], "ternary": []},
                2: {"binary": [("W", "Y", 5)], "ternary": []},
            }

            result = stage._run_composition_aware_attempt(
                descriptors,
                candidate_bins,
                anchor_indices=[0],
                remaining_indices=[1, 2],
                remaining_target=1,
                frontier_fraction=0.5,
                ternary_weight=1.0,
            )

            self.assertEqual(result["selected_indices"], [0, 1])

    def test_composition_aware_attempt_enforces_novelty_floor_before_composition_rerank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.create_stage(Path(tmp))
            descriptors = FakeArray([[0.0], [10.0], [9.5], [7.0]])
            candidate_bins = {
                0: {"binary": [("W", "Y", 10)], "ternary": []},
                1: {"binary": [("W", "Y", 10)], "ternary": []},
                2: {"binary": [("W", "Y", 10)], "ternary": []},
                3: {"binary": [("W", "Y", 5)], "ternary": []},
            }

            result = stage._run_composition_aware_attempt(
                descriptors,
                candidate_bins,
                anchor_indices=[0],
                remaining_indices=[1, 2, 3],
                remaining_target=1,
                frontier_fraction=1.0,
                ternary_weight=1.0,
            )

            self.assertIn(1, result["selected_indices"])
            self.assertNotIn(3, result["selected_indices"])

    def test_save_selected_structures_writes_elastic_training_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            stage = self.create_stage(project_dir)
            atoms = [
                FakeAtoms("A", [[0.0, 0.0, 0.0]]),
                FakeAtoms("B", [[1.0, 0.0, 0.0]]),
            ]
            atoms[1].info["perturbation_type"] = "elastic_stress"

            with patch("testpkg.modules.select.select.ase_write") as write_mock:
                stage._save_selected_structures(atoms, train_indices=[1], test_indices=[0])

            train_call = write_mock.call_args_list[0]
            self.assertEqual(train_call.args[1], [atoms[1]])

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
