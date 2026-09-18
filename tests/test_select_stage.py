import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")
from ase import Atoms  # noqa: E402


class StructureStub:
    """Minimal NepTrainKit structure boundary used by selection tests."""

    def __init__(self, num_atoms: int = 1):
        self.num_atoms = num_atoms


class CalculatorBoundary:
    """External calculator boundary returning real NumPy descriptors."""

    def __init__(self, path: str):
        self.path = path

    def get_structures_descriptor(self, batch, mean_descriptor=True):
        width = 3 if mean_descriptor else 2
        return np.ones((len(batch), width), dtype=float)


def _fake_farthest_point_sampling(descriptors, n_samples, min_dist):
    del descriptors, min_dist
    return np.arange(n_samples, dtype=int)


def _import_select_module():
    """Import the stage, scoping only the optional NepTrainKit boundary when absent."""
    try:
        import NepTrainKit  # noqa: F401
    except ImportError:
        nep_pkg = types.ModuleType("NepTrainKit")
        nep_pkg.__path__ = []
        nep_core = types.ModuleType("NepTrainKit.core")
        nep_core.__path__ = []
        nep_calc = types.ModuleType("NepTrainKit.core.calculator")
        nep_calc.NepCalculator = CalculatorBoundary
        nep_struct = types.ModuleType("NepTrainKit.core.structure")
        nep_struct.Structure = type(
            "StructureReader",
            (),
            {"read_multiple": staticmethod(lambda path: [StructureStub()])},
        )
        nep_io = types.ModuleType("NepTrainKit.core.io")
        nep_io.farthest_point_sampling = _fake_farthest_point_sampling
        modules = {
            "NepTrainKit": nep_pkg,
            "NepTrainKit.core": nep_core,
            "NepTrainKit.core.calculator": nep_calc,
            "NepTrainKit.core.structure": nep_struct,
            "NepTrainKit.core.io": nep_io,
        }
        with patch.dict(sys.modules, modules, clear=False):
            return importlib.import_module("modules.select.select")
    return importlib.import_module("modules.select.select")


select_module = _import_select_module()
SelectStage = select_module.SelectStage
DESCRIPTORS = importlib.import_module("common.descriptors")


def make_atoms(symbols: str = "Si", *, x: float = 0.0, composition: dict | None = None) -> Atoms:
    atoms = Atoms(
        symbols,
        positions=np.array([[x + i, 0.0, 0.0] for i in range(len(Atoms(symbols)))], dtype=float),
        cell=np.eye(3) * 5.0,
        pbc=True,
    )
    if composition is not None:
        atoms.info["composition"] = composition
    return atoms


def write_project_config(
    project_dir: Path,
    *,
    include_seed_structures: bool = False,
    include_single_element_elastic_stress_structures: bool = False,
    include_elastic_stress_structures: bool = False,
    composition_aware_fps: bool = False,
    target_train_count: int = 2,
    frontier_fraction: float = 0.10,
    ternary_weight: float = 1.0,
    adaptive_retries: int = 4,
    descriptor_floor_fraction: float = 0.95,
) -> None:
    config_dir = project_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "project.config").write_text(
        "\n".join(
            [
                "[project]",
                "random_seed=7",
                "",
                "[selection]",
                "descriptor_type=structure",
                "batch_size=2",
                f"target_train_count={target_train_count}",
                "target_test_count=1",
                "target_tolerance=1",
                "max_search_iterations=4",
                "test_pool_factor=1.0",
                f"include_seed_structures={'true' if include_seed_structures else 'false'}",
                "include_single_element_elastic_stress_structures="
                f"{'true' if include_single_element_elastic_stress_structures else 'false'}",
                "include_elastic_stress_structures="
                f"{'true' if include_elastic_stress_structures else 'false'}",
                f"composition_aware_fps={'true' if composition_aware_fps else 'false'}",
                f"composition_aware_fps_frontier_fraction={frontier_fraction}",
                f"composition_aware_fps_ternary_weight={ternary_weight}",
                f"composition_aware_fps_adaptive_retries={adaptive_retries}",
                f"composition_aware_fps_descriptor_floor_fraction={descriptor_floor_fraction}",
                "nep_model_file=nep89.txt",
                "",
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

    def test_prepare_requires_generated_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.create_stage(Path(tmp))
            with self.assertRaises(FileNotFoundError):
                stage.prepare()

    def test_prepare_returns_none_for_empty_structures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            generated = project_dir / "structures" / "generated" / "generated_structures.xyz"
            generated.parent.mkdir(parents=True)
            generated.write_text("", encoding="utf-8")
            stage = self.create_stage(project_dir)

            with patch.object(select_module.Structure, "read_multiple", return_value=[]):
                with patch.object(select_module, "ase_read", return_value=[]):
                    self.assertIsNone(stage.prepare())

    def test_run_executes_and_finalizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.create_stage(Path(tmp))
            prepared = {"structures": [StructureStub()], "ase_structures": [make_atoms()]}
            result = {"train_indices": [0], "test_indices": []}
            with patch.object(stage, "load_config", return_value=(Mock(), {})):
                with patch.object(stage, "prepare", return_value=prepared):
                    with patch.object(stage, "execute", return_value=result) as execute:
                        with patch.object(stage, "finalize") as finalize:
                            stage.run()
            execute.assert_called_once()
            finalize.assert_called_once_with(prepared, result)

    def test_load_config_defaults_and_reads_composition_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(project_dir)
            stage = self.create_stage(project_dir)
            _, settings = stage.load_config()
            self.assertFalse(settings["include_seed_structures"])
            self.assertFalse(settings["include_single_element_elastic_stress_structures"])
            self.assertFalse(settings["composition_aware_fps"])
            self.assertEqual(settings["composition_aware_fps_adaptive_retries"], 4)

    def test_load_config_rejects_invalid_composition_aware_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(project_dir, frontier_fraction=0.0)
            stage = self.create_stage(project_dir)
            with self.assertRaisesRegex(ValueError, "frontier_fraction"):
                stage.load_config()

    def test_descriptor_loader_reuses_cache_with_real_numpy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            cache_path = project_dir / "nep" / "datasets" / "descriptors.npy"
            cache_path.parent.mkdir(parents=True)
            expected = np.ones((3, 4), dtype=float)
            np.save(cache_path, expected)
            stage = self.create_stage(project_dir)
            structures = [StructureStub(), StructureStub(), StructureStub()]

            with patch.object(DESCRIPTORS, "NepCalculator") as calculator:
                result = DESCRIPTORS.load_or_compute_descriptors(
                    project_dir,
                    structures,
                    mean_descriptor=True,
                    batch_size=2,
                    nep_model_file="missing.nep",
                )

            np.testing.assert_array_equal(result, expected)
            calculator.assert_not_called()
            self.assertIsNotNone(stage)

    def test_execute_uses_shared_descriptor_loader_and_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(project_dir)
            stage = self.create_stage(project_dir)
            config, settings = stage.load_config()
            prepared = {
                "structures": [StructureStub(), StructureStub()],
                "ase_structures": [make_atoms(), make_atoms("Ge")],
            }
            descriptors = np.ones((2, 3), dtype=float)
            test_result = {
                "test_indices": [1],
                "test_min_dist": 0.2,
                "min_train_test_dist": 0.3,
                "mean_train_test_dist": 0.4,
            }
            with patch.object(select_module, "load_or_compute_descriptors", return_value=descriptors) as loader:
                with patch.object(stage, "_select_training_set", return_value=([0], 0.1)):
                    with patch.object(stage, "_select_test_set", return_value=test_result):
                        result = stage.execute(config, settings, prepared)

            loader.assert_called_once()
            np.testing.assert_array_equal(result["descriptors"], descriptors)
            self.assertEqual(result["train_indices"], [0])
            self.assertEqual(result["test_indices"], [1])

    def test_execute_includes_seed_and_elastic_anchor_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(
                project_dir,
                include_seed_structures=True,
                include_single_element_elastic_stress_structures=True,
                include_elastic_stress_structures=True,
            )
            stage = self.create_stage(project_dir)
            config, settings = stage.load_config()
            prepared = {
                "structures": [StructureStub() for _ in range(3)],
                "ase_structures": [make_atoms() for _ in range(3)],
            }
            with patch.object(select_module, "load_or_compute_descriptors", return_value=np.ones((3, 2))):
                with patch.object(stage, "_load_seed_indices", return_value=[0]) as seed:
                    with patch.object(stage, "_load_single_element_elastic_stress_indices", return_value=[1]) as single:
                        with patch.object(stage, "_load_elastic_stress_indices", return_value=[1, 2]) as elastic:
                            with patch.object(stage, "_select_training_set", return_value=([0, 1], 0.1)) as train:
                                with patch.object(stage, "_select_test_set", return_value={
                                    "test_indices": [], "test_min_dist": 0.0,
                                    "min_train_test_dist": float("inf"), "mean_train_test_dist": float("inf"),
                                }):
                                    result = stage.execute(config, settings, prepared)

            seed.assert_called_once()
            single.assert_called_once()
            elastic.assert_called_once()
            call_kwargs = train.call_args.kwargs
            self.assertEqual(call_kwargs["seed_indices"], [0])
            self.assertEqual(call_kwargs["single_element_elastic_indices"], [1])
            self.assertEqual(call_kwargs["elastic_indices"], [1, 2])
            self.assertEqual(result["train_anchor_count"], 3)

    def test_training_selection_preserves_and_deduplicates_anchors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.create_stage(Path(tmp))
            settings = {"target_train": 3, "composition_aware_fps": False, "mean_descriptor": True,
                        "tolerance": 1, "max_iterations": 4}
            descriptors = np.arange(8, dtype=float).reshape(4, 2)
            with patch.object(stage, "_fps_target_count", return_value=([0], 0.5)):
                indices, minimum = stage._select_training_set(
                    descriptors,
                    [StructureStub() for _ in range(4)],
                    settings,
                    seed_indices=[2, 0],
                    elastic_indices=[2],
                )
            self.assertEqual(indices, [0, 1, 2])
            self.assertAlmostEqual(minimum, 0.5)

    def test_training_selection_rejects_too_many_unique_anchors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.create_stage(Path(tmp))
            settings = {"target_train": 2}
            with self.assertRaisesRegex(ValueError, "unique anchors=3"):
                stage._select_training_set(
                    np.ones((4, 2)),
                    [StructureStub() for _ in range(4)],
                    settings,
                    seed_indices=[0, 1],
                    elastic_indices=[2],
                )

    def test_selection_helpers_use_real_numpy_distances(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.create_stage(Path(tmp))
            descriptors = np.array([[0.0], [1.0], [3.0]])
            self.assertAlmostEqual(stage._selected_mean_nearest_distance(descriptors, [0, 1, 2]), 4.0 / 3.0)
            self.assertAlmostEqual(stage._selected_positive_min_distance(np.array([[0.0], [0.0], [2.0]]), [0, 1, 2]), 2.0)
            np.testing.assert_allclose(stage._flatten_single_column_distances(np.array([[1.5], [2.5]])), [1.5, 2.5])

    def test_composition_projection_includes_binary_and_ternary_subsets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.create_stage(Path(tmp))
            binary = stage._composition_projection_bins(make_atoms("SiGe"))
            quaternary = stage._composition_projection_bins(make_atoms("SiGeAlCu"))
            self.assertEqual(len(binary["binary"]), 1)
            self.assertEqual(len(binary["ternary"]), 0)
            self.assertEqual(len(quaternary["binary"]), 6)
            self.assertEqual(len(quaternary["ternary"]), 4)

    def test_composition_aware_selection_preserves_anchor_and_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.create_stage(Path(tmp))
            settings = {
                "target_train": 3,
                "composition_aware_fps": True,
                "composition_aware_fps_frontier_fraction": 0.1,
                "composition_aware_fps_ternary_weight": 1.0,
                "composition_aware_fps_adaptive_retries": 1,
                "composition_aware_fps_descriptor_floor_fraction": 0.0,
                "mean_descriptor": True,
                "tolerance": 1,
                "max_iterations": 4,
            }
            ase_structures = [make_atoms("SiGe"), make_atoms("SiGe", x=1), make_atoms("SiAl", x=2)]
            descriptors = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
            indices, _ = stage._select_training_set(
                descriptors,
                [StructureStub() for _ in range(3)],
                settings,
                ase_structures=ase_structures,
                seed_indices=[0],
            )
            self.assertEqual(len(indices), 3)
            self.assertIn(0, indices)

    def test_composition_aware_selection_falls_back_for_unary_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.create_stage(Path(tmp))
            settings = {
                "target_train": 2,
                "composition_aware_fps": True,
                "composition_aware_fps_frontier_fraction": 0.1,
                "composition_aware_fps_ternary_weight": 1.0,
                "composition_aware_fps_adaptive_retries": 1,
                "composition_aware_fps_descriptor_floor_fraction": 0.95,
                "mean_descriptor": True,
                "tolerance": 1,
                "max_iterations": 4,
            }
            with patch.object(stage, "_fps_target_count", return_value=([0], 0.25)):
                indices, minimum = stage._select_training_set(
                    np.array([[0.0], [1.0], [2.0]]),
                    [StructureStub() for _ in range(3)],
                    settings,
                    ase_structures=[make_atoms(), make_atoms("Ge"), make_atoms("W")],
                )
            self.assertEqual(indices, [0])
            self.assertAlmostEqual(minimum, 0.25)

    def test_composition_aware_attempt_schedule_matches_retry_budget(self) -> None:
        self.assertEqual(
            select_module.SelectStage._composition_aware_attempt_schedule(0.1, 1.0, 3),
            [(0.1, 1.0), (0.05, 1.0), (0.2, 1.0)],
        )

    def test_composition_coverage_metrics_return_subset_means(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.create_stage(Path(tmp))
            ase_structures = [make_atoms("SiGe"), make_atoms("SiGe"), make_atoms("SiAl")]
            candidate_bins = {
                index: stage._composition_projection_bins(atoms)
                for index, atoms in enumerate(ase_structures)
            }
            metrics = stage._composition_coverage_metrics([0, 2], candidate_bins)
            self.assertGreaterEqual(metrics["binary_occupied_bin_fraction"], 0.0)
            self.assertLessEqual(metrics["binary_occupied_bin_fraction"], 1.0)
            self.assertGreaterEqual(metrics["binary_normalized_entropy"], 0.0)

    def test_pick_best_attempt_respects_descriptor_floor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.create_stage(Path(tmp))
            base = {
                "attempt_number": 1, "train_positive_min_dist": 1.0, "train_min_dist": 1.0,
                "train_mean_nn_dist": 1.0, "binary_occupied_bin_fraction": 0.2,
                "binary_normalized_entropy": 0.2, "ternary_occupied_bin_fraction": 0.0,
                "ternary_normalized_entropy": 0.0,
            }
            rejected = {**base, "attempt_number": 2, "train_positive_min_dist": 0.5,
                        "train_min_dist": 0.5, "train_mean_nn_dist": 0.5,
                        "binary_occupied_bin_fraction": 1.0}
            accepted = {**base, "attempt_number": 3, "train_positive_min_dist": 1.1,
                        "train_min_dist": 1.1, "train_mean_nn_dist": 1.1,
                        "binary_occupied_bin_fraction": 0.8}
            result = stage._pick_best_composition_aware_attempt([base, rejected, accepted], 0.9)
            self.assertEqual(result["attempt_number"], 3)

    def test_seed_loader_matches_real_ase_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            stage = self.create_stage(project_dir)
            seed_path = project_dir / "structures" / "seeds" / "base_structures.xyz"
            seed_path.parent.mkdir(parents=True)
            seed_path.write_text("1\n\nSi 0 0 0\n", encoding="utf-8")
            generated = [make_atoms(), make_atoms("Ge")]
            generated[0].info["seed_id"] = "seed_000000"
            generated[1].info["seed_id"] = "seed_000001"
            seed = make_atoms()
            seed.info["seed_id"] = "seed_000001"
            with patch.object(select_module, "ase_read", return_value=[seed]):
                self.assertEqual(stage._load_seed_indices(generated), [1])

    def test_elastic_helpers_detect_single_element_structures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.create_stage(Path(tmp))
            unary = make_atoms("Si2", composition={"Si": 1.0})
            unary.info["perturbation_type"] = "elastic_stress"
            binary = make_atoms("SiGe")
            binary.info["perturbation_type"] = "elastic_stress"
            self.assertTrue(stage._is_single_element_elastic_stress(unary))
            self.assertFalse(stage._is_single_element_elastic_stress(binary))
            self.assertEqual(stage._load_elastic_stress_indices([unary, binary]), [0, 1])
            self.assertEqual(stage._load_single_element_elastic_stress_indices([unary, binary]), [0])

    def test_save_selected_structures_writes_extxyz_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            stage = self.create_stage(project_dir)
            atoms = [make_atoms(), make_atoms("Ge")]
            stage._save_selected_structures(atoms, [0], [1])
            self.assertTrue((project_dir / "structures" / "selected" / "train.xyz").exists())
            self.assertTrue((project_dir / "structures" / "selected" / "test.xyz").exists())


if __name__ == "__main__":
    unittest.main()
