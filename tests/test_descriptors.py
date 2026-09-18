import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


def _import_descriptors_module():
    """Import descriptors, mocking only the optional NepTrainKit boundary if absent."""
    try:
        return importlib.import_module("common.descriptors")
    except ModuleNotFoundError as exc:
        if not str(exc.name).startswith("NepTrainKit"):
            raise

        calculator = types.ModuleType("NepTrainKit.core.calculator")
        calculator.NepCalculator = object
        core = types.ModuleType("NepTrainKit.core")
        core.__path__ = []
        package = types.ModuleType("NepTrainKit")
        package.__path__ = []
        modules = {
            "NepTrainKit": package,
            "NepTrainKit.core": core,
            "NepTrainKit.core.calculator": calculator,
        }
        with patch.dict(sys.modules, modules, clear=False):
            return importlib.import_module("common.descriptors")


DESCRIPTORS = _import_descriptors_module()


class CalculatorBoundary:
    """Boundary fake for NepTrainKit; descriptor values remain real NumPy arrays."""

    def __init__(self, path):
        self.path = path
        self.batch_sizes = []
        self.mean_descriptor_flags = []

    def get_structures_descriptor(self, batch, mean_descriptor=True):
        self.batch_sizes.append(len(batch))
        self.mean_descriptor_flags.append(mean_descriptor)
        width = 3 if mean_descriptor else 2
        return np.ones((len(batch), width))


class CalculatorV3Boundary:
    """Boundary fake for the NepTrainKit 3.x descriptor method."""

    def __init__(self, path):
        self.path = path
        self.batch_sizes = []
        self.mean_flags = []

    def descriptors(self, batch, mean=True):
        self.batch_sizes.append(len(batch))
        self.mean_flags.append(mean)
        width = 3 if mean else 2
        return np.ones((len(batch), width))


class DescriptorTests(unittest.TestCase):
    def test_descriptor_cache_path_uses_project_nep_dataset_folder(self) -> None:
        project_dir = Path("demo_project")

        result = DESCRIPTORS.descriptor_cache_path(project_dir)

        self.assertEqual(result, project_dir / "nep" / "datasets" / "descriptors.npy")

    def test_compute_descriptors_batched_splits_batches(self) -> None:
        calc = CalculatorBoundary("model.txt")

        descriptors = DESCRIPTORS.compute_descriptors_batched(
            calc,
            ["s0", "s1", "s2"],
            mean_descriptor=True,
            batch_size=2,
        )

        self.assertEqual(calc.batch_sizes, [2, 1])
        self.assertEqual(descriptors.shape, (3, 3))
        self.assertEqual(calc.mean_descriptor_flags, [True, True])

    def test_compute_descriptors_batched_forwards_mean_descriptor_flag(self) -> None:
        calc = CalculatorBoundary("model.txt")

        descriptors = DESCRIPTORS.compute_descriptors_batched(
            calc,
            ["s0", "s1"],
            mean_descriptor=False,
            batch_size=1,
        )

        self.assertEqual(calc.mean_descriptor_flags, [False, False])
        self.assertEqual(descriptors.shape, (2, 2))

    def test_compute_descriptors_batched_supports_neptrainkit_v3_api(self) -> None:
        calc = CalculatorV3Boundary("model.txt")

        descriptors = DESCRIPTORS.compute_descriptors_batched(
            calc,
            ["s0", "s1", "s2"],
            mean_descriptor=False,
            batch_size=2,
        )

        self.assertEqual(calc.batch_sizes, [2, 1])
        self.assertEqual(calc.mean_flags, [False, False])
        self.assertEqual(descriptors.shape, (3, 2))

    @pytest.mark.xfail(
        strict=True,
        reason="Phase 1 blocker P0-10: descriptor cache reuse must require input identity, not shape alone",
    )
    def test_equal_shape_cache_from_different_inputs_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            cache_path = project_dir / "nep" / "datasets"
            cache_path.mkdir(parents=True, exist_ok=True)
            np.save(cache_path / "descriptors.npy", np.ones((3, 4)))
            model_dir = project_dir / "config" / "nep"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "nep89.txt").write_text("stub", encoding="utf-8")

            with patch.object(
                DESCRIPTORS,
                "NepCalculator",
                side_effect=lambda path: CalculatorBoundary(path),
            ) as calc_cls:
                descriptors = DESCRIPTORS.load_or_compute_descriptors(
                    project_dir,
                    ["new-a", "new-b", "new-c"],
                    mean_descriptor=True,
                    batch_size=2,
                    nep_model_file="nep89.txt",
                )

            calc_cls.assert_called_once()
            self.assertEqual(descriptors.shape, (3, 4))

    def test_load_or_compute_descriptors_computes_and_saves_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            model_dir = project_dir / "config" / "nep"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "nep89.txt"
            model_path.write_text("stub", encoding="utf-8")

            with patch.object(DESCRIPTORS, "NepCalculator", CalculatorBoundary):
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
            np.save(cache_path / "descriptors.npy", np.ones((2, 9)))
            model_dir = project_dir / "config" / "nep"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "nep89.txt").write_text("stub", encoding="utf-8")

            with patch.object(DESCRIPTORS, "NepCalculator", CalculatorBoundary):
                with patch.object(DESCRIPTORS, "compute_descriptors_batched") as compute:
                    compute.return_value = np.ones((3, 2))
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
