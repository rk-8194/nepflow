import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

pytest.importorskip("NepTrainKit")

from common import descriptors as DESCRIPTORS  # noqa: E402


CACHE_SCHEMA_VERSION = "descriptor-cache-v1"
P0_10_XFAIL_REASON = "Phase 1 blocker P0-10: descriptor cache identity is incomplete"


class CacheStructure:
    """Small structure boundary carrying a stable cache identity."""

    def __init__(self, structure_id: str):
        self.structure_id = structure_id
        self.info = {"structure_id": structure_id}


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
    @staticmethod
    def cache_manifest_path(project_dir: Path) -> Path:
        return project_dir / "nep" / "datasets" / "descriptors.manifest.json"

    def write_cache_fixture(
        self,
        project_dir: Path,
        structures: list[CacheStructure],
        *,
        mean_descriptor: bool = True,
        schema_version: str = CACHE_SCHEMA_VERSION,
        cached_shape: tuple[int, int] = (3, 4),
        write_manifest: bool = True,
    ) -> Path:
        cache_dir = project_dir / "nep" / "datasets"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = np.arange(np.prod(cached_shape), dtype=float).reshape(cached_shape)
        np.save(cache_dir / "descriptors.npy", cached)

        model_dir = project_dir / "config" / "nep"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "nep89.txt"
        model_path.write_text("model-a", encoding="utf-8")

        if write_manifest:
            manifest = {
                "schema_version": schema_version,
                "structure_ids": [structure.structure_id for structure in structures],
                "model": {
                    "filename": model_path.name,
                    "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                },
                "settings": {"mean_descriptor": mean_descriptor},
                "descriptor_shape": list(cached.shape),
            }
            self.cache_manifest_path(project_dir).write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
        return model_path

    def assert_cache_recomputed(
        self,
        project_dir: Path,
        structures: list[CacheStructure],
        *,
        mean_descriptor: bool = True,
    ) -> None:
        recomputed = np.full((len(structures), 4), 7.0)
        with patch.object(DESCRIPTORS, "NepCalculator", return_value=Mock()):
            with patch.object(
                DESCRIPTORS,
                "compute_descriptors_batched",
                return_value=recomputed,
            ) as compute:
                descriptors = DESCRIPTORS.load_or_compute_descriptors(
                    project_dir,
                    structures,
                    mean_descriptor=mean_descriptor,
                    batch_size=2,
                    nep_model_file="nep89.txt",
                )

        compute.assert_called_once()
        np.testing.assert_array_equal(descriptors, recomputed)

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
        reason=P0_10_XFAIL_REASON,
    )
    def test_equal_shape_cache_from_changed_structure_content_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            cached_structures = [CacheStructure(f"structure-{i}") for i in range(3)]
            self.write_cache_fixture(project_dir, cached_structures)
            changed_structures = [
                CacheStructure("structure-0-changed"),
                cached_structures[1],
                cached_structures[2],
            ]

            self.assert_cache_recomputed(project_dir, changed_structures)

    @pytest.mark.xfail(strict=True, reason=P0_10_XFAIL_REASON)
    def test_equal_shape_cache_from_changed_structure_order_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            structures = [CacheStructure(f"structure-{i}") for i in range(3)]
            self.write_cache_fixture(project_dir, structures)

            self.assert_cache_recomputed(project_dir, list(reversed(structures)))

    @pytest.mark.xfail(strict=True, reason=P0_10_XFAIL_REASON)
    def test_equal_shape_cache_from_changed_model_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            structures = [CacheStructure(f"structure-{i}") for i in range(3)]
            model_path = self.write_cache_fixture(project_dir, structures)
            model_path.write_text("model-b", encoding="utf-8")

            self.assert_cache_recomputed(project_dir, structures)

    @pytest.mark.xfail(strict=True, reason=P0_10_XFAIL_REASON)
    def test_equal_shape_cache_from_changed_representation_settings_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            structures = [CacheStructure(f"structure-{i}") for i in range(3)]
            self.write_cache_fixture(project_dir, structures, mean_descriptor=True)

            self.assert_cache_recomputed(
                project_dir,
                structures,
                mean_descriptor=False,
            )

    @pytest.mark.xfail(strict=True, reason=P0_10_XFAIL_REASON)
    def test_equal_shape_cache_from_changed_schema_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            structures = [CacheStructure(f"structure-{i}") for i in range(3)]
            self.write_cache_fixture(
                project_dir,
                structures,
                schema_version="descriptor-cache-v0",
            )

            self.assert_cache_recomputed(project_dir, structures)

    def test_exact_descriptor_cache_identity_reuses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            structures = [CacheStructure(f"structure-{i}") for i in range(3)]
            self.write_cache_fixture(project_dir, structures)
            expected = np.load(project_dir / "nep" / "datasets" / "descriptors.npy")

            with patch.object(DESCRIPTORS, "compute_descriptors_batched") as compute:
                descriptors = DESCRIPTORS.load_or_compute_descriptors(
                    project_dir,
                    structures,
                    mean_descriptor=True,
                    batch_size=2,
                    nep_model_file="nep89.txt",
                )

            compute.assert_not_called()
            np.testing.assert_array_equal(descriptors, expected)

    @pytest.mark.xfail(strict=True, reason=P0_10_XFAIL_REASON)
    def test_missing_cache_identity_metadata_is_not_trusted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            structures = [CacheStructure(f"structure-{i}") for i in range(3)]
            self.write_cache_fixture(project_dir, structures, write_manifest=False)

            self.assert_cache_recomputed(project_dir, structures)

    @pytest.mark.xfail(strict=True, reason=P0_10_XFAIL_REASON)
    def test_malformed_cache_identity_metadata_is_not_trusted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            structures = [CacheStructure(f"structure-{i}") for i in range(3)]
            self.write_cache_fixture(project_dir, structures)
            self.cache_manifest_path(project_dir).write_text("{not-json", encoding="utf-8")

            self.assert_cache_recomputed(project_dir, structures)

    @pytest.mark.xfail(strict=True, reason=P0_10_XFAIL_REASON)
    def test_descriptor_and_manifest_shape_disagreement_invalidates_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            structures = [CacheStructure(f"structure-{i}") for i in range(3)]
            self.write_cache_fixture(project_dir, structures)
            manifest_path = self.cache_manifest_path(project_dir)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["descriptor_shape"] = [3, 5]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            self.assert_cache_recomputed(project_dir, structures)

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
