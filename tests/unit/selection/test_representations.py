"""Focused identity and persistence contracts for selection representations."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("NepTrainKit")

from nepflow.io.json import dumps
from nepflow.stages.selection import representations


class _Structure:
    def __init__(self, structure_id: str):
        self.structure_id = structure_id
        self.info = {"structure_id": structure_id}


def _write_cache(
    project_dir: Path,
    structures: list[_Structure],
    *,
    mean_descriptor: bool = True,
) -> np.ndarray:
    model_path = project_dir / "config" / "nep" / "nep89.txt"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("model-a", encoding="utf-8")
    values = np.arange(6, dtype=float).reshape(3, 2)
    cache_path = representations.descriptor_cache_path(project_dir)
    manifest_path = representations.descriptor_manifest_path(project_dir)
    model = {
        "filename": model_path.name,
        "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    manifest = representations._descriptor_manifest(
        [item.structure_id for item in structures],
        model,
        mean_descriptor,
        values,
    )
    representations._write_descriptor_cache(cache_path, manifest_path, values, manifest)
    return values


def test_exact_identity_reuses_cache_without_recalculation(tmp_path, monkeypatch) -> None:
    structures = [_Structure(f"structure-{index}") for index in range(3)]
    expected = _write_cache(tmp_path, structures)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("exact descriptor cache should be reused")

    monkeypatch.setattr(representations, "compute_descriptors_batched", fail_if_called)
    result = representations.load_or_calculate_representations(
        tmp_path,
        structures,
        mean_descriptor=True,
        batch_size=2,
        nep_model_file="nep89.txt",
    )

    np.testing.assert_array_equal(result, expected)


def test_manifest_serialization_remains_byte_for_byte_unchanged(tmp_path) -> None:
    structures = [_Structure(f"structure-{index}") for index in range(3)]
    _write_cache(tmp_path, structures)
    manifest_path = representations.descriptor_manifest_path(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest_path.read_text(encoding="utf-8") == dumps(manifest, indent=None)


def test_order_change_invalidates_equal_shape_cache(tmp_path, monkeypatch) -> None:
    structures = [_Structure(f"structure-{index}") for index in range(3)]
    _write_cache(tmp_path, structures)
    replacement = np.full((3, 2), 7.0)
    monkeypatch.setattr(representations, "NepCalculator", lambda _path: object())
    monkeypatch.setattr(
        representations,
        "compute_descriptors_batched",
        lambda *_args, **_kwargs: replacement,
    )

    result = representations.load_or_calculate_representations(
        tmp_path,
        list(reversed(structures)),
        mean_descriptor=True,
        batch_size=2,
        nep_model_file="nep89.txt",
    )

    np.testing.assert_array_equal(result, replacement)


def test_corrupt_manifest_invalidates_cache(tmp_path, monkeypatch) -> None:
    structures = [_Structure(f"structure-{index}") for index in range(3)]
    _write_cache(tmp_path, structures)
    representations.descriptor_manifest_path(tmp_path).write_text(
        "{not-json",
        encoding="utf-8",
    )
    replacement = np.full((3, 2), 9.0)
    monkeypatch.setattr(representations, "NepCalculator", lambda _path: object())
    monkeypatch.setattr(
        representations,
        "compute_descriptors_batched",
        lambda *_args, **_kwargs: replacement,
    )

    result = representations.load_or_calculate_representations(
        tmp_path,
        structures,
        mean_descriptor=True,
        batch_size=2,
        nep_model_file="nep89.txt",
    )

    np.testing.assert_array_equal(result, replacement)
    saved = json.loads(
        representations.descriptor_manifest_path(tmp_path).read_text(encoding="utf-8")
    )
    assert saved["structure_ids"] == [item.structure_id for item in structures]
