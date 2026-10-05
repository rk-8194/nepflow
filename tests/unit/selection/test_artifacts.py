"""Selection artifact ownership tests."""

import json

import pytest

pytest.importorskip("ase")

from ase import Atoms
from ase.io import read

from nepflow.errors import ArtifactError
from nepflow.stages.selection import artifacts as artifacts_module
from nepflow.stages.selection.artifacts import (
    SELECTION_MANIFEST_FILENAME,
    read_selection_manifest,
    write_selected_structures,
)


def test_artifact_writer_owns_established_train_and_test_outputs(tmp_path):
    structures = [
        Atoms("Si", positions=[[0.0, 0.0, 0.0]]),
        Atoms("Ge", positions=[[0.0, 0.0, 0.0]]),
    ]

    train_path, test_path = write_selected_structures(tmp_path, structures, [0], [1])

    assert train_path == tmp_path / "structures" / "selected" / "train.xyz"
    assert test_path == tmp_path / "structures" / "selected" / "test.xyz"
    assert read(train_path, index=0).get_chemical_symbols() == ["Si"]
    assert read(test_path, index=0).get_chemical_symbols() == ["Ge"]


def test_selection_manifest_verifies_published_artifacts(tmp_path):
    structures = [
        Atoms("Si", positions=[[0.0, 0.0, 0.0]]),
        Atoms("Ge", positions=[[0.0, 0.0, 0.0]]),
    ]
    write_selected_structures(tmp_path, structures, [0], [1])

    manifest = read_selection_manifest(tmp_path)
    assert manifest["schema_version"] == "nepflow.selection_artifact.v1"
    train_path = tmp_path / "structures" / "selected" / "train.xyz"
    train_path.write_text(train_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ArtifactError, match="changed|missing"):
        read_selection_manifest(tmp_path)


def test_selection_manifest_rejects_unknown_schema(tmp_path):
    structures = [
        Atoms("Si", positions=[[0.0, 0.0, 0.0]]),
        Atoms("Ge", positions=[[0.0, 0.0, 0.0]]),
    ]
    write_selected_structures(tmp_path, structures, [0], [1])
    manifest_path = tmp_path / "structures" / "selected" / SELECTION_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "nepflow.selection_artifact.v999"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactError, match="schema"):
        read_selection_manifest(tmp_path)


def test_selection_publication_rolls_back_existing_triplet(tmp_path, monkeypatch):
    structures = [
        Atoms("Si", positions=[[0.0, 0.0, 0.0]]),
        Atoms("Ge", positions=[[0.0, 0.0, 0.0]]),
    ]
    write_selected_structures(tmp_path, structures, [0], [1])
    selected_dir = tmp_path / "structures" / "selected"
    before = {
        path.name: path.read_bytes()
        for path in selected_dir.iterdir()
        if path.is_file()
    }

    def fail_manifest(*_args, **_kwargs):
        raise OSError("manifest publication failed")

    monkeypatch.setattr(artifacts_module, "write_json", fail_manifest)
    with pytest.raises(OSError, match="manifest publication failed"):
        write_selected_structures(tmp_path, structures, [1], [0])

    after = {path.name: path.read_bytes() for path in selected_dir.iterdir() if path.is_file()}
    assert after == before


def test_first_selection_publication_failure_leaves_no_partial_set(tmp_path, monkeypatch):
    structures = [
        Atoms("Si", positions=[[0.0, 0.0, 0.0]]),
        Atoms("Ge", positions=[[0.0, 0.0, 0.0]]),
    ]

    def fail_manifest(*_args, **_kwargs):
        raise OSError("manifest publication failed")

    monkeypatch.setattr(artifacts_module, "write_json", fail_manifest)
    with pytest.raises(OSError, match="manifest publication failed"):
        write_selected_structures(tmp_path, structures, [0], [1])

    selected_dir = tmp_path / "structures" / "selected"
    assert not any(selected_dir.iterdir())
