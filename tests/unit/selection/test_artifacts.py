"""Selection artifact ownership tests."""

import pytest

pytest.importorskip("ase")

from ase import Atoms
from ase.io import read

from nepflow.stages.selection.artifacts import write_selected_structures


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
