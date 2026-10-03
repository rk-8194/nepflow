"""Selection artifact and reporting contracts."""

import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("matplotlib")
pytest.importorskip("sklearn")

from ase import Atoms

from nepflow.stages.selection.artifacts import write_selected_structures
from nepflow.stages.selection.reports import plot_descriptor_space


def test_artifact_writer_preserves_train_and_test_paths(tmp_path) -> None:
    structures = [
        Atoms("Si", positions=[[0.0, 0.0, 0.0]]),
        Atoms("Ge", positions=[[0.0, 0.0, 0.0]]),
    ]

    train_path, test_path = write_selected_structures(
        tmp_path,
        structures,
        [0],
        [1],
    )

    assert train_path == tmp_path / "structures" / "selected" / "train.xyz"
    assert test_path == tmp_path / "structures" / "selected" / "test.xyz"
    assert train_path.exists()
    assert test_path.exists()


def test_descriptor_report_writes_png(tmp_path) -> None:
    output_path = tmp_path / "reports" / "descriptor_space.png"
    plot_descriptor_space(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        [0, 1],
        [2],
        output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
