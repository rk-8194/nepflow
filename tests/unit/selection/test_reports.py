"""Selection artifact and reporting contracts."""

import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("matplotlib")
pytest.importorskip("sklearn")

from ase import Atoms

from nepflow.stages.selection.artifacts import write_selected_structures
from nepflow.stages.selection.reports import (
    EntropyPresentationData,
    load_entropy_presentation_artifact,
    plot_candidate_bandwidth_summary,
    plot_descriptor_space,
    plot_projected_entropy_densities,
    write_entropy_presentation_artifact,
)


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


def test_entropy_presentation_artifact_roundtrip_and_report_variants(tmp_path) -> None:
    data = EntropyPresentationData(
        schema_version="entropy-selection-presentation-v1",
        scientific_fingerprint="science",
        representation_fingerprint="representation",
        pool_fingerprint="pool",
        bandwidth_fingerprint="bandwidth",
        candidate_ids=("c0", "c1", "c2"),
        structure_ids=("s0", "s1", "s2"),
        row_ids_sha256="rows",
        candidate_input_sha256="candidate-input",
        selected_candidate_ids=("c0",),
        test_candidate_ids=("c2",),
        descriptor_dimension_after=2,
        explained_variance=(0.75, 0.25),
        candidate_coords=np.asarray(
            [[-0.5, 0.0], [0.0, 0.5], [0.5, 0.0]],
            dtype=np.float64,
        ),
        bandwidth_summary=np.asarray(
            [[2.0, 0.5, 0.75, 1.0, 1.25, 1.5], [1, 1, 1, 1, 1, 1], [3, 2, 2, 2, 2, 2]],
            dtype=np.float64,
        ),
        x_edges=np.asarray([-1.0, 0.0, 1.0]),
        y_edges=np.asarray([-1.0, 0.0, 1.0]),
        p_grid=np.asarray([[0.2, 0.3], [0.1, 0.4]], dtype=np.float64),
        q_grid=np.asarray([[0.1, 0.4], [0.2, 0.3]], dtype=np.float64),
        p_mass=1.0,
        q_mass=1.0,
        presentation_config={"smoothing": "none"},
    )
    artifact_path = tmp_path / "reports" / "entropy_presentation-v1.npz"
    write_entropy_presentation_artifact(data, artifact_path)
    restored = load_entropy_presentation_artifact(artifact_path)

    assert restored.presentation_fingerprint == data.presentation_fingerprint
    assert np.array_equal(restored.p_grid, data.p_grid)
    plot_candidate_bandwidth_summary(restored, tmp_path / "reports" / "descriptor_bandwidth.png")
    plot_projected_entropy_densities(
        restored,
        tmp_path / "reports" / "entropy_probability_projection.png",
    )
    assert (tmp_path / "reports" / "descriptor_bandwidth.png").stat().st_size > 0
    assert (tmp_path / "reports" / "entropy_probability_projection.png").stat().st_size > 0
