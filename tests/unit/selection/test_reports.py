"""Selection artifact and reporting contracts."""

import json

import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("matplotlib")
pytest.importorskip("sklearn")

from ase import Atoms

from nepflow.stages.selection.algorithms.information_entropy.diagnostics import (
    EntropyScientificDiagnostics,
)
from nepflow.stages.selection.algorithms.information_entropy.bandwidth import build_entropy_pool
from nepflow.stages.selection.algorithms.information_entropy.models import FrozenBandwidths
from nepflow.stages.selection.artifacts import write_selected_structures
from nepflow.stages.selection.reports import (
    EntropyPresentationData,
    PresentationError,
    _validated_entropy_convergence_series,
    load_entropy_presentation_artifact,
    plot_candidate_bandwidth_summary,
    plot_descriptor_space,
    plot_entropy_convergence,
    plot_projected_entropy_densities,
    write_entropy_presentation_artifact,
    write_entropy_bandwidth_calibration_report,
)


def _convergence_diagnostics(history, *, final_cross_entropy, final_forward_kl):
    """Build the small restored-record surface needed by the presentation helper."""

    record = object.__new__(EntropyScientificDiagnostics)
    object.__setattr__(record, "selection_history", None)
    object.__setattr__(record, "selection_history_json", json.dumps(history))
    object.__setattr__(record, "tolerance", 1.0e-12)
    object.__setattr__(record, "acquisition_order", tuple(item["candidate_id"] for item in history))
    object.__setattr__(
        record,
        "selected_candidate_ids",
        tuple(item["candidate_id"] for item in history),
    )
    object.__setattr__(record, "final_cross_entropy", final_cross_entropy)
    object.__setattr__(record, "final_forward_kl", final_forward_kl)
    return record


def test_entropy_convergence_collapses_aggregate_anchor_baseline(tmp_path) -> None:
    history = [
        {
            "candidate_id": "a0",
            "reason": "anchor",
            "acquisition_iteration": 0,
            "selected_count": 2,
            "cross_entropy": 2.0,
            "forward_kl": 1.0,
        },
        {
            "candidate_id": "a1",
            "reason": "anchor",
            "acquisition_iteration": 0,
            "selected_count": 2,
            "cross_entropy": 2.0,
            "forward_kl": 1.0,
        },
        {
            "candidate_id": "g1",
            "reason": "greedy",
            "acquisition_iteration": 1,
            "selected_count": 3,
            "cross_entropy": 1.8,
            "forward_kl": 0.8,
        },
    ]
    diagnostics = _convergence_diagnostics(
        history,
        final_cross_entropy=1.8,
        final_forward_kl=0.8,
    )

    iterations, cross_entropy, forward_kl = _validated_entropy_convergence_series(diagnostics)

    assert iterations.tolist() == [0, 1]
    assert cross_entropy.tolist() == [2.0, 1.8]
    assert forward_kl is not None
    assert forward_kl.tolist() == [1.0, 0.8]
    plot_entropy_convergence(diagnostics, tmp_path / "reports" / "entropy_convergence.png")
    assert (tmp_path / "reports" / "entropy_convergence.png").stat().st_size > 0


def test_entropy_convergence_without_anchors_starts_at_greedy_iteration_one() -> None:
    history = [
        {
            "candidate_id": "g1",
            "reason": "greedy",
            "acquisition_iteration": 1,
            "selected_count": 1,
            "cross_entropy": 1.8,
            "forward_kl": None,
        },
        {
            "candidate_id": "g2",
            "reason": "greedy",
            "acquisition_iteration": 2,
            "selected_count": 2,
            "cross_entropy": 1.6,
            "forward_kl": None,
        },
    ]
    diagnostics = _convergence_diagnostics(
        history,
        final_cross_entropy=1.6,
        final_forward_kl=0.6,
    )

    iterations, cross_entropy, forward_kl = _validated_entropy_convergence_series(diagnostics)

    assert iterations.tolist() == [1, 2]
    assert cross_entropy.tolist() == [1.8, 1.6]
    assert forward_kl is None


def test_entropy_convergence_rejects_noncontiguous_greedy_iterations() -> None:
    history = [
        {
            "candidate_id": "g1",
            "reason": "greedy",
            "acquisition_iteration": 2,
            "selected_count": 1,
            "cross_entropy": 1.8,
            "forward_kl": 0.8,
        }
    ]
    diagnostics = _convergence_diagnostics(
        history,
        final_cross_entropy=1.8,
        final_forward_kl=0.8,
    )

    with pytest.raises(PresentationError, match="acquisition_iteration must be 1"):
        _validated_entropy_convergence_series(diagnostics)


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


def test_early_bandwidth_report_uses_per_candidate_medians_and_reuses_identity(tmp_path) -> None:
    pool = build_entropy_pool(
        np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        candidate_ids=["c0", "c1"],
        row_candidate_ids=["c0", "c0", "c1"],
        transform_fingerprint="transform",
        representation_fingerprint="representation",
    )
    bandwidths = FrozenBandwidths(
        radii=np.asarray([1.0, 2.0, 3.0]),
        bandwidths=np.asarray([5.0, 1.0, 7.0]),
        k=1,
        c=2.0,
        neighbour_fingerprint="neighbours",
        pool_fingerprint=pool.fingerprint,
        transform_fingerprint=pool.transform_fingerprint,
        fingerprint="frozen-bandwidths",
    )
    descriptors = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    first = write_entropy_bandwidth_calibration_report(
        pool,
        bandwidths,
        descriptors,
        tmp_path,
        calibration_fingerprint="calibration",
    )
    second = write_entropy_bandwidth_calibration_report(
        pool,
        bandwidths,
        descriptors,
        tmp_path,
        calibration_fingerprint="calibration",
    )
    assert first == second
    assert first["plot"]["candidate_median_bandwidths"] == [3.0, 7.0]
    assert (tmp_path / "descriptor_bandwidth_calibration.png").is_file()
    assert (tmp_path / "descriptor_bandwidth_calibration.manifest.json").is_file()


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
