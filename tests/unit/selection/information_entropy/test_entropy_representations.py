"""Regression coverage for the potential-independent entropy representation."""

import logging
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

import nepflow.stages.selection.representations as representation_module
from nepflow.errors import StateError
from nepflow.stages.selection.algorithms.information_entropy.bandwidth import build_entropy_pool
from nepflow.stages.selection.algorithms.information_entropy.neighbours import (
    compute_exact_neighbours,
)
from nepflow.stages.selection.reference_measure import reference_weights_fingerprint
from nepflow.stages.selection.representations import (
    LocalRepresentationConfig,
    build_local_environment_representation,
    fit_deterministic_whitening,
    load_or_calculate_local_representations,
)


def test_entropy_worker_import_does_not_load_neptrainkit() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    source_path = str(repo_root / "src")
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path
        if not existing_pythonpath
        else os.pathsep.join((source_path, existing_pythonpath))
    )
    script = """
import sys

from nepflow.stages.selection.representations import _compute_local_descriptor_task
from nepflow.stages.selection.stage import SelectionStage

assert _compute_local_descriptor_task is not None
assert SelectionStage is not None
assert not any(name.startswith("NepTrainKit") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def _atoms(symbols: str = "SiOSi") -> Atoms:
    return Atoms(
        symbols,
        positions=[[1.0, 1.0, 1.0], [2.4, 1.1, 1.0], [1.0, 2.2, 1.3]],
        cell=[12.0, 12.0, 12.0],
        pbc=False,
    )


def _build(candidate: Atoms, *, magnetic_mode: str = "structural"):
    return build_local_environment_representation(
        [candidate],
        config=LocalRepresentationConfig(
            cutoff=4.0,
            radial_bins=6,
            angular_bins=6,
            magnetic_mode=magnetic_mode,
            species=("Ge", "O", "Si"),
        ),
        candidate_ids=["candidate-a"],
        structure_ids=["structure-a"],
    )


def _dense_atoms() -> Atoms:
    return Atoms(
        symbols=["Si"] * 32,
        positions=[[0.9 * index, 0.0, 0.0] for index in range(32)],
        cell=[40.0, 40.0, 40.0],
        pbc=False,
    )


def _pool() -> list[Atoms]:
    return [_atoms(), _dense_atoms()]


def test_translation_and_global_spatial_rotation_invariance() -> None:
    original = _atoms()
    translated = original.copy()
    translated.positions += [3.5, -1.2, 2.0]

    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    rotated = original.copy()
    rotated.positions = original.positions @ rotation.T
    rotated.cell = original.cell.array @ rotation.T

    expected = _build(original).raw_descriptors
    np.testing.assert_allclose(_build(translated).raw_descriptors, expected, atol=1.0e-12)
    np.testing.assert_allclose(_build(rotated).raw_descriptors, expected, atol=1.0e-12)


def test_periodic_bcc_w_symmetry_fixture_reports_raw_and_final_knn_evidence(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Measure bcc-W equivalence without changing descriptor science.

    The fixture deliberately records, rather than assumes, how crystallographic
    equivalence propagates through raw descriptors, preliminary whitening, and
    final weighted whitening.  The rattled structure remains in the same pool
    so the regularised final transform has a meaningful non-zero eigenspectrum.
    """

    base = bulk("W", "bcc", a=3.165, cubic=True).repeat((4, 4, 4))
    translated = base.copy()
    translated.positions += np.asarray([0.371, -0.219, 0.143])
    translated.wrap()
    permutation = np.roll(np.arange(len(base), dtype=np.int64), 19)
    permuted = base[permutation]
    rattled = base.copy()
    rng = np.random.default_rng(128)
    rattled.positions += rng.normal(0.0, 0.01, size=rattled.positions.shape)
    rattled.wrap()

    config = LocalRepresentationConfig(
        cutoff=5.0,
        radial_bins=8,
        angular_bins=8,
        species=("Cr", "W"),
        whitening_tolerance=1.0e-12,
        whitening_regularization=1.0e-12,
        whitening_singular_policy="regularize",
    )
    representation = build_local_environment_representation(
        [base, translated, permuted, rattled],
        config=config,
        candidate_ids=["bcc-base", "bcc-translated", "bcc-permuted", "bcc-rattled"],
        structure_ids=["bcc-base", "bcc-translated", "bcc-permuted", "bcc-rattled"],
        local_descriptor_workers=1,
        reference_measure_tolerance=1.0e-6,
    )
    assert representation.raw_descriptors.shape == (512, 42)
    assert representation.descriptors.shape == (512, 42)

    blocks = {
        candidate: representation.raw_descriptors[offset * 128 : (offset + 1) * 128]
        for offset, candidate in enumerate(
            ("bcc-base", "bcc-translated", "bcc-permuted", "bcc-rattled")
        )
    }
    final_blocks = {
        candidate: representation.descriptors[offset * 128 : (offset + 1) * 128]
        for offset, candidate in enumerate(
            ("bcc-base", "bcc-translated", "bcc-permuted", "bcc-rattled")
        )
    }

    def pairwise_stats(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
        distances = np.sqrt(
            np.sum(
                (first[:, None, :] - second[None, :, :]) ** 2,
                axis=2,
                dtype=np.float64,
            )
        )
        return {
            "min": float(np.min(distances)),
            "median": float(np.median(distances)),
            "max": float(np.max(distances)),
        }

    preliminary_transform = fit_deterministic_whitening(
        representation.raw_descriptors,
        tolerance=1.0e-12,
        regularization=1.0e-12,
        singular_policy="regularize",
    )
    preliminary = preliminary_transform.apply(representation.raw_descriptors)
    preliminary_blocks = {
        candidate: preliminary[offset * 128 : (offset + 1) * 128]
        for offset, candidate in enumerate(
            ("bcc-base", "bcc-translated", "bcc-permuted", "bcc-rattled")
        )
    }
    neighbour_result = compute_exact_neighbours(
        representation.descriptors,
        8,
        chunk_size=16,
        representation_fingerprint=representation.transform_fingerprint,
    )
    floor = 1.0e-6
    raw_radii = neighbour_result.radii
    radius_evidence = {
        "minimum": float(np.min(raw_radii)),
        "p1": float(np.percentile(raw_radii, 1.0)),
        "median": float(np.percentile(raw_radii, 50.0)),
        "maximum": float(np.max(raw_radii)),
        "below_floor_rows": int(np.count_nonzero(raw_radii < floor)),
        "equal_floor_rows": int(np.count_nonzero(raw_radii == floor)),
        "at_or_above_floor_rows": int(np.count_nonzero(raw_radii >= floor)),
        "below_floor_locations": int(
            np.count_nonzero(raw_radii[neighbour_result.location_representatives] < floor)
        ),
        "distinct_locations": int(neighbour_result.unique_locations.shape[0]),
        "exact_duplicate_rows": int(
            raw_radii.size - neighbour_result.unique_locations.shape[0]
        ),
    }
    raw_unique_counts = {
        candidate: int(np.unique(values, axis=0).shape[0])
        for candidate, values in blocks.items()
    }
    preliminary_unique_counts = {
        candidate: int(np.unique(values, axis=0).shape[0])
        for candidate, values in preliminary_blocks.items()
    }
    final_unique_counts = {
        candidate: int(np.unique(values, axis=0).shape[0])
        for candidate, values in final_blocks.items()
    }
    caplog.set_level(logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(
        "bcc-W entropy evidence: raw_unique=%s, preliminary_unique=%s, final_unique=%s, "
        "raw_base_translated=%s, preliminary_base_translated=%s, final_base_translated=%s, "
        "raw_base_permuted=%s, preliminary_base_permuted=%s, final_base_permuted=%s, "
        "raw_base_rattled=%s, preliminary_base_rattled=%s, final_base_rattled=%s, "
        "regularised_dimensions=%d, near_null_raw=%d, near_null_final=%d, "
        "raw_eigenvalues=%s, final_eigenvalues=%s, radii=%s",
        raw_unique_counts,
        preliminary_unique_counts,
        final_unique_counts,
        pairwise_stats(blocks["bcc-base"], blocks["bcc-translated"]),
        pairwise_stats(preliminary_blocks["bcc-base"], preliminary_blocks["bcc-translated"]),
        pairwise_stats(final_blocks["bcc-base"], final_blocks["bcc-translated"]),
        pairwise_stats(blocks["bcc-base"], blocks["bcc-permuted"]),
        pairwise_stats(preliminary_blocks["bcc-base"], preliminary_blocks["bcc-permuted"]),
        pairwise_stats(final_blocks["bcc-base"], final_blocks["bcc-permuted"]),
        pairwise_stats(blocks["bcc-base"], blocks["bcc-rattled"]),
        pairwise_stats(preliminary_blocks["bcc-base"], preliminary_blocks["bcc-rattled"]),
        pairwise_stats(final_blocks["bcc-base"], final_blocks["bcc-rattled"]),
        representation.transform.retained_dimensions,
        int(np.count_nonzero(preliminary_transform.eigenvalues <= 1.0e-12)),
        int(np.count_nonzero(representation.transform.eigenvalues <= 1.0e-12)),
        preliminary_transform.eigenvalues.tolist(),
        representation.transform.eigenvalues.tolist(),
        radius_evidence,
    )
    assert any("bcc-W entropy evidence:" in record.getMessage() for record in caplog.records)
    assert np.any(np.linalg.norm(blocks["bcc-base"] - blocks["bcc-rattled"], axis=1) > 1.0e-12)


def test_atom_order_permutation_and_species_channels_are_respected() -> None:
    original = _atoms()
    permutation = [2, 0, 1]
    permuted = Atoms(
        symbols=[original.get_chemical_symbols()[index] for index in permutation],
        positions=original.positions[permutation],
        cell=original.cell,
        pbc=False,
    )
    first = _build(original)
    second = _build(permuted)
    np.testing.assert_allclose(second.raw_descriptors, first.raw_descriptors, atol=1.0e-12)
    assert [(row.candidate_id, row.structure_id) for row in first.rows] == [
        ("candidate-a", "structure-a")
    ] * len(first.rows)
    assert [row.atom_index for row in first.rows] == [
        row.atom_index for row in _build(original).rows
    ]

    changed_species = _atoms("GeOSi")
    assert not np.allclose(_build(changed_species).raw_descriptors, first.raw_descriptors)


def test_whitening_is_deterministic_and_singular_policy_is_explicit() -> None:
    values = np.asarray([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    first = fit_deterministic_whitening(values, singular_policy="drop")
    second = fit_deterministic_whitening(values, singular_policy="drop")
    np.testing.assert_array_equal(first.apply(values), second.apply(values))
    assert first.fingerprint == second.fingerprint
    assert first.retained_dimensions == 1
    with pytest.raises(ValueError, match="singular directions"):
        fit_deterministic_whitening(values, singular_policy="reject")


def test_weighted_whitening_matches_hand_calculated_probability_measure() -> None:
    values = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    weights = np.asarray([0.2, 0.3, 0.5])
    transform = fit_deterministic_whitening(
        values,
        target_weights=weights,
        singular_policy="reject",
    )
    expected_mean = np.sum(values * weights[:, None], axis=0)
    centred = values - expected_mean
    expected_covariance = centred.T @ (centred * weights[:, None])
    reconstructed = (
        transform.eigenvectors
        @ np.diag(transform.eigenvalues)
        @ transform.eigenvectors.T
    )
    np.testing.assert_allclose(transform.mean, expected_mean, atol=1.0e-14)
    np.testing.assert_allclose(reconstructed, expected_covariance, atol=1.0e-14)
    assert transform.weighting_fingerprint == reference_weights_fingerprint(weights)


def test_weighted_whitening_preserves_already_normalized_reference_bytes() -> None:
    occupancy = np.asarray([61, 29, 38], dtype=np.float64)
    probabilities = np.asarray(
        (1.0 / occupancy) / math.fsum(float(value) for value in 1.0 / occupancy),
        dtype=np.float64,
    )
    assert float(np.sum(probabilities, dtype=np.float64)) != 1.0
    assert math.isclose(
        float(np.sum(probabilities, dtype=np.float64)),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )

    values = np.arange(probabilities.size * 2, dtype=np.float64).reshape(probabilities.size, 2)
    transform = fit_deterministic_whitening(
        values,
        target_weights=probabilities,
        singular_policy="regularize",
    )

    assert transform.weighting_fingerprint == reference_weights_fingerprint(probabilities)


def test_weighted_whitening_normalizes_genuinely_unnormalized_inputs_once() -> None:
    values = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float64)
    supplied = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    original = supplied.copy()
    expected = supplied / np.sum(supplied, dtype=np.float64)

    transform = fit_deterministic_whitening(values, target_weights=supplied)

    np.testing.assert_array_equal(supplied, original)
    assert transform.weighting_fingerprint == reference_weights_fingerprint(expected)


def test_local_representation_and_entropy_pool_share_reference_weights() -> None:
    representation = _build(_atoms())
    pool = build_entropy_pool(representation)
    np.testing.assert_array_equal(pool.probabilities, representation.target_weights)
    assert pool.reference_weights_fingerprint == representation.transform.weighting_fingerprint
    assert pool.reference_weights_fingerprint == representation.reference_weights_fingerprint


def test_non_soc_magnetic_channels_are_global_spin_rotation_invariant() -> None:
    original = _atoms()
    original.set_array(
        "magnetic_moments",
        np.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    rotation = np.asarray(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        dtype=float,
    )
    rotated = original.copy()
    rotated.set_array("magnetic_moments", original.arrays["magnetic_moments"] @ rotation.T)
    first = _build(original, magnetic_mode="non_soc")
    second = _build(rotated, magnetic_mode="non_soc")
    np.testing.assert_allclose(second.raw_descriptors, first.raw_descriptors, atol=1.0e-12)


def test_structural_mode_rejects_repeated_geometry_but_magnetic_mode_distinguishes_states() -> None:
    first = _atoms()
    second = first.copy()
    first.set_array(
        "magnetic_moments",
        np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    second.set_array(
        "magnetic_moments",
        np.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    with pytest.raises(StateError, match="sharing a structural-only representation"):
        build_local_environment_representation(
            [first, second],
            candidate_ids=["candidate-fm", "candidate-afm"],
            structure_ids=["structure-a", "structure-a"],
        )
    result = build_local_environment_representation(
        [first, second],
        config=LocalRepresentationConfig(magnetic_mode="non_soc"),
        candidate_ids=["candidate-fm", "candidate-afm"],
        structure_ids=["structure-a", "structure-a"],
    )
    assert not np.allclose(result.raw_descriptors[0], result.raw_descriptors[-1])


def test_neighbour_list_is_built_once_per_candidate_and_not_on_cache_hit(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Atoms] = []
    original = representation_module.neighbor_list

    def counted_neighbor_list(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(representation_module, "neighbor_list", counted_neighbor_list)
    candidates = _pool()
    config = LocalRepresentationConfig(cutoff=4.0, radial_bins=4, angular_bins=4)
    load_or_calculate_local_representations(
        tmp_path,
        candidates,
        config=config,
        local_descriptor_workers=1,
        candidate_ids=["candidate-a", "candidate-b"],
        structure_ids=["structure-a", "structure-b"],
    )
    assert len(calls) == len(candidates)
    calls.clear()
    load_or_calculate_local_representations(
        tmp_path,
        candidates,
        config=config,
        local_descriptor_workers=1,
        candidate_ids=["candidate-a", "candidate-b"],
        structure_ids=["structure-a", "structure-b"],
    )
    assert calls == []


def test_serial_and_parallel_local_descriptors_are_identical() -> None:
    candidates = _pool()
    kwargs = {
        "config": LocalRepresentationConfig(cutoff=4.0, radial_bins=4, angular_bins=4),
        "candidate_ids": ["candidate-a", "candidate-b"],
        "structure_ids": ["structure-a", "structure-b"],
    }
    serial = build_local_environment_representation(
        candidates,
        local_descriptor_workers=1,
        **kwargs,
    )
    parallel = build_local_environment_representation(
        candidates,
        local_descriptor_workers=2,
        **kwargs,
    )
    np.testing.assert_array_equal(serial.raw_descriptors, parallel.raw_descriptors)
    np.testing.assert_array_equal(serial.descriptors, parallel.descriptors)
    np.testing.assert_array_equal(serial.transform.mean, parallel.transform.mean)
    np.testing.assert_array_equal(serial.transform.eigenvalues, parallel.transform.eigenvalues)
    np.testing.assert_array_equal(serial.transform.eigenvectors, parallel.transform.eigenvectors)
    assert serial.rows == parallel.rows
    assert serial.fingerprint == parallel.fingerprint


def test_local_descriptor_progress_is_bounded_and_cache_hits_are_silent(
    tmp_path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    candidates = [
        Atoms("Si", positions=[[float(index), 0.0, 0.0]], cell=[20.0] * 3, pbc=False)
        for index in range(7)
    ]
    kwargs = {
        "config": LocalRepresentationConfig(cutoff=2.0, radial_bins=2, angular_bins=2),
        "local_descriptor_workers": 1,
        "candidate_ids": [f"candidate-{index}" for index in range(7)],
        "structure_ids": [f"structure-{index}" for index in range(7)],
    }
    caplog.set_level(logging.INFO, logger=representation_module.__name__)
    load_or_calculate_local_representations(tmp_path, candidates, **kwargs)
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Local descriptor generation:")
    ]
    assert len(messages) <= 100
    assert messages[-1].startswith("Local descriptor generation: 7/7 structures (100%)")
    counts = [int(re.search(r"(\d+)/7", message).group(1)) for message in messages]
    assert counts == sorted(set(counts))

    caplog.clear()
    load_or_calculate_local_representations(tmp_path, candidates, **kwargs)
    assert not any(
        record.getMessage().startswith("Local descriptor generation:") for record in caplog.records
    )
