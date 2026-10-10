"""Regression coverage for the potential-independent entropy representation."""

import logging
import re

import numpy as np
import pytest
from ase import Atoms

import nepflow.stages.selection.representations as representation_module
from nepflow.errors import StateError
from nepflow.stages.selection.representations import (
    LocalRepresentationConfig,
    build_local_environment_representation,
    fit_deterministic_whitening,
    load_or_calculate_local_representations,
)


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
