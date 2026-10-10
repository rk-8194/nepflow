"""Regression coverage for the potential-independent entropy representation."""

import numpy as np
import pytest
from ase import Atoms

from nepflow.errors import StateError
from nepflow.stages.selection.representations import (
    LocalRepresentationConfig,
    build_local_environment_representation,
    fit_deterministic_whitening,
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
