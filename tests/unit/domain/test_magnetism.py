from io import StringIO

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from nepflow.domain import (
    CandidateIdentity,
    MagneticOrdering,
    MagneticState,
    MagneticStateIdentity,
    calculate_candidate_id,
    calculate_structure_id,
)


def test_magnetic_state_identity_is_deterministic_and_serializable() -> None:
    first = MagneticStateIdentity.from_components(
        [[0.0, 0.0, 2.5], [0.0, 0.0, -2.5]],
        [True, True],
        ordering="afm",
        propagation_vector=(0.5, 0.0, 0.0),
        orbit_phases=(0.0, 0.5),
    )
    second = MagneticStateIdentity.from_components(
        np.asarray([[0.0, 0.0, 2.5], [0.0, 0.0, -2.5]]),
        (True, True),
        ordering=MagneticOrdering.AFM,
        propagation_vector=(0.5, 0.0, 0.0),
        orbit_phases=(0.0, 0.5),
    )

    assert first.magnetic_state_id == second.magnetic_state_id
    assert first.to_dict()["moments"] == [list(vector) for vector in first.moments]
    assert all(isinstance(value, (bool, int)) for value in first.to_dict()["constraint_mask"])


def test_global_spin_inversion_is_equivalent_but_state_changes_are_not() -> None:
    state = MagneticStateIdentity.from_components(
        [[0.0, 0.0, 2.0], [0.0, 0.0, -1.0]],
        [True, False],
        ordering="afm",
    )
    inverted = MagneticStateIdentity.from_components(
        [[0.0, 0.0, -2.0], [0.0, 0.0, 1.0]],
        [True, False],
        ordering="afm",
    )
    changed = MagneticStateIdentity.from_components(
        [[0.0, 0.0, 2.1], [0.0, 0.0, -1.0]],
        [True, False],
        ordering="afm",
    )

    assert state.magnetic_state_id == inverted.magnetic_state_id
    assert state.magnetic_state_id != changed.magnetic_state_id
    assert (
        MagneticStateIdentity.from_components(
            [[0.0, 0.0, 2.0], [0.0, 0.0, -1.0]],
            [True, False],
            ordering="afm",
            global_spin_inversion_equivalent=False,
        ).magnetic_state_id
        != MagneticStateIdentity.from_components(
            [[0.0, 0.0, -2.0], [0.0, 0.0, 1.0]],
            [True, False],
            ordering="afm",
            global_spin_inversion_equivalent=False,
        ).magnetic_state_id
    )


def test_identity_uses_structure_canonical_atom_order() -> None:
    atoms = Atoms(
        ["Fe", "O", "Fe"],
        scaled_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]],
        cell=np.eye(3) * 4.0,
        pbc=True,
    )
    state_from_atoms = MagneticStateIdentity.from_atoms(
        atoms,
        [[0.0, 0.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, -2.0]],
        [True, False, True],
        ordering="afm",
    )
    state_already_canonical = MagneticStateIdentity.from_components(
        [[0.0, 0.0, 2.0], [0.0, 0.0, -2.0], [0.0, 0.0, 0.0]],
        [True, True, False],
        ordering="afm",
    )

    assert state_from_atoms.magnetic_state_id == state_already_canonical.magnetic_state_id


def test_same_structure_with_different_magnetic_states_has_distinct_candidate_ids() -> None:
    atoms = Atoms("Fe2", positions=[[0, 0, 0], [1, 1, 1]], cell=np.eye(3) * 3, pbc=True)
    structure_id = calculate_structure_id(atoms)
    fm = MagneticState(
        ordering=MagneticOrdering.FM,
        moment_set_name="medium",
        moments=((0.0, 0.0, 2.0), (0.0, 0.0, 2.0)),
        constraint_mask=(True, True),
    )
    afm = MagneticState(
        ordering=MagneticOrdering.AFM,
        moment_set_name="medium",
        moments=((0.0, 0.0, 2.0), (0.0, 0.0, -2.0)),
        constraint_mask=(True, True),
    )

    assert calculate_candidate_id(structure_id, fm) != calculate_candidate_id(structure_id, afm)
    assert CandidateIdentity(structure_id).candidate_id == structure_id
    assert not CandidateIdentity(structure_id).is_magnetic


def test_extxyz_serialization_fields_are_round_trip_safe() -> None:
    state = MagneticState(
        ordering=MagneticOrdering.FM,
        moment_set_name="nominal",
        moments=((0.0, 0.0, 2.0), (0.0, 0.0, 2.0)),
        constraint_mask=(True, True),
    )
    atoms = Atoms("Fe2", positions=[[0, 0, 0], [1, 1, 1]], cell=np.eye(3) * 3, pbc=True)
    arrays = state.to_extxyz_arrays()
    atoms.new_array("magnetic_moments", np.asarray(arrays["magnetic_moments"], dtype=float))
    atoms.new_array(
        "magnetic_constraint_mask",
        np.asarray(arrays["magnetic_constraint_mask"], dtype=bool),
    )
    candidate_id = calculate_candidate_id(calculate_structure_id(atoms), state)
    atoms.info.update(state.to_extxyz_info(candidate_id=candidate_id))

    output = StringIO()
    write(output, atoms, format="extxyz")
    restored = read(StringIO(output.getvalue()), format="extxyz")

    np.testing.assert_allclose(
        restored.arrays["magnetic_moments"], atoms.arrays["magnetic_moments"]
    )
    assert restored.arrays["magnetic_constraint_mask"].tolist() == [True, True]
    assert restored.info["magnetic_state_id"] == state.magnetic_state_id
    assert restored.info["candidate_id"] == candidate_id


def test_unsupported_ordering_and_invalid_moment_shape_fail() -> None:
    with pytest.raises(ValueError, match="unsupported magnetic ordering"):
        MagneticOrdering.parse("spiral")
    with pytest.raises(ValueError, match="three components"):
        MagneticStateIdentity.from_components([[1.0, 0.0]], [True])
