"""Regression tests for bounded generated-candidate distance validation."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ase")
from ase import Atoms

import nepflow.stages.generation.validation as validation_module
from nepflow.stages.generation.perturbations.models import PerturbationSettings
from nepflow.stages.generation.validation import validate_generated_candidate


def _candidate(
    positions: list[list[float]],
    *,
    cell: np.ndarray | None = None,
    pbc: tuple[bool, bool, bool] = (False, False, False),
) -> Atoms:
    return Atoms(
        "Si" * len(positions),
        positions=positions,
        cell=np.eye(3) * 20.0 if cell is None else cell,
        pbc=pbc,
    )


def _brute_issue(candidate: Atoms, threshold: float) -> tuple[str | None, float | None]:
    distances = candidate.get_all_distances(mic=bool(np.any(candidate.pbc)))
    upper = distances[np.triu_indices(len(candidate), k=1)]
    measured = None if upper.size == 0 else float(np.min(upper))
    if measured is not None and measured <= 1.0e-12:
        return "overlapping_atoms", measured
    if measured is not None and measured < threshold:
        return "too_close_atoms", measured
    return None, measured


@pytest.mark.parametrize("pbc", [(False, False, False), (True, False, False), (True, True, True)])
def test_bounded_search_matches_brute_force_on_small_random_cells(pbc) -> None:
    rng = np.random.RandomState(17)
    settings = PerturbationSettings(target_n_atoms=5, rattle_d_min=1.4)
    for _ in range(12):
        cell = np.array([[4.0, 0.2, 0.1], [0.1, 4.5, 0.3], [0.2, 0.4, 5.0]])
        positions = rng.random_sample((5, 3)) @ cell
        candidate = _candidate(positions.tolist(), cell=cell, pbc=pbc)
        expected_reason, expected_distance = _brute_issue(candidate, settings.rattle_d_min)
        issue = validate_generated_candidate(candidate, None, settings, "rattled")

        assert (None if issue is None else issue.reason) == expected_reason
        if expected_reason is not None:
            assert issue is not None
            assert issue.evidence["measured_distance"] == pytest.approx(expected_distance)


def test_exact_cutoff_is_accepted_and_near_cutoff_is_rejected() -> None:
    threshold = 1.5
    settings = PerturbationSettings(target_n_atoms=2, rattle_d_min=threshold)
    exact = _candidate([[0.0, 0.0, 0.0], [threshold, 0.0, 0.0]])
    near = _candidate([[0.0, 0.0, 0.0], [threshold - 1.0e-8, 0.0, 0.0]])

    assert validate_generated_candidate(exact, None, settings, "rattled") is None
    issue = validate_generated_candidate(near, None, settings, "rattled")
    assert issue is not None
    assert issue.reason == "too_close_atoms"
    assert issue.evidence["threshold"] == threshold


def test_wrapped_boundary_and_triclinic_minimum_images_are_checked() -> None:
    settings = PerturbationSettings(target_n_atoms=2, rattle_d_min=0.5)
    wrapped = _candidate(
        [[0.1, 0.1, 0.1], [9.9, 0.1, 0.1]],
        cell=np.eye(3) * 10.0,
        pbc=(True, True, True),
    )
    issue = validate_generated_candidate(wrapped, None, settings, "rattled")
    assert issue is not None
    assert issue.reason == "too_close_atoms"
    assert issue.evidence["measured_distance"] == pytest.approx(0.2)

    cell = np.array([[4.0, 0.0, 0.0], [1.5, 3.5, 0.0], [0.4, 0.6, 4.2]])
    triclinic = _candidate(
        [[0.0, 0.0, 0.0], [3.96, 0.03, 0.0]],
        cell=cell,
        pbc=(True, True, True),
    )
    brute = triclinic.get_all_distances(mic=True)[0, 1]
    triclinic_settings = PerturbationSettings(target_n_atoms=2, rattle_d_min=float(brute + 1.0e-6))
    issue = validate_generated_candidate(triclinic, None, triclinic_settings, "rattled")
    assert issue is not None
    assert issue.reason == "too_close_atoms"
    assert issue.evidence["measured_distance"] == pytest.approx(float(brute))


def test_no_pair_inside_cutoff_is_a_valid_search_result() -> None:
    candidate = _candidate([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    issue = validate_generated_candidate(
        candidate,
        None,
        PerturbationSettings(target_n_atoms=2, rattle_d_min=1.5),
        "rattled",
    )
    assert issue is None


def test_one_atom_and_periodic_small_cell_are_handled() -> None:
    one_atom = _candidate([[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=(True, True, True))
    assert (
        validate_generated_candidate(
            one_atom,
            None,
            PerturbationSettings(target_n_atoms=1, rattle_d_min=1.5),
            "rattled",
        )
        is None
    )

    small_cell = _candidate(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        cell=np.eye(3),
        pbc=(True, True, True),
    )
    issue = validate_generated_candidate(
        small_cell,
        None,
        PerturbationSettings(target_n_atoms=2, rattle_d_min=0.6),
        "rattled",
    )
    assert issue is not None
    assert issue.reason == "too_close_atoms"


def test_neighbour_search_failure_preserves_unavailable_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])

    def fail(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("controlled neighbour-list failure")

    monkeypatch.setattr(validation_module, "neighbor_list", fail)
    issue = validate_generated_candidate(
        candidate,
        None,
        PerturbationSettings(target_n_atoms=2, rattle_d_min=1.5),
        "rattled",
    )
    assert issue is not None
    assert issue.reason == "pair_distance_unavailable"


def test_large_validation_does_not_call_dense_distance_matrix() -> None:
    class NoDenseDistanceAtoms(Atoms):
        def get_all_distances(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("dense distance matrix must not be requested")

    count = 2000
    candidate = NoDenseDistanceAtoms(
        "Si" * count,
        positions=np.column_stack((np.arange(count, dtype=float) * 2.0, np.zeros((count, 2)))),
        cell=np.diag([count * 2.0, 10.0, 10.0]),
        pbc=(False, False, False),
    )
    assert (
        validate_generated_candidate(
            candidate,
            None,
            PerturbationSettings(target_n_atoms=count, rattle_d_min=1.5),
            "rattled",
        )
        is None
    )
