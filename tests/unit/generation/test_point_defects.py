"""Focused coverage for Phase 6 point-defect realization semantics."""

from collections import Counter

import numpy as np
import pytest

pytest.importorskip("ase")
from ase import Atoms
from ase.io import read, write

from nepflow.domain.identities import calculate_structure_id
from nepflow.stages.generation.generators.segregated import SegregatedGenerator
from nepflow.stages.generation.perturbations.coordinator import PerturbationCoordinator
from nepflow.stages.generation.perturbations.defects import (
    antisites,
    find_interstitial_site,
    interstitials,
    substitutions,
    vacancies,
)
from nepflow.stages.generation.perturbations.models import PerturbationSettings, derive_child_seed
from nepflow.stages.generation.perturbations.provenance import (
    annotate_generation_provenance,
)
from nepflow.stages.generation.validation import validate_generated_candidate


def _base() -> Atoms:
    atoms = Atoms(
        "Si2Ge2",
        positions=[[1, 1, 1], [8, 1, 1], [1, 8, 1], [1, 1, 8]],
        cell=np.eye(3) * 10.0,
        pbc=True,
    )
    atoms.info.update(
        {
            "elements": ["Si", "Ge"],
            "source": "point-defect-test",
            "configurational_type": "test_base",
        }
    )
    return atoms


def _annotate(candidate, base, family, **kwargs):
    return annotate_generation_provenance(candidate, base, family, **kwargs)


def _prepared_segregated_base() -> tuple[Atoms, Atoms, PerturbationSettings]:
    base = SegregatedGenerator(n_structures=1, random_seed=42).generate(
        {"Si": 0.5, "Ge": 0.5}, ["bcc"], target_n_atoms=128
    )[0]
    settings = PerturbationSettings(
        target_n_atoms=128,
        random_seed=1572714583,
        vacancy_range=(0.01, 0.01),
    )
    prepared = PerturbationCoordinator(settings=settings)._prepare_base_supercell(base, settings)
    return base, prepared, settings


def _assert_same_structure_and_provenance(left: Atoms, right: Atoms) -> None:
    assert calculate_structure_id(left) == calculate_structure_id(right)
    np.testing.assert_array_equal(left.numbers, right.numbers)
    np.testing.assert_allclose(left.positions, right.positions)
    np.testing.assert_allclose(left.cell.array, right.cell.array)
    assert left.info == right.info


def test_segregated_vacancy_slot_windows_are_continuous_and_deterministic() -> None:
    base, prepared, settings = _prepared_segregated_base()
    seed = 1572714583

    full = vacancies(prepared, base, 10, settings, None, _annotate, seed=seed, slot_start=0)
    first = vacancies(prepared, base, 8, settings, None, _annotate, seed=seed, slot_start=0)
    second = vacancies(
        prepared,
        base,
        2,
        settings,
        None,
        _annotate,
        seed=seed,
        slot_start=8,
        slot_stop=10,
    )

    for expected, actual in zip(full, first + second):
        _assert_same_structure_and_provenance(expected, actual)
    assert [item.info["random_seed"] for item in second] == [
        derive_child_seed(calculate_structure_id(base), seed, "vacancy", slot) for slot in (8, 9)
    ]
    assert [item.info["generation_provenance"]["operation_id"] for item in second] == [
        "vacancy:8",
        "vacancy:9",
    ]


def test_species_restricted_vacancy_and_impossible_species_failure() -> None:
    base = _base()
    settings = PerturbationSettings(
        vacancy_range=(0.25, 0.25), vacancy_species=("Ge",), random_seed=11
    )
    result = vacancies(base, base, 1, settings, None, _annotate, seed=11)[0]

    assert result.get_chemical_symbols().count("Ge") == 1
    assert result.info["affected_species"] == "Ge"
    assert result.info["requested_n_vacancies"] == 1
    assert result.info["realised_n_vacancies"] == 1
    assert result.info["requested_vacancy_concentration"] == 0.25
    assert result.info["realised_vacancy_concentration"] == 0.25

    with pytest.raises(ValueError, match="vacancy_species"):
        vacancies(
            base,
            base,
            1,
            PerturbationSettings(vacancy_species=("H",)),
            None,
            _annotate,
            seed=11,
        )


def test_substitution_and_antisite_species_semantics() -> None:
    base = _base()
    substitution = substitutions(
        base,
        base,
        1,
        PerturbationSettings(
            substitution_pairs=(("Si", "Ge"),),
            substitution_range=(0.25, 0.25),
            random_seed=13,
        ),
        None,
        _annotate,
        seed=13,
    )[0]
    assert Counter(substitution.get_chemical_symbols()) == Counter({"Ge": 3, "Si": 1})
    assert substitution.info["source_species"] == "Si"
    assert substitution.info["target_species"] == "Ge"
    assert substitution.info["affected_species"] == "Si,Ge"

    antisite = antisites(
        base,
        base,
        1,
        PerturbationSettings(
            antisite_pairs=(("Si", "Ge"),), antisite_range=(0.25, 0.25), random_seed=13
        ),
        None,
        _annotate,
        seed=13,
    )[0]
    assert Counter(antisite.get_chemical_symbols()) == Counter(base.get_chemical_symbols())
    assert antisite.get_chemical_symbols() != base.get_chemical_symbols()


def test_crystallographic_and_stochastic_interstitials_are_seeded() -> None:
    base = _base()
    configured = PerturbationSettings(
        interstitial_range=(0.25, 0.25),
        interstitial_sites=((0.5, 0.5, 0.5),),
        interstitial_d_min=1.0,
        random_seed=17,
    )
    configured_result = interstitials(base, base, 1, configured, None, _annotate, seed=17)[0]
    np.testing.assert_allclose(configured_result.positions[-1], [5.0, 5.0, 5.0])
    assert configured_result.info["interstitial_site_mode"] == "crystallographic"

    stochastic = PerturbationSettings(
        interstitial_range=(0.25, 0.25), interstitial_d_min=1.0, random_seed=19
    )
    first = interstitials(base, base, 1, stochastic, None, _annotate, seed=19)[0]
    second = interstitials(base, base, 1, stochastic, None, _annotate, seed=19)[0]
    np.testing.assert_allclose(first.positions, second.positions)
    assert first.info == second.info


def test_defect_and_periodic_image_separation_and_rejection() -> None:
    base = _base()
    settings = PerturbationSettings(
        interstitial_range=(0.5, 0.5),
        interstitial_sites=((0.45, 0.5, 0.5), (0.46, 0.5, 0.5)),
        interstitial_d_min=1.0,
        defect_defect_d_min=0.5,
        random_seed=23,
    )
    partial = interstitials(base, base, 1, settings, None, _annotate, seed=23)[0]
    assert partial.info["requested_n_interstitials"] == 2
    assert partial.info["realised_n_interstitials"] == 1
    issue = validate_generated_candidate(partial, base, settings, "interstitial")
    assert issue is not None
    assert issue.reason == "partial_interstitial_placement"

    cell = np.eye(3) * 5.0
    assert (
        find_interstitial_site(
            np.empty((0, 3)),
            cell,
            np.linalg.inv(cell),
            np.random.RandomState(1),
            d_min=0.0,
            max_attempts=3,
            periodic_image_d_min=5.1,
        )
        is None
    )


def test_point_defect_metadata_survives_extxyz_round_trip(tmp_path) -> None:
    base = _base()
    candidate = substitutions(
        base,
        base,
        1,
        PerturbationSettings(substitution_pairs=(("Si", "Ge"),), substitution_range=(0.25, 0.25)),
        None,
        _annotate,
        seed=29,
    )[0]
    path = tmp_path / "point_defect.xyz"
    write(path, candidate, format="extxyz")
    restored = read(path, format="extxyz")

    assert restored.info["perturbation_type"] == "substitution"
    assert restored.info["source_species"] == "Si"
    assert restored.info["target_species"] == "Ge"
    assert restored.info["affected_species"] == "Si,Ge"
    assert restored.info["requested_n_substitutions"] == 1
    assert restored.info["realised_n_substitutions"] == 1
