"""Explicit Phase 6 defect-to-magnetic chaining tests."""

import json

import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")
from ase import Atoms

from nepflow.config.models import MagnetismConfig
from nepflow.domain.magnetism import MagneticMomentSet
from nepflow.stages.generation.perturbations import PerturbationCoordinator
from nepflow.stages.generation.perturbations.defects import (
    antisites,
    substitutions,
)
from nepflow.stages.generation.perturbations.magnetism import (
    MagneticGenerator,
    UnsupportedMagneticTopologyError,
)
from nepflow.stages.generation.perturbations.models import (
    PerturbationCounts,
    PerturbationSettings,
)
from nepflow.stages.generation.perturbations.provenance import (
    annotate_generation_provenance,
)
from nepflow.stages.generation.supercell import build_target_supercell, mark_added_atoms_unmapped


def _annotate(candidate, base, family, **kwargs):
    return annotate_generation_provenance(candidate, base, family, **kwargs)


def _base(symbols: str = "Fe", target: int = 8) -> Atoms:
    n_atoms = len(Atoms(symbols))
    parent = Atoms(
        symbols,
        positions=np.asarray([[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]][:n_atoms]),
        cell=np.eye(3) * 3.0,
        pbc=True,
    )
    parent.info["configurational_type"] = "random_solid_solution"
    result = build_target_supercell(parent, target_n_atoms=target)
    assert result is not None
    return result


def _config(**overrides) -> MagnetismConfig:
    values = {
        "enabled": True,
        "target_potential_magnetic": True,
        "include_non_magnetic": False,
        "include_ferromagnetic": False,
        "include_antiferromagnetic": True,
        "moment_sets": (MagneticMomentSet("nominal", {"Fe": 2.0, "Ni": 1.0, "Co": 2.0}),),
        "max_afm_orderings": 100,
        "max_magnetic_variants_per_parent": 100,
        "max_magnetic_variants_per_defect": 100,
    }
    values.update(overrides)
    return MagnetismConfig(**values)


def test_pristine_and_vacancy_are_explicit_coordinator_chains() -> None:
    settings = PerturbationSettings(
        target_n_atoms=8,
        vacancy_range=(0.125, 0.125),
        n_volume_points=0,
        elastic_stress_enabled=False,
    )
    coordinator = PerturbationCoordinator(
        settings,
        magnetic_generator=MagneticGenerator(
            _config(max_afm_orderings=1, defect_families=("vacancy",))
        ),
    )
    candidates = coordinator.generate_candidates(
        [_base()],
        counts=PerturbationCounts(n_rattled=0, n_vacancies=1, n_interstitials=0),
        n_workers=1,
    )

    assert [candidate.info["perturbation_type"] for candidate in candidates] == [
        "unperturbed",
        "vacancy",
    ]
    assert all(candidate.info["magnetic_ordering"] == "afm" for candidate in candidates)
    assert len(candidates[1]) == len(candidates[0]) - 1
    assert candidates[1].arrays["parent_topology_mapped"].all()
    for candidate in candidates:
        provenance = json.loads(candidate.info["magnetic_provenance"])
        assert provenance["source_structure_id"] == candidate.info["magnetic_parent_structure_id"]
        assert (
            candidate.info["generation_provenance"]["perturbation_family"]
            == candidate.info["perturbation_type"]
        )


def test_substitution_and_antisite_retain_lattice_topology_for_afm() -> None:
    base = _base("FeNi", 4)
    substitution = substitutions(
        base,
        base,
        1,
        PerturbationSettings(
            substitution_pairs=(("Fe", "Co"),),
            substitution_range=(0.125, 0.125),
            random_seed=11,
        ),
        None,
        _annotate,
        seed=11,
    )[0]
    antisite = antisites(
        base,
        base,
        1,
        PerturbationSettings(
            antisite_pairs=(("Fe", "Ni"),),
            antisite_range=(0.125, 0.125),
            random_seed=12,
        ),
        None,
        _annotate,
        seed=12,
    )[0]
    generator = MagneticGenerator(_config())

    for defect in (substitution, antisite):
        result = generator.generate_result(defect)
        assert result.candidates
        assert all(candidate.info["magnetic_ordering"] == "afm" for candidate in result.candidates)
        np.testing.assert_array_equal(
            result.candidates[0].arrays["parent_site_index"],
            defect.arrays["parent_site_index"],
        )


def test_nonmagnetic_interstitial_is_zero_and_unconstrained_in_afm_host() -> None:
    base = _base("Fe", 2)
    defect = base.copy()
    defect += Atoms("O", positions=[[0.3, 0.3, 0.3]])
    mark_added_atoms_unmapped(defect, len(base))
    defect.info["perturbation_type"] = "interstitial"
    result = MagneticGenerator(_config(defect_families=("interstitial",))).expand_structures(
        [defect]
    )

    assert result.candidates
    candidate = result.candidates[0]
    assert candidate.arrays["magnetic_moments"][-1].tolist() == [0.0, 0.0, 0.0]
    assert not candidate.arrays["magnetic_constraint_mask"][-1]
    assert candidate.info["magnetic_ordering"] == "afm"


def test_unmapped_magnetic_interstitial_is_omitted_or_rejected_never_guessed() -> None:
    base = _base("Fe", 2)
    defect = base.copy()
    defect += Atoms("Fe", positions=[[0.3, 0.3, 0.3]])
    mark_added_atoms_unmapped(defect, len(base))
    defect.info["perturbation_type"] = "interstitial"

    omitted = MagneticGenerator(
        _config(defect_families=("interstitial",), unmapped_site_policy="skip_afm")
    ).expand_structures([defect])
    assert not omitted.candidates
    assert any(
        diagnostic.code == "UNMAPPED_MAGNETIC_SITE" for diagnostic in omitted.summary.diagnostics
    )

    with pytest.raises(UnsupportedMagneticTopologyError):
        MagneticGenerator(
            _config(defect_families=("interstitial",), unmapped_site_policy="reject")
        ).expand_structures([defect])


def test_defect_parent_limit_is_independent_and_scope_is_explicit() -> None:
    first = _base()
    first.info["perturbation_type"] = "vacancy"
    second = first.copy()
    second.positions[0] += [0.1, 0.0, 0.0]
    volume = first.copy()
    volume.info["perturbation_type"] = "volume_profile"
    config = _config(
        defect_families=("vacancy",),
        max_defect_parents=1,
        max_magnetic_variants_per_defect=1,
    )
    generator = MagneticGenerator(config)
    first_run = generator.expand_structures([first, second, volume])
    second_run = generator.expand_structures([first, second, volume])

    assert first_run.candidates[0].info["magnetic_ordering"] == "afm"
    assert "magnetic_ordering" not in first_run.candidates[1].info
    assert "magnetic_ordering" not in first_run.candidates[2].info
    assert [item.info["candidate_id"] for item in first_run.candidates] == [
        item.info["candidate_id"] for item in second_run.candidates
    ]


def test_defect_magnetism_is_opt_in_per_family() -> None:
    vacancy = _base()
    vacancy.info["perturbation_type"] = "vacancy"

    without_policy = MagneticGenerator(_config()).expand_structures([vacancy])
    with_policy = MagneticGenerator(_config(defect_families=("vacancy",))).expand_structures(
        [vacancy]
    )

    assert len(without_policy.candidates) == 1
    assert "magnetic_ordering" not in without_policy.candidates[0].info
    assert with_policy.candidates
    assert all(item.info["magnetic_ordering"] == "afm" for item in with_policy.candidates)


@pytest.mark.parametrize(
    "family",
    ("volume_profile", "elastic_stress", "rattled", "liquid", "surface", "grain_boundary"),
)
def test_structural_families_remain_unexpanded(family: str) -> None:
    structural = _base()
    structural.info["perturbation_type"] = family

    result = MagneticGenerator(
        _config(defect_families=("vacancy", "interstitial", "substitution", "antisite"))
    ).expand_structures([structural])

    assert len(result.candidates) == 1
    assert result.candidates[0].info["perturbation_type"] == family
    assert "magnetic_ordering" not in result.candidates[0].info
