"""Parent-site topology preservation across supercell and point-defect paths."""

import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")
from ase import Atoms
from ase.io import read, write

from nepflow.stages.generation.perturbations.defects import (
    antisites,
    interstitials,
    substitutions,
    vacancies,
)
from nepflow.stages.generation.perturbations.models import PerturbationSettings
from nepflow.stages.generation.perturbations.provenance import (
    annotate_generation_provenance,
)
from nepflow.stages.generation.supercell import (
    ParentTopologyError,
    build_target_supercell,
    ensure_parent_topology,
    require_parent_topology,
)


def _primitive(symbols: str = "Si") -> Atoms:
    count = len(Atoms(symbols))
    return Atoms(
        symbols,
        positions=np.zeros((count, 3)),
        cell=np.eye(3) * 3.0,
        pbc=True,
    )


def _annotate(candidate, base, family, **kwargs):
    return annotate_generation_provenance(candidate, base, family, **kwargs)


def _topology_arrays(atoms: Atoms) -> dict[str, np.ndarray]:
    return {
        name: np.array(atoms.arrays[name], copy=True)
        for name in (
            "parent_site_index",
            "parent_orbit_index",
            "parent_cell_translation",
            "parent_unwrapped_fractional",
            "parent_topology_mapped",
        )
    }


def _topology_row_keys(atoms: Atoms) -> set[tuple[int, tuple[int, int, int]]]:
    return {
        (int(site), tuple(int(value) for value in translation))
        for site, translation in zip(
            atoms.arrays["parent_site_index"], atoms.arrays["parent_cell_translation"]
        )
    }


def test_repeat_records_parent_sites_and_integer_translations() -> None:
    result = build_target_supercell(_primitive(), target_n_atoms=8)
    assert result is not None
    require_parent_topology(result)
    assert result.info["topology_supercell_repeat"] == (2, 2, 2)
    np.testing.assert_array_equal(result.arrays["parent_site_index"], np.zeros(8, dtype=int))
    np.testing.assert_array_equal(
        result.arrays["parent_cell_translation"],
        np.asarray(
            [
                (0, 0, 0),
                (0, 0, 1),
                (0, 1, 0),
                (0, 1, 1),
                (1, 0, 0),
                (1, 0, 1),
                (1, 1, 0),
                (1, 1, 1),
            ]
        ),
    )
    np.testing.assert_allclose(
        result.arrays["parent_unwrapped_fractional"],
        result.arrays["parent_cell_translation"],
    )


def test_multiple_orbits_and_assignment_are_deterministic() -> None:
    parent = Atoms(
        "NaCl",
        positions=[[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]],
        cell=np.eye(3) * 3.0,
        pbc=True,
    )
    first = ensure_parent_topology(parent)
    second = ensure_parent_topology(parent)
    np.testing.assert_array_equal(first.arrays["parent_orbit_index"], [0, 1])
    np.testing.assert_array_equal(
        first.arrays["parent_orbit_index"], second.arrays["parent_orbit_index"]
    )
    assert first.info["topology_source_space_group"] == second.info["topology_source_space_group"]


def test_vacancy_and_species_transformations_retain_lattice_topology() -> None:
    parent = Atoms(
        "NaCl",
        positions=[[0, 0, 0], [1.5, 1.5, 1.5]],
        cell=np.eye(3) * 3,
        pbc=True,
    )
    base = build_target_supercell(parent, target_n_atoms=4)
    assert base is not None
    original = _topology_arrays(base)

    vacancy = vacancies(
        base,
        base,
        1,
        PerturbationSettings(vacancy_range=(0.25, 0.25), random_seed=1),
        None,
        _annotate,
        seed=1,
    )[0]
    assert len(vacancy) == len(base) - max(1, int(0.25 * len(base)))
    assert _topology_row_keys(vacancy).issubset(_topology_row_keys(base))
    assert vacancy.arrays["parent_topology_mapped"].all()

    substitution = substitutions(
        base,
        base,
        1,
        PerturbationSettings(
            substitution_pairs=(("Na", "K"),), substitution_range=(0.25, 0.25), random_seed=2
        ),
        None,
        _annotate,
        seed=2,
    )[0]
    antisite = antisites(
        base,
        base,
        1,
        PerturbationSettings(
            antisite_pairs=(("Na", "Cl"),), antisite_range=(0.25, 0.25), random_seed=3
        ),
        None,
        _annotate,
        seed=3,
    )[0]
    for child in (substitution, antisite):
        for name, values in original.items():
            np.testing.assert_array_equal(child.arrays[name], values)


def test_interstitial_topology_distinguishes_stochastic_and_explicit_mapping() -> None:
    base = build_target_supercell(_primitive(), target_n_atoms=2)
    assert base is not None
    stochastic = interstitials(
        base,
        base,
        1,
        PerturbationSettings(
            interstitial_range=(0.5, 0.5),
            interstitial_d_min=0.1,
            interstitial_max_attempts=100,
            random_seed=4,
        ),
        None,
        _annotate,
        seed=4,
    )[0]
    assert not stochastic.arrays["parent_topology_mapped"][-1]
    assert stochastic.arrays["parent_site_index"][-1] == -1
    assert np.isnan(stochastic.arrays["parent_unwrapped_fractional"][-1]).all()

    explicit = interstitials(
        base,
        base,
        1,
        PerturbationSettings(
            interstitial_range=(0.5, 0.5),
            interstitial_d_min=0.1,
            interstitial_sites=(
                {
                    "fractional": (0.25, 0.25, 0.25),
                    "parent_site_index": 0,
                    "parent_orbit_index": 0,
                    "parent_cell_translation": (0, 0, 0),
                    "parent_unwrapped_fractional": (0.25, 0.25, 0.25),
                },
            ),
            random_seed=5,
        ),
        None,
        _annotate,
        seed=5,
    )[0]
    assert explicit.arrays["parent_topology_mapped"][-1]
    assert explicit.arrays["parent_site_index"][-1] == 0
    np.testing.assert_allclose(
        explicit.arrays["parent_unwrapped_fractional"][-1], (0.25, 0.25, 0.25)
    )


def test_topology_survives_copy_slice_and_extxyz_round_trip(tmp_path) -> None:
    source = build_target_supercell(_primitive(), target_n_atoms=4)
    assert source is not None
    copied = source.copy()
    sliced = source[[0, 2]]
    for child in (copied, sliced):
        require_parent_topology(child)
    np.testing.assert_array_equal(
        sliced.arrays["parent_cell_translation"], source.arrays["parent_cell_translation"][[0, 2]]
    )

    path = tmp_path / "topology.xyz"
    write(path, source, format="extxyz")
    restored = read(path, format="extxyz")
    require_parent_topology(restored)
    for name, values in _topology_arrays(source).items():
        np.testing.assert_array_equal(restored.arrays[name], values)
    assert restored.info["parent_topology_schema"] == source.info["parent_topology_schema"]


def test_missing_or_partial_topology_fails_explicitly() -> None:
    source = _primitive()
    with pytest.raises(ParentTopologyError, match="complete parent topology"):
        require_parent_topology(source)

    source.info["perturbation_type"] = "rattled"
    with pytest.raises(ParentTopologyError, match="derived structure"):
        ensure_parent_topology(source)

    partial = _primitive()
    partial.set_array("parent_site_index", np.zeros(len(partial), dtype=int))
    with pytest.raises(ParentTopologyError, match="incomplete"):
        ensure_parent_topology(partial)
