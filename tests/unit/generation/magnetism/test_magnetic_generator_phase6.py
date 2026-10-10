"""Phase 6 magnetic candidate generation tests."""

from io import StringIO

import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")
from ase import Atoms
from ase.io import read, write

from nepflow.config.models import MagnetismConfig
from nepflow.domain.magnetism import MagneticMomentSet, MagneticStateIdentity
from nepflow.stages.generation.perturbations.magnetism import (
    MagneticGenerator,
    enumerate_orbit_phases,
    is_commensurate,
)
from nepflow.stages.generation.supercell import build_target_supercell, mark_added_atoms_unmapped


def _config(**kwargs) -> MagnetismConfig:
    values = {
        "enabled": True,
        "target_potential_magnetic": True,
        "include_non_magnetic": True,
        "include_ferromagnetic": True,
        "include_antiferromagnetic": True,
        "moment_sets": (MagneticMomentSet("nominal", {"Fe": 2.5}),),
        "max_afm_orderings": 100,
        "max_magnetic_variants_per_parent": 100,
    }
    values.update(kwargs)
    return MagnetismConfig(**values)


def _supercell(symbols: str = "Fe", target: int = 8) -> Atoms:
    n_parent = len(Atoms(symbols))
    parent = Atoms(
        symbols,
        positions=np.asarray([[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]][:n_parent]),
        cell=np.eye(3) * 3.0,
        pbc=True,
    )
    result = build_target_supercell(parent, target_n_atoms=target)
    assert result is not None
    return result


def test_nm_fm_and_ligand_mask() -> None:
    result = MagneticGenerator(_config()).generate_result(_supercell("FeO", 4))

    assert result.candidates[0].info["magnetic_ordering"] == "nonmagnetic"
    fm = next(item for item in result.candidates if item.info["magnetic_ordering"] == "fm")
    assert fm.arrays["magnetic_moments"][:, 2].tolist() == [2.5, 0.0] * (len(fm) // 2)
    assert fm.arrays["magnetic_constraint_mask"].tolist() == [True, False] * (len(fm) // 2)


def test_simple_bipartite_afm_and_propagation_sign_pattern() -> None:
    result = MagneticGenerator(
        _config(include_non_magnetic=False, include_ferromagnetic=False)
    ).generate_result(_supercell("Fe", 8))
    target = next(
        item
        for item in result.candidates
        if tuple(item.info["magnetic_propagation_vector"]) == (0.5, 0.5, 0.5)
    )
    expected = [
        2.5 * (-1) ** int(sum(translation))
        for translation in target.arrays["parent_cell_translation"]
    ]
    np.testing.assert_allclose(target.arrays["magnetic_moments"][:, 2], expected)


def test_pure_cr_parent_keeps_normal_fm_and_afm_generation() -> None:
    result = MagneticGenerator(
        _config(moment_sets=(MagneticMomentSet("nominal", {"Cr": 2.5}),))
    ).generate_result(_supercell("Cr", 8))

    assert result.summary.configured_magnetic_site_count == 8
    assert result.summary.emitted_ferromagnetic == 1
    assert result.summary.emitted_antiferromagnetic > 0
    assert not any(
        diagnostic.code == "NO_MAGNETIC_SITES" for diagnostic in result.summary.diagnostics
    )


def test_multi_orbit_afm_enumerates_intracell_phase() -> None:
    parent = Atoms(
        "FeNi",
        positions=[[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]],
        cell=np.eye(3) * 3.0,
        pbc=True,
    )
    base = build_target_supercell(parent, target_n_atoms=4)
    assert base is not None
    config = _config(
        include_non_magnetic=False,
        include_ferromagnetic=False,
        moment_sets=(MagneticMomentSet("mixed", {"Fe": 2.0, "Ni": 1.0}),),
    )
    result = MagneticGenerator(config).generate_result(base)
    assert any(
        tuple(item.info["magnetic_propagation_vector"]) == (0.0, 0.0, 0.0)
        and tuple(item.info["magnetic_orbit_phases"]) == (1.0, -1.0)
        for item in result.candidates
    )


def test_half_period_commensurability_and_noncommensurate_rejection() -> None:
    assert is_commensurate((0.5, 0.0, 0.0), np.diag([2, 2, 2]))
    assert not is_commensurate((0.5, 0.0, 0.0), np.diag([3, 3, 3]))
    result = MagneticGenerator(
        _config(include_non_magnetic=False, include_ferromagnetic=False)
    ).generate_result(_supercell("Fe", 27))
    assert any(
        diagnostic.code == "INCOMMENSURATE_PROPAGATION"
        and diagnostic.propagation_vector == (0.5, 0.0, 0.0)
        for diagnostic in result.summary.diagnostics
    )


def test_orbit_phase_enumeration_and_global_inversion_deduplication() -> None:
    assert enumerate_orbit_phases([2, 0, 1]) == ((1, 1, 1), (1, -1, 1), (1, 1, -1), (1, -1, -1))
    first = MagneticStateIdentity.from_components(
        [[0.0, 0.0, 2.0], [0.0, 0.0, -1.0]], [True, True], ordering="afm"
    )
    inverted = MagneticStateIdentity.from_components(
        [[0.0, 0.0, -2.0], [0.0, 0.0, 1.0]], [True, True], ordering="afm"
    )
    assert first.magnetic_state_id == inverted.magnetic_state_id


def test_unmapped_magnetic_site_skips_afm_with_diagnostic() -> None:
    base = _supercell("Fe", 2)
    base += Atoms("Fe", positions=[[0.3, 0.3, 0.3]])
    mark_added_atoms_unmapped(base, 2)
    result = MagneticGenerator(
        _config(include_non_magnetic=False, include_ferromagnetic=True)
    ).generate_result(base)
    assert [item.info["magnetic_ordering"] for item in result.candidates] == ["fm"]
    assert any(
        diagnostic.code == "UNMAPPED_MAGNETIC_SITE" for diagnostic in result.summary.diagnostics
    )


def test_deterministic_order_and_explicit_limit() -> None:
    base = _supercell("Fe", 8)
    limited = _config(max_afm_orderings=2, include_non_magnetic=False, include_ferromagnetic=False)
    first = MagneticGenerator(limited).generate_result(base)
    second = MagneticGenerator(limited).generate_result(base)
    assert [item.info["candidate_id"] for item in first.candidates] == [
        item.info["candidate_id"] for item in second.candidates
    ]
    assert first.summary.retained_afm == 2
    assert first.summary.budget_truncated


def test_extxyz_round_trip_and_candidate_identity() -> None:
    result = MagneticGenerator(_config()).generate_result(_supercell("FeO", 4))
    output = StringIO()
    write(output, list(result.candidates), format="extxyz")
    restored = read(StringIO(output.getvalue()), index=":", format="extxyz")
    assert isinstance(restored, list)
    assert len(restored) == len(result.candidates)
    for original, loaded in zip(result.candidates, restored):
        np.testing.assert_allclose(
            original.arrays["magnetic_moments"], loaded.arrays["magnetic_moments"]
        )
        assert (
            loaded.arrays["magnetic_constraint_mask"].tolist()
            == original.arrays["magnetic_constraint_mask"].tolist()
        )
        assert loaded.info["candidate_id"] == original.info["candidate_id"]
        assert loaded.info["structure_id"] == original.info["structure_id"]

    structure_ids = {item.info["structure_id"] for item in result.candidates}
    candidate_ids = {item.info["candidate_id"] for item in result.candidates}
    assert len(structure_ids) == 1
    assert len(candidate_ids) == len(result.candidates)


def test_expansion_scope_does_not_touch_unselected_structural_families() -> None:
    pristine = _supercell("Fe", 2)
    pristine.info["perturbation_type"] = "unperturbed"
    rattled = pristine.copy()
    rattled.info["perturbation_type"] = "rattled"
    config = _config(
        include_non_magnetic=False,
        include_antiferromagnetic=False,
        defect_families=(),
    )
    result = MagneticGenerator(config).expand_structures([pristine, rattled])
    assert len(result.candidates) == 2
    assert result.candidates[0].info["magnetic_ordering"] == "fm"
    assert result.candidates[0].info["perturbation_type"] == "unperturbed"
    assert "magnetic_ordering" not in result.candidates[1].info


def test_structurally_selected_parent_without_configured_sites_emits_only_nm() -> None:
    result = MagneticGenerator(
        _config(moment_sets=(MagneticMomentSet("nominal", {"Cr": 2.5}),))
    ).expand_structures([_supercell("W", 4)])

    assert len(result.candidates) == 1
    assert result.candidates[0].info["magnetic_ordering"] == "nonmagnetic"
    assert result.summary.selected_structural_parents == 1
    assert result.summary.eligible_structural_parents == 1
    assert result.summary.selected_with_configured_magnetic_sites == 0
    assert result.summary.selected_without_configured_magnetic_sites == 1
    assert result.summary.zero_output_failures == 0
    assert result.summary.emitted_ferromagnetic == 0
    assert result.summary.emitted_antiferromagnetic == 0
    assert any(
        diagnostic.code == "NO_MAGNETIC_SITES" for diagnostic in result.summary.diagnostics
    )
