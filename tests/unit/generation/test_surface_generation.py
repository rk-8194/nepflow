"""Focused coverage for Phase 6 crystalline surface generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")
from ase import Atoms
from ase.build import bulk

import nepflow.stages.generation.perturbations.coordinator as coordinator_module
from nepflow.config.models import GenerationConfig
from nepflow.errors import ConfigurationError
from nepflow.stages.generation.perturbations.coordinator import PerturbationCoordinator
from nepflow.stages.generation.perturbations.models import (
    PerturbationCounts,
    PerturbationSettings,
)
from nepflow.stages.generation.perturbations.provenance import (
    annotate_generation_provenance,
)
from nepflow.stages.generation.perturbations.surfaces import (
    SurfaceConstructionError,
    measure_surface_geometry,
    surfaces,
)
from nepflow.stages.generation.validation import (
    validate_generated_candidate,
    validate_generation_config,
)


def _annotate(candidate, base, family, **kwargs):
    return annotate_generation_provenance(candidate, base, family, **kwargs)


def _base(kind: str = "bcc") -> Atoms:
    if kind == "hcp":
        atoms = bulk("Ti", "hcp", a=2.95, c=4.68)
        source = "sqs"
    else:
        atoms = bulk("Fe", kind, a=2.86, cubic=True)
        source = "mp_phase"
    atoms.info.update({"configurational_type": source, "composition": [atoms[0].symbol]})
    return atoms


def _settings(**overrides) -> PerturbationSettings:
    values = {
        "surface_enabled": True,
        "surface_miller_indices": ((1, 0, 0),),
        "surface_layers": 3,
        "surface_vacuum": 10.0,
        "surface_sources": ("all",),
    }
    values.update(overrides)
    return PerturbationSettings(**values)


def _mock_coordinator_surface_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace real slab construction in coordinator-only tests."""

    def fake_surfaces(
        parent,
        base,
        count,
        settings,
        rng,
        annotate,
        *,
        seed=None,
        slot_start=0,
        slot_stop=None,
    ):
        del count, rng, seed
        stop = len(settings.surface_miller_indices) if slot_stop is None else slot_stop
        output = []
        for slot in range(slot_start, min(stop, len(settings.surface_miller_indices))):
            candidate = parent.copy()
            candidate.set_cell(candidate.cell * (1.0 + 0.01 * (slot + 1)), scale_atoms=True)
            miller = tuple(settings.surface_miller_indices[slot])
            candidate.info["surface_miller_index"] = miller
            annotate(
                candidate,
                base,
                "surface",
                parameters={"surface_miller_index": miller},
                operation_id=f"surface:{slot}",
            )
            output.append(candidate)
        return output

    def thread_executor_factory(*, max_workers: int, mp_context):
        del mp_context
        from concurrent.futures import ThreadPoolExecutor

        return ThreadPoolExecutor(max_workers=max_workers)

    monkeypatch.setattr(
        coordinator_module,
        "build_target_supercell",
        lambda parent, target_n_atoms: parent.copy(),
    )
    monkeypatch.setattr(coordinator_module, "require_parent_topology", lambda _atoms: None)
    monkeypatch.setattr(
        coordinator_module,
        "surface_slot_count",
        lambda _parent, settings: len(settings.surface_miller_indices),
    )
    monkeypatch.setattr(coordinator_module, "surfaces", fake_surfaces)
    monkeypatch.setattr(coordinator_module, "validate_generated_candidate", lambda *args: None)
    monkeypatch.setattr(coordinator_module, "ProcessPoolExecutor", thread_executor_factory)


def test_bcc_surface_has_low_index_termination_and_provenance() -> None:
    parent = _base()
    candidate = surfaces(parent, parent, 1, _settings(), None, _annotate, seed=17)[0]

    assert candidate.info["perturbation_type"] == "surface"
    assert (
        candidate.info["parent_structure_id"]
        == candidate.info["generation_provenance"]["parent_structure_id"]
    )
    assert candidate.info["surface_miller_index"] == (1, 0, 0)
    assert candidate.info["surface_termination"] == "termination_0"
    assert candidate.info["surface_layers"] > 0
    assert candidate.info["surface_stoichiometry_change"] == {}
    assert candidate.info["surface_half_depth"] >= 6.0
    assert candidate.info["surface_bulk_core_atom_count"] >= 1
    assert validate_generated_candidate(candidate, parent, _settings(), "surface") is None


def test_surface_planner_records_target_band_and_repeat_decision() -> None:
    parent = _base()
    settings = _settings(
        target_n_atoms=32,
        surface_target_n_atoms=32,
        surface_target_tolerance=1.0,
        surface_max_n_atoms=128,
        surface_max_in_plane_repeat=(3, 3),
        surface_max_normal_repeat=8,
    )

    candidate = surfaces(parent, parent, 1, settings, None, _annotate, seed=18)[0]
    info = candidate.info

    assert info["surface_target_n_atoms"] == 32
    assert info["surface_target_tolerance"] == pytest.approx(1.0)
    assert info["surface_realized_atom_count"] == len(candidate)
    assert len(info["surface_planner_repeat"]) == 3
    assert info["surface_in_plane_repeat"] == info["surface_planner_repeat"][:2]
    assert info["surface_target_band"] is True
    assert validate_generated_candidate(candidate, parent, settings, "surface") is None


def test_surface_planner_fails_with_explicit_atom_limit_evidence() -> None:
    parent = _base()
    settings = _settings(
        target_n_atoms=32,
        surface_max_n_atoms=1,
        surface_max_in_plane_repeat=(1, 1),
        surface_max_normal_repeat=1,
    )

    with pytest.raises(SurfaceConstructionError, match="atom_limit"):
        surfaces(parent, parent, 1, settings, None, _annotate, seed=19)


def test_surface_planner_accepts_nearest_physical_candidate_outside_target_band() -> None:
    parent = _base()
    settings = _settings(
        surface_target_n_atoms=1,
        surface_target_tolerance=0.0,
        surface_max_n_atoms=64,
        surface_max_in_plane_repeat=(2, 2),
        surface_max_normal_repeat=6,
    )

    candidate = surfaces(parent, parent, 1, settings, None, _annotate, seed=20)[0]

    assert candidate.info["surface_realized_atom_count"] > 1
    assert candidate.info["surface_target_band"] is False
    assert validate_generated_candidate(candidate, parent, settings, "surface") is None


def test_surface_geometry_uses_the_actual_oblique_surface_normal() -> None:
    first = np.asarray([2.0, 0.0, 1.0])
    second = np.asarray([0.0, 3.0, 1.0])
    normal = np.cross(first, second)
    normal /= np.linalg.norm(normal)
    third = 12.0 * normal + 0.25 * first
    candidate = Atoms(
        "Fe2",
        positions=[2.0 * normal, 6.0 * normal],
        cell=np.asarray([first, second, third]),
        pbc=(True, True, False),
    )

    geometry = measure_surface_geometry(candidate)

    assert geometry.material_thickness == pytest.approx(4.0)
    assert geometry.material_thickness != pytest.approx(
        abs(candidate.positions[1, 2] - candidate.positions[0, 2])
    )
    assert geometry.half_depth == pytest.approx(2.0)
    assert geometry.in_plane_area == pytest.approx(7.0)
    assert geometry.shortest_in_plane_translation == pytest.approx(np.sqrt(5.0))
    assert geometry.realized_vacuum == pytest.approx(8.0)


def test_non_cubic_parent_vacuum_pbc_and_in_plane_repeat() -> None:
    parent = _base("hcp")
    candidate = surfaces(
        parent,
        parent,
        1,
        _settings(
            surface_in_plane_repeat=(2, 1),
            surface_min_in_plane_dimensions=(5.0, 4.0),
            surface_vacuum=8.0,
        ),
        None,
        _annotate,
        seed=3,
    )[0]

    assert tuple(bool(value) for value in candidate.pbc) == (True, True, False)
    assert candidate.info["surface_in_plane_repeat"][0] >= 2
    assert np.linalg.norm(candidate.cell[0]) >= 5.0
    assert np.linalg.norm(candidate.cell[1]) >= 4.0
    assert candidate.info["surface_vacuum"] == 8.0
    assert candidate.info["surface_reference_basis"] == "parent_stored_cell"
    np.testing.assert_allclose(candidate.info["surface_reference_basis_cell"], parent.cell.array)
    np.testing.assert_allclose(
        candidate.info["surface_parent_to_reference_transformation"], np.eye(3, dtype=int)
    )
    assert candidate.info["surface_realized_vacuum_angstrom"] >= 8.0
    assert np.isclose(
        candidate.info["surface_realized_vacuum_angstrom"],
        candidate.info["surface_realized_vacuum"],
    )


def test_termination_order_is_deterministic_and_policy_limits_it() -> None:
    parent = _base("hcp")
    settings = _settings(surface_layers=3, surface_termination_policy="all")
    first = surfaces(parent, parent, 2, settings, None, _annotate, seed=5)
    second = surfaces(parent, parent, 2, settings, None, _annotate, seed=5)

    assert [item.info["surface_termination"] for item in first] == [
        item.info["surface_termination"] for item in second
    ]
    np.testing.assert_allclose(first[0].positions, second[0].positions)
    limited = surfaces(
        parent,
        parent,
        1,
        _settings(surface_termination_policy="first"),
        None,
        _annotate,
        seed=5,
    )
    assert len(limited) == 1
    assert limited[0].info["surface_termination"] == "termination_0"


def test_surface_source_scope_filters_coordinator(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_coordinator_surface_boundary(monkeypatch)
    parent = _base()
    settings = _settings(surface_sources=("sqs",), n_volume_points=0, elastic_stress_enabled=False)
    coordinator = PerturbationCoordinator(settings)
    candidates = coordinator.generate_candidates(
        [parent], counts=PerturbationCounts(n_rattled=0, n_surfaces=1), n_workers=1
    )

    assert all(item.info.get("perturbation_type") != "surface" for item in candidates)


def test_disabled_surfaces_do_not_invoke_surface_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_coordinator_surface_boundary(monkeypatch)
    parent = _base()
    settings = _settings(
        surface_enabled=False,
        surface_miller_indices=((1, 0, 0), (1, 1, 0), (1, 1, 1)),
        n_volume_points=0,
        elastic_stress_enabled=False,
        target_n_atoms=2,
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("disabled surface generation must not call the backend")

    monkeypatch.setattr(
        "nepflow.stages.generation.perturbations.coordinator.surfaces",
        fail_if_called,
    )
    candidates = PerturbationCoordinator(settings).generate_candidates(
        [parent],
        counts=PerturbationCounts(
            n_rattled=0,
            n_vacancies=0,
            n_interstitials=0,
            n_surfaces=99,
        ),
        n_workers=1,
    )

    assert all(item.info.get("perturbation_type") != "surface" for item in candidates)


def test_requested_orientations_are_not_truncated_by_legacy_surface_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_coordinator_surface_boundary(monkeypatch)
    parent = _base()
    requested = ((1, 0, 0), (1, 1, 0), (1, 1, 1))
    settings = _settings(
        surface_miller_indices=requested,
        surface_termination_policy="first",
        n_volume_points=0,
        elastic_stress_enabled=False,
        target_n_atoms=2,
    )
    candidates = PerturbationCoordinator(settings).generate_candidates(
        [parent],
        counts=PerturbationCounts(
            n_rattled=0,
            n_vacancies=0,
            n_interstitials=0,
            n_surfaces=1,
        ),
        n_workers=1,
    )

    realized = {
        tuple(item.info["surface_miller_index"])
        for item in candidates
        if item.info.get("perturbation_type") == "surface"
    }
    assert realized == set(requested)


def test_layer_based_vacuum_is_physical_angstrom_not_hkl_planes() -> None:
    parent = _base()
    requested_vacuum = 10.0
    candidate = surfaces(
        parent,
        parent,
        1,
        _settings(
            surface_miller_indices=((1, 1, 1),),
            surface_vacuum=requested_vacuum,
        ),
        None,
        _annotate,
        seed=11,
    )[0]

    realized_vacuum = candidate.info["surface_realized_vacuum_angstrom"]
    assert realized_vacuum >= requested_vacuum
    # For bcc Fe(111), interpreting 10 as ten hkl planes would add roughly
    # one (111) spacing per unit and produce a materially larger vacuum.
    assert realized_vacuum < 1.5 * requested_vacuum
    assert candidate.info["surface_requested_vacuum_angstrom"] == requested_vacuum


def test_surface_validation_rejects_insufficient_depth_and_tampered_measurements() -> None:
    parent = _base()
    settings = _settings()
    candidate = surfaces(parent, parent, 1, settings, None, _annotate, seed=23)[0]

    too_deep_requirement = _settings(
        surface_min_half_depth=candidate.info["surface_half_depth"] + 1.0
    )
    issue = validate_generated_candidate(candidate, parent, too_deep_requirement, "surface")
    assert issue is not None
    assert issue.reason == "insufficient_surface_depth"

    no_core_requirement = _settings(
        surface_bulk_environment_radius=candidate.info["surface_half_depth"] + 1.0
    )
    issue = validate_generated_candidate(candidate, parent, no_core_requirement, "surface")
    assert issue is not None
    assert issue.reason == "insufficient_surface_bulk_core"

    tampered = candidate.copy()
    tampered.info = dict(candidate.info)
    tampered.info["surface_in_plane_area"] += 1.0
    issue = validate_generated_candidate(tampered, parent, settings, "surface")
    assert issue is not None
    assert issue.reason == "surface_geometry_provenance_mismatch"


def test_surface_validation_rejects_tampered_normal_and_duplicate_geometry_fields() -> None:
    parent = _base()
    settings = _settings()
    candidate = surfaces(parent, parent, 1, settings, None, _annotate, seed=24)[0]

    tampered_normal = candidate.copy()
    tampered_normal.info = dict(candidate.info)
    tampered_normal.info["surface_normal"] = tuple(
        -value for value in candidate.info["surface_normal"]
    )
    issue = validate_generated_candidate(tampered_normal, parent, settings, "surface")
    assert issue is not None
    assert issue.reason == "surface_geometry_provenance_mismatch"

    tampered_vacuum = candidate.copy()
    tampered_vacuum.info = dict(candidate.info)
    tampered_vacuum.info["surface_realized_vacuum"] += 1.0
    issue = validate_generated_candidate(tampered_vacuum, parent, settings, "surface")
    assert issue is not None
    assert issue.reason == "invalid_surface_geometry_provenance"

    tampered_depth = candidate.copy()
    tampered_depth.info = dict(candidate.info)
    tampered_depth.info["surface_slab_thickness"] += 1.0
    issue = validate_generated_candidate(tampered_depth, parent, settings, "surface")
    assert issue is not None
    assert issue.reason == "surface_geometry_provenance_mismatch"


def test_multicomponent_oxide_surface_records_species_aware_bulk_core() -> None:
    parent = bulk("MgO", "rocksalt", a=4.2, cubic=True)
    parent.info.update({"configurational_type": "mp_phase", "composition": ["Mg", "O"]})
    settings = _settings(surface_miller_indices=((1, 0, 0),))

    candidate = surfaces(parent, parent, 1, settings, None, _annotate, seed=29)[0]

    assert candidate.info["surface_bulk_core_atom_count"] >= 1
    assert candidate.info["surface_bulk_core_eligible_atom_count"] >= 1
    assert validate_generated_candidate(candidate, parent, settings, "surface") is None


def test_surface_stoichiometry_change_is_recorded_for_non_parent_composition() -> None:
    parent = Atoms(
        "Fe2Ni2",
        positions=[[0, 0, 0], [1.8, 1.8, 0], [0, 1.8, 1.8], [1.8, 0, 1.8]],
        cell=np.eye(3) * 3.6,
        pbc=True,
    )
    parent.info["configurational_type"] = "mp_phase"
    parent.info["composition"] = ["Fe", "Ni"]
    candidate = surfaces(
        parent,
        parent,
        1,
        _settings(surface_miller_indices=((1, 1, 1),)),
        None,
        _annotate,
        seed=2,
    )[0]

    assert "surface_stoichiometry_change" in candidate.info
    assert "surface_composition_change" in candidate.info
    assert "surface_atom_count_change" in candidate.info


def test_invalid_miller_and_impossible_geometry_fail_explicitly() -> None:
    parent = _base()
    with pytest.raises(SurfaceConstructionError, match="invalid"):
        surfaces(parent, parent, 1, _settings(surface_miller_indices=((0, 0, 0),)), None, _annotate)

    no_cell = Atoms("Fe", positions=[[0, 0, 0]])
    with pytest.raises(SurfaceConstructionError, match="failed to construct"):
        surfaces(no_cell, no_cell, 1, _settings(), None, _annotate)

    singular_cell = Atoms(
        "Fe",
        positions=[[0.0, 0.0, 0.0]],
        cell=np.zeros((3, 3)),
        pbc=True,
    )
    with pytest.raises(SurfaceConstructionError, match="reference basis"):
        surfaces(singular_cell, singular_cell, 1, _settings(), None, _annotate)

    with pytest.raises(SurfaceConstructionError, match="only"):
        surfaces(parent, parent, 99, _settings(surface_max_terminations=1), None, _annotate)


def test_surface_config_validation_rejects_invalid_miller_and_repeat() -> None:
    with pytest.raises(ConfigurationError, match="miller"):
        validate_generation_config(GenerationConfig(surface_miller_indices=((0, 0, 0),)))
    with pytest.raises(ConfigurationError, match="unsupported orientation"):
        validate_generation_config(GenerationConfig(surface_miller_indices=((1, 0, 1),)))
    with pytest.raises(ConfigurationError, match="surface_in_plane"):
        validate_generation_config(GenerationConfig(surface_in_plane_repeat=(0, 1)))
    with pytest.raises(ConfigurationError, match="surface_min_half_depth"):
        validate_generation_config(GenerationConfig(surface_min_half_depth=0.0))
    with pytest.raises(ConfigurationError, match="surface_min_bulk_core_atoms"):
        validate_generation_config(GenerationConfig(surface_min_bulk_core_atoms=0))
    with pytest.raises(ConfigurationError, match="surface_target_tolerance"):
        validate_generation_config(GenerationConfig(surface_target_tolerance=-0.1))
    with pytest.raises(ConfigurationError, match="surface_max_in_plane_repeat"):
        validate_generation_config(
            GenerationConfig(surface_in_plane_repeat=(2, 1), surface_max_in_plane_repeat=(1, 1))
        )


def test_surface_coordinator_serial_and_parallel_are_reproducible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_coordinator_surface_boundary(monkeypatch)
    parent = _base()
    settings = _settings(
        n_volume_points=0,
        elastic_stress_enabled=False,
        target_n_atoms=2,
    )
    counts = PerturbationCounts(n_rattled=0, n_surfaces=1)
    serial = PerturbationCoordinator(settings).generate_candidates(
        [parent], counts=counts, n_workers=1
    )
    parallel = PerturbationCoordinator(settings).generate_candidates(
        [parent], counts=counts, n_workers=2
    )
    serial_surfaces = [item for item in serial if item.info.get("perturbation_type") == "surface"]
    parallel_surfaces = [
        item for item in parallel if item.info.get("perturbation_type") == "surface"
    ]

    assert len(serial_surfaces) == len(parallel_surfaces) == 1
    np.testing.assert_allclose(serial_surfaces[0].positions, parallel_surfaces[0].positions)
    assert (
        serial_surfaces[0].info["generation_provenance"]
        == parallel_surfaces[0].info["generation_provenance"]
    )
