"""Focused coverage for Phase 6 crystalline surface generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")
from ase import Atoms
from ase.build import bulk

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
    assert validate_generated_candidate(candidate, parent, _settings(), "surface") is None


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


def test_surface_source_scope_filters_coordinator() -> None:
    parent = _base()
    settings = _settings(surface_sources=("sqs",), n_volume_points=0, elastic_stress_enabled=False)
    coordinator = PerturbationCoordinator(settings)
    candidates = coordinator.generate_candidates(
        [parent], counts=PerturbationCounts(n_rattled=0, n_surfaces=1), n_workers=1
    )

    assert all(item.info.get("perturbation_type") != "surface" for item in candidates)


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

    with pytest.raises(SurfaceConstructionError, match="only"):
        surfaces(parent, parent, 99, _settings(surface_max_terminations=1), None, _annotate)


def test_surface_config_validation_rejects_invalid_miller_and_repeat() -> None:
    with pytest.raises(ConfigurationError, match="miller"):
        validate_generation_config(GenerationConfig(surface_miller_indices=((0, 0, 0),)))
    with pytest.raises(ConfigurationError, match="surface_in_plane"):
        validate_generation_config(GenerationConfig(surface_in_plane_repeat=(0, 1)))


def test_surface_coordinator_serial_and_parallel_are_reproducible(tmp_path: Path) -> None:
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
