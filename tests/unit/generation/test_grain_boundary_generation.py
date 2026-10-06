"""Focused coverage for the supported Phase 6 grain-boundary path."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")
from ase import Atoms
from ase.build import bulk

from nepflow.errors import ConfigurationError
from nepflow.stages.generation.perturbations.coordinator import PerturbationCoordinator
from nepflow.stages.generation.perturbations.grain_boundaries import (
    GrainBoundaryConstructionError,
    grain_boundaries,
)
from nepflow.stages.generation.perturbations.models import (
    PerturbationCounts,
    PerturbationSettings,
)
from nepflow.stages.generation.perturbations.provenance import (
    annotate_generation_provenance,
)
from nepflow.stages.generation.validation import validate_generated_candidate


def _parent() -> Atoms:
    atoms = bulk("Fe", "bcc", a=2.86, cubic=True)
    atoms.info.update(
        {
            "configurational_type": "mp_phase",
            "composition": ["Fe"],
            "crystal_structure": "bcc",
        }
    )
    return atoms


def _settings(**overrides) -> PerturbationSettings:
    values = {
        "grain_boundary_enabled": True,
        "grain_boundary_sources": ("all",),
        "grain_boundary_expand_times": 2,
        "grain_boundary_overlap_tolerance": 0.7,
        "n_volume_points": 0,
        "elastic_stress_enabled": False,
    }
    values.update(overrides)
    return PerturbationSettings(**values)


def _annotate(candidate, base, family, **kwargs):
    return annotate_generation_provenance(candidate, base, family, **kwargs)


def test_supported_sigma5_symmetric_tilt_has_valid_geometry() -> None:
    parent = _parent()
    settings = _settings()
    candidate = grain_boundaries(parent, parent, 1, settings, None, _annotate, seed=13)[0]

    assert len(candidate) > 0
    assert tuple(bool(value) for value in candidate.pbc) == (True, True, True)
    assert np.linalg.matrix_rank(np.asarray(candidate.cell, dtype=float)) == 3
    assert candidate.info["grain_boundary_relationship"] == ("sigma5_[001]_36.869897645844_(210)")
    assert candidate.info["grain_boundary_sigma"] == 5
    assert candidate.info["grain_boundary_rotation_axis"] == (0, 0, 1)
    assert candidate.info["grain_boundary_plane"] == (2, 1, 0)
    assert validate_generated_candidate(candidate, parent, settings, "grain_boundary") is None


def test_grain_boundary_construction_is_deterministic() -> None:
    parent = _parent()
    settings = _settings()
    first = grain_boundaries(parent, parent, 1, settings, None, _annotate, seed=9)[0]
    second = grain_boundaries(parent, parent, 1, settings, None, _annotate, seed=9)[0]

    np.testing.assert_allclose(first.cell.array, second.cell.array)
    np.testing.assert_allclose(first.positions, second.positions)
    assert first.info == second.info


def test_overlap_removal_is_accounted_for_with_species_and_tolerance() -> None:
    parent = _parent()
    settings = _settings(grain_boundary_overlap_tolerance=0.7)
    candidate = grain_boundaries(parent, parent, 1, settings, None, _annotate, seed=1)[0]

    removed = candidate.info["grain_boundary_removed_species"]
    assert candidate.info["grain_boundary_removed_atom_count"] == sum(removed.values())
    assert candidate.info["grain_boundary_removed_atom_count"] > 0
    assert candidate.info["grain_boundary_overlap_tolerance"] == 0.7
    assert candidate.info["grain_boundary_overlap_removal_method"] == "pymatgen_merge_sites"
    assert (
        candidate.info["overlap_removed_count"]
        == candidate.info["grain_boundary_removed_atom_count"]
    )


def test_full_provenance_records_parent_and_final_cell_parameters() -> None:
    parent = _parent()
    candidate = grain_boundaries(parent, parent, 1, _settings(), None, _annotate, seed=4)[0]
    info = candidate.info

    assert info["parent_structure_id"] == info["generation_provenance"]["parent_structure_id"]
    for key in (
        "grain_boundary_cell",
        "grain_boundary_cell_lengths",
        "grain_boundary_grain_thickness",
        "grain_boundary_normal_repeat",
        "grain_boundary_join_plane",
    ):
        assert key in info


def test_source_scope_filters_grain_boundary_generation() -> None:
    parent = _parent()
    settings = _settings(grain_boundary_sources=("sqs",))
    candidates = PerturbationCoordinator(settings).generate_candidates(
        [parent],
        counts=PerturbationCounts(n_rattled=0, n_surfaces=0, n_grain_boundaries=1),
        n_workers=1,
    )

    assert all(
        candidate.info.get("perturbation_type") != "grain_boundary" for candidate in candidates
    )


def test_invalid_relationship_and_impossible_thickness_fail_explicitly() -> None:
    parent = _parent()
    with pytest.raises(GrainBoundaryConstructionError, match="rotation axis"):
        grain_boundaries(
            parent,
            parent,
            1,
            _settings(grain_boundary_rotation_axis=(1, 1, 1)),
            None,
            _annotate,
        )
    with pytest.raises(GrainBoundaryConstructionError, match="plane"):
        grain_boundaries(
            parent,
            parent,
            1,
            _settings(grain_boundary_plane=(1, 0, 0)),
            None,
            _annotate,
        )
    with pytest.raises(GrainBoundaryConstructionError, match="below"):
        grain_boundaries(
            parent,
            parent,
            1,
            _settings(grain_boundary_min_thickness=100.0),
            None,
            _annotate,
        )


def test_geometry_validation_reports_bad_pbc_or_missing_provenance() -> None:
    parent = _parent()
    settings = _settings()
    candidate = grain_boundaries(parent, parent, 1, settings, None, _annotate)[0]

    invalid_pbc = candidate.copy()
    invalid_pbc.set_pbc((True, True, False))
    issue = validate_generated_candidate(invalid_pbc, parent, settings, "grain_boundary")
    assert issue is not None
    assert issue.reason == "invalid_grain_boundary_pbc"

    missing = candidate.copy()
    del missing.info["grain_boundary_removed_species"]
    issue = validate_generated_candidate(missing, parent, settings, "grain_boundary")
    assert issue is not None
    assert issue.reason == "missing_grain_boundary_provenance"


def test_coordinator_deduplicates_and_retains_grain_boundary_provenance() -> None:
    parent = _parent()
    settings = _settings(target_n_atoms=2)
    coordinator = PerturbationCoordinator(settings)
    candidates = coordinator.generate_candidates(
        [parent],
        counts=PerturbationCounts(n_rattled=0, n_surfaces=0, n_grain_boundaries=1),
        n_workers=1,
    )
    boundary_candidates = [
        candidate
        for candidate in candidates
        if candidate.info.get("perturbation_type") == "grain_boundary"
    ]
    boundary_records = [
        record
        for record in coordinator.get_provenance_records()
        if record.provenance.perturbation_family == "grain_boundary"
    ]

    assert len(boundary_candidates) == 1
    assert len(boundary_records) == 1
    assert (
        boundary_records[0].provenance.parent_structure_id
        == boundary_candidates[0].info["parent_structure_id"]
    )


def test_invalid_grain_boundary_configuration_is_rejected() -> None:
    from nepflow.config.models import GenerationConfig
    from nepflow.stages.generation.validation import validate_generation_config

    with pytest.raises(ConfigurationError, match="Sigma"):
        validate_generation_config(GenerationConfig(grain_boundary_sigma=3))
