"""Issue 89 liquid-like perturbation contracts."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ase")
from ase import Atoms
from ase import md as ase_md

from nepflow.domain.identities import calculate_structure_id
from nepflow.stages.generation.perturbations.coordinator import (
    PerturbationCoordinator,
    execute_perturbation_task,
)
from nepflow.stages.generation.perturbations.liquid import (
    LiquidGenerationError,
    liquid_snapshots,
)
from nepflow.stages.generation.perturbations.models import (
    LIQUID_FIDELITY,
    LIQUID_METHOD,
    PerturbationCounts,
    PerturbationSettings,
    derive_child_seed,
)
from nepflow.stages.generation.perturbations.provenance import (
    annotate_generation_provenance,
)
from nepflow.stages.generation.validation import validate_generated_candidate


def base_atoms(*, source: str = "mp_phase") -> Atoms:
    atoms = Atoms(
        "Si4",
        positions=[[2, 2, 2], [6, 2, 2], [2, 6, 2], [2, 2, 6]],
        cell=np.eye(3) * 12.0,
        pbc=True,
    )
    atoms.info.update(
        {
            "seed_id": "liquid-fixture",
            "source": source,
            "configurational_type": source,
            "composition": {"Si": 1.0},
        }
    )
    return atoms


def settings(**overrides) -> PerturbationSettings:
    values = {
        "target_n_atoms": 4,
        "random_seed": 17,
        "liquid_enabled": True,
        "liquid_temperature_k": 1200.0,
        "liquid_timestep_fs": 1.0,
        "liquid_equilibration_steps": 2,
        "liquid_steps_between_snapshots": 3,
        "liquid_friction": 0.05,
    }
    values.update(overrides)
    return PerturbationSettings(**values)


def generate(
    base: Atoms,
    *,
    n_configurations: int = 2,
    n_snapshots: int = 3,
    seed: int = 17,
    **overrides,
) -> list[Atoms]:
    configured = settings(random_seed=seed, **overrides)
    return liquid_snapshots(
        base,
        base,
        n_configurations,
        n_snapshots,
        configured,
        seed,
        annotate_generation_provenance,
    )


def test_disabled_path_returns_no_snapshots_without_loading_ase_md() -> None:
    base = base_atoms()

    outputs = liquid_snapshots(
        base,
        base,
        2,
        3,
        PerturbationSettings(
            target_n_atoms=4,
            liquid_enabled=False,
            liquid_steps_between_snapshots=0,
        ),
        17,
        annotate_generation_provenance,
    )

    assert outputs == []


def test_exact_count_and_post_equilibration_snapshot_ordering() -> None:
    base = base_atoms()
    outputs = generate(base, n_configurations=2, n_snapshots=3)

    assert len(outputs) == 6
    assert [item.info["liquid_configuration_index"] for item in outputs] == [0, 0, 0, 1, 1, 1]
    assert [item.info["liquid_snapshot_index"] for item in outputs] == [0, 1, 2, 0, 1, 2]
    assert [item.info["liquid_snapshot_step"] for item in outputs] == [5, 8, 11, 5, 8, 11]
    assert all(item.info["liquid_snapshot_step"] > 2 for item in outputs)


def test_same_seed_is_reproducible_and_trajectory_seeds_are_distinct() -> None:
    base = base_atoms()
    first = generate(base, seed=23)
    second = generate(base, seed=23)

    for left, right in zip(first, second):
        np.testing.assert_array_equal(left.positions, right.positions)
        assert left.info == right.info
    expected = [
        derive_child_seed(calculate_structure_id(base), 23, "liquid", index) for index in (0, 1)
    ]
    assert [first[index * 3].info["liquid_effective_child_seed"] for index in (0, 1)] == expected
    assert expected[0] != expected[1]


def test_liquid_metadata_describes_method_settings_and_source() -> None:
    base = base_atoms()
    candidate = generate(base, n_configurations=1, n_snapshots=1)[0]
    info = candidate.info

    assert info["liquid_method"] == LIQUID_METHOD == "ase_langevin_lj"
    assert info["liquid_fidelity"] == LIQUID_FIDELITY
    assert info["liquid_target_temperature_k"] == 1200.0
    assert info["liquid_timestep_fs"] == 1.0
    assert info["liquid_friction"] == 0.05
    assert info["liquid_equilibration_steps"] == 2
    assert info["liquid_snapshot_spacing_steps"] == 3
    assert info["liquid_trajectory_index"] == 0
    assert info["liquid_snapshot_index"] == 0
    assert info["liquid_effective_child_seed"] == info["random_seed"]
    assert info["liquid_effective_seed"] == info["liquid_child_seed"]
    assert info["parent_structure_id"] == calculate_structure_id(base)
    assert info["source_composition"] == {"Si": 1.0}
    assert info["liquid_source_parent_structure_id"] == calculate_structure_id(base)
    assert info["liquid_source_composition"] == {"Si": 1.0}


def test_source_scope_filters_liquid_candidates_before_generation() -> None:
    base = base_atoms(source="sqs")
    configured = settings(
        liquid_sources=("mp_phase",), n_volume_points=0, elastic_stress_enabled=False
    )
    counts = PerturbationCounts(
        n_rattled=0,
        n_liquid_configurations=1,
        n_liquid_snapshots=1,
        n_vacancies=0,
        n_interstitials=0,
        n_gas_interstitials=0,
        n_substitutions=0,
        n_antisites=0,
        n_vacancy_interstitial=0,
        n_gas_in_vacancy=0,
        n_surfaces=0,
        n_grain_boundaries=0,
    )

    task = next(PerturbationCoordinator(settings=configured)._legacy_tasks([base], counts))
    result = execute_perturbation_task(task)

    assert [candidate.info["perturbation_type"] for candidate in result.candidates] == [
        "unperturbed"
    ]


def test_candidate_validation_rejects_missing_or_inconsistent_liquid_provenance() -> None:
    base = base_atoms()
    candidate = generate(base, n_configurations=1, n_snapshots=1)[0]
    candidate.info.pop("liquid_method")
    issue = validate_generated_candidate(candidate, base, settings(), "liquid")
    assert issue is not None
    assert issue.reason == "missing_liquid_provenance"

    candidate = generate(base, n_configurations=1, n_snapshots=1)[0]
    candidate.info["liquid_snapshot_step"] = 2
    issue = validate_generated_candidate(candidate, base, settings(), "liquid")
    assert issue is not None
    assert issue.reason == "invalid_liquid_provenance"


def test_ase_failure_is_explicitly_propagated(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_langevin(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("controlled ASE failure")

    monkeypatch.setattr(ase_md, "Langevin", fail_langevin)

    with pytest.raises(LiquidGenerationError, match="trajectory setup failed") as failure:
        generate(base_atoms(), n_configurations=1, n_snapshots=1)
    assert isinstance(failure.value.__cause__, RuntimeError)
    assert str(failure.value.__cause__) == "controlled ASE failure"
