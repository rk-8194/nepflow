"""Focused tests for the injected canonical generation coordinator."""

from pathlib import Path

import pytest

pytest.importorskip("ase")
from ase import Atoms
from ase.io import write

from nepflow.config.models import CompositionConfig, GenerationConfig
from nepflow.stages.generation.models import GenerationRequest
from nepflow.stages.generation.stage import GenerationStage


class FakeStore:
    def __init__(self) -> None:
        self.structures = []
        self.artifacts = []
        self.events = []

    def upsert_structure(self, record):
        self.structures.append(record)

    def record_artifact(self, artifact, **kwargs):
        self.artifacts.append((artifact, kwargs))

    def append_event(self, *args):
        self.events.append(args)


class FakeGenerator:
    def __init__(self, structures):
        self.structures = structures
        self.calls = []

    def generate(self, composition, crystal_structures, target_n_atoms):
        self.calls.append((dict(composition), tuple(crystal_structures), target_n_atoms))
        return [structure.copy() for structure in self.structures]


class FakeCoordinator:
    def __init__(self):
        self.calls = []
        self.summary = {"total": 1, "by_type": {"unperturbed": 1}, "by_config": {}}

    def process(self, base_structures, output_dir, **kwargs):
        self.calls.append((base_structures, output_dir, kwargs))
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "generated_structures.xyz"
        write(str(path), base_structures)
        return path

    def get_provenance_records(self):
        return ()

    def get_summary(self):
        return self.summary


def request(tmp_path: Path, store: FakeStore, *, seeds_only: bool = False) -> GenerationRequest:
    return GenerationRequest(
        project_name="demo",
        project_dir=tmp_path,
        composition=CompositionConfig(elements=("Si", "Ge"), composition_step=1.0),
        generation=GenerationConfig(
            crystal_structures=("fcc",),
            target_n_atoms=2,
            use_materials_project=False,
            use_random_solid_solution=False,
            use_sqs=False,
            use_segregated=False,
            n_workers=1,
        ),
        random_seed=7,
        state_store=store,
        seeds_only=seeds_only,
    )


def test_stage_retains_injected_generators_and_coordinator(tmp_path: Path) -> None:
    generator = FakeGenerator([Atoms("Si2", cell=[3, 3, 3], pbc=True)])
    coordinator = FakeCoordinator()
    stage = GenerationStage(
        generators=[("Injected", generator)],
        coordinator=coordinator,
    )

    store = FakeStore()
    stage.run(request=request(tmp_path, store, seeds_only=True))

    assert stage.generators == (("Injected", generator),)
    assert stage.coordinator is coordinator
    assert generator.calls
    assert generator.calls[0][1:] == (("fcc",), 2)


def test_stage_deduplicates_physical_structures_and_merges_provenance(tmp_path: Path) -> None:
    first = Atoms("Si2", cell=[3, 3, 3], pbc=True)
    first.info["source"] = "path-a"
    duplicate = first.copy()
    duplicate.info["source"] = "path-b"
    store = FakeStore()
    stage = GenerationStage(
        generators=[("Fake", FakeGenerator([first, duplicate]))],
        coordinator=FakeCoordinator(),
    )

    result = stage.run(request=request(tmp_path, store, seeds_only=True))

    assert result.status == "completed"
    assert len(result.base_structures) == 1
    assert result.base_structures[0].info["provenance_paths"] == ["path-a", "path-b"]
    assert len(store.structures) == 1
    assert result.manifest.artifact is not None
    assert store.artifacts
    assert store.events


def test_stage_passes_typed_perturbation_counts_and_persists_candidate(
    tmp_path: Path,
) -> None:
    base = Atoms("Si2", cell=[3, 3, 3], pbc=True)
    store = FakeStore()
    coordinator = FakeCoordinator()
    stage = GenerationStage(
        generators=[("Fake", FakeGenerator([base]))],
        coordinator=coordinator,
    )

    result = stage.run(request=request(tmp_path, store))

    assert result.summary["total"] == 1
    assert coordinator.calls[0][2]["n_liquid_configurations"] == 0
    assert coordinator.calls[0][2]["n_workers"] == 1
    assert result.manifest.candidate_artifact is not None
    assert any(event[3] == "candidate_artifact_persisted" for event in store.events)


def test_stage_preserves_requested_and_realized_compositions(tmp_path: Path) -> None:
    base = Atoms("Si3Ge", cell=[3, 3, 3], pbc=True)
    base.info["composition"] = {"Si": 0.5, "Ge": 0.5}
    base.info["actual_composition"] = {"Si": 0.75, "Ge": 0.25}
    store = FakeStore()
    stage = GenerationStage(
        generators=[("MaterialsProject", FakeGenerator([base]))],
        coordinator=FakeCoordinator(),
    )

    result = stage.run(request=request(tmp_path, store, seeds_only=True))

    assert result.base_structures[0].info["composition"] == {"Si": 0.5, "Ge": 0.5}
    assert result.base_structures[0].info["actual_composition"] == {
        "Si": 0.75,
        "Ge": 0.25,
    }
