"""Focused tests for the injected canonical generation coordinator."""

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("ase")
from ase import Atoms
from ase.io import read, write

from nepflow.config.models import CompositionConfig, GenerationConfig
from nepflow.domain.identities import ArtifactIdentity, calculate_structure_id
from nepflow.domain.structures import (
    GeneratedStructureRecord,
    StructureIdentity,
    StructureProvenance,
)
from nepflow.errors import StateError
from nepflow.io.hashing import sha256_file
from nepflow.stages.generation.models import GenerationExecutionMode, GenerationRequest
from nepflow.stages.generation.perturbations.provenance import annotate_generation_provenance
from nepflow.stages.generation.stage import GenerationStage
from nepflow.state.store import StateStore


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


class ResumeCoordinator(FakeCoordinator):
    def __init__(self, store: StateStore) -> None:
        super().__init__()
        self.store = store
        self.parents_present: list[bool] = []
        self.records = []

    def process(self, base_structures, output_dir, **kwargs):
        del kwargs
        self.parents_present = [
            self.store.get_structure(calculate_structure_id(base)) is not None
            for base in base_structures
        ]
        output_dir.mkdir(parents=True, exist_ok=True)
        candidate = base_structures[0].copy()
        record = annotate_generation_provenance(
            candidate,
            base_structures[0],
            "unperturbed",
            operation_id=f"candidate:{calculate_structure_id(base_structures[0])}",
        )
        self.records = [record]
        path = output_dir / "generated_structures.xyz"
        write(str(path), [candidate], format="extxyz")
        return path

    def get_provenance_records(self):
        return tuple(self.records)


class CountingStateStore(StateStore):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.top_level_transactions = 0

    @contextmanager
    def transaction(self):
        if not self.connection.in_transaction:
            self.top_level_transactions += 1
        with super().transaction():
            yield self


def generated_record(
    structure_id: str,
    operation_id: str,
    *,
    parent_structure_id: str | None = None,
    metadata: dict[str, str] | None = None,
) -> GeneratedStructureRecord:
    return GeneratedStructureRecord(
        identity=StructureIdentity(structure_id),
        provenance=StructureProvenance(
            parent_structure_id=parent_structure_id,
            generator="test",
            requested_composition={"Si": 1.0},
            realised_composition={"Si": 1.0},
            source_database_id=None,
            crystal_structure="fcc",
            perturbation_family="unperturbed",
            perturbation_parameters={},
            random_seed=1,
            operation_id=operation_id,
            code_version=None,
            config_fingerprint="test-config",
        ),
        metadata=metadata,
    )


def request(
    tmp_path: Path,
    store: FakeStore,
    *,
    seeds_only: bool = False,
    execution_mode: GenerationExecutionMode = GenerationExecutionMode.RESUME,
) -> GenerationRequest:
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
        execution_mode=execution_mode,
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
    assert result.manifest.candidate_path is not None
    assert result.manifest.candidate_artifact.path == str(result.manifest.candidate_path.resolve())
    assert result.manifest.candidate_artifact.sha256 == sha256_file(result.manifest.candidate_path)
    assert any(event[3] == "candidate_artifact_persisted" for event in store.events)


def test_restart_prepares_current_config_instead_of_reusing_prior_seed(
    tmp_path: Path,
) -> None:
    old_base = Atoms("Si2", cell=[3, 3, 3], pbc=True)
    new_base = Atoms("Ge2", cell=[4, 4, 4], pbc=True)
    store = FakeStore()

    initial_request = request(tmp_path, store)
    initial_stage = GenerationStage(generators=(), coordinator=FakeCoordinator())
    _, _ = initial_stage._persist_bases(initial_request, [old_base])

    restart_request = replace(
        initial_request,
        generation=replace(
            initial_request.generation,
            use_liquid=True,
            n_liquid_configurations=2,
            n_liquid_snapshots=3,
        ),
        execution_mode=GenerationExecutionMode.RESTART,
    )
    coordinator = FakeCoordinator()
    generator = FakeGenerator([new_base])
    stage = GenerationStage(generators=[("Current", generator)], coordinator=coordinator)

    result = stage.run(request=restart_request)

    assert result.manifest.resumed is False
    assert generator.calls
    assert result.base_structures[0].get_chemical_symbols() == ["Ge", "Ge"]
    assert coordinator.calls[0][2]["n_liquid_configurations"] == 2
    assert coordinator.calls[0][2]["n_liquid_snapshots"] == 3
    assert len([event for event in store.events if event[3] == "base_artifact_persisted"]) == 2


def test_seed_artifact_round_trip_defines_stable_canonical_identity(tmp_path: Path) -> None:
    base = Atoms(
        "Si2",
        positions=[[0.123456789012345, 0.234567890123456, 0.345678901234567], [1.7, 1.8, 1.9]],
        cell=[3.123456789012345, 3.234567890123456, 3.345678901234567],
        pbc=True,
    )
    store = FakeStore()
    stage = GenerationStage(generators=(), coordinator=FakeCoordinator())
    seed_request = request(tmp_path, store)

    canonical_bases, manifest = stage._persist_bases(seed_request, [base])
    first_read = list(read(str(manifest.path), index=":"))
    second_read = list(read(str(manifest.path), index=":"))

    expected_ids = [calculate_structure_id(item) for item in canonical_bases]
    assert [item.info["structure_id"] for item in first_read] == expected_ids
    assert [calculate_structure_id(item) for item in first_read] == expected_ids
    assert [item.info["structure_id"] for item in second_read] == expected_ids
    assert [calculate_structure_id(item) for item in second_read] == expected_ids


def test_perturbations_receive_and_persist_canonical_seed_parents(tmp_path: Path) -> None:
    base = Atoms(
        "Si2",
        positions=[[0.123456789012345, 0.234567890123456, 0.345678901234567], [1.7, 1.8, 1.9]],
        cell=[3.123456789012345, 3.234567890123456, 3.345678901234567],
        pbc=True,
    )
    with StateStore(tmp_path / "state.db") as store:
        seed_request = request(tmp_path, store)
        coordinator = ResumeCoordinator(store)
        stage = GenerationStage(
            generators=[("Fake", FakeGenerator([base]))],
            coordinator=coordinator,
        )

        result = stage.run(request=seed_request)
        persisted = list(read(str(result.manifest.path), index=":"))
        canonical_id = persisted[0].info["structure_id"]

        assert coordinator.parents_present == [True]
        assert calculate_structure_id(persisted[0]) == canonical_id
        assert coordinator.records[0].provenance.parent_structure_id == canonical_id
        assert store.get_structure(canonical_id) is not None


def test_mismatched_legacy_seed_identity_fails_before_perturbations(
    tmp_path: Path,
) -> None:
    base = Atoms("Si2", cell=[3.123456789012345, 3.2, 3.3], pbc=True)
    base.info["structure_id"] = "legacy-mismatched-id"
    store = StateStore(tmp_path / "state.db")
    seed_request = request(tmp_path, store)
    seed_request.seeds_file.parent.mkdir(parents=True, exist_ok=True)
    write(str(seed_request.seeds_file), [base], format="extxyz")
    artifact = ArtifactIdentity.from_file("generation_seed_structures", seed_request.seeds_file)
    store.append_event(
        "generation:demo:legacy-seed",
        "generation",
        "demo",
        "base_artifact_persisted",
        {"artifact": artifact.to_dict()},
    )
    coordinator = FakeCoordinator()
    stage = GenerationStage(generators=(), coordinator=coordinator)

    try:
        with pytest.raises(StateError, match="incompatible.*canonical structure_id"):
            stage.run(request=seed_request)
    finally:
        store.close()

    assert coordinator.calls == []


def test_generated_record_persistence_uses_one_top_level_transaction(tmp_path: Path) -> None:
    store = CountingStateStore(tmp_path / "state.db")
    stage = GenerationStage(generators=(), coordinator=FakeCoordinator())
    records = [generated_record(f"generated-{index}", f"operation-{index}") for index in range(100)]
    generation_request = request(tmp_path, store)

    try:
        stage._persist_generated_records(generation_request, records)

        assert store.top_level_transactions == 1
        assert all(store.get_structure(record.structure_id) is not None for record in records)
    finally:
        store.close()


def test_generated_record_batch_rolls_back_on_later_provenance_conflict(tmp_path: Path) -> None:
    with StateStore(tmp_path / "state.db") as store:
        stage = GenerationStage(generators=(), coordinator=FakeCoordinator())
        generation_request = request(tmp_path, store)
        store.upsert_structure(generated_record("existing", "conflicting-operation"))

        with pytest.raises(StateError, match="Structure provenance identity conflict"):
            stage._persist_generated_records(
                generation_request,
                [
                    generated_record("batch-first", "batch-operation-first"),
                    generated_record("batch-second", "conflicting-operation"),
                ],
            )

        assert store.get_structure("batch-first") is None
        assert store.get_structure("batch-second") is None
        assert store.get_structure("existing") is not None


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


def test_base_provenance_operation_is_namespaced_by_generation_configuration(
    tmp_path: Path,
) -> None:
    first = Atoms("Si2", cell=[3, 3, 3], pbc=True)
    first.info["seed_id"] = "seed_000000"
    second = first.copy()
    second.set_cell([4, 4, 4], scale_atoms=True)
    stage = GenerationStage(generators=(), coordinator=FakeCoordinator())

    with StateStore(tmp_path / "state.db") as store:
        first_request = GenerationRequest(
            project_name="demo",
            project_dir=tmp_path,
            composition=CompositionConfig(elements=("Si",)),
            generation=GenerationConfig(target_n_atoms=2),
            random_seed=7,
            state_store=store,
        )
        changed_request = GenerationRequest(
            project_name="demo",
            project_dir=tmp_path,
            composition=CompositionConfig(elements=("Si",)),
            generation=GenerationConfig(target_n_atoms=4),
            random_seed=7,
            state_store=store,
        )

        stage._persist_structure_records(first_request, [first])
        stage._persist_structure_records(changed_request, [second])

        first_provenance = store.get_structure_provenance(calculate_structure_id(first))
        second_provenance = store.get_structure_provenance(calculate_structure_id(second))

    assert first_provenance[0]["operation_id"] != second_provenance[0]["operation_id"]
    assert first_provenance[0]["config_fingerprint"] != second_provenance[0]["config_fingerprint"]


def test_resumed_seed_reconciles_missing_parents_before_perturbations_and_is_idempotent(
    tmp_path: Path,
) -> None:
    base = Atoms("Si2", cell=[3, 3, 3], pbc=True)
    base.info["seed_id"] = "seed_000000"
    stage = GenerationStage(generators=(), coordinator=FakeCoordinator())

    with StateStore(tmp_path / "state.db") as store:
        initial_request = request(tmp_path, FakeStore())
        initial_request = GenerationRequest(
            project_name=initial_request.project_name,
            project_dir=initial_request.project_dir,
            composition=initial_request.composition,
            generation=initial_request.generation,
            random_seed=initial_request.random_seed,
            state_store=store,
        )
        _, _ = stage._persist_bases(initial_request, [base])
        base_id = calculate_structure_id(base)
        store.connection.execute("DELETE FROM structure_provenance")
        store.connection.execute("DELETE FROM structures")

        first_coordinator = ResumeCoordinator(store)
        first_stage = GenerationStage(generators=(), coordinator=first_coordinator)
        first = first_stage.run(request=initial_request)

        assert first_coordinator.parents_present == [True]
        first_provenance = store.get_structure_provenance(base_id)
        first_operation_ids = [row["operation_id"] for row in first_provenance]
        assert len(first_provenance) == 2
        assert store.get_structure(base_id) is not None

        second_coordinator = ResumeCoordinator(store)
        second_stage = GenerationStage(generators=(), coordinator=second_coordinator)
        second_stage.run(request=initial_request)

        second_provenance = store.get_structure_provenance(base_id)
        assert second_coordinator.parents_present == [True]
        assert [row["operation_id"] for row in second_provenance] == first_operation_ids
        assert store.connection.execute("SELECT COUNT(*) FROM structures").fetchone()[0] == 1
        assert first.manifest.candidate_artifact is not None


def test_resumed_seed_identity_schema_conflict_fails_before_perturbations(
    tmp_path: Path,
) -> None:
    base = Atoms("Si2", cell=[3, 3, 3], pbc=True)
    stage = GenerationStage(generators=(), coordinator=FakeCoordinator())

    with StateStore(tmp_path / "state.db") as store:
        initial_request = request(tmp_path, FakeStore())
        initial_request = GenerationRequest(
            project_name=initial_request.project_name,
            project_dir=initial_request.project_dir,
            composition=initial_request.composition,
            generation=initial_request.generation,
            random_seed=initial_request.random_seed,
            state_store=store,
        )
        _, _ = stage._persist_bases(initial_request, [base])
        base_id = calculate_structure_id(base)
        store.connection.execute("DELETE FROM structure_provenance")
        store.connection.execute("DELETE FROM structures")
        store.connection.execute(
            "INSERT INTO structures (structure_id, identity_schema, metadata_json, created_at) "
            "VALUES (?, ?, ?, datetime('now'))",
            (base_id, "legacy-identity", "{}"),
        )

        coordinator = ResumeCoordinator(store)
        resumed_stage = GenerationStage(generators=(), coordinator=coordinator)
        with pytest.raises(StateError, match=base_id):
            resumed_stage.run(request=initial_request)

        assert coordinator.parents_present == []
        assert not (tmp_path / "structures" / "generated" / "generated_structures.xyz").exists()
