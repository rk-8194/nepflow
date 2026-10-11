from pathlib import Path

import pytest

pytest.importorskip("ase")
from ase import Atoms
from ase.io import write

from nepflow.config.models import CompositionConfig, GenerationConfig
from nepflow.domain.identities import ArtifactIdentity, calculate_structure_id
from nepflow.errors import ArtifactError
from nepflow.stages.generation.models import GenerationRequest
from nepflow.stages.generation.stage import GenerationStage
from nepflow.state.store import StateStore


class _Coordinator:
    def __init__(self) -> None:
        self.calls = 0

    def process(self, base_structures, output_dir, **kwargs):
        del kwargs
        self.calls += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "generated_structures.xyz"
        write(path, base_structures, format="extxyz")
        return path

    def get_summary(self):
        return {"total": 0, "by_type": {}, "by_config": {}}

    def get_provenance_records(self):
        return ()


def _request(root: Path, store: StateStore) -> GenerationRequest:
    return GenerationRequest(
        project_name="portable",
        project_dir=root,
        composition=CompositionConfig(elements=("Si",)),
        generation=GenerationConfig(
            crystal_structures=("fcc",),
            target_n_atoms=1,
            use_materials_project=False,
            use_random_solid_solution=False,
            use_sqs=False,
            use_segregated=False,
            n_workers=1,
        ),
        random_seed=1,
        state_store=store,
    )


def _persist_legacy_event(store: StateStore, old_path: str, digest: str, artifact_id: str) -> None:
    store.append_event(
        "generation:portable:legacy-base",
        "generation",
        "portable",
        "base_artifact_persisted",
        {
            "artifact": {
                "artifact_id": artifact_id,
                "artifact_type": "generation_seed_structures",
                "sha256": digest,
                "schema_version": "nepflow.artifact_identity.v1",
                "path": old_path,
            }
        },
    )


def test_legacy_absolute_seed_event_resumes_at_current_project_root(tmp_path: Path) -> None:
    seed_path = tmp_path / "structures" / "seeds" / "base_structures.xyz"
    seed_path.parent.mkdir(parents=True)
    seed = Atoms("Si")
    seed.info["structure_id"] = calculate_structure_id(seed)
    write(seed_path, seed, format="extxyz")
    artifact = ArtifactIdentity.from_file("generation_seed_structures", seed_path)

    with StateStore(tmp_path / "state.db") as store:
        _persist_legacy_event(
            store,
            r"C:\\old\\project\\structures\\seeds\\base_structures.xyz",
            str(artifact.sha256),
            artifact.artifact_id,
        )
        coordinator = _Coordinator()
        result = GenerationStage(generators=(), coordinator=coordinator).run(
            request=_request(tmp_path, store)
        )

        assert result.manifest.resumed is True
        assert coordinator.calls == 1
        assert result.manifest.to_dict()["path"] == "structures/seeds/base_structures.xyz"


def test_authoritative_seed_digest_mismatch_is_not_accepted(tmp_path: Path) -> None:
    seed_path = tmp_path / "structures" / "seeds" / "base_structures.xyz"
    seed_path.parent.mkdir(parents=True)
    write(seed_path, Atoms("Si"), format="extxyz")
    artifact = ArtifactIdentity.from_file("generation_seed_structures", seed_path)
    seed_path.write_text("not-the-authoritative-seed", encoding="utf-8")

    with StateStore(tmp_path / "state.db") as store:
        _persist_legacy_event(
            store,
            "/old/project/structures/seeds/base_structures.xyz",
            str(artifact.sha256),
            artifact.artifact_id,
        )
        with pytest.raises(ArtifactError, match="digest mismatch"):
            GenerationStage(generators=(), coordinator=_Coordinator()).run(
                request=_request(tmp_path, store)
            )


def test_authoritative_seed_missing_is_not_regenerated(tmp_path: Path) -> None:
    seed_path = tmp_path / "structures" / "seeds" / "base_structures.xyz"
    seed_path.parent.mkdir(parents=True)
    seed_path.write_bytes(b"seed")
    artifact = ArtifactIdentity.from_file("generation_seed_structures", seed_path)
    seed_path.unlink()

    with StateStore(tmp_path / "state.db") as store:
        _persist_legacy_event(
            store,
            "/old/project/structures/seeds/base_structures.xyz",
            str(artifact.sha256),
            artifact.artifact_id,
        )
        with pytest.raises(ArtifactError, match="missing"):
            GenerationStage(generators=(), coordinator=_Coordinator()).run(
                request=_request(tmp_path, store)
            )
