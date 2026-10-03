from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from nepflow.config.models import CompositionConfig, NepflowConfig, SlurmConfig
from nepflow.domain.datasets import DatasetIdentity, TrainingDatasetManifest
from nepflow.domain.identities import ArtifactIdentity
from nepflow.mlip.backend import TrainingInput
from nepflow.state.store import StateStore
from nepflow.stages.training.campaign import CampaignReconciliationResult
from nepflow.stages.training.stage import TrainingStage
from nepflow.workflow import StageContext, StageRunState


class RecordingBackend:
    def __init__(self) -> None:
        self.requests = []

    def render_training_input(self, request):
        self.requests.append(request)
        request.working_directory.mkdir(parents=True, exist_ok=True)
        content = "type 1 Si\n"
        (request.working_directory / "nep.in").write_text(content, encoding="utf-8")
        return TrainingInput(
            dataset=request.dataset,
            working_directory=request.working_directory,
            content=content,
            nep_in=ArtifactIdentity.from_bytes(
                "nep_input", content.encode(), path=str(request.working_directory / "nep.in")
            ),
            hyperparameters_hash=request.hyperparameters_hash,
        )

    def model_run_identity(self, inputs):
        return inputs.model_run_identity


class RecordingCampaign:
    def __init__(self, **kwargs) -> None:
        self.state_store = kwargs["state_store"]
        self.campaign_id = kwargs["campaign_id"]
        self.calls = []
        self._candidates = []

    def ensure(self, specification=None) -> None:
        self.calls.append(("ensure", specification))

    def candidates(self):
        return tuple(self._candidates)

    def create_run(self, training_input, *, candidate, dataset_path):
        self.calls.append(("create_run", candidate.ordinal, dataset_path))
        created = SimpleNamespace(run_directory=training_input.working_directory)
        self._candidates.append(
            SimpleNamespace(
                ordinal=candidate.ordinal,
                model_run_id=f"fake-model-{candidate.ordinal}",
                status="pending",
                terminal=False,
            )
        )
        return created

    def reconcile(self):
        raise AssertionError("disabled local training should not reconcile the scheduler")

    def snapshot(self):
        return CampaignReconciliationResult(
            campaign_id=self.campaign_id,
            status="pending",
            candidates=tuple(self._candidates),
            promotion_decision=None,
        )

    def promotion_decision(self):
        return None


def test_training_stage_delegates_preparation_to_backend_and_campaign(tmp_path: Path) -> None:
    dataset = TrainingDatasetManifest(
        DatasetIdentity(
            "dataset-stage",
            {"schema_version": "nepflow.dataset.v1", "records": []},
        ),
        (),
    )
    config = replace(
        NepflowConfig(),
        composition=CompositionConfig(elements=("Si",)),
        slurm=SlurmConfig(enabled=False, max_concurrent=1),
    )
    backend = RecordingBackend()
    campaigns = []

    def campaign_factory(**kwargs):
        campaign = RecordingCampaign(**kwargs)
        campaigns.append(campaign)
        return campaign

    stage = TrainingStage(backend=backend, campaign_factory=campaign_factory)
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    (dataset_path / "train.xyz").write_text("train\n", encoding="utf-8")
    (dataset_path / "test.xyz").write_text("test\n", encoding="utf-8")

    def assemble(_context, _config, _state_store):
        return dataset, dataset_path

    stage._assemble_dataset = assemble
    with StateStore(tmp_path / "state.db") as state_store:
        result = stage.run(
            StageContext(
                project_name="demo",
                project_dir=tmp_path,
                config_file=tmp_path / "project.config",
                state_file=tmp_path / "state.db",
                state_store=state_store,
                workflow_state=None,
                config=config,
                debug=False,
            )
        )

    assert result.status is StageRunState.RUNNING
    assert len(campaigns) == 1
    assert [call[0] for call in campaigns[0].calls] == ["ensure", "create_run"]
    assert len(backend.requests) == 1
    run_directory = backend.requests[0].working_directory
    assert (run_directory / "train.xyz").read_text(encoding="utf-8") == "train\n"
    assert (run_directory / "test.xyz").read_text(encoding="utf-8") == "test\n"
