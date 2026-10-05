"""Thin canonical training-stage orchestration."""

from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence

from nepflow.errors import StateError
from nepflow.hpc.resources import JobResources
from nepflow.hpc.scheduler import Scheduler
from nepflow.io.hashing import sha256_bytes, sha256_file
from nepflow.io.json import canonical_json_bytes
from nepflow.mlip.backend import TrainingInputRequest
from nepflow.mlip.nep.artifacts import create_model_run_manifest, read_model_run_manifest
from nepflow.mlip.nep.backend import NepBackend
from nepflow.mlip.nep.inputs import NepHyperparameters
from nepflow.workflow import StageContext, StageRunResult, StageRunState, WorkflowStage

from .campaign import TrainingCampaign
from .dataset import (
    DatasetBuildResult,
    DatasetSplit,
    build_training_dataset,
    load_materialized_dataset,
    prepare_training_dataset,
    resolve_selected_dft_results,
)
from .optimisation import ControlledSweep


class TrainingStage:
    """Compose dataset assembly, candidate preparation, and one campaign pass."""

    def __init__(
        self,
        *,
        scheduler: Scheduler | None = None,
        backend: Any | None = None,
        campaign_factory: Callable[..., TrainingCampaign] = TrainingCampaign,
        dataset_builder: Callable[..., DatasetBuildResult] = build_training_dataset,
        reader: Any | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.backend = backend
        self.campaign_factory = campaign_factory
        self.dataset_builder = dataset_builder
        self.reader = reader

    @staticmethod
    def _dataset_path(project_dir: Path) -> Path:
        """Validate explicit selection inputs for compatibility callers.

        The canonical stage no longer derives dataset identity or storage from
        these files; it resolves authoritative DFT records first.
        """

        selected_dir = Path(project_dir) / "structures" / "selected"
        for split in ("train", "test"):
            source = selected_dir / f"{split}.xyz"
            if not source.is_file():
                raise FileNotFoundError(
                    f"Selected {split} structures not found at {source}; run the select stage first"
                )
        raise StateError(
            "Dataset storage is keyed by the resolved authoritative dataset identity; "
            "call _assemble_dataset instead of deriving a path from selection files"
        )

    @staticmethod
    def _campaign_id(dataset_id: str, candidates: tuple[Any, ...]) -> str:
        payload = {
            "schema_version": "nepflow.training_campaign.v1",
            "dataset_id": dataset_id,
            "candidates": [
                {
                    "ordinal": candidate.ordinal,
                    "candidate_key": candidate.candidate_key,
                    "overrides": dict(candidate.overrides),
                }
                for candidate in candidates
            ],
        }
        return "campaign_" + sha256_bytes(canonical_json_bytes(payload))[:24]

    @staticmethod
    def _resources(config: Any) -> JobResources:
        walltime = config.slurm.train_nep_walltime or config.slurm.walltime
        return JobResources(nodes=1, gpus_per_node=1, mpi_ranks=1, walltime=walltime)

    @staticmethod
    def _materialize_dataset_inputs(dataset_path: Path, run_directory: Path) -> None:
        """Bind the exact immutable dataset files to one candidate directory."""

        for filename in ("train.xyz", "test.xyz"):
            source = Path(dataset_path) / filename
            if not source.is_file():
                raise StateError(f"Training dataset is missing required artifact: {source}")
            destination = Path(run_directory) / filename
            if destination.exists():
                if sha256_file(destination) != sha256_file(source):
                    raise StateError(
                        f"Candidate dataset artifact conflicts with the immutable dataset: {destination}"
                    )
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    def _config(self, context: StageContext) -> Any:
        if context.config is None:
            raise StateError("TrainingStage requires the injected typed project configuration")
        return context.config

    @staticmethod
    def _assert_authoritative_dataset(dataset_manifest: Any, state_store: Any) -> None:
        get_dataset = getattr(state_store, "get_dataset", None)
        if not callable(get_dataset):
            raise StateError("TrainingStage requires StateStore dataset identity APIs")
        dataset_id = dataset_manifest.identity.dataset_id
        row = get_dataset(dataset_id)
        persisted = (
            None
            if row is None or not isinstance(row, dict)
            else row.get("identity_json", row.get("identity"))
        )
        if not isinstance(persisted, dict) or persisted != dataset_manifest.identity.to_dict():
            raise StateError("Training dataset is not registered in authoritative StateStore")

    def _assemble_dataset(
        self,
        context: StageContext,
        config: Any,
        state_store: Any,
    ) -> tuple[Any, Path]:
        # Resolve exact selected DFT identities and accepted OUTCAR hashes
        # before consulting any materialized directory.  Selection files are
        # inputs to resolution, not the dataset identity.
        split_records = resolve_selected_dft_results(
            context.project_dir,
            state_store,
            reader=self.reader,
        )
        prepared_split_records: dict[DatasetSplit | str, Sequence[Any]] = {
            split: records for split, records in split_records.items()
        }
        preview_path = context.project_dir / "nep" / "datasets" / ".identity-preview"
        preparation = prepare_training_dataset(
            prepared_split_records,
            preview_path,
            train_virial=config.train_nep.train_virial,
            allow_partial=config.train_nep.allow_partial_dataset,
            state_store=state_store,
        )
        dataset_id = preparation.manifest.identity.dataset_id
        dataset_path = context.project_dir / "nep" / "datasets" / dataset_id
        if (dataset_path / ".dataset").is_file():
            manifest, metadata = load_materialized_dataset(dataset_path, state_store)
            if manifest.identity.to_dict() != preparation.manifest.identity.to_dict():
                raise StateError(
                    "Materialized training dataset conflicts with the resolved DFT identity"
                )
            if bool(metadata.get("virial_required", False)) != bool(config.train_nep.train_virial):
                raise StateError(
                    "Materialized dataset label schema does not match train_nep.train_virial"
                )
            if bool(metadata.get("partial_dataset_allowed", False)) != bool(
                config.train_nep.allow_partial_dataset
            ):
                raise StateError(
                    "Materialized dataset acceptance policy does not match training config"
                )
            return manifest, dataset_path

        result = self.dataset_builder(
            dataset_path,
            split_records,
            context.project_dir,
            train_virial=config.train_nep.train_virial,
            allow_partial=config.train_nep.allow_partial_dataset,
            state_store=state_store,
            project_id=context.project_name,
        )
        return result.manifest, dataset_path

    def _prepare_candidates(
        self,
        context: StageContext,
        config: Any,
        campaign: TrainingCampaign,
        dataset_manifest: Any,
        dataset_path: Path,
        backend: Any,
    ) -> None:
        candidates = ControlledSweep.from_mapping(
            config.train_nep,
            config.train_nep.sweep_mapping(),
        ).configurations()
        template = context.project_dir / "config" / "nep" / "nep.in"
        template_path = template if template.is_file() else None
        potentials_dir = context.project_dir / "nep" / "potentials"
        for candidate in candidates:
            existing = next(
                (
                    existing
                    for existing in campaign.candidates()
                    if existing.ordinal == candidate.ordinal
                ),
                None,
            )
            if existing is not None:
                self._ensure_model_manifest(
                    existing.run_directory,
                    dataset_path=dataset_path,
                    dataset_id=dataset_manifest.identity.dataset_id,
                    expected_model_run_id=existing.model_run_id,
                    expected_nep_in_sha256=existing.nep_in_sha256,
                    hyperparameters_hash=existing.hyperparameters_hash,
                    state_store=campaign.state_store,
                )
                continue
            run_directory = potentials_dir / f"{campaign.campaign_id}_{candidate.ordinal:04d}"
            self._materialize_dataset_inputs(dataset_path, run_directory)
            hyperparameters = NepHyperparameters.from_config(
                config.composition,
                candidate.hyperparameters,
            )
            request = TrainingInputRequest(
                dataset=dataset_manifest,
                hyperparameters=candidate.hyperparameters,
                composition=config.composition,
                working_directory=run_directory,
                hyperparameters_hash=hyperparameters.identity_hash(),
                template_path=template_path,
            )
            training_input = backend.render_training_input(request)
            created = campaign.create_run(
                training_input,
                candidate=candidate,
                dataset_path=dataset_path,
            )
            expected = backend.model_run_identity(training_input)
            self._ensure_model_manifest(
                created.run_directory,
                dataset_path=dataset_path,
                dataset_id=dataset_manifest.identity.dataset_id,
                expected_model_run_id=expected.model_run_id,
                expected_nep_in_sha256=expected.nep_in_sha256,
                hyperparameters_hash=training_input.hyperparameters_hash,
                state_store=campaign.state_store,
            )

    @staticmethod
    def _ensure_model_manifest(
        run_directory: Path,
        *,
        dataset_path: Path,
        dataset_id: str,
        expected_model_run_id: str,
        expected_nep_in_sha256: str,
        hyperparameters_hash: str,
        state_store: Any,
    ) -> None:
        manifest_path = Path(run_directory) / "model_run_manifest.json"
        if manifest_path.is_file():
            manifest = read_model_run_manifest(manifest_path)
            if any(
                manifest.get(key) != expected
                for key, expected in (
                    ("model_run_id", expected_model_run_id),
                    ("dataset_id", dataset_id),
                    ("nep_in_sha256", expected_nep_in_sha256),
                    ("hyperparameters_hash", hyperparameters_hash),
                    ("dataset_path", str(Path(dataset_path).resolve())),
                    ("nep_in_path", str((Path(run_directory) / "nep.in").resolve())),
                )
            ):
                raise StateError(
                    f"Model-run manifest conflicts with persisted candidate: {expected_model_run_id}"
                )
            return
        manifest = create_model_run_manifest(
            potential_path=Path(run_directory),
            dataset_path=dataset_path,
            dataset_id=dataset_id,
            nep_in_path=Path(run_directory) / "nep.in",
            hyperparameters_hash=hyperparameters_hash,
            state_store=state_store,
        )
        if manifest["model_run_id"] != expected_model_run_id:
            raise StateError("NEP backend and model-run manifest produced different identities")

    def run(self, context: StageContext) -> StageRunResult:
        if context.state_store is None:
            raise StateError("TrainingStage requires the authoritative StateStore")
        config = self._config(context)
        state_store = context.state_store
        dataset_manifest, dataset_path = self._assemble_dataset(
            context,
            config,
            state_store,
        )
        self._assert_authoritative_dataset(dataset_manifest, state_store)
        backend = self.backend or NepBackend(config.hpc.nep_command)
        scheduler = self.scheduler
        if config.slurm.enabled and scheduler is None:
            raise StateError("TrainingStage requires an injected scheduler when SLURM is enabled")
        # The campaign identity includes the effective candidate matrix, so a
        # restart reopens the same event stream rather than proposing trials
        # from mutable folder order.
        candidates = ControlledSweep.from_mapping(
            config.train_nep,
            config.train_nep.sweep_mapping(),
        ).configurations()
        campaign_id = self._campaign_id(dataset_manifest.identity.dataset_id, candidates)
        campaign = self.campaign_factory(
            campaign_id=campaign_id,
            dataset_id=dataset_manifest.identity.dataset_id,
            state_store=state_store,
            scheduler=scheduler,
            backend=backend,
            working_directory=context.project_dir / "nep" / "potentials",
            max_concurrent=config.slurm.max_concurrent,
            max_attempts=config.train_nep.max_resubmit + 1,
            resources=self._resources(config),
            dataset_path=dataset_path,
        )
        campaign.ensure(
            {
                "schema_version": "nepflow.training_campaign_spec.v2",
                "candidate_keys": [candidate.candidate_key for candidate in candidates],
                "candidate_matrix": [
                    {
                        "ordinal": candidate.ordinal,
                        "candidate_key": candidate.candidate_key,
                        "overrides": dict(candidate.overrides),
                        "hyperparameters": {
                            key: value
                            for key, value in asdict(candidate.hyperparameters).items()
                            if key != "sweep"
                        },
                    }
                    for candidate in candidates
                ],
                "max_attempts": config.train_nep.max_resubmit + 1,
                "max_concurrent": config.slurm.max_concurrent,
                "backend": {
                    "kind": f"{type(backend).__module__}.{type(backend).__qualname__}",
                    "command": list(getattr(backend, "command", ())),
                    "configured_command": config.hpc.nep_command,
                    "command_fingerprint": sha256_bytes(
                        canonical_json_bytes(config.hpc.nep_command)
                    ),
                },
                "resource_policy": asdict(self._resources(config)),
            }
        )
        self._prepare_candidates(
            context,
            config,
            campaign,
            dataset_manifest,
            dataset_path,
            backend,
        )

        if not config.slurm.enabled:
            result = campaign.snapshot()
        else:
            result = campaign.reconcile()
        if result.status == "failed":
            return StageRunResult(
                stage=WorkflowStage.TRAIN_NEP,
                status=StageRunState.FAILED,
                completed=False,
                message="All training candidates failed",
            )
        if result.ready_for_validation:
            return StageRunResult(
                stage=WorkflowStage.TRAIN_NEP,
                status=StageRunState.COMPLETED,
                advanced_to=WorkflowStage.VALIDATE,
                completed=True,
                message="Training campaign completed; promotion remains a separate decision",
            )
        return StageRunResult(
            stage=WorkflowStage.TRAIN_NEP,
            status=StageRunState.RUNNING,
            completed=False,
            message="Training campaign has active or pending candidates",
        )


__all__ = ["TrainingStage"]
