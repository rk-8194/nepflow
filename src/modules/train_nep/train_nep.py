"""NEP model training stage — prepare datasets and submit training jobs."""

import logging
import shlex
import shutil
from configparser import ConfigParser
from datetime import datetime
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np
from ase.io import read as ase_read
from ase.atoms import Atoms

from ..base import Stage
from nepflow.workflow.resubmission import SelfResubmitExit
from nepflow.state.store import StateStore
from nepflow.domain.datasets import DatasetIdentity
from nepflow.domain.identities import calculate_structure_id
from nepflow.dft.vasp.outputs import ResolvedVaspOutput, parse_outcar_result
from nepflow.io.hashing import sha256_file
from nepflow.mlip.nep.artifacts import (
    NepArtifactError,
    create_model_run_manifest,
    update_model_run_status,
)
from nepflow.io.atomic import atomic_write_text
from nepflow.io.json import read_json, write_json
from .launcher import run_launcher, read_train_status, write_train_status
from nepflow.stages.training.dataset import (
    DatasetSplit,
    build_dataset_metadata,
    build_training_dataset,
)
from nepflow.mlip.nep.inputs import (
    NepHyperparameters,
    NepInputRenderer,
    default_nep_template,
)
from nepflow.mlip.nep.backend import NepBackend
from nepflow.hpc.resources import JobResources, render_slurm_header

logger = logging.getLogger("nepflow.train_nep")


class TrainNepStage(Stage):
    """Prepare NEP training datasets from VASP results and submit training job."""

    def run(self) -> None:
        """Execute the stage with one authoritative ledger connection."""

        with StateStore(self.state_file) as state_store:
            return self._run_with_state_store(state_store)

    def _run_with_state_store(self, state_store: StateStore) -> None:
        """Execute NEP dataset preparation and training job submission/monitoring."""
        logger.info("NEP Training Stage")
        
        # Check if this is a resubmission
        status_file = self.project_dir / "nep" / ".train_nep_status"
        status = read_train_status(self.project_dir)
        if status_file.exists():
            status_value = status.get("status")
            if status_value not in {"running", "failed", "completed"}:
                raise ValueError(
                    "Training status has an invalid or missing status field"
                )
            if status_value == "completed":
                raise RuntimeError(
                    "Training status is already completed; refusing to start a fresh run"
                )
            if not isinstance(status.get("potential_path"), str) or not status["potential_path"].strip():
                raise ValueError(
                    "Training status is missing required potential_path for resume"
                )
        
        # If resubmission, skip dataset prep and go straight to launcher
        if status.get("status") in ["running", "failed"] and status.get("potential_path"):
            logger.info("Resubmitting from previous run")
            potential_path = Path(status["potential_path"])
            dataset_path = self._find_dataset_for_potential(
                potential_path,
                state_store=state_store,
                model_run_id=status.get("model_run_id"),
            )
            if status.get("dataset_path"):
                persisted_path = Path(status["dataset_path"]).resolve()
                if persisted_path != dataset_path.resolve():
                    raise RuntimeError(
                        "Training status dataset_path conflicts with the model-run manifest"
                    )
            
            # If dataset still exists, proceed with resubmission
            if dataset_path.exists():
                # Also check if the training script exists
                train_script = potential_path / "train_nep.sh"
                if train_script.exists():
                    config = self._load_config()
                    
                    try:
                        run_launcher(
                            config=config,
                            dataset_path=dataset_path,
                            potential_path=potential_path,
                            project_name=self.project_name,
                            project_dir=self.project_dir,
                            debug=self.debug,
                            slurm_deadline=self.slurm_deadline,
                            state_store=state_store,
                            model_run_id=status.get("model_run_id"),
                        )
                    except SelfResubmitExit as e:
                        logger.warning(f"Resubmit needed: {e}")
                        raise
                    
                    return
                else:
                    raise RuntimeError(
                        f"Persisted training state is missing its launcher script: {train_script}"
                    )
            else:
                raise RuntimeError(
                    f"Persisted training state is missing its dataset: {dataset_path}"
                )

        # === NEW SUBMISSION ===
        # Clear old status file before starting a new run.
        if status_file.exists():
            status_file.unlink()
            logger.debug("Cleared previous status file")
        
        logger.info("Preparing NEP training dataset")

        # Parse configuration
        config_path = self._find_config_file()
        config = self._load_config()
        logger.debug(f"Loaded config from {config_path}")
        hyperparameters = self._get_nep_hyperparameters(config)

        # Read train/test split from existing XYZ files
        logger.info("Step 1: Reading train/test split")
        try:
            train_structures, test_structures = self._read_train_test_split()
        except Exception as e:
            if self.debug:
                logger.warning(f"Could not read train/test split: {e}. Using debug mode.")
                train_structures, test_structures = self._generate_synthetic_split(
                    config.getint("project", "random_seed", fallback=0)
                )
            else:
                raise

        logger.info(
            f"Found {len(train_structures)} train structures and {len(test_structures)} test structures"
        )
        if all(isinstance(item, Atoms) for item in (*train_structures, *test_structures)):
            split_records = {
                DatasetSplit.TRAIN: self._resolve_state_results(
                    state_store, train_structures, DatasetSplit.TRAIN
                ),
                DatasetSplit.TEST: self._resolve_state_results(
                    state_store, test_structures, DatasetSplit.TEST
                ),
            }
        else:
            # Test/integration doubles may supply already-authoritative records
            # directly.  The production path above is the only path that
            # converts selected structures into dataset members.
            split_records = {
                DatasetSplit.TRAIN: train_structures,
                DatasetSplit.TEST: test_structures,
            }

        # Step 2: Create or find dataset folder
        logger.info("Step 2: Creating NEP dataset folder")
        dataset_path = self._get_or_create_dataset_folder()
        logger.info(f"Using dataset folder: {dataset_path}")

        # Extract config options
        train_virial = config.getboolean("train_nep", "train_virial", fallback=False)
        allow_partial = config.getboolean(
            "train_nep", "allow_partial_dataset", fallback=False
        )
        # Step 3: Assemble both explicit splits through the canonical dataset
        # boundary.  The stage still owns orchestration for now, but it no
        # longer owns DFT resolution, label validation, or manifest assembly.
        logger.info(
            "Step 3: Parsing OUTCAR files and writing XYZ datasets. Training virials: %s",
            train_virial,
        )
        dataset_result = build_training_dataset(
            dataset_path,
            split_records,
            self.project_dir,
            train_virial=train_virial,
            debug=self.debug,
            allow_partial=allow_partial,
            state_store=state_store,
            project_id=self.project_name,
        )
        train_count = dataset_result.train_count
        test_count = dataset_result.test_count
        train_report = dataset_result.reports[DatasetSplit.TRAIN].to_dict()
        test_report = dataset_result.reports[DatasetSplit.TEST].to_dict()
        metadata = dict(dataset_result.metadata)

        logger.info(f"Successfully processed {train_count} train and {test_count} test structures")

        # Step 4: Generate customized nep.in
        logger.info("Step 4: Generating customized nep.in")
        template_path = self.project_dir / "config" / "nep" / "nep.in"
        NepInputRenderer().render_hyperparameters(
            dataset=dataset_result.manifest,
            hyperparameters=hyperparameters,
            working_directory=dataset_path,
            template_path=template_path if template_path.exists() else None,
        )
        logger.info(f"nep.in written to {dataset_path / 'nep.in'}")

        # Step 5: Create training run folder
        logger.info("Step 5: Creating potential training folder")
        potential_path = self._create_potential_folder(
            config, train_count, hyperparameters
        )
        logger.info(f"Training will run in: {potential_path.name}")

        # Materialize the exact input files used by the canonical potential run.
        for filename in ("train.xyz", "test.xyz", "nep.in"):
            source = dataset_path / filename
            destination = potential_path / filename
            if source.exists() and source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
        dataset_id = metadata.get("dataset_id")
        if not dataset_id:
            raise RuntimeError(f"Dataset manifest has no dataset_id: {dataset_path / '.dataset'}")
        if state_store.get_dataset(str(dataset_id)) is None:
            # Compatibility doubles may return a prebuilt result without
            # having called the canonical builder's StateStore hook.
            state_store.upsert_dataset(
                DatasetIdentity(str(dataset_id), metadata),
                project_id=None,
                status="prepared",
            )
        model_manifest = create_model_run_manifest(
            potential_path=potential_path,
            dataset_path=dataset_path,
            dataset_id=dataset_id,
            nep_in_path=potential_path / "nep.in",
            hyperparameters_hash=hyperparameters.identity_hash(),
            state_store=state_store,
        )
        logger.info("Created model-run manifest for %s", model_manifest["model_run_id"])
        write_train_status(
            self.project_dir,
            potential_path=str(potential_path),
            dataset_path=str(dataset_path),
            model_run_id=model_manifest["model_run_id"],
            status="prepared",
            attempt=0,
        )

        # Step 6: Prepare the backend command and let the common launcher/
        # Scheduler boundary submit it.  The NEP backend never submits jobs.
        if config.getboolean("slurm", "enabled", fallback=False):
            logger.info("Step 6: Preparing SLURM training job")
            try:
                self._write_training_script(config, potential_path)
            except Exception as e:
                logger.error(f"Could not submit SLURM job: {e}")
                update_model_run_status(
                    potential_path,
                    "failed",
                    error=str(e),
                    state_store=state_store,
                    model_run_id=model_manifest["model_run_id"],
                )
                write_train_status(
                    self.project_dir,
                    potential_path=str(potential_path),
                    status="failed",
                    model_run_id=model_manifest["model_run_id"],
                    attempt=1,
                    error=str(e),
                )
                return

            # Step 7: Monitor training job
            logger.info("Step 7: Monitoring training job")
            try:
                run_launcher(
                    config=config,
                    dataset_path=dataset_path,
                    potential_path=potential_path,
                    project_name=self.project_name,
                    project_dir=self.project_dir,
                    debug=self.debug,
                    slurm_deadline=self.slurm_deadline,
                    state_store=state_store,
                    model_run_id=model_manifest["model_run_id"],
                )
            except SelfResubmitExit as e:
                logger.warning(f"Resubmit needed: {e}")
                raise
        else:
            logger.info("SLURM not enabled. To run training, use the dataset and nep.in files manually:")

    def _write_training_script(self, config: ConfigParser, potential_path: Path) -> Path:
        """Render a scheduler script from a backend command, without submitting it."""

        header_path = self.project_dir / "config" / "slurm" / "header.slurm"
        if not header_path.is_file():
            raise FileNotFoundError(f"SLURM header not found at {header_path}")
        command_text = config.get("hpc", "nep_command", fallback="").strip()
        if not command_text:
            raise ValueError("Required configuration hpc.nep_command is missing or blank")
        backend = NepBackend(command_text)
        walltime = config.get(
            "slurm", "train_nep_walltime",
            fallback=config.get("slurm", "walltime", fallback="24:00:00"),
        )
        rendered_header = render_slurm_header(
            JobResources(nodes=1, gpus_per_node=1, mpi_ranks=1, walltime=walltime),
            base_header=header_path.read_text(encoding="utf-8"),
            job_name=f"nep_train_{self.project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            stdout_path="train_nep_%j.log",
            stderr_path="train_nep_%j.err",
        )
        script_path = potential_path / "train_nep.sh"
        command_line = " ".join(
            part
            if not any(character in part for character in "'\";|&><`")
            else shlex.quote(part)
            for part in backend.command
        )
        script_path.write_text(
            "\n".join(
                (
                    rendered_header.rstrip(),
                    "",
                    f"cd {shlex.quote(str(potential_path))}",
                    command_line,
                )
            )
            + "\n",
            encoding="utf-8",
        )
        script_path.chmod(0o755)
        return script_path

    def _resolve_state_results(
        self,
        state_store: StateStore,
        structures: Sequence[Atoms],
        split: DatasetSplit,
    ) -> list[Any]:
        """Parse only OUTCAR artifacts accepted for exact StateStore rows."""

        calculations = state_store.list_dft_calculations(
            statuses=("completed",),
            selected_only=True,
        )
        by_structure: dict[str, list[dict[str, Any]]] = {}
        for calculation in calculations:
            identity = calculation.get("identity", {})
            structure_id = identity.get("structure_id", calculation.get("structure_id"))
            if structure_id:
                by_structure.setdefault(str(structure_id), []).append(calculation)

        resolved: list[Any] = []
        for atoms in structures:
            structure_id = str(atoms.info.get("structure_id", calculate_structure_id(atoms)))
            requested_calculation_id = atoms.info.get("calculation_id")
            candidates = by_structure.get(structure_id, [])
            if requested_calculation_id:
                candidates = [
                    row
                    for row in candidates
                    if row.get("calculation_id") == requested_calculation_id
                    or row.get("identity", {}).get("calculation_id") == requested_calculation_id
                ]
            if len(candidates) != 1:
                raise RuntimeError(
                    f"StateStore must resolve exactly one completed {split.value} DFT calculation "
                    f"for structure_id={structure_id}; found {len(candidates)}"
                )
            calculation = candidates[0]
            identity = dict(calculation.get("identity", {}))
            calculation_id = str(calculation["calculation_id"])
            attempt_id = calculation.get("accepted_attempt_id")
            artifacts = state_store.list_artifacts(originating_attempt_id=attempt_id)
            outcars = [
                artifact
                for artifact in artifacts
                if artifact.get("artifact_type") == "vasp_outcar"
            ]
            if len(outcars) != 1:
                raise RuntimeError(
                    f"StateStore has no unique accepted OUTCAR for {calculation_id}"
                )
            artifact = outcars[0]
            outcar_path = Path(str(artifact.get("path", "")))
            if not outcar_path.is_file() or sha256_file(outcar_path) != artifact.get("sha256"):
                raise RuntimeError(
                    f"Accepted OUTCAR artifact is missing or changed for {calculation_id}"
                )
            evidence = ResolvedVaspOutput(
                outcar_path=outcar_path,
                calculation_identity=tuple(sorted((str(k), str(v)) for k, v in identity.items())),
                verification_source="state_store_artifact",
            )
            result = parse_outcar_result(
                outcar_path,
                atoms,
                require_virial=False,
                calculation_identity=identity,
                identity_evidence=evidence,
            )
            if not result.accepted:
                raise RuntimeError(
                    f"StateStore OUTCAR could not be parsed for {calculation_id}: "
                    f"{result.rejection_reason}"
                )
            if result.structure_id != structure_id:
                raise RuntimeError(
                    f"StateStore OUTCAR structure identity changed for {calculation_id}"
                )
            resolved.append(result)
        return resolved

    def _find_dataset_for_potential(
        self,
        potential_path: Path,
        *,
        state_store: StateStore | None = None,
        model_run_id: str | None = None,
    ) -> Path:
        """Resolve a dataset through the authoritative model and dataset rows."""
        if state_store is not None:
            if not model_run_id:
                raise RuntimeError("Training restart requires an explicit model_run_id")
            model_run = state_store.get_model_run(model_run_id)
            if model_run is None:
                raise RuntimeError(f"Unknown authoritative model run: {model_run_id}")
            identity = model_run.get("identity", {})
            dataset_id = identity.get("dataset_id")
            dataset = state_store.get_dataset(str(dataset_id)) if dataset_id else None
            if dataset is None:
                raise RuntimeError(
                    f"Model run {model_run_id} has no authoritative dataset association"
                )
            run_metadata = model_run.get("execution_metadata", {})
            stored_potential = run_metadata.get("potential_path")
            if stored_potential and Path(str(stored_potential)).resolve() != potential_path.resolve():
                raise RuntimeError(
                    "Training restart potential_path conflicts with StateStore model identity"
                )
            dataset_manifest = dataset.get("manifest", {})
            dataset_path = Path(
                str(run_metadata.get("dataset_path", dataset_manifest.get("dataset_path", "")))
            )
            if not dataset_path.is_dir():
                raise RuntimeError(
                    f"Authoritative dataset {dataset_id} is missing its materialized path"
                )
            metadata = read_json(
                dataset_path / ".dataset",
                error_type=NepArtifactError,
                missing_error_type=NepArtifactError,
                require_object=True,
            )
            if metadata.get("dataset_id") != dataset_id:
                raise RuntimeError("Materialized dataset conflicts with StateStore identity")
            logger.debug("Resolved model run %s to dataset %s", model_run_id, dataset_path)
            return dataset_path

        raise RuntimeError("StateStore is required to resolve a training restart")

    def _get_nep_hyperparameters(self, config: ConfigParser) -> NepHyperparameters:
        """Resolve all exposed NEP settings through the canonical value object."""
        return NepHyperparameters.from_legacy_config(config)

    def _read_train_test_split(self) -> Tuple[List[Atoms], List[Atoms]]:
        """Read train and test structures from existing XYZ files."""
        train_path = self.project_dir / "structures" / "selected" / "train.xyz"
        test_path = self.project_dir / "structures" / "selected" / "test.xyz"

        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                f"Cannot find train/test XYZ files.\n"
                f"Expected:\n"
                f"  - {train_path}\n"
                f"  - {test_path}"
            )

        train_structures = ase_read(str(train_path), index=":", format="extxyz")
        test_structures = ase_read(str(test_path), index=":", format="extxyz")

        # Ensure lists
        if not isinstance(train_structures, list):
            train_structures = [train_structures]
        if not isinstance(test_structures, list):
            test_structures = [test_structures]

        return train_structures, test_structures

    def _generate_synthetic_split(
        self, random_seed: int = 0
    ) -> Tuple[List[Atoms], List[Atoms]]:
        """Generate synthetic structures for debug mode."""
        logger.warning("Generating synthetic train/test split for debug mode")
        rng = np.random.RandomState(random_seed)

        train_structures = []
        test_structures = []

        for i in range(5):
            atoms = Atoms(
                "W2",
                positions=[[0, 0, 0], [2.5, 0, 0]],
                cell=[[5, 0, 0], [0, 5, 0], [0, 0, 5]],
                pbc=True,
            )
            atoms.info["energy"] = -10.0 + i * 0.1
            atoms.arrays["forces"] = rng.rand(2, 3) * 0.1
            train_structures.append(atoms)

        for i in range(2):
            atoms = Atoms(
                "W2",
                positions=[[0, 0, 0], [2.5, 0, 0]],
                cell=[[5, 0, 0], [0, 5, 0], [0, 0, 5]],
                pbc=True,
            )
            atoms.info["energy"] = -10.0 + i * 0.2
            atoms.arrays["forces"] = rng.rand(2, 3) * 0.1
            test_structures.append(atoms)

        return train_structures, test_structures

    def _generate_nep_config(
        self,
        config: ConfigParser,
        dataset_path: Path,
        hyperparameters: NepHyperparameters | None = None,
    ) -> None:
        """Compatibility adapter around the canonical NEP input renderer."""
        hyperparameters = hyperparameters or self._get_nep_hyperparameters(config)
        template_path = self.project_dir / "config" / "nep" / "nep.in"
        if not template_path.exists():
            template_path = None
        content = NepInputRenderer().render_content(
            hyperparameters,
            template_path=template_path,
        )
        atomic_write_text(dataset_path / "nep.in", content)
        return

    @staticmethod
    def _get_default_nep_template() -> str:
        """Compatibility adapter for the canonical default template."""
        return default_nep_template()

    def _get_or_create_dataset_folder(self) -> Path:
        """Find the next dataset publication target without creating it."""
        datasets_dir = self.project_dir / "nep" / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)

        # Find existing dataset folders
        existing = [
            int(d.name.split("_")[1])
            for d in datasets_dir.iterdir()
            if d.is_dir() and d.name.startswith("dataset_")
        ]

        next_id = max(existing, default=0) + 1
        dataset_path = datasets_dir / f"dataset_{next_id:04d}"
        logger.debug(f"Reserved dataset publication target: dataset_{next_id:04d}")

        return dataset_path

    def _build_dataset_metadata(
        self,
        dataset_path: Path,
        train_report: dict,
        test_report: dict,
        *,
        train_virial: bool,
        allow_partial: bool,
    ) -> dict:
        """Compatibility adapter for the canonical manifest builder."""

        return build_dataset_metadata(
            dataset_path,
            train_report,
            test_report,
            train_virial=train_virial,
            allow_partial=allow_partial,
        )

    @staticmethod
    def _write_dataset_metadata(dataset_path: Path, metadata: dict) -> None:
        """Write the finalized content-derived .dataset manifest."""
        metadata_path = dataset_path / ".dataset"
        write_json(metadata_path, metadata)
        logger.debug("Wrote metadata to %s", metadata_path)

    @staticmethod
    def _generate_params_hash(hyperparameters: NepHyperparameters) -> str:
        """Generate the run identity directly from canonical hyperparameters."""
        return hyperparameters.identity_hash()

    def _create_potential_folder(
        self,
        config: ConfigParser,
        train_count: int,
        hyperparameters: NepHyperparameters | None = None,
    ) -> Path:
        """Create a potential folder whose identity covers every NEP setting."""
        potentials_dir = self.project_dir / "nep" / "potentials"
        potentials_dir.mkdir(parents=True, exist_ok=True)

        hyperparameters = hyperparameters or self._get_nep_hyperparameters(config)
        params_hash = self._generate_params_hash(hyperparameters)[:16]
        base_name = f"{train_count}_{params_hash}"

        # Find next available index for this exact hyperparameter identity.
        existing_indices = []
        for d in potentials_dir.iterdir():
            if d.is_dir() and d.name.startswith(base_name + "_"):
                try:
                    idx = int(d.name.split("_")[-1])
                    existing_indices.append(idx)
                except ValueError:
                    pass

        next_index = max(existing_indices, default=0) + 1
        folder_name = f"{base_name}_{next_index}"

        potential_path = potentials_dir / folder_name
        potential_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created potential folder: {folder_name}")

        return potential_path
