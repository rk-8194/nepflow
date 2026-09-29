"""NEP model training stage — prepare datasets and submit training jobs."""

import hashlib
import json
import logging
import shutil
from configparser import ConfigParser
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from ase.io import read as ase_read
from ase.atoms import Atoms

from ..base import Stage, SelfResubmitExit
from common.model_manifest import (
    MODEL_RUN_MANIFEST_FILENAME,
    ModelManifestError,
    create_model_run_manifest,
    read_model_run_manifest,
    update_model_run_status,
)
from .prepare import prepare_dataset
from .submit import submit_training_job
from .launcher import run_launcher, read_train_status, write_train_status

logger = logging.getLogger("nepflow.train_nep")


def _canonical_tokens(value: str) -> tuple[str, ...]:
    """Normalize a space/comma-separated numeric NEP setting."""
    tokens = value.replace(",", " ").split()
    normalized = []
    for token in tokens:
        try:
            normalized.append(format(float(token), ".15g"))
        except ValueError:
            normalized.append(token)
    return tuple(normalized)


@dataclass(frozen=True)
class NepHyperparameters:
    """Canonical NEP settings shared by rendering and run identity."""

    elements: tuple[str, ...]
    gas_elements: tuple[str, ...]
    cutoff: tuple[str, ...]
    n_max: tuple[str, ...]
    basis_size: tuple[str, ...]
    l_max: tuple[str, ...]
    neuron: tuple[str, ...]
    population: int
    batch: int
    generations: int
    outer_zbl: float
    charge_mode: int
    weights: tuple[float, ...]
    lambda_e: float
    lambda_f: float
    lambda_v: float
    lambda_shear: float

    @property
    def all_elements(self) -> tuple[str, ...]:
        return self.elements + self.gas_elements

    @staticmethod
    def _float(value: float) -> str:
        return format(value, ".15g")

    def canonical_dict(self) -> dict:
        """Return deterministic, JSON-serializable scientific settings."""
        return {
            "schema_version": "nep.hyperparameters.v1",
            "types": list(self.all_elements),
            "cutoff": list(self.cutoff),
            "n_max": list(self.n_max),
            "basis_size": list(self.basis_size),
            "l_max": list(self.l_max),
            "neuron": list(self.neuron),
            "population": self.population,
            "batch": self.batch,
            "generations": self.generations,
            "outer_zbl": self._float(self.outer_zbl),
            "charge_mode": self.charge_mode,
            "weights": [self._float(value) for value in self.weights],
            "lambda_e": self._float(self.lambda_e),
            "lambda_f": self._float(self.lambda_f),
            "lambda_v": self._float(self.lambda_v),
            "lambda_shear": self._float(self.lambda_shear),
        }

    def identity_hash(self) -> str:
        payload = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class TrainNepStage(Stage):
    """Prepare NEP training datasets from VASP results and submit training job."""

    def run(self) -> None:
        """Execute NEP dataset preparation and training job submission/monitoring."""
        logger.info("NEP Training Stage")
        
        # Check if this is a resubmission
        try:
            status = read_train_status(self.project_dir)
        except Exception as e:
            logger.debug(f"Could not read status: {e}")
            status = {}
        
        # If resubmission, skip dataset prep and go straight to launcher
        if status.get("status") in ["running", "failed"] and status.get("potential_path"):
            logger.info("Resubmitting from previous run")
            potential_path = Path(status["potential_path"])
            dataset_path = self._find_dataset_for_potential(potential_path)
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
                        )
                    except SelfResubmitExit as e:
                        logger.warning(f"Resubmit needed: {e}")
                        raise
                    
                    return
                else:
                    logger.warning("Training script not found — starting fresh")
            else:
                logger.info("Dataset no longer available — starting fresh run")

        # === NEW SUBMISSION ===
        # Clear old status file before starting fresh
        status_file = self.project_dir / "nep" / ".train_nep_status"
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
                train_structures, test_structures = self._generate_synthetic_split()
            else:
                raise

        logger.info(
            f"Found {len(train_structures)} train structures and {len(test_structures)} test structures"
        )

        # Step 2: Create or find dataset folder
        logger.info("Step 2: Creating NEP dataset folder")
        dataset_path = self._get_or_create_dataset_folder()
        logger.info(f"Using dataset folder: {dataset_path}")

        # Extract config options
        train_virial = config.getboolean("train_nep", "train_virial", fallback=False)
        allow_partial = config.getboolean(
            "train_nep", "allow_partial_dataset", fallback=False
        )
        train_report: dict = {}
        test_report: dict = {}

        # Step 3: Prepare datasets (parse OUTCAR and write XYZ files)
        logger.info("Step 3: Parsing OUTCAR files and writing XYZ datasets. Training virials: %s", train_virial)
        train_count = prepare_dataset(
            dataset_path=dataset_path / "train.xyz",
            ase_structures=train_structures,
            is_train=True,
            project_dir=self.project_dir,
            train_virial=train_virial,
            debug=self.debug,
            extraction_report=train_report,
        )
        test_count = prepare_dataset(
            dataset_path=dataset_path / "test.xyz",
            ase_structures=test_structures,
            is_train=False,
            project_dir=self.project_dir,
            train_virial=train_virial,
            debug=self.debug,
            extraction_report=test_report,
        )

        self._complete_extraction_report(train_report, len(train_structures), train_count)
        self._complete_extraction_report(test_report, len(test_structures), test_count)
        rejected_total = train_report["rejected_count"] + test_report["rejected_count"]
        if rejected_total and not allow_partial:
            raise RuntimeError(
                "Dataset creation rejected selected structures; "
                "set train_nep.allow_partial_dataset=true to allow explicit partial data"
            )

        if train_count == 0 or test_count == 0:
            raise RuntimeError("No valid structures found; cannot create a training dataset")

        metadata = self._build_dataset_metadata(
            dataset_path,
            train_report,
            test_report,
            train_virial=train_virial,
            allow_partial=allow_partial,
        )
        self._write_dataset_metadata(dataset_path, metadata)

        logger.info(f"Successfully processed {train_count} train and {test_count} test structures")

        # Step 4: Generate customized nep.in
        logger.info("Step 4: Generating customized nep.in")
        self._generate_nep_config(config, dataset_path, hyperparameters)
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
            if source.exists():
                shutil.copy2(source, potential_path / filename)
        dataset_id = metadata.get("dataset_id")
        if not dataset_id:
            raise RuntimeError(f"Dataset manifest has no dataset_id: {dataset_path / '.dataset'}")
        model_manifest = create_model_run_manifest(
            potential_path=potential_path,
            dataset_path=dataset_path,
            dataset_id=dataset_id,
            nep_in_path=potential_path / "nep.in",
            hyperparameters_hash=hyperparameters.identity_hash(),
        )
        logger.info("Created model-run manifest for %s", model_manifest["model_run_id"])

        # Step 6: Submit job and enter monitoring loop
        if config.getboolean("slurm", "enabled", fallback=False):
            logger.info("Step 6: Submitting SLURM training job")
            try:
                job_id = submit_training_job(
                    config=config,
                    dataset_path=dataset_path,
                    potential_path=potential_path,
                    project_dir=self.project_dir,
                    project_name=self.project_name,
                )
                logger.info(f"NEP training submitted as SLURM job {job_id}")
                # Save job_id and dataset_path to status file to track this submission
                # The launcher will monitor this job and only resubmit if it fails
                write_train_status(
                    self.project_dir,
                    potential_path=str(potential_path),
                    dataset_path=str(dataset_path),
                    model_run_id=model_manifest["model_run_id"],
                    job_id=job_id,
                    status="running",
                    attempt=1,
                )
            except Exception as e:
                logger.error(f"Could not submit SLURM job: {e}")
                update_model_run_status(
                    potential_path,
                    "failed",
                    error=str(e),
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
                )
            except SelfResubmitExit as e:
                logger.warning(f"Resubmit needed: {e}")
                raise
        else:
            logger.info("SLURM not enabled. To run training, use the dataset and nep.in files manually:")

    def _find_dataset_for_potential(self, potential_path: Path) -> Path:
        """Resolve the dataset from the potential's explicit run manifest."""
        manifest_path = potential_path / MODEL_RUN_MANIFEST_FILENAME
        try:
            manifest = read_model_run_manifest(manifest_path)
        except FileNotFoundError:
            raise
        except ModelManifestError as exc:
            raise RuntimeError(
                f"Cannot resolve dataset for {potential_path} without a valid model-run manifest"
            ) from exc
        dataset_path = Path(str(manifest.get("dataset_path", "")))
        dataset_id = manifest.get("dataset_id")
        if not dataset_id or not dataset_path.is_dir():
            raise RuntimeError(
                f"Model-run manifest has no valid dataset association: {manifest_path}"
            )
        try:
            dataset_metadata = json.loads(
                (dataset_path / ".dataset").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read dataset manifest: {dataset_path / '.dataset'}") from exc
        if dataset_metadata.get("dataset_id") != dataset_id:
            raise RuntimeError(
                f"Dataset identity mismatch for model run {manifest.get('model_run_id')}"
            )
        logger.debug("Resolved model run %s to dataset %s", manifest.get("model_run_id"), dataset_path)
        return dataset_path

    @staticmethod
    def _parse_list(config: ConfigParser, section: str, option: str) -> List[str]:
        """Parse comma-separated list from config."""
        value = config.get(section, option)
        return [item.strip() for item in value.split(",") if item.strip()]

    def _get_nep_hyperparameters(self, config: ConfigParser) -> NepHyperparameters:
        """Resolve all exposed NEP settings into one immutable value object."""
        try:
            elements = tuple(self._parse_list(config, "composition", "elements"))
        except Exception:
            elements = ("W", "O")

        try:
            gas_elements = tuple(self._parse_list(config, "composition", "gasElements"))
        except Exception:
            gas_elements = ()

        all_elements = elements + gas_elements
        weights_str = config.get("train_nep", "weights", fallback="")
        try:
            weights = tuple(
                float(weight)
                for weight in weights_str.replace(",", " ").split()
            )
            if len(weights) != len(all_elements):
                raise ValueError("weight count does not match element count")
        except ValueError:
            weights = tuple(1.0 for _ in all_elements)

        charge_mode = config.getint("train_nep", "charge_mode", fallback=0)
        if charge_mode not in (0, 1):
            logger.warning(
                "Unsupported train_nep.charge_mode=%s. Falling back to 0 (NEP).",
                charge_mode,
            )
            charge_mode = 0

        return NepHyperparameters(
            elements=elements,
            gas_elements=gas_elements,
            cutoff=_canonical_tokens(config.get("train_nep", "cutoff", fallback="6 5")),
            n_max=_canonical_tokens(config.get("train_nep", "n_max", fallback="4 4")),
            basis_size=_canonical_tokens(
                config.get("train_nep", "basis_size", fallback="8 8")
            ),
            l_max=_canonical_tokens(config.get("train_nep", "l_max", fallback="4 2 1")),
            neuron=_canonical_tokens(config.get("train_nep", "neuron", fallback="80")),
            population=config.getint("train_nep", "population", fallback=50),
            batch=config.getint("train_nep", "batch", fallback=3000),
            generations=config.getint("train_nep", "generation", fallback=250000),
            outer_zbl=config.getfloat("train_nep", "outerZBL", fallback=2.0),
            charge_mode=charge_mode,
            weights=weights,
            lambda_e=config.getfloat("train_nep", "lambda_e", fallback=1.0),
            lambda_f=config.getfloat("train_nep", "lambda_f", fallback=1.0),
            lambda_v=config.getfloat("train_nep", "lambda_v", fallback=1.0),
            lambda_shear=config.getfloat("train_nep", "lambda_shear", fallback=1.0),
        )

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

    def _generate_synthetic_split(self) -> Tuple[List[Atoms], List[Atoms]]:
        """Generate synthetic structures for debug mode."""
        logger.warning("Generating synthetic train/test split for debug mode")

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
            atoms.arrays["forces"] = np.random.rand(2, 3) * 0.1
            train_structures.append(atoms)

        for i in range(2):
            atoms = Atoms(
                "W2",
                positions=[[0, 0, 0], [2.5, 0, 0]],
                cell=[[5, 0, 0], [0, 5, 0], [0, 0, 5]],
                pbc=True,
            )
            atoms.info["energy"] = -10.0 + i * 0.2
            atoms.arrays["forces"] = np.random.rand(2, 3) * 0.1
            test_structures.append(atoms)

        return train_structures, test_structures

    def _generate_nep_config(
        self,
        config: ConfigParser,
        dataset_path: Path,
        hyperparameters: NepHyperparameters | None = None,
    ) -> None:
        """Generate customized nep.in based on project config."""
        # Read template
        template_path = self.project_dir / "config" / "nep" / "nep.in"
        if template_path.exists():
            template_lines = template_path.read_text().splitlines()
            logger.info(f"Using nep.in template from {template_path}")
        else:
            logger.warning(f"nep.in template not found at {template_path}. Using defaults.")
            template_lines = self._get_default_nep_template().splitlines()

        if not template_lines:
            logger.error("Template is empty! Cannot generate nep.in")
            return

        hyperparameters = hyperparameters or self._get_nep_hyperparameters(config)
        all_elements = hyperparameters.all_elements
        logger.info("NEP will train on %s element types: %s", len(all_elements), all_elements)

        population = hyperparameters.population
        batch = hyperparameters.batch
        generation = hyperparameters.generations
        charge_mode = hyperparameters.charge_mode
        outer_zbl = hyperparameters.outer_zbl
        weights = hyperparameters.weights
        cutoff = " ".join(hyperparameters.cutoff)
        n_max = " ".join(hyperparameters.n_max)
        basis_size = " ".join(hyperparameters.basis_size)
        l_max = " ".join(hyperparameters.l_max)
        neuron = " ".join(hyperparameters.neuron)
        lambda_e = hyperparameters.lambda_e
        lambda_f = hyperparameters.lambda_f
        lambda_v = hyperparameters.lambda_v
        lambda_shear = hyperparameters.lambda_shear

        # Build output
        output_lines = []
        for line in template_lines:
            stripped = line.strip()

            if stripped.startswith("type ") and "# type" not in line:
                output_lines.append(f"type {len(all_elements)} {' '.join(all_elements)}")
            elif stripped.startswith("type_weight ") and "# type" not in line:
                output_lines.append(f"type_weight {' '.join(str(w) for w in weights)}")
            elif stripped.startswith("cutoff ") and "# cutoff" not in line:
                output_lines.append(f"cutoff {cutoff}")
            elif stripped.startswith("n_max ") and "# n_max" not in line:
                output_lines.append(f"n_max {n_max}")
            elif stripped.startswith("basis_size ") and "# basis_size" not in line:
                output_lines.append(f"basis_size {basis_size}")
            elif stripped.startswith("l_max ") and "# l_max" not in line:
                output_lines.append(f"l_max {l_max}")
            elif stripped.startswith("neuron ") and "# neuron" not in line:
                output_lines.append(f"neuron {neuron}")
            elif stripped.startswith("population ") and "# population" not in line:
                output_lines.append(f"population {population}")
            elif stripped.startswith("batch ") and "# batch" not in line:
                output_lines.append(f"batch {batch}")
            elif stripped.startswith("generation ") and "# generation" not in line:
                output_lines.append(f"generation {generation}")
            elif stripped.startswith("charge_mode ") and "# charge_mode" not in line:
                output_lines.append(f"charge_mode {charge_mode}")
            elif stripped.startswith("zbl ") and "# zbl" not in line:
                output_lines.append(f"zbl {outer_zbl}")
            elif stripped.startswith("lambda_e ") and "# lambda" not in line:
                output_lines.append(f"lambda_e {lambda_e}")
            elif stripped.startswith("lambda_f ") and "# lambda" not in line:
                output_lines.append(f"lambda_f {lambda_f}")
            elif stripped.startswith("lambda_v ") and "# lambda" not in line:
                output_lines.append(f"lambda_v {lambda_v}")
            elif stripped.startswith("lambda_shear ") and "# lambda" not in line:
                output_lines.append(f"lambda_shear {lambda_shear}")
            else:
                output_lines.append(line)

        required_lines = [
            ("type", f"type {len(all_elements)} {' '.join(all_elements)}"),
            ("type_weight", f"type_weight {' '.join(str(w) for w in weights)}"),
            ("cutoff", f"cutoff {cutoff}"),
            ("n_max", f"n_max {n_max}"),
            ("basis_size", f"basis_size {basis_size}"),
            ("l_max", f"l_max {l_max}"),
            ("neuron", f"neuron {neuron}"),
            ("population", f"population {population}"),
            ("batch", f"batch {batch}"),
            ("generation", f"generation {generation}"),
            ("charge_mode", f"charge_mode {charge_mode}"),
            ("zbl", f"zbl {outer_zbl}"),
            ("lambda_e", f"lambda_e {lambda_e}"),
            ("lambda_f", f"lambda_f {lambda_f}"),
            ("lambda_v", f"lambda_v {lambda_v}"),
            ("lambda_shear", f"lambda_shear {lambda_shear}"),
        ]
        for key, rendered_line in required_lines:
            if not any(
                line.strip().startswith(f"{key} ") and f"# {key}" not in line
                for line in output_lines
            ):
                output_lines.append(rendered_line)

        # Write customized config
        output_path = dataset_path / "nep.in"
        content = "\n".join(output_lines)
        
        if not content.strip():
            logger.error(f"Generated nep.in is empty! Template had {len(template_lines)} lines, output has {len(output_lines)} lines.")
            return
        
        output_path.write_text(content)
        logger.info(f"Wrote {len(output_lines)} lines to {output_path}")

    @staticmethod
    def _get_default_nep_template() -> str:
        """Get default nep.in template if file not found."""
        return """# Do not change this value
version    4

# Training population and generation settings
population 50
batch 3000
generation 250000
charge_mode 0

# Element types and weights
type 2 W O
type_weight 1 1

# Zbl configuration - value is outer cuttoff of ZBL.
zbl        2

# Parameters to be ML optimised
cutoff     6 5
n_max      4 4
basis_size 8 8
l_max      4 2 1
neuron     80

# Loss function parameters
lambda_e   1
lambda_f   1
lambda_v   1
lambda_shear 1
"""

    def _get_or_create_dataset_folder(self) -> Path:
        """Find next available dataset_XXXX folder and create it."""
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
        dataset_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created dataset folder: dataset_{next_id:04d}")

        return dataset_path

    @staticmethod
    def _complete_extraction_report(report: dict, requested_count: int, accepted_count: int) -> None:
        """Normalize reports from the parser, including test doubles."""
        report.setdefault("requested_count", requested_count)
        report.setdefault("accepted_results", [])
        report.setdefault("accepted_content_records", [])
        report.setdefault("rejected_reason_counts", {})
        report["requested_count"] = requested_count
        report["accepted_count"] = accepted_count
        report["rejected_count"] = requested_count - accepted_count
        accounted_for = sum(report["rejected_reason_counts"].values())
        missing_rejections = report["rejected_count"] - accounted_for
        if missing_rejections > 0:
            report["rejected_reason_counts"]["record_rejected"] = (
                report["rejected_reason_counts"].get("record_rejected", 0)
                + missing_rejections
            )

    def _build_dataset_metadata(
        self,
        dataset_path: Path,
        train_report: dict,
        test_report: dict,
        *,
        train_virial: bool,
        allow_partial: bool,
    ) -> dict:
        """Build manifest data from accepted parse results and extraction outcomes."""
        canonical_records = []
        accepted_identities = []
        source_output_hashes = []
        for split, report in (("train", train_report), ("test", test_report)):
            for result in report.get("accepted_results", []):
                canonical_record = {
                    "split": split,
                    "structure_id": result.structure_id,
                    "calculation_identity": dict(result.calculation_identity),
                    "source_outcar_hash": result.source_outcar_hash,
                    "energy": result.energy_ev,
                    "forces": result.forces_ev_per_angstrom.tolist(),
                    "positions": result.positions_angstrom.tolist(),
                    "lattice": result.lattice_angstrom.tolist(),
                    "species": list(result.species),
                    "pbc": list(result.pbc),
                    "label_units": {
                        "energy": result.energy_unit,
                        "forces": result.force_unit,
                    },
                }
                if train_virial:
                    canonical_record["virial"] = (
                        result.virial_ev.tolist() if result.virial_ev is not None else None
                    )
                    canonical_record["label_units"]["virial"] = result.virial_unit
                    canonical_record["virial_convention"] = result.virial_convention
                canonical_records.append(canonical_record)
                accepted_identities.append(
                    {
                        "split": split,
                        "structure_id": result.structure_id,
                        "calculation_identity": dict(result.calculation_identity),
                        "source_outcar": result.source_outcar,
                        "source_outcar_hash": result.source_outcar_hash,
                    }
                )
                if result.source_outcar_hash:
                    source_output_hashes.append(result.source_outcar_hash)

        canonical_records.extend(train_report.get("accepted_content_records", []))
        canonical_records.extend(test_report.get("accepted_content_records", []))
        accepted_count = train_report["accepted_count"] + test_report["accepted_count"]
        if accepted_count and len(canonical_records) != accepted_count:
            raise RuntimeError(
                "Accepted dataset records are missing immutable content provenance"
            )

        label_schema = {
            "version": "nepflow.extxyz.labels.v1",
            "geometry": ["positions", "lattice", "species", "pbc"],
            "energy": True,
            "forces": True,
            "virial": train_virial,
        }
        units = {
            "energy": "eV",
            "forces": "eV/Angstrom",
            "virial": "eV",
        }
        identity_payload = {
            "schema_version": "nepflow.dataset.v1",
            "label_schema": label_schema,
            "units": units,
            "virial_convention": "positive_compression" if train_virial else None,
            "virial_tensor_convention": "cartesian_3x3" if train_virial else None,
            "records": canonical_records,
        }

        content_bytes = json.dumps(
            identity_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        dataset_id = f"dataset_{hashlib.sha256(content_bytes).hexdigest()}"

        train_reasons = dict(train_report["rejected_reason_counts"])
        test_reasons = dict(test_report["rejected_reason_counts"])
        rejection_reasons = {}
        for reason, count in (*train_reasons.items(), *test_reasons.items()):
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + count

        created = datetime.now().isoformat()
        return {
            "dataset_id": dataset_id,
            "dataset_schema_version": "nepflow.dataset.v1",
            "label_schema": label_schema,
            "created": created,
            "creation_timestamp": created,
            "dataset_folder": dataset_path.name,
            "requested_train_structures": train_report["requested_count"],
            "requested_test_structures": test_report["requested_count"],
            "accepted_train_structures": train_report["accepted_count"],
            "accepted_test_structures": test_report["accepted_count"],
            "rejected_train_structures": train_report["rejected_count"],
            "rejected_test_structures": test_report["rejected_count"],
            "train_structures": train_report["accepted_count"],
            "test_structures": test_report["accepted_count"],
            "total_structures": train_report["accepted_count"] + test_report["accepted_count"],
            "rejection_reason_counts": rejection_reasons,
            "exclusion_reasons": rejection_reasons,
            "train_rejection_reason_counts": train_reasons,
            "test_rejection_reason_counts": test_reasons,
            "accepted_calculation_identities": accepted_identities,
            "source_output_hashes": source_output_hashes,
            "virial_required": train_virial,
            "virial_included": train_virial,
            "units": units,
            "virial_convention": "positive_compression",
            "virial_tensor_convention": "cartesian_3x3",
            "partial_dataset_allowed": allow_partial,
        }

    @staticmethod
    def _write_dataset_metadata(dataset_path: Path, metadata: dict) -> None:
        """Write the finalized content-derived .dataset manifest."""
        metadata_path = dataset_path / ".dataset"
        metadata_path.write_text(json.dumps(metadata, indent=2))
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
