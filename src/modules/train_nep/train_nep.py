"""NEP model training stage — prepare datasets and submit training jobs."""

import hashlib
import json
import logging
from configparser import ConfigParser
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from ase.io import read as ase_read
from ase.atoms import Atoms

from ..base import Stage, SelfResubmitExit
from .prepare import prepare_dataset
from .submit import submit_training_job
from .launcher import run_launcher, read_train_status, write_train_status

logger = logging.getLogger("nepflow.train_nep")


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
            # Use saved dataset_path if available, otherwise try to find it
            if status.get("dataset_path"):
                dataset_path = Path(status["dataset_path"])
            else:
                try:
                    dataset_path = self._find_dataset_for_potential(potential_path)
                except FileNotFoundError:
                    logger.warning("Could not find dataset from previous run — starting fresh")
                    dataset_path = None
            
            # If dataset still exists, proceed with resubmission
            if dataset_path and dataset_path.exists():
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

        # Write dataset metadata
        self._write_dataset_metadata(
            dataset_path,
            len(train_structures),
            len(test_structures),
        )

        # Extract config options
        train_virial = config.getboolean("train_nep", "train_virial", fallback=False)

        # Step 3: Prepare datasets (parse OUTCAR and write XYZ files)
        logger.info("Step 3: Parsing OUTCAR files and writing XYZ datasets. Training virials: %s", train_virial)
        train_count = prepare_dataset(
            dataset_path=dataset_path / "train.xyz",
            ase_structures=train_structures,
            is_train=True,
            project_dir=self.project_dir,
            train_virial=train_virial,
            debug=self.debug,
        )
        test_count = prepare_dataset(
            dataset_path=dataset_path / "test.xyz",
            ase_structures=test_structures,
            is_train=False,
            project_dir=self.project_dir,
            train_virial=train_virial,
            debug=self.debug,
        )

        if train_count == 0 or test_count == 0:
            logger.error("No valid structures found. Cannot proceed.")
            return

        logger.info(f"Successfully processed {train_count} train and {test_count} test structures")

        # Step 4: Generate customized nep.in
        logger.info("Step 4: Generating customized nep.in")
        self._generate_nep_config(config, dataset_path)
        logger.info(f"nep.in written to {dataset_path / 'nep.in'}")

        # Step 5: Create training run folder
        logger.info("Step 5: Creating potential training folder")
        potential_path = self._create_potential_folder(config, train_count)
        logger.info(f"Training will run in: {potential_path.name}")

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
                    job_id=job_id,
                    status="running",
                    attempt=1,
                )
            except Exception as e:
                logger.error(f"Could not submit SLURM job: {e}")
                write_train_status(
                    self.project_dir,
                    potential_path=str(potential_path),
                    status="failed",
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
        """Find which dataset folder was used for a given potential.
        
        For now, assumes the latest dataset (dataset_XXXX with highest number).
        In future, could track this in potential folder metadata.
        """
        datasets_dir = self.project_dir / "nep" / "datasets"
        if not datasets_dir.exists():
            raise FileNotFoundError(f"No datasets found in {datasets_dir}")
        
        # Find all dataset folders and return the latest
        existing = [
            (int(d.name.split("_")[1]), d)
            for d in datasets_dir.iterdir()
            if d.is_dir() and d.name.startswith("dataset_")
        ]
        
        if not existing:
            raise FileNotFoundError(f"No dataset folders found in {datasets_dir}")
        
        dataset_path = max(existing)[1]
        logger.debug(f"Found dataset: {dataset_path}")
        return dataset_path

    @staticmethod
    def _parse_list(config: ConfigParser, section: str, option: str) -> List[str]:
        """Parse comma-separated list from config."""
        value = config.get(section, option)
        return [item.strip() for item in value.split(",") if item.strip()]

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

    def _generate_nep_config(self, config: ConfigParser, dataset_path: Path) -> None:
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

        # Parse elements
        try:
            elements = self._parse_list(config, "composition", "elements")
            logger.debug(f"Parsed {len(elements)} solid elements: {elements}")
        except Exception as e:
            logger.error(f"Failed to parse composition.elements: {e}. Using fallback W O.")
            elements = ["W", "O"]

        gas_elements = config.get("composition", "gasElements", fallback="")
        if gas_elements:
            try:
                gas_elements = self._parse_list(config, "composition", "gasElements")
                logger.debug(f"Parsed {len(gas_elements)} gas elements: {gas_elements}")
            except Exception as e:
                logger.warning(f"Failed to parse composition.gasElements: {e}")
                gas_elements = []
        else:
            gas_elements = []

        all_elements = elements + gas_elements
        logger.info(f"NEP will train on {len(all_elements)} element types: {all_elements}")

        # Get NEP training parameters
        population = config.getint("train_nep", "population", fallback=50)
        batch = config.getint("train_nep", "batch", fallback=3000)
        generation = config.getint("train_nep", "generation", fallback=250000)
        charge_mode = config.getint("train_nep", "charge_mode", fallback=0)
        outer_zbl = config.getfloat("train_nep", "outerZBL", fallback=2.0)
        weights_str = config.get("train_nep", "weights", fallback=None)
        if charge_mode not in (0, 1):
            logger.warning(
                "Unsupported train_nep.charge_mode=%s. Falling back to 0 (NEP).",
                charge_mode,
            )
            charge_mode = 0

        # Parse weights
        if weights_str:
            try:
                weights = [float(w.strip()) for w in weights_str.replace(",", " ").split()]
                if len(weights) != len(all_elements):
                    logger.warning(
                        f"Weight count ({len(weights)}) != element count ({len(all_elements)}). Using defaults."
                    )
                    weights = [1.0] * len(all_elements)
                else:
                    logger.debug(f"Parsed weights: {weights}")
            except ValueError as e:
                logger.warning(f"Could not parse weights '{weights_str}': {e}. Using defaults.")
                weights = [1.0] * len(all_elements)
        else:
            logger.debug("No weights specified, using uniform weights")
            weights = [1.0] * len(all_elements)

        # Get lambda values
        lambda_e = config.getfloat("train_nep", "lambda_e", fallback=1.0)
        lambda_f = config.getfloat("train_nep", "lambda_f", fallback=1.0)
        lambda_v = config.getfloat("train_nep", "lambda_v", fallback=1.0)
        logger.debug(f"Lambda values: E={lambda_e}, F={lambda_f}, V={lambda_v}")

        # Build output
        output_lines = []
        for line in template_lines:
            stripped = line.strip()

            if stripped.startswith("type ") and "# type" not in line:
                output_lines.append(f"type {len(all_elements)} {' '.join(all_elements)}")
            elif stripped.startswith("type_weight ") and "# type" not in line:
                output_lines.append(f"type_weight {' '.join(str(w) for w in weights)}")
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
            else:
                output_lines.append(line)

        if not any(line.strip().startswith("charge_mode ") for line in output_lines):
            insert_at = 0
            for i, line in enumerate(output_lines):
                if line.strip().startswith("generation "):
                    insert_at = i + 1
                    break
            output_lines.insert(insert_at, f"charge_mode {charge_mode}")

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

    def _write_dataset_metadata(
        self,
        dataset_path: Path,
        train_count: int,
        test_count: int,
    ) -> None:
        """Write .dataset metadata file."""
        metadata = {
            "dataset_id": dataset_path.name,
            "created": datetime.now().isoformat(),
            "train_structures": train_count,
            "test_structures": test_count,
            "total_structures": train_count + test_count,
        }

        metadata_path = dataset_path / ".dataset"
        metadata_path.write_text(json.dumps(metadata, indent=2))
        logger.debug(f"Wrote metadata to {metadata_path}")

    @staticmethod
    def _generate_params_hash(
        cutoff: str,
        n_max: str,
        basis_size: str,
        l_max: str,
        neuron: str,
    ) -> str:
        """Generate short hash from NEP parameters."""
        params_str = f"{cutoff}_{n_max}_{basis_size}_{l_max}_{neuron}"
        hash_obj = hashlib.md5(params_str.encode())
        return hash_obj.hexdigest()[:8]

    @staticmethod
    def _is_weights_uniform(weights_str: str, num_elements: int) -> bool:
        """Check if all weights are uniform (all 1.0)."""
        if not weights_str or not weights_str.strip():
            return True  # No weights specified = uniform
        try:
            weights = [float(w.strip()) for w in weights_str.replace(",", " ").split()]
            if len(weights) != num_elements:
                return False
            return all(abs(w - 1.0) < 1e-6 for w in weights)
        except ValueError:
            return False

    def _create_potential_folder(self, config: ConfigParser, train_count: int) -> Path:
        """Create and return potential folder with proper naming convention.
        
        Naming: [num_train][v?]_[zbl]_[w?]_[params]_[index]
        where:
        - [num_train] = number of training configurations
        - [v?] = "v" if lambda_v > 0, else omitted
        - [zbl] = outerZBL value if > 0
        - [w?] = "w" if weights are not uniform
        - [params] = short hash of NEP parameters
        - [index] = integer for multiple runs with same config
        """
        potentials_dir = self.project_dir / "nep" / "potentials"
        potentials_dir.mkdir(parents=True, exist_ok=True)

        # Gather NEP parameters
        lambda_v = config.getfloat("train_nep", "lambda_v", fallback=1.0)
        outer_zbl = config.getfloat("train_nep", "outerZBL", fallback=2.0)
        weights_str = config.get("train_nep", "weights", fallback="")

        # Parse composition for element count
        try:
            elements = self._parse_list(config, "composition", "elements")
            gas_elements = self._parse_list(config, "composition", "gasElements")
        except Exception:
            elements = ["W", "O"]
            gas_elements = []
        all_elements = elements + gas_elements

        # Parse params for hash
        try:
            cutoff = config.get("train_nep", "cutoff", fallback="6 5")
            n_max = config.get("train_nep", "n_max", fallback="4 4")
            basis_size = config.get("train_nep", "basis_size", fallback="8 8")
            l_max = config.get("train_nep", "l_max", fallback="4 2 1")
            neuron = config.get("train_nep", "neuron", fallback="80")
        except Exception:
            cutoff = n_max = basis_size = l_max = neuron = "default"

        # Build folder name components
        parts = [f"{train_count}"]

        # Add "v" if lambda_v > 0
        if lambda_v > 0:
            parts.append("v")

        # Add ZBL radius if > 0
        if outer_zbl > 0:
            parts.append(str(outer_zbl).replace(".", "_"))

        # Add "w" if weights are not uniform
        if not self._is_weights_uniform(weights_str, len(all_elements)):
            parts.append("w")

        # Add params hash
        params_hash = self._generate_params_hash(cutoff, n_max, basis_size, l_max, neuron)
        parts.append(params_hash)

        # Find next available index
        base_name = "_".join(parts)
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
