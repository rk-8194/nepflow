"""GPUMD validation stage — prepare structures, run validation, analyze results."""

import logging
from configparser import ConfigParser
from pathlib import Path

from ..base import Stage
from nepflow.workflow.resubmission import SelfResubmitExit
from nepflow.stages.validation.preparation import (
    prepare_validation_cases,
    validation_preparation_to_launcher_state,
)
from nepflow.stages.validation.protocols import ValidationCaseSpec
from nepflow.stages.validation.resolution import resolve_model_dataset
from .prepare import (
    finalize_nep_potential,
)
from .launcher import (
    run_validation_launcher,
    read_validation_status,
    write_validation_status,
)
from .analyze import (
    generate_comparison_csv,
    plot_comparison_results,
)

logger = logging.getLogger("nepflow.validate")


class ValidateStage(Stage):
    """Validate trained models with GPUMD simulations."""

    @staticmethod
    def _require_resume_state(status: dict) -> None:
        """Reject incomplete persisted state instead of guessing recovery."""
        if status.get("status") != "running":
            raise ValueError("Validation status must have status='running' to resume")
        required = (
            "model_run_id",
            "potential_path",
            "dataset_path",
            "dataset_name",
            "preparation_state",
            "validation_complete",
            "analysis_complete",
        )
        missing = [key for key in required if key not in status]
        if missing:
            raise ValueError(
                "Validation status is missing required fields: " + ", ".join(missing)
            )
        for field in ("model_run_id", "potential_path", "dataset_path", "dataset_name"):
            if not isinstance(status[field], str) or not status[field].strip():
                raise ValueError(f"Validation status field {field!r} must be a non-empty string")
        for field in ("validation_complete", "analysis_complete"):
            if not isinstance(status[field], bool):
                raise ValueError(f"Validation status field {field!r} must be a boolean")
        if not isinstance(status["preparation_state"], dict):
            raise ValueError("Validation status preparation_state must be an object")
        preparation_state = status["preparation_state"]
        for field in ("validation_root", "struct_count", "struct_folders", "cases"):
            if field not in preparation_state:
                raise ValueError(
                    f"Validation preparation_state is missing required field {field!r}"
                )
        if (
            not isinstance(preparation_state["validation_root"], str)
            or not preparation_state["validation_root"].strip()
        ):
            raise ValueError("Validation preparation_state validation_root must be a non-empty string")
        if not isinstance(preparation_state["struct_count"], int):
            raise ValueError("Validation preparation_state struct_count must be an integer")
        if not isinstance(preparation_state["struct_folders"], list):
            raise ValueError("Validation preparation_state struct_folders must be a list")
        if not isinstance(preparation_state["cases"], list):
            raise ValueError("Validation preparation_state cases must be a list")
        if preparation_state["struct_count"] != len(preparation_state["cases"]):
            raise ValueError(
                "Validation preparation_state struct_count does not match cases"
            )

    def _model_run_id(self, config: ConfigParser, status: dict) -> str:
        """Return the explicitly requested model-run identity."""
        persisted = status.get("model_run_id")
        if persisted:
            return str(persisted)
        for section in ("validate", "gpumd"):
            value = config.get(section, "model_run_id", fallback="").strip()
            if value:
                return value
        raise ValueError(
            "Validation requires an explicit model_run_id in [validate] or [gpumd]"
        )
    
    def run(self) -> None:
        """Execute GPUMD validation workflow.
        
        Stages:
        1. Finalize NEP potential (move nep.txt to gpumd structure)
        2. Prepare canonical identity-bound validation cases
        3. Submit and monitor GPUMD jobs
        4. Analyze results and generate plots
        """
        logger.info("=" * 70)
        logger.info("GPUMD VALIDATION STAGE")
        logger.info("=" * 70)
        
        # Load configuration
        config_path = self._find_config_file()
        config = self._load_config()
        logger.debug(f"Loaded config: {config_path}")
        
        # Check if this is a resubmission
        status_file = self.project_dir / "gpumd" / ".validation_status"
        status_exists = status_file.exists()
        status = read_validation_status(self.project_dir)
        if status_exists:
            if status.get("status") != "running":
                raise ValueError(
                    "Existing validation status has unsupported or missing status"
                )
            self._require_resume_state(status)
            if status["analysis_complete"] is True:
                logger.info("Validation analysis is already complete; nothing to resume")
                return
        
        # === RESUBMISSION PATH ===
        if status.get("status") == "running" and not status.get("analysis_complete"):
            logger.info("Resubmitting from previous run")
            model_run_id = self._model_run_id(config, status)
            logger.info("Resuming validation for model_run_id=%s", model_run_id)
            
            gpumd_potential_dir = Path(status["potential_path"])
            preparation_state = status.get("preparation_state")
            
            if preparation_state and gpumd_potential_dir.exists():
                try:
                    run_validation_launcher(
                        config=config,
                        preparation_state=preparation_state,
                        project_dir=self.project_dir,
                        debug=self.debug,
                        slurm_deadline=self.slurm_deadline,
                    )
                except SelfResubmitExit as e:
                    logger.warning(f"Resubmit needed: {e}")
                    raise
                
                # Reload the authoritative state written by the launcher.
                status = read_validation_status(self.project_dir)
                if status.get("validation_complete") is True:
                    logger.info("Validation jobs completed, proceeding to analysis...")
                    self._run_analysis(config, status)
                
                return
            raise RuntimeError(
                "Persisted validation state cannot be resumed: required potential "
                "or preparation artifacts are missing"
            )
        
        # === NEW SUBMISSION ===
        logger.info("Starting new validation run")
        model_run_id = self._model_run_id(config, status)
        
        # Phase 1: Finalize NEP potential
        logger.info("\n--- Phase 1: Finalizing NEP potential ---")
        try:
            gpumd_potential_dir, dataset_name = finalize_nep_potential(
                self.project_dir,
                model_run_id=model_run_id,
            )
            logger.info(f"NEP potential finalized at: {gpumd_potential_dir}")
        except FileNotFoundError as e:
            logger.error(f"Failed to finalize potential: {e}")
            raise
        
        # Phase 2: Prepare canonical cases.  This legacy stage only adapts the
        # typed result to the launcher's historical status shape.
        logger.info("\n--- Phase 2: Preparing identity-bound validation cases ---")
        try:
            preparation = prepare_validation_cases(
                self.project_dir,
                model_run_id,
                gpumd_potential_dir=gpumd_potential_dir,
            )
            preparation_state = validation_preparation_to_launcher_state(preparation)
            logger.info(
                "Prepared %d identity-bound cases for validation",
                len(preparation.cases),
            )
        except Exception as e:
            logger.error(f"Failed to prepare validation cases: {e}")
            raise
        
        # Save initial status
        status = {
            "status": "running",
            "model_run_id": model_run_id,
            "potential_path": str(gpumd_potential_dir),
            "dataset_path": str(preparation.dataset_path),
            "dataset_name": dataset_name,
            "preparation_state": preparation_state,
            "validation_complete": False,
            "analysis_complete": False,
        }
        write_validation_status(self.project_dir, **status)
        
        # Phase 3: Run validation launcher
        logger.info("\n--- Phase 3: Running GPUMD validation jobs ---")
        try:
            run_validation_launcher(
                config=config,
                preparation_state=preparation_state,
                project_dir=self.project_dir,
                debug=self.debug,
                slurm_deadline=self.slurm_deadline,
            )
        except SelfResubmitExit as e:
            logger.warning(f"Resubmit needed: {e}")
            raise
        
        # Reload the authoritative state written by the launcher. The launcher
        # is the only component that decides whether validation completed.
        status = read_validation_status(self.project_dir)
        if status.get("validation_complete") is not True:
            logger.warning("Validation launcher did not report successful completion")
            return
        logger.info("All validation jobs completed")
        
        # Phase 4: Analyze results
        logger.info("\n--- Phase 4: Analyzing validation results ---")
        self._run_analysis(config, status)
        
        logger.info("=" * 70)
        logger.info("VALIDATION STAGE COMPLETE")
        logger.info("=" * 70)
    
    def _run_analysis(self, config: ConfigParser, status: dict) -> None:
        """Run post-validation analysis and generate reports.
        
        Args:
            config: ConfigParser with settings
            status: Current validation status dict
        """
        try:
            preparation_state = status.get("preparation_state")
            dataset_name = status.get("dataset_name")
            model_run_id = status.get("model_run_id")
            if not isinstance(preparation_state, dict) or not dataset_name or not model_run_id:
                raise ValueError("Validation analysis state is incomplete")
            gpumd_potential_dir = Path(status["potential_path"])
            
            if not preparation_state:
                logger.warning("No preparation state available for analysis")
                return
            
            raw_cases = preparation_state.get("cases")
            if not isinstance(raw_cases, list) or not raw_cases:
                raise ValueError(
                    "Validation analysis requires persisted identity-bound cases"
                )
            cases = tuple(ValidationCaseSpec.from_mapping(item) for item in raw_cases)
            dataset_id = preparation_state.get("dataset_id")
            resolved = resolve_model_dataset(
                self.project_dir,
                str(model_run_id),
                dataset_id=str(dataset_id) if dataset_id else None,
            )

            # Paths
            validation_root = Path(preparation_state["validation_root"])
            
            # Get potential folder name
            potential_name = gpumd_potential_dir.name
            
            # Create CSV
            reports_dir = self.project_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            csv_path = reports_dir / f"{dataset_name}_{potential_name}_comparison.csv"
            logger.info(f"Generating comparison CSV: {csv_path}")
            
            generate_comparison_csv(
                validation_root=validation_root,
                test_xyz_path=None,
                output_csv_path=csv_path,
                cases=cases,
                model=resolved.model_run,
            )
            if not csv_path.exists() or csv_path.stat().st_size == 0:
                raise RuntimeError(f"Comparison CSV was not produced: {csv_path}")
            
            # Generate plots
            logger.info("Generating comparison plots...")
            plot_comparison_results(
                csv_path=csv_path,
                output_dir=reports_dir,
                dataset_name=dataset_name,
                potential_name=potential_name,
            )
            
            # Mark analysis complete
            persisted_status = read_validation_status(self.project_dir)
            persisted_status["analysis_complete"] = True
            write_validation_status(self.project_dir, **persisted_status)
            
            logger.info("Analysis complete")
        
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
