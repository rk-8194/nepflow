"""GPUMD validation stage — prepare structures, run validation, analyze results."""

import logging
from configparser import ConfigParser
from pathlib import Path

from ..base import Stage, SelfResubmitExit
from ..train_nep.launcher import read_train_status
from .prepare import (
    finalize_nep_potential,
    prepare_validation_structures,
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
        if not isinstance(status["preparation_state"], dict):
            raise ValueError("Validation status preparation_state must be an object")

    def _model_run_id(self, config: ConfigParser, status: dict) -> str:
        """Return the explicitly requested model-run identity."""
        persisted = status.get("model_run_id")
        if persisted:
            return str(persisted)
        for section in ("validate", "gpumd"):
            value = config.get(section, "model_run_id", fallback="").strip()
            if value:
                return value
        training_status = read_train_status(self.project_dir)
        if training_status.get("status") == "completed" and training_status.get("model_run_id"):
            return str(training_status["model_run_id"])
        raise ValueError(
            "Validation requires an explicit model_run_id in [validate] or [gpumd]"
        )
    
    def run(self) -> None:
        """Execute GPUMD validation workflow.
        
        Stages:
        1. Finalize NEP potential (move nep.txt to gpumd structure)
        2. Prepare validation structures (create struct folders, set replicates)
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
            else:
                logger.info("Preparation state not available — starting fresh")
        
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
        
        # Phase 2: Prepare structures
        logger.info("\n--- Phase 2: Preparing validation structures ---")
        config_gpumd_dir = self.project_dir / "config" / "gpumd"
        if not config_gpumd_dir.exists():
            logger.error(f"config/gpumd not found: {config_gpumd_dir}")
            raise FileNotFoundError(f"Missing config/gpumd directory")
        
        dataset_path = self.project_dir / "nep" / "datasets" / dataset_name
        
        try:
            preparation_state = prepare_validation_structures(
                dataset_path=dataset_path,
                gpumd_potential_dir=gpumd_potential_dir,
                project_dir=self.project_dir,
                config_gpumd_dir=config_gpumd_dir,
            )
            logger.info(f"Prepared {preparation_state['struct_count']} structures for validation")
        except Exception as e:
            logger.error(f"Failed to prepare structures: {e}")
            raise
        
        # Save initial status
        status = {
            "status": "running",
            "model_run_id": model_run_id,
            "potential_path": str(gpumd_potential_dir),
            "dataset_path": str(dataset_path),
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
            if not isinstance(preparation_state, dict) or not dataset_name:
                raise ValueError("Validation analysis state is incomplete")
            gpumd_potential_dir = Path(status["potential_path"])
            
            if not preparation_state:
                logger.warning("No preparation state available for analysis")
                return
            
            # Paths
            validation_root = Path(preparation_state["validation_root"])
            dataset_path = self.project_dir / "nep" / "datasets" / dataset_name
            test_xyz_path = dataset_path / "test.xyz"
            
            # Get potential folder name
            potential_name = gpumd_potential_dir.name
            
            # Create CSV
            reports_dir = self.project_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            csv_path = reports_dir / f"{dataset_name}_{potential_name}_comparison.csv"
            logger.info(f"Generating comparison CSV: {csv_path}")
            
            generate_comparison_csv(
                validation_root=validation_root,
                test_xyz_path=test_xyz_path,
                output_csv_path=csv_path,
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
    
