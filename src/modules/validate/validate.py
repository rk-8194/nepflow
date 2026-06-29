"""GPUMD validation stage — prepare structures, run validation, analyze results."""

import logging
from configparser import ConfigParser
from pathlib import Path

from ..base import Stage, SelfResubmitExit
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
        try:
            status = read_validation_status(self.project_dir)
        except Exception as e:
            logger.debug(f"Could not read status: {e}")
            status = {}
        
        # === RESUBMISSION PATH ===
        if status.get("status") == "running" and not status.get("analysis_complete"):
            logger.info("Resubmitting from previous run")
            
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
                
                # If launcher returns, check validation completion
                if status.get("validation_complete"):
                    logger.info("Validation jobs completed, proceeding to analysis...")
                    self._run_analysis(config, status)
                
                return
            else:
                logger.info("Preparation state not available — starting fresh")
        
        # === NEW SUBMISSION ===
        logger.info("Starting new validation run")
        
        # Phase 1: Finalize NEP potential
        logger.info("\n--- Phase 1: Finalizing NEP potential ---")
        try:
            gpumd_potential_dir, dataset_name = finalize_nep_potential(self.project_dir)
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
        
        # Update status: validation complete
        status["validation_complete"] = True
        write_validation_status(self.project_dir, **status)
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
            
            try:
                generate_comparison_csv(
                    validation_root=validation_root,
                    test_xyz_path=test_xyz_path,
                    output_csv_path=csv_path,
                )
            except Exception as e:
                logger.error(f"Failed to generate CSV: {e}")
                import traceback
                traceback.print_exc()
            
            # Generate plots
            logger.info("Generating comparison plots...")
            try:
                plot_comparison_results(
                    csv_path=csv_path,
                    output_dir=reports_dir,
                    dataset_name=dataset_name,
                    potential_name=potential_name,
                )
            except Exception as e:
                logger.error(f"Failed to generate plots: {e}")
                import traceback
                traceback.print_exc()
            
            # Mark analysis complete
            status["analysis_complete"] = True
            write_validation_status(self.project_dir, **status)
            
            logger.info("Analysis complete")
        
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
