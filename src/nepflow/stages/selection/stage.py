"""Thin orchestration boundary for structure selection."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from ase.io import read as ase_read
from NepTrainKit.core.structure import Structure

from nepflow.config.loader import find_config_path, load_config
from nepflow.config.models import SelectionConfig

from .artifacts import write_selected_structures
from .debug import run_debug_selection
from .models import SelectionResult
from .reports import plot_descriptor_space
from .representations import load_or_calculate_representations
from .strategy import (
    find_elastic_stress_indices,
    find_single_element_elastic_stress_indices,
    resolve_seed_indices,
    select_test_set,
    select_training_set,
)


logger = logging.getLogger("nepflow.selection.stage")


class SelectionStage:
    """Load candidates, compose selection services, and persist artifacts."""

    def __init__(
        self,
        project_name: str,
        config_file: Path,
        state_file: Path,
        project_dir: Path,
        debug: bool = False,
        slurm_deadline: float | None = None,
    ) -> None:
        self.project_name = project_name
        self.config_file = Path(config_file)
        self.state_file = Path(state_file)
        self.project_dir = Path(project_dir)
        self.debug = debug
        self.slurm_deadline = slurm_deadline

    def _find_config_file(self) -> Path:
        """Resolve the canonical project configuration path."""

        return find_config_path(self.project_dir, explicit_path=self.config_file)

    def run(self) -> None:
        logger.info("Running structure selection")
        if self.debug:
            run_debug_selection(self.project_dir)
            return

        config = load_config(
            self._find_config_file(),
            project_name=self.project_name,
            require_scientific_fields=False,
        )
        prepared = self.prepare()
        if prepared is None:
            return
        result = self.execute(config.selection, prepared)
        self.finalize(prepared, result)

    def prepare(self) -> dict | None:
        """Load the generated candidate structures needed for selection."""

        logger.info("")
        logger.info("Step 1: Loading generated structures")
        generated_path = (
            self.project_dir / "structures" / "generated" / "generated_structures.xyz"
        )
        if not generated_path.exists():
            raise FileNotFoundError(
                f"No generated structures found at {generated_path}\n"
                "Run the 'generate' stage first."
            )

        started = time.perf_counter()
        structures = Structure.read_multiple(str(generated_path))
        ase_structures = ase_read(str(generated_path), index=":", format="extxyz")
        if not isinstance(ase_structures, list):
            ase_structures = [ase_structures]
        elapsed = time.perf_counter() - started
        logger.info(
            "  Loaded %d candidate structures (%.1fs)",
            len(structures),
            elapsed,
        )
        if len(structures) == 0:
            logger.warning("No structures to select from - aborting")
            return None

        return {
            "generated_path": generated_path,
            "structures": structures,
            "ase_structures": ase_structures,
        }

    def execute(
        self,
        settings: SelectionConfig,
        prepared: dict,
    ) -> SelectionResult:
        """Compose representation, strategy, and result services."""

        logger.info("")
        logger.info("Step 2: Computing NEP descriptors")
        representations = load_or_calculate_representations(
            self.project_dir,
            prepared["structures"],
            mean_descriptor=settings.descriptor_type == "structure",
            batch_size=settings.batch_size,
            nep_model_file=settings.nep_model_file,
        )
        logger.info("  Descriptor shape: %s", representations.shape)
        logger.info("  Descriptor type: %s", settings.descriptor_type)

        seed_indices: list[int] = []
        if settings.include_seed_structures:
            seed_indices = resolve_seed_indices(
                self.project_dir,
                prepared["ase_structures"],
            )
            logger.info("  Seed anchors enabled: %d structures", len(seed_indices))

        single_element_elastic_indices: list[int] = []
        if settings.include_single_element_elastic_stress_structures:
            single_element_elastic_indices = find_single_element_elastic_stress_indices(
                prepared["ase_structures"]
            )

        elastic_indices: list[int] = []
        if settings.include_elastic_stress_structures:
            elastic_indices = find_elastic_stress_indices(prepared["ase_structures"])

        anchor_indices = sorted(
            set(seed_indices + single_element_elastic_indices + elastic_indices)
        )
        train_indices, train_min_dist = select_training_set(
            representations,
            prepared["structures"],
            settings,
            ase_structures=prepared["ase_structures"],
            seed_indices=seed_indices,
            single_element_elastic_indices=single_element_elastic_indices,
            elastic_indices=elastic_indices,
        )
        test_selection = select_test_set(
            representations,
            prepared["structures"],
            train_indices,
            settings,
        )

        return SelectionResult(
            descriptors=representations,
            train_indices=train_indices,
            train_min_dist=train_min_dist,
            train_seed_count=len(seed_indices),
            train_single_element_elastic_count=len(single_element_elastic_indices),
            train_elastic_count=len(elastic_indices),
            train_anchor_count=len(anchor_indices),
            train_fps_count=len(train_indices) - len(anchor_indices),
            test_indices=test_selection["test_indices"],
            test_min_dist=test_selection["test_min_dist"],
            min_train_test_dist=test_selection["min_train_test_dist"],
            mean_train_test_dist=test_selection["mean_train_test_dist"],
            seed_indices=seed_indices,
            single_element_elastic_indices=single_element_elastic_indices,
            elastic_indices=elastic_indices,
        )

    def finalize(self, prepared: dict, result: SelectionResult) -> None:
        """Write reports/artifacts and emit the established summary."""

        logger.info("")
        logger.info("Step 5: Plotting descriptor space")
        plot_descriptor_space(
            result.descriptors,
            result.train_indices,
            result.test_indices,
            self.project_dir / "reports" / "descriptor_space.png",
        )

        logger.info("")
        logger.info("Step 6: Saving selected structures")
        write_selected_structures(
            self.project_dir,
            prepared["ase_structures"],
            result.train_indices,
            result.test_indices,
        )

        total = len(prepared["structures"])
        logger.info("")
        logger.info("Selection complete from %d candidates:", total)
        logger.info(
            "  Training: %d structures (%d seed anchors, %d single-element "
            "elastic anchors, %d elastic anchors, %d unique anchors, %d "
            "FPS-selected; FPS min_distance=%.6f)",
            len(result.train_indices),
            result.train_seed_count,
            result.train_single_element_elastic_count,
            result.train_elastic_count,
            result.train_anchor_count,
            result.train_fps_count,
            result.train_min_dist,
        )
        logger.info(
            "  Test:     %d structures (FPS min_distance=%.6f)",
            len(result.test_indices),
            result.test_min_dist,
        )
        logger.info(
            "  Train<->test nearest-neighbour distance - min: %.6f, mean: %.6f",
            result.min_train_test_dist,
            result.mean_train_test_dist,
        )


__all__ = ["SelectionStage"]
