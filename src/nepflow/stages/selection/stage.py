"""Thin orchestration boundary for canonical structure selection."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ase.io import read as ase_read
from NepTrainKit.core.structure import Structure

from nepflow.config.models import NepflowConfig, SelectionConfig
from nepflow.errors import StateError
from nepflow.workflow.controller import StageContext

from .artifacts import write_selected_structures
from .debug import run_debug_selection
from .models import SelectionResult
from .persistence import candidate_ids as ordered_candidate_ids
from .persistence import (
    legacy_selection_run_id,
    persist_selection_result,
    restore_selection_result,
    selection_policy,
    selection_run_id,
)
from .persistence import structure_ids as ordered_structure_ids
from .reports import plot_descriptor_space
from .representations import (
    load_or_calculate_representations,
    validate_candidate_representation_identity,
)
from .sampling import calculate_composition_coverage_metrics, composition_projection_bins
from .strategy import (
    find_elastic_stress_indices,
    find_single_element_elastic_stress_indices,
    resolve_seed_indices,
    select_test_set,
    select_training_set,
)

logger = logging.getLogger(__name__)


class SelectionStage:
    """Compose selection services from an already validated stage context."""

    def __init__(
        self,
        context: StageContext | None = None,
        *,
        selection_config: SelectionConfig | None = None,
    ) -> None:
        self.context = context
        self.selection_config = selection_config

    def _active_context(self, context: StageContext | None) -> StageContext:
        active = context or self.context
        if active is None:
            raise TypeError("SelectionStage requires an injected StageContext")
        return active

    def _settings(self, context: StageContext) -> SelectionConfig:
        settings = self.selection_config
        if settings is None:
            root_config = context.config
            if isinstance(root_config, SelectionConfig):
                settings = root_config
            elif isinstance(root_config, NepflowConfig):
                settings = root_config.selection
        if not isinstance(settings, SelectionConfig):
            raise TypeError("SelectionStage requires the validated SelectionConfig on StageContext")
        return settings

    def run(self, context: StageContext | None = None) -> SelectionResult:
        """Run selection and return its scientific result, not its side effects."""

        active = self._active_context(context)
        settings = self._settings(active)
        state_store = active.state_store
        if state_store is None:
            raise StateError("SelectionStage requires the authoritative StateStore")

        logger.info("Running structure selection")
        prepared = self.prepare(active.project_dir)
        if prepared is None:
            raise ValueError("No generated structures are available for selection")

        started_at = datetime.now(timezone.utc).isoformat()
        if active.debug:
            result = run_debug_selection(
                active.project_dir,
                prepared["ase_structures"],
                random_seed=int(
                    getattr(getattr(active.config, "project", None), "random_seed", 42)
                ),
            )
        else:
            result = self.execute(settings, prepared, context=active)

        candidate_identity_ids = ordered_candidate_ids(prepared["ase_structures"])
        physical_structure_ids = ordered_structure_ids(prepared["ase_structures"])
        coverage_metrics = {}
        if settings.composition_aware_fps:
            candidate_bins = {
                index: composition_projection_bins(atoms)
                for index, atoms in enumerate(prepared["ase_structures"])
            }
            coverage_metrics = calculate_composition_coverage_metrics(
                result.train_indices,
                candidate_bins,
            )
        persist_selection_result(
            state_store,
            active.project_name,
            active.project_name,
            str(active.project_dir),
            settings,
            candidate_identity_ids,
            result,
            candidate_structure_ids=physical_structure_ids,
            coverage_metrics=coverage_metrics,
            started_at=started_at,
        )
        self.finalize(prepared, result, context=active)
        return result

    def prepare(self, project_dir: Path | None = None) -> dict[str, Any] | None:
        """Load the generated candidate structures needed for selection."""

        active_project_dir = Path(project_dir or self._active_context(None).project_dir)
        logger.info("")
        logger.info("Step 1: Loading generated structures")
        generated_path = (
            active_project_dir / "structures" / "generated" / "generated_structures.xyz"
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
        prepared: dict[str, Any],
        *,
        context: StageContext | None = None,
    ) -> SelectionResult:
        """Compose representation, strategy, and identity-safe result services."""

        active = context or self.context
        logger.info("")
        logger.info("Step 2: Computing NEP descriptors")
        project_dir = (
            active.project_dir
            if active is not None
            else Path(prepared["generated_path"]).parents[2]
        )
        candidate_identity_ids = ordered_candidate_ids(prepared["ase_structures"])
        physical_structure_ids = ordered_structure_ids(prepared["ase_structures"])
        validate_candidate_representation_identity(
            candidate_identity_ids,
            physical_structure_ids,
        )
        representations = load_or_calculate_representations(
            project_dir,
            prepared["structures"],
            mean_descriptor=settings.descriptor_type == "structure",
            batch_size=settings.batch_size,
            nep_model_file=settings.nep_model_file,
            candidate_ids=candidate_identity_ids,
            candidate_structure_ids=physical_structure_ids,
        )
        logger.info("  Descriptor shape: %s", representations.shape)
        logger.info("  Descriptor type: %s", settings.descriptor_type)

        if active is not None and active.state_store is not None:
            existing = active.state_store.get_selection_run(
                selection_run_id(active.project_name, candidate_identity_ids, settings)
            )
            if existing is None:
                existing = active.state_store.get_selection_run(
                    legacy_selection_run_id(
                        active.project_name,
                        candidate_identity_ids,
                        settings,
                    )
                )
            if existing is not None:
                if existing.get("status") != "completed":
                    raise StateError(
                        "Selection run exists but is not complete; explicit reconciliation "
                        "is required before selecting again"
                    )
                persisted_parameters = existing.get("parameters")
                if not isinstance(persisted_parameters, dict) or persisted_parameters.get(
                    "policy"
                ) != selection_policy(settings):
                    raise StateError("Persisted selection policy does not match current input")
                logger.info("  Reconciled completed selection by candidate identity")
                return restore_selection_result(
                    existing,
                    representations,
                    candidate_identity_ids,
                    candidate_structure_ids=physical_structure_ids,
                )

        seed_indices: list[int] = []
        if settings.include_seed_structures:
            seed_indices = resolve_seed_indices(
                project_dir,
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
            candidate_ids=candidate_identity_ids,
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

    def finalize(
        self,
        prepared: dict[str, Any],
        result: SelectionResult,
        *,
        context: StageContext | None = None,
    ) -> SelectionResult:
        """Write presentation artifacts without changing the scientific result."""

        active = context or self._active_context(None)
        logger.info("")
        logger.info("Step 5: Plotting descriptor space")
        plot_descriptor_space(
            result.descriptors,
            result.train_indices,
            result.test_indices,
            active.project_dir / "reports" / "descriptor_space.png",
        )

        logger.info("")
        logger.info("Step 6: Saving selected structures")
        write_selected_structures(
            active.project_dir,
            prepared["ase_structures"],
            result.train_indices,
            result.test_indices,
            candidate_ids=ordered_candidate_ids(prepared["ase_structures"]),
            structure_ids=ordered_structure_ids(prepared["ase_structures"]),
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
        return result


__all__ = ["SelectionStage"]
