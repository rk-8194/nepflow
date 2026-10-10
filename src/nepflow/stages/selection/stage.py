"""Thin orchestration boundary for canonical structure selection."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read as ase_read

from nepflow.config.models import NepflowConfig, SelectionConfig
from nepflow.errors import StateError
from nepflow.io.atomic import atomic_write_bytes
from nepflow.resources.budget import ResourceBudgetService, build_resource_budget
from nepflow.workflow.controller import StageContext

from .algorithms.information_entropy import (
    BandwidthCalibrationResult,
    EntropyBandwidthSettings,
    EntropyPool,
    FrozenBandwidths,
    SparseAtomicKernelGraph,
    SparseCandidateContributions,
    StreamedKernelExecutionSummary,
    build_entropy_diagnostics,
    build_entropy_pool,
)
from .artifacts import verify_selection_artifacts, write_selected_structures
from .debug import run_debug_selection
from .models import SelectionResult
from .persistence import candidate_ids as ordered_candidate_ids
from .persistence import (
    complete_selection_result,
    legacy_selection_run_id,
    mark_selection_pending,
    persist_selection_result,
    restore_selection_result,
    selection_policy,
    selection_run_id,
)
from .persistence import structure_ids as ordered_structure_ids
from .reports import (
    plot_descriptor_space,
    write_diagnostics_report,
    write_entropy_presentation_reports,
)
from .representations import (
    LocalEnvironmentRepresentation,
    LocalRepresentationConfig,
    load_or_calculate_local_representations,
    load_or_calculate_representations,
    validate_candidate_representation_identity,
)
from .sampling import calculate_composition_coverage_metrics, composition_projection_bins
from .strategy import (
    TrainingSelection,
    find_elastic_stress_indices,
    find_single_element_elastic_stress_indices,
    resolve_seed_indices,
    select_test_set,
    select_training_set,
)
from .test_selection import select_local_test_holdout

logger = logging.getLogger(__name__)


def _publication_paths(project_dir: Path) -> tuple[Path, ...]:
    selected = project_dir / "structures" / "selected"
    reports = project_dir / "reports"
    return (
        selected / "train.xyz",
        selected / "test.xyz",
        selected / "selection_manifest.json",
        reports / "descriptor_space.png",
        reports / "entropy_diagnostics.md",
        reports / "entropy_presentation-v1.npz",
        reports / "descriptor_bandwidth.png",
        reports / "entropy_probability_projection.png",
    )


def _snapshot_publication(project_dir: Path) -> dict[Path, bytes | None]:
    return {
        path: path.read_bytes() if path.is_file() else None
        for path in _publication_paths(project_dir)
    }


def _restore_publication(snapshot: dict[Path, bytes | None]) -> None:
    for path, content in snapshot.items():
        if content is None:
            path.unlink(missing_ok=True)
        else:
            atomic_write_bytes(path, content)


def _read_nep_structures(generated_path: Path) -> list[Any]:
    """Load legacy NepTrainKit structures only when the NEP path needs them."""

    from NepTrainKit.core.structure import Structure

    return Structure.read_multiple(str(generated_path))


def _aggregate_local_representations(
    representation: LocalEnvironmentRepresentation,
    candidate_ids: list[str] | tuple[str, ...],
) -> np.ndarray:
    """Produce candidate-mean plotting rows while entropy owns local geometry."""

    rows_by_candidate: dict[str, list[int]] = {candidate_id: [] for candidate_id in candidate_ids}
    for row_index, row in enumerate(representation.rows):
        rows_by_candidate[row.candidate_id].append(row_index)
    return np.asarray(
        [
            np.mean(representation.descriptors[rows_by_candidate[candidate_id]], axis=0)
            for candidate_id in candidate_ids
        ],
        dtype=np.float64,
    )


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
        runtime_budget: ResourceBudgetService | None = None
        if active.debug:
            result = run_debug_selection(
                active.project_dir,
                prepared["ase_structures"],
                random_seed=int(
                    getattr(getattr(active.config, "project", None), "random_seed", 42)
                ),
            )
        else:
            resource_config = getattr(active.config, "resources", None)
            runtime_budget = build_resource_budget(
                execution_mode=getattr(resource_config, "execution_mode", "auto"),
                memory_budget_bytes=getattr(resource_config, "memory_budget_bytes", None),
                scratch_budget_bytes=getattr(resource_config, "scratch_budget_bytes", None),
                reserved_headroom_bytes=getattr(
                    resource_config, "reserved_headroom_bytes", None
                ),
                safety_margin_fraction=getattr(
                    resource_config, "safety_margin_fraction", 0.10
                ),
                scratch_path=getattr(resource_config, "scratch_path", None) or None,
                worker_cap=getattr(resource_config, "worker_cap", None),
            )
            result = self.execute(
                settings,
                prepared,
                context=active,
                resource_budget=runtime_budget,
            )

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
        publication_snapshot = _snapshot_publication(active.project_dir)
        try:
            selection_run = state_store.get_selection_run(
                selection_run_id(active.project_name, candidate_identity_ids, settings)
            )
            if selection_run is None and settings.algorithm == "fps":
                selection_run = state_store.get_selection_run(
                    legacy_selection_run_id(
                        active.project_name,
                        candidate_identity_ids,
                        settings,
                    )
                )
            if selection_run is not None and selection_run.get("status") == "completed":
                try:
                    verify_selection_artifacts(
                        active.project_dir,
                        algorithm_id=result.algorithm_id,
                    )
                except Exception as exc:
                    logger.info("Completed selection needs artifact repair: %s", exc)
                    selection_run = mark_selection_pending(state_store, selection_run)
                    self.finalize(
                        prepared,
                        result,
                        context=active,
                        resource_budget=runtime_budget,
                        repair_only=True,
                    )
                    verify_selection_artifacts(
                        active.project_dir,
                        algorithm_id=result.algorithm_id,
                    )
                    complete_selection_result(state_store, selection_run)
                return result

            if selection_run is None:
                selection_run = persist_selection_result(
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
                    status="pending",
                )
            elif selection_run.get("status") != "pending":
                raise StateError(
                    "Selection run has unsupported recovery status: "
                    f"{selection_run.get('status')!r}"
                )
            self.finalize(prepared, result, context=active, resource_budget=runtime_budget)
            verify_selection_artifacts(active.project_dir, algorithm_id=result.algorithm_id)
            complete_selection_result(state_store, selection_run)
        except BaseException:
            _restore_publication(publication_snapshot)
            raise
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
        structures = _read_nep_structures(generated_path)
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
        resource_budget: ResourceBudgetService | None = None,
    ) -> SelectionResult:
        """Compose representation, strategy, and identity-safe result services."""

        active = context or self.context
        runtime_budget = resource_budget or build_resource_budget()
        logger.info("")
        logger.info("Step 2: Computing %s representations", settings.algorithm)
        project_dir = (
            active.project_dir
            if active is not None
            else Path(prepared["generated_path"]).parents[2]
        )
        candidate_identity_ids = ordered_candidate_ids(prepared["ase_structures"])
        physical_structure_ids = ordered_structure_ids(prepared["ase_structures"])
        candidate_count = len(candidate_identity_ids)
        if settings.algorithm == "information_entropy" and settings.target_train_count == 0:
            raise ValueError(
                "information-entropy production selection does not permit an empty training set; "
                "K=0 is only a mathematical optimizer no-op"
            )
        if (
            settings.algorithm == "information_entropy"
            and settings.target_train_count > candidate_count
        ):
            raise ValueError(
                "information-entropy target_train_count exceeds available candidates: "
                f"requested K={settings.target_train_count}, available M={candidate_count}"
            )

        seed_indices: list[int] = []
        if settings.include_seed_structures:
            seed_indices = resolve_seed_indices(project_dir, prepared["ase_structures"])
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
        if len(anchor_indices) > settings.target_train_count:
            raise ValueError(
                "Preselected anchor count exceeds target_train_count: "
                f"unique anchors={len(anchor_indices)}, "
                f"target_train_count={settings.target_train_count}"
            )
        if (
            settings.algorithm == "information_entropy"
            and settings.target_train_count == candidate_count
        ):
            logger.warning(
                "Entropy selection budget K=M=%d selects all candidates; no holdout remains",
                candidate_count,
            )
        local_representation: LocalEnvironmentRepresentation | None = None
        if settings.algorithm == "information_entropy":
            entropy = settings.entropy
            if not math.isfinite(entropy.beta) or entropy.beta <= 0.0:
                raise ValueError("selection.entropy.beta must be finite and positive")
            if entropy.optimizer_method not in {"lazy_greedy", "full_greedy"}:
                raise ValueError(
                    "selection.entropy.optimizer_method must be lazy_greedy or full_greedy"
                )
            EntropyBandwidthSettings(
                mode=entropy.bandwidth.mode,
                k=entropy.bandwidth.k,
                c=entropy.bandwidth.c,
                k_candidates=entropy.bandwidth.k_candidates,
                c_candidates=entropy.bandwidth.c_candidates,
                backend=entropy.bandwidth.backend,
                metric=entropy.bandwidth.metric,
                chunk_size=entropy.bandwidth.chunk_size,
                calibration_batch_size=None,
            )
            local_representation = load_or_calculate_local_representations(
                project_dir,
                prepared["ase_structures"],
                config=LocalRepresentationConfig(
                    magnetic_mode=settings.local_magnetic_mode,
                    cutoff=entropy.local_cutoff,
                    radial_bins=entropy.local_radial_bins,
                    angular_bins=entropy.local_angular_bins,
                    radial_sigma=entropy.local_radial_sigma,
                    angular_sigma=entropy.local_angular_sigma,
                    species=entropy.local_species,
                    whitening_tolerance=entropy.whitening_tolerance,
                    whitening_regularization=entropy.whitening_regularization,
                    whitening_singular_policy=entropy.whitening_singular_policy,
                ),
                local_descriptor_workers=settings.local_descriptor_workers,
                resource_budget=runtime_budget,
                candidate_ids=candidate_identity_ids,
                structure_ids=physical_structure_ids,
            )
            representations = _aggregate_local_representations(
                local_representation,
                candidate_identity_ids,
            )
            logger.info(
                "  Entropy local representation complete: N=%d atomic rows, d'=%d; "
                "candidate-mean plotting descriptors only: M=%d, d'=%d; test_policy=%s",
                len(local_representation.rows),
                local_representation.descriptors.shape[1],
                representations.shape[0],
                representations.shape[1],
                settings.test_selection_policy,
            )
        else:
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
        if settings.algorithm == "information_entropy":
            logger.info("  Descriptor type: candidate-mean local (plotting only)")
        else:
            logger.info("  Descriptor type: %s", settings.descriptor_type)

        if active is not None and active.state_store is not None:
            existing = active.state_store.get_selection_run(
                selection_run_id(active.project_name, candidate_identity_ids, settings)
            )
            if existing is None and settings.algorithm == "fps":
                existing = active.state_store.get_selection_run(
                    legacy_selection_run_id(
                        active.project_name,
                        candidate_identity_ids,
                        settings,
                    )
                )
            if existing is not None:
                if existing.get("status") not in {"completed", "pending"}:
                    raise StateError(
                        "Selection run exists in an unsupported recovery state; explicit "
                        "reconciliation is required before selecting again"
                    )
                persisted_parameters = existing.get("parameters")
                if not isinstance(persisted_parameters, dict) or persisted_parameters.get(
                    "policy"
                ) != selection_policy(settings):
                    raise StateError("Persisted selection policy does not match current input")
                if settings.algorithm == "information_entropy":
                    persisted_algorithm = persisted_parameters.get("algorithm_id")
                    if persisted_algorithm is None:
                        raise StateError(
                            "Persisted selection has no algorithm identity; refusing to reuse it"
                        )
                    if persisted_algorithm == "information_entropy":
                        entropy_record = persisted_parameters.get("entropy")
                        if not isinstance(entropy_record, dict):
                            raise StateError(
                                "Persisted information-entropy record is not a sparse entropy selection"
                            )
                        if persisted_parameters.get("candidate_ids") != candidate_identity_ids:
                            raise StateError(
                                "Persisted entropy selection requires the original ordered candidates"
                            )
                        if local_representation is None:
                            raise StateError("Entropy representation is unavailable for restore")
                        current_pool = build_entropy_pool(local_representation)
                        if entropy_record.get("pool_fingerprint") != current_pool.fingerprint:
                            raise StateError(
                                "Persisted entropy selection fingerprint does not match current representation"
                            )
                logger.info(
                    "  Reconciled %s selection by candidate identity",
                    existing.get("status", "completed"),
                )
                return restore_selection_result(
                    existing,
                    representations,
                    candidate_identity_ids,
                    candidate_structure_ids=physical_structure_ids,
                )

        training: Any = select_training_set(
            representations,
            prepared["structures"],
            settings,
            ase_structures=prepared["ase_structures"],
            seed_indices=seed_indices,
            single_element_elastic_indices=single_element_elastic_indices,
            elastic_indices=elastic_indices,
            candidate_ids=candidate_identity_ids,
            algorithm_id=settings.algorithm,
            local_representation=local_representation,
            candidate_structure_ids=physical_structure_ids,
            resource_budget=runtime_budget,
        )
        if isinstance(training, TrainingSelection):
            train_indices = training.indices
            train_min_dist = training.minimum_distance
            algorithm_result = training.algorithm_result
        else:  # Compatibility with injected legacy strategy doubles.
            train_indices, train_min_dist = training
            algorithm_result = None
        configured_test_policy = settings.test_selection_policy
        if settings.algorithm == "information_entropy" and configured_test_policy != (
            "candidate_mean_fps_legacy"
        ):
            if local_representation is None:
                raise StateError("Entropy test selection requires the local representation")
            test_selection = select_local_test_holdout(
                local_representation,
                prepared["ase_structures"],
                train_indices,
                anchor_indices=anchor_indices,
                target_count=settings.target_test_count,
                policy=configured_test_policy,
                candidate_ids=candidate_identity_ids,
                structure_ids=physical_structure_ids,
                atom_weighting=settings.test_atom_weighting,
                signature_bins=settings.test_signature_bins,
                novelty_quantile=settings.test_novelty_quantile,
                novelty_beta=settings.test_novelty_beta,
                resource_budget=runtime_budget,
            ).as_mapping()
            logger.info(
                "  Test set policy: %s (%d structures; local-environment representation)",
                test_selection["test_selection_policy"],
                len(test_selection["test_indices"]),
            )
        else:
            # FPS remains an independent peer.  Its historical candidate-vector
            # holdout is retained for FPS runs and only available to entropy as
            # the explicitly named legacy policy.
            test_selection = select_test_set(
                representations,
                prepared["structures"],
                train_indices,
                settings,
            )
        test_indices = [int(index) for index in test_selection["test_indices"]]
        if set(test_indices) & set(train_indices):
            raise StateError("Test selection returned a training candidate")
        train_physical_ids = {physical_structure_ids[index] for index in train_indices}
        test_physical_ids = {physical_structure_ids[index] for index in test_indices}
        if train_physical_ids & test_physical_ids:
            raise StateError("Test selection returned a physical identity from training")
        anchor_physical_ids = {physical_structure_ids[index] for index in anchor_indices}
        if anchor_physical_ids & test_physical_ids:
            raise StateError("Test selection returned a mandatory anchor")

        entropy_result = getattr(algorithm_result, "algorithm_result", None)
        entropy_history = []
        train_acquisition_order: list[str] = []
        train_entropy_values: dict[str, Any] = {}
        entropy_provenance: dict[str, Any] = {}
        entropy_diagnostics = None
        if entropy_result is not None and algorithm_result is not None:
            train_acquisition_order = list(entropy_result.acquisition_order)
            entropy_history = [dict(asdict(step)) for step in entropy_result.history.steps]
            train_entropy_values = {
                "train_entropy_objective": entropy_result.final_objective,
                "train_entropy_cross_entropy": entropy_result.final_cross_entropy,
                "train_entropy_forward_kl": entropy_result.final_forward_kl,
                "train_entropy_pool_fingerprint": entropy_result.pool_fingerprint,
                # Legacy result field retained as an operator-identity alias;
                # diagnostics use the explicit kernel_operator field.
                "train_entropy_graph_fingerprint": (
                    entropy_result.graph_fingerprint or entropy_result.kernel_operator_fingerprint
                ),
                "train_entropy_kernel_operator_fingerprint": (
                    entropy_result.kernel_operator_fingerprint
                ),
                "train_entropy_contributions_fingerprint": entropy_result.contributions_fingerprint,
                "train_entropy_state_fingerprint": entropy_result.state_fingerprint,
            }
            diagnostics = algorithm_result.diagnostics[0] if algorithm_result.diagnostics else {}
            calibration = diagnostics.get("calibration")
            graph = diagnostics.get("graph")
            contributions = diagnostics.get("contributions")
            pool = diagnostics.get("pool")
            entropy_provenance = {
                "greedy_method": entropy_result.method,
                "greedy_method_version": entropy_result.method_version,
                "beta": entropy_result.beta,
                "pool_fingerprint": entropy_result.pool_fingerprint,
                "transform_fingerprint": getattr(pool, "transform_fingerprint", None),
                "representation_fingerprint": getattr(pool, "representation_fingerprint", None),
                "calibration_fingerprint": getattr(calibration, "calibration_fingerprint", None),
                "selected_k": getattr(calibration, "k", None),
                "selected_c": getattr(calibration, "c", None),
                "graph_fingerprint": (
                    entropy_result.graph_fingerprint or entropy_result.kernel_operator_fingerprint
                ),
                "kernel_operator_fingerprint": entropy_result.kernel_operator_fingerprint,
                "atomic_graph_materialized": getattr(graph, "atomic_graph_materialized", True),
                "graph_edge_count": getattr(graph, "edge_count", None),
                "graph_array_bytes": getattr(graph, "array_bytes", None),
                "contributions_fingerprint": entropy_result.contributions_fingerprint,
                "contribution_entry_count": getattr(contributions, "entry_count", None),
                "contribution_array_bytes": getattr(contributions, "array_bytes", None),
            }
            if not isinstance(pool, EntropyPool):
                raise StateError("Information-entropy selection did not retain its pool")
            if not isinstance(calibration, BandwidthCalibrationResult):
                raise StateError("Information-entropy selection did not retain calibration")
            if not isinstance(graph, (SparseAtomicKernelGraph, StreamedKernelExecutionSummary)):
                raise StateError("Information-entropy selection did not retain kernel execution")
            if not isinstance(contributions, SparseCandidateContributions):
                raise StateError("Information-entropy selection did not retain contributions")
            if not isinstance(local_representation, LocalEnvironmentRepresentation):
                raise StateError("Information-entropy selection did not retain closure inputs")
            entropy_diagnostics = build_entropy_diagnostics(
                pool,
                calibration,
                graph,
                contributions,
                entropy_result,
                local_representation,
                test_candidate_ids=[
                    candidate_identity_ids[index] for index in test_selection["test_indices"]
                ],
                candidates=prepared["ase_structures"],
                selected_indices=train_indices,
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
            algorithm_id=(
                algorithm_result.algorithm_id
                if algorithm_result is not None
                else settings.algorithm
            ),
            algorithm_version=(
                algorithm_result.algorithm_version
                if algorithm_result is not None
                else "selection-request-v1"
            ),
            train_acquisition_order=train_acquisition_order,
            train_entropy_history=entropy_history,
            train_entropy_provenance=entropy_provenance,
            entropy_diagnostics=entropy_diagnostics,
            train_min_dist_applicable=settings.algorithm == "fps",
            train_fps_count_applicable=settings.algorithm == "fps",
            test_selection_policy=test_selection["test_selection_policy"],
            test_selection_version=test_selection["test_selection_version"],
            test_selection_provenance=test_selection["test_selection_provenance"],
            **train_entropy_values,
        )

    def finalize(
        self,
        prepared: dict[str, Any],
        result: SelectionResult,
        *,
        context: StageContext | None = None,
        resource_budget: ResourceBudgetService | None = None,
        repair_only: bool = False,
    ) -> SelectionResult:
        """Publish and verify selection artifacts without changing science."""

        active = context or self._active_context(None)
        logger.info("")
        logger.info("Step 5: Plotting descriptor space")
        descriptor_path = active.project_dir / "reports" / "descriptor_space.png"
        if not repair_only or not descriptor_path.is_file():
            plot_descriptor_space(
                result.descriptors,
                result.train_indices,
                result.test_indices,
                active.project_dir / "reports" / "descriptor_space.png",
                descriptor_label=(
                    "PCA of candidate-mean whitened local descriptors"
                    if result.algorithm_id == "information_entropy"
                    else "NEP descriptor space"
                ),
            )

        if repair_only:
            if result.entropy_diagnostics is not None:
                diagnostics_path = active.project_dir / "reports" / "entropy_diagnostics.md"
                if not diagnostics_path.is_file():
                    write_diagnostics_report(result.entropy_diagnostics, diagnostics_path)
                reports_dir = active.project_dir / "reports"
                presentation_paths = (
                    reports_dir / "entropy_presentation-v1.npz",
                    reports_dir / "descriptor_bandwidth.png",
                    reports_dir / "entropy_probability_projection.png",
                )
                if (
                    any(not path.is_file() for path in presentation_paths)
                    and isinstance(result.entropy_diagnostics.pool, EntropyPool)
                    and isinstance(result.entropy_diagnostics.bandwidths, FrozenBandwidths)
                ):
                    write_entropy_presentation_reports(
                        result.entropy_diagnostics,
                        result.descriptors,
                        reports_dir,
                        resource_budget=resource_budget,
                    )
            return result

        logger.info("")
        logger.info("Step 6: Saving selected structures")
        write_selected_structures(
            active.project_dir,
            prepared["ase_structures"],
            result.train_indices,
            result.test_indices,
            candidate_ids=ordered_candidate_ids(prepared["ase_structures"]),
            structure_ids=ordered_structure_ids(prepared["ase_structures"]),
            algorithm_id=result.algorithm_id,
            train_anchor_indices=(
                result.seed_indices + result.single_element_elastic_indices + result.elastic_indices
            ),
            train_acquisition_order=result.train_acquisition_order,
            entropy_diagnostics=(
                None
                if result.entropy_diagnostics is None
                else result.entropy_diagnostics.to_manifest(
                    active.project_dir / "structures" / "selected"
                )
            ),
            test_selection_policy=result.test_selection_policy,
            test_selection_version=result.test_selection_version,
            test_selection_provenance=result.test_selection_provenance,
        )
        if result.entropy_diagnostics is not None:
            write_diagnostics_report(
                result.entropy_diagnostics,
                active.project_dir / "reports" / "entropy_diagnostics.md",
            )
            if (
                isinstance(result.entropy_diagnostics.pool, EntropyPool)
                and isinstance(result.entropy_diagnostics.bandwidths, FrozenBandwidths)
            ):
                write_entropy_presentation_reports(
                    result.entropy_diagnostics,
                    result.descriptors,
                    active.project_dir / "reports",
                    resource_budget=resource_budget,
                )

        total = len(prepared["structures"])
        logger.info("")
        logger.info("Selection complete from %d candidates:", total)
        if result.algorithm_id == "information_entropy":
            logger.info(
                "  Training: %d structures via information_entropy (%d seed anchors, "
                "%d single-element elastic anchors, %d elastic anchors, %d unique anchors; "
                "F=%.6f, cross_entropy=%.6f, forward_KL=%.6f)",
                len(result.train_indices),
                result.train_seed_count,
                result.train_single_element_elastic_count,
                result.train_elastic_count,
                result.train_anchor_count,
                result.train_entropy_objective,
                result.train_entropy_cross_entropy,
                result.train_entropy_forward_kl,
            )
        else:
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
        if result.test_indices:
            if result.test_selection_policy == "candidate_mean_fps_legacy":
                logger.info(
                    "  Test:     %d structures (candidate_mean_fps_legacy; min_distance=%.6f)",
                    len(result.test_indices),
                    result.test_min_dist,
                )
            else:
                purpose = (
                    "representative predictive evaluation"
                    if result.test_selection_policy == "representative"
                    else "extrapolative stress test"
                )
                logger.info(
                    "  Test:     %d structures (%s; local-environment policy)",
                    len(result.test_indices),
                    purpose,
                )
        else:
            logger.info("  Test:     0 structures (no eligible candidates remain)")
        if result.min_train_test_dist is not None and result.mean_train_test_dist is not None:
            logger.info(
                "  Train<->test candidate-vector nearest-neighbour distance - min: %.6f, "
                "mean: %.6f",
                result.min_train_test_dist,
                result.mean_train_test_dist,
            )
        else:
            logger.info("  Train<->test candidate-vector distance: not applicable to local policy")
        return result


__all__ = ["SelectionStage"]
