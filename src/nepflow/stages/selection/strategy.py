"""Selection strategies and anchor resolution over canonical primitives."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from ase.io import read as ase_read

from nepflow.config.models import SelectionConfig
from nepflow.domain.identities import calculate_structure_id
from nepflow.errors import ValidationError
from nepflow.resources.budget import ResourceBudgetService

from .algorithms import (
    AlgorithmRegistry,
    SelectionAlgorithmRequest,
    SelectionAlgorithmResult,
    dispatch_selection_algorithm,
)
from .sampling import (
    calculate_cross_distance_stats,
    select_farthest_points_for_target,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrainingSelection:
    """Algorithm-neutral training selection with the typed algorithm result."""

    indices: list[int]
    minimum_distance: float | None
    algorithm_result: SelectionAlgorithmResult

    def __iter__(self):
        """Retain the historic ``indices, minimum_distance`` unpacking API."""

        yield self.indices
        yield self.minimum_distance


@dataclass(frozen=True, slots=True)
class SeedAnchorResolution:
    """One deterministic seed-to-generated-anchor resolution."""

    seed_id: str
    parent_structure_id: str
    candidate_index: int
    physical_structure_id: str
    resolution: str

    def as_mapping(self) -> dict[str, Any]:
        return {
            "seed_id": self.seed_id,
            "parent_structure_id": self.parent_structure_id,
            "candidate_index": self.candidate_index,
            "physical_structure_id": self.physical_structure_id,
            "resolution": self.resolution,
        }


def setting(settings: SelectionConfig | Mapping[str, Any], name: str) -> Any:
    """Read one setting from the typed config or a test mapping."""

    if isinstance(settings, Mapping):
        legacy_names = {
            "target_train_count": "target_train",
            "target_test_count": "target_test",
            "target_tolerance": "tolerance",
            "max_search_iterations": "max_iterations",
        }
        if name in settings:
            return settings[name]
        if name == "descriptor_type" and "mean_descriptor" in settings:
            return "structure" if settings["mean_descriptor"] else "atomic"
        return settings[legacy_names.get(name, name)]
    return getattr(settings, name)


def seed_anchor_policy(settings: SelectionConfig | Mapping[str, Any]) -> str:
    """Return the effective seed-anchor policy without changing FPS."""

    algorithm = str(setting(settings, "algorithm")).strip().lower()
    enabled = bool(_setting_or_default(settings, "include_seed_structures", False))
    if algorithm != "information_entropy":
        return "disabled_for_fps"
    return "enabled" if enabled else "opt_out"


def _setting_or_default(
    settings: SelectionConfig | Mapping[str, Any], name: str, default: Any
) -> Any:
    """Read an optional algorithm setting for legacy test mappings."""

    try:
        return setting(settings, name)
    except (AttributeError, KeyError):
        return default


def _algorithm_options(
    settings: SelectionConfig | Mapping[str, Any],
    *,
    ase_structures: Sequence[Any] | None = None,
    composition_aware: bool | None = None,
    local_representation: Any | None = None,
    resource_budget: ResourceBudgetService | None = None,
    checkpoint_root: Path | None = None,
    reports_dir: Path | None = None,
) -> dict[str, Any]:
    """Build algorithm options while retaining legacy mapping defaults."""

    options = {
        "ase_structures": tuple(ase_structures or ()),
        "composition_aware_fps": (
            _setting_or_default(settings, "composition_aware_fps", False)
            if composition_aware is None
            else composition_aware
        ),
        "composition_aware_fps_adaptive_retries": _setting_or_default(
            settings, "composition_aware_fps_adaptive_retries", 4
        ),
        "composition_aware_fps_descriptor_floor_fraction": _setting_or_default(
            settings, "composition_aware_fps_descriptor_floor_fraction", 0.95
        ),
        "composition_aware_fps_frontier_fraction": _setting_or_default(
            settings, "composition_aware_fps_frontier_fraction", 0.1
        ),
        "composition_aware_fps_ternary_weight": _setting_or_default(
            settings, "composition_aware_fps_ternary_weight", 1.0
        ),
        "descriptor_type": _setting_or_default(settings, "descriptor_type", "structure"),
        "fps_target_selector": select_farthest_points_for_target,
        "max_search_iterations": _setting_or_default(settings, "max_search_iterations", 30),
        "target_tolerance": _setting_or_default(settings, "target_tolerance", 50),
        "local_representation": local_representation,
        "background_mass": _setting_or_default(settings, "background_mass", 1.0e-12),
        "entropy_config": getattr(settings, "entropy", None),
        "reference_measure_tolerance": _setting_or_default(
            getattr(settings, "entropy", None) or settings,
            "reference_measure_tolerance",
            1.0e-6,
        ),
        "reference_measure_chunk_size": _setting_or_default(
            getattr(settings, "entropy", None) or settings,
            "reference_measure_chunk_size",
            1024,
        ),
        "resource_budget": resource_budget,
        "reports_dir": None if reports_dir is None else Path(reports_dir),
    }
    if checkpoint_root is not None:
        entropy_root = Path(checkpoint_root)
        options.update(
            {
                "calibration_checkpoint_path": entropy_root / "calibration",
                "contribution_checkpoint_path": entropy_root / "contributions",
                "contribution_storage_path": entropy_root / "candidate-contributions",
            }
        )
    return options


def select_training_set(
    representations: np.ndarray,
    structures: list,
    settings: SelectionConfig | Mapping[str, Any],
    *,
    ase_structures: list | None = None,
    seed_indices: list[int] | None = None,
    single_element_elastic_indices: list[int] | None = None,
    elastic_indices: list[int] | None = None,
    candidate_ids: Sequence[str] | None = None,
    algorithm_id: str = "information_entropy",
    algorithm_registry: AlgorithmRegistry | None = None,
    local_representation: Any | None = None,
    candidate_structure_ids: Sequence[str] | None = None,
    resource_budget: ResourceBudgetService | None = None,
    checkpoint_root: Path | None = None,
    reports_dir: Path | None = None,
) -> TrainingSelection:
    """Dispatch one training-selection algorithm with mandatory anchors."""

    logger.info("")
    logger.info("Step 3: Selecting training set via %s", algorithm_id)

    seed_indices = sorted(set(seed_indices or []))
    single_element_elastic_indices = sorted(set(single_element_elastic_indices or []))
    elastic_indices = sorted(set(elastic_indices or []))
    anchor_indices = sorted(set(seed_indices + single_element_elastic_indices + elastic_indices))
    target_train = setting(settings, "target_train_count")
    if len(anchor_indices) > target_train:
        raise ValueError(
            "Preselected anchor count exceeds target_train_count: "
            "seed anchors=%d, single-element elastic stress anchors=%d, "
            "all elastic stress anchors=%d, unique anchors=%d, target_train_count=%d"
            % (
                len(seed_indices),
                len(single_element_elastic_indices),
                len(elastic_indices),
                len(anchor_indices),
                target_train,
            )
        )

    if seed_indices:
        logger.info("  Preselected %d seed structures as training anchors", len(seed_indices))
    if single_element_elastic_indices:
        logger.info(
            "  Preselected %d single-element elastic stress structures as training anchors",
            len(single_element_elastic_indices),
        )
    if elastic_indices:
        logger.info(
            "  Preselected %d elastic stress structures as training anchors",
            len(elastic_indices),
        )
    composition_aware = _setting_or_default(settings, "composition_aware_fps", False)
    if composition_aware and algorithm_id == "fps":
        logger.info("  Composition-aware FPS enabled for training selection")

    ordered_candidate_ids = (
        tuple(str(index) for index in range(len(structures)))
        if candidate_ids is None
        else tuple(candidate_ids)
    )
    options = _algorithm_options(
        settings,
        ase_structures=ase_structures,
        composition_aware=composition_aware,
        local_representation=local_representation,
        resource_budget=resource_budget,
        checkpoint_root=checkpoint_root,
        reports_dir=reports_dir,
    )
    if candidate_structure_ids is not None:
        if len(candidate_structure_ids) != len(ordered_candidate_ids):
            raise ValueError("candidate structure IDs must match the candidate count")
        options["candidate_structure_ids"] = dict(
            zip(ordered_candidate_ids, map(str, candidate_structure_ids))
        )
    request = SelectionAlgorithmRequest(
        candidate_ids=ordered_candidate_ids,
        candidate_structures=structures,
        representations=representations,
        target_count=target_train,
        anchor_indices=tuple(anchor_indices),
        anchor_ids=tuple(ordered_candidate_ids[index] for index in anchor_indices),
        algorithm_id=algorithm_id,
        algorithm_version="selection-request-v1",
        options=options,
    )
    result = dispatch_selection_algorithm(request, registry=algorithm_registry)
    selected = list(result.selected_indices)
    minimum = result.minimum_distance
    if minimum is None:
        logger.info(
            "  Training set: %d structures (%s; entropy objective diagnostics recorded)",
            len(selected),
            algorithm_id,
        )
    else:
        logger.info(
            "  Training set: %d structures (min_distance=%.6f)",
            len(selected),
            minimum,
        )
    return TrainingSelection(selected, minimum, result)


def select_plain_fps_training_set(
    representations: np.ndarray,
    structures: list,
    settings: SelectionConfig | Mapping[str, Any],
    remaining_indices: list[int],
    remaining_target: int,
) -> tuple[list[int], float]:
    """Compatibility wrapper for the peer FPS algorithm."""

    local_structures = [structures[index] for index in remaining_indices]
    local_ids = tuple(str(index) for index in remaining_indices)
    request = SelectionAlgorithmRequest(
        candidate_ids=local_ids,
        candidate_structures=local_structures,
        representations=representations[remaining_indices],
        target_count=remaining_target,
        anchor_indices=(),
        anchor_ids=(),
        algorithm_id="fps",
        algorithm_version="selection-request-v1",
        options=_algorithm_options(settings),
    )
    result = dispatch_selection_algorithm(request)
    if result.minimum_distance is None:
        raise ValueError("FPS selection did not provide a minimum distance")
    return [remaining_indices[index] for index in result.selected_indices], result.minimum_distance


def select_composition_aware_training_set(
    representations: np.ndarray,
    structures: list,
    settings: SelectionConfig | Mapping[str, Any],
    ase_structures: list,
    anchor_indices: list[int],
    remaining_indices: list[int],
    remaining_target: int,
) -> tuple[list[int], float]:
    """Compatibility wrapper for composition-aware FPS."""

    candidate_ids = tuple(str(index) for index in range(len(structures)))
    options = _algorithm_options(
        settings,
        ase_structures=ase_structures,
        composition_aware=True,
    )
    options["eligible_indices"] = tuple(remaining_indices)
    request = SelectionAlgorithmRequest(
        candidate_ids=candidate_ids,
        candidate_structures=structures,
        representations=representations,
        target_count=len(anchor_indices) + remaining_target,
        anchor_indices=tuple(sorted(anchor_indices)),
        anchor_ids=tuple(candidate_ids[index] for index in sorted(anchor_indices)),
        algorithm_id="fps",
        algorithm_version="selection-request-v1",
        options=options,
    )
    result = dispatch_selection_algorithm(request)
    if result.minimum_distance is None:
        raise ValueError("FPS selection did not provide a minimum distance")
    return list(result.selected_indices), result.minimum_distance


def select_test_set(
    representations: np.ndarray,
    structures: list,
    train_indices: list[int],
    settings: SelectionConfig | Mapping[str, Any],
) -> dict[str, Any]:
    """Select the test set from candidates farthest from training structures."""

    logger.info("")
    logger.info("Step 4: Selecting test set via FPS (from remaining structures)")
    train_set = set(train_indices)
    remaining_indices = np.array(
        [index for index in range(len(structures)) if index not in train_set]
    )
    logger.info("  %d candidates remaining after training selection", len(remaining_indices))

    target_test = setting(settings, "target_test_count")
    if len(remaining_indices) == 0:
        logger.warning("  No structures remain for test set")
        return {
            "test_indices": [],
            "test_min_dist": 0.0,
            "min_train_test_dist": float("inf"),
            "mean_train_test_dist": float("inf"),
            "test_selection_policy": "candidate_mean_fps_legacy",
            "test_selection_version": "candidate-mean-fps-v1",
            "test_selection_provenance": {
                "representation": "candidate-level descriptor vectors",
                "selection_rule": "legacy candidate-mean FPS",
            },
        }

    if target_test >= len(remaining_indices):
        logger.info(
            "  target_test_count (%d) >= remaining (%d), using all remaining for test",
            target_test,
            len(remaining_indices),
        )
        test_indices = remaining_indices.tolist()
        minimum, mean_distance = calculate_cross_distance_stats(
            representations,
            train_indices,
            test_indices,
        )
        return {
            "test_indices": test_indices,
            "test_min_dist": 0.0,
            "min_train_test_dist": minimum,
            "mean_train_test_dist": mean_distance,
            "test_selection_policy": "candidate_mean_fps_legacy",
            "test_selection_version": "candidate-mean-fps-v1",
            "test_selection_provenance": {
                "representation": "candidate-level descriptor vectors",
                "selection_rule": "legacy candidate-mean FPS",
            },
        }

    from scipy.spatial.distance import cdist

    distances_to_train = cdist(
        representations[remaining_indices],
        representations[train_indices],
    ).min(axis=1)
    pool_size = max(
        target_test,
        int(len(remaining_indices) * setting(settings, "test_pool_factor")),
    )
    top_k = np.argsort(distances_to_train)[::-1][:pool_size]
    pool_indices = remaining_indices[top_k]
    logger.info(
        "  Pre-filtered to %d candidates farthest from training "
        "(top %.0f%%, min d_train in pool: %.6f, max: %.6f)",
        len(pool_indices),
        setting(settings, "test_pool_factor") * 100,
        distances_to_train[top_k[-1]],
        distances_to_train[top_k[0]],
    )

    test_local, test_min_dist = select_farthest_points_for_target(
        representations[pool_indices],
        [structures[index] for index in pool_indices],
        setting(settings, "descriptor_type") == "structure",
        target_test,
        setting(settings, "target_tolerance"),
        setting(settings, "max_search_iterations"),
        label="test",
    )
    test_indices = [int(pool_indices[index]) for index in test_local]
    minimum, mean_distance = calculate_cross_distance_stats(
        representations,
        train_indices,
        test_indices,
    )
    logger.info(
        "  Test set: %d structures (min_distance=%.6f)",
        len(test_indices),
        test_min_dist,
    )
    logger.info(
        "  Train<->test nearest-neighbour distance - min: %.6f, mean: %.6f",
        minimum,
        mean_distance,
    )
    return {
        "test_indices": test_indices,
        "test_min_dist": test_min_dist,
        "min_train_test_dist": minimum,
        "mean_train_test_dist": mean_distance,
        "test_selection_policy": "candidate_mean_fps_legacy",
        "test_selection_version": "candidate-mean-fps-v1",
        "test_selection_provenance": {
            "representation": "candidate-level descriptor vectors",
            "selection_rule": "legacy candidate-mean FPS",
        },
    }


def is_elastic_stress(atoms: Any) -> bool:
    """Return whether a structure is tagged as an elastic-stress sample."""

    return str(atoms.info.get("perturbation_type", "")) == "elastic_stress"


def is_single_element_structure(atoms: Any) -> bool:
    """Return whether composition metadata or symbols identify one element."""

    composition = atoms.info.get("composition")
    if isinstance(composition, dict):
        try:
            positive = [
                element for element, fraction in composition.items() if float(fraction) > 0.0
            ]
        except (TypeError, ValueError):
            positive = []
        if positive:
            return len(positive) == 1

    try:
        symbols = atoms.get_chemical_symbols()
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValidationError("Structure does not expose readable chemical symbols") from exc
    return len(set(symbols)) == 1


def _mapping_value(value: Any) -> Mapping[str, Any]:
    """Normalize nested extxyz provenance back into a mapping."""

    if isinstance(value, Mapping):
        return value
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, Mapping) else {}
    return {}


def _lineage_info(atoms: Any) -> tuple[str | None, str | None, str | None]:
    info = getattr(atoms, "info", {})
    if not isinstance(info, Mapping):
        info = {}
    provenance = _mapping_value(info.get("generation_provenance"))
    direct_provenance = _mapping_value(info.get("provenance"))
    parent = info.get("parent_structure_id")
    nested_parent = provenance.get(
        "parent_structure_id", direct_provenance.get("parent_structure_id")
    )
    if parent is not None and nested_parent is not None and str(parent) != str(nested_parent):
        return None, None, "__conflicting_lineage__"
    if parent is None:
        parent = nested_parent
    seed_id = info.get("seed_id")
    nested_seed_id = provenance.get("seed_id", direct_provenance.get("seed_id"))
    if seed_id is not None and nested_seed_id is not None and str(seed_id) != str(nested_seed_id):
        return None, None, "__conflicting_lineage__"
    if seed_id is None:
        seed_id = nested_seed_id
    family = info.get("perturbation_type")
    nested_family = provenance.get(
        "perturbation_family", direct_provenance.get("perturbation_family")
    )
    if (
        family is not None
        and nested_family is not None
        and str(family).lower() != str(nested_family).lower()
    ):
        return (
            None if seed_id is None else str(seed_id),
            None if parent is None else str(parent),
            "__conflicting_family__",
        )
    if family is None:
        family = nested_family
    return (
        None if seed_id is None else str(seed_id),
        None if parent is None else str(parent),
        None if family is None else str(family).strip().lower(),
    )


def resolve_seed_anchors(
    project_dir: Path,
    reference_ase: list,
) -> list[SeedAnchorResolution]:
    """Resolve each persisted seed to exactly one unperturbed candidate.

    Generated representatives carry both the persisted seed ID and the
    physical parent ID in ``generation_provenance``.  A direct physical match
    remains valid for legacy artifacts, but only when it is explicitly marked
    unperturbed; perturbation descendants are never accepted as fallbacks.
    """

    seeds_path = project_dir / "structures" / "seeds" / "base_structures.xyz"
    if not seeds_path.exists():
        raise FileNotFoundError(f"Seed structures requested, but file not found: {seeds_path}")

    logger.info("")
    logger.info("Step 2b: Loading seed structures")
    seed_ase = ase_read(str(seeds_path), index=":", format="extxyz")
    if not isinstance(seed_ase, list):
        seed_ase = [seed_ase]
    if len(seed_ase) == 0:
        raise ValueError(f"Seed inclusion enabled, but no structures were found in {seeds_path}")

    candidate_ids = [calculate_structure_id(atoms) for atoms in reference_ase]
    candidates: list[tuple[str | None, str | None, str | None, str]] = [
        (*_lineage_info(atoms), candidate_id)
        for atoms, candidate_id in zip(reference_ase, candidate_ids)
    ]
    resolutions: list[SeedAnchorResolution] = []
    used_candidates: dict[int, str] = {}
    for seed in seed_ase:
        info = getattr(seed, "info", {})
        persisted_seed_id = info.get("seed_id")
        seed_id = str(persisted_seed_id or info.get("structure_id") or calculate_structure_id(seed))
        parent_structure_id = calculate_structure_id(seed)
        matches: list[tuple[int, str]] = []
        for index, (
            candidate_seed_id,
            candidate_parent_id,
            family,
            candidate_id,
        ) in enumerate(candidates):
            physical_match = candidate_id == parent_structure_id
            if family in {"__conflicting_family__", "__conflicting_lineage__"} or (
                family is not None and family != "unperturbed"
            ):
                continue
            lineage_present = candidate_seed_id is not None or candidate_parent_id is not None
            seed_match = (
                candidate_seed_id is None
                or persisted_seed_id is None
                or candidate_seed_id == str(persisted_seed_id)
            )
            parent_match = candidate_parent_id is None or candidate_parent_id == parent_structure_id
            lineage_match = lineage_present and seed_match and parent_match
            direct_match = physical_match and family in {None, "unperturbed"}
            if lineage_match and family == "unperturbed":
                matches.append((index, "lineage"))
            elif direct_match and not lineage_present:
                matches.append((index, "physical_identity"))

        if len(matches) == 0:
            raise ValueError(
                "Seed inclusion enabled, but no canonical unperturbed generated "
                f"candidate resolved for seed {seed_id!r}; no physically identical "
                f"generated candidate or matching lineage was found (parent_structure_id="
                f"{parent_structure_id})"
            )
        if len(matches) > 1:
            descriptions = ", ".join(str(index) for index, _ in matches)
            raise ValueError(
                f"Seed inclusion enabled, but seed {seed_id!r} has ambiguous "
                f"unperturbed generated candidates at indices [{descriptions}]"
            )
        index, resolution = matches[0]
        previous_seed = used_candidates.get(index)
        if previous_seed is not None and previous_seed != seed_id:
            raise ValueError(
                "Seed inclusion enabled, but generated candidate index "
                f"{index} resolves to multiple seeds: {previous_seed!r} and {seed_id!r}"
            )
        used_candidates[index] = seed_id
        resolutions.append(
            SeedAnchorResolution(
                seed_id=seed_id,
                parent_structure_id=parent_structure_id,
                candidate_index=index,
                physical_structure_id=candidate_ids[index],
                resolution=resolution,
            )
        )

    logger.info(
        "  Matched %d/%d seed structures to canonical unperturbed representatives",
        len(resolutions),
        len(seed_ase),
    )
    return resolutions


def resolve_seed_indices(project_dir: Path, reference_ase: list) -> list[int]:
    """Backward-compatible index-only view of canonical seed resolution."""

    return sorted(item.candidate_index for item in resolve_seed_anchors(project_dir, reference_ase))


def find_elastic_stress_indices(reference_ase: list) -> list[int]:
    """Return all generated candidates carrying elastic-stress provenance."""

    logger.info("")
    logger.info("Step 2d: Loading elastic stress structures")
    indices = [index for index, atoms in enumerate(reference_ase) if is_elastic_stress(atoms)]
    if not indices:
        raise ValueError(
            "Elastic stress structure inclusion enabled, but no generated "
            "structures with perturbation_type=elastic_stress were found. "
            "Run the generate stage with elastic_stress_enabled=true."
        )
    logger.info(
        "  Matched %d elastic stress structures in generated candidates",
        len(indices),
    )
    return indices


def find_single_element_elastic_stress_indices(reference_ase: list) -> list[int]:
    """Return unary candidates carrying elastic-stress provenance."""

    logger.info("")
    logger.info("Step 2c: Loading single-element elastic stress structures")
    indices = [
        index
        for index, atoms in enumerate(reference_ase)
        if is_elastic_stress(atoms) and is_single_element_structure(atoms)
    ]
    if not indices:
        raise ValueError(
            "Single-element elastic stress structure inclusion enabled, but no "
            "generated structures with perturbation_type=elastic_stress and "
            "single-element composition were found. Run the generate stage with "
            "elastic_stress_enabled=true and check unary seed generation."
        )
    logger.info(
        "  Matched %d single-element elastic stress structures in generated candidates",
        len(indices),
    )
    return indices


__all__ = [
    "SeedAnchorResolution",
    "TrainingSelection",
    "find_elastic_stress_indices",
    "find_single_element_elastic_stress_indices",
    "is_elastic_stress",
    "is_single_element_structure",
    "resolve_seed_indices",
    "resolve_seed_anchors",
    "select_composition_aware_training_set",
    "select_plain_fps_training_set",
    "select_test_set",
    "select_training_set",
    "seed_anchor_policy",
    "setting",
]
