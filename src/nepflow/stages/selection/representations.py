"""Identity-safe representation calculation and cache ownership."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from NepTrainKit.core.calculator import NepCalculator

from nepflow.domain.identities import (
    DESCRIPTOR_CACHE_SCHEMA,
    DescriptorCacheIdentity,
    calculate_structure_id,
)
from nepflow.errors import StateError
from nepflow.io.atomic import atomic_write_bytes
from nepflow.io.hashing import sha256_bytes, sha256_file
from nepflow.io.json import read_json_object, write_json

logger = logging.getLogger(__name__)
DESCRIPTOR_CACHE_SCHEMA_VERSION = DESCRIPTOR_CACHE_SCHEMA


def compute_structure_descriptors(
    calc: NepCalculator,
    structures: list,
    *,
    mean_descriptor: bool,
) -> np.ndarray:
    """Compute one descriptor row per structure through NepTrainKit APIs.

    ``structures`` are passed in their existing order.  ``mean_descriptor``
    selects NepTrainKit's per-structure mean representation versus its native
    descriptor aggregation.  Returns a finite-compatible 2D array with shape
    ``(len(structures), n_features)``; the feature width is supplied by the
    model and is not assumed here.  Raises ``AttributeError`` when neither
    supported calculator API is available; shape and finiteness are validated
    by the cache-writing boundary.
    """

    descriptors = getattr(calc, "descriptors", None)
    if callable(descriptors):
        return np.asarray(descriptors(structures, mean=mean_descriptor))
    legacy_descriptors = getattr(calc, "get_structures_descriptor", None)
    if callable(legacy_descriptors):
        return np.asarray(
            legacy_descriptors(
                structures,
                mean_descriptor=mean_descriptor,
            )
        )
    raise AttributeError(
        "NepCalculator does not provide a supported descriptor API. "
        "Expected descriptors() or get_structures_descriptor()."
    )


def descriptor_cache_path(project_dir: Path) -> Path:
    """Return the descriptor cache path for a project."""

    return project_dir / "nep" / "datasets" / "descriptors.npy"


def descriptor_manifest_path(project_dir: Path) -> Path:
    """Return the descriptor-cache manifest path for a project."""

    return project_dir / "nep" / "datasets" / "descriptors.manifest.json"


def _structure_identity(structure: object) -> str:
    """Return the stable physical/input identity for one structure."""

    structure_id = getattr(structure, "structure_id", None)
    if structure_id is not None:
        structure_id = str(structure_id)
        if structure_id:
            return structure_id

    info = getattr(structure, "info", None)
    if isinstance(info, dict):
        for key in ("structure_id", "structure_hash"):
            identity = info.get(key)
            if identity is not None:
                identity = str(identity)
                if identity:
                    return identity

    try:
        return calculate_structure_id(structure)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "Structure has no stable identity; provide structure_id, "
            "structure_hash, or a real ASE structure"
        ) from exc


def structure_identity(structure: object) -> str:
    """Return the canonical identity used to match descriptor structures."""

    return _structure_identity(structure)


def validate_candidate_representation_identity(
    candidate_ids: Sequence[str],
    structure_ids: Sequence[str],
) -> None:
    """Reject state-distinct candidates that the current descriptors cannot distinguish.

    NepTrainKit descriptors are structurally grounded in this phase.  A
    repeated physical structure with multiple candidate IDs therefore cannot
    be sampled safely until a magnetic-aware representation exists in Phase 7.
    """

    if len(candidate_ids) != len(structure_ids):
        raise StateError("Candidate and structure identity counts do not match")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise StateError("Selection candidate IDs are not unique")
    by_structure: dict[str, set[str]] = {}
    for candidate_id, structure_id in zip(candidate_ids, structure_ids):
        by_structure.setdefault(str(structure_id), set()).add(str(candidate_id))
    conflicts = [structure_id for structure_id, values in by_structure.items() if len(values) > 1]
    if conflicts:
        raise StateError(
            "Selection cannot sample multiple candidate IDs sharing a structural-only "
            f"representation: structure_id={conflicts[0]}"
        )


def _model_identity(model_path: Path, model_filename: str) -> dict[str, str]:
    """Return the configured model filename and content hash."""

    if not model_path.exists():
        raise FileNotFoundError(
            f"NEP model not found at {model_path}\n"
            "Download NEP89 from:\n"
            "  https://github.com/brucefan1983/GPUMD/tree/master/potentials/nep/nep89_20250409\n"
            f"Place the model file as: {model_path}"
        )
    digest = sha256_file(model_path, required=True)
    if digest is None:
        raise StateError(f"Could not hash NEP model: {model_path}")
    return {
        "filename": model_filename,
        "sha256": digest,
    }


def _descriptor_manifest(
    structure_ids: list[str],
    model: dict[str, str],
    mean_descriptor: bool,
    descriptors: np.ndarray,
) -> dict[str, Any]:
    """Build the unchanged Phase 2 manifest representation."""

    return DescriptorCacheIdentity(
        structure_ids=tuple(structure_ids),
        model_filename=model["filename"],
        model_sha256=model["sha256"],
        mean_descriptor=bool(mean_descriptor),
        descriptor_shape=tuple(int(value) for value in descriptors.shape),
    ).to_manifest()


def _load_valid_cached_descriptors(
    descriptor_cache: Path,
    manifest_path: Path,
    structure_ids: list[str],
    model: dict[str, str],
    mean_descriptor: bool,
) -> np.ndarray | None:
    """Load a cache only when its complete identity matches current inputs."""

    if not descriptor_cache.exists() or not manifest_path.exists():
        return None

    try:
        manifest = read_json_object(manifest_path, error_type=StateError)
        if manifest.get("schema_version") != DESCRIPTOR_CACHE_SCHEMA_VERSION:
            return None
        if manifest.get("structure_ids") != structure_ids:
            return None
        manifest_model = manifest.get("model")
        if not isinstance(manifest_model, dict):
            return None
        if (
            manifest_model.get("filename") != model["filename"]
            or manifest_model.get("sha256") != model["sha256"]
        ):
            return None
        settings = manifest.get("settings")
        if not isinstance(settings, dict):
            return None
        if settings.get("mean_descriptor") is not bool(mean_descriptor):
            return None

        descriptors = np.load(descriptor_cache, allow_pickle=False)
        if not isinstance(descriptors, np.ndarray) or descriptors.ndim != 2:
            return None
        if manifest.get("descriptor_shape") != list(descriptors.shape):
            return None
        if descriptors.shape[0] != len(structure_ids):
            return None
        if not np.all(np.isfinite(descriptors)):
            return None
        expected_cache_hash = manifest.get("artifact_sha256")
        if not isinstance(expected_cache_hash, str) or not expected_cache_hash:
            return None
        actual_cache_hash = sha256_file(descriptor_cache, required=True, error_type=StateError)
        if actual_cache_hash != expected_cache_hash:
            return None
        return descriptors
    except (OSError, StateError, TypeError, ValueError, EOFError) as exc:
        logger.warning("  Ignoring invalid descriptor cache metadata: %s", exc)
        return None


def _validate_descriptors(descriptors: np.ndarray, structure_count: int) -> np.ndarray:
    """Validate representations before they become reusable cache state."""

    descriptors = np.asarray(descriptors)
    if descriptors.ndim != 2 or descriptors.shape[0] != structure_count:
        raise ValueError(
            "Computed descriptors must be a finite 2D array with one row per structure"
        )
    try:
        finite = np.all(np.isfinite(descriptors))
    except TypeError as exc:
        raise ValueError("Computed descriptors must contain numeric values") from exc
    if not finite:
        raise ValueError("Computed descriptors must contain only finite values")
    return descriptors


def _write_descriptor_cache(
    descriptor_cache: Path,
    manifest_path: Path,
    descriptors: np.ndarray,
    manifest: dict[str, Any],
) -> None:
    """Atomically write the array, then commit the unchanged manifest last."""

    array_buffer = BytesIO()
    np.save(array_buffer, descriptors, allow_pickle=False)
    array_bytes = array_buffer.getvalue()
    previous_array = descriptor_cache.read_bytes() if descriptor_cache.is_file() else None
    previous_manifest = manifest_path.read_bytes() if manifest_path.is_file() else None
    try:
        atomic_write_bytes(descriptor_cache, array_bytes)
        persisted_manifest = dict(manifest)
        persisted_manifest["artifact_sha256"] = sha256_bytes(array_bytes)
        write_json(manifest_path, persisted_manifest, indent=None)
    except BaseException:
        # The array and its identity manifest form one reusable cache record.
        # Restore the prior pair if publication of the second member fails.
        if previous_array is None:
            descriptor_cache.unlink(missing_ok=True)
        else:
            atomic_write_bytes(descriptor_cache, previous_array)
        if previous_manifest is None:
            manifest_path.unlink(missing_ok=True)
        else:
            atomic_write_bytes(manifest_path, previous_manifest)
        raise


def load_or_calculate_representations(
    project_dir: Path,
    structures: list,
    *,
    mean_descriptor: bool,
    batch_size: int,
    nep_model_file: str,
    candidate_ids: Sequence[str] | None = None,
    candidate_structure_ids: Sequence[str] | None = None,
) -> np.ndarray:
    """Load or calculate identity-matched NEP representations.

    Cache rows retain the input structure order and are reusable only when
    structure identities, model filename/content hash, aggregation setting,
    shape, and artifact hash all match.  ``batch_size`` bounds descriptor
    calculation memory.  Returns a ``(len(structures), n_features)`` array
    and atomically publishes a cache/manifest pair on a miss; raises when the
    model is absent or calculated descriptors are non-numeric, non-finite, or
    have the wrong row count.
    """

    descriptor_cache = descriptor_cache_path(project_dir)
    nep_model_path = project_dir / "config" / "nep" / nep_model_file
    structure_ids = [_structure_identity(structure) for structure in structures]
    if candidate_ids is not None:
        validate_candidate_representation_identity(
            candidate_ids,
            structure_ids if candidate_structure_ids is None else candidate_structure_ids,
        )
    model = _model_identity(nep_model_path, nep_model_file)
    manifest_path = descriptor_manifest_path(project_dir)

    descriptors = _load_valid_cached_descriptors(
        descriptor_cache,
        manifest_path,
        structure_ids,
        model,
        mean_descriptor,
    )
    if descriptors is not None:
        logger.info("  Loaded cached descriptors from %s", descriptor_cache)
        return descriptors

    calc = NepCalculator(str(nep_model_path))
    logger.info("  Loaded NEP model: %s", nep_model_file)
    descriptors = _validate_descriptors(
        compute_descriptors_batched(
            calc,
            structures,
            mean_descriptor=mean_descriptor,
            batch_size=batch_size,
        ),
        len(structures),
    )
    manifest = _descriptor_manifest(
        structure_ids,
        model,
        mean_descriptor,
        descriptors,
    )
    _write_descriptor_cache(descriptor_cache, manifest_path, descriptors, manifest)
    logger.info("  Saved descriptors to %s", descriptor_cache)
    return descriptors


def compute_descriptors_batched(
    calc: NepCalculator,
    structures: list,
    *,
    mean_descriptor: bool,
    batch_size: int,
) -> np.ndarray:
    """Compute representations in input order using bounded batches.

    ``batch_size`` is a positive number of structures per calculator call;
    output rows remain aligned with ``structures`` and are concatenated into a
    2D ``(len(structures), n_features)`` array.  The function logs progress
    and elapsed/estimated time as a non-persistent side effect.  Raises the
    calculator's errors or ``ValueError`` for an invalid batch size/empty
    result.
    """

    all_descriptors = []
    n = len(structures)
    t0 = time.perf_counter()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = structures[start:end]
        desc = compute_structure_descriptors(
            calc,
            batch,
            mean_descriptor=mean_descriptor,
        )
        all_descriptors.append(desc)
        elapsed = time.perf_counter() - t0
        rate = end / elapsed if elapsed > 0 else 0
        eta = (n - end) / rate if rate > 0 else 0
        logger.info(
            "  Batch %s/%s (%.0f%%) - %.1fs elapsed, ~%.0fs remaining",
            end,
            n,
            100 * end / n,
            elapsed,
            eta,
        )
    return np.concatenate(all_descriptors, axis=0)


__all__ = [
    "DESCRIPTOR_CACHE_SCHEMA_VERSION",
    "compute_descriptors_batched",
    "compute_structure_descriptors",
    "descriptor_cache_path",
    "descriptor_manifest_path",
    "load_or_calculate_representations",
    "structure_identity",
    "validate_candidate_representation_identity",
]
