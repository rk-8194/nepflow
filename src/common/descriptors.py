"""Shared NEP descriptor loading and computation helpers."""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from NepTrainKit.core.calculator import NepCalculator

from nepflow.errors import StateError
from nepflow.domain.identities import (
    DESCRIPTOR_CACHE_SCHEMA,
    DescriptorCacheIdentity,
    calculate_structure_id,
)
from nepflow.io.hashing import sha256_file
from nepflow.io.json import dumps, read_json

logger = logging.getLogger("nepflow.common.descriptors")


DESCRIPTOR_CACHE_SCHEMA_VERSION = DESCRIPTOR_CACHE_SCHEMA


def compute_structure_descriptors(
    calc: NepCalculator,
    structures: list,
    *,
    mean_descriptor: bool,
) -> np.ndarray:
    """Compute descriptors across NepTrainKit 2.x and 3.x calculator APIs."""
    if hasattr(calc, "descriptors"):
        return calc.descriptors(structures, mean=mean_descriptor)
    if hasattr(calc, "get_structures_descriptor"):
        return calc.get_structures_descriptor(
            structures,
            mean_descriptor=mean_descriptor,
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
            "structure_id, structure_hash, or a real ASE structure"
        ) from exc


def _model_identity(model_path: Path, model_filename: str) -> dict[str, str]:
    """Return the configured model filename and content hash."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"NEP model not found at {model_path}\n"
            f"Download NEP89 from:\n"
            f"  https://github.com/brucefan1983/GPUMD/tree/master/potentials/nep/nep89_20250409\n"
            f"Place the model file as: {model_path}"
        )
    return {
        "filename": model_filename,
        "sha256": sha256_file(model_path, required=True),
    }


def _descriptor_manifest(
    structure_ids: list[str],
    model: dict[str, str],
    mean_descriptor: bool,
    descriptors: np.ndarray,
) -> dict:
    """Build the deterministic manifest for one descriptor array."""
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
    """Load a cache only when its complete manifest matches current inputs."""
    if not descriptor_cache.exists() or not manifest_path.exists():
        return None

    try:
        manifest = read_json(manifest_path, error_type=StateError)
        if not isinstance(manifest, dict):
            return None
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
        return descriptors
    except (OSError, StateError, TypeError, ValueError, EOFError) as exc:
        logger.warning("  Ignoring invalid descriptor cache metadata: %s", exc)
        return None


def _validate_descriptors(descriptors: np.ndarray, structure_count: int) -> np.ndarray:
    """Validate the descriptor array before it becomes reusable cache state."""
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
    manifest: dict,
) -> None:
    """Atomically write descriptors, then commit their manifest last."""
    descriptor_cache.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=descriptor_cache.parent,
            prefix=f"{descriptor_cache.name}.",
            suffix=".tmp",
            delete=False,
        ) as descriptor_file:
            descriptor_temp = Path(descriptor_file.name)
            temporary_paths.append(descriptor_temp)
            np.save(descriptor_file, descriptors, allow_pickle=False)
            descriptor_file.flush()
            os.fsync(descriptor_file.fileno())

        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=manifest_path.parent,
            prefix=f"{manifest_path.name}.",
            suffix=".tmp",
            encoding="utf-8",
            delete=False,
        ) as manifest_file:
            manifest_temp = Path(manifest_file.name)
            temporary_paths.append(manifest_temp)
            manifest_file.write(dumps(manifest, indent=None))
            manifest_file.flush()
            os.fsync(manifest_file.fileno())

        os.replace(descriptor_temp, descriptor_cache)
        temporary_paths.remove(descriptor_temp)
        os.replace(manifest_temp, manifest_path)
        temporary_paths.remove(manifest_temp)
    finally:
        for temporary_path in temporary_paths:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def load_or_compute_descriptors(
    project_dir: Path,
    structures: list,
    *,
    mean_descriptor: bool,
    batch_size: int,
    nep_model_file: str,
) -> np.ndarray:
    """Load cached descriptors or compute them from the configured NEP model."""
    descriptor_cache = descriptor_cache_path(project_dir)
    nep_model_path = project_dir / "config" / "nep" / nep_model_file
    model = _model_identity(nep_model_path, nep_model_file)
    structure_ids = [_structure_identity(structure) for structure in structures]
    manifest_path = descriptor_manifest_path(project_dir)

    descriptors = _load_valid_cached_descriptors(
        descriptor_cache,
        manifest_path,
        structure_ids,
        model,
        mean_descriptor,
    )
    if descriptors is not None:
        logger.info(f"  Loaded cached descriptors from {descriptor_cache}")
        return descriptors

    calc = NepCalculator(str(nep_model_path))
    logger.info(f"  Loaded NEP model: {nep_model_file}")
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
    logger.info(f"  Saved descriptors to {descriptor_cache}")
    return descriptors


def compute_descriptors_batched(
    calc: NepCalculator,
    structures: list,
    *,
    mean_descriptor: bool,
    batch_size: int,
) -> np.ndarray:
    """Compute descriptors in batches to avoid OOM."""
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
            f"  Batch {end}/{n} ({100 * end / n:.0f}%) - "
            f"{elapsed:.1f}s elapsed, ~{eta:.0f}s remaining"
        )
    return np.concatenate(all_descriptors, axis=0)
