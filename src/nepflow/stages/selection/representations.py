"""Identity-safe representation calculation and cache ownership."""

from __future__ import annotations

import hashlib
import logging
import math
import multiprocessing as mp
import os
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from multiprocessing.context import BaseContext
from pathlib import Path
from typing import Any, cast

import numpy as np
from ase.neighborlist import neighbor_list

from nepflow.domain.identities import (
    DESCRIPTOR_CACHE_SCHEMA,
    DescriptorCacheIdentity,
    calculate_structure_id,
)
from nepflow.errors import StateError
from nepflow.io.atomic import atomic_write_bytes, atomic_write_stream
from nepflow.io.hashing import sha256_bytes, sha256_canonical_json, sha256_file
from nepflow.io.json import read_json_object, write_json
from nepflow.resources.budget import (
    ResourceBudgetService,
    ResourceCapacityError,
    build_resource_budget,
)

logger = logging.getLogger(__name__)
DESCRIPTOR_CACHE_SCHEMA_VERSION = DESCRIPTOR_CACHE_SCHEMA


def _load_nep_calculator() -> Any:
    """Load NepTrainKit only when the NEP descriptor backend is required."""

    try:
        from NepTrainKit.core.calculator import NepCalculator
    except ImportError as exc:  # pragma: no cover - depends on optional backend installation
        raise ImportError("NepTrainKit is required for the NEP descriptor representation") from exc
    return NepCalculator


def compute_structure_descriptors(
    calc: Any,
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

    descriptor_cache.parent.mkdir(parents=True, exist_ok=True)
    staging_fd, staging_name = tempfile.mkstemp(
        prefix=f".{descriptor_cache.name}.", suffix=".staging", dir=descriptor_cache.parent
    )
    os.close(staging_fd)
    staging_path = Path(staging_name)
    cache_backup = descriptor_cache.with_name(f".{descriptor_cache.name}.previous")
    manifest_backup = manifest_path.with_name(f".{manifest_path.name}.previous")
    cache_had_previous = descriptor_cache.is_file()
    manifest_had_previous = manifest_path.is_file()
    cache_backup.unlink(missing_ok=True)
    manifest_backup.unlink(missing_ok=True)
    try:
        artifact_sha256 = atomic_write_stream(
            staging_path,
            lambda handle: np.save(handle, descriptors, allow_pickle=False),
        )
        persisted_manifest = dict(manifest)
        persisted_manifest["artifact_sha256"] = artifact_sha256
        if cache_had_previous:
            os.replace(descriptor_cache, cache_backup)
        if manifest_had_previous:
            os.replace(manifest_path, manifest_backup)
        os.replace(staging_path, descriptor_cache)
        write_json(manifest_path, persisted_manifest, indent=None)
        cache_backup.unlink(missing_ok=True)
        manifest_backup.unlink(missing_ok=True)
    except BaseException:
        # The array and its identity manifest form one reusable cache record.
        # Restore the prior pair if publication of the second member fails.
        staging_path.unlink(missing_ok=True)
        if cache_backup.is_file():
            descriptor_cache.unlink(missing_ok=True)
            os.replace(cache_backup, descriptor_cache)
        elif not cache_had_previous:
            descriptor_cache.unlink(missing_ok=True)
        if manifest_backup.is_file():
            manifest_path.unlink(missing_ok=True)
            os.replace(manifest_backup, manifest_path)
        elif not manifest_had_previous:
            manifest_path.unlink(missing_ok=True)
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

    calculator_type = _load_nep_calculator()
    calc = calculator_type(str(nep_model_path))
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
    calc: Any,
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


# The local representation below is deliberately independent of NepTrainKit.
# It is a compact reference backend for information-entropy selection, not a
# replacement for the existing NEP/FPS representation path above.
LOCAL_REPRESENTATION_SCHEMA_VERSION = "local-environment-representation-v1"
LOCAL_REPRESENTATION_BACKEND = "invariant-radial-angular-v1"
LOCAL_ENVIRONMENT_ORDERING_VERSION = "candidate-order-atom-key-v1"
LOCAL_PREPROCESSING_VERSION = "full-pool-whitening-v1"
LOCAL_REPRESENTATION_CACHE_SCHEMA_VERSION = "local-representation-cache-v2"


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class LocalRepresentationConfig:
    """Scientific parameters for the potential-independent local backend.

    Each row contains a centre-species one-hot channel, smooth radial
    neighbour histograms, smooth neighbour-pair angular histograms, and,
    when requested, spin-magnitude and centre-neighbour spin-correlation
    channels.  Distances and dot products are the only geometric quantities;
    no Cartesian coordinate is concatenated to the descriptor.
    """

    cutoff: float = 5.0
    radial_bins: int = 8
    angular_bins: int = 8
    radial_sigma: float | None = None
    angular_sigma: float | None = None
    magnetic_mode: str = "structural"
    species: tuple[str, ...] = ()
    whitening_tolerance: float = 1.0e-12
    whitening_regularization: float = 1.0e-12
    whitening_singular_policy: str = "regularize"
    backend_version: str = LOCAL_REPRESENTATION_BACKEND

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.cutoff)) or self.cutoff <= 0.0:
            raise ValueError("cutoff must be a finite positive number")
        if int(self.radial_bins) != self.radial_bins or self.radial_bins < 1:
            raise ValueError("radial_bins must be a positive integer")
        if int(self.angular_bins) != self.angular_bins or self.angular_bins < 1:
            raise ValueError("angular_bins must be a positive integer")
        mode = str(self.magnetic_mode).strip().lower()
        aliases = {
            "none": "structural",
            "structural_only": "structural",
            "non-soc": "non_soc",
            "magnetic": "non_soc",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"structural", "non_soc"}:
            raise ValueError("magnetic_mode must be 'structural' or 'non_soc'")
        tolerance = float(self.whitening_tolerance)
        regularization = float(self.whitening_regularization)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("whitening_tolerance must be finite and positive")
        if not math.isfinite(regularization) or regularization < 0.0:
            raise ValueError("whitening_regularization must be finite and non-negative")
        policy = str(self.whitening_singular_policy).strip().lower()
        if policy not in {"drop", "regularize", "reject"}:
            raise ValueError("whitening_singular_policy must be one of: drop, regularize, reject")
        species = tuple(
            sorted({str(symbol).strip() for symbol in self.species if str(symbol).strip()})
        )
        if self.radial_sigma is not None and (
            not math.isfinite(float(self.radial_sigma)) or float(self.radial_sigma) <= 0.0
        ):
            raise ValueError("radial_sigma must be finite and positive when supplied")
        if self.angular_sigma is not None and (
            not math.isfinite(float(self.angular_sigma)) or float(self.angular_sigma) <= 0.0
        ):
            raise ValueError("angular_sigma must be finite and positive when supplied")
        object.__setattr__(self, "cutoff", float(self.cutoff))
        object.__setattr__(self, "radial_bins", int(self.radial_bins))
        object.__setattr__(self, "angular_bins", int(self.angular_bins))
        object.__setattr__(self, "magnetic_mode", mode)
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "whitening_tolerance", tolerance)
        object.__setattr__(self, "whitening_regularization", regularization)
        object.__setattr__(self, "whitening_singular_policy", policy)

    def to_dict(self, *, species: Sequence[str] | None = None) -> dict[str, Any]:
        """Return the complete scientific parameter payload."""

        selected_species = self.species if species is None else tuple(sorted(map(str, species)))
        return {
            "backend": LOCAL_REPRESENTATION_BACKEND,
            "backend_version": self.backend_version,
            "cutoff": self.cutoff,
            "radial_bins": self.radial_bins,
            "angular_bins": self.angular_bins,
            "radial_sigma": self.radial_sigma,
            "angular_sigma": self.angular_sigma,
            "species": list(selected_species),
            "magnetic_mode": self.magnetic_mode,
            "whitening_tolerance": self.whitening_tolerance,
            "whitening_regularization": self.whitening_regularization,
            "whitening_singular_policy": self.whitening_singular_policy,
        }


@dataclass(frozen=True, slots=True)
class LocalEnvironmentRow:
    """Identity mapping for one local descriptor row."""

    candidate_id: str
    structure_id: str
    atom_index: int

    @property
    def local_environment_index(self) -> int:
        """Readable alias for the atom index within the candidate."""

        return self.atom_index

    def to_dict(self, row_index: int | None = None) -> dict[str, Any]:
        result = {
            "candidate_id": self.candidate_id,
            "structure_id": self.structure_id,
            "atom_index": self.atom_index,
            "local_environment_index": self.atom_index,
        }
        if row_index is not None:
            result["row_index"] = int(row_index)
        return result


@dataclass(frozen=True, slots=True)
class WhiteningTransform:
    """Deterministic full-pool whitening transform and its fingerprints."""

    mean: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    retained_indices: tuple[int, ...]
    tolerance: float
    regularization: float
    singular_policy: str
    covariance_fingerprint: str
    eigenvalue_fingerprint: str
    fingerprint: str

    @property
    def retained_dimensions(self) -> int:
        return len(self.retained_indices)

    @property
    def transform_fingerprint(self) -> str:
        return self.fingerprint

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Apply ``Lambda**-1/2 V.T (phi - mean)`` to row-oriented input."""

        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.mean.shape[0]:
            raise ValueError("values must be a 2D array with the fitted feature width")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("values must contain only finite values")
        if not self.retained_indices:
            return np.empty((matrix.shape[0], 0), dtype=np.float64)
        indices = np.asarray(self.retained_indices, dtype=int)
        eigenvalues = self.eigenvalues[indices]
        if self.singular_policy == "regularize":
            eigenvalues = np.maximum(eigenvalues, self.tolerance) + self.regularization
        return (matrix - self.mean) @ self.eigenvectors[:, indices] / np.sqrt(eigenvalues)

    def to_manifest(self) -> dict[str, Any]:
        """Serialize fitted numerical state, not merely its dimensions."""

        return {
            "version": LOCAL_PREPROCESSING_VERSION,
            "mean": self.mean.tolist(),
            "eigenvalues": self.eigenvalues.tolist(),
            "eigenvectors": self.eigenvectors.tolist(),
            "retained_indices": list(self.retained_indices),
            "retained_dimensions": self.retained_dimensions,
            "tolerance": self.tolerance,
            "regularization": self.regularization,
            "singular_policy": self.singular_policy,
            "covariance_fingerprint": self.covariance_fingerprint,
            "eigenvalue_fingerprint": self.eigenvalue_fingerprint,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class LocalEnvironmentRepresentation:
    """Raw and whitened local rows with complete identity metadata."""

    raw_descriptors: np.ndarray
    descriptors: np.ndarray
    rows: tuple[LocalEnvironmentRow, ...]
    candidate_ids: tuple[str, ...]
    structure_ids: tuple[str, ...]
    config: LocalRepresentationConfig
    transform: WhiteningTransform
    fingerprint: str
    candidate_content_fingerprints: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.raw_descriptors.ndim != 2 or self.descriptors.ndim != 2:
            raise ValueError("local descriptors must be 2D arrays")
        if self.raw_descriptors.shape[0] != len(self.rows):
            raise ValueError("local descriptor rows and row metadata must have equal length")
        if self.descriptors.shape[0] != len(self.rows):
            raise ValueError("whitened descriptor rows and row metadata must have equal length")

    @property
    def local_descriptors(self) -> np.ndarray:
        """Alias used by information-entropy consumers."""

        return self.descriptors

    @property
    def row_mapping(self) -> tuple[LocalEnvironmentRow, ...]:
        return self.rows

    @property
    def transform_fingerprint(self) -> str:
        return self.transform.fingerprint


def _local_candidate_ids(
    candidates: Sequence[Any],
    candidate_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    if candidate_ids is not None:
        result = tuple(str(value) for value in candidate_ids)
        if len(result) != len(candidates):
            raise ValueError("candidate_ids must match the candidate count")
    else:
        values: list[str] = []
        for candidate in candidates:
            info = getattr(candidate, "info", {})
            identity = info.get("candidate_id") if isinstance(info, Mapping) else None
            values.append(str(identity or _structure_identity(candidate)))
        result = tuple(values)
    if any(not value.strip() for value in result):
        raise ValueError("candidate IDs must not be blank")
    if len(set(result)) != len(result):
        raise StateError("Selection candidate IDs are not unique")
    return result


def _local_structure_ids(
    candidates: Sequence[Any],
    structure_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    if structure_ids is not None:
        result = tuple(str(value) for value in structure_ids)
        if len(result) != len(candidates):
            raise ValueError("structure_ids must match the candidate count")
        return result
    return tuple(_structure_identity(candidate) for candidate in candidates)


def _coerce_local_representation_config(
    config: LocalRepresentationConfig | Mapping[str, Any] | None,
) -> LocalRepresentationConfig:
    if isinstance(config, LocalRepresentationConfig):
        return config
    parameters = dict(config or {})
    parameters.pop("backend", None)
    return LocalRepresentationConfig(**parameters)


def _magnetic_vectors(candidate: Any) -> np.ndarray:
    """Read magnetic vectors from the canonical generated-candidate fields."""

    arrays = getattr(candidate, "arrays", {})
    value: Any = None
    if isinstance(arrays, Mapping):
        for key in ("magnetic_moments", "initial_magmoms", "magmoms"):
            if key in arrays:
                value = arrays[key]
                break
    if value is None:
        getter = getattr(candidate, "get_magnetic_moments", None)
        if callable(getter):
            value = getter()
    if value is None:
        return np.zeros((len(candidate), 3), dtype=np.float64)
    moments = np.asarray(value, dtype=np.float64)
    if moments.ndim == 1:
        if moments.shape[0] != len(candidate):
            raise ValueError("magnetic moment count must match atom count")
        moments = np.column_stack((np.zeros_like(moments), np.zeros_like(moments), moments))
    if moments.shape != (len(candidate), 3) or not np.all(np.isfinite(moments)):
        raise ValueError("magnetic_moments must be a finite (n_atoms, 3) array")
    return moments


def _candidate_content_fingerprint(candidate: Any, *, magnetic_mode: str) -> str:
    payload: dict[str, Any] = {
        "symbols": list(candidate.get_chemical_symbols()),
        "positions": np.asarray(candidate.get_positions(), dtype=np.float64).tolist(),
        "cell": np.asarray(candidate.get_cell(), dtype=np.float64).tolist(),
        "pbc": np.asarray(candidate.pbc, dtype=bool).reshape(-1).tolist(),
    }
    if magnetic_mode == "non_soc":
        payload["magnetic_moments"] = _magnetic_vectors(candidate).tolist()
    return sha256_canonical_json(payload)


def _gaussian_histogram(value: float, centres: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((float(value) - centres) / sigma) ** 2)


@dataclass(frozen=True, slots=True)
class _LocalStructureContext:
    """Precomputed, per-structure inputs shared by every centre atom."""

    symbols: tuple[str, ...]
    species: tuple[str, ...]
    config: LocalRepresentationConfig
    species_index: dict[str, int]
    radial_centres: np.ndarray
    radial_sigma: float
    angular_centres: np.ndarray
    angular_sigma: float
    neighbours: tuple[tuple[tuple[int, np.ndarray, float], ...], ...]
    magnetic_moments: np.ndarray
    centre_one_hot: np.ndarray
    pair_indices: dict[tuple[int, int], int]


def _prepare_local_structure_context(
    candidate: Any,
    species: tuple[str, ...],
    config: LocalRepresentationConfig,
    magnetic_moments: np.ndarray,
) -> _LocalStructureContext:
    """Build one reusable neighbour and basis context for a candidate."""

    symbols = tuple(str(symbol) for symbol in candidate.get_chemical_symbols())
    positions = np.asarray(candidate.get_positions(), dtype=np.float64)
    cell = np.asarray(candidate.get_cell(), dtype=np.float64)
    species_index = {symbol: index for index, symbol in enumerate(species)}
    radial_centres = np.linspace(
        config.cutoff / (2.0 * config.radial_bins),
        config.cutoff - config.cutoff / (2.0 * config.radial_bins),
        config.radial_bins,
    )
    radial_sigma = config.radial_sigma or config.cutoff / config.radial_bins
    angular_centres = np.linspace(
        -1.0 + 1.0 / config.angular_bins,
        1.0 - 1.0 / config.angular_bins,
        config.angular_bins,
    )
    angular_sigma = config.angular_sigma or 2.0 / config.angular_bins
    grouped_neighbours: list[list[tuple[int, np.ndarray, float]]] = [[] for _ in symbols]
    indices, neighbour_indices, offsets = neighbor_list("ijS", candidate, config.cutoff)
    for source, target, offset in zip(indices, neighbour_indices, offsets):
        source_index = int(source)
        target_index = int(target)
        offset_vector = np.asarray(offset, dtype=np.float64) @ cell
        vector = positions[target_index] + offset_vector - positions[source_index]
        distance = float(np.linalg.norm(vector))
        if distance <= 1.0e-12 or distance > config.cutoff + 1.0e-12:
            continue
        grouped_neighbours[source_index].append((target_index, vector, distance))

    pair_indices: dict[tuple[int, int], int] = {}
    pair_channel = 0
    for first_species in range(len(species)):
        for second_species in range(first_species, len(species)):
            pair_indices[(first_species, second_species)] = pair_channel
            pair_channel += 1
    return _LocalStructureContext(
        symbols=symbols,
        species=species,
        config=config,
        species_index=species_index,
        radial_centres=radial_centres,
        radial_sigma=radial_sigma,
        angular_centres=angular_centres,
        angular_sigma=angular_sigma,
        neighbours=tuple(tuple(values) for values in grouped_neighbours),
        magnetic_moments=magnetic_moments,
        centre_one_hot=np.eye(len(species), dtype=np.float64),
        pair_indices=pair_indices,
    )


def _local_descriptor_for_atom(
    context: _LocalStructureContext,
    atom_index: int,
) -> np.ndarray:
    """Calculate one local descriptor from a precomputed structure context."""

    centre_symbol = context.symbols[atom_index]
    radial = np.zeros(
        (len(context.species), context.config.radial_bins),
        dtype=np.float64,
    )
    neighbours = context.neighbours[atom_index]
    for target_index, _, distance in neighbours:
        radial[context.species_index[context.symbols[target_index]]] += _gaussian_histogram(
            distance,
            context.radial_centres,
            context.radial_sigma,
        )

    angular = np.zeros(
        (
            len(context.species) * (len(context.species) + 1) // 2,
            context.config.angular_bins,
        ),
        dtype=np.float64,
    )
    for left in range(len(neighbours)):
        left_index, left_vector, left_distance = neighbours[left]
        for right_index, right_vector, right_distance in neighbours[left + 1 :]:
            cosine = float(np.dot(left_vector, right_vector) / (left_distance * right_distance))
            cosine = float(np.clip(cosine, -1.0, 1.0))
            left_species = context.species_index[context.symbols[left_index]]
            right_species = context.species_index[context.symbols[right_index]]
            pair = (min(left_species, right_species), max(left_species, right_species))
            angular[context.pair_indices[pair]] += _gaussian_histogram(
                cosine,
                context.angular_centres,
                context.angular_sigma,
            )

    features = [
        context.centre_one_hot[context.species_index[centre_symbol]],
        radial.reshape(-1),
        angular.reshape(-1),
    ]
    if context.config.magnetic_mode == "non_soc":
        centre_spin = context.magnetic_moments[atom_index]
        centre_magnitude = float(np.linalg.norm(centre_spin))
        magnetic_radial = np.zeros_like(radial)
        correlation = np.zeros(
            (len(context.species), context.config.angular_bins),
            dtype=np.float64,
        )
        centre_unit = centre_spin / centre_magnitude if centre_magnitude > 1.0e-12 else None
        for neighbour_index, _, distance in neighbours:
            neighbour_spin = context.magnetic_moments[neighbour_index]
            neighbour_magnitude = float(np.linalg.norm(neighbour_spin))
            magnetic_radial[context.species_index[context.symbols[neighbour_index]]] += (
                neighbour_magnitude
                * _gaussian_histogram(
                    distance,
                    context.radial_centres,
                    context.radial_sigma,
                )
            )
            if centre_unit is not None and neighbour_magnitude > 1.0e-12:
                relative_spin = float(np.dot(centre_unit, neighbour_spin / neighbour_magnitude))
            else:
                relative_spin = 0.0
            correlation[context.species_index[context.symbols[neighbour_index]]] += (
                _gaussian_histogram(
                    relative_spin,
                    context.angular_centres,
                    context.angular_sigma,
                )
            )
        features.extend(
            [
                np.asarray([centre_magnitude], dtype=np.float64),
                magnetic_radial.reshape(-1),
                correlation.reshape(-1),
            ]
        )
    return np.concatenate(features).astype(np.float64, copy=False)


@dataclass(frozen=True, slots=True)
class _LocalDescriptorTask:
    """One picklable, candidate-scoped raw descriptor task."""

    candidate_index: int
    candidate: Any
    species: tuple[str, ...]
    config: LocalRepresentationConfig


@dataclass(frozen=True, slots=True)
class _LocalDescriptorResult:
    """Raw descriptors and deterministic atom ordering from one candidate."""

    candidate_index: int
    descriptors: np.ndarray
    atom_order: tuple[int, ...]


class LocalDescriptorWorkerError(RuntimeError):
    """A candidate-scoped local descriptor worker failure."""


def _compute_local_descriptor_task(task: _LocalDescriptorTask) -> _LocalDescriptorResult:
    """Compute one candidate's raw local descriptors in a worker-safe function."""

    moments = (
        _magnetic_vectors(task.candidate)
        if task.config.magnetic_mode == "non_soc"
        else np.zeros((len(task.candidate), 3), dtype=np.float64)
    )
    context = _prepare_local_structure_context(
        task.candidate,
        task.species,
        task.config,
        moments,
    )
    local_values = [
        _local_descriptor_for_atom(context, atom_index)
        for atom_index in range(len(context.symbols))
    ]
    order = tuple(
        sorted(
            range(len(local_values)),
            key=lambda index: (
                tuple(float(round(value, 12)) for value in local_values[index]),
                context.symbols[index],
                index,
            ),
        )
    )
    feature_width = _local_feature_width(task.config, task.species)
    descriptors = np.asarray(
        [local_values[index] for index in order],
        dtype=np.float64,
    ).reshape((len(order), feature_width))
    return _LocalDescriptorResult(task.candidate_index, descriptors, order)


def _local_feature_width(
    config: LocalRepresentationConfig,
    species: Sequence[str],
) -> int:
    """Return the fixed raw descriptor width for one local environment."""

    species_count = len(species)
    pair_count = species_count * (species_count + 1) // 2
    width = species_count + species_count * config.radial_bins
    width += pair_count * config.angular_bins
    if config.magnetic_mode == "non_soc":
        width += 1 + species_count * config.radial_bins
        width += species_count * config.angular_bins
    return width


def _limit_local_worker_threads() -> None:
    """Limit native math-library pools inside a child process only."""

    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"


def _effective_local_cpu_allocation() -> int:
    """Return the CPUs available to this process and its scheduler allocation."""

    limits: list[int] = []
    affinity = cast(Any, getattr(os, "sched_getaffinity", None))
    if callable(affinity):
        try:
            limits.append(len(cast(Any, affinity(0))))
        except OSError:
            pass
    for name in (
        "SLURM_CPUS_PER_TASK",
        "SLURM_CPUS_ON_NODE",
        "SLURM_JOB_CPUS_PER_NODE",
    ):
        value = os.environ.get(name, "").strip()
        if not value:
            continue
        token = value.split(",", 1)[0].split("(", 1)[0].strip()
        try:
            limits.append(int(token))
        except ValueError:
            continue
    limits.append(os.cpu_count() or 1)
    return max(1, min(limit for limit in limits if limit > 0))


def _resolve_local_descriptor_workers(
    requested: int,
    candidate_count: int,
    worker_cap: int | None = None,
) -> int:
    """Resolve worker policy without exceeding the effective CPU allocation."""

    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 0:
        raise ValueError("local_descriptor_workers must be a non-negative integer")
    allocation = _effective_local_cpu_allocation()
    if worker_cap is not None:
        if isinstance(worker_cap, bool) or not isinstance(worker_cap, int) or worker_cap < 1:
            raise ValueError("worker_cap must be a positive integer when provided")
        allocation = min(allocation, worker_cap)
    if requested > allocation:
        raise ValueError(
            "local_descriptor_workers="
            f"{requested} exceeds the effective CPU allocation ({allocation}); "
            "reduce the value or use 0 for automatic selection"
        )
    if candidate_count <= 1:
        return 1
    if requested == 1:
        return 1
    if requested == 0:
        return max(1, min(8, allocation, candidate_count))
    return min(requested, candidate_count)


def _validate_local_descriptor_inflight_bytes(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("max_local_descriptor_inflight_bytes must be a positive integer")
    return value


def _estimate_local_descriptor_task_bytes(
    candidate: Any,
    feature_width: int,
    config: LocalRepresentationConfig,
    species: Sequence[str],
) -> int:
    """Conservatively estimate parent, IPC, worker and result residency."""

    atom_count = len(candidate)
    payload = 4096 + 128 * atom_count
    arrays = getattr(candidate, "arrays", {})
    if isinstance(arrays, Mapping):
        payload += sum(int(np.asarray(array).nbytes) for array in arrays.values())
    payload += int(np.asarray(getattr(candidate, "positions", ()), dtype=np.float64).nbytes)
    payload += int(np.asarray(getattr(candidate, "cell", ()), dtype=np.float64).nbytes)
    result = atom_count * feature_width * 8
    atom_order = atom_count * 8
    neighbour_scratch = atom_count * max(1, atom_count) * 16
    feature_scratch = atom_count * feature_width * 16 + len(species) * config.radial_bins * 16
    return int(payload + result + atom_order + neighbour_scratch + feature_scratch)


def _local_process_context() -> BaseContext:
    """Use a safe process start method for the current platform."""

    method = "spawn" if os.name == "nt" else "forkserver"
    return mp.get_context(method)


class _LocalDescriptorProgress:
    """Parent-owned, bounded whole-percent progress reporting."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.started = time.perf_counter()
        self.last_percent = 0

    def completed(self, count: int) -> None:
        percent = min(100, count * 100 // self.total)
        if percent <= self.last_percent:
            return
        elapsed = max(0.0, time.perf_counter() - self.started)
        remaining = max(0.0, elapsed / count * (self.total - count))
        logger.info(
            "Local descriptor generation: %d/%d structures (%d%%); "
            "elapsed %.3fs; estimated remaining %.3fs",
            count,
            self.total,
            percent,
            elapsed,
            remaining,
        )
        self.last_percent = percent


def _compute_local_raw_descriptors(
    candidates: Sequence[Any],
    candidate_ids: Sequence[str],
    structure_ids: Sequence[str],
    species: tuple[str, ...],
    config: LocalRepresentationConfig,
    local_descriptor_workers: int,
    *,
    max_local_descriptor_inflight_bytes: int | None = None,
    resource_budget: ResourceBudgetService | None = None,
) -> tuple[np.ndarray, list[LocalEnvironmentRow]]:
    """Compute raw rows serially or with bounded candidate-level processes."""

    runtime_budget = resource_budget or build_resource_budget()
    worker_count = _resolve_local_descriptor_workers(
        local_descriptor_workers,
        len(candidates),
        runtime_budget.budget.worker_cap,
    )
    managed_budget = runtime_budget.remaining_managed_budget
    if managed_budget is None:
        raise ResourceCapacityError(
            "local descriptor generation requires known runtime memory headroom; "
            "provide [resources] memory_budget_bytes. "
            f"mode={runtime_budget.budget.execution_mode}, "
            f"memory_budget_override={runtime_budget.budget.memory_budget_bytes is not None}; "
            f"{runtime_budget.budget.snapshot.diagnostic_summary()}",
            operation="local descriptor generation",
        )
    if max_local_descriptor_inflight_bytes is None:
        inflight_limit = managed_budget
    else:
        inflight_limit = _validate_local_descriptor_inflight_bytes(
            max_local_descriptor_inflight_bytes
        )
    logger.info(
        "Local descriptor workers: %d (effective allocation %d)",
        worker_count,
        _effective_local_cpu_allocation(),
    )
    feature_width = _local_feature_width(config, species)
    total_atoms = sum(len(candidate) for candidate in candidates)
    base_workspace_bytes = int(total_atoms * feature_width * 8)
    lease = runtime_budget.acquire(
        "local descriptor generation",
        max(1, base_workspace_bytes),
    )
    try:
        raw = np.empty((total_atoms, feature_width), dtype=np.float64)
    except BaseException:
        lease.close()
        raise
    rows: list[LocalEnvironmentRow] = []
    base_workspace_bytes = int(raw.nbytes)
    task_estimates = [
        _estimate_local_descriptor_task_bytes(candidate, feature_width, config, species)
        for candidate in candidates
    ]
    if base_workspace_bytes >= inflight_limit:
        lease.close()
        raise ValueError(
            "local descriptor in-flight budget is smaller than the existing raw workspace: "
            f"workspace_bytes={base_workspace_bytes}, max_local_descriptor_inflight_bytes="
            f"{inflight_limit}"
        )
    progress = _LocalDescriptorProgress(len(candidates))
    pending: dict[Future[_LocalDescriptorResult], int] = {}
    pending_bytes: dict[Future[_LocalDescriptorResult], int] = {}
    ready: dict[int, _LocalDescriptorResult] = {}
    ready_bytes: dict[int, int] = {}
    next_to_submit = 0
    next_to_publish = 0
    completed = 0
    write_offset = 0
    inflight_bytes = base_workspace_bytes

    def task_for(index: int) -> _LocalDescriptorTask:
        return _LocalDescriptorTask(index, candidates[index], species, config)

    def publish(result: _LocalDescriptorResult, estimated_bytes: int) -> None:
        nonlocal inflight_bytes, write_offset, next_to_publish
        index = result.candidate_index
        candidate = candidates[index]
        expected_atoms = len(candidate)
        if result.descriptors.shape != (expected_atoms, feature_width):
            raise ValueError(
                "local descriptor worker returned an invalid shape for "
                f"candidate index {index} ({candidate_ids[index]})"
            )
        if len(result.atom_order) != expected_atoms:
            raise ValueError(
                "local descriptor worker returned an invalid atom order for "
                f"candidate index {index} ({candidate_ids[index]})"
            )
        raw[write_offset : write_offset + expected_atoms] = result.descriptors
        rows.extend(
            LocalEnvironmentRow(candidate_ids[index], structure_ids[index], atom_index)
            for atom_index in result.atom_order
        )
        write_offset += expected_atoms
        next_to_publish += 1
        inflight_bytes -= estimated_bytes
        lease.resize(max(1, inflight_bytes))

    def admission_error(index: int, estimated_bytes: int) -> ValueError:
        lease.close()
        return ValueError(
            "local descriptor task exceeds in-flight byte budget before launch: "
            f"candidate_index={index}, candidate_id={candidate_ids[index]}, "
            f"atom_count={len(candidates[index])}, estimated_bytes={estimated_bytes}, "
            f"existing_workspace_bytes={base_workspace_bytes}, "
            f"max_local_descriptor_inflight_bytes={inflight_limit}"
        )

    if worker_count == 1:
        try:
            for index in range(len(candidates)):
                estimate = task_estimates[index]
                if base_workspace_bytes + estimate > inflight_limit:
                    raise admission_error(index, estimate)
                inflight_bytes += estimate
                lease.resize(inflight_bytes)
                try:
                    result = _compute_local_descriptor_task(task_for(index))
                except BaseException as exc:
                    inflight_bytes -= estimate
                    lease.resize(max(1, inflight_bytes))
                    raise LocalDescriptorWorkerError(
                        "Local descriptor generation failed for candidate index "
                        f"{index} ({candidate_ids[index]})"
                    ) from exc
                publish(result, estimate)
                completed += 1
                progress.completed(completed)
        except BaseException:
            lease.close()
            raise
        lease.close()
        return raw, rows

    executor: ProcessPoolExecutor | None = None
    try:
        executor = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=_local_process_context(),
            initializer=_limit_local_worker_threads,
        )
        bound = max(worker_count, 2 * worker_count)

        def submit_available() -> None:
            nonlocal next_to_submit, inflight_bytes
            while next_to_submit < len(candidates) and len(pending) + len(ready) < bound:
                estimate = task_estimates[next_to_submit]
                if inflight_bytes + estimate > inflight_limit:
                    if not pending and not ready:
                        raise admission_error(next_to_submit, estimate)
                    break
                lease.resize(inflight_bytes + estimate)
                future = executor.submit(_compute_local_descriptor_task, task_for(next_to_submit))
                pending[future] = next_to_submit
                pending_bytes[future] = estimate
                inflight_bytes += estimate
                next_to_submit += 1

        submit_available()

        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in sorted(done, key=lambda item: pending[item]):
                index = pending.pop(future)
                estimate = pending_bytes.pop(future)
                try:
                    result = future.result()
                except BaseException as exc:
                    raise LocalDescriptorWorkerError(
                        "Local descriptor worker failed for candidate index "
                        f"{index} ({candidate_ids[index]})"
                    ) from exc
                if result.candidate_index != index:
                    raise LocalDescriptorWorkerError(
                        "Local descriptor worker returned candidate index "
                        f"{result.candidate_index}, expected {index} ({candidate_ids[index]})"
                    )
                ready[index] = result
                ready_bytes[index] = estimate
                completed += 1
                progress.completed(completed)
            while next_to_publish in ready:
                index = next_to_publish
                publish(ready.pop(index), ready_bytes.pop(index))
            submit_available()
    except BaseException:
        for future in pending:
            future.cancel()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        lease.close()
        raise
    else:
        if executor is not None:
            executor.shutdown(wait=True)
    lease.close()
    return raw, rows


def _canonicalize_eigenvectors(
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Make eigenspaces deterministic, including repeated eigenvalues."""

    result = np.asarray(eigenvectors, dtype=np.float64).copy()
    dimension = result.shape[1]
    start = 0
    while start < dimension:
        end = start + 1
        scale = max(1.0, abs(float(eigenvalues[start])))
        while (
            end < dimension
            and abs(float(eigenvalues[end] - eigenvalues[start])) <= tolerance * scale
        ):
            end += 1
        block = result[:, start:end]
        if end - start > 1:
            projection = block @ block.T
            basis: list[np.ndarray] = []
            for axis in range(dimension):
                vector = projection[:, axis].copy()
                for previous in basis:
                    vector -= np.dot(previous, vector) * previous
                norm = float(np.linalg.norm(vector))
                if norm > tolerance:
                    basis.append(vector / norm)
                if len(basis) == end - start:
                    break
            if len(basis) == end - start:
                result[:, start:end] = np.column_stack(basis)
        for column in range(start, end):
            vector = result[:, column]
            pivot = int(np.argmax(np.abs(vector)))
            if vector[pivot] < 0.0:
                result[:, column] *= -1.0
        start = end
    return result


def fit_deterministic_whitening(
    values: np.ndarray,
    *,
    tolerance: float = 1.0e-12,
    regularization: float = 1.0e-12,
    singular_policy: str = "regularize",
) -> WhiteningTransform:
    """Fit the declared full-pool whitening transform in float64."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("whitening input must be a non-empty 2D array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("whitening input must contain only finite values")
    tolerance = float(tolerance)
    regularization = float(regularization)
    policy = str(singular_policy).strip().lower()
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if not math.isfinite(regularization) or regularization < 0.0:
        raise ValueError("regularization must be finite and non-negative")
    if policy not in {"drop", "regularize", "reject"}:
        raise ValueError("singular_policy must be one of: drop, regularize, reject")
    mean = np.asarray(np.mean(matrix, axis=0, dtype=np.float64), dtype=np.float64)
    centred = matrix - mean
    covariance = (centred.T @ centred) / float(matrix.shape[0])
    covariance = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(-eigenvalues, kind="stable")
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.float64)
    eigenvectors = _canonicalize_eigenvectors(eigenvectors, eigenvalues, tolerance)
    singular = eigenvalues <= tolerance
    if policy == "reject" and np.any(singular):
        raise ValueError("whitening covariance contains singular directions")
    retained = tuple(
        int(index)
        for index, value in enumerate(eigenvalues)
        if policy != "drop" or value > tolerance
    )
    covariance_fingerprint = sha256_bytes(np.asarray(covariance, dtype="<f8").tobytes())
    eigenvalue_fingerprint = sha256_bytes(np.asarray(eigenvalues, dtype="<f8").tobytes())
    fingerprint = sha256_canonical_json(
        {
            "version": LOCAL_PREPROCESSING_VERSION,
            "mean": sha256_bytes(np.asarray(mean, dtype="<f8").tobytes()),
            "eigenvalues": eigenvalue_fingerprint,
            "eigenvectors": sha256_bytes(np.asarray(eigenvectors, dtype="<f8").tobytes()),
            "retained_indices": list(retained),
            "tolerance": tolerance,
            "regularization": regularization,
            "singular_policy": policy,
        }
    )
    return WhiteningTransform(
        mean=mean,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        retained_indices=retained,
        tolerance=tolerance,
        regularization=regularization,
        singular_policy=policy,
        covariance_fingerprint=covariance_fingerprint,
        eigenvalue_fingerprint=eigenvalue_fingerprint,
        fingerprint=fingerprint,
    )


def _local_identity(
    candidate_ids: Sequence[str],
    structure_ids: Sequence[str],
    content_fingerprints: Sequence[str],
    config: LocalRepresentationConfig,
    species: Sequence[str],
    transform: WhiteningTransform,
) -> dict[str, Any]:
    return {
        "schema_version": LOCAL_REPRESENTATION_SCHEMA_VERSION,
        "candidate_ids": list(candidate_ids),
        "structure_ids": list(structure_ids),
        "candidate_content_fingerprints": list(content_fingerprints),
        "local_environment_ordering_version": LOCAL_ENVIRONMENT_ORDERING_VERSION,
        "representation": config.to_dict(species=species),
        "magnetic_mode": config.magnetic_mode,
        "preprocessing": {
            "version": LOCAL_PREPROCESSING_VERSION,
            "tolerance": config.whitening_tolerance,
            "regularization": config.whitening_regularization,
            "singular_policy": config.whitening_singular_policy,
        },
        "transform_fingerprint": transform.fingerprint,
        "software_versions": _local_software_versions(),
    }


def _ase_version() -> str:
    try:
        import ase

        return str(ase.__version__)
    except AttributeError:
        return "unknown"


def _local_software_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "ase": _ase_version(),
    }


def build_local_environment_representation(
    candidates: Sequence[Any],
    *,
    config: LocalRepresentationConfig | Mapping[str, Any] | None = None,
    candidate_ids: Sequence[str] | None = None,
    structure_ids: Sequence[str] | None = None,
    local_descriptor_workers: int = 0,
    max_local_descriptor_inflight_bytes: int | None = None,
    resource_budget: ResourceBudgetService | None = None,
) -> LocalEnvironmentRepresentation:
    """Build deterministic local rows and fit whitening on the complete pool."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    settings = _coerce_local_representation_config(config)
    ordered_candidate_ids = _local_candidate_ids(candidates, candidate_ids)
    ordered_structure_ids = _local_structure_ids(candidates, structure_ids)
    if settings.magnetic_mode == "structural":
        validate_candidate_representation_identity(ordered_candidate_ids, ordered_structure_ids)
    content_fingerprints = tuple(
        _candidate_content_fingerprint(candidate, magnetic_mode=settings.magnetic_mode)
        for candidate in candidates
    )
    all_species = tuple(
        sorted(
            set(settings.species).union(
                symbol for candidate in candidates for symbol in candidate.get_chemical_symbols()
            )
        )
    )
    if not all_species:
        raise ValueError("candidates must contain at least one chemical species")
    raw, rows = _compute_local_raw_descriptors(
        candidates,
        ordered_candidate_ids,
        ordered_structure_ids,
        all_species,
        settings,
        local_descriptor_workers,
        max_local_descriptor_inflight_bytes=max_local_descriptor_inflight_bytes,
        resource_budget=resource_budget,
    )
    if not np.all(np.isfinite(raw)):
        raise ValueError("local representation contains non-finite values")
    logger.info("Local descriptor whitening started")
    transform = fit_deterministic_whitening(
        raw,
        tolerance=settings.whitening_tolerance,
        regularization=settings.whitening_regularization,
        singular_policy=settings.whitening_singular_policy,
    )
    whitened = transform.apply(raw)
    logger.info("Local descriptor whitening completed")
    identity = _local_identity(
        ordered_candidate_ids,
        ordered_structure_ids,
        content_fingerprints,
        settings,
        all_species,
        transform,
    )
    fingerprint = sha256_canonical_json(identity)
    return LocalEnvironmentRepresentation(
        raw_descriptors=raw,
        descriptors=whitened,
        rows=tuple(rows),
        candidate_ids=ordered_candidate_ids,
        structure_ids=ordered_structure_ids,
        config=settings,
        transform=transform,
        fingerprint=fingerprint,
        candidate_content_fingerprints=content_fingerprints,
    )


def compute_local_environment_representation(
    *args: Any,
    **kwargs: Any,
) -> LocalEnvironmentRepresentation:
    """Compatibility name for :func:`build_local_environment_representation`."""

    return build_local_environment_representation(*args, **kwargs)


compute_local_environment_descriptors = compute_local_environment_representation
fit_whitening_transform = fit_deterministic_whitening


def local_representation_cache_path(project_dir: Path) -> Path:
    return project_dir / "nep" / "datasets" / "local_representations.npz"


def local_representation_manifest_path(project_dir: Path) -> Path:
    return project_dir / "nep" / "datasets" / "local_representations.manifest.json"


def _write_local_representation_cache(
    cache_path: Path,
    manifest_path: Path,
    representation: LocalEnvironmentRepresentation,
    identity: dict[str, Any],
) -> None:
    arrays = {
        "raw_descriptors": np.asarray(representation.raw_descriptors, dtype=np.float64),
        "descriptors": np.asarray(representation.descriptors, dtype=np.float64),
        "mean": np.asarray(representation.transform.mean, dtype=np.float64),
        "eigenvalues": np.asarray(representation.transform.eigenvalues, dtype=np.float64),
        "eigenvectors": np.asarray(representation.transform.eigenvectors, dtype=np.float64),
        "retained_indices": np.asarray(representation.transform.retained_indices, dtype=np.int64),
    }
    rows_payload = [row.to_dict(index) for index, row in enumerate(representation.rows)]
    persisted = {
        "schema_version": LOCAL_REPRESENTATION_CACHE_SCHEMA_VERSION,
        "identity": identity,
        "transform": representation.transform.to_manifest(),
        "rows": rows_payload,
        "shape": {
            "raw": list(representation.raw_descriptors.shape),
            "descriptors": list(representation.descriptors.shape),
        },
        "candidate_ids": list(representation.candidate_ids),
        "structure_ids": list(representation.structure_ids),
        "artifact": {
            "format": "npz-stream-v1",
            "arrays": {
                name: {
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                    "sha256": _sha256_array(array),
                }
                for name, array in arrays.items()
            },
            "rows_sha256": sha256_canonical_json(rows_payload),
        },
        "representation_fingerprint": representation.fingerprint,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    staging_fd, staging_name = tempfile.mkstemp(
        prefix=f".{cache_path.name}.", suffix=".staging", dir=cache_path.parent
    )
    os.close(staging_fd)
    staging_path = Path(staging_name)
    cache_backup = cache_path.with_name(f".{cache_path.name}.previous")
    manifest_backup = manifest_path.with_name(f".{manifest_path.name}.previous")
    cache_had_previous = cache_path.is_file()
    manifest_had_previous = manifest_path.is_file()
    cache_backup.unlink(missing_ok=True)
    manifest_backup.unlink(missing_ok=True)
    try:
        artifact_sha256 = atomic_write_stream(
            staging_path,
            lambda handle: np.savez(handle, **arrays),
        )
        persisted_manifest = dict(identity)
        persisted_manifest.update(persisted)
        persisted_manifest["artifact_sha256"] = artifact_sha256
        if cache_had_previous:
            os.replace(cache_path, cache_backup)
        if manifest_had_previous:
            os.replace(manifest_path, manifest_backup)
        os.replace(staging_path, cache_path)
        write_json(manifest_path, persisted_manifest, indent=None)
        cache_backup.unlink(missing_ok=True)
        manifest_backup.unlink(missing_ok=True)
    except BaseException:
        staging_path.unlink(missing_ok=True)
        if cache_backup.is_file():
            cache_path.unlink(missing_ok=True)
            os.replace(cache_backup, cache_path)
        elif not cache_had_previous:
            cache_path.unlink(missing_ok=True)
        if manifest_backup.is_file():
            manifest_path.unlink(missing_ok=True)
            os.replace(manifest_backup, manifest_path)
        elif not manifest_had_previous:
            manifest_path.unlink(missing_ok=True)
        raise


def _load_local_representation_cache(
    cache_path: Path,
    manifest_path: Path,
    expected_identity_without_transform: dict[str, Any],
) -> LocalEnvironmentRepresentation | None:
    if not cache_path.is_file() or not manifest_path.is_file():
        return None
    try:
        manifest = read_json_object(manifest_path, error_type=StateError)
        if manifest.get("schema_version") != LOCAL_REPRESENTATION_CACHE_SCHEMA_VERSION:
            return None
        identity = manifest.get("identity")
        if not isinstance(identity, dict):
            return None
        expected = dict(expected_identity_without_transform)
        expected_transform = identity.pop("transform_fingerprint", None)
        if identity != expected:
            return None
        if not isinstance(expected_transform, str) or not expected_transform:
            return None
        if manifest.get("artifact_sha256") != sha256_file(cache_path, required=True):
            return None
        artifact_manifest = manifest.get("artifact")
        if (
            not isinstance(artifact_manifest, dict)
            or artifact_manifest.get("format") != "npz-stream-v1"
            or not isinstance(artifact_manifest.get("arrays"), dict)
        ):
            return None
        with np.load(cache_path, allow_pickle=False) as archive:
            raw = np.asarray(archive["raw_descriptors"], dtype=np.float64)
            descriptors = np.asarray(archive["descriptors"], dtype=np.float64)
            mean = np.asarray(archive["mean"], dtype=np.float64)
            eigenvalues = np.asarray(archive["eigenvalues"], dtype=np.float64)
            eigenvectors = np.asarray(archive["eigenvectors"], dtype=np.float64)
            retained_array = np.asarray(archive["retained_indices"], dtype=np.int64)
            retained = tuple(int(value) for value in retained_array.tolist())
            loaded_arrays = {
                "raw_descriptors": raw,
                "descriptors": descriptors,
                "mean": mean,
                "eigenvalues": eigenvalues,
                "eigenvectors": eigenvectors,
                "retained_indices": retained_array,
            }
        for name, array in loaded_arrays.items():
            record = artifact_manifest["arrays"].get(name)
            if (
                not isinstance(record, dict)
                or record.get("dtype") != array.dtype.str
                or record.get("shape") != list(array.shape)
                or record.get("sha256") != _sha256_array(array)
            ):
                return None
        if raw.ndim != 2 or descriptors.ndim != 2 or raw.shape[0] != descriptors.shape[0]:
            return None
        shape = manifest.get("shape")
        if (
            not isinstance(shape, dict)
            or shape.get("raw") != list(raw.shape)
            or shape.get("descriptors") != list(descriptors.shape)
        ):
            return None
        if not all(
            np.all(np.isfinite(array))
            for array in (raw, descriptors, mean, eigenvalues, eigenvectors)
        ):
            return None
        transform_manifest = manifest.get("transform")
        if not isinstance(transform_manifest, dict):
            return None
        if transform_manifest.get("version") != LOCAL_PREPROCESSING_VERSION:
            return None
        transform = fit_deterministic_whitening(
            raw,
            tolerance=float(transform_manifest["tolerance"]),
            regularization=float(transform_manifest["regularization"]),
            singular_policy=str(transform_manifest["singular_policy"]),
        )
        if transform.fingerprint != expected_transform:
            return None
        if transform_manifest.get("fingerprint") != transform.fingerprint:
            return None
        if retained != transform.retained_indices:
            return None
        if (
            not np.array_equal(transform.mean, mean)
            or not np.array_equal(transform.eigenvalues, eigenvalues)
            or not np.array_equal(transform.eigenvectors, eigenvectors)
        ):
            return None
        if not np.array_equal(transform.apply(raw), descriptors):
            return None
        rows_payload = manifest.get("rows")
        if not isinstance(rows_payload, list) or len(rows_payload) != raw.shape[0]:
            return None
        if artifact_manifest.get("rows_sha256") != sha256_canonical_json(rows_payload):
            return None
        rows = tuple(
            LocalEnvironmentRow(
                str(item["candidate_id"]),
                str(item["structure_id"]),
                int(item["atom_index"]),
            )
            for item in rows_payload
        )
        if any(
            row.candidate_id not in expected_identity_without_transform["candidate_ids"]
            for row in rows
        ):
            return None
        candidate_ids = tuple(str(value) for value in manifest["candidate_ids"])
        structure_ids = tuple(str(value) for value in manifest["structure_ids"])
        if candidate_ids != tuple(expected_identity_without_transform["candidate_ids"]):
            return None
        if structure_ids != tuple(expected_identity_without_transform["structure_ids"]):
            return None
        fingerprint = str(manifest["representation_fingerprint"])
        expected_fingerprint = sha256_canonical_json(
            {**expected_identity_without_transform, "transform_fingerprint": expected_transform}
        )
        if fingerprint != expected_fingerprint:
            return None
        representation_parameters = dict(expected_identity_without_transform["representation"])
        representation_parameters.pop("backend", None)
        return LocalEnvironmentRepresentation(
            raw_descriptors=raw,
            descriptors=descriptors,
            rows=rows,
            candidate_ids=candidate_ids,
            structure_ids=structure_ids,
            config=LocalRepresentationConfig(**representation_parameters),
            transform=transform,
            fingerprint=fingerprint,
            candidate_content_fingerprints=tuple(
                expected_identity_without_transform["candidate_content_fingerprints"]
            ),
        )
    except (KeyError, OSError, StateError, TypeError, ValueError, EOFError):
        return None


def load_or_calculate_local_representations(
    project_dir: Path,
    candidates: Sequence[Any],
    *,
    config: LocalRepresentationConfig | Mapping[str, Any] | None = None,
    candidate_ids: Sequence[str] | None = None,
    structure_ids: Sequence[str] | None = None,
    local_descriptor_workers: int = 0,
    max_local_descriptor_inflight_bytes: int | None = None,
    resource_budget: ResourceBudgetService | None = None,
) -> LocalEnvironmentRepresentation:
    """Load exact-identity local rows or calculate and publish them."""

    settings = _coerce_local_representation_config(config)
    runtime_budget = resource_budget or build_resource_budget()
    _resolve_local_descriptor_workers(
        local_descriptor_workers,
        len(candidates),
        runtime_budget.budget.worker_cap,
    )
    if max_local_descriptor_inflight_bytes is not None:
        _validate_local_descriptor_inflight_bytes(max_local_descriptor_inflight_bytes)
    ordered_candidate_ids = _local_candidate_ids(candidates, candidate_ids)
    ordered_structure_ids = _local_structure_ids(candidates, structure_ids)
    if settings.magnetic_mode == "structural":
        validate_candidate_representation_identity(ordered_candidate_ids, ordered_structure_ids)
    species = tuple(
        sorted(
            set(settings.species).union(
                symbol for candidate in candidates for symbol in candidate.get_chemical_symbols()
            )
        )
    )
    content = tuple(
        _candidate_content_fingerprint(candidate, magnetic_mode=settings.magnetic_mode)
        for candidate in candidates
    )
    base_identity = {
        "schema_version": LOCAL_REPRESENTATION_SCHEMA_VERSION,
        "candidate_ids": list(ordered_candidate_ids),
        "structure_ids": list(ordered_structure_ids),
        "candidate_content_fingerprints": list(content),
        "local_environment_ordering_version": LOCAL_ENVIRONMENT_ORDERING_VERSION,
        "representation": settings.to_dict(species=species),
        "magnetic_mode": settings.magnetic_mode,
        "preprocessing": {
            "version": LOCAL_PREPROCESSING_VERSION,
            "tolerance": settings.whitening_tolerance,
            "regularization": settings.whitening_regularization,
            "singular_policy": settings.whitening_singular_policy,
        },
        "software_versions": _local_software_versions(),
    }
    cache_path = local_representation_cache_path(project_dir)
    manifest_path = local_representation_manifest_path(project_dir)
    cached = _load_local_representation_cache(cache_path, manifest_path, base_identity)
    if cached is not None:
        return cached
    result = build_local_environment_representation(
        candidates,
        config=settings,
        candidate_ids=ordered_candidate_ids,
        structure_ids=ordered_structure_ids,
        local_descriptor_workers=local_descriptor_workers,
        max_local_descriptor_inflight_bytes=max_local_descriptor_inflight_bytes,
        resource_budget=runtime_budget,
    )
    identity = _local_identity(
        ordered_candidate_ids,
        ordered_structure_ids,
        content,
        settings,
        species,
        result.transform,
    )
    _write_local_representation_cache(cache_path, manifest_path, result, identity)
    return result


__all__ = [
    "DESCRIPTOR_CACHE_SCHEMA_VERSION",
    "LOCAL_ENVIRONMENT_ORDERING_VERSION",
    "LOCAL_PREPROCESSING_VERSION",
    "LOCAL_REPRESENTATION_BACKEND",
    "LOCAL_REPRESENTATION_CACHE_SCHEMA_VERSION",
    "LOCAL_REPRESENTATION_SCHEMA_VERSION",
    "LocalEnvironmentRepresentation",
    "LocalEnvironmentRow",
    "LocalDescriptorWorkerError",
    "LocalRepresentationConfig",
    "WhiteningTransform",
    "build_local_environment_representation",
    "compute_descriptors_batched",
    "compute_local_environment_descriptors",
    "compute_local_environment_representation",
    "compute_structure_descriptors",
    "descriptor_cache_path",
    "descriptor_manifest_path",
    "fit_deterministic_whitening",
    "fit_whitening_transform",
    "load_or_calculate_local_representations",
    "load_or_calculate_representations",
    "local_representation_cache_path",
    "local_representation_manifest_path",
    "structure_identity",
    "validate_candidate_representation_identity",
]
