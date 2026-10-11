"""Optional exact, certified CuPy tiled k-nearest-neighbour backend.

CuPy is deliberately imported only after this backend is selected.  The
implementation uses the GPU to screen exhaustive float64 tiles and uses the
canonical CPU distance operation to certify one globally merged result per
source.  A boundary whose proof is inconclusive is rescanned in full on the
CPU; it is never accepted as an approximate top-k result.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from nepflow.io.hashing import sha256_canonical_json
from nepflow.resources.budget import ResourceBudgetService, ResourceCapacityError

from .checkpointing import IncrementalCheckpointStore, VersionedCheckpointStore
from .gpu_devices import (
    contiguous_gpu_ranges,
    normalize_gpu_devices,
    validate_visible_gpu_devices,
)
from .gpu_support import GPUCapacityError, GPUCertificationError, GPUDeviceValidationError
from .models import (
    GPU_NEIGHBOUR_BACKEND_ID,
    GPU_NEIGHBOUR_BACKEND_VERSION,
    NEIGHBOUR_DISTINCT_POLICY,
    NEIGHBOUR_METRIC,
)
from .neighbours import (
    DEFAULT_NEIGHBOUR_CACHE_BYTES,
    ExactNeighbourResult,
    RadiusQueryCapacityError,
    TiledCPUNeighbourIndex,
    _array_fingerprint,
    _validate_chunk_size,
    _validate_descriptors,
    _validate_k,
    _validate_row_ids,
)


def _gamma_bound(dimension: int) -> float:
    """Return an outward-rounded relative FP64 operation bound.

    The GPU tile computes one subtraction, square, and a floating-point sum per
    coordinate, followed by a square root.  The allowance is a conservative
    first-order ``gamma_n`` bound for the reduction and selection path.  It is
    used only when all intermediates are in the normal finite FP64 range; the
    underflow/overflow regime fails closed to exhaustive CPU verification.
    ``argpartition`` only orders the already-rounded non-negative scores and
    adds no arithmetic error to this distance interval.
    """

    if dimension < 1:
        raise GPUCertificationError("GPU kNN requires a positive descriptor dimension")
    unit_roundoff = np.finfo(np.float64).eps / 2.0
    operation_product = float((3 * int(dimension) + 8) * unit_roundoff)
    if not np.isfinite(operation_product) or operation_product >= 1.0:
        raise GPUCertificationError("GPU FP64 certification bound is not representable")
    relative = operation_product / (1.0 - operation_product)
    if not np.isfinite(relative) or relative <= 0.0:
        raise GPUCertificationError("GPU FP64 certification bound is not finite")
    return float(np.nextafter(relative, np.inf))


def _canonical_distances(query: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Compute the scientific CPU float64 distance primitive."""

    if targets.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    with np.errstate(over="raise", invalid="raise"):
        try:
            delta = targets - query
            distances = np.sqrt(np.sum(delta * delta, axis=1, dtype=np.float64))
        except FloatingPointError as exc:
            raise GPUCertificationError(
                "canonical CPU GPU-kNN verification overflowed or became invalid"
            ) from exc
    if not np.all(np.isfinite(distances)):
        raise GPUCertificationError(
            "canonical CPU GPU-kNN verification produced a non-finite distance"
        )
    return np.asarray(distances, dtype=np.float64)


class ExactTiledGPUNeighbourIndex(TiledCPUNeighbourIndex):
    """Exhaustive one-device GPU kNN with conservative exact certification."""

    backend = GPU_NEIGHBOUR_BACKEND_ID
    backend_version = GPU_NEIGHBOUR_BACKEND_VERSION
    metric = NEIGHBOUR_METRIC
    distinct_location_policy = NEIGHBOUR_DISTINCT_POLICY

    def __init__(
        self,
        descriptors: np.ndarray,
        *,
        device: int = 0,
        gpu_devices: Sequence[int] | str | int | None = None,
        gpu_tile_size: int = 128,
        max_device_bytes: int | None = None,
        tile_size: int | None = None,
        max_index_bytes: int | None = None,
        max_radius_query_bytes: int | None = None,
        max_neighbour_cache_bytes: int | None = DEFAULT_NEIGHBOUR_CACHE_BYTES,
        resource_budget: ResourceBudgetService | None = None,
    ) -> None:
        if isinstance(device, bool) or not isinstance(device, (int, np.integer)):
            raise ValueError("gpu_device must be a non-negative integer")
        if int(device) < 0:
            raise ValueError("gpu_device must be a non-negative integer")
        if (
            isinstance(gpu_tile_size, bool)
            or not isinstance(gpu_tile_size, (int, np.integer))
            or int(gpu_tile_size) < 1
        ):
            raise ValueError("gpu_tile_size must be a positive integer")
        self.gpu_devices = normalize_gpu_devices(gpu_devices, gpu_device=int(device))
        self.device = int(self.gpu_devices[0])
        self.requested_tile_size = int(gpu_tile_size)
        self._cp = self._load_cupy()
        self._validate_device()
        free_bytes, total_bytes = self._device_memory_info()
        if free_bytes < 1 or total_bytes < 1:
            raise GPUDeviceValidationError(f"CUDA device {self.device} reported no usable memory")
        if max_device_bytes is None:
            self.max_device_bytes = max(1, int(free_bytes * 0.80))
        else:
            if (
                isinstance(max_device_bytes, bool)
                or not isinstance(max_device_bytes, (int, np.integer))
                or int(max_device_bytes) < 1
            ):
                raise ValueError("max_gpu_neighbour_bytes must be positive when provided")
            self.max_device_bytes = min(int(max_device_bytes), int(free_bytes))
        self.device_total_bytes = int(total_bytes)
        self.device_free_bytes = int(free_bytes)
        values = _validate_descriptors(descriptors)
        self.certification_relative_bound = _gamma_bound(int(values.shape[1]))
        float_info = np.finfo(np.float64)
        max_safe_component = float(np.sqrt(float_info.max / (4.0 * int(values.shape[1]))))
        min_safe_component = float(np.sqrt(float_info.tiny) / (float_info.eps / 2.0))
        magnitudes = np.abs(values)
        self._gpu_fp64_certification_safe = bool(
            np.all(magnitudes <= max_safe_component)
            and not np.any((magnitudes > 0.0) & (magnitudes < min_safe_component))
        )
        super().__init__(
            values,
            tile_size=(self.requested_tile_size if tile_size is None else tile_size),
            max_index_bytes=max_index_bytes,
            max_radius_query_bytes=max_radius_query_bytes,
            max_neighbour_cache_bytes=max_neighbour_cache_bytes,
            resource_budget=resource_budget,
        )
        # Device selection, source/target tiling, and checkpoint chunking are
        # operational choices.  Keep them out of the scientific backend
        # fingerprint so an exact result can be resumed or compared across
        # those choices.
        self.fingerprint = sha256_canonical_json(
            {
                "backend": self.backend,
                "version": self.backend_version,
                "metric": self.metric,
                "distinct_location_policy": self.distinct_location_policy,
                "tie_policy": "distance-then-earliest-original-representative-v1",
                "radius_policy": "strict-float64-euclidean-less-than-v1",
                "locations": _array_fingerprint(self.unique_locations),
                "row_to_location": _array_fingerprint(self.row_to_location),
            }
        )
        source_count, target_count, required = self._tile_shape(1)
        del source_count, target_count
        if required > self.max_device_bytes:
            raise GPUCapacityError(
                "exact GPU kNN cannot fit one exhaustive source/target tile: "
                f"requested_bytes={required}, available_bytes={self.max_device_bytes}, "
                f"dimension={self.descriptors.shape[1]}",
                operation="exact tiled GPU kNN",
                requested_bytes=required,
                available_bytes=self.max_device_bytes,
            )
        self.source_tiles = 0
        self.target_tiles = 0
        self.fp64_evaluations = 0
        self.h2d_bytes = 0
        self.d2h_bytes = 0
        self.device_workspace_peak_bytes = 0
        self.host_workspace_peak_bytes = 0
        self.gpu_seconds = 0.0
        self.gpu_operation_seconds = 0.0
        self.gpu_h2d_seconds = 0.0
        self.gpu_d2h_seconds = 0.0
        self.gpu_distance_reduction_seconds = 0.0
        self.gpu_synchronization_seconds = 0.0
        self.gpu_launch_allocation_seconds = 0.0
        self.gpu_tile_wall_seconds = 0.0
        self.reference_uploads = 0
        self.reference_shards = 0
        self.resident_reference_bytes = 0
        self.global_candidate_state_bytes = 0
        self.certification_seconds = 0.0
        self.cpu_verification_seconds = 0.0
        self.certification_rescans = 0
        self.ambiguous_boundary_count = 0
        self.certified_tiles = 0
        self.final_certifications = 0
        self.per_device_metrics: dict[int, dict[str, Any]] = {}

    def _load_cupy(self) -> Any:
        try:
            import cupy as cp  # type: ignore[import-not-found]
        except ImportError as exc:
            raise GPUDeviceValidationError(
                "backend='exact_tiled_gpu_knn' requires optional CuPy"
            ) from exc
        return cp

    def _validate_device(self) -> None:
        try:
            validate_visible_gpu_devices(self._cp, self.gpu_devices)
        except Exception as exc:  # pragma: no cover - CUDA runtime dependent
            raise GPUDeviceValidationError(
                f"requested gpu_devices={self.gpu_devices} (gpu_device={self.device}) "
                f"are not visible: {exc}"
            ) from exc

    def _device_memory_info(self) -> tuple[int, int]:
        try:
            with self._cp.cuda.Device(self.device):
                free_bytes, total_bytes = self._cp.cuda.runtime.memGetInfo()
        except Exception as exc:  # pragma: no cover - CUDA runtime dependent
            raise GPUDeviceValidationError(
                f"unable to inspect CUDA device {self.device} memory: {exc}"
            ) from exc
        return int(free_bytes), int(total_bytes)

    def _tile_shape(
        self,
        source_count: int,
        *,
        resident_reference_bytes: int = 0,
        global_state_bytes: int = 0,
    ) -> tuple[int, int, int]:
        dimension = int(self.descriptors.shape[1])
        sources = max(1, min(int(source_count), self.requested_tile_size))
        targets = min(self.requested_tile_size, int(self.unique_locations.shape[0]))

        def required(source_rows: int, target_rows: int) -> int:
            # Vectorised subtraction/square storage, distance/reduction
            # buffers, descriptor transfers, persistent candidate state, and
            # any resident reference pool or shard. This is conservative so a
            # device allocation failure still fails explicitly.
            return int(
                global_state_bytes
                + resident_reference_bytes
                + 32 * source_rows * target_rows * dimension
                + 128 * source_rows * target_rows
                + 16 * (source_rows + target_rows) * dimension
            )

        while required(sources, targets) > self.max_device_bytes:
            if targets > 1:
                targets = max(1, targets // 2)
            elif sources > 1:
                sources = max(1, sources // 2)
            else:
                needed = required(sources, targets)
                raise GPUCapacityError(
                    "exact GPU kNN cannot fit one source/target tile: "
                    f"requested_bytes={needed}, available_bytes={self.max_device_bytes}",
                    operation="exact tiled GPU kNN",
                    requested_bytes=needed,
                    available_bytes=self.max_device_bytes,
                )
        return sources, targets, required(sources, targets)

    def _synchronize(self) -> float:
        """Synchronize the selected stream and return its wall-clock cost."""

        stream_getter = getattr(self._cp.cuda, "get_current_stream", None)
        if stream_getter is None:
            return 0.0
        stream = stream_getter()
        synchronize = getattr(stream, "synchronize", None)
        if synchronize is None:
            return 0.0
        started = time.perf_counter()
        synchronize()
        elapsed = max(0.0, time.perf_counter() - started)
        self.gpu_synchronization_seconds += elapsed
        return elapsed

    def _upload_descriptor_block(self, values: np.ndarray, operation: str) -> Any:
        """Upload one bounded descriptor block and record its transfer cost."""

        host_values = np.ascontiguousarray(values, dtype=np.float64)
        started = time.perf_counter()
        try:
            with self._cp.cuda.Device(self.device):
                gpu_values = self._cp.asarray(host_values, dtype=self._cp.float64)
                synchronization = self._synchronize()
        except Exception as exc:
            if isinstance(exc, (GPUCapacityError, GPUCertificationError)):
                raise
            if isinstance(exc, MemoryError) or "out of memory" in str(exc).lower():
                raise GPUCapacityError(
                    f"{operation} allocation exceeded the configured device capacity: "
                    f"device={self.device}, max_gpu_neighbour_bytes={self.max_device_bytes}",
                    operation=operation,
                    requested_bytes=int(host_values.nbytes),
                    available_bytes=self.max_device_bytes,
                ) from exc
            raise GPUCertificationError(
                f"{operation} upload failed on gpu_device={self.device}: {exc}"
            ) from exc
        self.h2d_bytes += int(host_values.nbytes)
        h2d_elapsed = max(0.0, time.perf_counter() - started)
        self.gpu_h2d_seconds += h2d_elapsed
        self.gpu_launch_allocation_seconds += max(
            0.0,
            h2d_elapsed - synchronization,
        )
        if operation in {
            "exact tiled GPU kNN reference pool",
            "exact tiled GPU kNN reference shard",
        }:
            self.reference_uploads += 1
        return gpu_values

    def _upload_source_batch(self, source: np.ndarray) -> Any:
        """Upload one source batch when it is not covered by a resident pool."""

        return self._upload_descriptor_block(source, "exact tiled GPU kNN source batch")

    def _upload_reference_shard(self, reference: np.ndarray) -> Any:
        return self._upload_descriptor_block(reference, "exact tiled GPU kNN reference shard")

    def _gpu_top_candidates(
        self,
        gpu_sources: Any,
        gpu_targets: Any,
        keep_count: int,
        *,
        target_offset: int,
        target_count: int,
    ) -> tuple[Any, Any, Any]:
        """Return GPU-local candidates and an outward-bound omission witness."""

        cp = self._cp
        source_count = int(gpu_sources.shape[0])
        dimension = int(gpu_sources.shape[1])
        reserved = int(
            32 * source_count * target_count * dimension
            + 128 * source_count * target_count
            + 16 * (source_count + target_count) * dimension
        )
        operation_started = time.perf_counter()
        compute_elapsed = 0.0
        try:
            with cp.cuda.Device(self.device):
                compute_started = time.perf_counter()
                delta = gpu_sources[:, None, :] - gpu_targets[None, :, :]
                delta *= delta
                distance_squared = cp.sum(delta, axis=2, dtype=cp.float64)
                isfinite = getattr(cp, "isfinite", None)
                all_values_finite = getattr(cp, "all", None)
                partition = cp.argpartition(
                    distance_squared,
                    kth=int(keep_count - 1),
                    axis=1,
                )[:, :keep_count]
                selected_squared = cp.take_along_axis(distance_squared, partition, axis=1)
                selected_distances = cp.sqrt(selected_squared)
                self._synchronize()
                compute_elapsed = max(0.0, time.perf_counter() - compute_started)
                self.gpu_distance_reduction_seconds += compute_elapsed
                selected_indices = partition + int(target_offset)
                local_bound = cp.full((source_count,), np.inf, dtype=cp.float64)
                if target_count > keep_count:
                    local_bound = cp.max(selected_distances, axis=1)
                if isfinite is not None and all_values_finite is not None:
                    # Keep the uncertainty witness on the device.  Calling
                    # bool() on a CuPy reduction here would force a scalar
                    # D2H synchronization for every reference tile.
                    finite_rows = all_values_finite(isfinite(distance_squared), axis=1)
                    local_bound = cp.where(finite_rows, local_bound, cp.nan)
        except Exception as exc:
            if isinstance(exc, (GPUCapacityError, GPUCertificationError)):
                raise
            if isinstance(exc, MemoryError) or "out of memory" in str(exc).lower():
                raise GPUCapacityError(
                    "GPU kNN tile allocation exceeded the configured device capacity: "
                    f"device={self.device}, max_gpu_neighbour_bytes={self.max_device_bytes}",
                    operation="exact tiled GPU kNN",
                    requested_bytes=reserved,
                    available_bytes=self.max_device_bytes,
                ) from exc
            raise GPUCertificationError(
                f"GPU kNN tile evaluation failed on gpu_device={self.device}: {exc}"
            ) from exc
        operation_elapsed = max(0.0, time.perf_counter() - operation_started)
        self.gpu_operation_seconds += operation_elapsed
        self.gpu_seconds += compute_elapsed
        self.gpu_launch_allocation_seconds += max(
            0.0,
            operation_elapsed - compute_elapsed,
        )
        self.fp64_evaluations += int(source_count * target_count)
        self.device_workspace_peak_bytes = max(
            self.device_workspace_peak_bytes,
            self.global_candidate_state_bytes + self.resident_reference_bytes + reserved,
        )
        return selected_indices, selected_distances, local_bound

    def _merge_gpu_candidates(
        self,
        global_indices: Any,
        global_distances: Any,
        omission_bounds: Any,
        merge_bounds: Any,
        source_start: int,
        local_indices: Any,
        local_distances: Any,
        local_bound: Any,
        existing_count: int,
        keep_count: int,
    ) -> None:
        """Merge one reference tile into persistent GPU top-k state."""

        cp = self._cp
        source_stop = source_start + int(local_indices.shape[0])
        omission_bounds[source_start:source_stop] = cp.minimum(
            omission_bounds[source_start:source_stop], local_bound
        )
        local_count = int(local_indices.shape[1])
        if existing_count == 0:
            global_indices[source_start:source_stop, :local_count] = local_indices
            global_distances[source_start:source_stop, :local_count] = local_distances
            return

        combined_indices = cp.concatenate(
            (
                global_indices[source_start:source_stop, :existing_count],
                local_indices,
            ),
            axis=1,
        )
        combined_distances = cp.concatenate(
            (
                global_distances[source_start:source_stop, :existing_count],
                local_distances,
            ),
            axis=1,
        )
        combined_count = int(combined_indices.shape[1])
        if combined_count <= keep_count:
            selected_indices = combined_indices
            selected_distances = combined_distances
        else:
            partition = cp.argpartition(
                combined_distances,
                kth=int(keep_count - 1),
                axis=1,
            )[:, :keep_count]
            selected_indices = cp.take_along_axis(combined_indices, partition, axis=1)
            selected_distances = cp.take_along_axis(combined_distances, partition, axis=1)
            merge_bounds[source_start:source_stop] = cp.minimum(
                merge_bounds[source_start:source_stop],
                cp.max(selected_distances, axis=1),
            )
        selected_count = int(selected_indices.shape[1])
        global_indices[source_start:source_stop, :selected_count] = selected_indices
        global_distances[source_start:source_stop, :selected_count] = selected_distances

    def _exact_rescan(
        self, query: np.ndarray, source_location: int, k: int
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute one source row exhaustively with the canonical CPU primitive."""

        started = time.perf_counter()
        try:
            distances = self._distance_values(query, self.unique_locations)
        except RadiusQueryCapacityError:
            raise
        except (FloatingPointError, ValueError) as exc:
            raise GPUCertificationError("canonical CPU GPU-kNN verification failed") from exc
        distances[int(source_location)] = np.inf
        sort_workspace = int(distances.nbytes + 8 * distances.size + 4096)
        self.host_workspace_peak_bytes = max(self.host_workspace_peak_bytes, sort_workspace)
        with self.resource_budget.acquire(
            "exact tiled GPU kNN CPU rescan ordering", max(1, sort_workspace)
        ):
            order = np.lexsort((self.location_representatives, distances))[: int(k)]
        elapsed = max(0.0, time.perf_counter() - started)
        return order.astype(np.int64, copy=False), distances[order], elapsed

    def _certify_final(
        self,
        query: np.ndarray,
        source_location: int,
        reported_locations: np.ndarray,
        reported_distances: np.ndarray,
        omission_bound: float,
        merge_bound: float,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Certify the globally merged GPU result or rescan one source exactly."""

        self.final_certifications += 1
        started = time.perf_counter()
        cpu_verification = 0.0

        def rescan() -> tuple[np.ndarray, np.ndarray]:
            nonlocal cpu_verification
            self.certification_rescans += 1
            locations, distances, elapsed = self._exact_rescan(query, source_location, k)
            cpu_verification += elapsed
            return locations, distances

        if not self._gpu_fp64_certification_safe:
            self.ambiguous_boundary_count += 1
            result = rescan()
            self.certification_seconds += max(0.0, time.perf_counter() - started - cpu_verification)
            self.cpu_verification_seconds += cpu_verification
            return result

        valid = (
            reported_locations.ndim == 1
            and reported_distances.ndim == 1
            and reported_locations.size == reported_distances.size
            and reported_locations.size >= int(k)
            and np.count_nonzero(reported_locations != int(source_location)) >= int(k)
            and np.all(reported_locations >= 0)
            and np.all(reported_locations < self.unique_locations.shape[0])
            and np.unique(reported_locations).size == reported_locations.size
            and int(source_location) in set(int(value) for value in reported_locations)
            and np.all(np.isfinite(reported_distances))
        )
        if not valid:
            result = rescan()
            self.certification_seconds += max(0.0, time.perf_counter() - started - cpu_verification)
            self.cpu_verification_seconds += cpu_verification
            return result

        canonical_started = time.perf_counter()
        try:
            canonical = self._distance_values(query, self.unique_locations[reported_locations])
        except RadiusQueryCapacityError:
            raise
        except (FloatingPointError, ValueError) as exc:
            raise GPUCertificationError("canonical CPU GPU-kNN verification failed") from exc
        cpu_verification += max(0.0, time.perf_counter() - canonical_started)
        order = np.lexsort(
            (
                self.location_representatives[reported_locations],
                canonical,
            )
        )
        reported_locations = reported_locations[order]
        canonical = canonical[order]
        reported_distances = reported_distances[order]
        nonself = reported_locations != int(source_location)
        certified_locations = reported_locations[nonself]
        certified_distances = canonical[nonself]
        reported_boundary = (
            float(certified_distances[int(k) - 1])
            if certified_distances.size >= int(k)
            else float(np.inf)
        )
        omitted_bound = min(float(omission_bound), float(merge_bound))
        omitted = certified_locations.size < int(self.unique_locations.shape[0] - 1)
        relative = self.certification_relative_bound
        lower = float(omitted_bound / (1.0 + relative))
        upper = float(reported_boundary * (1.0 + relative))
        # The gamma bound covers the GPU arithmetic.  The repeated adjacent
        # representable steps also cover the host-side division/product used
        # to turn that bound into an interval; this keeps the proof outward
        # rounded instead of relying on a nominal comparison tolerance.
        for _ in range(4):
            lower = float(np.nextafter(lower, 0.0))
            upper = float(np.nextafter(upper, np.inf))
        certifiable = bool(
            not omitted or (np.isfinite(lower) and np.isfinite(upper) and lower > upper)
        )
        if not certifiable:
            self.ambiguous_boundary_count += 1
            result = rescan()
        else:
            self.certified_tiles += 1
            result = (
                certified_locations[: int(k)],
                certified_distances[: int(k)],
            )
        self.certification_seconds += max(0.0, time.perf_counter() - started - cpu_verification)
        self.cpu_verification_seconds += cpu_verification
        return result

    def _compute_location_neighbours(
        self,
        k: int,
        *,
        chunk_size: int = 1024,
        source_range: tuple[int, int] | None = None,
        checkpoint_store: VersionedCheckpointStore | IncrementalCheckpointStore | None = None,
        checkpoint_name: str = "knn",
        checkpoint_identity: dict[str, Any] | None = None,
        checkpoint_every_locations: int = 1,
        resume: bool = True,
        preserve_checkpoint: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        order_k = _validate_k(k)
        block = _validate_chunk_size(chunk_size)
        if len(self.gpu_devices) > 1 and source_range is None:
            return self._compute_multi_location_neighbours(
                order_k,
                chunk_size=block,
                checkpoint_store=checkpoint_store,
                checkpoint_name=checkpoint_name,
                checkpoint_identity=checkpoint_identity,
                checkpoint_every_locations=checkpoint_every_locations,
                resume=resume,
                preserve_checkpoint=preserve_checkpoint,
            )
        cached = self._cached_neighbour_prefix(order_k)
        range_mode = source_range is not None
        if cached is not None and not range_mode:
            return cached
        location_count = int(self.unique_locations.shape[0])
        if location_count - 1 < order_k:
            raise ValueError(
                f"k={order_k} requires at least {order_k} other distinct descriptor locations; "
                f"the pool contains only {location_count - 1}"
            )
        if range_mode and checkpoint_store is not None:
            raise ValueError("source-range GPU kNN queries do not accept a checkpoint store")
        range_start, range_stop = (0, location_count) if source_range is None else source_range
        if (
            isinstance(range_start, bool)
            or isinstance(range_stop, bool)
            or not isinstance(range_start, (int, np.integer))
            or not isinstance(range_stop, (int, np.integer))
            or int(range_start) < 0
            or int(range_stop) > location_count
            or int(range_start) >= int(range_stop)
        ):
            raise ValueError("source_range must be a non-empty half-open location range")
        range_start = int(range_start)
        range_stop = int(range_stop)
        if checkpoint_store is not None and checkpoint_identity is None:
            raise ValueError("kNN checkpoint identity is required")
        if (
            isinstance(checkpoint_every_locations, bool)
            or not isinstance(checkpoint_every_locations, (int, np.integer))
            or int(checkpoint_every_locations) < 1
        ):
            raise ValueError("checkpoint_every_locations must be a positive integer")
        if checkpoint_store is not None and not resume:
            checkpoint_store.clear(checkpoint_name)

        neighbours = np.empty((location_count, order_k), dtype=np.int64)
        distances_out = np.empty((location_count, order_k), dtype=np.float64)
        source_start = range_start
        incremental_store: IncrementalCheckpointStore | None = None
        legacy_store: VersionedCheckpointStore | None = None
        if checkpoint_store is not None and not range_mode:
            if checkpoint_identity is None:
                raise ValueError("kNN checkpoint identity is required")
            if isinstance(checkpoint_store, IncrementalCheckpointStore):
                incremental_store = checkpoint_store
                legacy_store = VersionedCheckpointStore(checkpoint_store.root)
            elif isinstance(checkpoint_store, VersionedCheckpointStore):
                legacy_store = checkpoint_store
                incremental_store = IncrementalCheckpointStore(
                    checkpoint_store.root,
                    resource_budget=self.resource_budget,
                )
            else:
                raise TypeError("checkpoint_store must be a supported checkpoint store")
            if not resume:
                incremental_store.clear(checkpoint_name)
                legacy_store.clear(checkpoint_name)
            else:
                incremental_store.clean_stale()
            if resume and incremental_store.exists(checkpoint_name):
                _identity, state, arrays = incremental_store.load(
                    checkpoint_name,
                    expected_identity=checkpoint_identity,
                    k_max=order_k,
                    location_count=location_count,
                )
                source_start = int(state.get("cursor", -1))
                saved_locations = arrays.get("neighbour_locations")
                saved_distances = arrays.get("neighbour_distances")
            elif resume and legacy_store.exists(checkpoint_name):
                _identity, state, arrays = legacy_store.load(
                    checkpoint_name, expected_identity=checkpoint_identity
                )
                source_start = int(state.get("cursor", -1))
                saved_locations = arrays.get("neighbour_locations")
                saved_distances = arrays.get("neighbour_distances")
                if source_start < 1:
                    raise ValueError("GPU kNN legacy checkpoint has no completed range")
                neighbours[:source_start] = saved_locations
                distances_out[:source_start] = saved_distances
                incremental_store.write_chunk(
                    checkpoint_name,
                    identity=checkpoint_identity,
                    start=0,
                    neighbour_locations=neighbours[:source_start],
                    neighbour_distances=distances_out[:source_start],
                    k_max=order_k,
                    location_count=location_count,
                )
            else:
                saved_locations = None
                saved_distances = None
            if incremental_store.exists(checkpoint_name):
                if (
                    source_start < 0
                    or source_start > location_count
                    or saved_locations is None
                    or saved_distances is None
                    or saved_locations.shape != (source_start, order_k)
                    or saved_distances.shape != (source_start, order_k)
                    or saved_locations.dtype != np.dtype(np.int64)
                    or saved_distances.dtype != np.dtype(np.float64)
                ):
                    raise ValueError("GPU kNN checkpoint cursor or arrays are incompatible")
                neighbours[:source_start] = saved_locations
                distances_out[:source_start] = saved_distances

        if not range_mode and source_start == location_count:
            if incremental_store is not None and not preserve_checkpoint:
                incremental_store.clear(checkpoint_name)
                incremental_store.clean_stale()
                if legacy_store is not None:
                    legacy_store.clear(checkpoint_name)
            neighbours.setflags(write=False)
            distances_out.setflags(write=False)
            self._store_neighbour_cache(order_k, neighbours, distances_out)
            return neighbours, distances_out

        requested_sources = min(self.requested_tile_size, block, location_count)
        keep_count = min(location_count, order_k + 2)
        dimension = int(self.descriptors.shape[1])
        state_offset = range_start if range_mode else 0
        state_count = (range_stop - range_start) if range_mode else location_count
        global_state_bytes = int(state_count * keep_count * 16 + state_count * 16)
        full_reference_bytes = int(self.unique_locations.nbytes)
        try:
            source_tile, target_tile, _ = self._tile_shape(
                requested_sources,
                resident_reference_bytes=full_reference_bytes,
                global_state_bytes=global_state_bytes,
            )
            resident_full_reference = True
        except GPUCapacityError:
            source_tile, target_tile, _ = self._tile_shape(
                requested_sources,
                global_state_bytes=global_state_bytes,
            )
            resident_full_reference = False
        source_tile = max(1, min(source_tile, requested_sources))
        target_tile = max(1, min(target_tile, self.requested_tile_size, location_count))
        self.global_candidate_state_bytes = global_state_bytes
        self.resident_reference_bytes = full_reference_bytes if resident_full_reference else 0
        checkpoint_cursor = source_start

        def finish_checkpoint(completed_stop: int) -> None:
            nonlocal checkpoint_cursor
            if (
                incremental_store is None
                or checkpoint_identity is None
                or (
                    completed_stop - checkpoint_cursor < int(checkpoint_every_locations)
                    and completed_stop != location_count
                )
            ):
                return
            incremental_store.write_chunk(
                checkpoint_name,
                identity=checkpoint_identity,
                start=checkpoint_cursor,
                neighbour_locations=neighbours[checkpoint_cursor:completed_stop],
                neighbour_distances=distances_out[checkpoint_cursor:completed_stop],
                k_max=order_k,
                location_count=location_count,
            )
            checkpoint_cursor = completed_stop

        gpu_reference = None
        global_indices = None
        global_distances = None
        omission_bounds = None
        merge_bounds = None
        gpu_targets = None
        gpu_sources = None
        local_indices = None
        local_distances = None
        local_bound = None
        started = time.perf_counter()
        try:
            cp = self._cp
            try:
                with cp.cuda.Device(self.device):
                    global_indices = cp.full((state_count, keep_count), -1, dtype=cp.int64)
                    global_distances = cp.full((state_count, keep_count), np.inf, dtype=cp.float64)
                    omission_bounds = cp.full((state_count,), np.inf, dtype=cp.float64)
                    merge_bounds = cp.full((state_count,), np.inf, dtype=cp.float64)
                    self._synchronize()
            except Exception as exc:
                if isinstance(exc, MemoryError) or "out of memory" in str(exc).lower():
                    raise GPUCapacityError(
                        "GPU kNN persistent candidate state exceeded the configured capacity",
                        operation="exact tiled GPU kNN candidate state",
                        requested_bytes=global_state_bytes,
                        available_bytes=self.max_device_bytes,
                    ) from exc
                raise GPUCertificationError(
                    f"GPU kNN persistent candidate state allocation failed: {exc}"
                ) from exc
            self.device_workspace_peak_bytes = max(
                self.device_workspace_peak_bytes, global_state_bytes
            )
            if resident_full_reference:
                gpu_reference = self._upload_descriptor_block(
                    self.unique_locations,
                    "exact tiled GPU kNN reference pool",
                )

            for reference_start in range(0, location_count, target_tile):
                reference_stop = min(reference_start + target_tile, location_count)
                target_count = reference_stop - reference_start
                if resident_full_reference:
                    assert gpu_reference is not None
                    gpu_targets = gpu_reference[reference_start:reference_stop]
                else:
                    gpu_targets = self._upload_reference_shard(
                        self.unique_locations[reference_start:reference_stop]
                    )
                    self.reference_shards += 1
                for source_start_tile in range(source_start, range_stop, source_tile):
                    source_stop = min(source_start_tile + source_tile, range_stop)
                    batch_count = source_stop - source_start_tile
                    host_bytes = int(
                        2 * batch_count * dimension * 8
                        + batch_count * keep_count * 16
                        + 2 * batch_count * 8
                        + 2 * target_count * dimension * 8
                        + 4096
                    )
                    self.host_workspace_peak_bytes = max(self.host_workspace_peak_bytes, host_bytes)
                    lease = self.resource_budget.acquire(
                        "exact tiled GPU kNN source-target tile", max(1, host_bytes)
                    )
                    tile_started = time.perf_counter()
                    try:
                        if resident_full_reference:
                            assert gpu_reference is not None
                            gpu_sources = gpu_reference[source_start_tile:source_stop]
                        else:
                            gpu_sources = self._upload_source_batch(
                                self.unique_locations[source_start_tile:source_stop]
                            )
                        local_keep = min(target_count, keep_count)
                        local_indices, local_distances, local_bound = self._gpu_top_candidates(
                            gpu_sources,
                            gpu_targets,
                            local_keep,
                            target_offset=reference_start,
                            target_count=target_count,
                        )
                        assert (
                            global_indices is not None
                            and global_distances is not None
                            and omission_bounds is not None
                            and merge_bounds is not None
                        )
                        merge_started = time.perf_counter()
                        self._merge_gpu_candidates(
                            global_indices,
                            global_distances,
                            omission_bounds,
                            merge_bounds,
                            source_start_tile - state_offset,
                            local_indices,
                            local_distances,
                            local_bound,
                            min(reference_start, keep_count),
                            keep_count,
                        )
                        merge_synchronization = self._synchronize()
                        merge_elapsed = max(0.0, time.perf_counter() - merge_started)
                        self.gpu_operation_seconds += merge_elapsed
                        self.gpu_launch_allocation_seconds += max(
                            0.0, merge_elapsed - merge_synchronization
                        )
                    finally:
                        self.gpu_tile_wall_seconds += max(0.0, time.perf_counter() - tile_started)
                        lease.close()
                        gpu_sources = None
                        local_indices = None
                        local_distances = None
                        local_bound = None
                    self.target_tiles += 1
                    self.distance_evaluations += batch_count * target_count
                    self.unique_distance_evaluations += batch_count * target_count
                if not resident_full_reference:
                    del gpu_targets
            self.source_tiles += int((range_stop - source_start + source_tile - 1) // source_tile)

            for source_start_tile in range(source_start, range_stop, source_tile):
                source_stop = min(source_start_tile + source_tile, range_stop)
                state_start = source_start_tile - state_offset
                state_stop = source_stop - state_offset
                batch_count = source_stop - source_start_tile
                host_bytes = int(batch_count * keep_count * 16 + batch_count * 16 + 8192)
                self.host_workspace_peak_bytes = max(self.host_workspace_peak_bytes, host_bytes)
                lease = self.resource_budget.acquire(
                    "exact tiled GPU kNN final certification", max(1, host_bytes)
                )
                try:
                    assert (
                        global_indices is not None
                        and global_distances is not None
                        and omission_bounds is not None
                        and merge_bounds is not None
                    )
                    d2h_started = time.perf_counter()
                    with cp.cuda.Device(self.device):
                        host_indices = np.asarray(
                            cp.asnumpy(global_indices[state_start:state_stop]),
                            dtype=np.int64,
                        )
                        host_distances = np.asarray(
                            cp.asnumpy(global_distances[state_start:state_stop]),
                            dtype=np.float64,
                        )
                        host_omission = np.asarray(
                            cp.asnumpy(omission_bounds[state_start:state_stop]),
                            dtype=np.float64,
                        )
                        host_merge = np.asarray(
                            cp.asnumpy(merge_bounds[state_start:state_stop]),
                            dtype=np.float64,
                        )
                        synchronization = self._synchronize()
                    d2h_elapsed = max(0.0, time.perf_counter() - d2h_started)
                    self.d2h_bytes += int(
                        host_indices.nbytes
                        + host_distances.nbytes
                        + host_omission.nbytes
                        + host_merge.nbytes
                    )
                    self.gpu_d2h_seconds += d2h_elapsed
                    self.gpu_launch_allocation_seconds += max(0.0, d2h_elapsed - synchronization)
                    for offset, source_location in enumerate(range(source_start_tile, source_stop)):
                        locations, values = self._certify_final(
                            self.unique_locations[source_location],
                            source_location,
                            host_indices[offset],
                            host_distances[offset],
                            float(host_omission[offset]),
                            float(host_merge[offset]),
                            order_k,
                        )
                        if locations.size != order_k:
                            raise GPUCertificationError(
                                "GPU kNN exact reduction returned an incomplete neighbour row"
                            )
                        neighbours[source_location] = locations
                        distances_out[source_location] = values
                finally:
                    lease.close()
                self.index_queries += batch_count
                finish_checkpoint(source_stop)
        finally:
            gpu_reference = None
            gpu_targets = None
            gpu_sources = None
            local_indices = None
            local_distances = None
            local_bound = None
            global_indices = None
            global_distances = None
            omission_bounds = None
            merge_bounds = None
            try:
                with self._cp.cuda.Device(self.device):
                    for name in (
                        "get_default_memory_pool",
                        "get_default_pinned_memory_pool",
                    ):
                        pool_factory = getattr(self._cp, name, None)
                        if pool_factory is not None:
                            pool_factory().free_all_blocks()
            except Exception:
                pass
        if incremental_store is not None and not preserve_checkpoint:
            incremental_store.clear(checkpoint_name)
            incremental_store.clean_stale()
            if legacy_store is not None:
                legacy_store.clear(checkpoint_name)
        neighbours.setflags(write=False)
        distances_out.setflags(write=False)
        self._last_neighbour_compute_seconds = max(0.0, time.perf_counter() - started)
        if range_mode:
            return (
                np.asarray(neighbours[range_start:range_stop], dtype=np.int64),
                np.asarray(distances_out[range_start:range_stop], dtype=np.float64),
            )
        self._store_neighbour_cache(order_k, neighbours, distances_out)
        return neighbours, distances_out

    def query_neighbours_range(
        self,
        k: int,
        start: int,
        stop: int,
        *,
        chunk_size: int = 1024,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute one globally indexed contiguous unique-location range.

        The complete unique-location pool remains readonly on this device;
        only the source range is assigned to the worker.  This is the worker
        primitive used by the explicit multi-device scheduler.
        """

        if len(self.gpu_devices) != 1:
            raise ValueError("query_neighbours_range requires a single-device worker")
        return self._compute_location_neighbours(
            k,
            chunk_size=chunk_size,
            source_range=(start, stop),
        )

    def _compute_multi_location_neighbours(
        self,
        k: int,
        *,
        chunk_size: int,
        checkpoint_store: VersionedCheckpointStore | IncrementalCheckpointStore | None,
        checkpoint_name: str,
        checkpoint_identity: dict[str, Any] | None,
        checkpoint_every_locations: int,
        resume: bool,
        preserve_checkpoint: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run deterministic contiguous source ranges on explicit GPU workers.

        Workers are created for every requested visible ordinal before any
        range is published.  They are intentionally used through CuPy device
        contexts rather than a process-wide automatic device selector; the
        same exact screening, FP64 boundary proof and CPU canonical
        certification is therefore applied independently on every range.
        """

        location_count = int(self.unique_locations.shape[0])
        if checkpoint_store is not None and checkpoint_identity is None:
            raise ValueError("kNN checkpoint identity is required")
        if (
            isinstance(checkpoint_every_locations, bool)
            or not isinstance(checkpoint_every_locations, (int, np.integer))
            or int(checkpoint_every_locations) < 1
        ):
            raise ValueError("checkpoint_every_locations must be a positive integer")
        if checkpoint_store is not None and not resume:
            checkpoint_store.clear(checkpoint_name)

        neighbours = np.empty((location_count, k), dtype=np.int64)
        distances = np.empty((location_count, k), dtype=np.float64)
        cursor = 0
        if checkpoint_store is not None and resume and checkpoint_store.exists(checkpoint_name):
            if isinstance(checkpoint_store, IncrementalCheckpointStore):
                _identity, state, arrays = checkpoint_store.load(
                    checkpoint_name,
                    expected_identity=checkpoint_identity,
                    k_max=k,
                    location_count=location_count,
                )
            elif isinstance(checkpoint_store, VersionedCheckpointStore):
                _identity, state, arrays = checkpoint_store.load(
                    checkpoint_name, expected_identity=checkpoint_identity
                )
            else:
                raise TypeError("checkpoint_store must be a supported checkpoint store")
            cursor = int(state.get("cursor", -1))
            saved_locations = arrays.get("neighbour_locations")
            saved_distances = arrays.get("neighbour_distances")
            if (
                cursor < 0
                or cursor > location_count
                or saved_locations is None
                or saved_distances is None
                or saved_locations.shape != (cursor, k)
                or saved_distances.shape != (cursor, k)
                or saved_locations.dtype != np.dtype(np.int64)
                or saved_distances.dtype != np.dtype(np.float64)
            ):
                raise ValueError("multi-GPU kNN checkpoint cursor or arrays are incompatible")
            neighbours[:cursor] = saved_locations
            distances[:cursor] = saved_distances

        # Construction is deliberately eager: device loss/capacity failure on
        # any requested worker occurs before the first durable publication.
        worker_list: list[ExactTiledGPUNeighbourIndex] = []
        try:
            for device in self.gpu_devices:
                worker_list.append(
                    ExactTiledGPUNeighbourIndex(
                        self.descriptors,
                        device=device,
                        gpu_tile_size=self.requested_tile_size,
                        max_device_bytes=self.max_device_bytes,
                        tile_size=self.tile_size,
                        max_index_bytes=self.index_bytes,
                        max_radius_query_bytes=self.max_radius_query_bytes,
                        max_neighbour_cache_bytes=self.max_neighbour_cache_bytes,
                        resource_budget=self.resource_budget,
                    )
                )
        except BaseException:
            for worker in worker_list:
                worker.close()
            raise
        workers = tuple(worker_list)
        ranges = contiguous_gpu_ranges(location_count, self.gpu_devices)
        block = min(int(chunk_size), int(checkpoint_every_locations))

        def publish(start: int, stop: int) -> None:
            if checkpoint_store is None or checkpoint_identity is None:
                return
            if isinstance(checkpoint_store, IncrementalCheckpointStore):
                checkpoint_store.write_chunk(
                    checkpoint_name,
                    identity=checkpoint_identity,
                    start=start,
                    neighbour_locations=neighbours[start:stop],
                    neighbour_distances=distances[start:stop],
                    k_max=k,
                    location_count=location_count,
                )
            elif isinstance(checkpoint_store, VersionedCheckpointStore):
                checkpoint_store.write(
                    checkpoint_name,
                    identity=checkpoint_identity,
                    state={"cursor": stop},
                    arrays={
                        "neighbour_locations": neighbours[:stop],
                        "neighbour_distances": distances[:stop],
                    },
                )
            else:
                raise TypeError("checkpoint_store must be a supported checkpoint store")

        started = time.perf_counter()
        try:
            for worker_index, (_device, range_start, range_stop) in enumerate(ranges):
                if range_stop <= cursor:
                    continue
                start = max(range_start, cursor)
                while start < range_stop:
                    stop = min(start + block, range_stop)
                    worker_locations, worker_distances = workers[
                        worker_index
                    ].query_neighbours_range(k, start, stop, chunk_size=chunk_size)
                    neighbours[start:stop] = worker_locations
                    distances[start:stop] = worker_distances
                    publish(start, stop)
                    cursor = stop
                    start = stop
        except BaseException:
            raise
        finally:
            self.per_device_metrics = {
                int(worker.device): worker.memory_metrics() for worker in workers
            }
            for worker in workers:
                worker.close()
        if checkpoint_store is not None and not preserve_checkpoint:
            checkpoint_store.clear(checkpoint_name)
            checkpoint_store.clean_stale()
        neighbours.setflags(write=False)
        distances.setflags(write=False)
        self._last_neighbour_compute_seconds = max(0.0, time.perf_counter() - started)
        self._merge_multi_metrics()
        self._store_neighbour_cache(k, neighbours, distances)
        return neighbours, distances

    def _merge_multi_metrics(self) -> None:
        """Aggregate operational worker diagnostics without changing science."""

        if not self.per_device_metrics:
            return
        numeric_sum = (
            "gpu_source_tiles",
            "gpu_target_tiles",
            "gpu_reference_uploads",
            "gpu_reference_shards",
            "gpu_fp64_evaluations",
            "gpu_h2d_bytes",
            "gpu_d2h_bytes",
            "gpu_cpu_rescanned_count",
            "gpu_exact_boundary_recertifications",
        )
        for name in numeric_sum:
            setattr(
                self,
                name.replace("gpu_", ""),
                sum(int(metrics.get(name, 0) or 0) for metrics in self.per_device_metrics.values()),
            )
        self.source_tiles = sum(
            int(item.get("gpu_source_tiles", 0) or 0) for item in self.per_device_metrics.values()
        )
        self.target_tiles = sum(
            int(item.get("gpu_target_tiles", 0) or 0) for item in self.per_device_metrics.values()
        )
        self.reference_uploads = sum(
            int(item.get("gpu_reference_uploads", 0) or 0)
            for item in self.per_device_metrics.values()
        )
        self.reference_shards = sum(
            int(item.get("gpu_reference_shards", 0) or 0)
            for item in self.per_device_metrics.values()
        )
        self.fp64_evaluations = sum(
            int(item.get("gpu_fp64_evaluations", 0) or 0)
            for item in self.per_device_metrics.values()
        )
        self.h2d_bytes = sum(
            int(item.get("gpu_h2d_bytes", 0) or 0) for item in self.per_device_metrics.values()
        )
        self.d2h_bytes = sum(
            int(item.get("gpu_d2h_bytes", 0) or 0) for item in self.per_device_metrics.values()
        )
        self.certification_rescans = sum(
            int(item.get("gpu_cpu_rescanned_count", 0) or 0)
            for item in self.per_device_metrics.values()
        )
        self.ambiguous_boundary_count = sum(
            int(item.get("gpu_exact_boundary_recertifications", 0) or 0)
            for item in self.per_device_metrics.values()
        )
        self.device_workspace_peak_bytes = max(
            int(item.get("gpu_device_workspace_peak_bytes", 0) or 0)
            for item in self.per_device_metrics.values()
        )
        self.host_workspace_peak_bytes = max(
            int(item.get("gpu_host_workspace_peak_bytes", 0) or 0)
            for item in self.per_device_metrics.values()
        )
        self.gpu_seconds = sum(
            float(item.get("gpu_kernel_seconds", 0.0) or 0.0)
            for item in self.per_device_metrics.values()
        )
        self.gpu_operation_seconds = sum(
            float(item.get("gpu_operation_seconds_inclusive", 0.0) or 0.0)
            for item in self.per_device_metrics.values()
        )
        self.gpu_h2d_seconds = sum(
            float(item.get("gpu_h2d_seconds", 0.0) or 0.0)
            for item in self.per_device_metrics.values()
        )
        self.gpu_d2h_seconds = sum(
            float(item.get("gpu_d2h_seconds", 0.0) or 0.0)
            for item in self.per_device_metrics.values()
        )
        self.gpu_synchronization_seconds = sum(
            float(item.get("gpu_synchronization_seconds", 0.0) or 0.0)
            for item in self.per_device_metrics.values()
        )
        self.gpu_launch_allocation_seconds = sum(
            float(item.get("gpu_launch_allocation_seconds", 0.0) or 0.0)
            for item in self.per_device_metrics.values()
        )
        self.gpu_tile_wall_seconds = sum(
            float(item.get("gpu_tile_wall_seconds", 0.0) or 0.0)
            for item in self.per_device_metrics.values()
        )
        self.cpu_verification_seconds = sum(
            float(item.get("gpu_cpu_verification_seconds", 0.0) or 0.0)
            for item in self.per_device_metrics.values()
        )
        self.certification_seconds = sum(
            float(item.get("gpu_certification_seconds", 0.0) or 0.0)
            for item in self.per_device_metrics.values()
        )
        self.certified_tiles = sum(
            int(item.get("gpu_certified_tiles", 0) or 0)
            for item in self.per_device_metrics.values()
        )
        self.final_certifications = sum(
            int(item.get("gpu_final_certifications", 0) or 0)
            for item in self.per_device_metrics.values()
        )

    def memory_metrics(self) -> dict[str, Any]:
        metrics: dict[str, Any] = dict(super().memory_metrics())
        metrics.update(
            {
                "backend": self.backend,
                "backend_version": self.backend_version,
                "gpu_device": int(self.device),
                "gpu_devices": [int(device) for device in self.gpu_devices],
                "gpu_device_count": len(self.gpu_devices),
                "gpu_tile_size": int(self.requested_tile_size),
                "gpu_max_device_bytes": int(self.max_device_bytes),
                "gpu_total_bytes": int(self.device_total_bytes),
                "gpu_free_bytes": int(self.device_free_bytes),
                "gpu_source_tiles": int(self.source_tiles),
                "gpu_target_tiles": int(self.target_tiles),
                "gpu_reference_uploads": int(self.reference_uploads),
                "gpu_reference_shards": int(self.reference_shards),
                "gpu_resident_reference_bytes": int(self.resident_reference_bytes),
                "gpu_global_candidate_state_bytes": int(self.global_candidate_state_bytes),
                "gpu_fp64_evaluations": int(self.fp64_evaluations),
                "gpu_h2d_bytes": int(self.h2d_bytes),
                "gpu_d2h_bytes": int(self.d2h_bytes),
                "gpu_device_workspace_peak_bytes": int(self.device_workspace_peak_bytes),
                "gpu_host_workspace_peak_bytes": int(self.host_workspace_peak_bytes),
                "gpu_kernel_seconds": float(self.gpu_seconds),
                "gpu_distance_reduction_seconds": float(self.gpu_distance_reduction_seconds),
                "gpu_operation_seconds_inclusive": float(self.gpu_operation_seconds),
                "gpu_h2d_seconds": float(self.gpu_h2d_seconds),
                "gpu_d2h_seconds": float(self.gpu_d2h_seconds),
                "gpu_synchronization_seconds": float(self.gpu_synchronization_seconds),
                "gpu_launch_allocation_seconds": float(self.gpu_launch_allocation_seconds),
                "gpu_tile_wall_seconds": float(self.gpu_tile_wall_seconds),
                "gpu_kernel_seconds_definition": (
                    "inclusive FP64 distance/reduction section wall time, including its "
                    "synchronization and excluding measured transfers; not a hardware "
                    "kernel-only timestamp"
                ),
                "gpu_operation_seconds_definition": (
                    "inclusive per-source/reference-tile CuPy wall time for FP64 distance, "
                    "reduction, and GPU merge operations; transfers are separately reported"
                ),
                "gpu_launch_allocation_seconds_definition": (
                    "non-overlapping residual after measured transfers and distance section; "
                    "individual allocator and kernel-launch timestamps are unavailable"
                ),
                "gpu_synchronization_seconds_definition": (
                    "synchronization time recorded at transfer and computation boundaries; "
                    "diagnostic and not an exclusive phase"
                ),
                "knn_wall_seconds": float(self._last_neighbour_compute_seconds),
                "gpu_certification_seconds": float(self.certification_seconds),
                "gpu_cpu_verification_seconds": float(self.cpu_verification_seconds),
                "gpu_certification_rescans": int(self.certification_rescans),
                "gpu_ambiguous_boundary_count": int(self.ambiguous_boundary_count),
                "gpu_exact_boundary_recertifications": int(self.ambiguous_boundary_count),
                "gpu_certified_tiles": int(self.certified_tiles),
                "gpu_final_certifications": int(self.final_certifications),
                "gpu_certification_relative_bound": float(self.certification_relative_bound),
                "gpu_certification_contract": "fp64-gamma-bound-v2",
                "gpu_certification_fp64_safe": bool(self._gpu_fp64_certification_safe),
                "gpu_per_device": [metrics for _device, metrics in self.per_device_metrics.items()],
                "knn_throughput_locations_per_second": (
                    self.unique_locations.shape[0] / self._last_neighbour_compute_seconds
                    if self._last_neighbour_compute_seconds > 0.0
                    else 0.0
                ),
            }
        )
        return metrics

    def close(self) -> None:
        try:
            with self._cp.cuda.Device(self.device):
                for name in ("get_default_memory_pool", "get_default_pinned_memory_pool"):
                    pool_factory = getattr(self._cp, name, None)
                    if pool_factory is not None:
                        pool_factory().free_all_blocks()
        except Exception:
            pass
        super().close()


def _validate_gpu_result_limits(
    values: np.ndarray,
    k: int,
    max_neighbour_entries: int | None,
    remaining: int | None,
) -> None:
    required_entries = int(values.shape[0]) * int(k)
    if max_neighbour_entries is not None and (
        isinstance(max_neighbour_entries, bool)
        or not isinstance(max_neighbour_entries, (int, np.integer))
        or int(max_neighbour_entries) < 1
    ):
        raise ValueError("max_neighbour_entries must be a positive integer when provided")
    if max_neighbour_entries is not None and required_entries > int(max_neighbour_entries):
        raise ValueError("exact GPU neighbour result exceeds its explicit bounded-memory limit")
    requested = int(values.nbytes + 16 * required_entries + 128 * values.shape[0])
    if remaining is None:
        raise ResourceCapacityError(
            "exact GPU kNN requires a known host memory allowance",
            operation="exact tiled GPU kNN",
        )
    if requested > int(remaining):
        raise ResourceCapacityError(
            "exact GPU neighbour result exceeds its runtime memory allowance: "
            f"requested_bytes={requested}, available_bytes={remaining}",
            operation="exact tiled GPU kNN",
            requested_bytes=requested,
            available_bytes=remaining,
        )


def _resolve_device_limit(
    max_device_bytes: int | None,
    max_gpu_neighbour_bytes: int | None,
) -> int | None:
    if (
        max_device_bytes is not None
        and max_gpu_neighbour_bytes is not None
        and int(max_device_bytes) != int(max_gpu_neighbour_bytes)
    ):
        raise ValueError(
            "max_device_bytes and max_gpu_neighbour_bytes must agree when both are provided"
        )
    return max_gpu_neighbour_bytes if max_gpu_neighbour_bytes is not None else max_device_bytes


def _build_result(
    values: np.ndarray,
    active_index: ExactTiledGPUNeighbourIndex,
    order_k: int,
    radii: np.ndarray,
    neighbour_indices: np.ndarray,
    neighbour_distances: np.ndarray,
    representation_id: str,
    ordered_row_ids: tuple[str, ...],
) -> ExactNeighbourResult:
    identity = {
        "schema": active_index.backend_version,
        "representation": representation_id,
        "row_ids": list(ordered_row_ids),
        "metric": active_index.metric,
        "backend": active_index.backend,
        "backend_fingerprint": active_index.fingerprint,
        "distinct_location_policy": active_index.distinct_location_policy,
        "tie_policy": "distance-then-earliest-original-representative-v1",
        "k": int(order_k),
    }
    return ExactNeighbourResult(
        radii=radii,
        neighbour_indices=neighbour_indices,
        neighbour_distances=neighbour_distances,
        row_to_location=active_index.row_to_location,
        unique_locations=active_index.unique_locations,
        location_representatives=active_index.location_representatives,
        k=order_k,
        representation_fingerprint=representation_id,
        row_ids=ordered_row_ids,
        fingerprint=sha256_canonical_json(identity),
        backend=active_index.backend,
        backend_version=active_index.backend_version,
        metric=active_index.metric,
        distinct_location_policy=active_index.distinct_location_policy,
        backend_fingerprint=active_index.fingerprint,
    )


def compute_gpu_tiled_neighbours(
    descriptors: np.ndarray,
    k: int,
    *,
    index: ExactTiledGPUNeighbourIndex | None = None,
    chunk_size: int = 1024,
    representation_fingerprint: str | None = None,
    row_ids: Sequence[str] | None = None,
    max_neighbour_entries: int | None = None,
    max_device_bytes: int | None = None,
    max_gpu_neighbour_bytes: int | None = None,
    gpu_device: int = 0,
    gpu_devices: Sequence[int] | str | int | None = None,
    gpu_tile_size: int = 128,
    max_index_bytes: int | None = None,
    max_radius_query_bytes: int | None = None,
    max_neighbour_cache_bytes: int | None = DEFAULT_NEIGHBOUR_CACHE_BYTES,
    resource_budget: ResourceBudgetService | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    checkpoint_name: str = "knn",
    checkpoint_every_locations: int = 1,
    resume: bool = True,
    checkpoint_metrics: dict[str, Any] | None = None,
    preserve_checkpoint: bool = False,
) -> ExactNeighbourResult:
    values = _validate_descriptors(descriptors)
    order_k = _validate_k(k)
    _validate_chunk_size(chunk_size)
    service = resource_budget or (index.resource_budget if index is not None else None)
    if service is None:
        from nepflow.resources.budget import build_resource_budget

        service = build_resource_budget()
    _validate_gpu_result_limits(
        values, order_k, max_neighbour_entries, service.remaining_managed_budget
    )
    active_index = index or ExactTiledGPUNeighbourIndex(
        values,
        device=gpu_device,
        gpu_devices=gpu_devices,
        gpu_tile_size=gpu_tile_size,
        max_device_bytes=_resolve_device_limit(max_device_bytes, max_gpu_neighbour_bytes),
        tile_size=chunk_size,
        max_index_bytes=max_index_bytes,
        max_radius_query_bytes=max_radius_query_bytes,
        max_neighbour_cache_bytes=max_neighbour_cache_bytes,
        resource_budget=service,
    )
    if not isinstance(active_index, ExactTiledGPUNeighbourIndex) or not np.array_equal(
        active_index.descriptors, values
    ):
        raise ValueError("exact_tiled_gpu_knn index belongs to different descriptors")
    ordered_row_ids = _validate_row_ids(row_ids, values.shape[0])
    store = (
        None
        if checkpoint_path is None
        else IncrementalCheckpointStore(checkpoint_path, resource_budget=service)
    )
    identity = {
        "phase": "exact-knn",
        "schema": "exact-knn-checkpoint-v1",
        "backend": active_index.backend,
        "backend_version": active_index.backend_version,
        "descriptors": _array_fingerprint(values),
        "row_ids": list(ordered_row_ids),
        "unique_locations": _array_fingerprint(active_index.unique_locations),
        "row_to_location": _array_fingerprint(active_index.row_to_location),
        "k": order_k,
        "tie_policy": "distance-then-earliest-original-representative-v1",
    }
    radii, neighbours, distances = active_index.query_neighbours(
        order_k,
        chunk_size=chunk_size,
        checkpoint_store=store,
        checkpoint_name=checkpoint_name,
        checkpoint_identity=identity,
        checkpoint_every_locations=checkpoint_every_locations,
        resume=resume,
        preserve_checkpoint=preserve_checkpoint,
    )
    if checkpoint_metrics is not None:
        checkpoint_metrics.clear()
        if store is not None:
            checkpoint_metrics.update(store.metrics())
    representation_id = (
        _array_fingerprint(values)
        if representation_fingerprint is None
        else representation_fingerprint
    )
    if not isinstance(representation_id, str) or not representation_id.strip():
        raise ValueError("representation_fingerprint must be a non-blank string")
    return _build_result(
        values,
        active_index,
        order_k,
        radii,
        neighbours,
        distances,
        representation_id,
        ordered_row_ids,
    )


def compute_gpu_tiled_neighbours_many(
    descriptors: np.ndarray,
    ks: Sequence[int],
    *,
    index: ExactTiledGPUNeighbourIndex | None = None,
    chunk_size: int = 1024,
    representation_fingerprint: str | None = None,
    row_ids: Sequence[str] | None = None,
    max_neighbour_entries: int | None = None,
    max_device_bytes: int | None = None,
    max_gpu_neighbour_bytes: int | None = None,
    gpu_device: int = 0,
    gpu_devices: Sequence[int] | str | int | None = None,
    gpu_tile_size: int = 128,
    max_index_bytes: int | None = None,
    max_radius_query_bytes: int | None = None,
    max_neighbour_cache_bytes: int | None = DEFAULT_NEIGHBOUR_CACHE_BYTES,
    resource_budget: ResourceBudgetService | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    checkpoint_name: str = "knn",
    checkpoint_every_locations: int = 1,
    resume: bool = True,
    preserve_checkpoint: bool = False,
) -> dict[int, ExactNeighbourResult]:
    ordered = tuple(_validate_k(value) for value in ks)
    if not ordered or len(set(ordered)) != len(ordered):
        raise ValueError("ks must contain unique positive neighbour orders")
    values = _validate_descriptors(descriptors)
    maximum = max(ordered)
    service = resource_budget or (index.resource_budget if index is not None else None)
    if service is None:
        from nepflow.resources.budget import build_resource_budget

        service = build_resource_budget()
    _validate_gpu_result_limits(
        values, maximum, max_neighbour_entries, service.remaining_managed_budget
    )
    active_index = index or ExactTiledGPUNeighbourIndex(
        values,
        device=gpu_device,
        gpu_devices=gpu_devices,
        gpu_tile_size=gpu_tile_size,
        max_device_bytes=_resolve_device_limit(max_device_bytes, max_gpu_neighbour_bytes),
        tile_size=chunk_size,
        max_index_bytes=max_index_bytes,
        max_radius_query_bytes=max_radius_query_bytes,
        max_neighbour_cache_bytes=max_neighbour_cache_bytes,
        resource_budget=service,
    )
    if not isinstance(active_index, ExactTiledGPUNeighbourIndex) or not np.array_equal(
        active_index.descriptors, values
    ):
        raise ValueError("exact_tiled_gpu_knn index belongs to different descriptors")
    ordered_row_ids = _validate_row_ids(row_ids, values.shape[0])
    store = (
        None
        if checkpoint_path is None
        else IncrementalCheckpointStore(checkpoint_path, resource_budget=service)
    )
    identity = {
        "phase": "exact-knn",
        "schema": "exact-knn-checkpoint-v1",
        "backend": active_index.backend,
        "backend_version": active_index.backend_version,
        "descriptors": _array_fingerprint(values),
        "row_ids": list(ordered_row_ids),
        "unique_locations": _array_fingerprint(active_index.unique_locations),
        "row_to_location": _array_fingerprint(active_index.row_to_location),
        "k": maximum,
        "tie_policy": "distance-then-earliest-original-representative-v1",
    }
    queried = active_index.query_neighbours_many(
        ordered,
        chunk_size=chunk_size,
        checkpoint_store=store,
        checkpoint_name=checkpoint_name,
        checkpoint_identity=identity,
        checkpoint_every_locations=checkpoint_every_locations,
        resume=resume,
        preserve_checkpoint=preserve_checkpoint,
    )
    results: dict[int, ExactNeighbourResult] = {}
    representation_id = (
        _array_fingerprint(values)
        if representation_fingerprint is None
        else representation_fingerprint
    )
    if not isinstance(representation_id, str) or not representation_id.strip():
        raise ValueError("representation_fingerprint must be a non-blank string")
    for order_k in ordered:
        radii, neighbours, distances = queried[order_k]
        results[order_k] = _build_result(
            values,
            active_index,
            order_k,
            radii,
            neighbours,
            distances,
            representation_id,
            ordered_row_ids,
        )
    return results


# Match the naming convention of the CPU tiled entry points while retaining a
# descriptive GPU-first spelling for standalone callers.
compute_tiled_gpu_neighbours = compute_gpu_tiled_neighbours
compute_tiled_gpu_neighbours_many = compute_gpu_tiled_neighbours_many


__all__ = [
    "ExactTiledGPUNeighbourIndex",
    "GPUCapacityError",
    "GPUCertificationError",
    "GPUDeviceValidationError",
    "compute_gpu_tiled_neighbours",
    "compute_gpu_tiled_neighbours_many",
    "compute_tiled_gpu_neighbours",
    "compute_tiled_gpu_neighbours_many",
]
