"""Optional exact, certified CuPy tiled k-nearest-neighbour backend.

CuPy is deliberately imported only after this backend is selected.  The
implementation uses the GPU to screen exhaustive float64 tiles and uses the
canonical CPU distance operation to certify every retained boundary.  A tile
whose boundary cannot be proved safe is rescanned in full on the CPU; it is
never accepted as an approximate top-k result.
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
    TiledCPUNeighbourIndex,
    _array_fingerprint,
    _validate_chunk_size,
    _validate_descriptors,
    _validate_k,
    _validate_row_ids,
)


def _gamma_bound(dimension: int) -> float:
    """Return an outward-rounded relative FP64 operation bound.

    The GPU tile computes one subtraction, square, and sequential sum per
    coordinate, followed by a square root.  The allowance covers those
    operations plus the device reduction/selection path.  If the bound cannot
    be represented as a finite positive number, certification fails closed.
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


def _canonical_distances(
    query: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """Compute the scientific CPU float64 distance primitive."""

    if targets.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    with np.errstate(over="raise", invalid="raise"):
        try:
            distances = np.sqrt(
                np.sum((targets - query) ** 2, axis=1, dtype=np.float64)
            )
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
        if isinstance(gpu_tile_size, bool) or not isinstance(
            gpu_tile_size, (int, np.integer)
        ) or int(gpu_tile_size) < 1:
            raise ValueError("gpu_tile_size must be a positive integer")
        self.device = int(device)
        self.requested_tile_size = int(gpu_tile_size)
        self._cp = self._load_cupy()
        self._validate_device()
        free_bytes, total_bytes = self._device_memory_info()
        if free_bytes < 1 or total_bytes < 1:
            raise GPUDeviceValidationError(
                f"CUDA device {self.device} reported no usable memory"
            )
        if max_device_bytes is None:
            self.max_device_bytes = max(1, int(free_bytes * 0.80))
        else:
            if isinstance(max_device_bytes, bool) or not isinstance(
                max_device_bytes, (int, np.integer)
            ) or int(max_device_bytes) < 1:
                raise ValueError("max_gpu_neighbour_bytes must be positive when provided")
            self.max_device_bytes = min(int(max_device_bytes), int(free_bytes))
        self.device_total_bytes = int(total_bytes)
        self.device_free_bytes = int(free_bytes)
        values = _validate_descriptors(descriptors)
        self.certification_relative_bound = _gamma_bound(int(values.shape[1]))
        super().__init__(
            values,
            tile_size=(self.requested_tile_size if tile_size is None else tile_size),
            max_index_bytes=max_index_bytes,
            max_radius_query_bytes=max_radius_query_bytes,
            max_neighbour_cache_bytes=max_neighbour_cache_bytes,
            resource_budget=resource_budget,
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
        self.certification_seconds = 0.0
        self.cpu_verification_seconds = 0.0
        self.certification_rescans = 0
        self.ambiguous_boundary_count = 0
        self.certified_tiles = 0

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
            count = int(self._cp.cuda.runtime.getDeviceCount())
        except Exception as exc:  # pragma: no cover - CUDA runtime dependent
            raise GPUDeviceValidationError(
                f"unable to query visible CUDA devices for gpu_device={self.device}: {exc}"
            ) from exc
        if self.device >= count:
            raise GPUDeviceValidationError(
                f"gpu_device={self.device} is not visible; CUDA device count is {count}"
            )

    def _device_memory_info(self) -> tuple[int, int]:
        try:
            with self._cp.cuda.Device(self.device):
                free_bytes, total_bytes = self._cp.cuda.runtime.memGetInfo()
        except Exception as exc:  # pragma: no cover - CUDA runtime dependent
            raise GPUDeviceValidationError(
                f"unable to inspect CUDA device {self.device} memory: {exc}"
            ) from exc
        return int(free_bytes), int(total_bytes)

    def _tile_shape(self, source_count: int) -> tuple[int, int, int]:
        dimension = int(self.descriptors.shape[1])
        sources = max(1, min(int(source_count), self.requested_tile_size))
        targets = min(self.requested_tile_size, int(self.unique_locations.shape[0]))

        def required(source_rows: int, target_rows: int) -> int:
            # dist2, selected sqrt output, delta/square temporaries and
            # transferred descriptor tiles.  The dimension loop never creates
            # a source x target x dimension allocation.
            return int(
                64 * source_rows * target_rows
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

    def _upload_source_batch(self, source: np.ndarray) -> Any:
        """Upload one source batch, which is reused for every target tile."""

        source_values = np.ascontiguousarray(source, dtype=np.float64)
        started = time.perf_counter()
        try:
            with self._cp.cuda.Device(self.device):
                gpu_sources = self._cp.asarray(source_values, dtype=self._cp.float64)
                synchronization = self._synchronize()
        except Exception as exc:
            if isinstance(exc, (GPUCapacityError, GPUCertificationError)):
                raise
            if isinstance(exc, MemoryError) or "out of memory" in str(exc).lower():
                raise GPUCapacityError(
                    "GPU source-batch allocation exceeded the configured device capacity: "
                    f"device={self.device}, max_gpu_neighbour_bytes={self.max_device_bytes}",
                    operation="exact tiled GPU kNN source batch",
                    requested_bytes=int(source_values.nbytes),
                    available_bytes=self.max_device_bytes,
                ) from exc
            raise GPUCertificationError(
                f"GPU source-batch upload failed on gpu_device={self.device}: {exc}"
            ) from exc
        self.h2d_bytes += int(source_values.nbytes)
        h2d_elapsed = max(0.0, time.perf_counter() - started)
        self.gpu_h2d_seconds += h2d_elapsed
        self.gpu_launch_allocation_seconds += max(
            0.0,
            h2d_elapsed - synchronization,
        )
        return gpu_sources

    def _gpu_top_candidates(
        self,
        source: np.ndarray,
        targets: np.ndarray,
        keep_count: int,
        *,
        gpu_sources: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        cp = self._cp
        source_values = np.ascontiguousarray(source, dtype=np.float64)
        target_values = np.ascontiguousarray(targets, dtype=np.float64)
        reserved = int(
            64 * source_values.shape[0] * target_values.shape[0]
            + 16
            * (source_values.shape[0] + target_values.shape[0])
            * source_values.shape[1]
        )
        operation_started = time.perf_counter()
        h2d_started = time.perf_counter()
        h2d_elapsed = 0.0
        d2h_elapsed = 0.0
        compute_elapsed = 0.0
        try:
            with cp.cuda.Device(self.device):
                if gpu_sources is None:
                    gpu_sources = cp.asarray(source_values, dtype=cp.float64)
                    self.h2d_bytes += int(source_values.nbytes)
                gpu_targets = cp.asarray(target_values, dtype=cp.float64)
                self.h2d_bytes += int(target_values.nbytes)
                self._synchronize()
                h2d_elapsed = max(0.0, time.perf_counter() - h2d_started)
                self.gpu_h2d_seconds += h2d_elapsed

                compute_started = time.perf_counter()
                distance_squared = cp.zeros(
                    (source_values.shape[0], target_values.shape[0]), dtype=cp.float64
                )
                for dimension in range(source_values.shape[1]):
                    delta = (
                        gpu_targets[None, :, dimension]
                        - gpu_sources[:, None, dimension]
                    )
                    distance_squared += delta * delta
                isfinite = getattr(cp, "isfinite", None)
                all_values_finite = getattr(cp, "all", None)
                if isfinite is not None and all_values_finite is not None:
                    if not bool(all_values_finite(isfinite(distance_squared))):
                        raise GPUCertificationError(
                            "GPU kNN distance tile contains non-finite FP64 values; "
                            "exact certification failed"
                        )
                partition = cp.argpartition(
                    distance_squared,
                    kth=int(keep_count - 1),
                    axis=1,
                )[:, :keep_count]
                selected_squared = cp.take_along_axis(
                    distance_squared, partition, axis=1
                )
                selected_distances = cp.sqrt(selected_squared)
                self._synchronize()
                compute_elapsed = max(0.0, time.perf_counter() - compute_started)
                self.gpu_distance_reduction_seconds += compute_elapsed

                d2h_started = time.perf_counter()
                host_indices = np.asarray(cp.asnumpy(partition), dtype=np.int64)
                host_distances = np.asarray(
                    cp.asnumpy(selected_distances), dtype=np.float64
                )
                self._synchronize()
                self.d2h_bytes += int(host_indices.nbytes + host_distances.nbytes)
                d2h_elapsed = max(0.0, time.perf_counter() - d2h_started)
                self.gpu_d2h_seconds += d2h_elapsed
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
            operation_elapsed - h2d_elapsed - compute_elapsed - d2h_elapsed,
        )
        if not np.all(np.isfinite(host_distances)):
            raise GPUCertificationError(
                "GPU kNN produced non-finite boundary distances; exact certification failed"
            )
        self.fp64_evaluations += int(source_values.shape[0] * target_values.shape[0])
        self.device_workspace_peak_bytes = max(self.device_workspace_peak_bytes, reserved)
        return host_indices, host_distances

    def _merge(
        self,
        best_locations: np.ndarray,
        best_distances: np.ndarray,
        locations: np.ndarray,
        distances: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if locations.size == 0:
            return best_locations, best_distances
        combined_locations = np.concatenate((best_locations, locations))
        combined_distances = np.concatenate((best_distances, distances))
        order = np.lexsort(
            (
                self.location_representatives[combined_locations],
                combined_distances,
            )
        )[:k]
        return combined_locations[order], combined_distances[order]

    def _certify_tile(
        self,
        query: np.ndarray,
        source_location: int,
        target_locations: np.ndarray,
        reported_locations: np.ndarray,
        reported_distances: np.ndarray,
        best_locations: np.ndarray,
        best_distances: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Certify a GPU top-k tile or exactly rescan the whole tile."""

        nonself = reported_locations != source_location
        reported_locations = reported_locations[nonself]
        reported_distances = reported_distances[nonself]
        report_order = np.lexsort(
            (
                self.location_representatives[reported_locations],
                reported_distances,
            )
        )
        reported_locations = reported_locations[report_order]
        reported_distances = reported_distances[report_order]
        verification_started = time.perf_counter()
        canonical = _canonical_distances(
            query, self.unique_locations[reported_locations]
        )
        self.cpu_verification_seconds += max(
            0.0, time.perf_counter() - verification_started
        )
        best_locations, best_distances = self._merge(
            best_locations, best_distances, reported_locations, canonical, k
        )
        omitted = reported_locations.size < int(target_locations.size - 1)
        if not omitted:
            self.certified_tiles += 1
            return best_locations, best_distances
        if reported_locations.size <= k or best_locations.size < k:
            certifiable = False
        else:
            boundary = float(reported_distances[k])
            relative = self.certification_relative_bound
            lower = boundary / (1.0 + relative)
            upper = float(best_distances[k - 1]) * (1.0 + relative)
            lower = float(np.nextafter(lower, 0.0))
            upper = float(np.nextafter(upper, np.inf))
            certifiable = bool(np.isfinite(lower) and np.isfinite(upper) and lower > upper)
            if not certifiable:
                self.ambiguous_boundary_count += 1
        if certifiable:
            self.certified_tiles += 1
            return best_locations, best_distances

        self.certification_rescans += 1
        verification_started = time.perf_counter()
        rescan_locations = target_locations[target_locations != source_location]
        rescan_distances = _canonical_distances(
            query, self.unique_locations[rescan_locations]
        )
        self.cpu_verification_seconds += max(
            0.0, time.perf_counter() - verification_started
        )
        return self._merge(
            best_locations,
            best_distances,
            rescan_locations,
            rescan_distances,
            k,
        )

    def _compute_location_neighbours(
        self,
        k: int,
        *,
        chunk_size: int = 1024,
        checkpoint_store: VersionedCheckpointStore | IncrementalCheckpointStore | None = None,
        checkpoint_name: str = "knn",
        checkpoint_identity: dict[str, Any] | None = None,
        checkpoint_every_locations: int = 1,
        resume: bool = True,
        preserve_checkpoint: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        order_k = _validate_k(k)
        block = _validate_chunk_size(chunk_size)
        cached = self._cached_neighbour_prefix(order_k)
        if cached is not None:
            return cached
        location_count = int(self.unique_locations.shape[0])
        if location_count - 1 < order_k:
            raise ValueError(
                f"k={order_k} requires at least {order_k} other distinct descriptor locations; "
                f"the pool contains only {location_count - 1}"
            )
        if checkpoint_store is not None and checkpoint_identity is None:
            raise ValueError("kNN checkpoint identity is required")
        if isinstance(checkpoint_every_locations, bool) or not isinstance(
            checkpoint_every_locations, (int, np.integer)
        ) or int(checkpoint_every_locations) < 1:
            raise ValueError("checkpoint_every_locations must be a positive integer")
        if checkpoint_store is not None and not resume:
            checkpoint_store.clear(checkpoint_name)

        neighbours = np.empty((location_count, order_k), dtype=np.int64)
        distances_out = np.empty((location_count, order_k), dtype=np.float64)
        source_start = 0
        incremental_store: IncrementalCheckpointStore | None = None
        legacy_store: VersionedCheckpointStore | None = None
        if checkpoint_store is not None:
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

        requested_sources = min(self.requested_tile_size, block, location_count)
        source_tile, target_tile, _ = self._tile_shape(requested_sources)
        source_tile = max(1, min(source_tile, requested_sources))
        target_tile = max(1, min(target_tile, self.requested_tile_size, location_count))
        dimension = int(self.descriptors.shape[1])
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

        started = time.perf_counter()
        for source_start_tile in range(source_start, location_count, source_tile):
            source_stop = min(source_start_tile + source_tile, location_count)
            self.source_tiles += 1
            source_values = self.unique_locations[source_start_tile:source_stop]
            batch_count = source_stop - source_start_tile
            best_locations = [np.empty(0, dtype=np.int64) for _ in range(batch_count)]
            best_distances = [np.empty(0, dtype=np.float64) for _ in range(batch_count)]
            gpu_sources = None
            for target_start in range(0, location_count, target_tile):
                target_stop = min(target_start + target_tile, location_count)
                target_count = target_stop - target_start
                keep_count = min(target_count, order_k + 2)
                host_bytes = int(
                    2 * batch_count * dimension * 8
                    + 2 * target_count * dimension * 8
                    + 40 * target_count * dimension
                    + target_count * 16
                    + batch_count * keep_count * 16
                    + batch_count * order_k * 16
                    + (batch_count + target_count) * 8
                    + 4096
                )
                self.host_workspace_peak_bytes = max(
                    self.host_workspace_peak_bytes, host_bytes
                )
                lease = self.resource_budget.acquire(
                    "exact tiled GPU kNN source-target tile", max(1, host_bytes)
                )
                tile_started = time.perf_counter()
                target_locations = None
                target_values = None
                reported_indices = None
                reported_distances = None
                try:
                    if gpu_sources is None:
                        gpu_sources = self._upload_source_batch(source_values)
                    target_locations = np.arange(
                        target_start, target_stop, dtype=np.int64
                    )
                    target_values = self.unique_locations[target_locations]
                    assert target_values is not None
                    reported_indices, reported_distances = self._gpu_top_candidates(
                        source_values,
                        target_values,
                        keep_count,
                        gpu_sources=gpu_sources,
                    )
                    assert (
                        target_locations is not None
                        and reported_indices is not None
                        and reported_distances is not None
                    )
                    certification_started = time.perf_counter()
                    for offset, source_location in enumerate(
                        range(source_start_tile, source_stop)
                    ):
                        reported_locations = target_locations[reported_indices[offset]]
                        best_locations[offset], best_distances[offset] = self._certify_tile(
                            source_values[offset],
                            source_location,
                            target_locations,
                            reported_locations,
                            reported_distances[offset],
                            best_locations[offset],
                            best_distances[offset],
                            order_k,
                        )
                    self.certification_seconds += max(
                        0.0, time.perf_counter() - certification_started
                    )
                finally:
                    self.gpu_tile_wall_seconds += max(
                        0.0, time.perf_counter() - tile_started
                    )
                    lease.close()
                    target_locations = None
                    target_values = None
                    reported_indices = None
                    reported_distances = None
                self.target_tiles += 1
                self.distance_evaluations += batch_count * target_count
                self.unique_distance_evaluations += batch_count * target_count
            del gpu_sources
            for offset, source_location in enumerate(range(source_start_tile, source_stop)):
                if best_locations[offset].size != order_k:
                    raise GPUCertificationError(
                        "GPU kNN exact reduction returned an incomplete neighbour row"
                    )
                neighbours[source_location] = best_locations[offset]
                distances_out[source_location] = best_distances[offset]
            self.index_queries += batch_count
            finish_checkpoint(source_stop)
        if incremental_store is not None and not preserve_checkpoint:
            incremental_store.clear(checkpoint_name)
            incremental_store.clean_stale()
            if legacy_store is not None:
                legacy_store.clear(checkpoint_name)
        neighbours.setflags(write=False)
        distances_out.setflags(write=False)
        self._last_neighbour_compute_seconds = max(0.0, time.perf_counter() - started)
        self._store_neighbour_cache(order_k, neighbours, distances_out)
        return neighbours, distances_out

    def memory_metrics(self) -> dict[str, Any]:
        metrics = super().memory_metrics()
        metrics.update(
            {
                "backend": self.backend,
                "backend_version": self.backend_version,
                "gpu_device": int(self.device),
                "gpu_tile_size": int(self.requested_tile_size),
                "gpu_max_device_bytes": int(self.max_device_bytes),
                "gpu_total_bytes": int(self.device_total_bytes),
                "gpu_free_bytes": int(self.device_free_bytes),
                "gpu_source_tiles": int(self.source_tiles),
                "gpu_target_tiles": int(self.target_tiles),
                "gpu_fp64_evaluations": int(self.fp64_evaluations),
                "gpu_h2d_bytes": int(self.h2d_bytes),
                "gpu_d2h_bytes": int(self.d2h_bytes),
                "gpu_device_workspace_peak_bytes": int(self.device_workspace_peak_bytes),
                "gpu_host_workspace_peak_bytes": int(self.host_workspace_peak_bytes),
                "gpu_kernel_seconds": float(self.gpu_seconds),
                "gpu_distance_reduction_seconds": float(
                    self.gpu_distance_reduction_seconds
                ),
                "gpu_operation_seconds_inclusive": float(self.gpu_operation_seconds),
                "gpu_h2d_seconds": float(self.gpu_h2d_seconds),
                "gpu_d2h_seconds": float(self.gpu_d2h_seconds),
                "gpu_synchronization_seconds": float(self.gpu_synchronization_seconds),
                "gpu_launch_allocation_seconds": float(
                    self.gpu_launch_allocation_seconds
                ),
                "gpu_tile_wall_seconds": float(self.gpu_tile_wall_seconds),
                "gpu_kernel_seconds_definition": (
                    "inclusive FP64 distance/reduction section wall time, including its "
                    "synchronization and excluding measured transfers; not a hardware "
                    "kernel-only timestamp"
                ),
                "gpu_operation_seconds_definition": (
                    "inclusive per-target-tile CuPy wall time, including target transfers; "
                    "source-batch upload is included in gpu_h2d_seconds and tile wall time"
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
                "gpu_certified_tiles": int(self.certified_tiles),
                "gpu_certification_relative_bound": float(
                    self.certification_relative_bound
                ),
                "gpu_certification_contract": "fp64-gamma-bound-v1",
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
        raise ValueError(
            "exact GPU neighbour result exceeds its explicit bounded-memory limit"
        )
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
    return (
        max_gpu_neighbour_bytes
        if max_gpu_neighbour_bytes is not None
        else max_device_bytes
    )


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
    service = resource_budget or (
        index.resource_budget if index is not None else None
    )
    if service is None:
        from nepflow.resources.budget import build_resource_budget

        service = build_resource_budget()
    _validate_gpu_result_limits(
        values, order_k, max_neighbour_entries, service.remaining_managed_budget
    )
    active_index = index or ExactTiledGPUNeighbourIndex(
        values,
        device=gpu_device,
        gpu_tile_size=gpu_tile_size,
        max_device_bytes=_resolve_device_limit(
            max_device_bytes, max_gpu_neighbour_bytes
        ),
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
        gpu_tile_size=gpu_tile_size,
        max_device_bytes=_resolve_device_limit(
            max_device_bytes, max_gpu_neighbour_bytes
        ),
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
