"""Optional, certified CuPy execution for exact compact-kernel supports.

This module intentionally has no CuPy import at module scope.  The CPU
selection path therefore remains importable on installations without CUDA.
CuPy is used only to screen bounded source/target tiles.  Distances retained
by the scientific operator are always recomputed by the canonical NumPy
float64 primitive before they are returned.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from nepflow.resources.budget import ResourceBudgetService, ResourceCapacityError

from .gpu_devices import (
    contiguous_gpu_ranges,
    normalize_gpu_devices,
    validate_visible_gpu_devices,
)
from .models import KERNEL_SUPPORT_EXECUTION_GPU_VERSION


class GPUSupportError(RuntimeError):
    """Base error for an explicitly requested GPU support execution."""


class GPUDeviceValidationError(GPUSupportError):
    """The requested CUDA device is unavailable or not an ordinal."""


class GPUCertificationError(GPUSupportError):
    """A GPU result could not be certified against the CPU contract."""


class GPUCapacityError(ResourceCapacityError, GPUSupportError):
    """A device or host tile cannot fit the declared bounded workspace."""


def _canonical_distances(
    descriptors: np.ndarray, source_index: int, target_indices: np.ndarray
) -> np.ndarray:
    """Return the canonical NumPy float64 distances used by the CPU backend."""

    source = descriptors[int(source_index)]
    targets = descriptors[np.asarray(target_indices, dtype=np.int64)]
    with np.errstate(over="raise", invalid="raise"):
        try:
            distances = np.sqrt(np.sum((targets - source) ** 2, axis=1))
        except FloatingPointError as exc:
            raise GPUCertificationError(
                "canonical CPU distance verification overflowed or became invalid"
            ) from exc
    if not np.all(np.isfinite(distances)):
        raise GPUCertificationError("canonical CPU distance verification was non-finite")
    return np.asarray(distances, dtype=np.float64)


def _validate_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    integer = int(value)
    if integer < 1:
        raise ValueError(f"{name} must be a positive integer")
    return integer


class ExactTiledGPUSupport:
    """Bounded one-device GPU screening with canonical CPU certification.

    The interface is deliberately source-tile oriented so a future multi-GPU
    provider can partition this object without changing the kernel callers.
    ``support_many`` retains original row indices and returns source-major CSR
    arrays in the order requested by the caller.
    """

    backend = "exact_tiled_gpu"
    backend_version = KERNEL_SUPPORT_EXECUTION_GPU_VERSION
    metric = "euclidean"

    def __init__(
        self,
        descriptors: np.ndarray,
        *,
        device: int = 0,
        gpu_devices: Sequence[int] | str | int | None = None,
        tile_size: int = 128,
        max_device_bytes: int | None = None,
        max_host_bytes: int | None = None,
        resource_budget: ResourceBudgetService | None = None,
        capacity_context: str | None = None,
    ) -> None:
        values = np.asarray(descriptors)
        if values.dtype != np.dtype(np.float64) or values.ndim != 2 or values.size == 0:
            raise ValueError("GPU support descriptors must be a non-empty float64 matrix")
        if not np.all(np.isfinite(values)):
            raise ValueError("GPU support descriptors must be finite")
        if isinstance(device, bool) or not isinstance(device, (int, np.integer)):
            raise ValueError("gpu_device must be a non-negative integer")
        if int(device) < 0:
            raise ValueError("gpu_device must be a non-negative integer")
        self.descriptors = np.ascontiguousarray(values, dtype=np.float64)
        self.gpu_devices = normalize_gpu_devices(gpu_devices, gpu_device=int(device))
        self.device = int(self.gpu_devices[0])
        self.requested_tile_size = _validate_positive_int(tile_size, "gpu_tile_size")
        self.resource_budget = resource_budget
        self.capacity_context = capacity_context
        self._cp = self._load_cupy()
        self._validate_device()

        # Construct every requested worker before the first query.  This makes
        # a missing device or insufficient per-device capacity fail atomically
        # and keeps the readonly descriptor pool complete on every worker.
        self._workers: tuple[ExactTiledGPUSupport, ...] = ()
        if len(self.gpu_devices) > 1:
            worker_list: list[ExactTiledGPUSupport] = []
            try:
                for worker_device in self.gpu_devices:
                    worker_list.append(
                        ExactTiledGPUSupport(
                            self.descriptors,
                            device=worker_device,
                            tile_size=tile_size,
                            max_device_bytes=max_device_bytes,
                            max_host_bytes=max_host_bytes,
                            resource_budget=resource_budget,
                            capacity_context=capacity_context,
                        )
                    )
            except BaseException:
                for worker in worker_list:
                    worker.close()
                raise
            self._workers = tuple(worker_list)
            self.source_tile_size = min(worker.source_tile_size for worker in self._workers)
            self.max_device_bytes = max(worker.max_device_bytes for worker in self._workers)
            self.max_host_bytes = max(worker.max_host_bytes for worker in self._workers)
            self.device_total_bytes = max(worker.device_total_bytes for worker in self._workers)
            self.device_free_bytes = max(worker.device_free_bytes for worker in self._workers)
            self.device_workspace_peak_bytes = 0
            self.host_workspace_peak_bytes = 0
            self.source_tiles = 0
            self.target_tiles = 0
            self.fp64_evaluations = 0
            self.gpu_launch_count = 0
            self.d2h_transfer_count = 0
            self.ambiguous_boundary_count = 0
            self.cpu_rescanned_count = 0
            self.h2d_bytes = 0
            self.d2h_bytes = 0
            self.h2d_seconds = 0.0
            self.d2h_seconds = 0.0
            self.gpu_seconds = 0.0
            self.cpu_verification_seconds = 0.0
            self.support_query_count = 0
            self.support_entries = 0
            self.grouped_radius_query_count = 0
            self.reused_radius_scans = 0
            self.conservative_candidate_count = 0
            self.support_query_seconds = 0.0
            self.support_wall_seconds = 0.0
            return

        free_bytes, total_bytes = self._device_memory_info()
        if total_bytes < 1 or free_bytes < 1:
            raise GPUDeviceValidationError(f"CUDA device {self.device} reported no usable memory")
        # These fields are part of the worker availability contract and are
        # also consumed by the aggregate multi-device provider.
        self.device_free_bytes = int(free_bytes)
        self.device_total_bytes = int(total_bytes)
        if max_device_bytes is not None:
            self.max_device_bytes = min(
                _validate_positive_int(max_device_bytes, "max_gpu_support_bytes"),
                free_bytes,
            )
        else:
            # The provider owns only a bounded workspace and leaves headroom
            # for the CUDA runtime and unrelated allocations.
            self.max_device_bytes = max(1, int(free_bytes * 0.80))
        if self.max_device_bytes < 40:
            raise GPUCapacityError(
                "GPU support tile requires at least 40 bytes of device workspace: "
                f"device={self.device}, max_gpu_support_bytes={self.max_device_bytes}",
                operation="exact tiled GPU support",
                requested_bytes=40,
                available_bytes=self.max_device_bytes,
            )

        remaining_host = (
            None if resource_budget is None else resource_budget.remaining_managed_budget
        )
        if remaining_host is None and max_host_bytes is None:
            raise GPUCapacityError(
                "exact tiled GPU support requires a known host memory allowance",
                operation="exact tiled GPU support",
            )
        host_limit = max_host_bytes if max_host_bytes is not None else remaining_host
        if remaining_host is not None:
            if host_limit is None:  # narrowed by the preceding allowance check
                raise GPUCapacityError(
                    "exact tiled GPU support has no allocatable host workspace",
                    operation="exact tiled GPU support",
                    requested_bytes=1,
                    available_bytes=remaining_host,
                )
            host_limit = min(int(host_limit), int(remaining_host))
        if host_limit is None:
            raise GPUCapacityError(
                "exact tiled GPU support has no allocatable host workspace",
                operation="exact tiled GPU support",
                requested_bytes=1,
                available_bytes=host_limit,
            )
        if int(host_limit) < 1:
            raise GPUCapacityError(
                "exact tiled GPU support has no allocatable host workspace",
                operation="exact tiled GPU support",
                requested_bytes=1,
                available_bytes=host_limit,
            )
        self.max_host_bytes = int(host_limit)
        rows, dimension = self.descriptors.shape
        per_source = max(48 * rows + 16 * dimension + 32768, 16 * rows + 1)
        self.source_tile_size = min(
            self.requested_tile_size,
            max(1, self.max_host_bytes // per_source),
        )
        if self.source_tile_size < 1 or per_source > self.max_host_bytes:
            raise GPUCapacityError(
                "one exact GPU source support row exceeds the host workspace bound: "
                f"rows={rows}, requested_bytes={per_source}, "
                f"available_bytes={self.max_host_bytes}",
                operation="exact tiled GPU support",
                requested_bytes=per_source,
                available_bytes=self.max_host_bytes,
            )
        device_source_tile_size, _, _ = self._tile_shape(self.source_tile_size)
        self.source_tile_size = min(self.source_tile_size, device_source_tile_size)

        self.source_tiles = 0
        self.target_tiles = 0
        self.fp64_evaluations = 0
        self.gpu_launch_count = 0
        self.d2h_transfer_count = 0
        self.ambiguous_boundary_count = 0
        self.cpu_rescanned_count = 0
        self.h2d_bytes = 0
        self.d2h_bytes = 0
        self.h2d_seconds = 0.0
        self.d2h_seconds = 0.0
        self.device_workspace_peak_bytes = 0
        self.host_workspace_peak_bytes = 0
        self.gpu_seconds = 0.0
        self.cpu_verification_seconds = 0.0
        self.support_query_count = 0
        self.support_entries = 0
        self.grouped_radius_query_count = 0
        self.reused_radius_scans = 0
        self.conservative_candidate_count = 0
        self.support_query_seconds = 0.0
        self.support_wall_seconds = 0.0

    def _load_cupy(self) -> Any:
        try:
            import cupy as cp  # type: ignore[import-not-found]
        except ImportError as exc:
            raise GPUSupportError(
                "kernel_support_execution='exact_tiled_gpu' requires optional CuPy"
            ) from exc
        return cp

    def _validate_device(self) -> None:
        try:
            validate_visible_gpu_devices(self._cp, self.gpu_devices)
        except Exception as exc:  # pragma: no cover - depends on CUDA runtime
            raise GPUDeviceValidationError(
                f"requested gpu_devices={self.gpu_devices} (gpu_device={self.device}) "
                f"are not visible: {exc}"
            ) from exc

    def _device_memory_info(self) -> tuple[int, int]:
        try:
            with self._cp.cuda.Device(self.device):
                free_bytes, total_bytes = self._cp.cuda.runtime.memGetInfo()
        except Exception as exc:  # pragma: no cover - depends on CUDA runtime
            raise GPUDeviceValidationError(
                f"unable to inspect CUDA device {self.device} memory: {exc}"
            ) from exc
        return int(free_bytes), int(total_bytes)

    def _tile_shape(self, source_count: int) -> tuple[int, int, int]:
        dimension = int(self.descriptors.shape[1])
        source_count = max(1, min(int(source_count), self.requested_tile_size))
        target_count = min(self.requested_tile_size, int(self.descriptors.shape[0]))

        def required(sources: int, targets: int) -> int:
            # dist2, sqrt output, square/delta temporaries, plus transferred
            # descriptor tiles.  The dimension loop never creates an N x N x d
            # array, and this estimate reserves its largest temporary.
            return int(40 * sources * targets + 8 * (sources + targets) * dimension)

        while required(source_count, target_count) > self.max_device_bytes:
            if target_count > 1:
                target_count = max(1, target_count // 2)
            elif source_count > 1:
                source_count = max(1, source_count // 2)
            else:
                needed = required(source_count, target_count)
                raise GPUCapacityError(
                    "exact GPU support cannot fit one source/target tile: "
                    f"requested_bytes={needed}, available_bytes={self.max_device_bytes}",
                    operation="exact tiled GPU support",
                    requested_bytes=needed,
                    available_bytes=self.max_device_bytes,
                )
        return source_count, target_count, required(source_count, target_count)

    def _screen_tile(
        self,
        source_indices: np.ndarray,
        target_indices: np.ndarray,
        radii: np.ndarray,
    ) -> np.ndarray:
        cp = self._cp
        source_count, target_count, reserved_bytes = self._tile_shape(source_indices.size)
        del source_count, target_count
        source_values = np.ascontiguousarray(self.descriptors[source_indices])
        target_values = np.ascontiguousarray(self.descriptors[target_indices])
        self.h2d_bytes += int(source_values.nbytes + target_values.nbytes)
        gpu_started = time.perf_counter()
        with cp.cuda.Device(self.device):
            transfer_started = time.perf_counter()
            gpu_sources = cp.asarray(source_values, dtype=cp.float64)
            gpu_targets = cp.asarray(target_values, dtype=cp.float64)
            self.h2d_seconds += max(0.0, time.perf_counter() - transfer_started)
            distance_squared = cp.zeros(
                (source_indices.size, target_indices.size), dtype=cp.float64
            )
            for dimension in range(self.descriptors.shape[1]):
                delta = gpu_targets[None, :, dimension] - gpu_sources[:, None, dimension]
                distance_squared += delta * delta
            gpu_distances = cp.sqrt(distance_squared)
            transfer_started = time.perf_counter()
            host_distances = np.asarray(cp.asnumpy(gpu_distances), dtype=np.float64)
            self.d2h_seconds += max(0.0, time.perf_counter() - transfer_started)
        self.gpu_seconds += max(0.0, time.perf_counter() - gpu_started)
        self.gpu_launch_count += 1
        self.d2h_transfer_count += 1
        self.d2h_bytes += int(host_distances.nbytes)
        self.fp64_evaluations += int(source_indices.size * target_indices.size)
        self.device_workspace_peak_bytes = max(self.device_workspace_peak_bytes, reserved_bytes)

        # This is a conservative absolute-error enclosure for subtraction,
        # squaring, summation and square-root in IEEE float64.  If scaling the
        # enclosure is itself unsafe, every candidate is CPU certified rather
        # than making an uncertified decision.
        source_norm = np.sqrt(np.sum(np.abs(source_values) ** 2, axis=1, dtype=np.float64))
        target_norm = np.sqrt(np.sum(np.abs(target_values) ** 2, axis=1, dtype=np.float64))
        scale = np.maximum(1.0, source_norm[:, None] + target_norm[None, :])
        with np.errstate(over="ignore", invalid="ignore"):
            bound = np.finfo(np.float64).eps * (1024.0 + 32.0 * self.descriptors.shape[1]) * scale
        bound = np.asarray(bound, dtype=np.float64)
        bound[~np.isfinite(bound)] = np.inf
        lower = np.maximum(0.0, host_distances - bound)
        potential = lower < radii[:, None]
        if not np.all(np.isfinite(host_distances)):
            # The canonical CPU primitive is authoritative for overflow or
            # invalid device results; force certification rather than allowing
            # a NaN comparison to become an exclusion.
            potential = np.ones_like(host_distances, dtype=bool)
        self.ambiguous_boundary_count += int(
            np.count_nonzero(np.abs(host_distances - radii[:, None]) <= bound)
        )
        self.conservative_candidate_count += int(np.count_nonzero(potential))
        self.host_workspace_peak_bytes = max(
            self.host_workspace_peak_bytes,
            int(source_values.nbytes + target_values.nbytes + host_distances.nbytes + bound.nbytes),
        )
        return potential

    def support_many(
        self,
        source_indices: Sequence[int] | np.ndarray,
        radii: Sequence[float] | np.ndarray,
        *,
        _record_group: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return an exact source-major CSR support superset.

        ``radii`` may be either ``(source_count,)`` for the historical
        single-radius query or ``(candidate_count, source_count)``.  For the
        latter, one GPU distance pass is screened with the largest radius for
        each source and canonical CPU distances are returned for that finite
        superset.  Callers reconstruct each candidate's strict support with
        ``distances < radii[candidate]``; this preserves duplicate rows,
        source self-inclusion, and the original target order without retaining
        a C-by-N-by-N workspace.
        """

        sources = np.asarray(source_indices, dtype=np.int64)
        requested_radii = np.asarray(radii, dtype=np.float64)
        if sources.ndim != 1 or requested_radii.ndim not in (1, 2):
            raise ValueError(
                "source_indices must be one-dimensional and radii must have shape "
                "(source_count,) or (candidate_count, source_count)"
            )
        if requested_radii.shape[-1] != sources.size:
            raise ValueError("source_indices and radii must have aligned source dimensions")
        if requested_radii.ndim == 2 and requested_radii.shape[0] == 0:
            raise ValueError("support_many requires at least one radius candidate")
        scales = (
            requested_radii
            if requested_radii.ndim == 1
            else np.max(requested_radii, axis=0)
        )
        if sources.size == 0:
            raise ValueError("support_many requires at least one source")
        if np.any(sources < 0) or np.any(sources >= self.descriptors.shape[0]):
            raise ValueError("source_indices contains an out-of-range row")
        if not np.all(np.isfinite(requested_radii)) or np.any(requested_radii <= 0.0):
            raise ValueError("GPU support radii must be finite and strictly positive")
        if _record_group and requested_radii.ndim == 2 and requested_radii.shape[0] > 1:
            self.grouped_radius_query_count += 1
            self.reused_radius_scans += max(0, int(requested_radii.shape[0]) - 1)
        if self._workers:
            started = time.perf_counter()
            target_parts: list[np.ndarray] = []
            distance_parts: list[np.ndarray] = []
            offsets = np.zeros(sources.size + 1, dtype=np.int64)
            ranges = contiguous_gpu_ranges(int(sources.size), self.gpu_devices)
            executor = ThreadPoolExecutor(max_workers=len(self._workers))
            futures: dict[int, Any] = {}
            try:
                # One thread owns one explicitly selected worker.  Every
                # worker enters its own CUDA context for all CuPy operations;
                # no peer access or cross-device array operation is needed.
                for worker_index, (_device, start, stop) in enumerate(ranges):
                    if start == stop:
                        continue
                    futures[worker_index] = executor.submit(
                        self._workers[worker_index]._support_many_on_device,
                        sources[start:stop],
                        requested_radii[:, start:stop]
                        if requested_radii.ndim == 2
                        else scales[start:stop],
                    )
                for worker_index, (_device, start, stop) in enumerate(ranges):
                    if start == stop:
                        continue
                    worker_offsets, worker_targets, worker_distances = futures[
                        worker_index
                    ].result()
                    target_parts.append(worker_targets)
                    distance_parts.append(worker_distances)
                    offsets[start + 1 : stop + 1] = worker_offsets[1:] + offsets[start]
            except BaseException:
                for future in futures.values():
                    future.cancel()
                raise
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
            self.support_wall_seconds += max(0.0, time.perf_counter() - started)
            targets = (
                np.asarray(np.concatenate(target_parts), dtype=np.int64)
                if target_parts
                else np.empty(0, dtype=np.int64)
            )
            distances = (
                np.asarray(np.concatenate(distance_parts), dtype=np.float64)
                if distance_parts
                else np.empty(0, dtype=np.float64)
            )
            self._aggregate_worker_metrics()
            return offsets, targets, distances
        device_source_tile_size, _, _ = self._tile_shape(sources.size)
        source_tile_limit = min(self.source_tile_size, device_source_tile_size)
        if sources.size > source_tile_limit:
            requested_host = int(
                48 * sources.size * self.descriptors.shape[0]
                + 16 * sources.size * self.descriptors.shape[1]
                + 32768
            )
            if requested_host > self.max_host_bytes:
                raise GPUCapacityError(
                    "GPU support source batch exceeds its bounded host output allowance: "
                    f"requested_sources={sources.size}, source_tile_size={source_tile_limit}, "
                    f"requested_bytes={requested_host}, available_bytes={self.max_host_bytes}",
                    operation="exact tiled GPU support",
                    requested_bytes=requested_host,
                    available_bytes=self.max_host_bytes,
                )
            target_parts: list[np.ndarray] = []
            distance_parts: list[np.ndarray] = []
            offsets = np.zeros(sources.size + 1, dtype=np.int64)
            for start in range(0, sources.size, source_tile_limit):
                stop = min(start + source_tile_limit, sources.size)
                tile_offsets, tile_targets, tile_distances = self.support_many(
                    sources[start:stop],
                    requested_radii[:, start:stop]
                    if requested_radii.ndim == 2
                    else scales[start:stop],
                    _record_group=False,
                )
                target_parts.append(tile_targets)
                distance_parts.append(tile_distances)
                offsets[start + 1 : stop + 1] = tile_offsets[1:] + offsets[start]
            return (
                offsets,
                np.asarray(np.concatenate(target_parts), dtype=np.int64),
                np.asarray(np.concatenate(distance_parts), dtype=np.float64),
            )
        started = time.perf_counter()
        offsets = np.zeros(sources.size + 1, dtype=np.int64)
        target_count = int(self.descriptors.shape[0])
        with (
            self.resource_budget.acquire(
                "exact tiled GPU support",
                min(
                    self.max_host_bytes,
                    max(
                        1,
                        48 * sources.size * target_count
                        + 16 * sources.size * self.descriptors.shape[1]
                        + 32768,
                    ),
                ),
            )
            if self.resource_budget is not None
            else _NullLease()
        ):
            source_targets: list[list[np.ndarray]] = [[] for _ in range(sources.size)]
            source_distances: list[list[np.ndarray]] = [[] for _ in range(sources.size)]
            _, target_tile_size, _ = self._tile_shape(sources.size)
            target_cursor = 0
            while target_cursor < target_count:
                target_stop = min(target_cursor + target_tile_size, target_count)
                target_tile = np.arange(target_cursor, target_stop, dtype=np.int64)
                potential = self._screen_tile(sources, target_tile, scales)
                for local_source, source_index in enumerate(sources):
                    candidates = target_tile[potential[local_source]]
                    if candidates.size == 0:
                        continue
                    verification_started = time.perf_counter()
                    canonical = _canonical_distances(
                        self.descriptors, int(source_index), candidates
                    )
                    self.cpu_verification_seconds += max(
                        0.0, time.perf_counter() - verification_started
                    )
                    self.cpu_rescanned_count += int(candidates.size)
                    keep = canonical < float(scales[local_source])
                    source_targets[local_source].append(candidates[keep])
                    source_distances[local_source].append(canonical[keep])
                self.target_tiles += 1
                target_cursor = target_stop
            for local_source in range(sources.size):
                targets = (
                    np.concatenate(source_targets[local_source]).astype(np.int64, copy=False)
                    if source_targets[local_source]
                    else np.empty(0, dtype=np.int64)
                )
                distances = (
                    np.concatenate(source_distances[local_source]).astype(np.float64, copy=False)
                    if source_distances[local_source]
                    else np.empty(0, dtype=np.float64)
                )
                target_parts_size = int(targets.size)
                offsets[local_source + 1] = offsets[local_source] + target_parts_size
                if targets.size and np.any(np.diff(targets) <= 0):
                    raise GPUCertificationError(
                        "GPU support rows were not produced in deterministic target order"
                    )
                # Keep the source-local arrays in a compact list until the
                # final bounded CSR materialisation below.
                source_targets[local_source] = [targets]
                source_distances[local_source] = [distances]
            targets = (
                np.asarray(np.concatenate([parts[0] for parts in source_targets]), dtype=np.int64)
                if sources.size
                else np.empty(0, dtype=np.int64)
            )
            distances = (
                np.asarray(
                    np.concatenate([parts[0] for parts in source_distances]), dtype=np.float64
                )
                if sources.size
                else np.empty(0, dtype=np.float64)
            )
            self.host_workspace_peak_bytes = max(
                self.host_workspace_peak_bytes,
                int(targets.nbytes + distances.nbytes),
            )
        self.source_tiles += 1
        self.support_query_count += int(sources.size)
        self.support_entries += int(targets.size)
        self.support_query_seconds += max(0.0, time.perf_counter() - started)
        return offsets, targets, distances

    def _support_many_on_device(
        self,
        source_indices: np.ndarray,
        radii: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run one single-device worker while its CUDA context is current."""

        with self._cp.cuda.Device(self.device):
            return self.support_many(source_indices, radii, _record_group=False)

    def operational_metrics(self) -> dict[str, Any]:
        if self._workers:
            self._aggregate_worker_metrics()
            per_device = [worker.operational_metrics() for worker in self._workers]
            aggregate = {
                "support_execution": self.backend,
                "support_execution_version": self.backend_version,
                "gpu_devices": list(self.gpu_devices),
                "gpu_device_count": len(self.gpu_devices),
                "gpu_device": self.device,
                "gpu_tile_size": self.requested_tile_size,
                "gpu_max_device_bytes": self.max_device_bytes,
                "gpu_total_bytes": self.device_total_bytes,
                "gpu_free_bytes": self.device_free_bytes,
                "gpu_source_tile_size": self.source_tile_size,
                "gpu_device_workspace_peak_bytes": self.device_workspace_peak_bytes,
                "gpu_host_workspace_peak_bytes": self.host_workspace_peak_bytes,
                "gpu_source_tiles": self.source_tiles,
                "gpu_target_tiles": self.target_tiles,
                "gpu_fp64_evaluations": self.fp64_evaluations,
                "gpu_launch_count": self.gpu_launch_count,
                "gpu_d2h_transfer_count": self.d2h_transfer_count,
                "gpu_ambiguous_boundary_count": self.ambiguous_boundary_count,
                "gpu_exact_boundary_recertifications": self.ambiguous_boundary_count,
                "gpu_cpu_rescanned_count": self.cpu_rescanned_count,
                "gpu_conservative_candidate_count": self.conservative_candidate_count,
                "gpu_grouped_radius_query_count": self.grouped_radius_query_count,
                "gpu_reused_radius_scans": self.reused_radius_scans,
                "gpu_h2d_bytes": self.h2d_bytes,
                "gpu_d2h_bytes": self.d2h_bytes,
                "gpu_h2d_seconds": self.h2d_seconds,
                "gpu_d2h_seconds": self.d2h_seconds,
                "gpu_seconds": self.gpu_seconds,
                "gpu_strict_radius_screening_seconds": self.gpu_seconds,
                "gpu_active_seconds_sum": float(self.gpu_seconds),
                "gpu_wall_seconds": float(self.support_wall_seconds),
                "gpu_parallelism_estimate": (
                    self.gpu_seconds / self.support_wall_seconds
                    if self.support_wall_seconds > 0.0
                    else 0.0
                ),
                "gpu_cpu_verification_seconds": self.cpu_verification_seconds,
                "support_query_count": self.support_query_count,
                "support_entries": self.support_entries,
                "support_query_seconds": self.support_query_seconds,
                "support_query_wall_seconds": self.support_wall_seconds,
                "support_query_throughput_per_second": (
                    self.support_query_count / self.support_query_seconds
                    if self.support_query_seconds > 0.0
                    else 0.0
                ),
                "gpu_per_device": per_device,
            }
            return aggregate
        free_bytes, total_bytes = self._device_memory_info()
        metrics = {
            "support_execution": self.backend,
            "support_execution_version": self.backend_version,
            "gpu_device": self.device,
            "gpu_devices": [self.device],
            "gpu_device_count": 1,
            "gpu_tile_size": self.requested_tile_size,
            "gpu_max_device_bytes": self.max_device_bytes,
            "gpu_free_bytes": free_bytes,
            "gpu_total_bytes": total_bytes,
            "gpu_device_workspace_peak_bytes": self.device_workspace_peak_bytes,
            "gpu_device_occupancy_fraction": (
                self.device_workspace_peak_bytes / self.max_device_bytes
                if self.max_device_bytes
                else 0.0
            ),
            "gpu_source_tiles": self.source_tiles,
            "gpu_target_tiles": self.target_tiles,
            "gpu_fp64_evaluations": self.fp64_evaluations,
            "gpu_launch_count": self.gpu_launch_count,
            "gpu_d2h_transfer_count": self.d2h_transfer_count,
            "gpu_ambiguous_boundary_count": self.ambiguous_boundary_count,
            "gpu_exact_boundary_recertifications": self.ambiguous_boundary_count,
            "gpu_cpu_rescanned_count": self.cpu_rescanned_count,
            "gpu_conservative_candidate_count": self.conservative_candidate_count,
            "gpu_grouped_radius_query_count": self.grouped_radius_query_count,
            "gpu_reused_radius_scans": self.reused_radius_scans,
            "gpu_h2d_bytes": self.h2d_bytes,
            "gpu_d2h_bytes": self.d2h_bytes,
            "gpu_h2d_seconds": self.h2d_seconds,
            "gpu_d2h_seconds": self.d2h_seconds,
            "gpu_seconds": self.gpu_seconds,
            "gpu_strict_radius_screening_seconds": self.gpu_seconds,
            "gpu_active_seconds_sum": float(self.gpu_seconds),
            "gpu_wall_seconds": float(self.support_query_seconds),
            "gpu_parallelism_estimate": 1.0 if self.support_query_seconds > 0.0 else 0.0,
            "gpu_cpu_verification_seconds": self.cpu_verification_seconds,
            "gpu_host_workspace_peak_bytes": self.host_workspace_peak_bytes,
            "support_query_count": self.support_query_count,
            "support_entries": self.support_entries,
            "support_query_seconds": self.support_query_seconds,
            "support_query_wall_seconds": self.support_query_seconds,
            "support_query_throughput_per_second": (
                self.support_query_count / self.support_query_seconds
                if self.support_query_seconds > 0.0
                else 0.0
            ),
        }
        metrics["gpu_per_device"] = [dict(metrics)]
        return metrics

    def _aggregate_worker_metrics(self) -> None:
        """Refresh bounded aggregate counters after a multi-device query."""

        if not self._workers:
            return
        metrics = [worker.operational_metrics() for worker in self._workers]
        for name in (
            "source_tiles",
            "target_tiles",
            "fp64_evaluations",
            "gpu_launch_count",
            "d2h_transfer_count",
            "ambiguous_boundary_count",
            "cpu_rescanned_count",
            "conservative_candidate_count",
            "h2d_bytes",
            "d2h_bytes",
            "support_query_count",
            "support_entries",
        ):
            setattr(
                self,
                name,
                sum(
                    int(
                        item.get(
                            {
                                "source_tiles": "gpu_source_tiles",
                                "target_tiles": "gpu_target_tiles",
                                "fp64_evaluations": "gpu_fp64_evaluations",
                                "gpu_launch_count": "gpu_launch_count",
                                "d2h_transfer_count": "gpu_d2h_transfer_count",
                                "ambiguous_boundary_count": "gpu_ambiguous_boundary_count",
                                "cpu_rescanned_count": "gpu_cpu_rescanned_count",
                                "conservative_candidate_count": "gpu_conservative_candidate_count",
                                "h2d_bytes": "gpu_h2d_bytes",
                                "d2h_bytes": "gpu_d2h_bytes",
                                "support_query_count": "support_query_count",
                                "support_entries": "support_entries",
                            }[name],
                            0,
                        )
                        or 0
                    )
                    for item in metrics
                ),
            )
        self.h2d_seconds = sum(
            float(item.get("gpu_h2d_seconds", 0.0) or 0.0) for item in metrics
        )
        self.d2h_seconds = sum(
            float(item.get("gpu_d2h_seconds", 0.0) or 0.0) for item in metrics
        )
        self.device_workspace_peak_bytes = max(
            int(item.get("gpu_device_workspace_peak_bytes", 0) or 0) for item in metrics
        )
        self.host_workspace_peak_bytes = max(
            int(item.get("gpu_host_workspace_peak_bytes", 0) or 0) for item in metrics
        )
        self.gpu_seconds = sum(float(item.get("gpu_seconds", 0.0) or 0.0) for item in metrics)
        self.cpu_verification_seconds = sum(
            float(item.get("gpu_cpu_verification_seconds", 0.0) or 0.0) for item in metrics
        )
        self.support_query_seconds = sum(
            float(item.get("support_query_seconds", 0.0) or 0.0) for item in metrics
        )

    def close(self) -> None:
        """Release device-pool allocations held by this provider and workers."""

        for worker in self._workers:
            worker.close()
        try:
            with self._cp.cuda.Device(self.device):
                for name in ("get_default_memory_pool", "get_default_pinned_memory_pool"):
                    pool_factory = getattr(self._cp, name, None)
                    if pool_factory is not None:
                        pool_factory().free_all_blocks()
        except Exception:
            pass


class _NullLease:
    def __enter__(self) -> "_NullLease":
        return self

    def __exit__(self, *_args: object) -> None:
        return None
