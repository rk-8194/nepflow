"""Explicit CUDA device-list validation and deterministic range scheduling.

The information-entropy GPU backends deliberately treat device ordinals as
*process-visible* ordinals.  In a Slurm job this is normally the ordinal after
``CUDA_VISIBLE_DEVICES`` remapping, never a physical node ordinal.  Keeping the
policy in one small module prevents the kNN and support providers from
silently applying different allocation rules.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from numbers import Integral
from typing import Any


def normalize_gpu_devices(
    gpu_devices: Sequence[int] | str | int | None = None,
    *,
    gpu_device: int = 0,
) -> tuple[int, ...]:
    """Return a validated, ordered, unique explicit visible-device list.

    ``gpu_device`` is retained as the compatibility spelling for one GPU.  A
    supplied list is authoritative; a non-default legacy ordinal combined
    with that list is rejected instead of being silently ignored.
    """

    if isinstance(gpu_device, bool) or not isinstance(gpu_device, Integral):
        raise ValueError("gpu_device must be a non-negative integer")
    if gpu_devices is None:
        values: list[Any] = [gpu_device]
    elif isinstance(gpu_devices, str):
        raw = [part.strip() for part in gpu_devices.split(",")]
        if not raw or any(not part for part in raw):
            raise ValueError("gpu_devices must be a comma-separated list of ordinals")
        values = raw
    elif isinstance(gpu_devices, Integral) and not isinstance(gpu_devices, bool):
        values = [gpu_devices]
        if gpu_device != 0 and int(gpu_device) != int(gpu_devices):
            raise ValueError("gpu_device and gpu_devices specify different devices")
    else:
        try:
            values = list(gpu_devices)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError(
                "gpu_devices must be an ordinal sequence or comma-separated string"
            ) from exc

    if not values:
        raise ValueError("gpu_devices must contain at least one device")
    if gpu_device != 0:
        try:
            legacy_device = int(gpu_device)
        except (TypeError, ValueError) as exc:
            raise ValueError("gpu_device must be a non-negative integer") from exc
        if not values or int(values[0]) != legacy_device:
            raise ValueError("gpu_device and gpu_devices specify different devices")
    result: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("gpu_devices must contain non-negative integer ordinals")
        if not isinstance(value, Integral) and not isinstance(value, str):
            raise ValueError("gpu_devices must contain non-negative integer ordinals")
        try:
            integer = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("gpu_devices must contain non-negative integer ordinals") from exc
        if isinstance(value, str) and value.strip() != str(integer):
            raise ValueError("gpu_devices must contain non-negative integer ordinals")
        if integer < 0:
            raise ValueError("gpu_devices must contain non-negative integer ordinals")
        if integer in result:
            raise ValueError(f"gpu_devices contains duplicate ordinal {integer}")
        result.append(integer)
    return tuple(result)


def _slurm_allocated_ordinals() -> set[int] | None:
    """Best-effort parse of Slurm's physical ordinal allocation metadata."""

    raw = os.environ.get("SLURM_STEP_GPUS") or os.environ.get("SLURM_JOB_GPUS")
    if raw:
        values: set[int] = set()
        for token in raw.replace("gpu:", "").replace("gpu", "").split(","):
            token = token.strip()
            if not token:
                continue
            match = re.fullmatch(r"(?:\[)?(\d+)(?:-(\d+))?(?:\])?", token)
            if match is None:
                return None
            start = int(match.group(1))
            stop = int(match.group(2) or start)
            if stop < start:
                return None
            values.update(range(start, stop + 1))
        return values or None
    count = os.environ.get("SLURM_GPUS_ON_NODE")
    if count and count.isdigit():
        return set(range(int(count)))
    return None


def validate_visible_gpu_devices(cp: Any, devices: Sequence[int]) -> tuple[int, ...]:
    """Validate requested process-visible ordinals before any GPU work starts."""

    ordered = tuple(int(device) for device in devices)
    try:
        visible_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:  # pragma: no cover - CUDA runtime dependent
        raise RuntimeError(f"unable to query visible CUDA devices: {exc}") from exc
    if visible_count < 1:
        raise RuntimeError("the explicitly requested GPU backend has no visible CUDA devices")

    # CUDA_VISIBLE_DEVICES remaps physical IDs to the process-visible ordinal
    # space.  The runtime count is authoritative, while this check provides a
    # clearer error for malformed/empty allocation environments.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        tokens = [part.strip() for part in visible.split(",")]
        if not tokens or any(part in {"", "-1", "NoDevFiles"} for part in tokens):
            raise RuntimeError("CUDA_VISIBLE_DEVICES exposes no usable CUDA devices")
        if len(tokens) != visible_count:
            raise RuntimeError(
                "CUDA runtime visibility disagrees with CUDA_VISIBLE_DEVICES: "
                f"runtime={visible_count}, CUDA_VISIBLE_DEVICES={visible!r}"
            )
    elif (allocated := _slurm_allocated_ordinals()) is not None:
        # Without remapping, Slurm's allocation list is the only safe source
        # of physical ordinals.  Never guess that the first N devices are ours.
        if not set(ordered).issubset(allocated):
            raise RuntimeError(
                "requested gpu_devices are outside the Slurm GPU allocation: "
                f"requested={ordered}, allocated={sorted(allocated)}"
            )

    invalid = [device for device in ordered if device < 0 or device >= visible_count]
    if invalid:
        raise RuntimeError(
            "requested gpu_devices are not visible process ordinals: "
            f"requested={ordered}, visible_count={visible_count}, invalid={invalid}"
        )
    return ordered


def contiguous_gpu_ranges(total: int, devices: Sequence[int]) -> tuple[tuple[int, int, int], ...]:
    """Partition ``[0, total)`` into balanced contiguous ``(device,start,stop)`` ranges."""

    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ValueError("total must be a non-negative integer")
    ordered = tuple(devices)
    if not ordered:
        raise ValueError("at least one GPU device is required")
    base, remainder = divmod(total, len(ordered))
    cursor = 0
    ranges: list[tuple[int, int, int]] = []
    for offset, device in enumerate(ordered):
        length = base + (1 if offset < remainder else 0)
        ranges.append((int(device), cursor, cursor + length))
        cursor += length
    return tuple(ranges)


__all__ = [
    "contiguous_gpu_ranges",
    "normalize_gpu_devices",
    "validate_visible_gpu_devices",
]
