"""Transactional, checksummed checkpoints for long-running entropy phases."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from nepflow.io.hashing import sha256_canonical_json
from nepflow.resources.budget import ResourceBudgetService, ResourceCapacityError

CHECKPOINT_SCHEMA_VERSION = "entropy-checkpoint-v1"
INCREMENTAL_KNN_CHECKPOINT_SCHEMA_VERSION = "entropy-knn-incremental-v1"


class CheckpointValidationError(ValueError):
    """Raised when a checkpoint cannot be safely resumed."""


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(name: str) -> str:
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError("checkpoint name must be a simple non-empty filename")
    return name


class VersionedCheckpointStore:
    """Publish complete checkpoint generations through an atomic pointer.

    A generation contains a JSON state record and NumPy ``.npy`` arrays.  The
    pointer is replaced only after every file has been flushed and checksummed;
    readers therefore see either the previous complete generation or the new
    complete generation, never an in-flight append.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _pointer(self, name: str) -> Path:
        return self.root / f"{_safe_name(name)}.pointer.json"

    def write(
        self,
        name: str,
        *,
        identity: Mapping[str, Any],
        state: Mapping[str, Any],
        arrays: Mapping[str, np.ndarray] | None = None,
    ) -> str:
        checkpoint_name = _safe_name(name)
        generation = f"{checkpoint_name}.generation-{uuid.uuid4().hex}"
        temporary = self.root / f".{generation}.tmp"
        temporary.mkdir()
        try:
            array_entries: dict[str, dict[str, Any]] = {}
            for array_name, value in (arrays or {}).items():
                if not isinstance(array_name, str) or Path(array_name).name != array_name:
                    raise ValueError("checkpoint array names must be simple filenames")
                array = np.asarray(value)
                if array.dtype == np.dtype("O"):
                    raise ValueError("checkpoint arrays must not use object dtype")
                array_path = temporary / f"{array_name}.npy"
                with array_path.open("wb") as handle:
                    np.save(handle, array, allow_pickle=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                array_entries[array_name] = {
                    "name": array_path.name,
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                    "sha256": _digest_file(array_path),
                }
            payload = {
                "schema": CHECKPOINT_SCHEMA_VERSION,
                "name": checkpoint_name,
                "identity": dict(identity),
                "identity_fingerprint": sha256_canonical_json(dict(identity)),
                "state": dict(state),
                "arrays": array_entries,
            }
            manifest_path = temporary / "manifest.json"
            with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            final = self.root / generation
            os.replace(temporary, final)
            pointer_temporary = self.root / f".{checkpoint_name}.pointer-{uuid.uuid4().hex}.tmp"
            pointer_payload = {"schema": CHECKPOINT_SCHEMA_VERSION, "generation": generation}
            with pointer_temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(pointer_payload, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(pointer_temporary, self._pointer(checkpoint_name))
            return generation
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    save = write

    def load(
        self,
        name: str,
        *,
        expected_identity: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
        checkpoint_name = _safe_name(name)
        pointer_path = self._pointer(checkpoint_name)
        try:
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            if pointer.get("schema") != CHECKPOINT_SCHEMA_VERSION:
                raise CheckpointValidationError("checkpoint pointer schema is incompatible")
            generation = pointer["generation"]
            if not isinstance(generation, str) or Path(generation).name != generation:
                raise CheckpointValidationError("checkpoint generation path is invalid")
            generation_path = self.root / generation
            manifest = json.loads((generation_path / "manifest.json").read_text(encoding="utf-8"))
        except CheckpointValidationError:
            raise
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise CheckpointValidationError("checkpoint pointer or manifest is unreadable") from exc
        if manifest.get("schema") != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointValidationError("checkpoint schema is incompatible")
        identity = manifest.get("identity")
        if not isinstance(identity, dict):
            raise CheckpointValidationError("checkpoint has no scientific identity")
        if sha256_canonical_json(identity) != manifest.get("identity_fingerprint"):
            raise CheckpointValidationError("checkpoint scientific identity checksum mismatch")
        if expected_identity is not None and sha256_canonical_json(
            dict(expected_identity)
        ) != manifest.get("identity_fingerprint"):
            raise CheckpointValidationError(
                "checkpoint scientific identity does not match the current ordered computation"
            )
        array_manifest = manifest.get("arrays", {})
        if not isinstance(array_manifest, dict):
            raise CheckpointValidationError("checkpoint array manifest is invalid")
        arrays: dict[str, np.ndarray] = {}
        for array_name, entry in array_manifest.items():
            if not isinstance(entry, dict):
                raise CheckpointValidationError(
                    f"checkpoint array metadata is invalid: {array_name}"
                )
            filename = entry.get("name")
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise CheckpointValidationError(f"checkpoint array path is invalid: {array_name}")
            path = generation_path / filename
            if not path.is_file():
                raise CheckpointValidationError(f"checkpoint array is missing: {array_name}")
            if _digest_file(path) != entry.get("sha256"):
                raise CheckpointValidationError(f"checkpoint array checksum mismatch: {array_name}")
            try:
                array = np.load(path, allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise CheckpointValidationError(
                    f"checkpoint array is unreadable: {array_name}"
                ) from exc
            if str(array.dtype) != entry.get("dtype") or list(array.shape) != entry.get("shape"):
                raise CheckpointValidationError(f"checkpoint array metadata mismatch: {array_name}")
            array.setflags(write=False)
            arrays[array_name] = array
        state = manifest.get("state")
        if not isinstance(state, dict):
            raise CheckpointValidationError("checkpoint state is invalid")
        return identity, state, arrays

    restore = load

    def exists(self, name: str) -> bool:
        return self._pointer(_safe_name(name)).is_file()

    def clear(self, name: str) -> None:
        checkpoint_name = _safe_name(name)
        pointer = self._pointer(checkpoint_name)
        generation = None
        if pointer.is_file():
            try:
                generation = json.loads(pointer.read_text(encoding="utf-8")).get("generation")
            except (OSError, ValueError):
                generation = None
        try:
            pointer.unlink()
        except FileNotFoundError:
            pass
        if isinstance(generation, str) and Path(generation).name == generation:
            shutil.rmtree(self.root / generation, ignore_errors=True)

    def clean_stale(self, *, keep_names: set[str] | None = None) -> int:
        """Remove only unreferenced generations and temporary directories."""

        keep = set(keep_names or ())
        referenced = set()
        for pointer in self.root.glob("*.pointer.json"):
            try:
                generation = json.loads(pointer.read_text(encoding="utf-8")).get("generation")
            except (OSError, ValueError):
                continue
            if isinstance(generation, str):
                referenced.add(generation)
        removed = 0
        for path in self.root.iterdir():
            if not path.is_dir() or path.name in referenced or path.name in keep:
                continue
            if path.name.startswith(".") or ".generation-" in path.name:
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
        return removed


class IncrementalCheckpointStore:
    """Transactional append-only store for canonical indexed kNN chunks.

    Data chunks are immutable directories.  Every commit writes a new
    manifest which references the old chunks plus one new contiguous range;
    only after that manifest is durable is the pointer atomically promoted.
    The existing :class:`VersionedCheckpointStore` remains unchanged for LOO
    and candidate aggregation checkpoints.
    """

    _ARRAYS = ("neighbour_locations", "neighbour_distances")

    def __init__(
        self,
        root: str | Path,
        *,
        resource_budget: ResourceBudgetService | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.resource_budget = resource_budget
        self._stats: dict[str, float | int] = {
            "bytes_written": 0,
            "bytes_read": 0,
            "fsync_seconds": 0.0,
            "digest_seconds": 0.0,
            "commits": 0,
        }

    def _pointer(self, name: str) -> Path:
        return self.root / f"{_safe_name(name)}.incremental.pointer.json"

    @staticmethod
    def _safe_file_name(value: Any) -> bool:
        return (
            isinstance(value, str)
            and value
            and Path(value).name == value
            and value not in {".", ".."}
        )

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        try:
            descriptor = os.open(str(path), flags)
        except (OSError, ValueError):
            return
        try:
            try:
                os.fsync(descriptor)
            except OSError:
                if os.name != "nt":
                    raise
        finally:
            os.close(descriptor)

    def _add_stat(self, name: str, value: float | int) -> None:
        if isinstance(value, float):
            self._stats[name] = float(self._stats.get(name, 0.0)) + max(0.0, value)
        else:
            self._stats[name] = int(self._stats.get(name, 0)) + int(value)

    def _retained_bytes(self) -> int:
        total = 0
        try:
            for path in self.root.iterdir():
                if ".incremental.manifest-" in path.name and path.is_file():
                    total += int(path.stat().st_size)
                elif ".incremental.chunk-" in path.name and path.is_dir():
                    total += sum(
                        int(child.stat().st_size)
                        for child in path.rglob("*")
                        if child.is_file()
                    )
        except OSError as exc:
            raise CheckpointValidationError(
                "unable to inspect incremental checkpoint volume"
            ) from exc
        return total

    def metrics(self) -> dict[str, float | int | None]:
        result: dict[str, float | int | None] = dict(self._stats)
        result["retained_bytes"] = self._retained_bytes()
        result["scratch_budget_bytes"] = (
            None
            if self.resource_budget is None
            else self.resource_budget.budget.scratch_budget_bytes
        )
        return result

    def _admit(self, requested_bytes: int) -> None:
        requested = max(1, int(requested_bytes))
        retained = self._retained_bytes()
        if self.resource_budget is not None:
            limit = self.resource_budget.budget.scratch_budget_bytes
            if limit is not None and retained + requested > limit:
                raise ResourceCapacityError(
                    "incremental kNN checkpoint exceeds the managed scratch budget: "
                    f"retained_bytes={retained}, requested_bytes={requested}, "
                    f"scratch_budget_bytes={limit}",
                    operation="incremental kNN checkpoint",
                    requested_bytes=requested,
                    scratch_needed_bytes=retained + requested,
                    scratch_available_bytes=limit,
                )
            if limit is not None:
                with self.resource_budget.acquire(
                    "incremental kNN checkpoint write", 1, scratch_bytes=requested
                ):
                    return
        try:
            available = int(shutil.disk_usage(self.root).free)
        except OSError as exc:
            raise ResourceCapacityError(
                "incremental kNN checkpoint scratch availability is unreadable",
                operation="incremental kNN checkpoint",
                requested_bytes=requested,
            ) from exc
        if available < requested:
            raise ResourceCapacityError(
                "incremental kNN checkpoint exceeds available scratch capacity: "
                f"requested_bytes={requested}, available_bytes={available}",
                operation="incremental kNN checkpoint",
                requested_bytes=requested,
                scratch_needed_bytes=requested,
                scratch_available_bytes=available,
            )

    def _read_manifest_with_name(self, name: str) -> tuple[str, dict[str, Any]]:
        checkpoint_name = _safe_name(name)
        try:
            pointer = json.loads(self._pointer(checkpoint_name).read_text(encoding="utf-8"))
            if pointer.get("schema") != INCREMENTAL_KNN_CHECKPOINT_SCHEMA_VERSION:
                raise CheckpointValidationError(
                    "incremental checkpoint pointer schema is incompatible"
                )
            manifest_name = pointer.get("manifest")
            if not self._safe_file_name(manifest_name):
                raise CheckpointValidationError("incremental checkpoint manifest path is invalid")
            manifest = json.loads(
                (self.root / manifest_name).read_text(encoding="utf-8")
            )
        except CheckpointValidationError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise CheckpointValidationError("incremental checkpoint is unreadable") from exc
        if (
            manifest.get("schema") != INCREMENTAL_KNN_CHECKPOINT_SCHEMA_VERSION
            or manifest.get("name") != checkpoint_name
        ):
            raise CheckpointValidationError("incremental checkpoint manifest is incompatible")
        return manifest_name, manifest

    def _read_manifest(self, name: str) -> dict[str, Any]:
        _, manifest = self._read_manifest_with_name(name)
        return manifest

    @staticmethod
    def _validate_identity(
        manifest: Mapping[str, Any], expected_identity: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        identity = manifest.get("identity")
        if not isinstance(identity, dict):
            raise CheckpointValidationError("incremental checkpoint identity is missing")
        if sha256_canonical_json(identity) != manifest.get("identity_fingerprint"):
            raise CheckpointValidationError("incremental checkpoint identity checksum mismatch")
        if expected_identity is not None and sha256_canonical_json(
            dict(expected_identity)
        ) != manifest.get("identity_fingerprint"):
            raise CheckpointValidationError(
                "incremental checkpoint identity does not match the ordered input"
            )
        return identity

    def _load_chain(
        self,
        name: str,
        *,
        expected_identity: Mapping[str, Any] | None,
        k_max: int | None,
        location_count: int | None,
    ) -> tuple[dict[str, Any], int, int, int, list[dict[str, Any]]]:
        manifest_name, manifest = self._read_manifest_with_name(name)
        identity: dict[str, Any] | None = None
        saved_k: int | None = None
        saved_count: int | None = None
        newest_cursor: int | None = None
        newest_chunk_count: int | None = None
        reverse_chunks: list[dict[str, Any]] = []
        seen: set[str] = set()
        while True:
            if manifest_name in seen:
                raise CheckpointValidationError("incremental checkpoint manifest chain loops")
            seen.add(manifest_name)
            current_identity = self._validate_identity(manifest, expected_identity)
            current_k = manifest.get("k_max")
            current_count = manifest.get("location_count")
            current_cursor = manifest.get("cursor")
            current_chunk_count = manifest.get("chunk_count")
            chunk = manifest.get("chunk")
            if (
                isinstance(current_k, bool)
                or not isinstance(current_k, int)
                or current_k < 1
                or isinstance(current_count, bool)
                or not isinstance(current_count, int)
                or current_count < 1
                or isinstance(current_cursor, bool)
                or not isinstance(current_cursor, int)
                or current_cursor < 1
                or current_cursor > current_count
                or isinstance(current_chunk_count, bool)
                or not isinstance(current_chunk_count, int)
                or current_chunk_count < 1
                or not isinstance(chunk, dict)
            ):
                raise CheckpointValidationError("incremental checkpoint manifest header is invalid")
            if identity is None:
                identity = current_identity
                saved_k = current_k
                saved_count = current_count
                newest_cursor = current_cursor
                newest_chunk_count = current_chunk_count
                if k_max is not None and saved_k != int(k_max):
                    raise CheckpointValidationError("incremental checkpoint k_max does not match")
                if location_count is not None and saved_count != int(location_count):
                    raise CheckpointValidationError(
                        "incremental checkpoint location_count does not match"
                    )
            elif (
                current_identity != identity
                or current_k != saved_k
                or current_count != saved_count
            ):
                raise CheckpointValidationError("incremental checkpoint manifest chain disagrees")
            reverse_chunks.append(chunk)
            previous = manifest.get("previous_manifest")
            if previous is None:
                break
            if not self._safe_file_name(previous) or previous in seen:
                raise CheckpointValidationError(
                    "incremental checkpoint previous manifest is invalid"
                )
            try:
                manifest = json.loads((self.root / previous).read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError) as exc:
                raise CheckpointValidationError(
                    "incremental checkpoint previous manifest is unreadable"
                ) from exc
            if manifest.get("schema") != INCREMENTAL_KNN_CHECKPOINT_SCHEMA_VERSION or manifest.get(
                "name"
            ) != _safe_name(name):
                raise CheckpointValidationError(
                    "incremental checkpoint previous manifest is invalid"
                )
            manifest_name = previous
        assert identity is not None and saved_k is not None and saved_count is not None
        assert newest_cursor is not None
        assert newest_chunk_count is not None
        if newest_chunk_count != len(reverse_chunks):
            raise CheckpointValidationError(
                "incremental checkpoint chunk count is not authoritative"
            )
        if reverse_chunks[-1].get("start") != 0:
            raise CheckpointValidationError("incremental checkpoint chain does not start at zero")
        chunks = list(reversed(reverse_chunks))
        expected_start = 0
        for chunk in chunks:
            start = chunk.get("start")
            stop = chunk.get("stop")
            length = chunk.get("length")
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(stop, bool)
                or not isinstance(stop, int)
                or isinstance(length, bool)
                or not isinstance(length, int)
                or length < 1
                or start != expected_start
                or stop != start + length
            ):
                raise CheckpointValidationError(
                    "incremental checkpoint chain contains a gap or overlap"
                )
            expected_start = int(chunk["stop"])
        if expected_start != newest_cursor:
            raise CheckpointValidationError("incremental checkpoint cursor is not authoritative")
        return identity, saved_k, saved_count, newest_cursor, chunks

    def write_chunk(
        self,
        name: str,
        *,
        identity: Mapping[str, Any],
        start: int,
        neighbour_locations: np.ndarray,
        neighbour_distances: np.ndarray,
        k_max: int,
        location_count: int,
    ) -> dict[str, float | int]:
        """Durably commit one new contiguous range."""

        checkpoint_name = _safe_name(name)
        if (
            isinstance(start, bool)
            or not isinstance(start, (int, np.integer))
            or isinstance(k_max, bool)
            or not isinstance(k_max, (int, np.integer))
            or isinstance(location_count, bool)
            or not isinstance(location_count, (int, np.integer))
        ):
            raise ValueError("incremental checkpoint range parameters must be integers")
        start = int(start)
        k_max = int(k_max)
        location_count = int(location_count)
        indices = np.asarray(neighbour_locations)
        distances = np.asarray(neighbour_distances)
        if start < 0 or k_max < 1 or location_count < 1:
            raise ValueError("incremental checkpoint range parameters are invalid")
        if (
            indices.dtype != np.dtype(np.int64)
            or indices.ndim != 2
            or indices.shape[1] != k_max
        ):
            raise ValueError("incremental checkpoint indices must be int64 with shape (L, k_max)")
        if (
            distances.dtype != np.dtype(np.float64)
            or distances.ndim != 2
            or distances.shape != indices.shape
        ):
            raise ValueError(
                "incremental checkpoint distances must be float64 with the index shape"
            )
        length = int(indices.shape[0])
        stop = start + length
        if length < 1 or stop > location_count:
            raise ValueError("incremental checkpoint range is outside location_count")

        previous_manifest: str | None = None
        previous_chunk_count = 0
        if self.exists(checkpoint_name):
            previous_manifest, manifest = self._read_manifest_with_name(checkpoint_name)
            self._validate_identity(manifest, identity)
            if manifest.get("k_max") != k_max or manifest.get("location_count") != location_count:
                raise CheckpointValidationError("incremental checkpoint dimensions do not match")
            if manifest.get("cursor") != start:
                raise CheckpointValidationError("incremental checkpoint chunk is not contiguous")
            previous_chunk_count = manifest.get("chunk_count", 0)
            if (
                isinstance(previous_chunk_count, bool)
                or not isinstance(previous_chunk_count, int)
                or previous_chunk_count < 1
            ):
                raise CheckpointValidationError("incremental checkpoint chunk count is invalid")
        elif start != 0:
            raise CheckpointValidationError("incremental checkpoint is missing its initial chunk")

        self._admit(indices.nbytes + distances.nbytes + 32768)
        chunk_name = (
            f"{checkpoint_name}.incremental.chunk-{start:012d}-{stop:012d}-"
            f"{uuid.uuid4().hex}"
        )
        temporary = self.root / f".{chunk_name}.tmp"
        final = self.root / chunk_name
        temporary.mkdir()
        manifest_temporary: Path | None = None
        pointer_temporary: Path | None = None
        started = time.perf_counter()
        bytes_written = 0
        try:
            array_entries: dict[str, dict[str, Any]] = {}
            for array_name, array in zip(self._ARRAYS, (indices, distances), strict=True):
                path = temporary / f"{array_name}.npy"
                with path.open("wb") as handle:
                    np.save(handle, array, allow_pickle=False)
                    handle.flush()
                    sync_started = time.perf_counter()
                    os.fsync(handle.fileno())
                    self._add_stat("fsync_seconds", time.perf_counter() - sync_started)
                digest_started = time.perf_counter()
                digest = _digest_file(path)
                self._add_stat("digest_seconds", time.perf_counter() - digest_started)
                byte_count = int(path.stat().st_size)
                bytes_written += byte_count
                array_entries[array_name] = {
                    "name": f"{chunk_name}/{path.name}",
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                    "offset": start,
                    "length": length,
                    "bytes": byte_count,
                    "sha256": digest,
                }
            self._fsync_directory(temporary)
            os.replace(temporary, final)
            self._fsync_directory(self.root)

            manifest_name = f"{checkpoint_name}.incremental.manifest-{uuid.uuid4().hex}.json"
            manifest_payload = {
                "schema": INCREMENTAL_KNN_CHECKPOINT_SCHEMA_VERSION,
                "name": checkpoint_name,
                "identity": dict(identity),
                "identity_fingerprint": sha256_canonical_json(dict(identity)),
                "k_max": k_max,
                "location_count": location_count,
                "cursor": stop,
                "chunk_count": previous_chunk_count + 1,
                "previous_manifest": previous_manifest,
                "chunk": {
                    "start": start,
                    "stop": stop,
                    "length": length,
                    "arrays": array_entries,
                },
            }
            manifest_temporary = self.root / f".{manifest_name}.tmp"
            with manifest_temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(manifest_payload, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                sync_started = time.perf_counter()
                os.fsync(handle.fileno())
                self._add_stat("fsync_seconds", time.perf_counter() - sync_started)
            os.replace(manifest_temporary, self.root / manifest_name)
            self._fsync_directory(self.root)

            pointer_payload = {
                "schema": INCREMENTAL_KNN_CHECKPOINT_SCHEMA_VERSION,
                "manifest": manifest_name,
            }
            pointer_temporary = self.root / (
                f".{checkpoint_name}.incremental.pointer-{uuid.uuid4().hex}.tmp"
            )
            with pointer_temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(pointer_payload, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                sync_started = time.perf_counter()
                os.fsync(handle.fileno())
                self._add_stat("fsync_seconds", time.perf_counter() - sync_started)
            os.replace(pointer_temporary, self._pointer(checkpoint_name))
            self._fsync_directory(self.root)
            self._add_stat("bytes_written", bytes_written)
            self._add_stat("commits", 1)
            return {
                "bytes_written": bytes_written,
                "write_seconds": max(0.0, time.perf_counter() - started),
            }
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        finally:
            if manifest_temporary is not None:
                manifest_temporary.unlink(missing_ok=True)
            if pointer_temporary is not None:
                pointer_temporary.unlink(missing_ok=True)

    def load(
        self,
        name: str,
        *,
        expected_identity: Mapping[str, Any] | None = None,
        k_max: int | None = None,
        location_count: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
        """Validate and restore the complete committed prefix read-only."""

        identity, saved_k, saved_count, cursor, chunks = self._load_chain(
            name,
            expected_identity=expected_identity,
            k_max=k_max,
            location_count=location_count,
        )

        expected_offset = 0
        index_parts: list[np.ndarray] = []
        distance_parts: list[np.ndarray] = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise CheckpointValidationError("incremental checkpoint chunk is invalid")
            start = chunk.get("start")
            stop = chunk.get("stop")
            length = chunk.get("length")
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(stop, bool)
                or not isinstance(stop, int)
                or isinstance(length, bool)
                or not isinstance(length, int)
                or start != expected_offset
                or length < 1
                or stop != start + length
            ):
                raise CheckpointValidationError("incremental checkpoint contains a gap or overlap")
            array_manifest = chunk.get("arrays")
            if not isinstance(array_manifest, dict):
                raise CheckpointValidationError("incremental checkpoint arrays are invalid")
            loaded: dict[str, np.ndarray] = {}
            for array_name in self._ARRAYS:
                entry = array_manifest.get(array_name)
                if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                    raise CheckpointValidationError(
                        "incremental checkpoint array metadata is invalid"
                    )
                filename = entry["name"]
                parts = filename.split("/", 1)
                if (
                    len(parts) != 2
                    or not self._safe_file_name(parts[0])
                    or not self._safe_file_name(parts[1])
                ):
                    raise CheckpointValidationError("incremental checkpoint array path is invalid")
                path = self.root / filename
                try:
                    path.relative_to(self.root)
                except ValueError as exc:
                    raise CheckpointValidationError(
                        "incremental checkpoint array path escapes the checkpoint root"
                    ) from exc
                if not path.is_file():
                    raise CheckpointValidationError("incremental checkpoint array is missing")
                digest_started = time.perf_counter()
                digest = _digest_file(path)
                byte_count = int(path.stat().st_size)
                self._add_stat("digest_seconds", time.perf_counter() - digest_started)
                self._add_stat("bytes_read", byte_count)
                if digest != entry.get("sha256") or byte_count != entry.get("bytes"):
                    raise CheckpointValidationError(
                        "incremental checkpoint array checksum mismatch"
                    )
                try:
                    array = np.load(path, allow_pickle=False)
                except (OSError, ValueError) as exc:
                    raise CheckpointValidationError(
                        "incremental checkpoint array is unreadable"
                    ) from exc
                expected_dtype = np.dtype(np.int64 if array_name == self._ARRAYS[0] else np.float64)
                if (
                    array.dtype != expected_dtype
                    or array.shape != (length, saved_k)
                    or entry.get("dtype") != str(expected_dtype)
                    or entry.get("shape") != [length, saved_k]
                    or entry.get("offset") != start
                    or entry.get("length") != length
                ):
                    raise CheckpointValidationError(
                        "incremental checkpoint array metadata mismatch"
                    )
                array.setflags(write=False)
                loaded[array_name] = array
            index_parts.append(loaded[self._ARRAYS[0]])
            distance_parts.append(loaded[self._ARRAYS[1]])
            expected_offset = stop
        if expected_offset != cursor:
            raise CheckpointValidationError("incremental checkpoint cursor is not authoritative")
        indices = np.concatenate(index_parts, axis=0)
        distances = np.concatenate(distance_parts, axis=0)
        indices.setflags(write=False)
        distances.setflags(write=False)
        return (
            identity,
            {
                "cursor": cursor,
                "k_max": saved_k,
                "location_count": saved_count,
                "chunk_count": len(chunks),
            },
            {self._ARRAYS[0]: indices, self._ARRAYS[1]: distances},
        )

    restore = load

    def exists(self, name: str) -> bool:
        return self._pointer(_safe_name(name)).is_file()

    def clear(self, name: str) -> None:
        self._pointer(_safe_name(name)).unlink(missing_ok=True)
        self.clean_stale()

    def clean_stale(self, *, keep_names: set[str] | None = None) -> int:
        """Remove only unreferenced incremental generations and temp files."""

        keep = set(keep_names or ())
        manifests: set[str] = set()
        chunks: set[str] = set()
        for pointer in self.root.glob("*.incremental.pointer.json"):
            try:
                payload = json.loads(pointer.read_text(encoding="utf-8"))
                manifest_name = payload.get("manifest")
                if not self._safe_file_name(manifest_name):
                    raise CheckpointValidationError(
                        "incremental checkpoint pointer manifest is invalid"
                    )
                current_name: str | None = manifest_name
                visited: set[str] = set()
                while current_name is not None and current_name not in visited:
                    visited.add(current_name)
                    manifests.add(current_name)
                    manifest = json.loads(
                        (self.root / current_name).read_text(encoding="utf-8")
                    )
                    chunk = manifest.get("chunk", {})
                    for entry in chunk.get("arrays", {}).values():
                        filename = entry.get("name") if isinstance(entry, dict) else None
                        if isinstance(filename, str) and "/" in filename:
                            chunks.add(filename.split("/", 1)[0])
                    previous = manifest.get("previous_manifest")
                    if previous is not None and not self._safe_file_name(previous):
                        raise CheckpointValidationError(
                            "incremental checkpoint previous manifest is invalid"
                        )
                    current_name = previous
                if current_name is not None and current_name in visited:
                    raise CheckpointValidationError(
                        "incremental checkpoint manifest chain loops"
                    )
            except (OSError, TypeError, ValueError, AttributeError) as exc:
                raise CheckpointValidationError(
                    "unable to inspect a referenced incremental checkpoint generation"
                ) from exc
        removed = 0
        for path in self.root.iterdir():
            if path.name in keep or path.name in manifests or path.name in chunks:
                continue
            stale = (
                (".incremental.manifest-" in path.name and path.is_file())
                or (".incremental.chunk-" in path.name and path.is_dir())
                or (path.name.startswith(".") and ".incremental." in path.name)
            )
            if not stale:
                continue
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)
            except OSError as exc:
                raise CheckpointValidationError(
                    f"unable to clean stale incremental checkpoint entry: {path.name}"
                ) from exc
            if path.exists():
                raise CheckpointValidationError(
                    f"stale incremental checkpoint entry remains after cleanup: {path.name}"
                )
            removed += 1
        return removed


class CandidateAggregationCheckpoint:
    """Append-only, row-granular checkpoint for candidate CSR aggregation."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "aggregation-manifest.json"

    def initialize(self, identity: Mapping[str, Any], candidate_ids: tuple[str, ...]) -> None:
        fingerprint = sha256_canonical_json(dict(identity))
        if self.manifest_path.is_file():
            try:
                manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise CheckpointValidationError(
                    "candidate aggregation manifest is unreadable"
                ) from exc
            if (
                manifest.get("schema") != CHECKPOINT_SCHEMA_VERSION
                or manifest.get("identity_fingerprint") != fingerprint
                or tuple(manifest.get("candidate_ids", ())) != tuple(candidate_ids)
            ):
                raise CheckpointValidationError(
                    "candidate aggregation checkpoint identity or candidate order does not match"
                )
            return
        self._write_manifest(
            {
                "schema": CHECKPOINT_SCHEMA_VERSION,
                "identity": dict(identity),
                "identity_fingerprint": fingerprint,
                "candidate_ids": list(candidate_ids),
                "completed": {},
            }
        )

    def _read_manifest(self) -> dict[str, Any]:
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CheckpointValidationError("candidate aggregation manifest is unreadable") from exc
        if manifest.get("schema") != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointValidationError(
                "candidate aggregation checkpoint schema is incompatible"
            )
        return manifest

    def _write_manifest(self, manifest: Mapping[str, Any]) -> None:
        temporary = self.root / f".aggregation-manifest-{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(dict(manifest), handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.manifest_path)
        finally:
            temporary.unlink(missing_ok=True)

    def write_row(
        self,
        candidate_index: int,
        candidate_id: str,
        target_indices: np.ndarray,
        values: np.ndarray,
        *,
        source_cursor: int,
        q_completed: int,
        implicit_edges: int,
        source_support: np.ndarray | None = None,
        self_edge_count: int = 0,
        source_mass_max_deviation: float = 0.0,
    ) -> None:
        manifest = self._read_manifest()
        row_name = f"row-{int(candidate_index):08d}"
        temporary = self.root / f".{row_name}-{uuid.uuid4().hex}.tmp"
        final = self.root / row_name
        temporary.mkdir()
        try:
            target_path = temporary / "target_indices.npy"
            values_path = temporary / "values.npy"
            for path, array in ((target_path, target_indices), (values_path, values)):
                with path.open("wb") as handle:
                    np.save(handle, np.asarray(array), allow_pickle=False)
                    handle.flush()
                    os.fsync(handle.fileno())
            source_support_path = None
            if source_support is not None:
                source_support_path = temporary / "source_support.npy"
                with source_support_path.open("wb") as handle:
                    np.save(
                        handle,
                        np.asarray(source_support, dtype=np.int64),
                        allow_pickle=False,
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
            row_manifest = {
                "candidate_id": candidate_id,
                "source_cursor": int(source_cursor),
                "q_completed": int(q_completed),
                "implicit_edges": int(implicit_edges),
                "self_edge_count": int(self_edge_count),
                "source_mass_max_deviation": float(source_mass_max_deviation),
                "target_sha256": _digest_file(target_path),
                "values_sha256": _digest_file(values_path),
            }
            if source_support_path is not None:
                row_manifest["source_support_sha256"] = _digest_file(source_support_path)
            with (temporary / "manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(row_manifest, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            if final.exists():
                shutil.rmtree(final)
            os.replace(temporary, final)
            completed = dict(manifest.get("completed", {}))
            completed[str(int(candidate_index))] = row_manifest
            manifest["completed"] = completed
            self._write_manifest(manifest)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def load_rows(self) -> dict[int, tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
        manifest = self._read_manifest()
        result: dict[int, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
        for index_text, row_manifest in manifest.get("completed", {}).items():
            index = int(index_text)
            row_dir = self.root / f"row-{index:08d}"
            if not row_dir.is_dir() or not isinstance(row_manifest, dict):
                raise CheckpointValidationError(
                    "candidate aggregation checkpoint row is incomplete"
                )
            target_path = row_dir / "target_indices.npy"
            values_path = row_dir / "values.npy"
            if not target_path.is_file() or not values_path.is_file():
                raise CheckpointValidationError(
                    f"candidate aggregation checkpoint row is incomplete: {index}"
                )
            if _digest_file(target_path) != row_manifest.get("target_sha256") or _digest_file(
                values_path
            ) != row_manifest.get("values_sha256"):
                raise CheckpointValidationError(
                    f"candidate aggregation checkpoint row checksum mismatch: {index}"
                )
            try:
                targets = np.load(target_path, allow_pickle=False)
                values = np.load(values_path, allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise CheckpointValidationError(
                    f"candidate aggregation checkpoint row is unreadable: {index}"
                ) from exc
            source_support_name = row_dir / "source_support.npy"
            source_support_digest = row_manifest.get("source_support_sha256")
            if source_support_digest is not None:
                if not source_support_name.is_file() or _digest_file(source_support_name) != (
                    source_support_digest
                ):
                    raise CheckpointValidationError(
                        f"candidate aggregation checkpoint source support is invalid: {index}"
                    )
                try:
                    source_support = np.load(source_support_name, allow_pickle=False)
                except (OSError, ValueError) as exc:
                    raise CheckpointValidationError(
                        f"candidate aggregation checkpoint source support is unreadable: {index}"
                    ) from exc
                if source_support.dtype != np.dtype(np.int64) or source_support.ndim != 1:
                    raise CheckpointValidationError(
                        f"candidate aggregation checkpoint source support is invalid: {index}"
                    )
                row_manifest = dict(row_manifest)
                row_manifest["_source_support"] = source_support
            result[index] = (targets, values, row_manifest)
        return result

    def clean_stale(self) -> int:
        """Remove only unreferenced row generations and temporary fragments."""

        if not self.manifest_path.is_file():
            return 0
        manifest = self._read_manifest()
        completed = manifest.get("completed", {})
        if not isinstance(completed, dict):
            raise CheckpointValidationError(
                "candidate aggregation checkpoint completed rows are invalid"
            )
        referenced = {f"row-{int(index):08d}" for index in completed}
        removed = 0
        for path in self.root.iterdir():
            stale_row = (
                path.is_dir() and path.name.startswith("row-") and path.name not in referenced
            )
            temporary = path.name.startswith(".")
            if not stale_row and not temporary:
                continue
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
            removed += 1
        return removed

    def clear(self) -> None:
        for path in self.root.iterdir():
            if path.is_dir() and (path.name.startswith("row-") or path.name.startswith(".")):
                shutil.rmtree(path, ignore_errors=True)
            elif path.is_file() and path.name.startswith("."):
                path.unlink(missing_ok=True)
        self.manifest_path.unlink(missing_ok=True)


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "INCREMENTAL_KNN_CHECKPOINT_SCHEMA_VERSION",
    "CheckpointValidationError",
    "CandidateAggregationCheckpoint",
    "IncrementalCheckpointStore",
    "VersionedCheckpointStore",
]
