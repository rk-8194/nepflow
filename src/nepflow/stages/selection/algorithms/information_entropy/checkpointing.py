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
_COMPACT_ROW_ID_LIMIT = 1024


class CheckpointValidationError(ValueError):
    """Raised when a checkpoint cannot be safely resumed."""


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    """Durably publish directory-entry replacements where the platform allows it."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(str(path), flags)
    except (OSError, ValueError):
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            # Windows does not expose a generally usable directory fsync.
            if os.name != "nt":
                raise
    finally:
        os.close(descriptor)


def _atomic_replace(source: Path, destination: Path) -> None:
    """Replace a path, tolerating short-lived Windows scanner locks."""

    attempts = 8 if os.name == "nt" else 1
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(0.01 * (attempt + 1))


def _safe_name(name: str) -> str:
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError("checkpoint name must be a simple non-empty filename")
    return name


def _compact_incremental_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Keep large ordered row-id identities out of every chunk manifest.

    The ordered row-id sequence remains part of the checkpoint identity, but a
    digest is sufficient for resume validation and avoids repeating hundreds of
    megabytes of metadata across a long incremental manifest chain.
    """

    compact = dict(identity)
    row_ids = compact.get("row_ids")
    if isinstance(row_ids, list) and len(row_ids) > _COMPACT_ROW_ID_LIMIT:
        compact["row_ids_fingerprint"] = sha256_canonical_json(row_ids)
        compact["row_ids_count"] = len(row_ids)
        del compact["row_ids"]
    return compact


class VersionedCheckpointStore:
    """Publish complete checkpoint generations through an atomic pointer.

    A generation contains a JSON state record and NumPy ``.npy`` arrays.  The
    pointer is replaced only after every file has been flushed and checksummed;
    readers therefore see either the previous complete generation or the new
    complete generation, never an in-flight append.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        retention_generations: int | None = None,
        resource_budget: ResourceBudgetService | None = None,
        retention_name: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if retention_generations is not None and (
            isinstance(retention_generations, bool)
            or not isinstance(retention_generations, int)
            or retention_generations < 1
        ):
            raise ValueError("retention_generations must be a positive integer when provided")
        self.retention_generations = retention_generations
        self.resource_budget = resource_budget
        if retention_name is not None:
            retention_name = _safe_name(retention_name)
        self.retention_name = retention_name
        self._retained_bytes_cache: int | None = None
        self._retention_index: dict[str, list[str]] = {}
        self._stats: dict[str, float | int] = {
            "bytes_written": 0,
            "array_bytes_written": 0,
            "physical_bytes_written": 0,
            "metadata_bytes_written": 0,
            "bytes_read": 0,
            "fsync_seconds": 0.0,
            "digest_seconds": 0.0,
            "write_seconds": 0.0,
            "checkpoint_wall_seconds": 0.0,
            "cleanup_seconds": 0.0,
            "commits": 0,
            "cleanup_calls": 0,
            "stale_generation_count": 0,
            "stale_generation_bytes": 0,
            "retired_generation_count": 0,
            "retired_generation_bytes": 0,
            "peak_retained_bytes": 0,
            "max_retained_scratch_bytes": 0,
        }
        if self.retention_generations is not None:
            self._recover_rolling_generations()

    def _add_stat(self, name: str, value: float | int) -> None:
        if isinstance(value, float):
            self._stats[name] = float(self._stats.get(name, 0.0)) + max(0.0, value)
        else:
            self._stats[name] = int(self._stats.get(name, 0)) + int(value)

    def _retained_bytes(self) -> int:
        if self.retention_generations is not None and self._retained_bytes_cache is not None:
            return int(self._retained_bytes_cache)
        total = 0
        try:
            for path in self.root.iterdir():
                if path.is_dir() and ".generation-" in path.name:
                    total += sum(
                        int(child.stat().st_size)
                        for child in path.rglob("*")
                        if child.is_file()
                    )
                elif (
                    path.is_file()
                    and path.name.endswith(".pointer.json")
                    and not path.name.endswith(".incremental.pointer.json")
                ):
                    total += int(path.stat().st_size)
        except OSError as exc:
            raise CheckpointValidationError(
                "unable to inspect checkpoint volume"
            ) from exc
        if self.retention_generations is not None:
            self._retained_bytes_cache = total
        return total

    def metrics(self) -> dict[str, float | int | None]:
        result: dict[str, float | int | None] = dict(self._stats)
        result["retained_bytes"] = self._retained_bytes()
        result["peak_retained_bytes"] = max(
            int(result.get("peak_retained_bytes", 0)), int(result["retained_bytes"])
        )
        self._stats["peak_retained_bytes"] = int(result["peak_retained_bytes"])
        result["max_retained_scratch_bytes"] = max(
            int(self._stats.get("max_retained_scratch_bytes", 0)),
            int(result["peak_retained_bytes"]),
        )
        self._stats["max_retained_scratch_bytes"] = int(result["max_retained_scratch_bytes"])
        result["scratch_budget_bytes"] = (
            None
            if self.resource_budget is None
            else self.resource_budget.budget.scratch_budget_bytes
        )
        return result

    def _retention_path(self, name: str) -> Path:
        return self.root / f".{_safe_name(name)}.retention.json"

    def _read_retention_index(self, name: str) -> list[str]:
        if self.retention_generations is None:
            return []
        checkpoint_name = _safe_name(name)
        if self.retention_name is not None and checkpoint_name != self.retention_name:
            raise ValueError(
                f"rolling retention is scoped to checkpoint {self.retention_name!r}"
            )
        cached = self._retention_index.get(checkpoint_name)
        if cached is not None:
            return list(cached)
        path = self._retention_path(checkpoint_name)
        if not path.is_file():
            result: list[str] = []
        else:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                values = payload.get("generations")
                if not isinstance(values, list):
                    raise ValueError("retention generations are not a list")
                result = []
                for value in values:
                    if (
                        not isinstance(value, str)
                        or Path(value).name != value
                        or not value.startswith(f"{checkpoint_name}.generation-")
                    ):
                        raise ValueError("retention generation name is invalid")
                    if value not in result:
                        result.append(value)
            except (OSError, TypeError, ValueError) as exc:
                raise CheckpointValidationError(
                    "checkpoint retention metadata is unreadable"
                ) from exc
        self._retention_index[checkpoint_name] = result
        return list(result)

    def _write_retention_index(self, name: str, generations: list[str]) -> None:
        if self.retention_generations is None:
            return
        checkpoint_name = _safe_name(name)
        if self.retention_name is not None and checkpoint_name != self.retention_name:
            raise ValueError(
                f"rolling retention is scoped to checkpoint {self.retention_name!r}"
            )
        path = self._retention_path(checkpoint_name)
        temporary = self.root / f".{checkpoint_name}.retention-{uuid.uuid4().hex}.tmp"
        payload = {
            "schema": CHECKPOINT_SCHEMA_VERSION,
            "name": checkpoint_name,
            "generations": list(generations[-self.retention_generations :]),
        }
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                sync_started = time.perf_counter()
                os.fsync(handle.fileno())
                self._add_stat("fsync_seconds", time.perf_counter() - sync_started)
            self._add_stat("physical_bytes_written", int(temporary.stat().st_size))
            self._add_stat("metadata_bytes_written", int(temporary.stat().st_size))
            _atomic_replace(temporary, path)
            _fsync_directory(self.root)
        finally:
            temporary.unlink(missing_ok=True)
        self._retention_index[checkpoint_name] = list(payload["generations"])

    def _referenced_generations(self) -> set[str]:
        referenced: set[str] = set()
        for pointer in self.root.glob("*.pointer.json"):
            if pointer.name.endswith(".incremental.pointer.json"):
                continue
            try:
                payload = json.loads(pointer.read_text(encoding="utf-8"))
                generation = payload.get("generation")
            except (OSError, TypeError, ValueError) as exc:
                raise CheckpointValidationError(
                    f"checkpoint pointer is unreadable: {pointer.name}"
                ) from exc
            if payload.get("schema") != CHECKPOINT_SCHEMA_VERSION:
                raise CheckpointValidationError(
                    f"checkpoint pointer schema is incompatible: {pointer.name}"
                )
            if not isinstance(generation, str) or Path(generation).name != generation:
                raise CheckpointValidationError(
                    f"checkpoint pointer generation is invalid: {pointer.name}"
                )
            if not (self.root / generation).is_dir():
                raise CheckpointValidationError(
                    f"checkpoint pointer generation is missing: {generation}"
                )
            referenced.add(generation)
        return referenced

    @staticmethod
    def _entry_bytes(path: Path) -> int:
        if path.is_file():
            return int(path.stat().st_size)
        if path.is_dir():
            return sum(
                int(child.stat().st_size) for child in path.rglob("*") if child.is_file()
            )
        return 0

    def _remove_generation_if_unreferenced(
        self, path: Path, *, referenced: set[str] | None = None
    ) -> int:
        if not path.is_dir() or ".generation-" not in path.name:
            return 0
        active = referenced if referenced is not None else self._referenced_generations()
        if path.name in active:
            return 0
        byte_count = self._entry_bytes(path)
        try:
            shutil.rmtree(path)
        except OSError as exc:
            raise CheckpointValidationError(
                f"unable to retire stale checkpoint generation: {path.name}"
            ) from exc
        if path.exists():
            raise CheckpointValidationError(
                f"stale checkpoint generation remains after retirement: {path.name}"
            )
        self._add_stat("stale_generation_count", 1)
        self._add_stat("stale_generation_bytes", byte_count)
        self._add_stat("retired_generation_count", 1)
        self._add_stat("retired_generation_bytes", byte_count)
        if self._retained_bytes_cache is not None:
            self._retained_bytes_cache = max(0, self._retained_bytes_cache - byte_count)
        return byte_count

    def _recover_rolling_generations(self) -> None:
        """Recover a bounded rolling set once when a store is opened."""

        referenced = self._referenced_generations()
        names: set[str] = set()
        if self.retention_name is not None:
            names.add(self.retention_name)
        else:
            for pointer in self.root.glob("*.pointer.json"):
                if pointer.name.endswith(".incremental.pointer.json"):
                    continue
                names.add(pointer.name[: -len(".pointer.json")])
            for path in self.root.iterdir():
                if path.is_dir() and ".generation-" in path.name:
                    names.add(path.name.split(".generation-", 1)[0])
        for name in sorted(names):
            _safe_name(name)
            indexed = self._read_retention_index(name)
            active = next(
                (
                    generation
                    for generation in referenced
                    if generation.startswith(f"{name}.generation-")
                ),
                None,
            )
            keep = [generation for generation in indexed if (self.root / generation).is_dir()]
            if active is not None:
                keep = [generation for generation in keep if generation != active] + [active]
            keep = keep[-int(self.retention_generations or 1) :]
            for path in self.root.iterdir():
                if path.is_dir() and path.name.startswith(f"{name}.generation-"):
                    if path.name not in keep:
                        self._remove_generation_if_unreferenced(path, referenced=referenced)
            if keep:
                self._retention_index[name] = keep
                self._write_retention_index(name, keep)
            else:
                self._retention_path(name).unlink(missing_ok=True)
        self._retained_bytes_cache = None
        self._retained_bytes()

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
        if self.retention_name is not None and checkpoint_name != self.retention_name:
            raise ValueError(
                f"rolling retention is scoped to checkpoint {self.retention_name!r}"
            )
        generation = f"{checkpoint_name}.generation-{uuid.uuid4().hex}"
        temporary = self.root / f".{generation}.tmp"
        write_started = time.perf_counter()
        if self.retention_generations is not None:
            estimated_array_bytes = sum(
                int(np.asarray(value).nbytes) + 512 for value in (arrays or {}).values()
            )
            estimated_metadata_bytes = (
                8192
                + 256 * len(arrays or {})
                + len(json.dumps(dict(identity), sort_keys=True, separators=(",", ":")))
                + len(json.dumps(dict(state), sort_keys=True, separators=(",", ":")))
            )
            requested_scratch = estimated_array_bytes + estimated_metadata_bytes
            retained = self._retained_bytes()
            scratch_needed = retained + requested_scratch
            self._stats["max_retained_scratch_bytes"] = max(
                int(self._stats.get("max_retained_scratch_bytes", 0)), scratch_needed
            )
            if self.resource_budget is not None:
                limit = self.resource_budget.budget.scratch_budget_bytes
                if limit is None:
                    raise ResourceCapacityError(
                        "LOO checkpoint scratch availability is unknown",
                        operation="LOO checkpoint write",
                        requested_bytes=requested_scratch,
                        scratch_needed_bytes=scratch_needed,
                        scratch_available_bytes=None,
                    )
                if scratch_needed > limit:
                    raise ResourceCapacityError(
                        "LOO rolling checkpoint exceeds the managed scratch budget: "
                        f"retained_bytes={retained}, requested_bytes={requested_scratch}, "
                        f"scratch_needed_bytes={scratch_needed}, scratch_budget_bytes={limit}",
                        operation="LOO checkpoint write",
                        requested_bytes=requested_scratch,
                        scratch_needed_bytes=scratch_needed,
                        scratch_available_bytes=limit,
                    )
                lease = self.resource_budget.acquire(
                    "LOO checkpoint write",
                    max(1, estimated_metadata_bytes),
                    scratch_bytes=scratch_needed,
                )
            else:
                lease = None
            try:
                return self._write_generation(
                    checkpoint_name,
                    generation,
                    temporary,
                    identity=identity,
                    state=state,
                    arrays=arrays,
                    write_started=write_started,
                )
            finally:
                if lease is not None:
                    lease.close()
        return self._write_generation(
            checkpoint_name,
            generation,
            temporary,
            identity=identity,
            state=state,
            arrays=arrays,
            write_started=write_started,
        )

    def _write_generation(
        self,
        checkpoint_name: str,
        generation: str,
        temporary: Path,
        *,
        identity: Mapping[str, Any],
        state: Mapping[str, Any],
        arrays: Mapping[str, np.ndarray] | None,
        write_started: float,
    ) -> str:
        temporary.mkdir()
        final = self.root / generation
        pointer_temporary: Path | None = None
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
                    sync_started = time.perf_counter()
                    os.fsync(handle.fileno())
                    self._add_stat("fsync_seconds", time.perf_counter() - sync_started)
                array_bytes = int(array_path.stat().st_size)
                self._add_stat("array_bytes_written", array_bytes)
                self._add_stat("bytes_written", array_bytes)
                self._add_stat("physical_bytes_written", array_bytes)
                digest_started = time.perf_counter()
                digest = _digest_file(array_path)
                self._add_stat("digest_seconds", time.perf_counter() - digest_started)
                array_entries[array_name] = {
                    "name": array_path.name,
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                    "bytes": array_bytes,
                    "sha256": digest,
                }
            payload = {
                "schema": CHECKPOINT_SCHEMA_VERSION,
                "name": checkpoint_name,
                "identity": dict(identity),
                "identity_fingerprint": sha256_canonical_json(dict(identity)),
                "state": dict(state),
                "state_fingerprint": sha256_canonical_json(dict(state)),
                "arrays": array_entries,
            }
            manifest_path = temporary / "manifest.json"
            with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                sync_started = time.perf_counter()
                os.fsync(handle.fileno())
                self._add_stat("fsync_seconds", time.perf_counter() - sync_started)
            self._add_stat("physical_bytes_written", int(manifest_path.stat().st_size))
            self._add_stat("metadata_bytes_written", int(manifest_path.stat().st_size))
            _atomic_replace(temporary, final)
            _fsync_directory(self.root)
            pointer_temporary = self.root / f".{checkpoint_name}.pointer-{uuid.uuid4().hex}.tmp"
            pointer_payload = {"schema": CHECKPOINT_SCHEMA_VERSION, "generation": generation}
            with pointer_temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(pointer_payload, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                sync_started = time.perf_counter()
                os.fsync(handle.fileno())
                self._add_stat("fsync_seconds", time.perf_counter() - sync_started)
            self._add_stat("physical_bytes_written", int(pointer_temporary.stat().st_size))
            self._add_stat("metadata_bytes_written", int(pointer_temporary.stat().st_size))
            _atomic_replace(pointer_temporary, self._pointer(checkpoint_name))
            _fsync_directory(self.root)

            if self.retention_generations is not None:
                previous = self._read_retention_index(checkpoint_name)
                retained_generations = [
                    value
                    for value in previous
                    if value != generation and (self.root / value).is_dir()
                ]
                retained_generations.append(generation)
                retained_generations = retained_generations[-int(self.retention_generations) :]
                referenced = self._referenced_generations()
                cleanup_started = time.perf_counter()
                for value in previous:
                    if value not in retained_generations:
                        self._remove_generation_if_unreferenced(
                            self.root / value, referenced=referenced
                        )
                self._add_stat("cleanup_seconds", time.perf_counter() - cleanup_started)
                self._stats["cleanup_calls"] = int(self._stats.get("cleanup_calls", 0)) + 1
                self._write_retention_index(checkpoint_name, retained_generations)
                self._retained_bytes_cache = None
                self._retained_bytes()
            self._add_stat("commits", 1)
            elapsed = max(0.0, time.perf_counter() - write_started)
            self._add_stat("write_seconds", elapsed)
            self._add_stat("checkpoint_wall_seconds", elapsed)
            self._stats["peak_retained_bytes"] = max(
                int(self._stats.get("peak_retained_bytes", 0)), self._retained_bytes()
            )
            return generation
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            if pointer_temporary is not None:
                pointer_temporary.unlink(missing_ok=True)
            if final.exists():
                self._remove_generation_if_unreferenced(final)
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
            digest_started = time.perf_counter()
            digest = _digest_file(path)
            self._add_stat("digest_seconds", time.perf_counter() - digest_started)
            self._add_stat("bytes_read", int(path.stat().st_size))
            if digest != entry.get("sha256"):
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
        if sha256_canonical_json(state) != manifest.get("state_fingerprint"):
            raise CheckpointValidationError("checkpoint state checksum mismatch")
        return identity, state, arrays

    restore = load

    def exists(self, name: str) -> bool:
        return self._pointer(_safe_name(name)).is_file()

    def has_artifacts(self, name: str) -> bool:
        """Return whether a pointer or unpublished retained generation exists."""

        checkpoint_name = _safe_name(name)
        if self._pointer(checkpoint_name).is_file() or self._retention_path(
            checkpoint_name
        ).is_file():
            return True
        return any(
            path.is_dir() and path.name.startswith(f"{checkpoint_name}.generation-")
            for path in self.root.iterdir()
        )

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
            self._remove_generation_if_unreferenced(self.root / generation)
        if self.retention_generations is not None:
            self._retention_index.pop(checkpoint_name, None)
            self._retention_path(checkpoint_name).unlink(missing_ok=True)
            self._retained_bytes_cache = None
            self._retained_bytes()

    def clean_stale(
        self,
        *,
        keep_names: set[str] | None = None,
        checkpoint_names: set[str] | None = None,
    ) -> int:
        """Remove only unreferenced generations and temporary directories."""

        cleanup_started = time.perf_counter()
        keep = set(keep_names or ())
        scope = set(checkpoint_names or ())
        if self.retention_name is not None:
            scope.add(self.retention_name)
        if scope:
            scope = {_safe_name(name) for name in scope}
        referenced = self._referenced_generations()
        removed = 0
        for path in self.root.iterdir():
            if path.is_dir() and (path.name in referenced or path.name in keep):
                continue
            if path.is_dir() and (path.name.startswith(".") or ".generation-" in path.name):
                generation_name = path.name.split(".generation-", 1)[0]
                if scope and generation_name.lstrip(".") not in scope:
                    continue
                if ".generation-" in path.name:
                    self._remove_generation_if_unreferenced(path, referenced=referenced)
                else:
                    shutil.rmtree(path, ignore_errors=True)
                removed += 1
            elif path.is_file() and path.name.startswith(".") and path.name.endswith(".tmp"):
                if scope and not any(path.name.startswith(f".{name}.") for name in scope):
                    continue
                path.unlink(missing_ok=True)
                removed += 1
        self._stats["cleanup_calls"] = int(self._stats.get("cleanup_calls", 0)) + 1
        self._add_stat("cleanup_seconds", time.perf_counter() - cleanup_started)
        if self.retention_generations is not None:
            self._retained_bytes_cache = None
            self._retained_bytes()
        return removed


class IncrementalCheckpointStore:
    """Transactional append-only store for canonical indexed kNN chunks.

    Data chunks are immutable directories.  Every commit writes a new
    manifest which references the old chunks plus one new contiguous range;
    only after that manifest is durable is the pointer atomically promoted.
    The versioned store remains the compatibility path for LOO and candidate
    aggregation checkpoints; both stores expose operational write diagnostics.
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
        self._identity_cache: dict[str, tuple[Mapping[str, Any], dict[str, Any]]] = {}
        self._validated_heads: dict[str, tuple[Mapping[str, Any], int, int, str]] = {}
        self._stats: dict[str, float | int] = {
            "bytes_written": 0,
            "array_bytes_written": 0,
            "physical_bytes_written": 0,
            "metadata_bytes_written": 0,
            "bytes_read": 0,
            "array_bytes_read": 0,
            "physical_bytes_read": 0,
            "metadata_bytes_read": 0,
            "fsync_seconds": 0.0,
            "digest_seconds": 0.0,
            "write_seconds": 0.0,
            "commits": 0,
            "peak_retained_bytes": 0,
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
                elif path.is_file() and path.name.endswith(".incremental.pointer.json"):
                    total += int(path.stat().st_size)
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
        pointer_path = self._pointer(checkpoint_name)
        try:
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            pointer_bytes = int(pointer_path.stat().st_size)
            self._add_stat("metadata_bytes_read", pointer_bytes)
            self._add_stat("physical_bytes_read", pointer_bytes)
            if pointer.get("schema") != INCREMENTAL_KNN_CHECKPOINT_SCHEMA_VERSION:
                raise CheckpointValidationError(
                    "incremental checkpoint pointer schema is incompatible"
                )
            manifest_name = pointer.get("manifest")
            if not self._safe_file_name(manifest_name):
                raise CheckpointValidationError("incremental checkpoint manifest path is invalid")
            manifest_path = self.root / manifest_name
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_bytes = int(manifest_path.stat().st_size)
            self._add_stat("metadata_bytes_read", manifest_bytes)
            self._add_stat("physical_bytes_read", manifest_bytes)
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
        manifest: Mapping[str, Any],
        expected_identity: Mapping[str, Any] | None,
        expected_fingerprints: set[str] | None = None,
    ) -> dict[str, Any]:
        identity = manifest.get("identity")
        if not isinstance(identity, dict):
            raise CheckpointValidationError("incremental checkpoint identity is missing")
        if sha256_canonical_json(identity) != manifest.get("identity_fingerprint"):
            raise CheckpointValidationError("incremental checkpoint identity checksum mismatch")
        if expected_identity is not None:
            expected_fingerprints = expected_fingerprints or {
                sha256_canonical_json(dict(expected_identity)),
                sha256_canonical_json(_compact_incremental_identity(expected_identity)),
            }
            if manifest.get("identity_fingerprint") not in expected_fingerprints:
                raise CheckpointValidationError(
                    "incremental checkpoint identity does not match the ordered input"
                )
        return _compact_incremental_identity(identity)

    def _load_chain(
        self,
        name: str,
        *,
        expected_identity: Mapping[str, Any] | None,
        k_max: int | None,
        location_count: int | None,
    ) -> tuple[str, dict[str, Any], int, int, int, list[dict[str, Any]]]:
        manifest_name, manifest = self._read_manifest_with_name(name)
        head_manifest_name = manifest_name
        identity: dict[str, Any] | None = None
        saved_k: int | None = None
        saved_count: int | None = None
        newest_cursor: int | None = None
        newest_chunk_count: int | None = None
        reverse_chunks: list[dict[str, Any]] = []
        reverse_chunk_counts: list[int] = []
        seen: set[str] = set()
        expected_fingerprints = None
        if expected_identity is not None:
            expected_fingerprints = {
                sha256_canonical_json(dict(expected_identity)),
                sha256_canonical_json(_compact_incremental_identity(expected_identity)),
            }
        while True:
            if manifest_name in seen:
                raise CheckpointValidationError("incremental checkpoint manifest chain loops")
            seen.add(manifest_name)
            current_identity = self._validate_identity(
                manifest,
                expected_identity,
                expected_fingerprints,
            )
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
            chunk_stop = chunk.get("stop")
            if (
                isinstance(chunk_stop, bool)
                or not isinstance(chunk_stop, int)
                or chunk_stop != current_cursor
            ):
                raise CheckpointValidationError(
                    "incremental checkpoint cursor does not match its chunk"
                )
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
            reverse_chunk_counts.append(current_chunk_count)
            previous = manifest.get("previous_manifest")
            if previous is None:
                break
            if not self._safe_file_name(previous) or previous in seen:
                raise CheckpointValidationError(
                    "incremental checkpoint previous manifest is invalid"
                )
            try:
                previous_path = self.root / previous
                manifest = json.loads(previous_path.read_text(encoding="utf-8"))
                previous_bytes = int(previous_path.stat().st_size)
                self._add_stat("metadata_bytes_read", previous_bytes)
                self._add_stat("physical_bytes_read", previous_bytes)
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
        chunk_counts = list(reversed(reverse_chunk_counts))
        if any(count != index for index, count in enumerate(chunk_counts, start=1)):
            raise CheckpointValidationError(
                "incremental checkpoint chunk count chain is not authoritative"
            )
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
        return head_manifest_name, identity, saved_k, saved_count, newest_cursor, chunks

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
        cached_identity = self._identity_cache.get(checkpoint_name)
        if cached_identity is not None and cached_identity[0] is identity:
            stored_identity = cached_identity[1]
        else:
            stored_identity = _compact_incremental_identity(identity)
            self._identity_cache[checkpoint_name] = (identity, stored_identity)
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
            current_manifest_name, _ = self._read_manifest_with_name(checkpoint_name)
            previous_manifest = current_manifest_name
            validated_head = self._validated_heads.get(checkpoint_name)
            if validated_head is None or validated_head[0] is not identity:
                validated_manifest_name, _, _, saved_count, saved_cursor, chunks = self._load_chain(
                    checkpoint_name,
                    expected_identity=identity,
                    k_max=k_max,
                    location_count=location_count,
                )
                self._validated_heads[checkpoint_name] = (
                    identity,
                    saved_cursor,
                    len(chunks),
                    validated_manifest_name,
                )
                previous_chunk_count = len(chunks)
            else:
                _, saved_cursor, previous_chunk_count, validated_manifest_name = validated_head
                if validated_manifest_name != current_manifest_name:
                    raise CheckpointValidationError(
                        "incremental checkpoint pointer changed since the cached head was validated"
                    )
                saved_count = location_count
            if saved_count != location_count:
                raise CheckpointValidationError("incremental checkpoint dimensions do not match")
            if saved_cursor != start:
                raise CheckpointValidationError("incremental checkpoint chunk is not contiguous")
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
        array_bytes_written = 0
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
                array_bytes_written += byte_count
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
            _atomic_replace(temporary, final)
            self._fsync_directory(self.root)

            manifest_name = f"{checkpoint_name}.incremental.manifest-{uuid.uuid4().hex}.json"
            manifest_payload = {
                "schema": INCREMENTAL_KNN_CHECKPOINT_SCHEMA_VERSION,
                "name": checkpoint_name,
                "identity": stored_identity,
                "identity_fingerprint": sha256_canonical_json(stored_identity),
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
            manifest_bytes = int(manifest_temporary.stat().st_size)
            _atomic_replace(manifest_temporary, self.root / manifest_name)
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
            pointer_bytes = int(pointer_temporary.stat().st_size)
            _atomic_replace(pointer_temporary, self._pointer(checkpoint_name))
            self._validated_heads[checkpoint_name] = (
                identity,
                stop,
                previous_chunk_count + 1,
                manifest_name,
            )
            self._fsync_directory(self.root)
            bytes_written = array_bytes_written + manifest_bytes + pointer_bytes
            metadata_bytes_written = manifest_bytes + pointer_bytes
            self._add_stat("array_bytes_written", array_bytes_written)
            self._add_stat("bytes_written", array_bytes_written)
            self._add_stat("physical_bytes_written", bytes_written)
            self._add_stat("metadata_bytes_written", metadata_bytes_written)
            self._add_stat("commits", 1)
            self._add_stat("write_seconds", time.perf_counter() - started)
            self._validated_heads[checkpoint_name] = (
                identity,
                stop,
                previous_chunk_count + 1,
                manifest_name,
            )
            self._stats["peak_retained_bytes"] = max(
                int(self._stats.get("peak_retained_bytes", 0)), self._retained_bytes()
            )
            return {
                "bytes_written": array_bytes_written,
                "array_bytes_written": array_bytes_written,
                "metadata_bytes_written": metadata_bytes_written,
                "physical_bytes_written": bytes_written,
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

        manifest_name, identity, saved_k, saved_count, cursor, chunks = self._load_chain(
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
                self._add_stat("array_bytes_read", byte_count)
                self._add_stat("physical_bytes_read", byte_count)
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
        cache_identity = expected_identity if expected_identity is not None else identity
        self._identity_cache[_safe_name(name)] = (
            cache_identity,
            _compact_incremental_identity(cache_identity),
        )
        self._validated_heads[_safe_name(name)] = (
            cache_identity,
            cursor,
            len(chunks),
            manifest_name,
        )
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
        checkpoint_name = _safe_name(name)
        self._identity_cache.pop(checkpoint_name, None)
        self._validated_heads.pop(checkpoint_name, None)
        self._pointer(checkpoint_name).unlink(missing_ok=True)
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
    """Batched, row-granular checkpoint for candidate CSR aggregation.

    Candidate directories are published atomically as soon as their arrays are
    durable.  The growing manifest is published only at bounded commit points,
    so a cancelled run still resumes from a contiguous durable prefix without
    rewriting all previous metadata for every candidate.
    """

    DEFAULT_COMMIT_EVERY_CANDIDATES = 32
    DEFAULT_COMMIT_EVERY_SECONDS = 30.0

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "aggregation-manifest.json"
        self._manifest_cache: dict[str, Any] | None = None
        self._pending_rows: dict[str, dict[str, Any]] = {}
        self._bytes_on_disk: int | None = None
        self._manifest_bytes = 0
        self._metrics: dict[str, float | int] = {
            "manifest_commits": 0,
            "manifest_bytes_written": 0,
            "row_bytes_written": 0,
            "manifest_seconds": 0.0,
            "row_serialization_seconds": 0.0,
            "filesystem_stat_seconds": 0.0,
        }

    def _path_bytes(self, path: Path) -> int:
        started = time.perf_counter()
        try:
            if path.is_file():
                return int(path.stat().st_size)
            if not path.is_dir():
                return 0
            total = 0
            for child in path.rglob("*"):
                if child.is_file():
                    total += int(child.stat().st_size)
            return int(total)
        except OSError as exc:
            raise CheckpointValidationError(
                f"unable to account checkpoint bytes below {path}"
            ) from exc
        finally:
            self._metrics["filesystem_stat_seconds"] = float(
                self._metrics["filesystem_stat_seconds"]
            ) + max(0.0, time.perf_counter() - started)

    def _refresh_bytes(self) -> None:
        self._bytes_on_disk = self._path_bytes(self.root)
        try:
            self._manifest_bytes = (
                int(self.manifest_path.stat().st_size)
                if self.manifest_path.is_file()
                else 0
            )
        except OSError as exc:
            raise CheckpointValidationError(
                "unable to stat candidate aggregation manifest"
            ) from exc

    def _row_bytes(self, path: Path) -> int:
        """Account one known row layout without scanning the checkpoint root."""

        started = time.perf_counter()
        if not path.exists():
            return 0
        if not path.is_dir():
            raise CheckpointValidationError(f"candidate aggregation row is not a directory: {path}")
        total = 0
        known = {
            "target_indices.npy",
            "values.npy",
            "source_support.npy",
            "manifest.json",
        }
        try:
            children = {child.name: child for child in path.iterdir()}
            if set(children).difference(known):
                raise CheckpointValidationError(
                    f"candidate aggregation row contains an unknown file: {path.name}"
                )
            for child in children.values():
                if not child.is_file():
                    raise CheckpointValidationError(
                        f"candidate aggregation row contains a non-file entry: {child}"
                    )
                total += int(child.stat().st_size)
        except OSError as exc:
            raise CheckpointValidationError(
                f"unable to account candidate aggregation row: {path}"
            ) from exc
        finally:
            self._metrics["filesystem_stat_seconds"] = float(
                self._metrics["filesystem_stat_seconds"]
            ) + max(0.0, time.perf_counter() - started)
        return int(total)

    @property
    def bytes_on_disk(self) -> int:
        """Durable bytes, refreshed only on initialization/explicit cleanup."""

        if self._bytes_on_disk is None:
            self._refresh_bytes()
        assert self._bytes_on_disk is not None
        return int(self._bytes_on_disk)

    @property
    def manifest_bytes(self) -> int:
        if self._bytes_on_disk is None:
            self._refresh_bytes()
        return int(self._manifest_bytes)

    @property
    def pending_row_count(self) -> int:
        return len(self._pending_rows)

    def metrics_snapshot(self) -> dict[str, float | int]:
        return dict(self._metrics)

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
            if not isinstance(manifest.get("completed"), dict):
                raise CheckpointValidationError(
                    "candidate aggregation checkpoint completed rows are invalid"
                )
            self._manifest_cache = manifest
        else:
            self._manifest_cache = {
                "schema": CHECKPOINT_SCHEMA_VERSION,
                "identity": dict(identity),
                "identity_fingerprint": fingerprint,
                "candidate_ids": list(candidate_ids),
                "completed": {},
            }
            self._write_manifest(self._manifest_cache)
        self._pending_rows.clear()
        # The only recursive accounting pass is at initialization/resume.
        self._refresh_bytes()

    def _read_manifest(self) -> dict[str, Any]:
        if self._manifest_cache is not None:
            return self._manifest_cache
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CheckpointValidationError("candidate aggregation manifest is unreadable") from exc
        if manifest.get("schema") != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointValidationError(
                "candidate aggregation checkpoint schema is incompatible"
            )
        self._manifest_cache = manifest
        return manifest

    def _write_manifest(self, manifest: Mapping[str, Any]) -> int:
        started = time.perf_counter()
        temporary = self.root / f".aggregation-manifest-{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(dict(manifest), handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            byte_count = int(temporary.stat().st_size)
            _atomic_replace(temporary, self.manifest_path)
            _fsync_directory(self.root)
            self._metrics["manifest_bytes_written"] = int(
                self._metrics["manifest_bytes_written"]
            ) + byte_count
            self._metrics["manifest_commits"] = int(self._metrics["manifest_commits"]) + 1
            return byte_count
        finally:
            self._metrics["manifest_seconds"] = float(
                self._metrics["manifest_seconds"]
            ) + max(0.0, time.perf_counter() - started)
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
        commit: bool = True,
    ) -> None:
        self._read_manifest()
        key = str(int(candidate_index))
        if key in self._pending_rows:
            raise CheckpointValidationError(
                f"candidate aggregation checkpoint row is already pending: {candidate_index}"
            )
        row_name = f"row-{int(candidate_index):08d}"
        temporary = self.root / f".{row_name}-{uuid.uuid4().hex}.tmp"
        final = self.root / row_name
        started = time.perf_counter()
        previous_bytes = self._row_bytes(final) if final.exists() else 0
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
            _atomic_replace(temporary, final)
            final_bytes = self._row_bytes(final)
            self._bytes_on_disk = max(
                0,
                self.bytes_on_disk - previous_bytes + final_bytes,
            )
            self._metrics["row_bytes_written"] = int(
                self._metrics["row_bytes_written"]
            ) + final_bytes
            self._pending_rows[key] = row_manifest
            if commit:
                self.commit()
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        finally:
            self._metrics["row_serialization_seconds"] = float(
                self._metrics["row_serialization_seconds"]
            ) + max(0.0, time.perf_counter() - started)

    def commit(self) -> bool:
        """Publish all complete pending rows in one atomic manifest snapshot."""

        if not self._pending_rows:
            return False
        manifest = self._read_manifest()
        completed = dict(manifest.get("completed", {}))
        completed.update(
            {
                key: self._pending_rows[key]
                for key in sorted(self._pending_rows, key=lambda value: int(value))
            }
        )
        updated = dict(manifest)
        updated["completed"] = completed
        old_manifest_bytes = self.manifest_bytes
        new_manifest_bytes = self._write_manifest(updated)
        self._manifest_cache = updated
        self._manifest_bytes = new_manifest_bytes
        self._bytes_on_disk = max(
            0,
            self.bytes_on_disk - old_manifest_bytes + new_manifest_bytes,
        )
        self._pending_rows.clear()
        return True

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
        self._refresh_bytes()
        return removed

    def clear(self) -> None:
        for path in self.root.iterdir():
            if path.is_dir() and (path.name.startswith("row-") or path.name.startswith(".")):
                shutil.rmtree(path, ignore_errors=True)
            elif path.is_file() and path.name.startswith("."):
                path.unlink(missing_ok=True)
        self.manifest_path.unlink(missing_ok=True)
        self._manifest_cache = None
        self._pending_rows.clear()
        self._bytes_on_disk = 0
        self._manifest_bytes = 0


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "INCREMENTAL_KNN_CHECKPOINT_SCHEMA_VERSION",
    "CheckpointValidationError",
    "CandidateAggregationCheckpoint",
    "IncrementalCheckpointStore",
    "VersionedCheckpointStore",
]
