"""Transactional, checksummed checkpoints for long-running entropy phases."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from nepflow.io.hashing import sha256_canonical_json

CHECKPOINT_SCHEMA_VERSION = "entropy-checkpoint-v1"


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
    "CheckpointValidationError",
    "CandidateAggregationCheckpoint",
    "VersionedCheckpointStore",
]
