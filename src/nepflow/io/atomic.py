"""Atomic filesystem writes for authoritative NEPFlow artifacts."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path


def _sync_directory(directory: Path) -> None:
    """Sync a replaced file's parent directory when durability is requested.

    Directory descriptors cannot be opened on every supported platform (notably
    Windows), so directory syncing remains an explicit platform exception there.
    """

    if os.name == "nt":
        return
    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_write_bytes(
    path: str | Path,
    data: bytes,
    *,
    durable: bool = True,
) -> None:
    """Replace *path* with *data* without exposing a partial file.

    The temporary file is created beside the target so ``os.replace`` remains
    an atomic same-filesystem operation.  Any failed temporary write or replace
    removes its abandoned temporary file and leaves an existing target alone.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    file_descriptor: int | None = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(file_descriptor, "wb") as handle:
            file_descriptor = None
            handle.write(data)
            handle.flush()
            if durable:
                os.fsync(handle.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
        if durable:
            _sync_directory(target.parent)
    except BaseException:
        # Cleanup must also run for KeyboardInterrupt/SystemExit so an
        # interrupted atomic write never leaves an open descriptor or temp
        # artifact.  The original exception is always re-raised below.
        if file_descriptor is not None:
            try:
                os.close(file_descriptor)
            except OSError:
                # The original write failure remains authoritative.
                file_descriptor = None
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                temporary_path = None
            except OSError:
                # Cleanup is best effort; preserve the original write error.
                temporary_path = None
        raise


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
    durable: bool = True,
) -> None:
    """Atomically write UTF-8 (or explicitly selected) text."""

    atomic_write_bytes(path, text.encode(encoding), durable=durable)


def atomic_write_stream(
    path: str | Path,
    writer: Callable[[object], None],
    *,
    durable: bool = True,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Stream an artifact to a same-directory temporary file and publish it.

    The callback receives the open binary file.  The completed temporary file
    is hashed incrementally before ``os.replace`` publishes it, so callers do
    not need to construct a second in-memory copy of a large artifact.
    """

    import hashlib

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    file_descriptor: int | None = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(file_descriptor, "w+b") as handle:
            file_descriptor = None
            writer(handle)
            handle.flush()
            if durable:
                os.fsync(handle.fileno())
            handle.seek(0)
            digest = hashlib.sha256()
            while True:
                block = handle.read(chunk_size)
                if not block:
                    break
                digest.update(block)
            handle.seek(0)
        os.replace(temporary_path, target)
        temporary_path = None
        if durable:
            _sync_directory(target.parent)
        return digest.hexdigest()
    except BaseException:
        if file_descriptor is not None:
            try:
                os.close(file_descriptor)
            except OSError:
                pass
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass
        raise


__all__ = ["atomic_write_bytes", "atomic_write_stream", "atomic_write_text"]
