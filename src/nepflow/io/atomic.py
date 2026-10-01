"""Atomic filesystem writes for authoritative NEPFlow artifacts."""

from __future__ import annotations

import os
import tempfile
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
        if file_descriptor is not None:
            try:
                os.close(file_descriptor)
            except OSError:
                pass
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
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


__all__ = ["atomic_write_bytes", "atomic_write_text"]
