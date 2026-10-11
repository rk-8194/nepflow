"""Project-scoped path contracts for persisted artifacts.

Persisted project artifacts use paths relative to the project root.  Runtime
callers may still use absolute ``Path`` objects; these helpers are the single
boundary where the two representations are converted and validated.
"""

from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath


def _is_absolute_path(value: str | Path) -> bool:
    """Recognise native and Windows absolute paths on every host OS."""

    raw = str(value)
    return Path(raw).is_absolute() or PureWindowsPath(raw).is_absolute()


def _raise(error_type: type[Exception], message: str) -> None:
    raise error_type(message)


def _resolved_inside(path: Path, root: Path, *, field: str, error_type: type[Exception]) -> Path:
    try:
        resolved = path.resolve(strict=False)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise error_type(
            f"{field} must resolve inside project root {root}: {path}"
        ) from exc
    return resolved


def project_relative_path(
    path: str | Path,
    project_root: str | Path,
    *,
    field: str = "project artifact path",
    error_type: type[Exception] = ValueError,
) -> str:
    """Return a validated POSIX project-relative representation of ``path``.

    Absolute paths are accepted only when they already belong to the supplied
    project root.  Relative paths are interpreted from that root, so traversal
    and out-of-root paths are rejected deterministically.
    """

    root = Path(project_root).resolve(strict=False)
    raw = str(path)
    if not raw.strip():
        _raise(error_type, f"{field} must not be blank")
    if _is_absolute_path(raw) and os.name != "nt" and PureWindowsPath(raw).is_absolute():
        _raise(error_type, f"{field} uses an absolute Windows path: {raw}")
    candidate = Path(raw) if Path(raw).is_absolute() else root / Path(raw)
    resolved = _resolved_inside(candidate, root, field=field, error_type=error_type)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:  # pragma: no cover - guarded by _resolved_inside
        raise error_type(f"{field} is outside project root {root}: {path}") from exc
    return relative.as_posix()


def resolve_project_path(
    value: str | Path,
    project_root: str | Path,
    *,
    field: str = "project artifact path",
    allow_legacy_absolute: bool = False,
    error_type: type[Exception] = ValueError,
) -> Path:
    """Resolve a persisted project path and reject traversal/out-of-root use.

    New persisted values must be relative.  ``allow_legacy_absolute`` is a
    narrow compatibility switch for consumers that explicitly support legacy
    state; even then, an absolute path must already be under the active root.
    It never authorises an arbitrary external path.
    """

    root = Path(project_root).resolve(strict=False)
    raw = str(value)
    if not raw.strip():
        _raise(error_type, f"{field} must not be blank")
    native_absolute = Path(raw).is_absolute()
    windows_absolute = PureWindowsPath(raw).is_absolute()
    if (native_absolute or windows_absolute) and not allow_legacy_absolute:
        _raise(error_type, f"{field} must be project-relative: {raw}")
    if windows_absolute and os.name != "nt":
        _raise(error_type, f"{field} uses an unsupported Windows path: {raw}")
    candidate = Path(raw) if native_absolute else root / Path(raw)
    return _resolved_inside(candidate, root, field=field, error_type=error_type)


def validate_project_relative_path(
    value: str | Path,
    *,
    field: str = "project path",
    error_type: type[Exception] = ValueError,
) -> Path:
    """Validate a path-valued project setting without resolving it at I/O time."""

    raw = str(value)
    if _is_absolute_path(raw):
        _raise(error_type, f"{field} must be relative to the project root: {raw}")
    normalised = Path(raw)
    if not raw.strip() or normalised == Path("."):
        _raise(error_type, f"{field} must name a project-relative directory")
    if any(part == ".." for part in normalised.parts):
        _raise(error_type, f"{field} must not traverse outside the project root: {raw}")
    return normalised


__all__ = [
    "project_relative_path",
    "resolve_project_path",
    "validate_project_relative_path",
]
