"""Typed execution of external processes.

This module owns process invocation mechanics only.  It deliberately does not
interpret scheduler output or assign meaning to scheduler return codes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any

from nepflow.errors import ProcessError


_CommandPart = str | os.PathLike[str]


def _normalise_command(command: Sequence[_CommandPart]) -> tuple[str, ...]:
    """Convert an argument-list command to an immutable, text-only tuple."""

    if isinstance(command, (str, bytes)):
        raise TypeError("ProcessRunner commands must be an argument list, not a string")

    values: list[str] = []
    for part in command:
        if isinstance(part, bytes):
            raise TypeError("ProcessRunner command arguments must be text")
        value = os.fspath(part)
        if not isinstance(value, str):
            raise TypeError("ProcessRunner command arguments must be text")
        values.append(value)
    if not values:
        raise ValueError("ProcessRunner command must not be empty")
    return tuple(values)


def _normalise_environment(
    env: Mapping[str, str] | None,
    *,
    inherit_environment: bool,
) -> dict[str, str]:
    """Build an explicit environment overlay or a full replacement."""

    values = {str(key): str(value) for key, value in (env or {}).items()}
    if inherit_environment:
        inherited = dict(os.environ)
        inherited.update(values)
        return inherited
    return values


def _text_output(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _display_command(command: Sequence[str]) -> str:
    """Return a shell-escaped representation for logs, never for execution."""

    return shlex.join(command)


@dataclass(frozen=True, slots=True)
class ProcessResult:
    """Result of one external process invocation."""

    command: tuple[str, ...]
    cwd: str | None
    returncode: int
    stdout: str | None
    stderr: str | None
    duration_seconds: float

    @property
    def ok(self) -> bool:
        """Whether the process returned zero."""

        return self.returncode == 0


class ProcessRunner:
    """Run argument-list commands with one consistent failure contract."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("nepflow.hpc.process")

    def run(
        self,
        command: Sequence[_CommandPart],
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        inherit_environment: bool = True,
        timeout: float | None = None,
        check: bool = True,
        capture_output: bool = True,
    ) -> ProcessResult:
        """Run an argument-list command without shell interpretation.

        ``env`` overlays the current environment by default.  Pass
        ``inherit_environment=False`` for an explicit full replacement.
        ``check=False`` returns nonzero results for callers that intentionally
        classify them; execution failures and timeouts always raise
        :class:`~nepflow.errors.ProcessError`.
        """

        normalised = _normalise_command(command)
        return self._execute(
            normalised,
            cwd=cwd,
            env=env,
            inherit_environment=inherit_environment,
            timeout=timeout,
            check=check,
            capture_output=capture_output,
            shell=False,
        )

    def run_shell(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        inherit_environment: bool = True,
        timeout: float | None = None,
        check: bool = True,
        capture_output: bool = True,
    ) -> ProcessResult:
        """Run an explicitly requested shell command.

        This is separate from :meth:`run` because shell parsing is never an
        implicit fallback.  Callers must preserve a concrete reason for using
        a shell command, such as an existing command string with redirection.
        """

        if not isinstance(command, str) or not command:
            raise TypeError("ProcessRunner shell commands must be non-empty text")
        return self._execute(
            (command,),
            cwd=cwd,
            env=env,
            inherit_environment=inherit_environment,
            timeout=timeout,
            check=check,
            capture_output=capture_output,
            shell=True,
        )

    def _execute(
        self,
        command: tuple[str, ...],
        *,
        cwd: str | Path | None,
        env: Mapping[str, str] | None,
        inherit_environment: bool,
        timeout: float | None,
        check: bool,
        capture_output: bool,
        shell: bool,
    ) -> ProcessResult:
        cwd_value = None if cwd is None else os.fspath(cwd)
        if not isinstance(cwd_value, (str, type(None))):
            raise TypeError("ProcessRunner cwd must be a path or text")
        environment = _normalise_environment(
            env,
            inherit_environment=inherit_environment,
        )
        display_command = _display_command(command)
        self.logger.debug(
            "Running external command: %s (cwd=%s)",
            display_command,
            cwd_value or "<inherited>",
        )

        started = time.monotonic()
        try:
            completed = subprocess.run(
                command[0] if shell else command,
                shell=shell,
                cwd=cwd_value,
                env=environment,
                timeout=timeout,
                check=False,
                capture_output=capture_output,
                text=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise ProcessError(
                f"External command timed out after {timeout}s: {display_command}",
                command=command,
                cwd=cwd_value,
                returncode=None,
                stdout=_text_output(exc.stdout),
                stderr=_text_output(exc.stderr),
                kind="timeout",
            ) from exc
        except FileNotFoundError as exc:
            raise ProcessError(
                f"Executable not found for external command: {display_command}",
                command=command,
                cwd=cwd_value,
                returncode=None,
                stdout=None,
                stderr=None,
                kind="not_found",
            ) from exc
        except OSError as exc:
            raise ProcessError(
                f"Could not execute external command {display_command}: {exc}",
                command=command,
                cwd=cwd_value,
                returncode=None,
                stdout=None,
                stderr=None,
                kind="os_error",
            ) from exc

        result = ProcessResult(
            command=command,
            cwd=cwd_value,
            returncode=completed.returncode,
            stdout=_text_output(completed.stdout),
            stderr=_text_output(completed.stderr),
            duration_seconds=time.monotonic() - started,
        )
        if check and not result.ok:
            raise ProcessError(
                f"External command exited with status {result.returncode}: {display_command}",
                command=result.command,
                cwd=result.cwd,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                kind="nonzero",
            )
        return result


__all__ = ["ProcessError", "ProcessResult", "ProcessRunner"]
