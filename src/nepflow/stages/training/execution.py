"""Materialize and submit one NEP training attempt."""

from __future__ import annotations

from dataclasses import asdict
import logging
import os
from pathlib import Path
import shlex
import shutil
from typing import Any, Callable, Mapping

from nepflow.errors import StateError
from nepflow.hpc.resources import JobResources
from nepflow.io.atomic import atomic_write_text
from nepflow.io.hashing import sha256_bytes, sha256_file
from nepflow.io.json import canonical_json_bytes, to_jsonable


logger = logging.getLogger(__name__)


class TrainingExecution:
    """Own mutable attempt directories without owning campaign state."""

    def __init__(
        self,
        *,
        backend: Any,
        resources: JobResources | None,
        input_for: Callable[[Any], Any],
    ) -> None:
        self.backend = backend
        self.resources = resources
        self.input_for = input_for

    @staticmethod
    def attempt_directory(candidate: Any, attempt_number: int) -> Path:
        return candidate.run_directory / "attempts" / f"a{attempt_number:04d}"

    def write_script(
        self,
        candidate: Any,
        attempt: Any,
        *,
        rewrite: bool = True,
    ) -> tuple[Path, str, Mapping[str, Any]]:
        """Materialize and fingerprint one deterministic attempt command."""

        execution_directory = attempt.execution_directory or self.attempt_directory(
            candidate, attempt.attempt_number
        )
        execution_directory.mkdir(parents=True, exist_ok=True)
        self._copy_inputs(candidate.run_directory, execution_directory)
        command = tuple(self.backend.training_command(self.input_for(candidate)))
        if not command:
            raise StateError(f"NEP backend returned an empty command for {candidate.model_run_id}")
        script_path = execution_directory / "train_nep.sh"
        script_content = self._script_content(execution_directory, command)
        self._write_script_file(script_path, script_content, rewrite, attempt)
        execution_config = self._execution_config(
            candidate, attempt, command, execution_directory, script_path
        )
        execution_config_hash = sha256_bytes(canonical_json_bytes(execution_config))
        return script_path, execution_config_hash, execution_config

    @staticmethod
    def _copy_inputs(source_directory: Path, execution_directory: Path) -> None:
        for filename in ("nep.in", "train.xyz", "test.xyz"):
            source = source_directory / filename
            if not source.is_file():
                continue
            destination = execution_directory / filename
            if destination.is_file() and sha256_file(destination) != sha256_file(source):
                raise StateError(
                    f"Attempt execution input conflicts with candidate identity: {destination}"
                )
            if not destination.is_file():
                shutil.copy2(source, destination)

    @staticmethod
    def _script_content(execution_directory: Path, command: tuple[Any, ...]) -> str:
        return (
            "#!/bin/sh\nset -eu\ncd "
            + shlex.quote(str(execution_directory))
            + "\n"
            + " ".join(shlex.quote(os.path.expandvars(str(part))) for part in command)
            + "\n"
        )

    @staticmethod
    def _write_script_file(
        script_path: Path,
        script_content: str,
        rewrite: bool,
        attempt: Any,
    ) -> None:
        script_matches = (
            script_path.is_file()
            and script_path.read_text(encoding="utf-8") == script_content
        )
        if not script_matches and not rewrite:
            raise StateError(
                f"Persisted training script or execution command changed for "
                f"active attempt {attempt.attempt_id}"
            )
        if not script_matches:
            atomic_write_text(script_path, script_content, encoding="utf-8")
        try:
            script_path.chmod(0o755)
        except OSError:
            logger.debug("Could not mark training script executable: %s", script_path)

    def _execution_config(
        self,
        candidate: Any,
        attempt: Any,
        command: tuple[Any, ...],
        execution_directory: Path,
        script_path: Path,
    ) -> dict[str, Any]:
        return to_jsonable(
            {
                "attempt_number": attempt.attempt_number,
                "model_run_id": candidate.model_run_id,
                "command": list(command),
                "resources": asdict(self.resources) if self.resources is not None else None,
                "execution_directory": str(execution_directory.resolve()),
                "script_path": str(script_path.resolve()),
                "script_sha256": sha256_file(script_path),
            }
        )

    @staticmethod
    def job_id(value: Any) -> str:
        job_id = getattr(value, "job_id", None)
        if job_id is None and isinstance(value, Mapping):
            job_id = value.get("job_id")
        if not isinstance(job_id, str) or not job_id.strip():
            raise StateError("Scheduler submission returned no job_id")
        return job_id

    def submit(self, scheduler: Any, attempt: Any, job_name: str) -> str:
        submit_script = getattr(scheduler, "submit_script", None)
        if callable(submit_script):
            result = submit_script(
                attempt.script_path,
                resources=self.resources,
                job_name=job_name,
                cwd=attempt.execution_directory,
            )
        else:
            submit = getattr(scheduler, "submit", None)
            if not callable(submit):
                raise TypeError("TrainingCampaign scheduler has no submission API")
            result = submit(
                ("sbatch", str(attempt.script_path)),
                cwd=attempt.execution_directory,
                resources=self.resources,
            )
        return self.job_id(result)


__all__ = ["TrainingExecution"]
