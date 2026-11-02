"""Materialize and submit one NEP training attempt."""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping

from nepflow.errors import StateError
from nepflow.hpc.resources import JobResources, render_shell_script
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
        site_prologue: tuple[str, ...] = (),
    ) -> None:
        self.backend = backend
        self.resources = resources
        self.input_for = input_for
        self.site_prologue = tuple(site_prologue)

    @staticmethod
    def attempt_directory(
        candidate: Any,
        attempt_number: int,
        execution_generation: int = 1,
    ) -> Path:
        generation_root = (
            candidate.run_directory / "attempts"
            if execution_generation == 1
            else candidate.run_directory / "attempts" / f"g{execution_generation:04d}"
        )
        return generation_root / f"a{attempt_number:04d}"

    def write_script(
        self,
        candidate: Any,
        attempt: Any,
        *,
        rewrite: bool = True,
    ) -> tuple[Path, str, Mapping[str, Any]]:
        """Materialize and fingerprint one deterministic attempt command."""

        execution_directory = attempt.execution_directory or self.attempt_directory(
            candidate,
            attempt.attempt_number,
            getattr(attempt, "execution_generation", 1),
        )
        execution_directory.mkdir(parents=True, exist_ok=True)
        self._copy_inputs(candidate, attempt, execution_directory)
        command = tuple(self.backend.training_command(self.input_for(candidate)))
        if not command:
            raise StateError(f"NEP backend returned an empty command for {candidate.model_run_id}")
        script_path = execution_directory / "train_nep.sh"
        script_content = self._script_content(
            execution_directory,
            command,
            model_run_id=candidate.model_run_id,
            attempt_id=attempt.attempt_id,
            attempt_number=attempt.attempt_number,
            execution_generation=getattr(attempt, "execution_generation", 1),
            generation_offset=getattr(attempt, "generation_offset", 0),
            segment_target_generations=getattr(
                attempt,
                "segment_target_generations",
                getattr(candidate, "target_generations", 0),
            ),
        )
        self._write_script_file(script_path, script_content, rewrite, attempt)
        execution_config = self._execution_config(
            candidate, attempt, command, execution_directory, script_path
        )
        execution_config_hash = sha256_bytes(canonical_json_bytes(execution_config))
        return script_path, execution_config_hash, execution_config

    def _copy_inputs(self, candidate: Any, attempt: Any, execution_directory: Path) -> None:
        source_directory = Path(candidate.run_directory)
        for filename in ("nep.in", "train.xyz", "test.xyz"):
            if filename == "nep.in":
                continue
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

        training_input = self.input_for(candidate)
        segment_target = int(
            getattr(attempt, "segment_target_generations", 0)
            or getattr(candidate, "target_generations", 0)
        )
        attempt_input = execution_directory / "nep.in"
        materialize = getattr(self.backend, "materialize_attempt_input", None)
        if callable(materialize) and segment_target > 0:
            materialize(training_input, attempt_input, segment_target)
        else:
            source = source_directory / "nep.in"
            if not source.is_file():
                raise StateError(f"Canonical NEP input is missing: {source}")
            if attempt_input.is_file() and sha256_file(attempt_input) != sha256_file(source):
                raise StateError(
                    f"Attempt execution input conflicts with candidate identity: {attempt_input}"
                )
            if not attempt_input.is_file():
                shutil.copy2(source, attempt_input)

        restart_source = getattr(attempt, "restart_source_attempt_id", None)
        if restart_source is None:
            return
        metadata = getattr(attempt, "checkpoint_metadata", None)
        if not isinstance(metadata, Mapping):
            raise StateError(
                f"Attempt {attempt.attempt_id} has no persisted checkpoint metadata for resume"
            )
        for name, path_key, hash_key in (
            ("nep.restart", "restart_path", "restart_sha256"),
            ("nep.txt", "model_path", "model_sha256"),
        ):
            source = Path(str(metadata.get(path_key, ""))).resolve()
            expected_hash = str(metadata.get(hash_key, ""))
            if not source.is_file() or not expected_hash or sha256_file(source) != expected_hash:
                raise StateError(
                    f"Persisted checkpoint artifact for {restart_source} is missing or changed: "
                    f"{source}"
                )
            destination = execution_directory / name
            shutil.copy2(source, destination)
            if sha256_file(destination) != expected_hash:
                raise StateError(
                    f"Copied checkpoint artifact failed hash verification: {destination}"
                )

    def _script_content(
        self,
        execution_directory: Path,
        command: tuple[Any, ...],
        *,
        model_run_id: str = "",
        attempt_id: str = "",
        attempt_number: int = 0,
        execution_generation: int = 1,
        generation_offset: int = 0,
        segment_target_generations: int = 0,
    ) -> str:
        success_marker = ".nepflow_training_success.json"
        marker_template = json.dumps(
            {
                "schema_version": "nepflow.training_success.v2",
                "model_run_id": model_run_id,
                "attempt_id": attempt_id,
                "attempt_number": attempt_number,
                "execution_generation": execution_generation,
                "generation_offset": generation_offset,
                "segment_target_generations": segment_target_generations,
                "process_exit_code": 0,
                "completion_timestamp": "__NEPFLOW_COMPLETION_TIMESTAMP__",
                "command": [str(part) for part in command],
            },
            separators=(",", ":"),
        )
        marker_prefix, marker_suffix = marker_template.split(
            '"__NEPFLOW_COMPLETION_TIMESTAMP__"', 1
        )
        body = (
            "set -euo pipefail\ncd "
            + shlex.quote(str(execution_directory))
            + "\n"
            + "success_marker="
            + shlex.quote(success_marker)
            + "\n"
            + "rm -f -- \"$success_marker\"\n"
            + "success_marker_tmp=\"${success_marker}.tmp.$$\"\n"
            + "trap 'rm -f -- \"$success_marker_tmp\"' EXIT\n"
            + " ".join(shlex.quote(os.path.expandvars(str(part))) for part in command)
            + "\n"
            + "completion_timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)\n"
            + "printf '%s%s%s\\n' "
            + shlex.quote(marker_prefix + '"')
            + " \"$completion_timestamp\" "
            + shlex.quote('"' + marker_suffix)
            + " > \"$success_marker_tmp\"\n"
            + "mv -f -- \"$success_marker_tmp\" \"$success_marker\"\n"
        )
        return render_shell_script(body, site_prologue=self.site_prologue)

    @staticmethod
    def _write_script_file(
        script_path: Path,
        script_content: str,
        rewrite: bool,
        attempt: Any,
    ) -> None:
        script_matches = (
            script_path.is_file() and script_path.read_text(encoding="utf-8") == script_content
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
        attempt_input_sha256 = sha256_file(execution_directory / "nep.in")
        return to_jsonable(
            {
                "target_generations": getattr(candidate, "target_generations", 0),
                "generation_offset": getattr(attempt, "generation_offset", 0),
                "segment_target_generations": getattr(
                    attempt,
                    "segment_target_generations",
                    getattr(candidate, "target_generations", 0),
                ),
                "attempt_nep_in_sha256": attempt_input_sha256,
                "attempt_local_nep_in_sha256": attempt_input_sha256,
                "restart_source_attempt_id": getattr(
                    attempt, "restart_source_attempt_id", None
                ),
                "checkpoint_metadata": getattr(attempt, "checkpoint_metadata", None),
                "restart_checkpoint_sha256": (
                    getattr(attempt, "checkpoint_metadata", {}).get("restart_sha256")
                    if isinstance(getattr(attempt, "checkpoint_metadata", None), Mapping)
                    else None
                ),
                "model_checkpoint_sha256": (
                    getattr(attempt, "checkpoint_metadata", {}).get("model_sha256")
                    if isinstance(getattr(attempt, "checkpoint_metadata", None), Mapping)
                    else None
                ),
                "attempt_number": attempt.attempt_number,
                "execution_generation": getattr(attempt, "execution_generation", 1),
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
