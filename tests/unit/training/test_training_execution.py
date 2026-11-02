from __future__ import annotations

from pathlib import Path

from nepflow.stages.training.execution import TrainingExecution


def test_training_script_records_success_only_after_backend_command(tmp_path: Path) -> None:
    execution = TrainingExecution(
        backend=object(),
        resources=None,
        input_for=lambda _candidate: None,
    )

    script = execution._script_content(
        tmp_path,
        ("nep", "--quiet"),
        model_run_id="model-run-1",
        attempt_id="model-run-1:attempt:1",
        attempt_number=1,
    )

    command_position = script.index("nep --quiet")
    marker_position = script.index("mv -f --")
    assert "set -euo pipefail" in script
    assert "rm -f -- \"$success_marker\"" in script
    assert ".nepflow_training_success.json" in script
    assert "model-run-1" in script
    assert command_position < marker_position


def test_later_execution_generation_uses_a_distinct_attempt_directory(tmp_path: Path) -> None:
    candidate = type("Candidate", (), {"run_directory": tmp_path})()

    assert TrainingExecution.attempt_directory(candidate, 1) == tmp_path / "attempts" / "a0001"
    assert TrainingExecution.attempt_directory(candidate, 1, 2) == (
        tmp_path / "attempts" / "g0002" / "a0001"
    )
