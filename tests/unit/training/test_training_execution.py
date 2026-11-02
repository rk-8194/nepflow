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
