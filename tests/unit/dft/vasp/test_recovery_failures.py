import pytest

import nepflow.dft.vasp.recovery as recovery_module
from nepflow.config.models import NepflowConfig
from nepflow.dft.backend import DftCompletionEvidence, DftFailureEvidence, DftInputArtifacts
from nepflow.dft.vasp.backend import VaspBackend
from nepflow.dft.vasp.failures import (
    VaspFailureEvidence,
    classify_failure,
)
from nepflow.dft.vasp.inputs import hash_incar_text
from nepflow.dft.vasp.recovery import (
    build_retry_levels_for_config,
    decide_retry,
    write_incar_resource_parameters,
)
from nepflow.domain.identities import DftCalculationIdentity
from nepflow.errors import VaspError
from nepflow.hpc.process import ProcessResult


def test_failure_classification_distinguishes_oom_from_incomplete(tmp_path) -> None:
    oom = tmp_path / "oom"
    oom.mkdir()
    (oom / ".vasp_oom_detected").write_text("1")
    (oom / ".vasp_exit_code").write_text("137")
    evidence = VaspFailureEvidence.from_job_directory(oom)
    assert (oom / ".vasp_exit_code").read_text() == "137"
    assert classify_failure(evidence).kind == "out_of_memory"

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    assert (
        classify_failure(VaspFailureEvidence.from_job_directory(incomplete)).kind
        == "incomplete_output"
    )


def test_recovery_decision_records_the_next_resource_level() -> None:
    levels = build_retry_levels_for_config(1, 2, 1, NepflowConfig())
    decision = decide_retry(levels, 0, max_retry_level=6)
    assert decision.retry
    assert decision.retry_level == 1
    assert decision.reason == "oom_escalation"
    exhausted = decide_retry(levels, len(levels), max_retry_level=6)
    assert not exhausted.retry


def test_backend_classifies_process_evidence_without_fabricating_paths() -> None:
    calculation = DftCalculationIdentity("structure", "incar", "potcar")
    backend = VaspBackend()
    failed = backend.classify_failure(
        DftFailureEvidence(
            calculation=calculation,
            completion=DftCompletionEvidence(completed=False),
            process=ProcessResult(
                command=("vasp_std",),
                cwd=None,
                returncode=9,
                stdout="",
                stderr="",
                duration_seconds=0.1,
            ),
        )
    )
    assert failed.kind == "vasp_execution_failed"
    assert not failed.recoverable

    oom = backend.classify_failure(
        DftFailureEvidence(
            calculation=calculation,
            completion=DftCompletionEvidence(completed=False),
            markers=("oom_kill",),
        )
    )
    assert oom.kind == "out_of_memory"
    assert oom.recoverable


def test_backend_reads_runner_exit_evidence_for_scheduler_reconciliation(tmp_path) -> None:
    calculation = DftCalculationIdentity("structure", "incar", "potcar")
    working = tmp_path / "failed"
    working.mkdir()
    (working / ".vasp_exit_code").write_text("9")
    backend = VaspBackend()
    completion = backend.parse_completion(
        DftInputArtifacts(calculation=calculation, working_directory=working, files=())
    )

    assert completion.returncode == 9
    failure = backend.classify_failure(
        DftFailureEvidence(calculation=calculation, completion=completion)
    )
    assert failure.kind == "vasp_execution_failed"
    assert not failure.recoverable


@pytest.mark.parametrize(
    ("name", "log", "exit_code", "scheduler_state", "expected_kind", "recoverable"),
    (
        (
            "missing-executable",
            "prterun was unable to find the specified executable file: vasp_std\n",
            None,
            "failed",
            "vasp_execution_failed",
            False,
        ),
        ("runner-exit", "", 127, None, "vasp_execution_failed", False),
        ("slurm-oom", "", None, "oom", "out_of_memory", True),
        ("runner-oom", "", 137, None, "out_of_memory", True),
        ("timeout", "", None, "timeout", "timeout", False),
        ("cancelled", "", None, "cancelled", "cancelled", False),
        ("scheduler-failed", "", None, "failed", "scheduler_execution_failed", False),
        ("empty", "", None, None, "incomplete_output", True),
    ),
)
def test_backend_classifies_scheduler_and_runner_evidence(
    tmp_path,
    name: str,
    log: str,
    exit_code: int | None,
    scheduler_state: str | None,
    expected_kind: str,
    recoverable: bool,
) -> None:
    working = tmp_path / name
    working.mkdir()
    if log:
        (working / "vasp_output.log").write_text(log)
    if exit_code is not None:
        (working / ".vasp_exit_code").write_text(str(exit_code))
    calculation = DftCalculationIdentity("structure", "incar", "potcar")

    failure = VaspBackend().classify_failure(
        DftFailureEvidence(
            calculation=calculation,
            completion=DftCompletionEvidence(completed=False),
            working_directory=working,
            scheduler_state=scheduler_state,
            scheduler_reason="test scheduler reason" if scheduler_state else None,
        )
    )

    assert failure.kind == expected_kind
    assert failure.recoverable is recoverable


def test_recovery_application_fails_explicitly_and_preserves_scientific_hash(tmp_path) -> None:
    with pytest.raises(VaspError):
        write_incar_resource_parameters(tmp_path, 4, 2)

    incar = tmp_path / "INCAR"
    original = "ENCUT = 520\nNCORE = 2\nKPAR = 1\n"
    incar.write_text(original)
    before = hash_incar_text(original)
    write_incar_resource_parameters(tmp_path, 8, 4)
    after = incar.read_text()
    assert "NCORE = 8" in after
    assert "KPAR = 4" in after
    assert hash_incar_text(after) == before


def test_recovery_write_failure_is_typed(monkeypatch, tmp_path) -> None:
    incar = tmp_path / "INCAR"
    incar.write_text("NCORE = 2\nKPAR = 1\n")

    def fail_write(path, *args, **kwargs):
        if path == incar:
            raise OSError("read-only test")

    monkeypatch.setattr(recovery_module, "atomic_write_text", fail_write)
    with pytest.raises(VaspError, match="Could not apply"):
        write_incar_resource_parameters(tmp_path, 8, 4)
