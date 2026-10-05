import pytest

import nepflow.dft.vasp.recovery as recovery_module
from nepflow.config.models import NepflowConfig
from nepflow.dft.backend import DftCompletionEvidence, DftFailureEvidence
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
    evidence = VaspFailureEvidence.from_job_directory(oom)
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
