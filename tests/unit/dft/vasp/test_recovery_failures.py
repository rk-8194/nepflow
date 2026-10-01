from configparser import ConfigParser

from nepflow.dft.vasp.failures import (
    VaspFailureEvidence,
    classify_failure,
)
from nepflow.dft.vasp.recovery import build_retry_levels_for_gpu, decide_retry


def test_failure_classification_distinguishes_oom_from_incomplete(tmp_path) -> None:
    oom = tmp_path / "oom"
    oom.mkdir()
    (oom / ".vasp_oom_detected").write_text("1")
    evidence = VaspFailureEvidence.from_job_directory(oom)
    assert classify_failure(evidence).kind == "out_of_memory"

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    assert classify_failure(
        VaspFailureEvidence.from_job_directory(incomplete)
    ).kind == "incomplete_output"


def test_recovery_decision_records_the_next_resource_level() -> None:
    config = ConfigParser()
    levels = build_retry_levels_for_gpu(1, 2, 1, config)
    decision = decide_retry(levels, 0, max_retry_level=6)
    assert decision.retry
    assert decision.retry_level == 1
    assert decision.reason == "oom_escalation"
    exhausted = decide_retry(levels, len(levels), max_retry_level=6)
    assert not exhausted.retry
