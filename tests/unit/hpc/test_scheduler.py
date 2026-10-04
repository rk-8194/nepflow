from pathlib import Path

import pytest

from nepflow.errors import ProcessError, SchedulerError
from nepflow.hpc.jobs import SchedulerJobState
from nepflow.hpc.process import ProcessResult
from nepflow.hpc.resources import JobResources, render_sbatch_directives, render_slurm_header
from nepflow.hpc.slurm import (
    SACCT_FORMAT,
    SQUEUE_FORMAT,
    SlurmScheduler,
    parse_sacct_output,
    parse_sbatch_job_id,
    parse_squeue_output,
)


class StubProcessRunner:
    def __init__(self, responses: list[ProcessResult | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def run(self, command, **kwargs):
        command_tuple = tuple(str(part) for part in command)
        self.calls.append((command_tuple, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def completed(
    command: tuple[str, ...], stdout: str = "", stderr: str = "", returncode: int = 0
) -> ProcessResult:
    return ProcessResult(
        command=command,
        cwd=None,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=0.0,
    )


def test_sbatch_job_id_parser_and_malformed_output() -> None:
    assert parse_sbatch_job_id("Submitted batch job 12345\n") == "12345"
    with pytest.raises(SchedulerError, match="Malformed sbatch output"):
        parse_sbatch_job_id("sbatch accepted the request")


def test_squeue_parser_handles_active_jobs_and_empty_success() -> None:
    records = parse_squeue_output("123|train_demo|RUNNING|None\n124|train_wait|PENDING|Resources\n")
    assert [record.state for record in records] == [
        SchedulerJobState.RUNNING,
        SchedulerJobState.PENDING,
    ]
    assert records[1].reason == "Resources"
    assert parse_squeue_output("") == ()


def test_squeue_parser_keeps_unknown_state_explicit() -> None:
    record = parse_squeue_output("123|demo|NEW_VENDOR_STATE|site reason\n")[0]
    assert record.state is SchedulerJobState.UNKNOWN
    assert record.raw_state == "NEW_VENDOR_STATE"


def test_sacct_parser_maps_terminal_states_and_paths() -> None:
    records = parse_sacct_output(
        "123|demo|COMPLETED|None|0:0|/tmp/out.log|/tmp/err.log\n"
        "124|demo|OUT_OF_MEMORY|oom|137:0|out|err\n"
        "125|demo|CANCELLED|signal|1:0|-|-\n"
        "126|demo|TIMEOUT|time limit|1:0|out|err\n"
        "127|demo|FAILED|node failure|1:9|out|err\n"
    )
    assert [record.state for record in records] == [
        SchedulerJobState.COMPLETED,
        SchedulerJobState.OOM,
        SchedulerJobState.CANCELLED,
        SchedulerJobState.TIMEOUT,
        SchedulerJobState.FAILED,
    ]
    assert records[0].exit_code == 0
    assert records[0].stdout_path == Path("/tmp/out.log")
    assert records[0].stderr_path == Path("/tmp/err.log")


def test_resources_validate_gpu_and_memory_semantics() -> None:
    resources = JobResources(
        nodes=2,
        gpus_per_node=4,
        mpi_ranks=8,
        cpus_per_task=2,
        memory_per_node="64G",
        walltime="01:02:03",
    )
    assert resources.total_gpus == 8
    assert resources.tasks == 8
    assert "#SBATCH --gpus-per-node=4" in render_sbatch_directives(resources)
    assert "#SBATCH --mem=64G" in render_sbatch_directives(resources)

    with pytest.raises(ValueError, match="mutually exclusive"):
        JobResources(memory_per_node="64G", memory_total="128G")
    with pytest.raises(ValueError, match="total_gpus"):
        JobResources(nodes=2, gpus_per_node=4, total_gpus=7)
    with pytest.raises(ValueError, match="nodes"):
        JobResources(nodes=0)


def test_shared_header_replaces_resources_but_preserves_site_prologue() -> None:
    header = "#!/bin/bash\n#SBATCH --partition=gpu\n#SBATCH --nodes=99\nmodule load cuda\nexport SITE_FLAG=1\n"
    rendered = render_slurm_header(
        JobResources(nodes=2, gpus_per_node=4, mpi_ranks=8, walltime="02:00:00"),
        base_header=header,
        job_name="demo",
    )
    assert "#SBATCH --nodes=2" in rendered
    assert "#SBATCH --nodes=99" not in rendered
    assert "#SBATCH --partition=gpu" in rendered
    assert "module load cuda" in rendered
    assert "export SITE_FLAG=1" in rendered


def test_failed_squeue_is_scheduler_error_not_empty_query() -> None:
    runner = StubProcessRunner(
        [
            completed(("squeue",), stderr="scheduler unavailable", returncode=1),
        ]
    )
    scheduler = SlurmScheduler(process_runner=runner, user="demo")

    with pytest.raises(SchedulerError) as raised:
        scheduler.list_active_jobs()

    assert raised.value.kind == "command_failed"
    assert raised.value.returncode == 1


def test_scheduler_query_and_job_name_lookup() -> None:
    runner = StubProcessRunner(
        [
            completed(("squeue",), "123|demo_job|RUNNING|None\n"),
            completed(("squeue",), "123|demo_job|RUNNING|None\n"),
        ]
    )
    scheduler = SlurmScheduler(process_runner=runner, user="demo")
    result = scheduler.list_active_jobs(name_prefix="demo_")
    assert len(result.jobs) == 1
    assert scheduler.find_job_by_name("demo_job").job_id == "123"
    assert all("--format" in call[0] for call in runner.calls)
    assert SQUEUE_FORMAT in runner.calls[0][0]


def test_cancel_uses_typed_scheduler_boundary() -> None:
    runner = StubProcessRunner([completed(("scancel", "123"), "cancelled\n")])
    result = SlurmScheduler(process_runner=runner).cancel("123")
    assert result.job_id == "123"
    assert runner.calls[0][0] == ("scancel", "123")


def test_submit_returns_typed_job_id_and_rejects_malformed_output() -> None:
    runner = StubProcessRunner(
        [
            completed(("sbatch", "script.slurm"), "Submitted batch job 123\n"),
            completed(("sbatch", "script.slurm"), "accepted\n"),
        ]
    )
    scheduler = SlurmScheduler(process_runner=runner)
    result = scheduler.submit(["sbatch", "script.slurm"])
    assert result.job_id == "123"
    with pytest.raises(SchedulerError, match="Malformed sbatch output"):
        scheduler.submit(["sbatch", "script.slurm"])


def test_accounting_reconciles_a_job_that_left_the_queue() -> None:
    runner = StubProcessRunner(
        [
            completed(("squeue",), ""),
            completed(("sacct",), "123|demo|COMPLETED|None|0:0|out|err\n"),
        ]
    )
    result = SlurmScheduler(process_runner=runner).reconcile("123")
    assert result.source == "sacct"
    assert result.job.state is SchedulerJobState.COMPLETED
    assert result.job.stdout_path == Path("out")
    assert SACCT_FORMAT in runner.calls[1][0]


def test_reconciliation_does_not_infer_completion_without_accounting() -> None:
    runner = StubProcessRunner(
        [
            completed(("squeue",), ""),
            completed(("sacct",), ""),
        ]
    )
    result = SlurmScheduler(process_runner=runner).reconcile("123")
    assert result.job.state is SchedulerJobState.NOT_FOUND


def test_process_failure_is_translated_with_context() -> None:
    process_error = ProcessError(
        "squeue unavailable",
        command=("squeue",),
        cwd=None,
        returncode=None,
        stderr="not found",
        kind="not_found",
    )
    runner = StubProcessRunner([process_error])
    with pytest.raises(SchedulerError) as raised:
        SlurmScheduler(process_runner=runner).list_active_jobs()
    assert raised.value.kind == "process_not_found"
    assert raised.value.command == ("squeue",)
