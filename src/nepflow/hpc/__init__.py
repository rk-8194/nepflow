"""High-performance computing and scheduler integrations for NEPFlow."""

from .jobs import (
    AccountingResult,
    CancellationResult,
    QueueQueryResult,
    QueueStatusResult,
    ReconciledJobResult,
    SchedulerJobState,
    SlurmJobRecord,
    SubmissionResult,
)
from .process import MonotonicElapsedTimer, ProcessError, ProcessResult, ProcessRunner
from .resources import (
    DEFAULT_SITE_ENVIRONMENT_PROLOGUE,
    SITE_ENVIRONMENT_PROLOGUE,
    JobResources,
    load_site_environment_prologue,
    normalize_site_environment_prologue,
    render_sbatch_directives,
    render_shell_script,
    render_slurm_header,
    require_resolvable_command_without_prologue,
)
from .scheduler import Scheduler
from .slurm import (
    SlurmScheduler,
    map_slurm_state,
    parse_sacct_output,
    parse_sbatch_job_id,
    parse_squeue_output,
)

__all__ = [
    "AccountingResult",
    "CancellationResult",
    "DEFAULT_SITE_ENVIRONMENT_PROLOGUE",
    "JobResources",
    "MonotonicElapsedTimer",
    "ProcessError",
    "ProcessResult",
    "ProcessRunner",
    "QueueQueryResult",
    "QueueStatusResult",
    "ReconciledJobResult",
    "Scheduler",
    "SchedulerJobState",
    "SITE_ENVIRONMENT_PROLOGUE",
    "SlurmJobRecord",
    "SlurmScheduler",
    "SubmissionResult",
    "map_slurm_state",
    "load_site_environment_prologue",
    "normalize_site_environment_prologue",
    "parse_sacct_output",
    "parse_sbatch_job_id",
    "parse_squeue_output",
    "render_sbatch_directives",
    "render_shell_script",
    "render_slurm_header",
    "require_resolvable_command_without_prologue",
]
