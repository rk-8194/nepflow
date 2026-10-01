"""Stable exception hierarchy for NEPFlow failure domains."""


class NepflowError(Exception):
    """Base class for expected NEPFlow failures."""


class ConfigurationError(NepflowError):
    """Raised when project or runtime configuration is invalid."""


class ValidationError(NepflowError):
    """Raised when required validation fails."""


class StateError(NepflowError):
    """Raised when workflow state is missing, malformed, or inconsistent."""


class SchedulerError(NepflowError):
    """Raised when scheduler interaction fails."""


class ProcessError(NepflowError):
    """Raised when an external process cannot be executed successfully."""

    def __init__(
        self,
        message: str,
        *,
        command: tuple[str, ...],
        cwd: str | None,
        returncode: int | None,
        stdout: str | None = None,
        stderr: str | None = None,
        kind: str = "execution",
    ) -> None:
        self.command = command
        self.cwd = cwd
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.kind = kind
        super().__init__(message)


class BackendError(NepflowError):
    """Raised when an external scientific backend fails."""


class VaspError(BackendError):
    """Raised for VASP backend failures."""


class MlipError(BackendError):
    """Raised for machine-learning interatomic potential failures."""


class ArtifactError(NepflowError):
    """Raised when a required workflow artifact is invalid or unavailable."""
