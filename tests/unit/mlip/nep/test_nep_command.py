from typing import cast

import pytest

from nepflow.mlip.backend import TrainingInput
from nepflow.mlip.nep.backend import NepBackend


def test_backend_command_is_argument_oriented_and_scheduler_independent() -> None:
    backend = NepBackend("/opt/gpumd/bin/nep --quiet")
    command = backend.training_command(cast(TrainingInput, None))

    assert command == ("/opt/gpumd/bin/nep", "--quiet")
    assert "sbatch" not in command
    assert ";" not in command


def test_missing_nep_command_is_an_explicit_error() -> None:
    with pytest.raises(ValueError, match="command"):
        NepBackend("")
