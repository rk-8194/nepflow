import sys
from pathlib import Path

import pytest

from nepflow.errors import ProcessError
from nepflow.hpc.process import MonotonicElapsedTimer, ProcessRunner


def python_command(source: str, *arguments: str) -> list[str]:
    return [sys.executable, "-c", source, *arguments]


def test_string_commands_require_explicit_shell_api() -> None:
    with pytest.raises(TypeError, match="argument list"):
        ProcessRunner().run("echo not-an-argument-list")  # type: ignore[arg-type]


def test_elapsed_timer_uses_injected_monotonic_seconds() -> None:
    ticks = iter((10.0, 10.75))
    timer = MonotonicElapsedTimer(lambda: next(ticks))
    assert timer.elapsed_seconds == 0.75


def test_success_captures_stdout_and_stderr() -> None:
    runner = ProcessRunner()

    result = runner.run(python_command("import sys; print('out'); print('err', file=sys.stderr)"))

    assert result.ok
    assert result.returncode == 0
    assert result.stdout == "out\n"
    assert result.stderr == "err\n"


def test_nonzero_exit_raises_typed_failure_with_context(tmp_path: Path) -> None:
    runner = ProcessRunner()
    command = python_command("import sys; print('out'); print('err', file=sys.stderr); sys.exit(7)")

    with pytest.raises(ProcessError) as raised:
        runner.run(command, cwd=tmp_path)

    error = raised.value
    assert error.kind == "nonzero"
    assert error.command == tuple(command)
    assert error.cwd == str(tmp_path)
    assert error.returncode == 7
    assert error.stdout == "out\n"
    assert error.stderr == "err\n"


def test_missing_executable_is_typed_failure(tmp_path: Path) -> None:
    runner = ProcessRunner()
    missing = str(tmp_path / "not-an-executable")

    with pytest.raises(ProcessError) as raised:
        runner.run([missing])

    error = raised.value
    assert error.kind == "not_found"
    assert error.command == (missing,)
    assert error.returncode is None


def test_timeout_is_typed_failure() -> None:
    runner = ProcessRunner()

    with pytest.raises(ProcessError) as raised:
        runner.run(
            python_command("import time; time.sleep(2)"),
            timeout=0.05,
        )

    assert raised.value.kind == "timeout"
    assert raised.value.returncode is None


def test_working_directory_is_explicit(tmp_path: Path) -> None:
    result = ProcessRunner().run(
        python_command("import os; print(os.getcwd())"),
        cwd=tmp_path,
    )

    assert result.stdout == f"{tmp_path}\n"
    assert result.cwd == str(tmp_path)


def test_environment_overlay_and_full_replacement(monkeypatch: pytest.MonkeyPatch) -> None:
    variable = "NEPFLOW_PROCESS_RUNNER_TEST"
    monkeypatch.setenv(variable, "inherited")
    command = python_command(
        f"import os; print(os.environ.get('{variable}', 'missing')); "
        f"print(os.environ.get('NEPFLOW_PROCESS_OVERLAY', 'missing'))"
    )

    overlay = ProcessRunner().run(
        command,
        env={"NEPFLOW_PROCESS_OVERLAY": "overlay"},
    )
    assert overlay.stdout == "inherited\noverlay\n"

    replacement = ProcessRunner().run(
        command,
        env={"NEPFLOW_PROCESS_OVERLAY": "replacement"},
        inherit_environment=False,
    )
    assert replacement.stdout == "missing\nreplacement\n"


def test_argument_boundaries_are_not_reinterpreted_by_a_shell() -> None:
    argument = "value with spaces; echo should-not-run & [special]"
    result = ProcessRunner().run(python_command("import sys; print(sys.argv[1])", argument))

    assert result.stdout == f"{argument}\n"


def test_captured_stdout_and_stderr_do_not_deadlock() -> None:
    result = ProcessRunner().run(
        python_command("import sys; sys.stdout.write('o' * 100000); sys.stderr.write('e' * 100000)")
    )

    assert result.ok
    assert result.stdout == "o" * 100000
    assert result.stderr == "e" * 100000


def test_nonzero_result_can_be_classified_without_losing_status() -> None:
    result = ProcessRunner().run(
        python_command("import sys; sys.exit(3)"),
        check=False,
    )

    assert not result.ok
    assert result.returncode == 3
