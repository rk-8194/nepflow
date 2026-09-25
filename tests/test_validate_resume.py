import json
import tempfile
from configparser import ConfigParser
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")
from ase import Atoms  # noqa: E402

from modules.validate import launcher as launcher_module  # noqa: E402
from modules.validate import prepare as prepare_module  # noqa: E402
from modules.validate import validate as validate_stage_module  # noqa: E402
from modules.validate.launcher import (  # noqa: E402
    read_validation_status,
    write_validation_status,
)
from modules.validate.validate import ValidateStage  # noqa: E402


def make_validate_stage(project_dir: Path) -> ValidateStage:
    return ValidateStage(
        project_name="demo",
        config_file=project_dir / "config" / "demo.yaml",
        state_file=project_dir / "state.db",
        project_dir=project_dir,
        debug=False,
    )


def make_config() -> ConfigParser:
    config = ConfigParser()
    config["slurm"] = {
        "max_concurrent": "1",
        "poll_interval": "0",
        "max_retry_level": "1",
    }
    return config


def prepare_state_fixture(root: Path) -> tuple[dict, Path]:
    dataset_path = root / "nep" / "datasets" / "dataset_0001"
    potential_path = root / "gpumd" / "dataset_0001" / "potential_0001"
    config_gpumd_dir = root / "config" / "gpumd"
    dataset_path.mkdir(parents=True, exist_ok=True)
    potential_path.mkdir(parents=True, exist_ok=True)
    config_gpumd_dir.mkdir(parents=True, exist_ok=True)
    (potential_path / "nep.txt").write_text(
        "version 4\ntype 1 Si\ncutoff 6 5 112 60\n",
        encoding="utf-8",
    )
    (config_gpumd_dir / "run.in_validate").write_text("replicate 2 2 2\n", encoding="utf-8")

    atoms = Atoms(
        "Si",
        positions=[[0.0, 0.0, 0.0]],
        cell=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        pbc=True,
    )
    with (
        patch.object(prepare_module, "parse_test_xyz", return_value=[{"atoms": atoms}]),
        patch.object(prepare_module, "calculate_required_replicates", return_value=(2, 2, 2)),
    ):
        state = prepare_module.prepare_validation_structures(
            dataset_path=dataset_path,
            gpumd_potential_dir=potential_path,
            project_dir=root,
            config_gpumd_dir=config_gpumd_dir,
        )

    return state, state["struct_folders"][0]["path"]


@pytest.mark.xfail(
    strict=True,
    reason="Phase 2 blocker P0-2: validation preparation state must be JSON serializable",
)
def test_preparation_state_serializes_without_raw_paths() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        state, _ = prepare_state_fixture(Path(tmp))

    json.dumps(state)


@pytest.mark.xfail(
    strict=True,
    reason="Phase 2 blocker P0-2: persisted validation state must round-trip structure paths",
)
def test_persisted_preparation_paths_round_trip_after_reload() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        state, struct_path = prepare_state_fixture(root)
        write_validation_status(root, preparation_state=state)
        reloaded = read_validation_status(root)

        persisted_path = reloaded["preparation_state"]["struct_folders"][0]["path"]
        assert Path(persisted_path) == struct_path


def resume_state(root: Path) -> tuple[ValidateStage, dict, Path]:
    potential_path = root / "gpumd" / "dataset_0001" / "potential_0001"
    potential_path.mkdir(parents=True, exist_ok=True)
    preparation_state = {
        "validation_root": str(potential_path / "validation"),
        "struct_count": 0,
        "struct_folders": [],
    }
    status = {
        "status": "running",
        "potential_path": str(potential_path),
        "dataset_path": str(root / "nep" / "datasets" / "dataset_0001"),
        "dataset_name": "dataset_0001",
        "preparation_state": preparation_state,
        "validation_complete": False,
        "analysis_complete": False,
    }
    write_validation_status(root, **status)
    return make_validate_stage(root), status, potential_path


@pytest.mark.xfail(
    strict=True,
    reason="Phase 2 blocker P0-5: resumed validation must reload launcher-updated completion state",
)
def test_resumed_stage_observes_persisted_launcher_completion_before_analysis() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        stage, _, _ = resume_state(root)
        config = make_config()

        def launcher_completes_on_disk(*args: object, **kwargs: object) -> None:
            write_validation_status(root, validation_complete=True)

        with (
            patch.object(stage, "_find_config_file", return_value=root / "config" / "demo.yaml"),
            patch.object(stage, "_load_config", return_value=config),
            patch.object(
                validate_stage_module,
                "run_validation_launcher",
                side_effect=launcher_completes_on_disk,
            ),
            patch.object(stage, "_run_analysis") as analysis_mock,
        ):
            stage.run()

        assert read_validation_status(root)["validation_complete"] is True
        analysis_mock.assert_called_once()


def test_missing_required_resume_status_fields_fail_explicitly() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_validation_status(root, status="running")
        stage = make_validate_stage(root)

        with (
            patch.object(stage, "_find_config_file", return_value=root / "config" / "demo.yaml"),
            patch.object(stage, "_load_config", return_value=make_config()),
        ):
            with pytest.raises((KeyError, ValueError)):
                stage.run()


@pytest.mark.xfail(
    strict=True,
    reason="Phase 2 blocker P0-5: malformed validation state must fail fast instead of becoming an empty state",
)
def test_malformed_validation_status_is_an_explicit_failure() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        status_file = root / "gpumd" / ".validation_status"
        status_file.parent.mkdir(parents=True, exist_ok=True)
        status_file.write_text("{not-json", encoding="utf-8")

        with pytest.raises((ValueError, json.JSONDecodeError)):
            read_validation_status(root)


@pytest.mark.xfail(
    strict=True,
    reason="Phase 3 deferred scheduler abstraction: scheduler-query failure must be distinguishable from no running jobs",
)
def test_scheduler_query_failure_is_not_an_empty_running_job_set() -> None:
    with patch.object(launcher_module.subprocess, "run", side_effect=OSError("squeue unavailable")):
        with pytest.raises(RuntimeError):
            launcher_module._get_running_job_ids()


@pytest.mark.xfail(
    strict=True,
    reason="Phase 3 deferred scheduler abstraction: query failure must not mark submitted jobs completed",
)
def test_scheduler_query_failure_does_not_complete_submitted_validation_job() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        struct_dir = root / "validation" / "struct_0000"
        struct_dir.mkdir(parents=True, exist_ok=True)
        (struct_dir / "out.xyz").write_text("completed output\n", encoding="utf-8")
        preparation_state = {
            "validation_root": str(root / "validation"),
            "struct_folders": [{"name": "struct_0000"}],
        }
        write_validation_status(
            root,
            struct_status={
                "struct_0000": {
                    "status": "submitted",
                    "job_id": "job-123",
                    "attempts": 1,
                }
            },
            completed_count=0,
            validation_complete=False,
        )

        query_failure = launcher_module.subprocess.CompletedProcess(
            args=["squeue"], returncode=1, stdout="", stderr="scheduler unavailable"
        )
        config = make_config()
        with patch.object(launcher_module.subprocess, "run", return_value=query_failure):
            try:
                launcher_module.run_validation_launcher(config, preparation_state, root)
            except RuntimeError:
                pass

        persisted = read_validation_status(root)
        assert persisted.get("validation_complete") is not True
        assert persisted["struct_status"]["struct_0000"]["status"] != "completed"


@pytest.mark.xfail(
    strict=True,
    reason="Phase 2 blocker P0-5: failed analysis must not persist analysis_complete",
)
def test_analysis_failure_cannot_mark_validation_analysis_complete() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        stage = make_validate_stage(root)
        status = {
            "status": "running",
            "potential_path": str(root / "gpumd" / "dataset_0001" / "potential_0001"),
            "dataset_name": "dataset_0001",
            "preparation_state": {"validation_root": str(root / "validation")},
            "analysis_complete": False,
        }
        write_validation_status(root, **status)

        with (
            patch.object(
                validate_stage_module,
                "generate_comparison_csv",
                side_effect=RuntimeError("analysis input failed"),
            ),
            patch.object(
                validate_stage_module,
                "plot_comparison_results",
                side_effect=RuntimeError("plot input failed"),
            ),
        ):
            try:
                stage._run_analysis(make_config(), status)
            except Exception:
                pass

        assert read_validation_status(root).get("analysis_complete") is not True
