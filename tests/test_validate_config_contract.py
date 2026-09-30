"""Regression tests for validation runtime project-context handling.

The project directory is runtime context supplied by the workflow/stage. It is
not scientific configuration and must not be reconstructed from a
``paths.project_dir`` setting.
"""

import tempfile
from configparser import ConfigParser
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")

from modules.init.init import InitStage  # noqa: E402
from modules.validate import launcher as launcher_module  # noqa: E402


def initialized_default_config(project_dir: Path) -> ConfigParser:
    """Render the real current initialization config shape."""
    stage = InitStage(
        project_name="demo",
        config_file=project_dir / "config" / "demo.yaml",
        state_file=project_dir / "state.db",
        project_dir=project_dir,
        debug=False,
    )
    rendered = stage._render_default_config(
        {
            "materialsproject_api_key": "",
            "elements": "Si",
            "gasElements": "",
            "crystal_structures": "diamond",
            "target_n_atoms": "64",
            "scp_address": "user@host:/srv/nepflow",
        }
    )
    config = ConfigParser()
    config.read_string(rendered)
    return config


def write_header(project_dir: Path, marker: str) -> None:
    header = project_dir / "config" / "slurm" / "header.slurm"
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_text(f"#!/bin/bash\n# {marker}\n", encoding="utf-8")


def test_initialized_config_does_not_define_project_dir_as_scientific_config() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = initialized_default_config(Path(tmp))

    assert config.has_section("paths")
    assert not config.has_option("paths", "project_dir")


def test_slurm_script_uses_explicit_runtime_project_dir_without_config_workaround() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp) / "project_demo"
        struct_dir = project_dir / "gpumd" / "validation" / "struct_0000"
        struct_dir.mkdir(parents=True)
        write_header(project_dir, "RUNTIME HEADER")
        config = initialized_default_config(project_dir)

        assert not config.has_option("paths", "project_dir")
        script = launcher_module._generate_slurm_script(
            struct_dir,
            "gpumd_val_struct_0000",
            config,
            project_dir=project_dir,
        )

        assert "# RUNTIME HEADER" in script
        assert f"cd {struct_dir}" in script


def test_wrong_config_project_dir_cannot_override_runtime_context() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        project_dir = root / "runtime_project"
        wrong_dir = root / "stale_config_project"
        struct_dir = project_dir / "gpumd" / "validation" / "struct_0000"
        struct_dir.mkdir(parents=True)
        write_header(project_dir, "RUNTIME HEADER")
        write_header(wrong_dir, "WRONG CONFIG HEADER")

        config = initialized_default_config(project_dir)
        config["paths"]["project_dir"] = str(wrong_dir)

        script = launcher_module._generate_slurm_script(
            struct_dir,
            "gpumd_val_struct_0000",
            config,
            project_dir=project_dir,
        )

        assert "# RUNTIME HEADER" in script
        assert "# WRONG CONFIG HEADER" not in script


def test_validation_launcher_threads_runtime_project_dir_to_submission_boundary() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp) / "project_demo"
        struct_dir = project_dir / "gpumd" / "validation" / "struct_0000"
        struct_dir.mkdir(parents=True)
        config = initialized_default_config(project_dir)
        config["slurm"]["max_concurrent"] = "1"
        config["slurm"]["poll_interval"] = "0"
        config["slurm"]["max_retry_level"] = "1"
        preparation_state = {
            "validation_root": str(struct_dir.parent),
            "struct_folders": [{"name": "struct_0000"}],
        }

        with (
            patch.object(launcher_module, "_get_running_job_ids", return_value=set()),
            patch.object(
                launcher_module,
                "submit_struct_validation_job",
                return_value="job-123",
            ) as submit_mock,
        ):
            launcher_module.run_validation_launcher(
                config,
                preparation_state,
                project_dir,
                debug=False,
            )

        submit_mock.assert_called_once_with(
            struct_dir,
            config,
            project_dir=project_dir,
            debug=False,
        )


def test_missing_runtime_project_dir_is_explicit_error_not_guessed_fallback() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        struct_dir = root / "gpumd" / "validation" / "struct_0000"
        struct_dir.mkdir(parents=True)
        config = initialized_default_config(root)

        with pytest.raises(ValueError, match="project_dir|runtime"):
            launcher_module._generate_slurm_script(
                struct_dir,
                "gpumd_val_struct_0000",
                config,
                project_dir=None,
            )


def test_missing_runtime_slurm_header_is_explicit_error() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp) / "project_demo"
        struct_dir = project_dir / "gpumd" / "validation" / "struct_0000"
        struct_dir.mkdir(parents=True)
        config = initialized_default_config(project_dir)

        with pytest.raises(FileNotFoundError, match="header\\.slurm"):
            launcher_module._generate_slurm_script(
                struct_dir,
                "gpumd_val_struct_0000",
                config,
                project_dir=project_dir,
            )


def test_missing_gpumd_command_is_explicit_error() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp) / "project_demo"
        struct_dir = project_dir / "gpumd" / "validation" / "struct_0000"
        struct_dir.mkdir(parents=True)
        write_header(project_dir, "RUNTIME HEADER")
        config = initialized_default_config(project_dir)
        config.remove_option("hpc", "gpumd_command")

        with pytest.raises(ValueError, match="hpc\\.gpumd_command"):
            launcher_module._generate_slurm_script(
                struct_dir,
                "gpumd_val_struct_0000",
                config,
                project_dir=project_dir,
            )
