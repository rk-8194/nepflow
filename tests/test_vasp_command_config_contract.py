import tempfile
from configparser import ConfigParser
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")

from modules.init.init import InitStage  # noqa: E402
from modules.run_vasp import run_vasp as run_vasp_module  # noqa: E402
from modules.run_vasp.run_vasp import RunVaspStage  # noqa: E402


def initialized_default_config(project_dir: Path) -> ConfigParser:
    """Render the real config shape produced by initialization."""
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


def make_stage(project_dir: Path) -> RunVaspStage:
    return RunVaspStage(
        project_name="demo",
        config_file=project_dir / "config" / "demo.ini",
        state_file=project_dir / "state.db",
        project_dir=project_dir,
        debug=False,
    )


def write_header(project_dir: Path) -> None:
    header = project_dir / "config" / "slurm" / "header.slurm"
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_text("#!/bin/bash\n", encoding="utf-8")


def test_initialized_hpc_command_is_handed_to_shared_vasp_script() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp) / "project_demo"
        write_header(project_dir)
        config = initialized_default_config(project_dir)
        stage = make_stage(project_dir)

        with (
            patch.object(stage, "_find_config_file", return_value=project_dir / "config" / "project.config"),
            patch.object(stage, "_load_config", return_value=config),
            patch.object(run_vasp_module, "prepare_jobs"),
            patch.object(run_vasp_module, "write_shared_vasp_script") as write_script_mock,
            patch.object(run_vasp_module, "run_launcher"),
        ):
            stage.run()

        assert write_script_mock.call_args.args[2] == config["hpc"]["vasp_command"]


def test_slurm_command_cannot_override_hpc_command() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp) / "project_demo"
        write_header(project_dir)
        config = initialized_default_config(project_dir)
        expected_command = config["hpc"]["vasp_command"]
        config["slurm"]["vasp_command"] = "wrong-command --from-slurm"
        stage = make_stage(project_dir)

        with (
            patch.object(stage, "_find_config_file", return_value=project_dir / "config" / "project.config"),
            patch.object(stage, "_load_config", return_value=config),
            patch.object(run_vasp_module, "prepare_jobs"),
            patch.object(run_vasp_module, "write_shared_vasp_script") as write_script_mock,
            patch.object(run_vasp_module, "run_launcher"),
        ):
            stage.run()

        assert write_script_mock.call_args.args[2] == expected_command
        assert write_script_mock.call_args.args[2] != config["slurm"]["vasp_command"]


@pytest.mark.parametrize("command", [None, "   "], ids=["missing", "blank"])
def test_missing_or_blank_hpc_command_is_an_explicit_error(command: str | None) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp) / "project_demo"
        config = initialized_default_config(project_dir)
        if command is None:
            config["hpc"].pop("vasp_command")
        else:
            config["hpc"]["vasp_command"] = command
        stage = make_stage(project_dir)

        with (
            patch.object(stage, "_find_config_file", return_value=project_dir / "config" / "project.config"),
            patch.object(stage, "_load_config", return_value=config),
        ):
            with pytest.raises(ValueError, match="hpc\\.vasp_command"):
                stage.run()
