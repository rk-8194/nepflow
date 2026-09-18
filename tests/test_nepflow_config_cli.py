import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

for package_name in ("ase", "hiphive", "mp_api", "pymatgen", "icet", "NepTrainKit"):
    pytest.importorskip(package_name)

import nepflow  # noqa: E402


class ConfigCliTests(unittest.TestCase):
    def test_config_flag_opens_project_config_in_vim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            project_dir = output_dir / "project_w"
            config_file = project_dir / "config" / "project.config"
            config_file.parent.mkdir(parents=True, exist_ok=True)
            config_file.write_text("[project]\nname=w\n", encoding="utf-8")

            with patch.object(
                nepflow.sys,
                "argv",
                [
                    "nepflow.py",
                    "--project",
                    "w",
                    "--config",
                    "--output-dir",
                    str(output_dir),
                ],
            ):
                with patch.object(nepflow.subprocess, "run") as run_mock:
                    nepflow.main()

            run_mock.assert_called_once_with(
                ["vim", str(config_file)],
                check=False,
                cwd=str(project_dir),
            )

    def test_resolve_resubmit_command_prefers_original_slurm_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            submit_dir = Path(tmp)
            script_path = submit_dir / "submit.slurm"
            script_path.write_text("#!/bin/bash\n", encoding="utf-8")
            scontrol_output = f"JobId=123 Command={script_path} WorkDir={submit_dir}"

            with patch.dict(os.environ, {"SLURM_JOB_ID": "123"}, clear=False):
                with patch.object(
                    nepflow.subprocess,
                    "run",
                    return_value=SimpleNamespace(stdout=scontrol_output),
                ) as run_mock:
                    command, cwd, source = nepflow._resolve_resubmit_command()

            self.assertEqual(command, ["sbatch", str(script_path)])
            self.assertEqual(cwd, submit_dir)
            self.assertEqual(source, "scontrol job 123")
            run_mock.assert_called_once_with(
                ["scontrol", "show", "job", "123"],
                capture_output=True,
                text=True,
                check=True,
            )

    def test_resolve_resubmit_command_falls_back_to_submit_dir_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            submit_dir = Path(tmp)
            script_path = submit_dir / "submit.slurm"
            script_path.write_text("#!/bin/bash\n", encoding="utf-8")

            with patch.dict(
                os.environ,
                {"SLURM_JOB_ID": "123", "SLURM_SUBMIT_DIR": str(submit_dir)},
                clear=False,
            ):
                with patch.object(
                    nepflow.subprocess,
                    "run",
                    side_effect=FileNotFoundError("scontrol not found"),
                ):
                    command, cwd, source = nepflow._resolve_resubmit_command()

            self.assertEqual(command, ["sbatch", str(script_path)])
            self.assertEqual(cwd, submit_dir)
            self.assertEqual(source, f"submit.slurm in {submit_dir}")


if __name__ == "__main__":
    unittest.main()
