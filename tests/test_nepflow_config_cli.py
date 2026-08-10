import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def install_nepflow_import_stubs() -> None:
    for name in [
        "numpy",
        "ase",
        "hiphive",
        "mp_api",
        "pymatgen",
        "icet",
        "NepTrainKit",
    ]:
        sys.modules.setdefault(name, types.ModuleType(name))

    workflow_module = types.ModuleType("workflow")
    workflow_module.WorkflowController = type("WorkflowController", (), {})
    sys.modules.setdefault("workflow", workflow_module)

    logging_module = types.ModuleType("logging_config")
    logging_module.setup_logging = lambda *args, **kwargs: None
    sys.modules.setdefault("logging_config", logging_module)

    modules_module = types.ModuleType("modules")
    modules_module.SelfResubmitExit = type("SelfResubmitExit", (Exception,), {})
    sys.modules.setdefault("modules", modules_module)


def load_nepflow_module():
    install_nepflow_import_stubs()
    spec = importlib.util.spec_from_file_location("nepflow", ROOT / "nepflow.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["nepflow"] = module
    spec.loader.exec_module(module)
    return module


class ConfigCliTests(unittest.TestCase):
    def test_config_flag_opens_project_config_in_vim(self) -> None:
        nepflow = load_nepflow_module()

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            project_dir = output_dir / "project_w"
            config_file = project_dir / "config" / "project.config"
            config_file.parent.mkdir(parents=True, exist_ok=True)
            config_file.write_text("[project]\nname=w\n", encoding="utf-8")

            with patch.object(
                sys,
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
        nepflow = load_nepflow_module()

        with tempfile.TemporaryDirectory() as tmp:
            submit_dir = Path(tmp)
            script_path = submit_dir / "submit.slurm"
            script_path.write_text("#!/bin/bash\n", encoding="utf-8")

            scontrol_output = (
                f"JobId=123 Command={script_path} WorkDir={submit_dir}"
            )

            with patch.dict(os.environ, {"SLURM_JOB_ID": "123"}, clear=False):
                with patch.object(
                    nepflow.subprocess,
                    "run",
                    return_value=types.SimpleNamespace(stdout=scontrol_output),
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
        nepflow = load_nepflow_module()

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
