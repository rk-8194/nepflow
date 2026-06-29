import importlib.util
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


if __name__ == "__main__":
    unittest.main()
