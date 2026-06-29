import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def load_init_module():
    package_names = [
        "testpkg",
        "testpkg.modules",
        "testpkg.modules.init",
    ]
    for name in package_names:
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module

    ase_module = types.ModuleType("ase")
    ase_module.__path__ = []
    sys.modules["ase"] = ase_module

    ase_data_module = types.ModuleType("ase.data")
    ase_data_module.chemical_symbols = [
        "",
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Cr",
        "Y",
        "Zr",
        "W",
    ]
    sys.modules["ase.data"] = ase_data_module

    base_module = types.ModuleType("testpkg.modules.base")

    class Stage:
        def __init__(
            self,
            project_name,
            config_file,
            state_file,
            project_dir,
            debug=False,
            slurm_deadline=None,
        ):
            self.project_name = project_name
            self.config_file = Path(config_file)
            self.state_file = Path(state_file)
            self.project_dir = Path(project_dir)
            self.debug = debug
            self.slurm_deadline = slurm_deadline

    base_module.Stage = Stage
    sys.modules["testpkg.modules.base"] = base_module

    spec = importlib.util.spec_from_file_location(
        "testpkg.modules.init.init",
        SRC / "modules" / "init" / "init.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["testpkg.modules.init.init"] = module
    spec.loader.exec_module(module)
    return module


InitModule = load_init_module()
ConfigPrompt = InitModule.ConfigPrompt
InitStage = InitModule.InitStage


class InitConfigPromptTests(unittest.TestCase):
    def _make_stage(self, project_dir: Path) -> InitStage:
        return InitStage(
            project_name="demo",
            config_file=project_dir / "config" / "demo.yaml",
            state_file=project_dir / "state.db",
            project_dir=project_dir,
        )

    def test_setup_config_prompts_for_materials_project_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            (project_dir / "config").mkdir(parents=True, exist_ok=True)
            stage = self._make_stage(project_dir)

            with patch.dict(os.environ, {}, clear=True):
                with patch(
                    "testpkg.modules.init.init.input",
                    side_effect=[
                        "mp-test-key",
                        "w,cr,y,zr",
                        "",
                        "BCC, fcc",
                        "",
                        "user@host:/opt/nepflow",
                    ],
                ) as input_mock:
                    stage._setup_config()

            config_text = (project_dir / "config" / "project.config").read_text(encoding="utf-8")
            self.assertIn("api_key=mp-test-key", config_text)
            self.assertIn("elements=W,Cr,Y,Zr", config_text)
            self.assertIn("gasElements=", config_text)
            self.assertIn("crystal_structures=bcc,fcc", config_text)
            self.assertIn("target_n_atoms=128", config_text)
            self.assertIn("scp_address=user@host:/opt/nepflow", config_text)
            self.assertEqual(input_mock.call_count, 6)

    def test_unknown_element_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            (project_dir / "config").mkdir(parents=True, exist_ok=True)
            stage = self._make_stage(project_dir)

            with patch.dict(os.environ, {}, clear=True):
                with patch(
                    "testpkg.modules.init.init.input",
                    side_effect=["mp-test-key", "W,Xx"],
                ):
                    with self.assertRaisesRegex(ValueError, "Unknown element: Xx"):
                        stage._setup_config()

    def test_blank_elements_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            (project_dir / "config").mkdir(parents=True, exist_ok=True)
            stage = self._make_stage(project_dir)

            with patch.dict(os.environ, {}, clear=True):
                with patch(
                    "testpkg.modules.init.init.input",
                    side_effect=["mp-test-key", ""],
                ):
                    with self.assertRaisesRegex(ValueError, "At least one element is required"):
                        stage._setup_config()

    def test_prompt_registry_is_generic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            stage = self._make_stage(project_dir)

            prompts = (
                ConfigPrompt(
                    key="first_value",
                    label="First value",
                    message="First value",
                    env_var="FIRST_VALUE",
                ),
                ConfigPrompt(
                    key="secret_value",
                    label="Secret value",
                    message="Secret value",
                ),
            )

            with patch.dict(os.environ, {"FIRST_VALUE": "from-env"}, clear=True):
                with patch("testpkg.modules.init.init.input", return_value="from-prompt"):
                    values = stage._collect_prompt_values(prompts)

            self.assertEqual(
                values,
                {
                    "first_value": "from-env",
                    "secret_value": "from-prompt",
                },
            )


if __name__ == "__main__":
    unittest.main()
