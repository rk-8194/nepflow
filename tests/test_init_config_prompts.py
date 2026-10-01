import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")

from modules.init.init import ConfigPrompt, InitStage
from nepflow.state import StateStore


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
                    "modules.init.init.input",
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
            self.assertIn("gas_elements=", config_text)
            self.assertIn("crystal_structures=bcc,fcc", config_text)
            self.assertIn("target_n_atoms=128", config_text)
            self.assertIn("scp_address=user@host:/opt/nepflow", config_text)
            self.assertIn("composition_aware_fps=false", config_text)
            self.assertIn("composition_aware_fps_frontier_fraction=0.10", config_text)
            self.assertIn("composition_aware_fps_ternary_weight=1.0", config_text)
            self.assertIn("composition_aware_fps_adaptive_retries=4", config_text)
            self.assertIn(
                "composition_aware_fps_descriptor_floor_fraction=0.95",
                config_text,
            )
            self.assertEqual(input_mock.call_count, 6)

    def test_materials_project_environment_key_is_not_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            (project_dir / "config").mkdir(parents=True, exist_ok=True)
            stage = self._make_stage(project_dir)

            with patch.dict(os.environ, {"MP_API_KEY": "environment-secret"}, clear=True):
                with patch(
                    "modules.init.init.input",
                    side_effect=[
                        "W",
                        "",
                        "BCC",
                        "",
                        "user@host:/opt/nepflow",
                    ],
                ):
                    stage._setup_config()

            config_text = (project_dir / "config" / "project.config").read_text(
                encoding="utf-8"
            )
            self.assertIn("api_key=\n", config_text)
            self.assertNotIn("environment-secret", config_text)

    def test_unknown_element_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            (project_dir / "config").mkdir(parents=True, exist_ok=True)
            stage = self._make_stage(project_dir)

            with patch.dict(os.environ, {}, clear=True):
                with patch("modules.init.init.input", side_effect=["mp-test-key", "W,Xx"]):
                    with self.assertRaisesRegex(ValueError, "Unknown element: Xx"):
                        stage._setup_config()

    def test_blank_elements_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            (project_dir / "config").mkdir(parents=True, exist_ok=True)
            stage = self._make_stage(project_dir)

            with patch.dict(os.environ, {}, clear=True):
                with patch("modules.init.init.input", side_effect=["mp-test-key", ""]):
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
                with patch("modules.init.init.input", return_value="from-prompt"):
                    values = stage._collect_prompt_values(prompts)

            self.assertEqual(
                values,
                {
                    "first_value": "from-env",
                    "secret_value": "from-prompt",
                },
            )

    def test_run_initializes_authoritative_state_db(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            stage = self._make_stage(project_dir)

            with patch.dict(os.environ, {}, clear=True):
                with patch(
                    "modules.init.init.input",
                    side_effect=[
                        "",
                        "W",
                        "",
                        "BCC",
                        "",
                        "user@host:/opt/nepflow",
                    ],
                ):
                    stage.run()

            with StateStore(project_dir / "state.db") as store:
                project = store.get_project("demo")

            assert project is not None
            assert project["root_path"] == str(project_dir)


if __name__ == "__main__":
    unittest.main()
