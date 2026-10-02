import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")

from modules.init.init import ConfigPrompt, InitStage
from nepflow.config import load_config
from nepflow.errors import ConfigurationError, StateError
from nepflow.state import CURRENT_SCHEMA_VERSION, StateStore
from nepflow.workflow.stages import StageRunState, WorkflowStage


class InitConfigPromptTests(unittest.TestCase):
    def _make_stage(self, project_dir: Path) -> InitStage:
        return InitStage(
            project_name="demo",
            config_file=project_dir / "config" / "demo.yaml",
            state_file=project_dir / "state.db",
            project_dir=project_dir,
        )

    def _valid_config_text(self, stage: InitStage) -> str:
        return stage._render_default_config(
            {
                "materialsproject_api_key": "",
                "elements": "W",
                "gas_elements": "",
                "crystal_structures": "bcc",
                "target_n_atoms": "128",
                "scp_address": "",
            }
        )

    def _install_valid_config(self, stage: InitStage) -> Path:
        config_path = stage.project_dir / "config" / "project.config"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(self._valid_config_text(stage), encoding="utf-8")
        return config_path

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

            config = load_config(project_dir / "config" / "project.config")
            with StateStore(project_dir / "state.db") as store:
                project = store.get_project("demo")
                stage_run = store.get_stage_run("demo:init")

            assert project is not None
            assert project["root_path"] == str(project_dir)
            assert config.project.name == "demo"
            assert project["config_fingerprint"] == project["metadata"]["config_fingerprint"]
            assert project["metadata"]["config_schema_version"] == config.schema_version
            assert project["metadata"]["state_schema_version"] == CURRENT_SCHEMA_VERSION
            assert stage_run is not None
            assert stage_run["stage"] == WorkflowStage.INIT.value
            assert stage_run["status"] == StageRunState.RUNNING.value

    def test_repeated_init_is_non_destructive_and_does_not_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            stage = self._make_stage(project_dir)
            prompts = ["", "W", "", "BCC", "", "user@host:/opt/nepflow"]
            with patch.dict(os.environ, {}, clear=True), patch(
                "modules.init.init.input", side_effect=prompts
            ):
                stage.run()

            config_path = project_dir / "config" / "project.config"
            state_path = project_dir / "state.db"
            first_config = config_path.read_bytes()
            with StateStore(state_path) as store:
                first_project = store.get_project("demo")
                first_stage = store.get_stage_run("demo:init")

            with patch("modules.init.init.input") as input_mock:
                stage.run()

            assert config_path.read_bytes() == first_config
            input_mock.assert_not_called()
            with StateStore(state_path) as store:
                assert store.get_project("demo") == first_project
                assert store.get_stage_run("demo:init") == first_stage

    def test_invalid_rendered_config_does_not_create_authoritative_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            stage = self._make_stage(project_dir)
            invalid_config = self._valid_config_text(stage).replace(
                "schema_version=1", "schema_version=999", 1
            )

            with patch.object(stage, "_render_default_config", return_value=invalid_config):
                with pytest.raises(ConfigurationError, match="schema_version"):
                    with patch.dict(os.environ, {}, clear=True), patch(
                        "modules.init.init.input", side_effect=["", "W", "", "BCC", "", ""]
                    ):
                        stage.run()

            assert not (project_dir / "config" / "project.config").exists()
            assert not (project_dir / "state.db").exists()

    def test_state_transaction_rolls_back_project_when_stage_record_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            stage = self._make_stage(project_dir)

            with patch.object(
                StateStore,
                "upsert_stage_run",
                side_effect=RuntimeError("stage write failed"),
            ):
                with pytest.raises(RuntimeError, match="stage write failed"):
                    with patch.dict(os.environ, {}, clear=True), patch(
                        "modules.init.init.input", side_effect=["", "W", "", "BCC", "", ""]
                    ):
                        stage.run()

            assert not (project_dir / "config" / "project.config").exists()
            assert not (project_dir / "state.db").exists()

            with patch.dict(os.environ, {}, clear=True), patch(
                "modules.init.init.input", side_effect=["", "W", "", "BCC", "", ""]
            ):
                stage.run()

            with StateStore(project_dir / "state.db") as store:
                assert store.get_project("demo") is not None
                assert store.get_stage_run("demo:init") is not None

    def test_corrupt_existing_state_fails_without_replacing_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            state_path = project_dir / "state.db"
            state_path.write_bytes(b"not a sqlite database")
            stage = self._make_stage(project_dir)

            with pytest.raises(StateError, match="state database"):
                stage.run()

            assert state_path.read_bytes() == b"not a sqlite database"

    def test_existing_state_without_config_requires_explicit_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            with StateStore(project_dir / "state.db"):
                pass

            with pytest.raises(ConfigurationError, match="canonical config"):
                self._make_stage(project_dir).run()

    def test_legacy_marker_requires_explicit_migration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            (project_dir / ".project").write_text("generate", encoding="utf-8")

            with pytest.raises(StateError, match="explicit migration"):
                self._make_stage(project_dir).run()

    def test_existing_config_without_state_requires_explicit_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            stage = self._make_stage(project_dir)
            self._install_valid_config(stage)

            with pytest.raises(StateError, match="state.db is missing"):
                stage.run()

            assert not (project_dir / "state.db").exists()

    def test_existing_state_without_project_requires_explicit_migration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            stage = self._make_stage(project_dir)
            self._install_valid_config(stage)
            with StateStore(project_dir / "state.db"):
                pass

            with pytest.raises(StateError, match="no project record"):
                stage.run()

            with StateStore(project_dir / "state.db") as store:
                assert store.get_project("demo") is None

    def test_existing_project_without_stage_requires_explicit_migration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            stage = self._make_stage(project_dir)
            with patch.dict(os.environ, {}, clear=True), patch(
                "modules.init.init.input", side_effect=["", "W", "", "BCC", "", ""]
            ):
                stage.run()

            with StateStore(project_dir / "state.db") as store:
                store.connection.execute(
                    "DELETE FROM stage_runs WHERE project_id = ?", ("demo",)
                )

            with pytest.raises(StateError, match="no stage history"):
                stage.run()

            with StateStore(project_dir / "state.db") as store:
                assert store.get_stage_run("demo:init") is None


if __name__ == "__main__":
    unittest.main()
