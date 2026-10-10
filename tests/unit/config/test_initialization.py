import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")

from nepflow.cli_wizard import CONFIG_PROMPTS, ConfigPrompt, ConfigWizard
from nepflow.config import load_config, render_default_config
from nepflow.errors import ConfigurationError, StateError
from nepflow.stages.generation.generators.materials_project import (
    build_materials_project_fetcher,
)
from nepflow.state import CURRENT_SCHEMA_VERSION, StateStore
from nepflow.workflow import initialization as initialization_module
from nepflow.workflow.initialization import ProjectCreationService
from nepflow.workflow.stages import StageRunState, WorkflowStage


class InitConfigPromptTests(unittest.TestCase):
    def _make_service(self, project_dir: Path) -> ProjectCreationService:
        return ProjectCreationService(
            project_name="demo",
            config_file=project_dir / "config" / "demo.yaml",
            state_file=project_dir / "state.db",
            project_dir=project_dir,
        )

    def _valid_config_text(self, service: ProjectCreationService) -> str:
        return render_default_config(
            service.project_name,
            {
                "materialsproject_api_key": "",
                "elements": "W",
                "gas_elements": "",
                "crystal_structures": "bcc",
                "target_n_atoms": "128",
                "scp_address": "",
            },
        )

    def _install_valid_config(self, service: ProjectCreationService) -> Path:
        config_path = service.project_dir / "config" / "project.config"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(self._valid_config_text(service), encoding="utf-8")
        return config_path

    def _collect_values(
        self,
        service: ProjectCreationService,
        responses: list[str],
    ) -> tuple[dict[str, str], Mock]:
        input_mock = Mock(side_effect=responses)
        wizard = ConfigWizard(service.project_name, input_fn=input_mock)
        return wizard.collect_prompt_values(CONFIG_PROMPTS), input_mock

    def test_setup_config_never_persists_materials_project_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            (project_dir / "config").mkdir(parents=True, exist_ok=True)
            service = self._make_service(project_dir)

            with patch.dict(os.environ, {}, clear=True):
                values, input_mock = self._collect_values(
                    service,
                    [
                        "w,cr,y,zr",
                        "",
                        "BCC, fcc",
                        "",
                        "user@host:/opt/nepflow",
                    ],
                )
                service.setup_config(
                    prompt_values={**values, "materialsproject_api_key": "mp-test-key"}
                )

            config_text = (project_dir / "config" / "project.config").read_text(encoding="utf-8")
            self.assertIn("api_key=\n", config_text)
            self.assertNotIn("mp-test-key", config_text)
            self.assertIn("elements=W,Cr,Y,Zr", config_text)
            self.assertIn("gas_elements=", config_text)
            self.assertIn("crystal_structures=bcc,fcc", config_text)
            self.assertIn("target_n_atoms=128", config_text)
            self.assertIn("scp_address=user@host:/opt/nepflow", config_text)
            self.assertIn("algorithm=information_entropy", config_text)
            self.assertIn("composition_aware_fps=false", config_text)
            self.assertIn("composition_aware_fps_frontier_fraction=0.10", config_text)
            self.assertIn("composition_aware_fps_ternary_weight=1.0", config_text)
            self.assertIn("composition_aware_fps_adaptive_retries=4", config_text)
            self.assertIn(
                "composition_aware_fps_descriptor_floor_fraction=0.95",
                config_text,
            )
            self.assertEqual(input_mock.call_count, 5)

    def test_materials_project_environment_key_is_not_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            (project_dir / "config").mkdir(parents=True, exist_ok=True)
            service = self._make_service(project_dir)

            with patch.dict(os.environ, {"MP_API_KEY": "environment-secret"}, clear=True):
                values, _ = self._collect_values(
                    service,
                    ["W", "", "BCC", "", "user@host:/opt/nepflow"],
                )
                service.setup_config(prompt_values=values)
                fetcher = build_materials_project_fetcher(
                    cache_dir=project_dir / "mp-cache",
                    client=object(),
                )

            config_text = (project_dir / "config" / "project.config").read_text(encoding="utf-8")
            self.assertIn("api_key=\n", config_text)
            self.assertNotIn("environment-secret", config_text)
            self.assertEqual(fetcher.api_key, "environment-secret")

    def test_unknown_element_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._make_service(Path(tmp))
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "Unknown element: Xx"):
                    self._collect_values(service, ["W,Xx"])

    def test_blank_elements_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._make_service(Path(tmp))
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "At least one element is required"):
                    self._collect_values(service, [""])

    def test_prompt_registry_is_generic(self) -> None:
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
        input_mock = Mock(return_value="from-prompt")
        with patch.dict(os.environ, {"FIRST_VALUE": "from-env"}, clear=True):
            values = ConfigWizard("demo", input_fn=input_mock).collect_prompt_values(prompts)

        self.assertEqual(
            values,
            {"first_value": "from-env", "secret_value": "from-prompt"},
        )

    def test_run_initializes_authoritative_state_db(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            service = self._make_service(project_dir)
            with patch.dict(os.environ, {}, clear=True):
                values, _ = self._collect_values(
                    service,
                    ["W", "", "BCC", "", "user@host:/opt/nepflow"],
                )
            service.run(prompt_values=values)

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
            assert config.magnetism.include_non_magnetic is True
            assert config.magnetism.moment_sets == ()
            assert stage_run is not None
            assert stage_run["stage"] == WorkflowStage.INIT.value
            assert stage_run["status"] == StageRunState.RUNNING.value

    def test_repeated_init_is_non_destructive_and_does_not_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            service = self._make_service(project_dir)
            values, _ = self._collect_values(
                service,
                ["W", "", "BCC", "", "user@host:/opt/nepflow"],
            )
            service.run(prompt_values=values)

            config_path = project_dir / "config" / "project.config"
            state_path = project_dir / "state.db"
            first_config = config_path.read_bytes()
            with StateStore(state_path) as store:
                first_project = store.get_project("demo")
                first_stage = store.get_stage_run("demo:init")

            with patch("builtins.input") as input_mock, patch("builtins.print"):
                service.run()

            assert config_path.read_bytes() == first_config
            input_mock.assert_not_called()
            with StateStore(state_path) as store:
                assert store.get_project("demo") == first_project
                assert store.get_stage_run("demo:init") == first_stage

    def test_creation_service_performs_no_terminal_io(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._make_service(Path(tmp))
            values, _ = self._collect_values(
                service,
                ["W", "", "BCC", "", "user@host:/opt/nepflow"],
            )
            with patch("builtins.input") as input_mock, patch("builtins.print") as print_mock:
                service.run(prompt_values=values)
            input_mock.assert_not_called()
            print_mock.assert_not_called()

    def test_invalid_rendered_config_does_not_create_authoritative_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._make_service(Path(tmp))
            invalid_config = self._valid_config_text(service).replace(
                "schema_version=1", "schema_version=999", 1
            )
            values, _ = self._collect_values(
                service,
                ["W", "", "BCC", "", ""],
            )

            with pytest.raises(ConfigurationError, match="schema_version"):
                with patch.object(
                    initialization_module,
                    "render_default_config",
                    return_value=invalid_config,
                ):
                    service.run(prompt_values=values)

            assert not (service.project_dir / "config" / "project.config").exists()
            assert not (service.project_dir / "state.db").exists()

    def test_state_transaction_rolls_back_project_when_stage_record_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._make_service(Path(tmp))
            values, _ = self._collect_values(
                service,
                ["W", "", "BCC", "", ""],
            )
            with patch.object(
                StateStore,
                "upsert_stage_run",
                side_effect=RuntimeError("stage write failed"),
            ):
                with pytest.raises(RuntimeError, match="stage write failed"):
                    service.run(prompt_values=values)

            assert not (service.project_dir / "config" / "project.config").exists()
            assert not (service.project_dir / "state.db").exists()

            service.run(prompt_values=values)
            with StateStore(service.state_file) as store:
                assert store.get_project("demo") is not None
                assert store.get_stage_run("demo:init") is not None

    def test_corrupt_existing_state_fails_without_replacing_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            state_path = project_dir / "state.db"
            state_path.write_bytes(b"not a sqlite database")
            service = self._make_service(project_dir)

            with pytest.raises(StateError, match="state database"):
                service.run()

            assert state_path.read_bytes() == b"not a sqlite database"

    def test_existing_state_without_config_requires_explicit_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            with StateStore(project_dir / "state.db"):
                pass

            with pytest.raises(ConfigurationError, match="canonical config"):
                self._make_service(project_dir).run()

    def test_legacy_marker_requires_explicit_migration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            (project_dir / ".project").write_text("generate", encoding="utf-8")

            with pytest.raises(StateError, match="explicit migration"):
                self._make_service(project_dir).run()

    def test_existing_config_without_state_requires_explicit_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            service = self._make_service(project_dir)
            self._install_valid_config(service)

            with pytest.raises(StateError, match="state.db is missing"):
                service.run()

            assert not (project_dir / "state.db").exists()

    def test_existing_state_without_project_requires_explicit_migration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            service = self._make_service(project_dir)
            self._install_valid_config(service)
            with StateStore(project_dir / "state.db"):
                pass

            with pytest.raises(StateError, match="no project record"):
                service.run()

            with StateStore(project_dir / "state.db") as store:
                assert store.get_project("demo") is None

    def test_existing_project_without_stage_requires_explicit_migration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            service = self._make_service(project_dir)
            values, _ = self._collect_values(
                service,
                ["W", "", "BCC", "", ""],
            )
            service.run(prompt_values=values)

            with StateStore(project_dir / "state.db") as store:
                store.connection.execute("DELETE FROM stage_runs WHERE project_id = ?", ("demo",))

            with pytest.raises(StateError, match="no stage history"):
                service.run()

            with StateStore(project_dir / "state.db") as store:
                assert store.get_stage_run("demo:init") is None


if __name__ == "__main__":
    unittest.main()
