import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")
from ase import Atoms  # noqa: E402

from modules.generate import generate as generate_module


GenerateStage = generate_module.GenerateStage


def write_project_config(
    project_dir: Path,
    include_generation: bool = True,
    scp_address: str = "",
) -> None:
    config_dir = project_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "[project]",
        "random_seed=7",
        "",
        "[composition]",
        "elements=Si,Ge",
        "composition_step=0.5",
        "include_pure_elements=true",
        "include_binaries=true",
        "include_ternaries=false",
        "",
    ]
    if include_generation:
        lines.extend(
            [
                "[generation]",
                "crystal_structures=bcc,fcc",
                "target_n_atoms=16",
                "n_workers=1",
                "use_materials_project=false",
                "use_random_solid_solution=false",
                "use_sqs=false",
                "use_segregated=false",
                "",
            ]
        )
    lines.extend(["[hpc]", f"scp_address={scp_address}", ""])
    (config_dir / "project.config").write_text("\n".join(lines), encoding="utf-8")


def make_atoms(symbols: str = "Si") -> Atoms:
    n_atoms = len(Atoms(symbols))
    return Atoms(
        symbols,
        positions=np.arange(3 * n_atoms).reshape((-1, 3)).astype(float),
        cell=np.eye(3) * 3.0,
        pbc=True,
    )


class GenerateStageTests(unittest.TestCase):
    def create_stage(self, project_dir: Path) -> GenerateStage:
        return GenerateStage(
            project_name="demo",
            config_file=project_dir / "config" / "demo.yaml",
            state_file=project_dir / "state.db",
            project_dir=project_dir,
            debug=False,
        )

    def test_run_seeds_only_skips_perturbations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(project_dir)
            stage = self.create_stage(project_dir)
            bases = [make_atoms()]

            with patch.object(stage, "prepare", return_value=bases) as prepare:
                with patch.object(stage, "execute") as execute:
                    stage.run(seeds_only=True)

            prepare.assert_called_once()
            execute.assert_not_called()

    def test_run_seeds_only_offers_upload_when_scp_address_is_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "project_demo"
            write_project_config(project_dir, scp_address="user@host:/opt/nepflow")
            stage = self.create_stage(project_dir)

            with patch.object(stage, "prepare", return_value=[make_atoms()]):
                with patch("builtins.input", return_value="y") as input_mock:
                    with patch("subprocess.run") as run_mock:
                        stage.run(seeds_only=True)

            input_mock.assert_called_once()
            run_mock.assert_called_once_with(
                [
                    "scp",
                    "-r",
                    str(project_dir),
                    "user@host:/opt/nepflow/projects/project_demo",
                ],
                check=True,
            )

    def test_resume_if_needed_reuses_existing_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(project_dir)
            seeds_file = project_dir / "structures" / "seeds" / "base_structures.xyz"
            seeds_file.parent.mkdir(parents=True, exist_ok=True)
            seeds_file.write_text("stub", encoding="utf-8")
            stage = self.create_stage(project_dir)
            config, settings = stage.load_config()
            bases = [make_atoms()]

            with patch.object(stage, "_load_saved_bases", return_value=bases) as load:
                with patch.object(stage, "prepare") as prepare:
                    result = stage.resume_if_needed(config, settings)

            load.assert_called_once_with(seeds_file)
            prepare.assert_not_called()
            self.assertIs(result, bases)

    def test_load_config_requires_generation_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(project_dir, include_generation=False)
            stage = self.create_stage(project_dir)

            with self.assertRaisesRegex(ValueError, r"Config missing \[generation\]"):
                stage.load_config()

    def test_prepare_returns_none_when_no_generators_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(project_dir)
            stage = self.create_stage(project_dir)
            config, settings = stage.load_config()

            with patch.object(stage, "_build_compositions", return_value=[{"Si": 1.0}]):
                with patch.object(stage, "_build_generators", return_value=[]):
                    result = stage.prepare(config, settings)

            self.assertIsNone(result)

    def test_run_executes_and_finalizes_after_prepare(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(project_dir)
            stage = self.create_stage(project_dir)
            bases = [make_atoms()]
            summary = {"total": 1, "by_type": {"unperturbed": 1}, "by_config": {"debug": 1}}

            with patch.object(stage, "resume_if_needed", return_value=bases) as resume:
                with patch.object(stage, "execute", return_value=summary) as execute:
                    with patch.object(stage, "finalize") as finalize:
                        stage.run()

            resume.assert_called_once()
            execute.assert_called_once()
            finalize.assert_called_once_with(summary)

    def test_build_engine_passes_elastic_stress_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(project_dir)
            stage = self.create_stage(project_dir)
            config, settings = stage.load_config()

            with patch.object(generate_module, "PerturbationEngine") as engine_cls:
                stage._build_engine(config, settings["target_n_atoms"], settings["random_seed"])

            kwargs = engine_cls.call_args.kwargs
            self.assertTrue(kwargs["elastic_stress_enabled"])
            self.assertEqual(kwargs["elastic_strain_amplitudes"], [-0.02, -0.01, -0.005, 0.005, 0.01, 0.02])
            self.assertNotIn("strain_limit", kwargs)
            self.assertEqual(kwargs["rattle_std_min"], 0.015)
            self.assertEqual(kwargs["rattle_std_max"], 0.06)

    def test_build_engine_accepts_liquid_perturbation_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(project_dir)
            config_path = project_dir / "config" / "project.config"
            config_text = config_path.read_text(encoding="utf-8")
            config_path.write_text(
                config_text.replace(
                    "use_segregated=false",
                    "\n".join(
                        [
                            "use_segregated=false",
                            "use_liquid=true",
                            "n_liquid_configurations=4",
                            "n_liquid_snapshots=7",
                            "liquid_temperature=2500",
                            "liquid_timestep_fs=1.5",
                            "liquid_equilibration_steps=25",
                            "liquid_steps_between_snapshots=11",
                            "liquid_friction=0.05",
                        ]
                    ),
                ),
                encoding="utf-8",
            )
            stage = self.create_stage(project_dir)
            config, settings = stage.load_config()

            with patch.object(generate_module, "PerturbationEngine") as engine_cls:
                stage._build_engine(config, settings["target_n_atoms"], settings["random_seed"])

            kwargs = engine_cls.call_args.kwargs
            self.assertTrue(kwargs["liquid_enabled"])
            self.assertEqual(kwargs["liquid_temperature_k"], 2500.0)
            self.assertEqual(kwargs["liquid_timestep_fs"], 1.5)
            self.assertEqual(kwargs["liquid_equilibration_steps"], 25)
            self.assertEqual(kwargs["liquid_steps_between_snapshots"], 11)
            self.assertEqual(kwargs["liquid_friction"], 0.05)

    def test_execute_passes_liquid_perturbation_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(project_dir)
            config_path = project_dir / "config" / "project.config"
            config_text = config_path.read_text(encoding="utf-8")
            config_path.write_text(
                config_text.replace(
                    "use_segregated=false",
                    "\n".join(
                        [
                            "use_segregated=false",
                            "use_liquid=true",
                            "n_liquid_configurations=4",
                            "n_liquid_snapshots=7",
                        ]
                    ),
                ),
                encoding="utf-8",
            )
            stage = self.create_stage(project_dir)
            config, settings = stage.load_config()
            engine = Mock()
            engine.get_summary.return_value = {"total": 0, "by_type": {}, "by_config": {}}

            with patch.object(stage, "_build_engine", return_value=engine):
                summary = stage.execute(config, settings, [make_atoms()])

            self.assertEqual(summary, engine.get_summary.return_value)
            kwargs = engine.process.call_args.kwargs
            self.assertEqual(kwargs["n_liquid_configurations"], 4)
            self.assertEqual(kwargs["n_liquid_snapshots"], 7)

    def test_prepare_logs_explicit_materials_project_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(project_dir)
            stage = self.create_stage(project_dir)
            config, settings = stage.load_config()
            first = make_atoms("WC")
            first.info.update({"formula": "WC", "material_id": "mp-1894", "structure_name": "unknown"})
            second = make_atoms("W")
            second.info.update({"formula": "W", "material_id": "mp-91", "structure_name": "bcc"})
            generator = Mock()
            generator.generate.return_value = [first, second]

            with patch.object(stage, "_build_compositions", return_value=[{"W": 1.0}]):
                with patch.object(stage, "_build_generators", return_value=[("MaterialsProject", generator)]):
                    with patch.object(generate_module.logger, "info") as info_mock:
                        bases = stage.prepare(config, settings)

            self.assertEqual(len(bases), 2)
            logged_messages = [" ".join(str(arg) for arg in call.args) for call in info_mock.call_args_list]
            self.assertTrue(any("MaterialsProject" in message for message in logged_messages))
            self.assertTrue(any("WC" in message and "mp-1894" in message for message in logged_messages))
            self.assertTrue(any("W" in message and "mp-91" in message and "bcc" in message for message in logged_messages))


if __name__ == "__main__":
    unittest.main()
