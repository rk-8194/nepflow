import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


class FakeAtoms:
    def __init__(self, *args, **kwargs):
        self.info = {}
        self.cell = kwargs.get(
            "cell",
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )
        self.pbc = [True, True, True]

    def copy(self):
        other = FakeAtoms()
        other.info = dict(self.info)
        other.cell = self.cell
        return other

    def set_chemical_symbols(self, symbols):
        self.info["symbols"] = list(symbols)

    def get_chemical_symbols(self):
        return self.info.get("symbols", ["Si"])

    def get_cell(self):
        return self.cell

    def get_scaled_positions(self):
        return [[0.0, 0.0, 0.0] for _ in self.get_chemical_symbols()]

    def rattle(self, stdev, seed):
        self.info["rattle"] = (stdev, seed)

    def append(self, atom):
        self.info.setdefault("appended", []).append(atom)

    def __mul__(self, other):
        return self.copy()


class FakeAtom:
    def __init__(self, symbol=None, position=None):
        self.symbol = symbol
        self.position = position


class FakeCompositionGrid:
    def __init__(self, elements, **kwargs):
        self.elements = elements

    def generate(self):
        return [{self.elements[0]: 1.0}] if self.elements else []

    @staticmethod
    def format_composition(comp):
        return "-".join(f"{k}{v}" for k, v in comp.items())


class FakePerturbationEngine:
    last_instance = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.process_kwargs = {}
        FakePerturbationEngine.last_instance = self

    def process(self, *args, **kwargs):
        self.process_kwargs = dict(kwargs)
        return None

    def get_summary(self):
        return {"total": 0, "by_type": {}, "by_config": {}}


class _FakeGeneratorBase:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class FakeMaterialsProjectGenerator(_FakeGeneratorBase):
    def generate(self, *args, **kwargs):
        first = FakeAtoms()
        first.info.update({
            "formula": "WC",
            "material_id": "mp-1894",
            "structure_name": "unknown",
        })
        second = FakeAtoms()
        second.info.update({
            "formula": "W",
            "material_id": "mp-91",
            "structure_name": "bcc",
        })
        return [first, second]


class FakeRandomSolidSolutionGenerator(_FakeGeneratorBase):
    pass


class FakeSegregatedGenerator(_FakeGeneratorBase):
    pass


class FakeSQSGenerator(_FakeGeneratorBase):
    pass


def install_test_stubs():
    package_names = [
        "testpkg",
        "testpkg.modules",
        "testpkg.modules.generate",
    ]
    for name in package_names:
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module

    ase_module = types.ModuleType("ase")
    ase_module.Atoms = FakeAtoms
    ase_module.Atom = FakeAtom
    sys.modules["ase"] = ase_module

    ase_build = types.ModuleType("ase.build")
    ase_build.bulk = lambda *args, **kwargs: FakeAtoms()
    sys.modules["ase.build"] = ase_build

    ase_io = types.ModuleType("ase.io")
    ase_io.write = lambda *args, **kwargs: None
    sys.modules["ase.io"] = ase_io

    common_package = types.ModuleType("common")
    common_package.__path__ = []
    sys.modules["common"] = common_package

    identity_spec = importlib.util.spec_from_file_location(
        "common.structure_identity",
        SRC / "common" / "structure_identity.py",
    )
    identity_module = importlib.util.module_from_spec(identity_spec)
    sys.modules["common.structure_identity"] = identity_module
    assert identity_spec.loader is not None
    identity_spec.loader.exec_module(identity_module)

    base_spec = importlib.util.spec_from_file_location(
        "testpkg.modules.base",
        SRC / "modules" / "base.py",
    )
    base_module = importlib.util.module_from_spec(base_spec)
    sys.modules["testpkg.modules.base"] = base_module
    assert base_spec.loader is not None
    base_spec.loader.exec_module(base_module)

    composition_module = types.ModuleType("testpkg.modules.generate.composition")
    composition_module.CompositionGrid = FakeCompositionGrid
    sys.modules["testpkg.modules.generate.generators.composition"] = composition_module

    generators_package = types.ModuleType("testpkg.modules.generate.generators")
    generators_package.__path__ = []
    sys.modules["testpkg.modules.generate.generators"] = generators_package

    configurational_module = types.ModuleType("testpkg.modules.generate.generators.configurational")
    configurational_module.MaterialsProjectGenerator = FakeMaterialsProjectGenerator
    configurational_module.RandomSolidSolutionGenerator = FakeRandomSolidSolutionGenerator
    configurational_module.SegregatedGenerator = FakeSegregatedGenerator
    configurational_module.SQSGenerator = FakeSQSGenerator
    sys.modules["testpkg.modules.generate.generators.configurational"] = configurational_module

    materials_project_module = types.ModuleType("testpkg.modules.generate.generators.materials_project")
    materials_project_module.get_materials_project_fetcher = lambda config: object()
    sys.modules["testpkg.modules.generate.generators.materials_project"] = materials_project_module

    structure_generation_module = types.ModuleType("testpkg.modules.generate.generators.structure_generation")
    structure_generation_module.PerturbationEngine = FakePerturbationEngine
    sys.modules["testpkg.modules.generate.generators.structure_generation"] = structure_generation_module

    generators_package.CompositionGrid = FakeCompositionGrid
    generators_package.MaterialsProjectGenerator = getattr(
        configurational_module, "MaterialsProjectGenerator"
    )
    generators_package.RandomSolidSolutionGenerator = getattr(
        configurational_module, "RandomSolidSolutionGenerator"
    )
    generators_package.SegregatedGenerator = getattr(
        configurational_module, "SegregatedGenerator"
    )
    generators_package.SQSGenerator = getattr(configurational_module, "SQSGenerator")
    generators_package.PerturbationEngine = FakePerturbationEngine
    generators_package.get_materials_project_fetcher = (
        materials_project_module.get_materials_project_fetcher
    )


def load_generate_stage():
    install_test_stubs()
    spec = importlib.util.spec_from_file_location(
        "testpkg.modules.generate.generate",
        SRC / "modules" / "generate" / "generate.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["testpkg.modules.generate.generate"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.GenerateStage


GenerateStage = load_generate_stage()


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
    lines.extend(
        [
            "[hpc]",
            f"scp_address={scp_address}",
            "",
        ]
    )
    (config_dir / "project.config").write_text("\n".join(lines), encoding="utf-8")


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
            bases = [FakeAtoms()]

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
            bases = [FakeAtoms()]

            with patch.object(stage, "prepare", return_value=bases):
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
            bases = [FakeAtoms()]
            config, settings = stage.load_config()

            with patch.object(stage, "_load_saved_bases", return_value=bases) as load_saved_bases:
                with patch.object(stage, "prepare") as prepare:
                    result = stage.resume_if_needed(config, settings)

            load_saved_bases.assert_called_once()
            prepare.assert_not_called()
            self.assertIs(result, bases)

    def test_run_raises_for_missing_generation_section(self) -> None:
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
            bases = [FakeAtoms()]
            summary = {"total": 1, "by_type": {"unperturbed": 1}, "by_config": {"debug": 1}}

            with patch.object(stage, "resume_if_needed", return_value=bases) as resume_if_needed:
                with patch.object(stage, "execute", return_value=summary) as execute:
                    with patch.object(stage, "finalize") as finalize:
                        stage.run()

            resume_if_needed.assert_called_once()
            execute.assert_called_once()
            finalize.assert_called_once_with(summary)

    def test_build_engine_uses_elastic_stress_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(project_dir)
            stage = self.create_stage(project_dir)
            config, settings = stage.load_config()

            engine = stage._build_engine(
                config,
                settings["target_n_atoms"],
                settings["random_seed"],
            )

            self.assertTrue(engine.kwargs["elastic_stress_enabled"])
            self.assertEqual(
                engine.kwargs["elastic_strain_amplitudes"],
                [-0.02, -0.01, -0.005, 0.005, 0.01, 0.02],
            )
            self.assertNotIn("strain_limit", engine.kwargs)
            self.assertEqual(engine.kwargs["rattle_std_min"], 0.015)
            self.assertEqual(engine.kwargs["rattle_std_max"], 0.06)

    def test_build_engine_accepts_elastic_stress_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(project_dir)
            config_path = project_dir / "config" / "project.config"
            config_text = config_path.read_text(encoding="utf-8")
            config_path.write_text(
                config_text.replace(
                    "n_workers=1",
                    "\n".join(
                        [
                            "n_workers=1",
                            "elastic_stress_enabled=false",
                            "elastic_strain_amplitudes=-0.03,0.03",
                            "rattle_std_min=0.01",
                            "rattle_std_max=0.08",
                        ]
                    ),
                ),
                encoding="utf-8",
            )
            stage = self.create_stage(project_dir)
            config, settings = stage.load_config()

            engine = stage._build_engine(
                config,
                settings["target_n_atoms"],
                settings["random_seed"],
            )

            self.assertFalse(engine.kwargs["elastic_stress_enabled"])
            self.assertEqual(engine.kwargs["elastic_strain_amplitudes"], [-0.03, 0.03])
            self.assertNotIn("strain_limit", engine.kwargs)
            self.assertEqual(engine.kwargs["rattle_std_min"], 0.01)
            self.assertEqual(engine.kwargs["rattle_std_max"], 0.08)

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

            engine = stage._build_engine(
                config,
                settings["target_n_atoms"],
                settings["random_seed"],
            )

            self.assertTrue(engine.kwargs["liquid_enabled"])
            self.assertEqual(engine.kwargs["liquid_temperature_k"], 2500.0)
            self.assertEqual(engine.kwargs["liquid_timestep_fs"], 1.5)
            self.assertEqual(engine.kwargs["liquid_equilibration_steps"], 25)
            self.assertEqual(engine.kwargs["liquid_steps_between_snapshots"], 11)
            self.assertEqual(engine.kwargs["liquid_friction"], 0.05)

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

            FakePerturbationEngine.last_instance = None
            summary = stage.execute(config, settings, [FakeAtoms()])

            self.assertEqual(summary, {"total": 0, "by_type": {}, "by_config": {}})
            self.assertIsNotNone(FakePerturbationEngine.last_instance)
            self.assertEqual(
                FakePerturbationEngine.last_instance.process_kwargs["n_liquid_configurations"],
                4,
            )
            self.assertEqual(
                FakePerturbationEngine.last_instance.process_kwargs["n_liquid_snapshots"],
                7,
            )

    def test_prepare_logs_explicit_materials_project_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            write_project_config(project_dir)
            config_path = project_dir / "config" / "project.config"
            config_text = config_path.read_text(encoding="utf-8")
            config_path.write_text(
                config_text.replace(
                    "use_materials_project=false",
                    "use_materials_project=true",
                ),
                encoding="utf-8",
            )
            stage = self.create_stage(project_dir)
            config, settings = stage.load_config()

            with patch("testpkg.modules.generate.generate.logger.info") as info_mock:
                bases = stage.prepare(config, settings)

            self.assertIsNotNone(bases)
            logged_messages = [" ".join(str(arg) for arg in call.args) for call in info_mock.call_args_list]
            self.assertTrue(any("MaterialsProject" in message for message in logged_messages))
            self.assertTrue(any("WC" in message and "mp-1894" in message for message in logged_messages))
            self.assertTrue(any("W" in message and "mp-91" in message and "bcc" in message for message in logged_messages))


if __name__ == "__main__":
    unittest.main()
