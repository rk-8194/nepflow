import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def ensure_package(name: str) -> None:
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


for package_name in (
    "testpkg",
    "testpkg.modules",
    "testpkg.modules.train_nep",
):
    ensure_package(package_name)

numpy_module = types.ModuleType("numpy")
numpy_module.ndarray = object
sys.modules["numpy"] = numpy_module

ase_module = types.ModuleType("ase")
ase_module.__path__ = []
sys.modules["ase"] = ase_module

ase_atoms = types.ModuleType("ase.atoms")
ase_atoms.Atoms = object
sys.modules["ase.atoms"] = ase_atoms

ase_io = types.ModuleType("ase.io")
ase_io.read = lambda *args, **kwargs: []
sys.modules["ase.io"] = ase_io

base_module = load_module("testpkg.modules.base", SRC / "modules" / "base.py")

prepare_module = types.ModuleType("testpkg.modules.train_nep.prepare")
prepare_module.prepare_dataset = lambda *args, **kwargs: 0
sys.modules["testpkg.modules.train_nep.prepare"] = prepare_module

submit_module = types.ModuleType("testpkg.modules.train_nep.submit")
submit_module.submit_training_job = lambda *args, **kwargs: "0"
sys.modules["testpkg.modules.train_nep.submit"] = submit_module

launcher_module = types.ModuleType("testpkg.modules.train_nep.launcher")
launcher_module.run_launcher = lambda *args, **kwargs: None
launcher_module.read_train_status = lambda *args, **kwargs: {}
launcher_module.write_train_status = lambda *args, **kwargs: None
sys.modules["testpkg.modules.train_nep.launcher"] = launcher_module

train_nep_module = load_module(
    "testpkg.modules.train_nep.train_nep",
    SRC / "modules" / "train_nep" / "train_nep.py",
)
TrainNepStage = train_nep_module.TrainNepStage


class TrainNepConfigTests(unittest.TestCase):
    def create_stage(self, project_dir: Path) -> TrainNepStage:
        return TrainNepStage(
            project_name="demo",
            config_file=project_dir / "config" / "demo.yaml",
            state_file=project_dir / "state.db",
            project_dir=project_dir,
            debug=False,
        )

    def test_generate_nep_config_copies_lambda_shear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            config_dir = project_dir / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            dataset_dir = project_dir / "nep" / "datasets" / "dataset_0001"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / "project.config").write_text(
                "\n".join(
                    [
                        "[composition]",
                        "elements=Si,Ge",
                        "",
                        "[train_nep]",
                        "population=99",
                        "batch=123",
                        "generation=456",
                        "lambda_e=2.0",
                        "lambda_f=3.0",
                        "lambda_v=4.0",
                        "lambda_shear=5.0",
                    ]
                ),
                encoding="utf-8",
            )

            stage = self.create_stage(project_dir)
            stage._generate_nep_config(stage._load_config(), dataset_dir)

            nep_in = (dataset_dir / "nep.in").read_text(encoding="utf-8")
            self.assertIn("lambda_e 2.0", nep_in)
            self.assertIn("lambda_f 3.0", nep_in)
            self.assertIn("lambda_v 4.0", nep_in)
            self.assertIn("lambda_shear 5.0", nep_in)


if __name__ == "__main__":
    unittest.main()
