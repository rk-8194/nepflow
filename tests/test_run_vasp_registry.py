import configparser
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
STUBBED_MODULES = ("numpy", "ase", "ase.io")
ORIGINAL_MODULES = {name: sys.modules.get(name) for name in STUBBED_MODULES}


class FakeMask:
    def __init__(self, values):
        self.values = values

    def sum(self):
        return sum(1 for value in self.values if value)


class FakeArray:
    def __init__(self, values):
        self.values = values

    def __eq__(self, other):
        return FakeMask([value == other for value in self.values])

    def __iter__(self):
        return iter(self.values)


class FakeIndices(list):
    def tolist(self):
        return list(self)


def install_stubs():
    numpy_module = types.ModuleType("numpy")
    numpy_module.array = lambda values, dtype=None: FakeArray(list(values))
    numpy_module.where = lambda mask: (FakeIndices([i for i, ok in enumerate(mask.values) if ok]),)
    numpy_module.pi = 3.141592653589793
    numpy_module.ceil = lambda value: value
    numpy_module.unique = lambda values: sorted(set(values))
    numpy_module.linalg = types.SimpleNamespace(norm=lambda value: 1.0)
    sys.modules["numpy"] = numpy_module

    ase_module = types.ModuleType("ase")
    ase_module.__path__ = []
    sys.modules["ase"] = ase_module

    ase_io = types.ModuleType("ase.io")
    ase_io._frames = {}
    ase_io.iread = lambda path, format=None: iter(ase_io._frames[str(path)])
    ase_io.read = lambda *args, **kwargs: None
    sys.modules["ase.io"] = ase_io

    for name in ("testpkg", "testpkg.modules", "testpkg.modules.run_vasp"):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    return ase_io


class FakeAtoms:
    def __init__(self, symbols, scaled_positions, cell=None):
        self._symbols = symbols
        self._scaled_positions = scaled_positions
        self._cell = cell or [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        self.info = {}

    def get_chemical_symbols(self):
        return list(self._symbols)

    def get_cell(self):
        return self._cell

    def get_scaled_positions(self):
        return self._scaled_positions


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def restore_stubs():
    for name, module in ORIGINAL_MODULES.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


ASE_IO = install_stubs()
common = load_module(
    "testpkg.modules.run_vasp._common",
    SRC / "modules" / "run_vasp" / "_common.py",
)
load_module(
    "testpkg.modules.base",
    SRC / "modules" / "base.py",
)
prepare = load_module(
    "testpkg.modules.run_vasp.prepare",
    SRC / "modules" / "run_vasp" / "prepare.py",
)
launcher = load_module(
    "testpkg.modules.run_vasp.launcher",
    SRC / "modules" / "run_vasp" / "launcher.py",
)
restore_stubs()


class RunVaspRegistryTests(unittest.TestCase):
    def make_project(self, root: Path, name: str = "demo") -> Path:
        project_dir = root / "projects" / f"project_{name}"
        (project_dir / "config" / "vasp").mkdir(parents=True, exist_ok=True)
        (project_dir / "structures" / "selected").mkdir(parents=True, exist_ok=True)
        (project_dir / "config" / "vasp" / "INCAR").write_text(
            "ENCUT = 520\nNCORE = 16\nKPAR = 1\n",
            encoding="utf-8",
        )
        (project_dir / "config" / "vasp" / "POTCAR_Si").write_bytes(b"Si-potcar-v1")
        (project_dir / "structures" / "selected" / "train.xyz").write_text("train", encoding="utf-8")
        (project_dir / "structures" / "selected" / "test.xyz").write_text("test", encoding="utf-8")
        return project_dir

    def test_structure_hash_changes_with_structure(self) -> None:
        first = FakeAtoms(["Si"], [[0.0, 0.0, 0.0]])
        moved = FakeAtoms(["Si"], [[0.25, 0.0, 0.0]])
        larger_cell = FakeAtoms(
            ["Si"],
            [[0.0, 0.0, 0.0]],
            cell=[[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )

        self.assertNotEqual(common.hash_structure(first), common.hash_structure(moved))
        self.assertNotEqual(common.hash_structure(first), common.hash_structure(larger_cell))

    def test_incar_hash_ignores_resource_params(self) -> None:
        original = "ENCUT = 520\nNCORE = 16\nKPAR = 1\nISMEAR = 0\n"
        rewritten = "ENCUT = 520\nNCORE = 64\nKPAR = 4\nISMEAR = 0\n"

        self.assertEqual(common.hash_incar_text(original), common.hash_incar_text(rewritten))

    def test_potcar_hash_changes_when_content_changes(self) -> None:
        self.assertNotEqual(
            common.hash_potcar_bytes(b"Si-potcar-v1"),
            common.hash_potcar_bytes(b"Si-potcar-v2"),
        )

    def test_prepare_marks_registry_match_as_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = self.make_project(root)
            atoms = FakeAtoms(["Si"], [[0.0, 0.0, 0.0]])
            ASE_IO._frames[str(project_dir / "structures" / "selected" / "train.xyz")] = [atoms]
            ASE_IO._frames[str(project_dir / "structures" / "selected" / "test.xyz")] = []

            completed_job = root / "projects" / "project_old" / "vasp" / "jobs" / "train" / "struct_0007"
            completed_job.mkdir(parents=True)
            (completed_job / "OUTCAR").write_text("General timing\n", encoding="utf-8")
            incar = prepare.inject_incar_defaults("ENCUT = 520\nNCORE = 16\nKPAR = 1\n", configparser.ConfigParser())
            incar_hash = common.hash_incar_text(incar)
            potcar_hash = common.hash_potcar_bytes(b"Si-potcar-v1")
            structure_hash = common.hash_structure(atoms)
            common.upsert_registry_entry(
                root,
                incar_hash,
                potcar_hash,
                structure_hash,
                {"job_path": str(completed_job.resolve())},
            )

            prepare.prepare_jobs(
                configparser.ConfigParser(),
                project_dir / "config" / "vasp",
                project_dir / "structures" / "selected",
                project_dir / "vasp" / "jobs",
                ["train", "test"],
                project_dir=project_dir,
                project_name="demo",
            )

            status = json.loads(
                (project_dir / "vasp" / "jobs" / "train" / "struct_0000" / ".vasp_status")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(status["status"], "reused")
            self.assertEqual(Path(status["reused_from"]), completed_job.resolve())

    def test_prepare_does_not_reuse_when_incar_differs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = self.make_project(root)
            (project_dir / "config" / "vasp" / "INCAR").write_text("ENCUT = 600\n", encoding="utf-8")
            atoms = FakeAtoms(["Si"], [[0.0, 0.0, 0.0]])
            ASE_IO._frames[str(project_dir / "structures" / "selected" / "train.xyz")] = [atoms]
            ASE_IO._frames[str(project_dir / "structures" / "selected" / "test.xyz")] = []

            completed_job = root / "projects" / "project_old" / "vasp" / "jobs" / "train" / "struct_0007"
            completed_job.mkdir(parents=True)
            (completed_job / "OUTCAR").write_text("General timing\n", encoding="utf-8")
            common.upsert_registry_entry(
                root,
                common.hash_incar_text("ENCUT = 520\n"),
                common.hash_potcar_bytes(b"Si-potcar-v1"),
                common.hash_structure(atoms),
                {"job_path": str(completed_job.resolve())},
            )

            prepare.prepare_jobs(
                configparser.ConfigParser(),
                project_dir / "config" / "vasp",
                project_dir / "structures" / "selected",
                project_dir / "vasp" / "jobs",
                ["train", "test"],
                project_dir=project_dir,
                project_name="demo",
            )

            status = json.loads(
                (project_dir / "vasp" / "jobs" / "train" / "struct_0000" / ".vasp_status")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(status["status"], "pending")

    def test_prepare_does_not_reuse_when_potcar_differs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = self.make_project(root)
            (project_dir / "config" / "vasp" / "POTCAR_Si").write_bytes(b"Si-potcar-v2")
            atoms = FakeAtoms(["Si"], [[0.0, 0.0, 0.0]])
            ASE_IO._frames[str(project_dir / "structures" / "selected" / "train.xyz")] = [atoms]
            ASE_IO._frames[str(project_dir / "structures" / "selected" / "test.xyz")] = []

            completed_job = root / "projects" / "project_old" / "vasp" / "jobs" / "train" / "struct_0007"
            completed_job.mkdir(parents=True)
            (completed_job / "OUTCAR").write_text("General timing\n", encoding="utf-8")
            incar = prepare.inject_incar_defaults("ENCUT = 520\nNCORE = 16\nKPAR = 1\n", configparser.ConfigParser())
            common.upsert_registry_entry(
                root,
                common.hash_incar_text(incar),
                common.hash_potcar_bytes(b"Si-potcar-v1"),
                common.hash_structure(atoms),
                {"job_path": str(completed_job.resolve())},
            )

            prepare.prepare_jobs(
                configparser.ConfigParser(),
                project_dir / "config" / "vasp",
                project_dir / "structures" / "selected",
                project_dir / "vasp" / "jobs",
                ["train", "test"],
                project_dir=project_dir,
                project_name="demo",
            )

            status = json.loads(
                (project_dir / "vasp" / "jobs" / "train" / "struct_0000" / ".vasp_status")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(status["status"], "pending")

    def test_register_completed_job_writes_registry_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            struct_dir = root / "projects" / "project_demo" / "vasp" / "jobs" / "train" / "struct_0042"
            struct_dir.mkdir(parents=True)
            identity = {
                "structure_hash": "structure-hash",
                "incar_hash": "incar-hash",
                "potcar_hash": "potcar-hash",
            }
            (struct_dir / ".vasp_identity").write_text(json.dumps(identity), encoding="utf-8")
            (struct_dir / "OUTCAR").write_text("General timing\n", encoding="utf-8")

            launcher._register_completed_job(
                struct_dir,
                root,
                "demo",
                "train",
                42,
            )

            registry = common.read_completed_registry(root)
            entry = registry["jobs"]["incar-hash"]["potcar-hash"]["structure-hash"]
            self.assertEqual(Path(entry["job_path"]), struct_dir.resolve())
            self.assertEqual(entry["project_name"], "demo")
            self.assertEqual(entry["dataset"], "train")
            self.assertEqual(entry["selected_index"], 42)
            self.assertIn("completed_at", entry)


if __name__ == "__main__":
    unittest.main()
