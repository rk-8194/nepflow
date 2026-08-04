import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
STUBBED_MODULES = (
    "numpy",
    "ase",
    "ase.atoms",
    "ase.io",
    "common",
    "common.structure_identity",
)
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

    @property
    def shape(self):
        if self.values and isinstance(self.values[0], list):
            return (len(self.values), len(self.values[0]))
        return (len(self.values),)

    @property
    def flat(self):
        for value in self.values:
            if isinstance(value, list):
                for nested in value:
                    yield nested
            else:
                yield value

    @property
    def array(self):
        return self

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        return self.values[item]


class FakeIndices(list):
    def tolist(self):
        return list(self)


class FakePbc(list):
    def tolist(self):
        return list(self)


def install_stubs():
    numpy_module = types.ModuleType("numpy")
    numpy_module.ndarray = FakeArray
    numpy_module.array = lambda values, dtype=None: FakeArray(list(values))
    numpy_module.zeros = lambda shape: FakeArray([[0.0 for _ in range(shape[1])] for _ in range(shape[0])])
    numpy_module.where = lambda mask: (FakeIndices([i for i, ok in enumerate(mask.values) if ok]),)
    numpy_module.pi = 3.141592653589793
    numpy_module.ceil = lambda value: value
    numpy_module.unique = lambda values: sorted(set(values))
    numpy_module.linalg = types.SimpleNamespace(norm=lambda value: 1.0)
    sys.modules["numpy"] = numpy_module

    ase_module = types.ModuleType("ase")
    ase_module.__path__ = []
    sys.modules["ase"] = ase_module

    ase_atoms = types.ModuleType("ase.atoms")
    ase_atoms.Atoms = object
    sys.modules["ase.atoms"] = ase_atoms

    ase_io = types.ModuleType("ase.io")
    ase_io._outcar_atoms = {}
    ase_io.read = lambda path, *args, **kwargs: ase_io._outcar_atoms[str(path)]
    ase_io.iread = lambda *args, **kwargs: iter([])
    sys.modules["ase.io"] = ase_io

    common_module = types.ModuleType("common")
    common_module.__path__ = []
    sys.modules["common"] = common_module

    def hash_structure(atoms):
        payload = {
            "symbols": atoms.get_chemical_symbols(),
            "cell": [list(row) for row in atoms.get_cell()],
            "scaled_positions": [list(row) for row in atoms.get_scaled_positions()],
        }
        return "stub-structure-hash-" + json.dumps(payload, sort_keys=True)

    structure_identity = types.ModuleType("common.structure_identity")
    structure_identity.hash_structure = hash_structure
    sys.modules["common.structure_identity"] = structure_identity

    for name in (
        "testpkg",
        "testpkg.modules",
        "testpkg.modules.run_vasp",
        "testpkg.modules.train_nep",
    ):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    return ase_io


class FakeAtoms:
    def __init__(self, symbols, scaled_positions):
        self._symbols = symbols
        self._scaled_positions = scaled_positions
        self.info = {}
        self.arrays = {}
        self.pbc = FakePbc([True, True, True])
        self.calc = object()

    def get_chemical_symbols(self):
        return list(self._symbols)

    def get_cell(self):
        return FakeArray([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

    def get_scaled_positions(self):
        return self._scaled_positions

    def get_positions(self):
        return FakeArray(self._scaled_positions)

    def get_potential_energy(self):
        return -1.25

    def get_forces(self):
        return FakeArray([[0.1, 0.0, 0.0] for _ in self._symbols])

    def get_volume(self):
        return 1.0


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
    "testpkg.modules.run_vasp.prepare",
    SRC / "modules" / "run_vasp" / "prepare.py",
)
load_module(
    "testpkg.modules.train_nep._common",
    SRC / "modules" / "train_nep" / "_common.py",
)
train_prepare = load_module(
    "testpkg.modules.train_nep.prepare",
    SRC / "modules" / "train_nep" / "prepare.py",
)
restore_stubs()


def current_incar_hash() -> str:
    incar = train_prepare.inject_incar_defaults("ENCUT = 520\n", train_prepare.ConfigParser())
    return common.hash_incar_text(incar)


class TrainNepPrepareRegistryTests(unittest.TestCase):
    def make_project(self, root: Path) -> Path:
        project_dir = root / "projects" / "project_demo"
        (project_dir / "config" / "vasp").mkdir(parents=True, exist_ok=True)
        (project_dir / "vasp" / "jobs" / "train").mkdir(parents=True, exist_ok=True)
        (project_dir / "config" / "vasp" / "INCAR").write_text("ENCUT = 520\n", encoding="utf-8")
        (project_dir / "config" / "vasp" / "POTCAR_Si").write_bytes(b"Si-potcar-v1")
        return project_dir

    def write_identity_job(self, project_dir: Path, struct_name: str, atoms: FakeAtoms, reused_from: Path | None = None) -> Path:
        struct_dir = project_dir / "vasp" / "jobs" / "train" / struct_name
        struct_dir.mkdir(parents=True, exist_ok=True)
        identity = {
            "structure_hash": common.hash_structure(atoms),
            "incar_hash": current_incar_hash(),
            "potcar_hash": common.hash_potcar_bytes(b"Si-potcar-v1"),
        }
        (struct_dir / ".vasp_identity").write_text(json.dumps(identity), encoding="utf-8")
        status = {"status": "completed"}
        if reused_from is not None:
            status = {"status": "reused", "reused_from": str(reused_from.resolve())}
        (struct_dir / ".vasp_status").write_text(json.dumps(status), encoding="utf-8")
        return struct_dir

    def test_resolves_outcar_from_reused_registry_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = self.make_project(root)
            atoms = FakeAtoms(["Si"], [[0.0, 0.0, 0.0]])
            old_job = root / "projects" / "project_old" / "vasp" / "jobs" / "train" / "struct_0002"
            old_job.mkdir(parents=True)
            old_outcar = old_job / "OUTCAR"
            old_outcar.write_text("General timing\n", encoding="utf-8")
            ASE_IO._outcar_atoms[str(old_outcar)] = atoms

            common.upsert_registry_entry(
                root,
                current_incar_hash(),
                common.hash_potcar_bytes(b"Si-potcar-v1"),
                common.hash_structure(atoms),
                {"job_path": str(old_job.resolve())},
            )

            parsed = list(train_prepare._parse_structures([atoms], True, project_dir))

            self.assertEqual(len(parsed), 1)
            self.assertEqual(parsed[0]["energy"], -1.25)

    def test_resolves_by_hash_not_selected_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = self.make_project(root)
            first = FakeAtoms(["Si"], [[0.0, 0.0, 0.0]])
            second = FakeAtoms(["Si"], [[0.25, 0.0, 0.0]])
            struct_dir = self.write_identity_job(project_dir, "struct_0009", second)
            outcar = struct_dir / "OUTCAR"
            outcar.write_text("General timing\n", encoding="utf-8")
            ASE_IO._outcar_atoms[str(outcar)] = second

            parsed = list(train_prepare._parse_structures([first, second], True, project_dir))

            self.assertEqual(len(parsed), 1)
            self.assertEqual(parsed[0]["energy"], -1.25)

    def test_skips_incomplete_registry_outcar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = self.make_project(root)
            atoms = FakeAtoms(["Si"], [[0.0, 0.0, 0.0]])
            old_job = root / "projects" / "project_old" / "vasp" / "jobs" / "train" / "struct_0002"
            old_job.mkdir(parents=True)
            (old_job / "OUTCAR").write_text("not complete\n", encoding="utf-8")

            common.upsert_registry_entry(
                root,
                current_incar_hash(),
                common.hash_potcar_bytes(b"Si-potcar-v1"),
                common.hash_structure(atoms),
                {"job_path": str(old_job.resolve())},
            )

            parsed = list(train_prepare._parse_structures([atoms], True, project_dir))

            self.assertEqual(parsed, [])

    def test_write_xyz_file_includes_nep_virial_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "train.xyz"
            structures = iter([
                {
                    "energy": -1.25,
                    "pbc": [True, True, True],
                    "lattice": FakeArray([
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ]),
                    "species": ["Si"],
                    "positions": FakeArray([[0.1, 0.2, 0.3]]),
                    "forces": FakeArray([[0.4, 0.5, 0.6]]),
                    "virial": FakeArray([
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0],
                    ]),
                }
            ])

            count = train_prepare._write_xyz_file(
                output_path,
                structures,
                include_virial=True,
            )

            lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(count, 1)
            self.assertEqual(lines[0], "1")
            self.assertIn("energy=-1.2500000000", lines[1])
            self.assertIn('pbc="T T T"', lines[1])
            self.assertIn(
                'Lattice="1.0000000000 0.0000000000 0.0000000000 '
                '0.0000000000 1.0000000000 0.0000000000 '
                '0.0000000000 0.0000000000 1.0000000000"',
                lines[1],
            )
            self.assertIn("Properties=species:S:1:pos:R:3:force:R:3", lines[1])
            self.assertIn(
                'virial="1.0000000000 2.0000000000 3.0000000000 '
                '4.0000000000 5.0000000000 6.0000000000 '
                '7.0000000000 8.0000000000 9.0000000000"',
                lines[1],
            )
            self.assertEqual(
                lines[2].split(),
                [
                    "Si",
                    "0.1000000000",
                    "0.2000000000",
                    "0.3000000000",
                    "0.4000000000",
                    "0.5000000000",
                    "0.6000000000",
                ],
            )

    def test_write_xyz_file_omits_missing_virial_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "test.xyz"
            structures = iter([
                {
                    "energy": -1.25,
                    "pbc": [True, True, True],
                    "lattice": FakeArray([
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ]),
                    "species": ["Si"],
                    "positions": FakeArray([[0.1, 0.2, 0.3]]),
                    "forces": FakeArray([[0.4, 0.5, 0.6]]),
                    "virial": None,
                }
            ])

            count = train_prepare._write_xyz_file(
                output_path,
                structures,
                include_virial=True,
            )

            lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(count, 1)
            self.assertNotIn("virial=", lines[1])
            self.assertIn("Properties=species:S:1:pos:R:3:force:R:3", lines[1])


if __name__ == "__main__":
    unittest.main()
