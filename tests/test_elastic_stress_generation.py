import sys
import importlib.util
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class FakeMatrix:
    def __init__(self, rows):
        self.rows = [[float(value) for value in row] for row in rows]

    def __getitem__(self, item):
        if isinstance(item, tuple):
            row, col = item
            return self.rows[row][col]
        if isinstance(item, slice):
            return self
        return self.rows[item]

    def __setitem__(self, item, value):
        row, col = item
        self.rows[row][col] = float(value)

    def __matmul__(self, other):
        other_rows = other.rows if isinstance(other, FakeMatrix) else other
        return FakeMatrix([
            [
                sum(self.rows[i][k] * other_rows[k][j] for k in range(3))
                for j in range(3)
            ]
            for i in range(3)
        ])

    def __sub__(self, other):
        return FakeMatrix([
            [self.rows[i][j] - other.rows[i][j] for j in range(3)]
            for i in range(3)
        ])

    def __mul__(self, scalar):
        return FakeMatrix([
            [value * float(scalar) for value in row]
            for row in self.rows
        ])

    def copy(self):
        return FakeMatrix([list(row) for row in self.rows])

    def reshape(self, *_args):
        return self

    def tolist(self):
        return [value for row in self.rows for value in row]


class FakeRandomState:
    def __init__(self, seed):
        self.seed = seed

    def uniform(self, low, high):
        return (float(low) + float(high)) / 2.0


def fake_eye(n):
    return FakeMatrix([
        [1.0 if i == j else 0.0 for j in range(n)]
        for i in range(n)
    ])


def fake_det(matrix):
    m = matrix.rows
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


numpy_module = types.ModuleType("numpy")
numpy_module.ndarray = FakeMatrix
numpy_module.eye = fake_eye
numpy_module.linalg = types.SimpleNamespace(det=fake_det)
numpy_module.random = types.SimpleNamespace(RandomState=FakeRandomState)
sys.modules["numpy"] = numpy_module


class FakeAtoms:
    def __init__(self, symbols, positions, cell, pbc=True):
        self.symbols = symbols
        self.positions = positions
        self.cell = cell.copy() if isinstance(cell, FakeMatrix) else FakeMatrix(cell)
        self.pbc = pbc
        self.info = {}
        self.calc = None

    def copy(self):
        other = FakeAtoms(
            self.symbols,
            self.positions.copy(),
            self.cell.copy(),
            self.pbc,
        )
        other.info = dict(self.info)
        other.calc = self.calc
        return other

    def __len__(self):
        return len(self.positions)

    def set_cell(self, cell, scale_atoms=False):  # noqa: ARG002
        self.cell = cell.copy() if isinstance(cell, FakeMatrix) else FakeMatrix(cell)

    def get_volume(self):
        return float(abs(fake_det(self.cell)))


ase_module = types.ModuleType("ase")
ase_module.Atom = object
ase_module.Atoms = FakeAtoms
ase_module.units = types.SimpleNamespace(fs=1.0)
sys.modules["ase"] = ase_module

ase_calculators = types.ModuleType("ase.calculators")
ase_calculators.__path__ = []
sys.modules["ase.calculators"] = ase_calculators


class FakeLennardJones:
    pass


ase_lj_module = types.ModuleType("ase.calculators.lj")
ase_lj_module.LennardJones = FakeLennardJones
sys.modules["ase.calculators.lj"] = ase_lj_module


class FakeLangevin:
    def __init__(self, atoms, timestep, temperature_K, friction):
        self.atoms = atoms
        self.timestep = timestep
        self.temperature_K = temperature_K
        self.friction = friction
        self._callbacks = []

    def attach(self, callback, interval):
        self._callbacks.append((callback, interval))

    def run(self, steps):
        for callback, interval in self._callbacks:
            for _ in range(int(steps // interval)):
                callback()


ase_md_module = types.ModuleType("ase.md")
ase_md_module.Langevin = FakeLangevin
sys.modules["ase.md"] = ase_md_module

ase_veldist_module = types.ModuleType("ase.md.velocitydistribution")
ase_veldist_module.MaxwellBoltzmannDistribution = lambda *args, **kwargs: None
ase_veldist_module.Stationary = lambda *args, **kwargs: None
ase_veldist_module.ZeroRotation = lambda *args, **kwargs: None
sys.modules["ase.md.velocitydistribution"] = ase_veldist_module

ase_io_module = types.ModuleType("ase.io")
ase_io_module.write = lambda *args, **kwargs: None
sys.modules["ase.io"] = ase_io_module

common_package = types.ModuleType("common")
common_package.__path__ = []
sys.modules["common"] = common_package

structure_identity_module = types.ModuleType("common.structure_identity")
structure_identity_module.annotate_structure_hashes = lambda *args, **kwargs: None
sys.modules["common.structure_identity"] = structure_identity_module
common_package.structure_identity = structure_identity_module

spec = importlib.util.spec_from_file_location(
    "structure_generation_under_test",
    SRC / "modules" / "generate" / "generators" / "structure_generation.py",
)
structure_generation = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(structure_generation)
PerturbationEngine = structure_generation.PerturbationEngine


class ElasticStressGenerationTests(unittest.TestCase):
    def make_base(self) -> FakeAtoms:
        atoms = FakeAtoms(
            "Si2",
            positions=[[0.0, 0.0, 0.0], [1.25, 1.25, 1.25]],
            cell=fake_eye(3) * 3.0,
            pbc=True,
        )
        atoms.info.update({
            "seed_id": "seed_000001",
            "configurational_type": "test_base",
            "crystal_structure": "diamond",
        })
        return atoms

    def test_elastic_stress_set_creates_all_modes_for_each_amplitude(self) -> None:
        base = self.make_base()
        engine = PerturbationEngine(
            target_n_atoms=2,
            elastic_strain_amplitudes=[-0.01, 0.01],
        )

        structures = engine._elastic_stress_set(base, base)

        self.assertEqual(len(structures), 18)
        modes = {atoms.info["elastic_mode"] for atoms in structures}
        self.assertEqual(
            modes,
            {
                "normal_xx",
                "normal_yy",
                "normal_zz",
                "coupled_xy",
                "coupled_xz",
                "coupled_yz",
                "shear_xy",
                "shear_xz",
                "shear_yz",
            },
        )
        amplitudes = {atoms.info["strain_amplitude"] for atoms in structures}
        self.assertEqual(amplitudes, {-0.01, 0.01})

    def test_elastic_stress_metadata_is_inherited(self) -> None:
        base = self.make_base()
        engine = PerturbationEngine(
            target_n_atoms=2,
            elastic_strain_amplitudes=[0.01],
        )

        structure = engine._elastic_stress_set(base, base)[0]

        self.assertEqual(structure.info["perturbation_type"], "elastic_stress")
        self.assertEqual(structure.info["seed_id"], "seed_000001")
        self.assertEqual(structure.info["configurational_type"], "test_base")
        self.assertEqual(structure.info["crystal_structure"], "diamond")
        self.assertEqual(len(structure.info["strain_matrix"]), 9)

    def test_elastic_stress_disabled_returns_no_structures(self) -> None:
        base = self.make_base()
        engine = PerturbationEngine(
            target_n_atoms=2,
            elastic_stress_enabled=False,
            elastic_strain_amplitudes=[0.01],
        )

        self.assertEqual(engine._elastic_stress_set(base, base), [])

    def test_mode_cells_are_modified_as_expected(self) -> None:
        base = self.make_base()
        engine = PerturbationEngine(
            target_n_atoms=2,
            elastic_strain_amplitudes=[0.01],
        )
        structures = {
            atoms.info["elastic_mode"]: atoms
            for atoms in engine._elastic_stress_set(base, base)
        }

        normal = structures["normal_xx"]
        self.assertAlmostEqual(normal.cell[0, 0], 3.03)
        self.assertAlmostEqual(normal.cell[1, 1], 3.0)
        self.assertAlmostEqual(normal.cell[2, 2], 3.0)

        coupled = structures["coupled_xy"]
        self.assertAlmostEqual(coupled.get_volume(), base.get_volume(), places=10)

        shear = structures["shear_xy"]
        self.assertNotEqual(float(shear.cell[0, 1]), 0.0)
        self.assertNotEqual(float(shear.cell[1, 0]), 0.0)

    def test_rattle_std_defaults_expand_sampling_range(self) -> None:
        engine = PerturbationEngine(target_n_atoms=2)

        self.assertAlmostEqual(engine.rattle_std_min, 0.03)
        self.assertAlmostEqual(engine.rattle_std_max, 0.03)

        sampled = engine._sample_rattle_stds(3)

        self.assertEqual(sampled, [0.03, 0.03, 0.03])

    def test_rattle_std_sampling_uses_configured_range(self) -> None:
        engine = PerturbationEngine(
            target_n_atoms=2,
            rattle_std=0.03,
            rattle_std_min=0.01,
            rattle_std_max=0.07,
        )

        sampled = engine._sample_rattle_stds(2)

        self.assertEqual(sampled, [0.01, 0.07])

    def test_rattle_std_sampling_steps_across_range(self) -> None:
        engine = PerturbationEngine(
            target_n_atoms=2,
            rattle_std=0.03,
            rattle_std_min=0.01,
            rattle_std_max=0.07,
        )

        sampled = engine._sample_rattle_stds(4)

        self.assertEqual(sampled, [0.01, 0.03, 0.05, 0.07])

    def test_liquid_snapshots_are_tagged_as_perturbations(self) -> None:
        base = self.make_base()
        engine = PerturbationEngine(
            target_n_atoms=2,
            liquid_enabled=True,
            liquid_temperature_k=2500.0,
            liquid_timestep_fs=1.5,
            liquid_equilibration_steps=2,
            liquid_steps_between_snapshots=3,
            liquid_friction=0.05,
        )

        structures = engine._liquid_snapshots(base, base, n_configurations=2, n_snapshots=2)

        self.assertEqual(len(structures), 4)
        self.assertEqual(
            [atoms.info["liquid_configuration_index"] for atoms in structures],
            [0, 0, 1, 1],
        )
        self.assertEqual(
            [atoms.info["liquid_snapshot_index"] for atoms in structures],
            [0, 1, 0, 1],
        )
        for atoms in structures:
            self.assertEqual(atoms.info["perturbation_type"], "liquid")
            self.assertEqual(atoms.info["configurational_type"], "test_base")
            self.assertEqual(atoms.info["liquid_temperature_k"], 2500.0)
            self.assertEqual(atoms.info["liquid_timestep_fs"], 1.5)


if __name__ == "__main__":
    unittest.main()
