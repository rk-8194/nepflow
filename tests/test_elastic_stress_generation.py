import unittest
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")

from ase import Atoms
from modules.generate.generators.structure_generation import PerturbationEngine


class _FakeDynamics:
    def __init__(self, *_args, **_kwargs):
        self.callback = None
        self.interval = None

    def attach(self, callback, interval):
        self.callback = callback
        self.interval = interval

    def run(self, steps):
        if self.callback is None:
            return
        for step in range(self.interval, steps + 1, self.interval):
            del step
            self.callback()


class ElasticStressGenerationTests(unittest.TestCase):
    def make_base(self) -> Atoms:
        atoms = Atoms(
            "Si2",
            positions=[[0.0, 0.0, 0.0], [1.25, 1.25, 1.25]],
            cell=np.eye(3) * 3.0,
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
        np.testing.assert_allclose(
            sorted(atoms.info["strain_amplitude"] for atoms in structures),
            [-0.01] * 9 + [0.01] * 9,
            atol=1e-12,
        )

    def test_elastic_stress_metadata_is_inherited(self) -> None:
        base = self.make_base()
        engine = PerturbationEngine(target_n_atoms=2, elastic_strain_amplitudes=[0.01])

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
        engine = PerturbationEngine(target_n_atoms=2, elastic_strain_amplitudes=[0.01])
        structures = {
            atoms.info["elastic_mode"]: atoms
            for atoms in engine._elastic_stress_set(base, base)
        }

        normal = structures["normal_xx"]
        self.assertAlmostEqual(normal.cell[0, 0], 3.03, places=12)
        self.assertAlmostEqual(normal.cell[1, 1], 3.0, places=12)
        self.assertAlmostEqual(normal.cell[2, 2], 3.0, places=12)

        coupled = structures["coupled_xy"]
        self.assertAlmostEqual(coupled.get_volume(), base.get_volume(), places=10)

        shear = structures["shear_xy"]
        self.assertNotEqual(float(shear.cell[0, 1]), 0.0)
        self.assertNotEqual(float(shear.cell[1, 0]), 0.0)

    def test_rattle_std_defaults_expand_sampling_range(self) -> None:
        engine = PerturbationEngine(target_n_atoms=2)

        self.assertAlmostEqual(engine.rattle_std_min, 0.03, places=12)
        self.assertAlmostEqual(engine.rattle_std_max, 0.03, places=12)
        np.testing.assert_allclose(engine._sample_rattle_stds(3), [0.03, 0.03, 0.03], atol=1e-12)

    def test_rattle_std_sampling_uses_configured_range(self) -> None:
        engine = PerturbationEngine(
            target_n_atoms=2,
            rattle_std=0.03,
            rattle_std_min=0.01,
            rattle_std_max=0.07,
        )

        np.testing.assert_allclose(engine._sample_rattle_stds(2), [0.01, 0.07], atol=1e-12)

    def test_rattle_std_sampling_steps_across_range(self) -> None:
        engine = PerturbationEngine(
            target_n_atoms=2,
            rattle_std=0.03,
            rattle_std_min=0.01,
            rattle_std_max=0.07,
        )

        np.testing.assert_allclose(
            engine._sample_rattle_stds(4),
            [0.01, 0.03, 0.05, 0.07],
            atol=1e-12,
        )

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

        with (
            patch("ase.calculators.lj.LennardJones", return_value=object()),
            patch("ase.md.Langevin", _FakeDynamics),
            patch("ase.md.velocitydistribution.MaxwellBoltzmannDistribution"),
            patch("ase.md.velocitydistribution.Stationary"),
            patch("ase.md.velocitydistribution.ZeroRotation"),
        ):
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
