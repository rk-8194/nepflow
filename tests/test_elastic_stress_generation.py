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


class _SeededFakeDynamics:
    """MD boundary whose state is reproducible only when a seed is supplied."""

    def __init__(self, atoms, *_args, **kwargs):
        self.atoms = atoms
        self.callback = None
        self.interval = None
        seed = kwargs.get("random_seed")
        self.rng = np.random.default_rng(seed)

    def attach(self, callback, interval):
        self.callback = callback
        self.interval = interval

    def run(self, steps):
        for step in range(steps):
            self.atoms.positions += self.rng.normal(0.0, 1.0e-3, self.atoms.positions.shape)
            if self.callback is not None and (step + 1) % self.interval == 0:
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

    def test_normal_strain_uses_explicit_diagonal_tensor_convention(self) -> None:
        amplitude = 0.02

        expected = np.eye(3)
        expected[0, 0] = 1.0 + amplitude

        np.testing.assert_allclose(
            PerturbationEngine._normal_strain_matrix(amplitude, 0),
            expected,
            atol=1e-12,
        )

    def test_coupled_strain_preserves_volume_with_compensating_axis(self) -> None:
        amplitude = 0.02

        expected = np.diag(
            [1.0 + amplitude, 1.0 - amplitude, 1.0 / (1.0 - amplitude**2)]
        )

        np.testing.assert_allclose(
            PerturbationEngine._coupled_strain_matrix(amplitude, 0, 1),
            expected,
            atol=1e-12,
        )
        self.assertAlmostEqual(np.linalg.det(expected), 1.0, places=12)

    def test_shear_amplitude_is_tensor_shear_not_engineering_shear(self) -> None:
        amplitude = 0.03
        matrix = PerturbationEngine._shear_strain_matrix(amplitude, 0, 1)

        # Tensor-shear convention: epsilon_xy = epsilon_yx = amplitude.
        self.assertAlmostEqual(matrix[0, 1], amplitude, places=12)
        self.assertAlmostEqual(matrix[1, 0], amplitude, places=12)
        self.assertNotAlmostEqual(matrix[0, 1], 2.0 * amplitude, places=12)

    def test_normal_and_shear_positive_negative_amplitudes_are_symmetric(self) -> None:
        amplitude = 0.025
        identity = np.eye(3)

        normal_positive = PerturbationEngine._normal_strain_matrix(amplitude, 2)
        normal_negative = PerturbationEngine._normal_strain_matrix(-amplitude, 2)
        shear_positive = PerturbationEngine._shear_strain_matrix(amplitude, 1, 2)
        shear_negative = PerturbationEngine._shear_strain_matrix(-amplitude, 1, 2)

        np.testing.assert_allclose(
            normal_positive - identity,
            -(normal_negative - identity),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            shear_positive - identity,
            -(shear_negative - identity),
            atol=1e-12,
        )

    def test_elastic_generation_is_independent_of_random_state(self) -> None:
        base = self.make_base()
        first = PerturbationEngine(
            target_n_atoms=2,
            random_seed=7,
            elastic_strain_amplitudes=[-0.01, 0.01],
        )._elastic_stress_set(base, base)
        second = PerturbationEngine(
            target_n_atoms=2,
            random_seed=991,
            elastic_strain_amplitudes=[-0.01, 0.01],
        )._elastic_stress_set(base, base)

        self.assertEqual(len(first), len(second))
        for first_atoms, second_atoms in zip(first, second):
            np.testing.assert_allclose(first_atoms.cell.array, second_atoms.cell.array)
            self.assertEqual(first_atoms.info, second_atoms.info)

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

    @pytest.mark.xfail(
        strict=True,
        reason="Phase 1 blocker P0-14: liquid MD must receive an explicit reproducibility seed",
    )
    def test_liquid_snapshots_repeat_with_same_seed(self) -> None:
        base = self.make_base()
        kwargs = {
            "target_n_atoms": 2,
            "random_seed": 17,
            "liquid_enabled": True,
            "liquid_equilibration_steps": 1,
            "liquid_steps_between_snapshots": 1,
        }

        with (
            patch("ase.calculators.lj.LennardJones", return_value=object()),
            patch("ase.md.Langevin", _SeededFakeDynamics),
            patch("ase.md.velocitydistribution.MaxwellBoltzmannDistribution"),
            patch("ase.md.velocitydistribution.Stationary"),
            patch("ase.md.velocitydistribution.ZeroRotation"),
        ):
            first = PerturbationEngine(**kwargs)._liquid_snapshots(
                base,
                base,
                n_configurations=1,
                n_snapshots=1,
            )
            second = PerturbationEngine(**kwargs)._liquid_snapshots(
                base,
                base,
                n_configurations=1,
                n_snapshots=1,
            )

        np.testing.assert_allclose(first[0].positions, second[0].positions)


if __name__ == "__main__":
    unittest.main()
