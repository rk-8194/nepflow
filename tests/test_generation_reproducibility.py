import unittest
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")
pytest.importorskip("hiphive")
from ase import Atoms  # noqa: E402

from modules.generate.generators import structure_generation as structure_generation_module  # noqa: E402
from modules.generate.generators.structure_generation import PerturbationEngine  # noqa: E402


class GenerationReproducibilityTests(unittest.TestCase):
    @staticmethod
    def make_base() -> Atoms:
        atoms = Atoms(
            "Si16",
            positions=np.arange(48, dtype=float).reshape((-1, 3)),
            cell=np.eye(3) * 20.0,
            pbc=True,
        )
        atoms.info.update(
            {
                "seed_id": "seed_000007",
                "source": "reproducibility-fixture",
                "elements": ["Si"],
                "configurational_type": "test_base",
            }
        )
        return atoms

    @staticmethod
    def assert_structures_equal(first: list[Atoms], second: list[Atoms]) -> None:
        if len(first) != len(second):
            raise AssertionError(f"different structure counts: {len(first)} != {len(second)}")
        for first_atoms, second_atoms in zip(first, second):
            np.testing.assert_array_equal(first_atoms.numbers, second_atoms.numbers)
            np.testing.assert_allclose(first_atoms.positions, second_atoms.positions)
            np.testing.assert_allclose(first_atoms.cell.array, second_atoms.cell.array)
            self_info = dict(first_atoms.info)
            other_info = dict(second_atoms.info)
            if self_info != other_info:
                raise AssertionError(f"different structure metadata: {self_info} != {other_info}")

    def test_rattled_outputs_repeat_with_same_seed_without_global_rng(self) -> None:
        base = self.make_base()
        kwargs = {
            "target_n_atoms": len(base),
            "random_seed": 21,
            "rattle_std": 0.03,
        }

        with patch(
            "hiphive.structure_generation.generate_mc_rattled_structures",
            side_effect=RuntimeError("use deterministic Gaussian fallback"),
        ):
            first = PerturbationEngine(**kwargs)._rattled(base, base, n=3)
            second = PerturbationEngine(**kwargs)._rattled(base, base, n=3)

        self.assert_structures_equal(first, second)

    def test_primary_rattling_path_repeats_with_same_seed(self) -> None:
        base = self.make_base()
        ambient_rng = np.random.RandomState(90210)

        def fake_primary_rattling(atoms, n_structures, rattle_std, d_min, **kwargs):
            del d_min
            supplied_rng = kwargs.get("rng")
            if supplied_rng is None:
                supplied_rng = kwargs.get("random_state")
            supplied_seed = kwargs.get("random_seed")
            if supplied_seed is None:
                supplied_seed = kwargs.get("seed")

            if supplied_rng is None and supplied_seed is None:
                # Model hiphive's uncontrolled ambient state.  The test must
                # not pass by forcing the Gaussian fallback.
                rng = ambient_rng
            elif supplied_rng is not None:
                rng = supplied_rng
            else:
                rng = np.random.RandomState(supplied_seed)

            outputs = []
            for _ in range(n_structures):
                rattled = atoms.copy()
                rattled.positions += rng.normal(
                    0.0, rattle_std, rattled.positions.shape
                )
                outputs.append(rattled)
            return outputs

        with patch(
            "hiphive.structure_generation.generate_mc_rattled_structures",
            side_effect=fake_primary_rattling,
        ):
            first = PerturbationEngine(
                target_n_atoms=len(base),
                random_seed=21,
            )._rattled(base, base, n=1)
            second = PerturbationEngine(
                target_n_atoms=len(base),
                random_seed=21,
            )._rattled(base, base, n=1)

        self.assert_structures_equal(first, second)

    def test_different_effective_seeds_reach_primary_rattling_boundary(self) -> None:
        base = self.make_base()
        received_seeds = []

        def fake_primary_rattling(atoms, n_structures, rattle_std, d_min, **kwargs):
            del d_min, rattle_std
            received_seeds.append(kwargs["seed"])
            return [atoms.copy() for _ in range(n_structures)]

        with patch(
            "hiphive.structure_generation.generate_mc_rattled_structures",
            side_effect=fake_primary_rattling,
        ):
            PerturbationEngine(
                target_n_atoms=len(base),
                random_seed=21,
            )._rattled(base, base, n=1)
            PerturbationEngine(
                target_n_atoms=len(base),
                random_seed=22,
            )._rattled(base, base, n=1)

        self.assertEqual(received_seeds, [21, 22])

    def test_stochastic_vacancy_choices_repeat_with_same_seed(self) -> None:
        base = self.make_base()
        kwargs = {
            "target_n_atoms": len(base),
            "random_seed": 31,
            "vacancy_range": (0.25, 0.25),
        }

        first = PerturbationEngine(**kwargs)._vacancies(base, base, n=3)
        second = PerturbationEngine(**kwargs)._vacancies(base, base, n=3)

        self.assert_structures_equal(first, second)

    def test_different_seeds_can_change_stochastic_vacancy_choices(self) -> None:
        base = self.make_base()
        first = PerturbationEngine(
            target_n_atoms=len(base),
            random_seed=31,
            vacancy_range=(0.25, 0.25),
        )._vacancies(base, base, n=1)
        second = PerturbationEngine(
            target_n_atoms=len(base),
            random_seed=32,
            vacancy_range=(0.25, 0.25),
        )._vacancies(base, base, n=1)

        self.assertFalse(
            np.array_equal(first[0].numbers, second[0].numbers)
            and np.allclose(first[0].positions, second[0].positions)
        )

    def test_worker_engine_parameters_record_random_seed(self) -> None:
        engine = PerturbationEngine(target_n_atoms=16, random_seed=1234)

        self.assertEqual(engine._engine_params()["random_seed"], 1234)

    def test_worker_derived_seed_is_recorded_on_stochastic_candidate(self) -> None:
        base = self.make_base()
        parent_engine = PerturbationEngine(
            target_n_atoms=len(base),
            random_seed=1234,
            n_volume_points=0,
            elastic_stress_enabled=False,
        )
        child_seed = 9876

        def fake_primary_rattling(atoms, n_structures, rattle_std, d_min, **kwargs):
            del d_min, rattle_std, kwargs
            return [atoms.copy() for _ in range(n_structures)]

        with patch(
            "hiphive.structure_generation.generate_mc_rattled_structures",
            side_effect=fake_primary_rattling,
        ):
            results = structure_generation_module._process_one_base(
                (
                    base,
                    parent_engine._engine_params(),
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    child_seed,
                )
            )

        rattled = [
            atoms for atoms in results if atoms.info["perturbation_type"] == "rattled"
        ]
        self.assertEqual(len(rattled), 1)
        self.assertEqual(rattled[0].info["random_seed"], child_seed)
        self.assertNotEqual(rattled[0].info["random_seed"], parent_engine._random_seed)

    def test_stochastic_candidate_records_parent_and_seed_provenance(self) -> None:
        base = self.make_base()
        candidate = PerturbationEngine(
            target_n_atoms=len(base),
            random_seed=1234,
            vacancy_range=(0.25, 0.25),
        )._vacancies(base, base, n=1)[0]

        self.assertEqual(candidate.info["seed_id"], base.info["seed_id"])
        self.assertEqual(candidate.info["source"], base.info["source"])
        self.assertEqual(candidate.info["perturbation_type"], "vacancy")
        self.assertEqual(candidate.info["n_vacancies"], 4)
        self.assertEqual(candidate.info["random_seed"], 1234)


if __name__ == "__main__":
    unittest.main()
