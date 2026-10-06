import pickle
import unittest
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")
pytest.importorskip("hiphive")
from ase import Atoms  # noqa: E402

from nepflow.domain.identities import calculate_structure_id  # noqa: E402
from nepflow.stages.generation.perturbations.coordinator import (  # noqa: E402
    PerturbationCoordinator,
    execute_perturbation_task,
)
from nepflow.stages.generation.perturbations.defects import vacancies  # noqa: E402
from nepflow.stages.generation.perturbations.displacements import rattled  # noqa: E402
from nepflow.stages.generation.perturbations.models import (  # noqa: E402
    PerturbationCounts,
    PerturbationSettings,
    PerturbationTask,
    derive_child_seed,
)
from nepflow.stages.generation.perturbations.provenance import (  # noqa: E402
    annotate_generation_provenance,
)


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

    @staticmethod
    def settings(**kwargs) -> PerturbationSettings:
        values = {"target_n_atoms": 16}
        values.update(kwargs)
        return PerturbationSettings(**values)

    def test_rattling_failure_is_not_replaced_by_gaussian_output(self) -> None:
        base = self.make_base()
        kwargs = {
            "target_n_atoms": len(base),
            "random_seed": 21,
            "rattle_std": 0.03,
        }

        with patch(
            "hiphive.structure_generation.generate_mc_rattled_structures",
            side_effect=RuntimeError("primary rattling unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Gaussian substitution is disabled"):
                rattled(
                    base,
                    base,
                    3,
                    GenerationReproducibilityTests.settings(**kwargs),
                    21,
                    annotate_generation_provenance,
                )

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
                rattled.positions += rng.normal(0.0, rattle_std, rattled.positions.shape)
                outputs.append(rattled)
            return outputs

        with patch(
            "hiphive.structure_generation.generate_mc_rattled_structures",
            side_effect=fake_primary_rattling,
        ):
            first = rattled(
                base,
                base,
                1,
                self.settings(random_seed=21),
                21,
                annotate_generation_provenance,
            )
            second = rattled(
                base,
                base,
                1,
                self.settings(random_seed=21),
                21,
                annotate_generation_provenance,
            )

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
            rattled(
                base, base, 1, self.settings(random_seed=21), 21, annotate_generation_provenance
            )
            rattled(
                base, base, 1, self.settings(random_seed=22), 22, annotate_generation_provenance
            )

        base_id = calculate_structure_id(base)
        self.assertEqual(
            received_seeds,
            [
                derive_child_seed(base_id, 21, "rattled", 0),
                derive_child_seed(base_id, 22, "rattled", 0),
            ],
        )

    def test_rattle_slots_receive_distinct_child_seeds(self) -> None:
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
            outputs = rattled(
                base,
                base,
                3,
                self.settings(random_seed=21),
                21,
                annotate_generation_provenance,
            )

        expected = [
            derive_child_seed(calculate_structure_id(base), 21, "rattled", index)
            for index in range(3)
        ]
        self.assertEqual(received_seeds, expected)
        self.assertEqual([item.info["random_seed"] for item in outputs], expected)
        self.assertEqual(len(set(received_seeds)), 3)

    def test_unrelated_family_does_not_change_rattled_output(self) -> None:
        base = self.make_base()

        def fake_primary_rattling(atoms, n_structures, rattle_std, d_min, **kwargs):
            del d_min
            rng = np.random.RandomState(kwargs["seed"])
            outputs = []
            for _ in range(n_structures):
                rattled_atoms = atoms.copy()
                rattled_atoms.positions += rng.normal(
                    0.0, rattle_std, rattled_atoms.positions.shape
                )
                outputs.append(rattled_atoms)
            return outputs

        settings = self.settings(
            random_seed=1234,
            n_volume_points=0,
            elastic_stress_enabled=False,
            rattle_std=0.03,
            vacancy_range=(0.25, 0.25),
        )
        task_kwargs = {
            "base": base,
            "base_structure_id": calculate_structure_id(base),
            "settings": settings,
            "seed": 5678,
        }

        with patch(
            "hiphive.structure_generation.generate_mc_rattled_structures",
            side_effect=fake_primary_rattling,
        ):
            without_vacancies = execute_perturbation_task(
                PerturbationTask(
                    **task_kwargs,
                    counts=PerturbationCounts(
                        n_rattled=1,
                        n_vacancies=0,
                        n_interstitials=0,
                    ),
                )
            )
            with_vacancies = execute_perturbation_task(
                PerturbationTask(
                    **task_kwargs,
                    counts=PerturbationCounts(
                        n_rattled=1,
                        n_vacancies=1,
                        n_interstitials=0,
                    ),
                )
            )

        first_rattled = [
            item
            for item in without_vacancies.candidates
            if item.info["perturbation_type"] == "rattled"
        ]
        second_rattled = [
            item
            for item in with_vacancies.candidates
            if item.info["perturbation_type"] == "rattled"
        ]
        self.assert_structures_equal(first_rattled, second_rattled)

    def test_stochastic_vacancy_choices_repeat_with_same_seed(self) -> None:
        base = self.make_base()
        kwargs = {
            "target_n_atoms": len(base),
            "random_seed": 31,
            "vacancy_range": (0.25, 0.25),
        }

        first = vacancies(
            base,
            base,
            3,
            self.settings(**kwargs),
            np.random.RandomState(31),
            annotate_generation_provenance,
            seed=31,
        )
        second = vacancies(
            base,
            base,
            3,
            self.settings(**kwargs),
            np.random.RandomState(31),
            annotate_generation_provenance,
            seed=31,
        )

        self.assert_structures_equal(first, second)

    def test_different_seeds_can_change_stochastic_vacancy_choices(self) -> None:
        base = self.make_base()
        first = vacancies(
            base,
            base,
            1,
            self.settings(random_seed=31, vacancy_range=(0.25, 0.25)),
            np.random.RandomState(31),
            annotate_generation_provenance,
            seed=31,
        )
        second = vacancies(
            base,
            base,
            1,
            self.settings(random_seed=32, vacancy_range=(0.25, 0.25)),
            np.random.RandomState(32),
            annotate_generation_provenance,
            seed=32,
        )

        self.assertFalse(
            np.array_equal(first[0].numbers, second[0].numbers)
            and np.allclose(first[0].positions, second[0].positions)
        )

    def test_worker_engine_parameters_record_random_seed(self) -> None:
        settings = self.settings(random_seed=1234)

        self.assertEqual(settings.random_seed, 1234)

    def test_serial_and_parallel_task_results_keep_base_order(self) -> None:
        first = self.make_base()
        second = self.make_base()
        second.info["seed_id"] = "seed_000008"
        second.info["source"] = "reproducibility-fixture-second"
        settings = self.settings(
            random_seed=1234,
            n_volume_points=0,
            elastic_stress_enabled=False,
        )
        counts = PerturbationCounts(n_rattled=0, n_vacancies=0, n_interstitials=0)

        serial = PerturbationCoordinator(settings=settings).generate_candidates(
            [first, second], counts=counts, n_workers=1
        )
        parallel = PerturbationCoordinator(settings=settings).generate_candidates(
            [first, second], counts=counts, n_workers=2
        )

        self.assert_structures_equal(serial, parallel)
        self.assertEqual(
            [item.info["seed_id"] for item in serial],
            ["seed_000007", "seed_000008"],
        )

    def test_worker_derived_seed_is_recorded_on_stochastic_candidate(self) -> None:
        base = self.make_base()
        child_seed = 9876
        received_seeds = []

        def fake_primary_rattling(atoms, n_structures, rattle_std, d_min, **kwargs):
            del d_min, rattle_std
            received_seeds.append(kwargs["seed"])
            return [atoms.copy() for _ in range(n_structures)]

        with patch(
            "hiphive.structure_generation.generate_mc_rattled_structures",
            side_effect=fake_primary_rattling,
        ):
            result = execute_perturbation_task(
                PerturbationTask(
                    base=base,
                    base_structure_id=calculate_structure_id(base),
                    settings=self.settings(
                        random_seed=1234,
                        n_volume_points=0,
                        elastic_stress_enabled=False,
                    ),
                    counts=PerturbationCounts(
                        n_rattled=1,
                        n_vacancies=0,
                        n_interstitials=0,
                    ),
                    seed=child_seed,
                )
            )
            results = list(result.candidates)
            restored = pickle.loads(pickle.dumps(result))

        rattled = [atoms for atoms in results if atoms.info["perturbation_type"] == "rattled"]
        self.assertEqual(len(rattled), 1)
        expected_seed = derive_child_seed(calculate_structure_id(base), child_seed, "rattled", 0)
        self.assertEqual(received_seeds, [expected_seed])
        self.assertEqual(rattled[0].info["random_seed"], expected_seed)
        self.assertNotEqual(rattled[0].info["random_seed"], child_seed)
        self.assertNotEqual(rattled[0].info["random_seed"], 1234)
        self.assertEqual(
            [calculate_structure_id(item) for item in restored.candidates],
            [calculate_structure_id(item) for item in result.candidates],
        )

    def test_stochastic_candidate_records_parent_and_seed_provenance(self) -> None:
        base = self.make_base()
        candidate = vacancies(
            base,
            base,
            1,
            self.settings(random_seed=1234, vacancy_range=(0.25, 0.25)),
            np.random.RandomState(1234),
            annotate_generation_provenance,
            seed=1234,
        )[0]

        self.assertEqual(candidate.info["seed_id"], base.info["seed_id"])
        self.assertEqual(candidate.info["source"], base.info["source"])
        self.assertEqual(candidate.info["perturbation_type"], "vacancy")
        self.assertEqual(candidate.info["n_vacancies"], 4)
        self.assertEqual(
            candidate.info["random_seed"],
            derive_child_seed(calculate_structure_id(base), 1234, "vacancy", 0),
        )


if __name__ == "__main__":
    unittest.main()
