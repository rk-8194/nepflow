import unittest
from collections import Counter

import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")
from ase import Atoms  # noqa: E402

from nepflow.stages.generation.generators.composition_primitives import (  # noqa: E402
    allocate_crystal_quota,
)
from nepflow.stages.generation.generators.random_solution import (  # noqa: E402
    RandomSolidSolutionGenerator,
)
from nepflow.stages.generation.generators.segregated import SegregatedGenerator  # noqa: E402
from nepflow.stages.generation.generators.sqs import SQSGenerator  # noqa: E402


def make_atoms(symbols: str = "Si4") -> Atoms:
    atoms = Atoms(
        symbols,
        positions=np.arange(3 * len(Atoms(symbols))).reshape((-1, 3)).astype(float),
        cell=np.eye(3) * 3.0,
        pbc=True,
    )
    return atoms


class SQSBackendBoundary:
    """Deterministic fake for the injected external SQS backend boundary."""

    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.fail_on_call = fail_on_call

    def generate(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("primary SQS generation unavailable")
        return make_atoms("Si4")


class ConfigurationalGeneratorTests(unittest.TestCase):
    composition = {"Si": 0.5, "Ge": 0.5}
    crystal_structures = ["bcc", "fcc"]

    def assert_per_composition_quota(self, results: list[Atoms], expected_total: int) -> None:
        counts = Counter(atoms.info["crystal_structure"] for atoms in results)
        # n_structures is the total quota for one composition.  This test
        # contract allocates an even split when the quota covers both crystals.
        self.assertEqual(len(results), expected_total)
        self.assertEqual(
            counts,
            Counter({"bcc": expected_total // 2, "fcc": expected_total // 2}),
        )

    def test_crystal_quota_allocation_balances_in_configured_order(self) -> None:
        self.assertEqual(
            allocate_crystal_quota(5, ["bcc", "fcc"]),
            [("bcc", 3), ("fcc", 2)],
        )

    def test_crystal_quota_allocation_gives_remainder_to_first_crystal(self) -> None:
        self.assertEqual(
            allocate_crystal_quota(1, ["bcc", "fcc"]),
            [("bcc", 1), ("fcc", 0)],
        )

    def test_crystal_quota_allocation_zero_has_no_outputs(self) -> None:
        self.assertEqual(
            allocate_crystal_quota(0, ["bcc", "fcc"]),
            [("bcc", 0), ("fcc", 0)],
        )

    def test_crystal_quota_allocation_rejects_invalid_crystal_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            allocate_crystal_quota(4, ["bcc", "bcc"])
        with self.assertRaisesRegex(ValueError, "at least one"):
            allocate_crystal_quota(1, [])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            allocate_crystal_quota(-1, ["bcc"])

    def test_random_solution_reversed_crystal_order_uses_ordered_remainder(self) -> None:
        results = RandomSolidSolutionGenerator(
            n_structures=3,
            random_seed=7,
        ).generate(self.composition, ["fcc", "bcc"], target_n_atoms=8)

        self.assertEqual(
            [atoms.info["crystal_structure"] for atoms in results],
            ["fcc", "fcc", "bcc"],
        )

    def test_random_solution_quota_is_per_composition_and_deterministic(self) -> None:
        first = RandomSolidSolutionGenerator(
            n_structures=4,
            random_seed=7,
        ).generate(self.composition, self.crystal_structures, target_n_atoms=8)
        second = RandomSolidSolutionGenerator(
            n_structures=4,
            random_seed=7,
        ).generate(self.composition, self.crystal_structures, target_n_atoms=8)

        self.assert_per_composition_quota(first, expected_total=4)
        self.assertEqual(
            [atoms.get_chemical_symbols() for atoms in first],
            [atoms.get_chemical_symbols() for atoms in second],
        )

    def test_random_solution_different_seeds_change_assignment(self) -> None:
        first = RandomSolidSolutionGenerator(
            n_structures=1,
            random_seed=7,
        ).generate(self.composition, ["bcc"], target_n_atoms=16)
        second = RandomSolidSolutionGenerator(
            n_structures=1,
            random_seed=8,
        ).generate(self.composition, ["bcc"], target_n_atoms=16)

        self.assertNotEqual(
            first[0].get_chemical_symbols(),
            second[0].get_chemical_symbols(),
        )

    def test_random_solution_same_seed_repeats_assignment_independently_of_quota(self) -> None:
        first = RandomSolidSolutionGenerator(
            n_structures=2,
            random_seed=17,
        ).generate(self.composition, ["bcc"], target_n_atoms=16)
        second = RandomSolidSolutionGenerator(
            n_structures=2,
            random_seed=17,
        ).generate(self.composition, ["bcc"], target_n_atoms=16)

        self.assertEqual(
            [atoms.get_chemical_symbols() for atoms in first],
            [atoms.get_chemical_symbols() for atoms in second],
        )

    def test_segregated_quota_is_per_composition_and_deterministic(self) -> None:
        first = SegregatedGenerator(
            n_structures=4,
            random_seed=7,
        ).generate(self.composition, self.crystal_structures, target_n_atoms=8)
        second = SegregatedGenerator(
            n_structures=4,
            random_seed=7,
        ).generate(self.composition, self.crystal_structures, target_n_atoms=8)

        self.assert_per_composition_quota(first, expected_total=4)
        self.assertEqual(
            [atoms.get_chemical_symbols() for atoms in first],
            [atoms.get_chemical_symbols() for atoms in second],
        )

    def test_sqs_quota_is_per_composition_and_deterministic(self) -> None:
        first = SQSGenerator(
            n_structures=4,
            random_seed=7,
            backend=SQSBackendBoundary(),
        ).generate(self.composition, self.crystal_structures, target_n_atoms=8)
        second = SQSGenerator(
            n_structures=4,
            random_seed=7,
            backend=SQSBackendBoundary(),
        ).generate(self.composition, self.crystal_structures, target_n_atoms=8)

        self.assert_per_composition_quota(first, expected_total=4)
        self.assertEqual(
            [atoms.get_chemical_symbols() for atoms in first],
            [atoms.get_chemical_symbols() for atoms in second],
        )

    def test_sqs_success_preserves_sqs_provenance(self) -> None:
        results = SQSGenerator(
            n_structures=1,
            random_seed=7,
            backend=SQSBackendBoundary(),
        ).generate(self.composition, ["bcc"], target_n_atoms=8)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].info["configurational_type"], "sqs")
        self.assertEqual(results[0].info["composition"], self.composition)
        self.assertEqual(results[0].info["crystal_structure"], "bcc")
        self.assertEqual(results[0].info["source"], "sqs-Si0.5-Ge0.5-bcc-0")

    def test_sqs_failure_on_later_slot_does_not_return_partial_results(self) -> None:
        backend = SQSBackendBoundary(fail_on_call=2)
        with self.assertRaisesRegex(
            RuntimeError,
            "composition.*crystal structure.*fcc.*primary SQS generation unavailable",
        ):
            SQSGenerator(
                n_structures=2,
                random_seed=7,
                backend=backend,
            ).generate(self.composition, self.crystal_structures, target_n_atoms=8)

        self.assertEqual([call["random_seed"] for call in backend.calls], [7, 8])

    def test_unary_sqs_returns_empty_without_backend_call(self) -> None:
        generator = SQSGenerator(
            n_structures=1,
            random_seed=7,
            backend=SQSBackendBoundary(fail_on_call=1),
        )
        self.assertEqual(generator.generate({"Si": 1.0}, ["bcc"], target_n_atoms=8), [])

    def test_sqs_backend_failure_is_not_replaced_with_an_alternative(
        self,
    ) -> None:
        generator = SQSGenerator(
            n_structures=1,
            random_seed=7,
            backend=SQSBackendBoundary(fail_on_call=1),
        )
        with self.assertRaisesRegex(RuntimeError, "primary SQS generation unavailable"):
            generator.generate(self.composition, ["bcc"], target_n_atoms=8)

    def test_sqs_backend_failure_is_wrapped_at_the_generator_boundary(self) -> None:
        class UnavailableBackend:
            def generate(self, **kwargs):
                del kwargs
                raise ImportError("icet backend unavailable")

        with self.assertRaisesRegex(RuntimeError, "icet backend unavailable"):
            SQSGenerator(backend=UnavailableBackend()).generate(
                self.composition,
                ["bcc"],
                target_n_atoms=8,
            )


if __name__ == "__main__":
    unittest.main()
