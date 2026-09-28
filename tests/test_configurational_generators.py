import sys
import unittest
from collections import Counter
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")
from ase import Atoms  # noqa: E402

from modules.generate.generators import configurational as configurational_module  # noqa: E402


P0_12_XFAIL_REASON = (
    "Phase 1 blocker P0-12: per-composition quotas must cover requested crystals"
)
P0_13_XFAIL_REASON = (
    "Phase 1 blocker P0-13: public SQS generation must fail fast on unavailable paths"
)


def make_atoms(symbols: str = "Si4") -> Atoms:
    atoms = Atoms(
        symbols,
        positions=np.arange(3 * len(Atoms(symbols))).reshape((-1, 3)).astype(float),
        cell=np.eye(3) * 3.0,
        pbc=True,
    )
    return atoms


class ConfigurationalGeneratorTests(unittest.TestCase):
    composition = {"Si": 0.5, "Ge": 0.5}
    crystal_structures = ["bcc", "fcc"]

    def assert_per_composition_quota(
        self, results: list[Atoms], expected_total: int
    ) -> None:
        counts = Counter(atoms.info["crystal_structure"] for atoms in results)
        # n_structures is the total quota for one composition.  This test
        # contract allocates an even split when the quota covers both crystals.
        self.assertEqual(len(results), expected_total)
        self.assertEqual(
            counts,
            Counter({"bcc": expected_total // 2, "fcc": expected_total // 2}),
        )

    @pytest.mark.xfail(strict=True, reason=P0_12_XFAIL_REASON)
    def test_random_solution_quota_is_per_composition_and_deterministic(self) -> None:
        first = configurational_module.RandomSolidSolutionGenerator(
            n_structures=4,
            random_seed=7,
        ).generate(self.composition, self.crystal_structures, target_n_atoms=8)
        second = configurational_module.RandomSolidSolutionGenerator(
            n_structures=4,
            random_seed=7,
        ).generate(self.composition, self.crystal_structures, target_n_atoms=8)

        self.assert_per_composition_quota(first, expected_total=4)
        self.assertEqual(
            [atoms.get_chemical_symbols() for atoms in first],
            [atoms.get_chemical_symbols() for atoms in second],
        )

    def test_random_solution_different_seeds_change_assignment(self) -> None:
        first = configurational_module.RandomSolidSolutionGenerator(
            n_structures=1,
            random_seed=7,
        ).generate(self.composition, ["bcc"], target_n_atoms=16)
        second = configurational_module.RandomSolidSolutionGenerator(
            n_structures=1,
            random_seed=8,
        ).generate(self.composition, ["bcc"], target_n_atoms=16)

        self.assertNotEqual(
            first[0].get_chemical_symbols(),
            second[0].get_chemical_symbols(),
        )

    @pytest.mark.xfail(strict=True, reason=P0_12_XFAIL_REASON)
    def test_segregated_quota_is_per_composition_and_deterministic(self) -> None:
        first = configurational_module.SegregatedGenerator(
            n_structures=4,
            random_seed=7,
        ).generate(self.composition, self.crystal_structures, target_n_atoms=8)
        second = configurational_module.SegregatedGenerator(
            n_structures=4,
            random_seed=7,
        ).generate(self.composition, self.crystal_structures, target_n_atoms=8)

        self.assert_per_composition_quota(first, expected_total=4)
        self.assertEqual(
            [atoms.get_chemical_symbols() for atoms in first],
            [atoms.get_chemical_symbols() for atoms in second],
        )

    @pytest.mark.xfail(strict=True, reason=P0_12_XFAIL_REASON)
    def test_sqs_quota_is_per_composition_and_deterministic(self) -> None:
        fake_icet = ModuleType("icet")
        fake_icet_tools = ModuleType("icet.tools")
        fake_structure_generation = ModuleType("icet.tools.structure_generation")
        fake_icet.ClusterSpace = lambda *args, **kwargs: object()
        fake_structure_generation.generate_sqs_from_supercells = (
            lambda **kwargs: make_atoms("Si4")
        )
        fake_structure_generation._get_sqs_cluster_vector = lambda *args, **kwargs: None

        with patch.dict(
            sys.modules,
            {
                "icet": fake_icet,
                "icet.tools": fake_icet_tools,
                "icet.tools.structure_generation": fake_structure_generation,
            },
        ):
            first = configurational_module.SQSGenerator(
                n_structures=4,
                random_seed=7,
            ).generate(self.composition, self.crystal_structures, target_n_atoms=8)
            second = configurational_module.SQSGenerator(
                n_structures=4,
                random_seed=7,
            ).generate(self.composition, self.crystal_structures, target_n_atoms=8)

        self.assert_per_composition_quota(first, expected_total=4)
        self.assertEqual(
            [atoms.get_chemical_symbols() for atoms in first],
            [atoms.get_chemical_symbols() for atoms in second],
        )

    @pytest.mark.xfail(strict=True, reason=P0_13_XFAIL_REASON)
    def test_sqs_generate_fails_fast_when_primary_and_fallback_are_unavailable(
        self,
    ) -> None:
        fake_icet = ModuleType("icet")
        fake_icet_tools = ModuleType("icet.tools")
        fake_structure_generation = ModuleType("icet.tools.structure_generation")
        fake_icet.ClusterSpace = lambda *args, **kwargs: object()

        def fail_primary(**kwargs):
            raise RuntimeError("primary SQS generation unavailable")

        fake_structure_generation.generate_sqs_from_supercells = fail_primary
        fake_structure_generation._get_sqs_cluster_vector = lambda *args, **kwargs: None

        # Deliberately omit generate_target_structure: no equivalent fallback
        # is available, so the public boundary must report the failure.
        with patch.dict(
            sys.modules,
            {
                "icet": fake_icet,
                "icet.tools": fake_icet_tools,
                "icet.tools.structure_generation": fake_structure_generation,
            },
        ):
            generator = configurational_module.SQSGenerator(
                n_structures=1,
                random_seed=7,
            )
            with self.assertRaises(RuntimeError):
                generator.generate(self.composition, ["bcc"], target_n_atoms=8)

    @pytest.mark.xfail(strict=True, reason=P0_13_XFAIL_REASON)
    def test_sqs_generate_fails_fast_when_icet_is_unavailable(self) -> None:
        generator = configurational_module.SQSGenerator(n_structures=1, random_seed=7)

        with patch.dict(sys.modules, {"icet": None}):
            with self.assertRaises((ImportError, RuntimeError)):
                generator.generate(self.composition, ["bcc"], target_n_atoms=8)


if __name__ == "__main__":
    unittest.main()
