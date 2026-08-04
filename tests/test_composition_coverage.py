import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UTILITY = ROOT / "utilities" / "plot_composition_coverage.py"


def load_module():
    if "ase" not in sys.modules:
        ase_module = types.ModuleType("ase")
        ase_module.__path__ = []
        sys.modules["ase"] = ase_module
    ase_io = types.ModuleType("ase.io")
    ase_io.read = lambda *args, **kwargs: []
    sys.modules["ase.io"] = ase_io

    spec = importlib.util.spec_from_file_location(
        "testpkg.utilities.plot_composition_coverage",
        UTILITY,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULE = load_module()


class CompositionCoverageTests(unittest.TestCase):
    def composition(self, index, fractions):
        return MODULE.StructureComposition(
            structure_index=index,
            formula="".join(sorted(fractions)),
            total_atoms=100,
            unique_elements=tuple(sorted(fractions)),
            element_counts={element: int(value * 100) for element, value in fractions.items()},
            element_fractions=dict(fractions),
        )

    def test_binary_projection_uses_subset_normalization(self):
        compositions = [
            self.composition(0, {"W": 0.75, "Y": 0.25}),
            self.composition(1, {"Cr": 0.20, "W": 0.60, "Y": 0.20}),
            self.composition(2, {"W": 1.0}),
        ]

        projections = MODULE._collect_binary_projections(compositions)
        w_y = projections[("W", "Y")]
        fractions = sorted(point.normalized_fraction_b for point in w_y)

        self.assertEqual(len(w_y), 2)
        self.assertAlmostEqual(fractions[0], 0.25)
        self.assertAlmostEqual(fractions[1], 0.25)

    def test_ternary_projection_includes_quaternary_structures(self):
        compositions = [
            self.composition(0, {"Cr": 0.25, "W": 0.50, "Y": 0.25}),
            self.composition(1, {"Cr": 0.20, "W": 0.40, "Y": 0.20, "Zr": 0.20}),
        ]

        projections = MODULE._collect_ternary_projections(compositions)

        self.assertIn(("Cr", "W", "Y"), projections)
        self.assertIn(("Cr", "W", "Zr"), projections)
        self.assertIn(("Cr", "Y", "Zr"), projections)
        self.assertIn(("W", "Y", "Zr"), projections)

        cr_w_y = projections[("Cr", "W", "Y")]
        self.assertEqual(len(cr_w_y), 2)
        self.assertEqual(cr_w_y[0].barycentric, (0.25, 0.5, 0.25))
        self.assertAlmostEqual(cr_w_y[1].barycentric[0], 0.25)
        self.assertAlmostEqual(cr_w_y[1].barycentric[1], 0.5)
        self.assertAlmostEqual(cr_w_y[1].barycentric[2], 0.25)

    def test_uniform_binary_distribution_scores_better_than_clustered(self):
        uniform = [
            MODULE.BinaryProjection(("W", "Y"), idx, value)
            for idx, value in enumerate([0.05, 0.25, 0.45, 0.65, 0.85])
        ]
        clustered = [
            MODULE.BinaryProjection(("W", "Y"), idx, value)
            for idx, value in enumerate([0.49, 0.50, 0.50, 0.51, 0.50])
        ]

        uniform_summary, _ = MODULE._summarize_binary_subset(("W", "Y"), uniform, bins=5)
        clustered_summary, _ = MODULE._summarize_binary_subset(("W", "Y"), clustered, bins=5)

        self.assertGreater(uniform_summary.occupied_bin_fraction, clustered_summary.occupied_bin_fraction)
        self.assertGreater(uniform_summary.normalized_entropy, clustered_summary.normalized_entropy)
        self.assertLess(uniform_summary.max_bin_fraction, clustered_summary.max_bin_fraction)
        self.assertLess(uniform_summary.gini, clustered_summary.gini)

    def test_binary_summary_groups_binary_and_ternary_structures_in_same_bin(self):
        projections = [
            MODULE.BinaryProjection(("W", "Y"), 0, 0.25),
            MODULE.BinaryProjection(("W", "Y"), 1, 0.25),
            MODULE.BinaryProjection(("W", "Y"), 2, 0.75),
        ]

        summary, counts = MODULE._summarize_binary_subset(("W", "Y"), projections, bins=4)

        self.assertEqual(summary.structure_count, 3)
        self.assertEqual(max(counts), 2)
        self.assertEqual(summary.occupied_bins, 2)

    def test_ternary_summary_detects_occupied_bins(self):
        projections = [
            MODULE.TernaryProjection(("Cr", "W", "Y"), 0, (0.25, 0.50, 0.25)),
            MODULE.TernaryProjection(("Cr", "W", "Y"), 1, (0.24, 0.51, 0.25)),
            MODULE.TernaryProjection(("Cr", "W", "Y"), 2, (0.50, 0.25, 0.25)),
        ]

        summary, counts = MODULE._summarize_ternary_subset(("Cr", "W", "Y"), projections, resolution=8)

        self.assertEqual(summary.structure_count, 3)
        self.assertGreaterEqual(summary.occupied_bins, 2)
        self.assertEqual(sum(counts.values()), 3)


if __name__ == "__main__":
    unittest.main()
