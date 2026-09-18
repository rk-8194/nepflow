import unittest

from utilities import plot_composition_coverage as module


class CompositionCoverageTests(unittest.TestCase):
    def composition(self, index, fractions):
        return module.StructureComposition(
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

        projections = module._collect_binary_projections(compositions)
        w_y = projections[("W", "Y")]
        fractions = sorted(point.normalized_fraction_b for point in w_y)

        self.assertEqual(len(w_y), 2)
        self.assertAlmostEqual(fractions[0], 0.25, places=12)
        self.assertAlmostEqual(fractions[1], 0.25, places=12)

    def test_ternary_projection_includes_quaternary_structures(self):
        compositions = [
            self.composition(0, {"Cr": 0.25, "W": 0.50, "Y": 0.25}),
            self.composition(1, {"Cr": 0.20, "W": 0.40, "Y": 0.20, "Zr": 0.20}),
        ]

        projections = module._collect_ternary_projections(compositions)

        self.assertIn(("Cr", "W", "Y"), projections)
        self.assertIn(("Cr", "W", "Zr"), projections)
        self.assertIn(("Cr", "Y", "Zr"), projections)
        self.assertIn(("W", "Y", "Zr"), projections)

        cr_w_y = projections[("Cr", "W", "Y")]
        self.assertEqual(len(cr_w_y), 2)
        self.assertEqual(cr_w_y[0].barycentric, (0.25, 0.5, 0.25))
        self.assertAlmostEqual(cr_w_y[1].barycentric[0], 0.25, places=12)
        self.assertAlmostEqual(cr_w_y[1].barycentric[1], 0.5, places=12)
        self.assertAlmostEqual(cr_w_y[1].barycentric[2], 0.25, places=12)

    def test_uniform_binary_distribution_scores_better_than_clustered(self):
        uniform = [
            module.BinaryProjection(("W", "Y"), idx, value)
            for idx, value in enumerate([0.05, 0.25, 0.45, 0.65, 0.85])
        ]
        clustered = [
            module.BinaryProjection(("W", "Y"), idx, value)
            for idx, value in enumerate([0.49, 0.50, 0.50, 0.51, 0.50])
        ]

        uniform_summary, _ = module._summarize_binary_subset(("W", "Y"), uniform, bins=5)
        clustered_summary, _ = module._summarize_binary_subset(("W", "Y"), clustered, bins=5)

        self.assertGreater(uniform_summary.occupied_bin_fraction, clustered_summary.occupied_bin_fraction)
        self.assertGreater(uniform_summary.normalized_entropy, clustered_summary.normalized_entropy)
        self.assertLess(uniform_summary.max_bin_fraction, clustered_summary.max_bin_fraction)
        self.assertLess(uniform_summary.gini, clustered_summary.gini)

    def test_binary_summary_groups_binary_and_ternary_structures_in_same_bin(self):
        projections = [
            module.BinaryProjection(("W", "Y"), 0, 0.25),
            module.BinaryProjection(("W", "Y"), 1, 0.25),
            module.BinaryProjection(("W", "Y"), 2, 0.75),
        ]

        summary, counts = module._summarize_binary_subset(("W", "Y"), projections, bins=4)

        self.assertEqual(summary.structure_count, 3)
        self.assertEqual(max(counts), 2)
        self.assertEqual(summary.occupied_bins, 2)

    def test_ternary_summary_detects_occupied_bins(self):
        projections = [
            module.TernaryProjection(("Cr", "W", "Y"), 0, (0.25, 0.50, 0.25)),
            module.TernaryProjection(("Cr", "W", "Y"), 1, (0.24, 0.51, 0.25)),
            module.TernaryProjection(("Cr", "W", "Y"), 2, (0.50, 0.25, 0.25)),
        ]

        summary, counts = module._summarize_ternary_subset(("Cr", "W", "Y"), projections, resolution=8)

        self.assertEqual(summary.structure_count, 3)
        self.assertGreaterEqual(summary.occupied_bins, 2)
        self.assertEqual(sum(counts.values()), 3)


if __name__ == "__main__":
    unittest.main()
