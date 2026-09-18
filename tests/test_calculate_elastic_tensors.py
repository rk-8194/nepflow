import unittest
from pathlib import Path

import numpy as np

from utilities import calculate_elastic_tensors as module


class CalculateElasticTensorsTests(unittest.TestCase):
    def test_strain_matrix_to_voigt_uses_engineering_shear(self) -> None:
        strain = [
            0.0, 0.01, 0.0,
            0.01, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ]

        voigt = module.strain_matrix_to_voigt(strain)

        np.testing.assert_allclose(
            voigt,
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.02]),
            rtol=0.0,
            atol=1e-12,
        )

    def test_fit_elastic_tensor_returns_current_contract_for_isotropic_fixture(self) -> None:
        c11 = 240.0
        c12 = 140.0
        c44 = 80.0
        expected = np.array([
            [c11, c12, c12, 0.0, 0.0, 0.0],
            [c12, c11, c12, 0.0, 0.0, 0.0],
            [c12, c12, c11, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, c44, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, c44, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, c44],
        ])

        strains = [
            np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.01, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0]),
            np.array([0.01, -0.01, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.01, 0.0, -0.01, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.01, -0.01, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.02, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.02, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.02]),
        ]

        records = [
            module.ElasticRecord(
                group_key="seed_000001",
                source_label=f"mode_{idx}",
                structure_hash=f"structure-{idx}",
                reference_hash="reference-000001",
                formula="Si2",
                material_id=None,
                structure_name=None,
                elastic_mode=f"mode_{idx}",
                strain_amplitude=0.01,
                strain_voigt=strain,
                stress_voigt_gpa=expected @ strain,
                outcar_path=Path(f"struct_{idx:04d}/OUTCAR"),
            )
            for idx, strain in enumerate(strains)
        ]

        fitted, raw, mode, offsets = module.fit_elastic_tensor(records)

        np.testing.assert_allclose(fitted, expected, atol=1e-10, rtol=0.0)
        np.testing.assert_allclose(raw, expected, atol=1e-10, rtol=0.0)
        self.assertEqual(mode, "general")
        np.testing.assert_allclose(offsets, np.zeros(6), atol=1e-10, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
