import importlib.util
import sys
import unittest
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
    raise unittest.SkipTest("numpy is required for elastic tensor tests") from exc


ROOT = Path(__file__).resolve().parents[1]
UTILITY = ROOT / "utilities" / "calculate_elastic_tensors.py"

spec = importlib.util.spec_from_file_location("calculate_elastic_tensors", UTILITY)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
try:
    spec.loader.exec_module(module)
except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
    raise unittest.SkipTest("elastic tensor utility dependencies are unavailable") from exc


class CalculateElasticTensorsTests(unittest.TestCase):
    def test_strain_matrix_to_voigt_uses_engineering_shear(self) -> None:
        strain = [
            0.0, 0.01, 0.0,
            0.01, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ]

        voigt = module.strain_matrix_to_voigt(strain)

        np.testing.assert_allclose(voigt, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.02]))

    def test_fit_elastic_tensor_recovers_known_isotropic_tensor(self) -> None:
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

        records = []
        for idx, strain in enumerate(strains):
            records.append(
                module.ElasticRecord(
                    group_key="seed_000001",
                    structure_index=idx,
                    elastic_mode=f"mode_{idx}",
                    strain_amplitude=0.01,
                    strain_voigt=strain,
                    stress_voigt_gpa=expected @ strain,
                    outcar_path=Path(f"struct_{idx:04d}/OUTCAR"),
                    metadata={"seed_id": "seed_000001"},
                )
            )

        fitted = module.fit_elastic_tensor(records)

        np.testing.assert_allclose(fitted, expected, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
