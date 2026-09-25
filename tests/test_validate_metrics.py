import csv
import tempfile
from configparser import ConfigParser
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")

from modules.validate import analyze as analyze_module  # noqa: E402
from modules.validate import validate as validate_stage_module  # noqa: E402
from modules.validate.validate import ValidateStage  # noqa: E402
from ase.io import read as ase_read  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "structures"
DFT_FIXTURE = FIXTURES / "dft_reference.extxyz.fixture"
ML_FIXTURE = FIXTURES / "gpumd_static_prediction.extxyz.fixture"


def fixture_properties(path: Path) -> dict:
    atoms = ase_read(str(path), index=0, format="extxyz")
    return {
        "atoms_count": len(atoms),
        "energy": float(atoms.get_potential_energy()),
        "forces": np.asarray(atoms.arrays["force"], dtype=float),
        "virial": np.asarray(atoms.info["virial"], dtype=float),
        "positions": np.asarray(atoms.positions, dtype=float),
        "species": list(atoms.get_chemical_symbols()),
    }


def comparison_inputs() -> tuple[dict, dict]:
    dft = fixture_properties(DFT_FIXTURE)
    ml = fixture_properties(ML_FIXTURE)
    dft_second = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in dft.items()}
    ml_second = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in ml.items()}
    # Two distinct, hand-checkable per-atom energy errors: 0.125 and 0.25 eV.
    dft_second["energy"] = -10.5
    ml_second["energy"] = -10.0
    return {0: dft, 1: dft_second}, [ml, ml_second]


def write_model_output(root: Path, frame_count: int = 2) -> Path:
    validation_root = root / "validation"
    for struct_idx in range(frame_count):
        struct_dir = validation_root / f"struct_{struct_idx:04d}"
        struct_dir.mkdir(parents=True, exist_ok=True)
        (struct_dir / "out.xyz").write_text(
            ML_FIXTURE.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    return validation_root


def generate_report(root: Path) -> Path:
    dft_data, ml_predictions = comparison_inputs()
    output_path = root / "comparison.csv"
    validation_root = write_model_output(root, frame_count=len(dft_data))

    with (
        patch.object(analyze_module, "parse_dft_properties", return_value=dft_data),
        patch.object(
            analyze_module,
            "parse_gpumd_output",
            side_effect=[
                (prediction["atoms_count"], prediction["positions"], prediction)
                for prediction in ml_predictions
            ],
        ),
    ):
        analyze_module.generate_comparison_csv(
            validation_root=validation_root,
            test_xyz_path=DFT_FIXTURE,
            output_csv_path=output_path,
        )

    return output_path


def test_paired_reference_fixtures_are_intentionally_different() -> None:
    dft = fixture_properties(DFT_FIXTURE)
    ml = fixture_properties(ML_FIXTURE)

    assert dft["energy"] == -10.5
    assert ml["energy"] == -10.25
    assert not np.allclose(dft["forces"], ml["forces"])
    assert not np.allclose(dft["virial"], ml["virial"])


@pytest.mark.xfail(
    strict=True,
    reason="Phase 2 blocker P0-4: DFT parser must consume paired energy and force labels",
)
def test_dft_parser_consumes_fixture_energy_forces_and_virial() -> None:
    parsed = analyze_module.parse_dft_properties(DFT_FIXTURE)
    record = parsed[0]

    assert record["energy"] == -10.5
    np.testing.assert_allclose(record["forces"], [[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    assert record["virial"] is not None


@pytest.mark.xfail(
    strict=True,
    reason="Phase 2 blocker P0-4: model parser must expose genuine energy, force, and virial predictions",
)
def test_model_parser_exposes_fixture_predictions() -> None:
    atoms_count, _, predictions = analyze_module.parse_gpumd_output(ML_FIXTURE)
    expected = fixture_properties(ML_FIXTURE)

    assert atoms_count == expected["atoms_count"]
    assert predictions["energy"] == -10.25
    np.testing.assert_allclose(predictions["forces"], expected["forces"])
    np.testing.assert_allclose(predictions["virial"], expected["virial"])


@pytest.mark.xfail(
    strict=True,
    reason="Phase 2 blocker P0-4: comparison output must use distinct model energy and explicit eV/atom error",
)
def test_comparison_reports_model_energy_and_hand_checkable_per_atom_error() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        report_path = generate_report(Path(tmp))
        rows = list(csv.DictReader(report_path.open(newline="", encoding="utf-8")))

    assert rows
    dft_per_atom = float(rows[0]["energy_per_atom_dft"])
    ml_per_atom = float(rows[0]["energy_per_atom_ml"])
    assert dft_per_atom == -5.25
    assert ml_per_atom == -5.125
    assert float(rows[2]["energy_per_atom_ml"]) == -5.0
    assert float(rows[0]["energy_mae"]) == pytest.approx(0.1875)
    assert float(rows[0]["energy_rmse"]) == pytest.approx(np.sqrt(0.0390625))


@pytest.mark.xfail(
    strict=True,
    reason="Phase 2 blocker P0-4: comparison output must preserve predicted force components and magnitudes",
)
def test_comparison_reports_force_components_and_hand_checkable_metrics() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        report_path = generate_report(Path(tmp))
        rows = list(csv.DictReader(report_path.open(newline="", encoding="utf-8")))

    ml = fixture_properties(ML_FIXTURE)
    predicted_components = np.asarray(
        [[float(row["force_x_ml"]), float(row["force_y_ml"]), float(row["force_z_ml"])] for row in rows]
    )
    np.testing.assert_allclose(predicted_components, np.vstack([ml["forces"], ml["forces"]]))
    assert float(rows[0]["force_component_mae"]) == pytest.approx(0.01)
    assert float(rows[0]["force_component_rmse"]) == pytest.approx(0.01290994449)
    assert float(rows[0]["force_magnitude_mae"]) == pytest.approx(0.01989668415)
    assert float(rows[0]["force_magnitude_rmse"]) == pytest.approx(0.01990345882)


@pytest.mark.xfail(
    strict=True,
    reason="Phase 2 blocker P0-4: comparison output must preserve predicted virial values and convention",
)
def test_comparison_reports_virial_components_and_errors() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        report_path = generate_report(Path(tmp))
        rows = list(csv.DictReader(report_path.open(newline="", encoding="utf-8")))

    ml = fixture_properties(ML_FIXTURE)
    predicted = np.asarray(
        [[float(row[f"virial_{axis}_ml"]) for axis in ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz")] for row in rows]
    )
    np.testing.assert_allclose(predicted[0].reshape(3, 3), ml["virial"])
    assert float(rows[0]["virial_mae"]) == pytest.approx(5.18426037271)
    assert float(rows[0]["virial_rmse"]) == pytest.approx(5.81117973296)


@pytest.mark.parametrize("missing_key", ["energy", "forces", "virial"])
@pytest.mark.xfail(
    strict=True,
    reason="Phase 2 blocker P0-4: missing required validation labels must raise instead of defaulting or writing partial reports",
)
def test_missing_required_prediction_is_an_error(missing_key: str) -> None:
    dft_data, ml_predictions = comparison_inputs()
    # This represents a validation configuration in which the DFT reference
    # labels are present and virial checking is enabled when missing_key is virial.
    ml_predictions[0][missing_key] = None

    with tempfile.TemporaryDirectory() as tmp:
        validation_root = write_model_output(Path(tmp))
        output_path = Path(tmp) / "comparison.csv"
        with (
            patch.object(analyze_module, "parse_dft_properties", return_value=dft_data),
            patch.object(
                analyze_module,
                "parse_gpumd_output",
                side_effect=[
                    (prediction["atoms_count"], prediction["positions"], prediction)
                    for prediction in ml_predictions
                ],
            ),
        ):
            with pytest.raises((ValueError, RuntimeError)):
                analyze_module.generate_comparison_csv(
                    validation_root=validation_root,
                    test_xyz_path=DFT_FIXTURE,
                    output_csv_path=output_path,
                )


@pytest.mark.xfail(
    strict=True,
    reason="Phase 2 blocker P0-4: plotting must reject missing model predictions rather than substituting zeros or DFT values",
)
def test_plotting_cannot_substitute_missing_model_predictions() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "comparison.csv"
        csv_path.write_text(
            "struct_id,atom_id,species,energy_per_atom_dft,energy_per_atom_ml,force_magnitude_dft,force_magnitude_ml\n"
            "0,0,Si,-5.25,,0.1,\n"
            "0,1,Si,-5.20,,0.2,\n",
            encoding="utf-8",
        )

        with pytest.raises((ValueError, RuntimeError)):
            analyze_module.plot_comparison_results(
                csv_path=csv_path,
                output_dir=Path(tmp) / "reports",
                dataset_name="dataset_0001",
                potential_name="potential_0001",
            )


@pytest.mark.xfail(
    strict=True,
    reason="Phase 2 blocker P0-4: validation cannot persist analysis_complete without required metric output",
)
def test_validation_analysis_cannot_complete_without_required_metrics() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        stage = ValidateStage(
            project_name="demo",
            config_file=root / "config" / "demo.yaml",
            state_file=root / "state.db",
            project_dir=root,
            debug=False,
        )
        status = {
            "status": "running",
            "potential_path": str(root / "gpumd" / "dataset_0001" / "potential_0001"),
            "dataset_name": "dataset_0001",
            "preparation_state": {"validation_root": str(root / "validation")},
            "analysis_complete": False,
        }
        (root / "gpumd").mkdir(parents=True, exist_ok=True)
        with patch.object(validate_stage_module, "generate_comparison_csv", return_value=None):
            with patch.object(validate_stage_module, "plot_comparison_results", return_value=None):
                analysis_error = None
                try:
                    stage._run_analysis(ConfigParser(), status)
                except Exception as exc:
                    analysis_error = exc

        from modules.validate.launcher import read_validation_status

        assert analysis_error is not None or read_validation_status(root).get("analysis_complete") is not True


def test_runtime_is_not_an_accuracy_metric_in_current_report_contract() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        report_path = generate_report(Path(tmp))
        fields = next(csv.reader(report_path.open(newline="", encoding="utf-8")))

    assert not any("runtime" in field.lower() or "performance" in field.lower() for field in fields)
