import csv
import shutil
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
from ase.io import read as ase_read, write as ase_write  # noqa: E402
from nepflow.domain.identities import ArtifactIdentity, ModelRunIdentity, StructureIdentity  # noqa: E402
from nepflow.domain.models import ModelArtifactMetadata, ModelRunRecord  # noqa: E402
from nepflow.mlip.gpumd import GpumdBackend  # noqa: E402
from nepflow.mlip.simulation import StaticPrediction, StaticPredictionRequest  # noqa: E402
from nepflow.errors import MlipError, ValidationError  # noqa: E402
from nepflow.stages.validation.protocols import (  # noqa: E402
    ValidationCaseSpec,
    ValidationReference,
)


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
        "cell": np.asarray(atoms.cell, dtype=float),
        "pbc": np.asarray(atoms.pbc, dtype=bool),
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


def backend_prediction_side_effect(predictions: list[dict]):
    iterator = iter(predictions)

    def parse(_request, _output_path):
        prediction = next(iterator)
        return StaticPrediction(
            StructureIdentity("fixture-output"),
            ModelRunIdentity("report", "nep-in", "hyperparameters"),
            prediction["atoms_count"],
            prediction["energy"],
            prediction["forces"],
            prediction["virial"],
            True,
            species=tuple(prediction["species"]),
            positions_angstrom=prediction["positions"],
            cell_angstrom=prediction["cell"],
            pbc=tuple(prediction["pbc"]),
            atom_mapping=tuple(range(prediction["atoms_count"])),
        )

    return parse


def report_model() -> ModelRunRecord:
    artifact = ArtifactIdentity.from_bytes("nep-model", b"fixture-report-model")
    return ModelRunRecord(
        ModelRunIdentity("fixture-dataset", "fixture-nep-input", "fixture-hyperparameters"),
        ModelArtifactMetadata(model=artifact, status="completed"),
    )


def report_cases(
    root: Path,
    dft_data: dict,
    *,
    factors: tuple[int, int, int] = (1, 1, 1),
) -> tuple[ModelRunRecord, tuple[ValidationCaseSpec, ...]]:
    model = report_model()
    cases = []
    for ordinal, dft in sorted(dft_data.items()):
        case_dir = root / "validation" / f"struct_{ordinal:04d}"
        reference = ValidationReference(
            structure=StructureIdentity(f"fixture-structure-{ordinal}"),
            species=tuple(dft["species"]),
            positions_angstrom=dft["positions"],
            cell_angstrom=dft["cell"],
            pbc=tuple(dft["pbc"]),
            energy_ev=dft["energy"],
            forces_ev_per_angstrom=dft["forces"],
            virial_ev=dft["virial"],
        )
        cases.append(
            ValidationCaseSpec.create(
                ordinal=ordinal,
                model_run_id=model.model_run_id,
                dataset_id=model.identity.dataset_id,
                reference=reference,
                input_path=case_dir / "model.xyz",
                working_directory=case_dir,
                output_path=case_dir / "out.xyz",
                replicates=factors,
                virial_requested=dft["virial"] is not None,
            )
        )
    return model, tuple(cases)


def write_model_output(root: Path, frame_count: int = 2) -> Path:
    validation_root = root / "validation"
    for struct_idx in range(frame_count):
        struct_dir = validation_root / f"struct_{struct_idx:04d}"
        struct_dir.mkdir(parents=True, exist_ok=True)
        (struct_dir / "run.in").write_text(
            "replicate 1 1 1\nrun 1\n", encoding="utf-8"
        )
        (struct_dir / "out.xyz").write_text(
            ML_FIXTURE.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    return validation_root


def generate_report(root: Path) -> Path:
    dft_data, ml_predictions = comparison_inputs()
    output_path = root / "comparison.csv"
    validation_root = write_model_output(root, frame_count=len(dft_data))
    model, cases = report_cases(root, dft_data)

    with (
        patch.object(analyze_module, "parse_dft_properties", return_value=dft_data),
        patch.object(
            GpumdBackend,
            "parse_prediction",
            side_effect=backend_prediction_side_effect(ml_predictions),
        ),
    ):
        analyze_module.generate_comparison_csv(
            validation_root=validation_root,
            test_xyz_path=None,
            output_csv_path=output_path,
            cases=cases,
            model=model,
        )

    return output_path


def test_paired_reference_fixtures_are_intentionally_different() -> None:
    dft = fixture_properties(DFT_FIXTURE)
    ml = fixture_properties(ML_FIXTURE)

    assert dft["energy"] == -10.5
    assert ml["energy"] == -10.25
    assert not np.allclose(dft["forces"], ml["forces"])
    assert not np.allclose(dft["virial"], ml["virial"])


def test_dft_parser_consumes_fixture_energy_forces_and_virial() -> None:
    parsed = analyze_module.parse_dft_properties(DFT_FIXTURE)
    record = parsed[0]

    assert record["energy"] == -10.5
    np.testing.assert_allclose(record["forces"], [[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    assert record["virial"] is not None


def test_model_parser_exposes_fixture_predictions() -> None:
    artifact = ArtifactIdentity.from_bytes("nep-model", b"fixture-model")
    model = ModelRunRecord(
        ModelRunIdentity("dataset", "nep-input", "hyperparameters"),
        ModelArtifactMetadata(model=artifact, status="completed"),
    )
    request = StaticPredictionRequest(
        StructureIdentity("fixture-structure"),
        model,
        ML_FIXTURE,
        ML_FIXTURE.parent,
        atom_count=2,
        virial_requested=True,
    )
    prediction = GpumdBackend().parse_prediction(request, ML_FIXTURE)
    expected = fixture_properties(ML_FIXTURE)

    assert prediction.atom_count == expected["atoms_count"]
    assert prediction.energy_ev == -10.25
    np.testing.assert_allclose(prediction.forces_ev_per_angstrom, expected["forces"])
    np.testing.assert_allclose(prediction.virial_ev, expected["virial"])


def test_unmocked_extxyz_parser_to_csv_path_preserves_real_predictions() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dft_data, _ = comparison_inputs()
        validation_root = write_model_output(root, frame_count=1)
        model, cases = report_cases(root, {0: dft_data[0]})
        report_path = root / "comparison.csv"
        analyze_module.generate_comparison_csv(
            validation_root=validation_root,
            test_xyz_path=None,
            output_csv_path=report_path,
            cases=cases,
            model=model,
        )
        rows = list(csv.DictReader(report_path.open(newline="", encoding="utf-8")))

    assert len(rows) == 2
    assert float(rows[0]["energy_per_atom_ml"]) == pytest.approx(-5.125)
    assert float(rows[0]["energy_mae"]) == pytest.approx(0.125)
    assert float(rows[0]["force_component_mae"]) == pytest.approx(0.01)
    assert float(rows[0]["virial_mae"]) == pytest.approx(5.18426037271)


def test_replicated_model_output_normalizes_to_reference_metrics() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dft_data, _ = comparison_inputs()
        reference_root = root / "reference"
        reference_struct_dir = reference_root / "validation" / "struct_0000"
        reference_struct_dir.mkdir(parents=True)
        (reference_struct_dir / "run.in").write_text(
            "replicate 1 1 1\nrun 1\n", encoding="utf-8"
        )
        shutil.copy2(ML_FIXTURE, reference_struct_dir / "out.xyz")
        reference_report = reference_root / "comparison.csv"
        reference_model, reference_cases = report_cases(
            reference_root, {0: dft_data[0]}
        )
        analyze_module.generate_comparison_csv(
            validation_root=reference_root / "validation",
            test_xyz_path=None,
            output_csv_path=reference_report,
            cases=reference_cases,
            model=reference_model,
        )
        reference_rows = list(
            csv.DictReader(reference_report.open(newline="", encoding="utf-8"))
        )

        validation_root = root / "validation"
        struct_dir = validation_root / "struct_0000"
        struct_dir.mkdir(parents=True)
        (struct_dir / "run.in").write_text("replicate 2 1 1\nrun 1\n", encoding="utf-8")
        model = ase_read(str(ML_FIXTURE), index=0, format="extxyz")
        repeated = model.repeat((2, 1, 1))
        repeated.calc = None
        repeated.info["energy"] = model.get_potential_energy() * 2
        repeated.info["virial"] = np.asarray(model.info["virial"], dtype=float) * 2
        ase_write(struct_dir / "out.xyz", repeated, format="extxyz")
        report_path = root / "comparison.csv"
        model_record, cases = report_cases(
            root, {0: dft_data[0]}, factors=(2, 1, 1)
        )

        analyze_module.generate_comparison_csv(
            validation_root=validation_root,
            test_xyz_path=None,
            output_csv_path=report_path,
            cases=cases,
            model=model_record,
        )
        rows = list(csv.DictReader(report_path.open(newline="", encoding="utf-8")))

    assert len(rows) == 4
    assert float(rows[0]["energy_per_atom_ml"]) == pytest.approx(-5.125)
    assert float(rows[0]["energy_mae"]) == pytest.approx(float(reference_rows[0]["energy_mae"]))
    assert float(rows[0]["force_component_mae"]) == pytest.approx(
        float(reference_rows[0]["force_component_mae"])
    )
    assert float(rows[0]["virial_mae"]) == pytest.approx(float(reference_rows[0]["virial_mae"]))


def test_displaced_model_frame_cannot_be_paired_with_dft_reference() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        validation_root = write_model_output(root, frame_count=1)
        dft_data, _ = comparison_inputs()
        model_record, cases = report_cases(root, {0: dft_data[0]})
        output = ase_read(str(validation_root / "struct_0000" / "out.xyz"), format="extxyz")
        output.positions[0, 0] += 0.25
        ase_write(validation_root / "struct_0000" / "out.xyz", output, format="extxyz")

        with pytest.raises(MlipError, match="positions|configuration"):
            analyze_module.generate_comparison_csv(
                validation_root=validation_root,
                test_xyz_path=None,
                output_csv_path=root / "comparison.csv",
                cases=cases,
                model=model_record,
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
    assert float(rows[0]["energy_error_per_atom"]) == pytest.approx(0.125)
    assert float(rows[2]["energy_per_atom_ml"]) == -5.0
    assert float(rows[2]["energy_error_per_atom"]) == pytest.approx(0.25)
    assert float(rows[0]["energy_mae"]) == pytest.approx(0.1875)
    assert float(rows[0]["energy_rmse"]) == pytest.approx(np.sqrt(0.0390625))


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
def test_missing_required_prediction_is_an_error(missing_key: str) -> None:
    dft_data, ml_predictions = comparison_inputs()
    # This represents a validation configuration in which the DFT reference
    # labels are present and virial checking is enabled when missing_key is virial.
    ml_predictions[0][missing_key] = None

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        validation_root = write_model_output(root)
        model, cases = report_cases(root, dft_data)
        output_path = Path(tmp) / "comparison.csv"
        with (
            patch.object(
                GpumdBackend,
                "parse_prediction",
                side_effect=backend_prediction_side_effect(ml_predictions),
            ),
        ):
            with pytest.raises(ValidationError):
                analyze_module.generate_comparison_csv(
                    validation_root=validation_root,
                    test_xyz_path=None,
                    output_csv_path=output_path,
                    cases=cases,
                    model=model,
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
