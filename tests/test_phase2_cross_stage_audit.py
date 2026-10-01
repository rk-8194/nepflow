import csv
import json
import shutil
import tempfile
from configparser import ConfigParser
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")

from ase.io import read as ase_read  # noqa: E402
from ase.calculators.singlepoint import SinglePointCalculator  # noqa: E402

from common.model_manifest import (  # noqa: E402
    compute_model_run_id,
    create_model_run_manifest,
    update_model_run_status,
    validate_model_run_manifest,
)
from nepflow.domain.identities import calculate_structure_id  # noqa: E402
from nepflow.dft.vasp.inputs import read_identity  # noqa: E402
from nepflow.dft.vasp.outputs import parse_outcar_result  # noqa: E402
from nepflow.io.hashing import sha256_file  # noqa: E402
from modules.train_nep import prepare as train_prepare  # noqa: E402
from modules.train_nep.train_nep import TrainNepStage  # noqa: E402
from modules.validate.analyze import generate_comparison_csv  # noqa: E402
from modules.validate.launcher import (  # noqa: E402
    read_validation_status,
    write_validation_status,
)
from modules.validate.prepare import (  # noqa: E402
    find_latest_potential_and_dataset,
    find_model_run_and_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
OUTCAR_FIXTURE = ROOT / "tests" / "fixtures" / "outcar" / "valid_outcar"
DFT_FIXTURE = ROOT / "tests" / "fixtures" / "structures" / "dft_reference.extxyz.fixture"
ML_FIXTURE = ROOT / "tests" / "fixtures" / "structures" / "gpumd_static_prediction.extxyz.fixture"


def read_dft_fixture_as_vasp_result():
    """Adapt the NEP-facing fixture to ASE's VASP calculator contract."""
    atoms = ase_read(str(DFT_FIXTURE), format="extxyz")
    assert "force" in atoms.arrays
    assert np.asarray(atoms.arrays["force"]).shape == (len(atoms), 3)

    energy = atoms.get_potential_energy()
    forces = np.asarray(atoms.arrays["force"], dtype=float).copy()
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=energy,
        forces=forces,
    )
    return atoms


def audit_config() -> ConfigParser:
    config = ConfigParser()
    config["composition"] = {"elements": "Si"}
    config["train_nep"] = {"charge_mode": "0"}
    return config


def empty_report(requested_count: int, accepted_results: list) -> dict:
    accepted_count = len(accepted_results)
    return {
        "requested_count": requested_count,
        "accepted_count": accepted_count,
        "rejected_count": requested_count - accepted_count,
        "rejected_reason_counts": {},
        "accepted_results": accepted_results,
    }


def test_deterministic_dft_to_validation_identity_trace() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp)
        dataset_path = project_dir / "nep" / "datasets" / "dataset_0001"
        dataset_path.mkdir(parents=True)

        # Selected physical structure -> VASP calculation identity -> exact OUTCAR hash.
        vasp_job = project_dir / "vasp" / "jobs" / "train" / "struct_0000"
        vasp_job.mkdir(parents=True)
        outcar_path = vasp_job / "OUTCAR"
        shutil.copy2(OUTCAR_FIXTURE, outcar_path)
        selected = read_dft_fixture_as_vasp_result()
        calculation_identity = {
            "structure_hash": calculate_structure_id(selected),
            "incar_hash": "incar-hash-v1",
            "potcar_hash": "potcar-hash-v1",
        }
        (vasp_job / ".vasp_identity").write_text(
            json.dumps(calculation_identity), encoding="utf-8"
        )

        accepted = parse_outcar_result(
            outcar_path,
            selected,
            require_virial=True,
            calculation_identity=calculation_identity,
            reader=lambda _path: read_dft_fixture_as_vasp_result(),
            identity_reader=read_identity,
        )

        assert accepted.accepted, accepted.rejection_reason
        assert accepted.structure_id == calculation_identity["structure_hash"]
        assert accepted.calculation_identity == tuple(sorted(calculation_identity.items()))
        assert accepted.source_outcar_hash == sha256_file(outcar_path)

        # Accepted records -> content-derived dataset manifest.
        train_report = {
            "requested_count": 2,
            "accepted_count": 1,
            "rejected_count": 1,
            "rejected_reason_counts": {"completed_outcar_missing": 1},
            "accepted_results": [accepted],
        }
        test_report = empty_report(1, [accepted])
        stage = TrainNepStage(
            project_name="demo",
            config_file=project_dir / "config" / "demo.ini",
            state_file=project_dir / "state.db",
            project_dir=project_dir,
            debug=False,
        )
        metadata = stage._build_dataset_metadata(
            dataset_path,
            train_report,
            test_report,
            train_virial=True,
            allow_partial=True,
        )
        stage._write_dataset_metadata(dataset_path, metadata)
        shutil.copy2(DFT_FIXTURE, dataset_path / "train.xyz")
        shutil.copy2(DFT_FIXTURE, dataset_path / "test.xyz")

        persisted_dataset = json.loads((dataset_path / ".dataset").read_text(encoding="utf-8"))
        assert persisted_dataset["dataset_id"] == metadata["dataset_id"]
        assert persisted_dataset["requested_train_structures"] == 2
        assert persisted_dataset["accepted_train_structures"] == 1
        assert persisted_dataset["rejected_train_structures"] == 1
        assert persisted_dataset["rejection_reason_counts"] == {"completed_outcar_missing": 1}
        assert persisted_dataset["accepted_calculation_identities"][0]["structure_id"] == accepted.structure_id
        assert persisted_dataset["source_output_hashes"] == [accepted.source_outcar_hash] * 2
        assert persisted_dataset["label_schema"]["virial"] is True
        assert persisted_dataset["units"]["energy"] == "eV"
        assert persisted_dataset["virial_convention"] == "positive_compression"

        # Rendered NEP input -> immutable hyperparameter identity.
        config = audit_config()
        hyperparameters = stage._get_nep_hyperparameters(config)
        stage._generate_nep_config(config, dataset_path, hyperparameters)
        nep_in = dataset_path / "nep.in"
        nep_in_hash = sha256_file(nep_in)
        hyperparameters_hash = hyperparameters.identity_hash()
        variant_config = audit_config()
        variant_config["train_nep"]["charge_mode"] = "1"
        assert (
            stage._get_nep_hyperparameters(variant_config).identity_hash()
            != hyperparameters_hash
        )

        # Model manifest binds the exact copied input and completed artifact.
        potential_path = project_dir / "nep" / "potentials" / "potential_0001"
        potential_path.mkdir(parents=True)
        shutil.copy2(nep_in, potential_path / "nep.in")
        (potential_path / "nep.txt").write_text(
            "version 4\ntype 1 Si\ncutoff 6 5 112 60\n", encoding="utf-8"
        )
        model_manifest = create_model_run_manifest(
            potential_path=potential_path,
            dataset_path=dataset_path,
            dataset_id=metadata["dataset_id"],
            nep_in_path=potential_path / "nep.in",
            hyperparameters_hash=hyperparameters_hash,
        )
        completed_manifest = update_model_run_status(potential_path, "completed")
        validated_manifest = validate_model_run_manifest(
            potential_path / "model_run_manifest.json",
            expected_model_run_id=model_manifest["model_run_id"],
        )
        assert completed_manifest["model_run_id"] == validated_manifest["model_run_id"]
        assert validated_manifest["dataset_id"] == metadata["dataset_id"]
        assert validated_manifest["nep_in_sha256"] == sha256_file(potential_path / "nep.in")
        assert validated_manifest["potential_artifact_sha256"] == sha256_file(potential_path / "nep.txt")
        assert validated_manifest["nep_in_sha256"] == nep_in_hash

        resolved_potential, resolved_dataset = find_model_run_and_dataset(
            project_dir, validated_manifest["model_run_id"]
        )
        assert resolved_potential == potential_path.resolve()
        assert resolved_dataset == dataset_path.resolve()
        with pytest.raises(ValueError, match="model_run_id"):
            find_latest_potential_and_dataset(project_dir)

        # Storage relocation does not change the scientific dataset/model identities.
        relocated_result = replace(accepted, source_outcar=str(project_dir / "moved" / "OUTCAR"))
        relocated_metadata = stage._build_dataset_metadata(
            project_dir / "moved" / "dataset_0001",
            {**train_report, "accepted_results": [relocated_result]},
            {**test_report, "accepted_results": [relocated_result]},
            train_virial=True,
            allow_partial=True,
        )
        assert relocated_metadata["dataset_id"] == metadata["dataset_id"]
        relocated_nep = project_dir / "moved" / "nep.in"
        relocated_nep.parent.mkdir(parents=True)
        shutil.copy2(potential_path / "nep.in", relocated_nep)
        assert compute_model_run_id(
            dataset_id=metadata["dataset_id"],
            nep_in_sha256=sha256_file(relocated_nep),
            hyperparameters_hash=hyperparameters_hash,
        ) == validated_manifest["model_run_id"]

        # Exact model/dataset association is persisted in validation status.
        validation_root = potential_path / "validation"
        struct_dir = validation_root / "struct_0000"
        struct_dir.mkdir(parents=True)
        (struct_dir / "run.in").write_text("replicate 1 1 1\nrun 1\n", encoding="utf-8")
        shutil.copy2(ML_FIXTURE, struct_dir / "out.xyz")
        write_validation_status(
            project_dir,
            model_run_id=validated_manifest["model_run_id"],
            potential_path=str(potential_path),
            dataset_path=str(dataset_path),
            dataset_name=dataset_path.name,
            preparation_state={
                "validation_root": str(validation_root),
                "struct_count": 1,
                "struct_folders": [{"name": "struct_0000", "path": str(struct_dir)}],
            },
            validation_complete=True,
            analysis_complete=True,
        )
        validation_status = read_validation_status(project_dir)
        assert validation_status["model_run_id"] == validated_manifest["model_run_id"]
        assert Path(validation_status["dataset_path"]).resolve() == dataset_path.resolve()

        # Actual serialized DFT/ML output produces deliberately non-zero metrics.
        report_path = project_dir / "reports" / "comparison.csv"
        generate_comparison_csv(validation_root, dataset_path / "test.xyz", report_path)
        rows = list(csv.DictReader(report_path.open(newline="", encoding="utf-8")))
        assert rows
        assert float(rows[0]["energy_error_per_atom"]) == pytest.approx(0.125)
        assert float(rows[0]["force_component_mae"]) > 0.0
        assert "virial_mae" in rows[0]
