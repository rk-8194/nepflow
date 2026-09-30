import json
import tempfile
import unittest
from configparser import ConfigParser
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")
from ase import Atoms  # noqa: E402
from ase.calculators.singlepoint import SinglePointCalculator  # noqa: E402

from modules.run_vasp import _common as common
from modules.train_nep import prepare as train_prepare
from modules.train_nep import train_nep as train_stage_module


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def make_atoms(
    symbols: str = "Si2",
    *,
    energy: float = -10.5,
    forces: np.ndarray | None = None,
    stress: np.ndarray | None = None,
    energy_available: bool = True,
    forces_available: bool = True,
    has_calculator: bool = True,
    positions: np.ndarray | None = None,
) -> Atoms:
    n_atoms = len(Atoms(symbols))
    if positions is None:
        positions = np.array([[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]][:n_atoms], dtype=float)
        if n_atoms > len(positions):
            positions = np.vstack([positions, np.zeros((n_atoms - len(positions), 3))])
    atoms = Atoms(
        symbols,
        positions=positions,
        cell=np.eye(3) * 3.0,
        pbc=True,
    )
    if has_calculator:
        results = {}
        if energy_available:
            results["energy"] = energy
        if forces_available:
            results["forces"] = (
                np.asarray(forces, dtype=float)
                if forces is not None
                else np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]][:n_atoms])
            )
        if stress is not None:
            results["stress"] = np.asarray(stress, dtype=float)
        if results:
            atoms.calc = SinglePointCalculator(atoms, **results)
    return atoms


def current_incar_hash() -> str:
    incar = train_prepare.inject_incar_defaults("ENCUT = 520\n", ConfigParser())
    return common.hash_incar_text(incar)


class TrainNepPrepareRegistryTests(unittest.TestCase):
    def make_project(self, root: Path) -> Path:
        project_dir = root / "projects" / "project_demo"
        (project_dir / "config" / "vasp").mkdir(parents=True, exist_ok=True)
        (project_dir / "vasp" / "jobs" / "train").mkdir(parents=True, exist_ok=True)
        (project_dir / "config" / "vasp" / "INCAR").write_text("ENCUT = 520\n", encoding="utf-8")
        (project_dir / "config" / "vasp" / "POTCAR_Si").write_bytes(b"Si-potcar-v1")
        return project_dir

    def write_identity_job(
        self,
        project_dir: Path,
        struct_name: str,
        atoms: Atoms,
        reused_from: Path | None = None,
    ) -> Path:
        struct_dir = project_dir / "vasp" / "jobs" / "train" / struct_name
        struct_dir.mkdir(parents=True, exist_ok=True)
        identity = {
            "structure_hash": common.hash_structure(atoms),
            "incar_hash": current_incar_hash(),
            "potcar_hash": common.hash_potcar_bytes(b"Si-potcar-v1"),
        }
        (struct_dir / ".vasp_identity").write_text(json.dumps(identity), encoding="utf-8")
        status = {"status": "completed"}
        if reused_from is not None:
            status = {"status": "reused", "reused_from": str(reused_from.resolve())}
        (struct_dir / ".vasp_status").write_text(json.dumps(status), encoding="utf-8")
        return struct_dir

    @staticmethod
    def fixture_atoms(**kwargs) -> Atoms:
        return make_atoms(**kwargs)

    def parsed_outcar(self, **kwargs) -> Atoms:
        return self.fixture_atoms(**kwargs)

    def test_extracts_energy_from_valid_outcar_fixture(self) -> None:
        outcar = FIXTURES / "outcar" / "valid_outcar"
        with patch.object(train_prepare, "ase_read", return_value=self.parsed_outcar()):
            result = train_prepare._parse_outcar(outcar, self.fixture_atoms())

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["energy"], -10.5, places=12)

    def test_extracts_per_atom_forces_from_valid_outcar_fixture(self) -> None:
        outcar = FIXTURES / "outcar" / "valid_outcar"
        with patch.object(train_prepare, "ase_read", return_value=self.parsed_outcar()):
            result = train_prepare._parse_outcar(outcar, self.fixture_atoms())

        self.assertIsNotNone(result)
        np.testing.assert_allclose(result["forces"], [[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])

    def test_converts_fixture_virial_with_explicit_units_and_order(self) -> None:
        virial = train_prepare.parse_virial_from_outcar(
            FIXTURES / "outcar" / "valid_outcar", volume=27.0
        )

        expected = -np.arange(1.0, 10.0).reshape(3, 3) * 27.0 / 1602.17663
        np.testing.assert_allclose(virial, expected, rtol=0.0, atol=1e-12)

    def test_rejects_incomplete_outcar_fixture_before_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = self.make_project(root)
            atoms = self.fixture_atoms()
            struct_dir = self.write_identity_job(project_dir, "struct_0000", atoms)
            (struct_dir / "OUTCAR").write_text(
                (FIXTURES / "outcar" / "incomplete_outcar").read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            parsed = list(train_prepare._parse_structures([atoms], True, project_dir))

            self.assertEqual(parsed, [])

    def test_rejects_unparsable_outcar_instead_of_falling_back(self) -> None:
        outcar = FIXTURES / "outcar" / "completed_without_stress"
        with patch.object(train_prepare, "ase_read", side_effect=RuntimeError("bad OUTCAR")):
            result = train_prepare._parse_outcar(outcar, self.fixture_atoms())

        self.assertIsNone(result)

    def test_malformed_vasp_identity_is_not_treated_as_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            struct_dir = Path(tmp) / "struct_0000"
            struct_dir.mkdir()
            (struct_dir / ".vasp_identity").write_text("{malformed", encoding="utf-8")

            with self.assertRaises(ValueError):
                train_prepare._read_identity(struct_dir)

    def test_debug_dataset_rejects_unlabelled_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = root / "project_demo"
            project_dir.mkdir()
            atoms = Atoms(
                "Si",
                positions=[[0.0, 0.0, 0.0]],
                cell=np.eye(3) * 3.0,
                pbc=True,
            )

            with self.assertRaisesRegex(ValueError, "missing (an )?(energy|force) label"):
                train_prepare.prepare_dataset(
                    dataset_path=root / "train.xyz",
                    ase_structures=[atoms],
                    is_train=True,
                    project_dir=project_dir,
                    debug=True,
                )

    def test_rejects_missing_energy(self) -> None:
        outcar = FIXTURES / "outcar" / "valid_outcar"
        parsed_atoms = self.fixture_atoms(energy_available=False)
        selected_atoms = self.fixture_atoms(has_calculator=False)
        with patch.object(train_prepare, "ase_read", return_value=parsed_atoms):
            result = train_prepare._parse_outcar(outcar, selected_atoms)

        self.assertIsNone(result)

    def test_rejects_missing_forces(self) -> None:
        outcar = FIXTURES / "outcar" / "valid_outcar"
        parsed_atoms = self.fixture_atoms(forces_available=False)
        with patch.object(train_prepare, "ase_read", return_value=parsed_atoms):
            result = train_prepare._parse_outcar(outcar, self.fixture_atoms())

        self.assertIsNone(result)

    def test_rejects_atom_count_mismatch(self) -> None:
        outcar = FIXTURES / "outcar" / "valid_outcar"
        parsed_atoms = make_atoms("Si")
        with patch.object(train_prepare, "ase_read", return_value=parsed_atoms):
            result = train_prepare._parse_outcar(outcar, self.fixture_atoms())

        self.assertIsNone(result)

    def test_rejects_species_mismatch(self) -> None:
        outcar = FIXTURES / "outcar" / "valid_outcar"
        parsed_atoms = make_atoms("Ge2")
        with patch.object(train_prepare, "ase_read", return_value=parsed_atoms):
            result = train_prepare._parse_outcar(outcar, self.fixture_atoms())

        self.assertIsNone(result)

    def test_rejects_missing_required_virial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = self.make_project(root)
            atoms = self.fixture_atoms()
            struct_dir = self.write_identity_job(project_dir, "struct_0000", atoms)
            outcar = struct_dir / "OUTCAR"
            outcar.write_text(
                (FIXTURES / "outcar" / "completed_without_stress").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            with patch.object(train_prepare, "ase_read", return_value=self.fixture_atoms()):
                count = train_prepare.prepare_dataset(
                    dataset_path=root / "train.xyz",
                    ase_structures=[atoms],
                    is_train=True,
                    project_dir=project_dir,
                    train_virial=True,
                )

        self.assertEqual(count, 0)

    def prepare_dataset_after_ase_parse_failure(self) -> tuple[int, str]:
        """Run the production dataset path with a completed but unparsable OUTCAR."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = self.make_project(root)
            # A parser failure must leave the production dataset empty.
            atoms = self.fixture_atoms(has_calculator=False)
            struct_dir = self.write_identity_job(project_dir, "struct_0000", atoms)
            (struct_dir / "OUTCAR").write_text(
                (FIXTURES / "outcar" / "valid_outcar").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            dataset_path = root / "train.xyz"

            with patch.object(train_prepare, "ase_read", side_effect=RuntimeError("bad OUTCAR")):
                count = train_prepare.prepare_dataset(
                    dataset_path=dataset_path,
                    ase_structures=[atoms],
                    is_train=True,
                    project_dir=project_dir,
                )

            rendered = dataset_path.read_text(encoding="utf-8")

        return count, rendered

    def test_production_dataset_path_rejects_placeholder_energy(self) -> None:
        count, rendered = self.prepare_dataset_after_ase_parse_failure()

        self.assertEqual(count, 0, "an unparsable OUTCAR must not produce an accepted dataset record")
        self.assertNotIn("energy=-1.0000000000", rendered)

    def test_production_dataset_path_rejects_placeholder_forces(self) -> None:
        count, rendered = self.prepare_dataset_after_ase_parse_failure()

        self.assertEqual(count, 0, "an unparsable OUTCAR must not produce an accepted dataset record")
        atom_rows = [line for line in rendered.splitlines() if line.startswith("Si ")]
        self.assertEqual(atom_rows, [])

    def test_reused_output_is_accepted_when_identity_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = self.make_project(root)
            atoms = self.fixture_atoms()
            old_job = root / "projects" / "project_old" / "vasp" / "jobs" / "train" / "struct_0002"
            old_job.mkdir(parents=True)
            old_outcar = old_job / "OUTCAR"
            old_outcar.write_text(
                (FIXTURES / "outcar" / "valid_outcar").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            self.write_identity_job(project_dir, "struct_0000", atoms, reused_from=old_job)

            with patch.object(train_prepare, "ase_read", return_value=self.fixture_atoms()):
                parsed = list(train_prepare._parse_structures([atoms], True, project_dir))

        self.assertEqual(len(parsed), 1)
        self.assertAlmostEqual(parsed[0]["energy"], -10.5, places=12)

    def test_reused_output_is_rejected_when_identity_does_not_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = self.make_project(root)
            selected_atoms = self.fixture_atoms()
            different_atoms = make_atoms(positions=np.array([[0.25, 0.0, 0.0], [1.5, 1.5, 1.5]]))
            old_job = root / "projects" / "project_old" / "vasp" / "jobs" / "train" / "struct_0002"
            old_job.mkdir(parents=True)
            (old_job / "OUTCAR").write_text(
                (FIXTURES / "outcar" / "valid_outcar").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            self.write_identity_job(project_dir, "struct_0000", different_atoms, reused_from=old_job)

            parsed = list(train_prepare._parse_structures([selected_atoms], True, project_dir))

        self.assertEqual(parsed, [])

    def test_resolves_outcar_from_reused_registry_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = self.make_project(root)
            atoms = make_atoms("Si")
            old_job = root / "projects" / "project_old" / "vasp" / "jobs" / "train" / "struct_0002"
            old_job.mkdir(parents=True)
            old_outcar = old_job / "OUTCAR"
            old_outcar.write_text("General timing\n", encoding="utf-8")
            common.upsert_registry_entry(
                root,
                current_incar_hash(),
                common.hash_potcar_bytes(b"Si-potcar-v1"),
                common.hash_structure(atoms),
                {"job_path": str(old_job.resolve())},
            )

            with patch.object(train_prepare, "ase_read", return_value=make_atoms("Si")):
                parsed = list(train_prepare._parse_structures([atoms], True, project_dir))

        self.assertEqual(len(parsed), 1)
        self.assertAlmostEqual(parsed[0]["energy"], -10.5, places=12)

    def test_resolves_by_hash_not_selected_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = self.make_project(root)
            first = make_atoms("Si", positions=np.array([[0.0, 0.0, 0.0]]))
            second = make_atoms("Si", positions=np.array([[0.25, 0.0, 0.0]]))
            struct_dir = self.write_identity_job(project_dir, "struct_0009", second)
            (struct_dir / "OUTCAR").write_text("General timing\n", encoding="utf-8")

            with patch.object(train_prepare, "ase_read", return_value=make_atoms("Si")):
                parsed = list(train_prepare._parse_structures([first, second], True, project_dir))

        self.assertEqual(len(parsed), 1)
        self.assertAlmostEqual(parsed[0]["energy"], -10.5, places=12)

    def test_skips_incomplete_registry_outcar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = self.make_project(root)
            atoms = make_atoms("Si")
            old_job = root / "projects" / "project_old" / "vasp" / "jobs" / "train" / "struct_0002"
            old_job.mkdir(parents=True)
            (old_job / "OUTCAR").write_text("not complete\n", encoding="utf-8")
            common.upsert_registry_entry(
                root,
                current_incar_hash(),
                common.hash_potcar_bytes(b"Si-potcar-v1"),
                common.hash_structure(atoms),
                {"job_path": str(old_job.resolve())},
            )

            parsed = list(train_prepare._parse_structures([atoms], True, project_dir))

        self.assertEqual(parsed, [])

    def test_write_xyz_file_includes_nep_virial_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "train.xyz"
            structures = iter(
                [
                    {
                        "energy": -1.25,
                        "pbc": [True, True, True],
                        "lattice": np.eye(3),
                        "species": ["Si"],
                        "positions": np.array([[0.1, 0.2, 0.3]]),
                        "forces": np.array([[0.4, 0.5, 0.6]]),
                        "virial": np.arange(1.0, 10.0).reshape(3, 3),
                    }
                ]
            )

            count = train_prepare._write_xyz_file(output_path, structures, include_virial=True)
            lines = output_path.read_text(encoding="utf-8").splitlines()

            self.assertEqual(count, 1)
            self.assertEqual(lines[0], "1")
            self.assertIn("energy=-1.2500000000", lines[1])
            self.assertIn('pbc="T T T"', lines[1])
            self.assertIn("Properties=species:S:1:pos:R:3:force:R:3", lines[1])
            self.assertIn(
                'virial="1.0000000000 2.0000000000 3.0000000000 '
                '4.0000000000 5.0000000000 6.0000000000 '
                '7.0000000000 8.0000000000 9.0000000000"',
                lines[1],
            )
            self.assertEqual(
                lines[2].split(),
                ["Si", "0.1000000000", "0.2000000000", "0.3000000000", "0.4000000000", "0.5000000000", "0.6000000000"],
            )

    def test_write_xyz_file_omits_missing_virial_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "test.xyz"
            structures = iter(
                [
                    {
                        "energy": -1.25,
                        "pbc": [True, True, True],
                        "lattice": np.eye(3),
                        "species": ["Si"],
                        "positions": np.array([[0.1, 0.2, 0.3]]),
                        "forces": np.array([[0.4, 0.5, 0.6]]),
                        "virial": None,
                    }
                ]
            )

            train_prepare._write_xyz_file(output_path, structures, include_virial=True)
            lines = output_path.read_text(encoding="utf-8").splitlines()

            self.assertNotIn("virial=", lines[1])
            self.assertIn("Properties=species:S:1:pos:R:3:force:R:3", lines[1])


class TrainNepMetadataTests(unittest.TestCase):
    def make_project(self, root: Path) -> Path:
        project_dir = root / "projects" / "project_demo"
        (project_dir / "config" / "vasp").mkdir(parents=True, exist_ok=True)
        (project_dir / "vasp" / "jobs" / "train").mkdir(parents=True, exist_ok=True)
        (project_dir / "config" / "vasp" / "INCAR").write_text("ENCUT = 520\n", encoding="utf-8")
        (project_dir / "config" / "vasp" / "POTCAR_Si").write_bytes(b"Si-potcar-v1")
        return project_dir

    def run_stage_with_extraction_counts(
        self,
        root: Path,
        *,
        selected_train: int,
        selected_test: int,
        accepted_train: int,
        accepted_test: int,
        allow_partial: bool = True,
    ) -> dict:
        project_dir = self.make_project(root)
        dataset_path = project_dir / "nep" / "datasets" / "dataset_0001"
        dataset_path.mkdir(parents=True, exist_ok=True)
        config = ConfigParser()
        config["composition"] = {"elements": "Si"}
        config["train_nep"] = {
            "train_virial": "false",
            "allow_partial_dataset": str(allow_partial).lower(),
        }
        config["slurm"] = {"enabled": "false"}
        config_path = project_dir / "config" / "demo.ini"
        with config_path.open("w", encoding="utf-8") as handle:
            config.write(handle)
        stage = train_stage_module.TrainNepStage(
            project_name="demo",
            config_file=project_dir / "config" / "demo.ini",
            state_file=project_dir / "state.db",
            project_dir=project_dir,
            debug=False,
        )

        def fake_prepare_dataset(*, is_train, extraction_report, **_kwargs):
            accepted_count = accepted_train if is_train else accepted_test
            split = "train" if is_train else "test"
            extraction_report["accepted_results"] = [
                self.accepted_result(
                    f"{split}-{index}",
                    -1.0 - index,
                    f"{split}-hash-{index}",
                )
                for index in range(accepted_count)
            ]
            return accepted_count

        def fake_generate_nep_config(*_args, **_kwargs):
            (dataset_path / "nep.in").write_text(
                "type 1 Si\ncutoff 6 5\n",
                encoding="utf-8",
            )

        with (
            patch.object(stage, "_load_config", return_value=config),
            patch.object(
                stage,
                "_read_train_test_split",
                return_value=(
                    [object() for _ in range(selected_train)],
                    [object() for _ in range(selected_test)],
                ),
            ),
            patch.object(stage, "_get_or_create_dataset_folder", return_value=dataset_path),
            patch.object(train_stage_module, "prepare_dataset", side_effect=fake_prepare_dataset),
            patch.object(stage, "_generate_nep_config", side_effect=fake_generate_nep_config),
            patch.object(stage, "_create_potential_folder", return_value=dataset_path),
        ):
            stage.run()

        return json.loads((dataset_path / ".dataset").read_text(encoding="utf-8"))

    def test_metadata_counts_match_actual_accepted_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metadata = self.run_stage_with_extraction_counts(
                Path(tmp), selected_train=2, selected_test=1, accepted_train=1, accepted_test=1
            )

        self.assertEqual(metadata["train_structures"], 1)
        self.assertEqual(metadata["test_structures"], 1)
        self.assertEqual(metadata["total_structures"], 2)

    def test_metadata_exposes_exclusion_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metadata = self.run_stage_with_extraction_counts(
                Path(tmp), selected_train=2, selected_test=1, accepted_train=1, accepted_test=1
            )

        self.assertIn("requested_train_structures", metadata)
        self.assertIn("accepted_train_structures", metadata)
        self.assertIn("rejected_train_structures", metadata)
        self.assertIn("exclusion_reasons", metadata)
        self.assertTrue(metadata["exclusion_reasons"])

    def test_rejections_fail_before_manifest_without_partial_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(RuntimeError):
                self.run_stage_with_extraction_counts(
                    root,
                    selected_train=2,
                    selected_test=1,
                    accepted_train=1,
                    accepted_test=1,
                    allow_partial=False,
                )

            manifest = root / "projects" / "project_demo" / "nep" / "datasets" / "dataset_0001" / ".dataset"
            self.assertFalse(manifest.exists())

    @staticmethod
    def accepted_result(
        structure_id: str,
        energy: float,
        source_hash: str,
        source_path: str | None = None,
        virial_scale: float = 1.0,
        force_unit: str = "eV/Angstrom",
        virial_convention: str = "positive_compression",
    ):
        return train_prepare.VaspParseResult(
            structure_id=structure_id,
            calculation_identity=(
                ("incar_hash", "incar-1"),
                ("potcar_hash", "potcar-1"),
                ("structure_hash", structure_id),
            ),
            source_outcar=source_path or f"/tmp/{structure_id}/OUTCAR",
            source_outcar_hash=source_hash,
            status="accepted",
            rejection_reason=None,
            energy_ev=energy,
            forces_ev_per_angstrom=np.array([[0.1, 0.0, 0.0]]),
            virial_ev=np.eye(3) * virial_scale,
            positions_angstrom=np.array([[0.0, 0.0, 0.0]]),
            lattice_angstrom=np.eye(3),
            species=("Si",),
            pbc=(True, True, True),
            force_unit=force_unit,
            virial_convention=virial_convention,
        )

    def test_dataset_id_changes_when_accepted_content_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = self.make_project(Path(tmp))
            stage = train_stage_module.TrainNepStage(
                project_name="demo",
                config_file=project_dir / "config" / "demo.ini",
                state_file=project_dir / "state.db",
                project_dir=project_dir,
                debug=False,
            )
            base_report = {
                "requested_count": 1,
                "accepted_count": 1,
                "rejected_count": 0,
                "rejected_reason_counts": {},
                "accepted_results": [self.accepted_result("calc-1", -1.0, "hash-1")],
            }
            metadata_a = stage._build_dataset_metadata(
                project_dir / "nep" / "datasets" / "dataset_0001",
                base_report,
                {"requested_count": 0, "accepted_count": 0, "rejected_count": 0, "rejected_reason_counts": {}, "accepted_results": []},
                train_virial=True,
                allow_partial=False,
            )
            changed_report = {**base_report, "accepted_results": [self.accepted_result("calc-1", -2.0, "hash-1")]}
            metadata_b = stage._build_dataset_metadata(
                project_dir / "nep" / "datasets" / "dataset_0001",
                changed_report,
                {"requested_count": 0, "accepted_count": 0, "rejected_count": 0, "rejected_reason_counts": {}, "accepted_results": []},
                train_virial=True,
                allow_partial=False,
            )

        self.assertNotEqual(metadata_a["dataset_id"], metadata_b["dataset_id"])

    def test_dataset_id_ignores_storage_path_and_unused_virial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = self.make_project(Path(tmp))
            stage = train_stage_module.TrainNepStage(
                project_name="demo",
                config_file=project_dir / "config" / "demo.ini",
                state_file=project_dir / "state.db",
                project_dir=project_dir,
                debug=False,
            )
            base = self.accepted_result(
                "calc-1", -1.0, "hash-1", source_path="C:/one/OUTCAR", virial_scale=1.0
            )
            moved_and_changed_virial = self.accepted_result(
                "calc-1", -1.0, "hash-1", source_path="D:/two/OUTCAR", virial_scale=2.0
            )
            report_a = {
                "requested_count": 1,
                "accepted_count": 1,
                "rejected_count": 0,
                "rejected_reason_counts": {},
                "accepted_results": [base],
            }
            report_b = {**report_a, "accepted_results": [moved_and_changed_virial]}
            empty_report = {
                "requested_count": 0,
                "accepted_count": 0,
                "rejected_count": 0,
                "rejected_reason_counts": {},
                "accepted_results": [],
            }
            metadata_a = stage._build_dataset_metadata(
                project_dir / "nep" / "datasets" / "dataset_0001",
                report_a,
                empty_report,
                train_virial=False,
                allow_partial=False,
            )
            metadata_b = stage._build_dataset_metadata(
                project_dir / "nep" / "datasets" / "dataset_0001",
                report_b,
                empty_report,
                train_virial=False,
                allow_partial=False,
            )

        self.assertEqual(metadata_a["dataset_id"], metadata_b["dataset_id"])

    def test_dataset_id_includes_virial_only_when_materialized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = self.make_project(Path(tmp))
            stage = train_stage_module.TrainNepStage(
                project_name="demo",
                config_file=project_dir / "config" / "demo.ini",
                state_file=project_dir / "state.db",
                project_dir=project_dir,
                debug=False,
            )
            low_virial = self.accepted_result("calc-1", -1.0, "hash-1", virial_scale=1.0)
            high_virial = self.accepted_result("calc-1", -1.0, "hash-1", virial_scale=2.0)
            report_a = {
                "requested_count": 1,
                "accepted_count": 1,
                "rejected_count": 0,
                "rejected_reason_counts": {},
                "accepted_results": [low_virial],
            }
            report_b = {**report_a, "accepted_results": [high_virial]}
            empty_report = {
                "requested_count": 0,
                "accepted_count": 0,
                "rejected_count": 0,
                "rejected_reason_counts": {},
                "accepted_results": [],
            }
            without_virial_a = stage._build_dataset_metadata(
                project_dir / "nep" / "datasets" / "dataset_0001",
                report_a,
                empty_report,
                train_virial=False,
                allow_partial=False,
            )
            without_virial_b = stage._build_dataset_metadata(
                project_dir / "nep" / "datasets" / "dataset_0001",
                report_b,
                empty_report,
                train_virial=False,
                allow_partial=False,
            )
            with_virial_a = stage._build_dataset_metadata(
                project_dir / "nep" / "datasets" / "dataset_0001",
                report_a,
                empty_report,
                train_virial=True,
                allow_partial=False,
            )
            with_virial_b = stage._build_dataset_metadata(
                project_dir / "nep" / "datasets" / "dataset_0001",
                report_b,
                empty_report,
                train_virial=True,
                allow_partial=False,
            )
            different_units = self.accepted_result(
                "calc-1",
                -1.0,
                "hash-1",
                force_unit="hartree/bohr",
                virial_convention="positive_tension",
            )
            with_different_units = stage._build_dataset_metadata(
                project_dir / "nep" / "datasets" / "dataset_0001",
                {**report_a, "accepted_results": [different_units]},
                empty_report,
                train_virial=True,
                allow_partial=False,
            )

        self.assertEqual(without_virial_a["dataset_id"], without_virial_b["dataset_id"])
        self.assertNotEqual(with_virial_a["dataset_id"], with_virial_b["dataset_id"])
        self.assertNotEqual(without_virial_a["dataset_id"], with_virial_a["dataset_id"])
        self.assertNotEqual(with_virial_a["dataset_id"], with_different_units["dataset_id"])

    def test_metadata_preserves_ordered_identity_hash_and_virial_convention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = self.make_project(Path(tmp))
            stage = train_stage_module.TrainNepStage(
                project_name="demo",
                config_file=project_dir / "config" / "demo.ini",
                state_file=project_dir / "state.db",
                project_dir=project_dir,
                debug=False,
            )
            train_report = {
                "requested_count": 2,
                "accepted_count": 2,
                "rejected_count": 0,
                "rejected_reason_counts": {},
                "accepted_results": [
                    self.accepted_result("calc-1", -1.0, "hash-1"),
                    self.accepted_result("calc-2", -2.0, "hash-2"),
                ],
            }
            empty_report = {
                "requested_count": 0,
                "accepted_count": 0,
                "rejected_count": 0,
                "rejected_reason_counts": {},
                "accepted_results": [],
            }
            metadata = stage._build_dataset_metadata(
                project_dir / "nep" / "datasets" / "dataset_0001",
                train_report,
                empty_report,
                train_virial=True,
                allow_partial=False,
            )

        self.assertEqual(
            [entry["structure_id"] for entry in metadata["accepted_calculation_identities"]],
            ["calc-1", "calc-2"],
        )
        self.assertEqual(metadata["source_output_hashes"], ["hash-1", "hash-2"])
        self.assertTrue(metadata["virial_required"])
        self.assertTrue(metadata["virial_included"])
        self.assertEqual(metadata["units"]["virial"], "eV")
        self.assertEqual(metadata["virial_convention"], "positive_compression")


if __name__ == "__main__":
    unittest.main()
