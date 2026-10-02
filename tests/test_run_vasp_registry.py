import json
import tempfile
import unittest
from configparser import ConfigParser
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")

from ase import Atoms
from ase.io import write as ase_write
from nepflow.config import load_config, render_default_config
from nepflow.domain.calculations import DftResultArtifact
from nepflow.domain.identities import (
    ArtifactIdentity,
    DftCalculationIdentity,
    StructureIdentity,
    calculate_structure_id,
)
from nepflow.dft.backend import DftInputArtifacts
from nepflow.dft.vasp.inputs import (
    hash_incar_text,
    hash_potcar_bytes,
    inject_incar_defaults,
    read_identity,
)
from nepflow.dft.vasp.registry import (
    VASP_REGISTRY_VERSION,
    completed_jobs_registry_path,
    get_registry_entry,
    read_completed_registry,
    read_status,
    upsert_registry_entry,
    write_status,
)
from nepflow.errors import ArtifactError, StateError
from nepflow.stages.dft import calculation_identities_match, prepare_calculations
from nepflow.stages.dft import DftExecutionRecord
from nepflow.state import StateStore


class RunVaspRegistryTests(unittest.TestCase):
    def make_project(self, root: Path, name: str = "demo") -> Path:
        project_dir = root / "projects" / f"project_{name}"
        (project_dir / "config" / "vasp").mkdir(parents=True, exist_ok=True)
        (project_dir / "structures" / "selected").mkdir(parents=True, exist_ok=True)
        (project_dir / "config" / "vasp" / "INCAR").write_text(
            "ENCUT = 520\nNCORE = 16\nKPAR = 1\n",
            encoding="utf-8",
        )
        (project_dir / "config" / "vasp" / "POTCAR_Si").write_bytes(b"Si-potcar-v1")
        (project_dir / "config" / "project.config").write_text(
            render_default_config(
                "demo",
                {
                    "materialsproject_api_key": "",
                    "elements": "Si",
                    "gas_elements": "",
                    "crystal_structures": "bcc",
                    "target_n_atoms": "1",
                    "scp_address": "",
                },
            ),
            encoding="utf-8",
        )
        return project_dir

    @staticmethod
    def silicon_atoms() -> Atoms:
        return Atoms(
            "Si",
            scaled_positions=[[0.0, 0.0, 0.0]],
            cell=np.eye(3),
            pbc=True,
        )

    def run_prepare_jobs(self, project_dir: Path, atoms: Atoms) -> None:
        selected_dir = project_dir / "structures" / "selected"
        ase_write(selected_dir / "train.xyz", atoms, format="extxyz")
        ase_write(selected_dir / "test.xyz", atoms, format="extxyz")
        config = load_config(
            project_dir / "config" / "project.config",
            project_name="demo",
        )
        with StateStore(project_dir / "state.db") as state_store:
            prepare_calculations(
                config=config,
                project_dir=project_dir,
                project_name="demo",
                state_store=state_store,
                datasets=("train", "test"),
            )

    def test_structure_hash_changes_with_structure(self) -> None:
        first = self.silicon_atoms()
        moved = self.silicon_atoms()
        moved.set_scaled_positions([[0.25, 0.0, 0.0]])
        larger_cell = self.silicon_atoms()
        larger_cell.set_cell([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        self.assertNotEqual(calculate_structure_id(first), calculate_structure_id(moved))
        self.assertNotEqual(calculate_structure_id(first), calculate_structure_id(larger_cell))

    def test_incar_hash_ignores_resource_params(self) -> None:
        original = "ENCUT = 520\nNCORE = 16\nKPAR = 1\nISMEAR = 0\n"
        rewritten = "ENCUT = 520\nNCORE = 64\nKPAR = 4\nISMEAR = 0\n"

        self.assertEqual(hash_incar_text(original), hash_incar_text(rewritten))

    def test_effective_incar_defaults_participate_in_identity(self) -> None:
        template = "ENCUT = 520\n"
        first_config = ConfigParser()
        first_config["vasp"] = {"kspacing": "0.30", "kgamma": ".TRUE."}
        second_config = ConfigParser()
        second_config["vasp"] = {"kspacing": "0.35", "kgamma": ".TRUE."}

        first_effective = inject_incar_defaults(template, first_config)
        second_effective = inject_incar_defaults(template, second_config)

        self.assertNotEqual(
            hash_incar_text(first_effective),
            hash_incar_text(second_effective),
        )
        self.assertNotEqual(
            hash_incar_text(template),
            hash_incar_text(first_effective),
        )

    def test_scheduler_and_resource_provenance_do_not_change_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atoms = self.silicon_atoms()
            incar = inject_incar_defaults(
                "ENCUT = 520\nNCORE = 16\nKPAR = 1\n",
                ConfigParser(),
            )
            identity = (
                hash_incar_text(incar),
                hash_potcar_bytes(b"Si-potcar-v1"),
                calculate_structure_id(atoms),
            )

            for index, attempt_provenance in enumerate(
                (
                    {"slurm_job_id": "1001", "nodes": 1, "gpus": 1, "walltime": "01:00:00"},
                    {"slurm_job_id": "2002", "nodes": 4, "gpus": 8, "walltime": "24:00:00"},
                )
            ):
                project_dir = self.make_project(root, f"demo_{index}")
                completed_job = (
                    root
                    / "projects"
                    / f"project_old_{index}"
                    / "vasp"
                    / "jobs"
                    / "train"
                    / "struct_0007"
                )
                completed_job.mkdir(parents=True)
                (completed_job / "OUTCAR").write_text(
                    "General timing\n",
                    encoding="utf-8",
                )
                upsert_registry_entry(
                    root,
                    identity[0],
                    identity[1],
                    identity[2],
                    {"job_path": str(completed_job.resolve()), **attempt_provenance},
                )

                self.run_prepare_jobs(project_dir, atoms)

                status = json.loads(
                    (
                        project_dir
                        / "vasp"
                        / "jobs"
                        / "train"
                        / "struct_0000"
                        / ".vasp_status"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(status["status"], "reused")

    def test_missing_identity_fields_never_match(self) -> None:
        expected = {
            "structure_hash": "structure-hash",
            "incar_hash": "incar-hash",
            "potcar_hash": "potcar-hash",
        }

        self.assertFalse(
            calculation_identities_match(
                {"structure_hash": "structure-hash", "incar_hash": "incar-hash"},
                expected,
            )
        )

    def test_potcar_hash_changes_when_content_changes(self) -> None:
        self.assertNotEqual(
            hash_potcar_bytes(b"Si-potcar-v1"),
            hash_potcar_bytes(b"Si-potcar-v2"),
        )

    def test_prepare_marks_registry_match_as_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = self.make_project(root)
            atoms = self.silicon_atoms()

            completed_job = root / "projects" / "project_old" / "vasp" / "jobs" / "train" / "struct_0007"
            completed_job.mkdir(parents=True)
            (completed_job / "OUTCAR").write_text("General timing\n", encoding="utf-8")
            incar = inject_incar_defaults(
                "ENCUT = 520\nNCORE = 16\nKPAR = 1\n",
                ConfigParser(),
            )
            upsert_registry_entry(
                root,
                hash_incar_text(incar),
                hash_potcar_bytes(b"Si-potcar-v1"),
                calculate_structure_id(atoms),
                {
                    "job_path": str(completed_job.resolve()),
                    "slurm_job_id": "12345",
                    "nodes": 1,
                    "gpus": 1,
                    "walltime": "01:00:00",
                },
            )

            self.run_prepare_jobs(project_dir, atoms)

            status = json.loads(
                (project_dir / "vasp" / "jobs" / "train" / "struct_0000" / ".vasp_status")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(status["status"], "reused")
            self.assertEqual(Path(status["reused_from"]), completed_job.resolve())
            self.assertNotEqual(status["status"], "pending")

    def test_prepare_does_not_reuse_when_structure_differs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = self.make_project(root)
            atoms = self.silicon_atoms()
            changed = self.silicon_atoms()
            changed.set_scaled_positions([[0.25, 0.0, 0.0]])
            completed_job = root / "projects" / "project_old" / "vasp" / "jobs" / "train" / "struct_0007"
            completed_job.mkdir(parents=True)
            (completed_job / "OUTCAR").write_text("General timing\n", encoding="utf-8")
            incar = inject_incar_defaults(
                "ENCUT = 520\nNCORE = 16\nKPAR = 1\n",
                ConfigParser(),
            )
            upsert_registry_entry(
                root,
                hash_incar_text(incar),
                hash_potcar_bytes(b"Si-potcar-v1"),
                calculate_structure_id(changed),
                {"job_path": str(completed_job.resolve())},
            )

            self.run_prepare_jobs(project_dir, atoms)

            status = json.loads(
                (project_dir / "vasp" / "jobs" / "train" / "struct_0000" / ".vasp_status")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(status["status"], "pending")

    def test_prepare_does_not_reuse_incomplete_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = self.make_project(root)
            atoms = self.silicon_atoms()
            incomplete_job = root / "projects" / "project_old" / "vasp" / "jobs" / "train" / "struct_0007"
            incomplete_job.mkdir(parents=True)
            (incomplete_job / "OUTCAR").write_text("not complete\n", encoding="utf-8")
            incar = inject_incar_defaults(
                "ENCUT = 520\nNCORE = 16\nKPAR = 1\n",
                ConfigParser(),
            )
            upsert_registry_entry(
                root,
                hash_incar_text(incar),
                hash_potcar_bytes(b"Si-potcar-v1"),
                calculate_structure_id(atoms),
                {"job_path": str(incomplete_job.resolve())},
            )

            self.run_prepare_jobs(project_dir, atoms)

            status = json.loads(
                (project_dir / "vasp" / "jobs" / "train" / "struct_0000" / ".vasp_status")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(status["status"], "pending")

    def test_prepare_does_not_reuse_stale_registry_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = self.make_project(root)
            atoms = self.silicon_atoms()
            stale_job = root / "projects" / "project_old" / "vasp" / "jobs" / "train" / "missing"
            incar = inject_incar_defaults(
                "ENCUT = 520\nNCORE = 16\nKPAR = 1\n",
                ConfigParser(),
            )
            upsert_registry_entry(
                root,
                hash_incar_text(incar),
                hash_potcar_bytes(b"Si-potcar-v1"),
                calculate_structure_id(atoms),
                {"job_path": str(stale_job.resolve())},
            )

            self.run_prepare_jobs(project_dir, atoms)

            status = json.loads(
                (project_dir / "vasp" / "jobs" / "train" / "struct_0000" / ".vasp_status")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(status["status"], "pending")

    def test_prepare_does_not_reuse_when_incar_differs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = self.make_project(root)
            (project_dir / "config" / "vasp" / "INCAR").write_text(
                "ENCUT = 600\n",
                encoding="utf-8",
            )
            atoms = self.silicon_atoms()
            completed_job = root / "projects" / "project_old" / "vasp" / "jobs" / "train" / "struct_0007"
            completed_job.mkdir(parents=True)
            (completed_job / "OUTCAR").write_text("General timing\n", encoding="utf-8")
            upsert_registry_entry(
                root,
                hash_incar_text("ENCUT = 520\n"),
                hash_potcar_bytes(b"Si-potcar-v1"),
                calculate_structure_id(atoms),
                {"job_path": str(completed_job.resolve())},
            )

            self.run_prepare_jobs(project_dir, atoms)

            status = json.loads(
                (project_dir / "vasp" / "jobs" / "train" / "struct_0000" / ".vasp_status")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(status["status"], "pending")

    def test_prepare_does_not_reuse_when_potcar_differs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = self.make_project(root)
            (project_dir / "config" / "vasp" / "POTCAR_Si").write_bytes(b"Si-potcar-v2")
            atoms = self.silicon_atoms()
            completed_job = root / "projects" / "project_old" / "vasp" / "jobs" / "train" / "struct_0007"
            completed_job.mkdir(parents=True)
            (completed_job / "OUTCAR").write_text("General timing\n", encoding="utf-8")
            incar = inject_incar_defaults(
                "ENCUT = 520\nNCORE = 16\nKPAR = 1\n",
                ConfigParser(),
            )
            upsert_registry_entry(
                root,
                hash_incar_text(incar),
                hash_potcar_bytes(b"Si-potcar-v1"),
                calculate_structure_id(atoms),
                {"job_path": str(completed_job.resolve())},
            )

            self.run_prepare_jobs(project_dir, atoms)

            status = json.loads(
                (project_dir / "vasp" / "jobs" / "train" / "struct_0000" / ".vasp_status")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(status["status"], "pending")

    def test_completed_result_is_recorded_in_state_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            struct_dir = root / "projects" / "project_demo" / "vasp" / "jobs" / "train" / "struct_0042"
            struct_dir.mkdir(parents=True)
            (struct_dir / "OUTCAR").write_text("General timing\n", encoding="utf-8")
            calculation = DftCalculationIdentity(
                structure_id="structure-hash",
                incar_hash="incar-hash",
                potcar_hash="potcar-hash",
            )
            inputs = DftInputArtifacts(
                calculation=calculation,
                working_directory=struct_dir,
                files=(struct_dir / "OUTCAR",),
            )
            artifact = DftResultArtifact(
                calculation=calculation,
                outcar=ArtifactIdentity.from_file("vasp_outcar", struct_dir / "OUTCAR"),
                status="completed",
            )

            with StateStore(root / "state.db") as state_store:
                state_store.upsert_structure(StructureIdentity("structure-hash"))
                state_store.save_execution(
                    DftExecutionRecord(
                        inputs=inputs,
                        attempt_id=f"{calculation.calculation_id}:attempt:1",
                        status="completed",
                    ),
                    artifact=artifact,
                )
                calculation_row = state_store.get_dft_calculation(
                    calculation.calculation_id
                )
                artifacts = state_store.list_artifacts()

            self.assertIsNotNone(calculation_row)
            self.assertEqual(calculation_row["status"], "completed")
            self.assertEqual(artifacts[0]["artifact_type"], "vasp_outcar")

    def test_prepare_fails_on_malformed_status_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = self.make_project(root)
            struct_dir = project_dir / "vasp" / "jobs" / "train" / "struct_0000"
            struct_dir.mkdir(parents=True)
            (struct_dir / ".vasp_status").write_text("{malformed", encoding="utf-8")

            with self.assertRaises(StateError):
                self.run_prepare_jobs(project_dir, self.silicon_atoms())

    def test_prepare_fails_on_malformed_identity_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = self.make_project(root)
            struct_dir = project_dir / "vasp" / "jobs" / "train" / "struct_0000"
            struct_dir.mkdir(parents=True)
            write_status(struct_dir, "submitted")
            (struct_dir / ".vasp_identity").write_text("{malformed", encoding="utf-8")

            with self.assertRaises(StateError):
                self.run_prepare_jobs(project_dir, self.silicon_atoms())

    def test_prepare_fails_on_corrupt_registry_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = self.make_project(root)
            completed_jobs_registry_path(root).write_text(
                "{malformed",
                encoding="utf-8",
            )

            with self.assertRaises(ArtifactError):
                self.run_prepare_jobs(project_dir, self.silicon_atoms())

    def test_status_json_list_is_corrupt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            struct_dir = Path(tmp)
            (struct_dir / ".vasp_status").write_text("[]", encoding="utf-8")

            with self.assertRaises(StateError):
                read_status(struct_dir)

    def test_status_missing_status_field_is_corrupt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            struct_dir = Path(tmp)
            (struct_dir / ".vasp_status").write_text(
                json.dumps({"retry_level": 0}),
                encoding="utf-8",
            )

            with self.assertRaises(StateError):
                read_status(struct_dir)

    def test_identity_missing_required_hash_is_corrupt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            struct_dir = Path(tmp)
            (struct_dir / ".vasp_identity").write_text(
                json.dumps(
                    {
                        "structure_hash": "structure-hash",
                        "incar_hash": "incar-hash",
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(StateError):
                read_identity(struct_dir)

    def test_registry_missing_jobs_field_is_corrupt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = completed_jobs_registry_path(Path(tmp))
            registry_path.write_text(
                json.dumps({"version": VASP_REGISTRY_VERSION}),
                encoding="utf-8",
            )

            with self.assertRaises(ArtifactError):
                read_completed_registry(Path(tmp))

    def test_registry_unsupported_version_is_corrupt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = completed_jobs_registry_path(Path(tmp))
            registry_path.write_text(
                json.dumps(
                    {
                        "version": VASP_REGISTRY_VERSION + 1,
                        "jobs": {},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ArtifactError):
                read_completed_registry(Path(tmp))

    def test_registry_jobs_must_be_an_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = completed_jobs_registry_path(Path(tmp))
            registry_path.write_text(
                json.dumps(
                    {
                        "version": VASP_REGISTRY_VERSION,
                        "jobs": [],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ArtifactError):
                read_completed_registry(Path(tmp))

    def test_matching_registry_entry_must_have_job_path(self) -> None:
        for malformed_entry in ([], {}):
            with self.subTest(malformed_entry=malformed_entry):
                registry = {
                    "version": VASP_REGISTRY_VERSION,
                    "jobs": {
                        "incar-hash": {
                            "potcar-hash": {
                                "structure-hash": malformed_entry,
                            }
                        }
                    },
                }

                with self.assertRaises(ArtifactError):
                    get_registry_entry(
                        registry,
                        "incar-hash",
                        "potcar-hash",
                        "structure-hash",
                    )

    def test_absent_state_keeps_initial_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            struct_dir = root / "struct_0000"
            struct_dir.mkdir()

            self.assertEqual(
                read_status(struct_dir),
                {"status": "pending", "retry_level": 0},
            )
            self.assertEqual(read_identity(struct_dir), {})
            self.assertEqual(
                read_completed_registry(root),
                {"version": VASP_REGISTRY_VERSION, "jobs": {}},
            )


if __name__ == "__main__":
    unittest.main()
