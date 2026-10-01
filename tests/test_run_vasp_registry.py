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

from ase import Atoms
from modules.run_vasp import _common as common
from modules.run_vasp import launcher, prepare
from nepflow.domain.identities import calculate_structure_id


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
        (project_dir / "structures" / "selected" / "train.xyz").write_text(
            "train",
            encoding="utf-8",
        )
        (project_dir / "structures" / "selected" / "test.xyz").write_text(
            "test",
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
        def fake_iread(path, **_kwargs):
            return iter([atoms] if "train.xyz" in str(path) else [])

        with patch.object(prepare, "iread", side_effect=fake_iread):
            prepare.prepare_jobs(
                ConfigParser(),
                project_dir / "config" / "vasp",
                project_dir / "structures" / "selected",
                project_dir / "vasp" / "jobs",
                ["train", "test"],
                project_dir=project_dir,
                project_name="demo",
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

        self.assertEqual(common.hash_incar_text(original), common.hash_incar_text(rewritten))

    def test_effective_incar_defaults_participate_in_identity(self) -> None:
        template = "ENCUT = 520\n"
        first_config = ConfigParser()
        first_config["vasp"] = {"kspacing": "0.30", "kgamma": ".TRUE."}
        second_config = ConfigParser()
        second_config["vasp"] = {"kspacing": "0.35", "kgamma": ".TRUE."}

        first_effective = prepare.inject_incar_defaults(template, first_config)
        second_effective = prepare.inject_incar_defaults(template, second_config)

        self.assertNotEqual(
            common.hash_incar_text(first_effective),
            common.hash_incar_text(second_effective),
        )
        self.assertNotEqual(
            common.hash_incar_text(template),
            common.hash_incar_text(first_effective),
        )

    def test_scheduler_and_resource_provenance_do_not_change_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atoms = self.silicon_atoms()
            incar = prepare.inject_incar_defaults(
                "ENCUT = 520\nNCORE = 16\nKPAR = 1\n",
                ConfigParser(),
            )
            identity = (
                common.hash_incar_text(incar),
                common.hash_potcar_bytes(b"Si-potcar-v1"),
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
                common.upsert_registry_entry(
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
            prepare._identity_matches(
                {"structure_hash": "structure-hash", "incar_hash": "incar-hash"},
                expected,
            )
        )

    def test_potcar_hash_changes_when_content_changes(self) -> None:
        self.assertNotEqual(
            common.hash_potcar_bytes(b"Si-potcar-v1"),
            common.hash_potcar_bytes(b"Si-potcar-v2"),
        )

    def test_prepare_marks_registry_match_as_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = self.make_project(root)
            atoms = self.silicon_atoms()

            completed_job = root / "projects" / "project_old" / "vasp" / "jobs" / "train" / "struct_0007"
            completed_job.mkdir(parents=True)
            (completed_job / "OUTCAR").write_text("General timing\n", encoding="utf-8")
            incar = prepare.inject_incar_defaults(
                "ENCUT = 520\nNCORE = 16\nKPAR = 1\n",
                ConfigParser(),
            )
            common.upsert_registry_entry(
                root,
                common.hash_incar_text(incar),
                common.hash_potcar_bytes(b"Si-potcar-v1"),
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
            incar = prepare.inject_incar_defaults(
                "ENCUT = 520\nNCORE = 16\nKPAR = 1\n",
                ConfigParser(),
            )
            common.upsert_registry_entry(
                root,
                common.hash_incar_text(incar),
                common.hash_potcar_bytes(b"Si-potcar-v1"),
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
            incar = prepare.inject_incar_defaults(
                "ENCUT = 520\nNCORE = 16\nKPAR = 1\n",
                ConfigParser(),
            )
            common.upsert_registry_entry(
                root,
                common.hash_incar_text(incar),
                common.hash_potcar_bytes(b"Si-potcar-v1"),
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
            incar = prepare.inject_incar_defaults(
                "ENCUT = 520\nNCORE = 16\nKPAR = 1\n",
                ConfigParser(),
            )
            common.upsert_registry_entry(
                root,
                common.hash_incar_text(incar),
                common.hash_potcar_bytes(b"Si-potcar-v1"),
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
            common.upsert_registry_entry(
                root,
                common.hash_incar_text("ENCUT = 520\n"),
                common.hash_potcar_bytes(b"Si-potcar-v1"),
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
            incar = prepare.inject_incar_defaults(
                "ENCUT = 520\nNCORE = 16\nKPAR = 1\n",
                ConfigParser(),
            )
            common.upsert_registry_entry(
                root,
                common.hash_incar_text(incar),
                common.hash_potcar_bytes(b"Si-potcar-v1"),
                calculate_structure_id(atoms),
                {"job_path": str(completed_job.resolve())},
            )

            self.run_prepare_jobs(project_dir, atoms)

            status = json.loads(
                (project_dir / "vasp" / "jobs" / "train" / "struct_0000" / ".vasp_status")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(status["status"], "pending")

    def test_register_completed_job_writes_registry_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            struct_dir = root / "projects" / "project_demo" / "vasp" / "jobs" / "train" / "struct_0042"
            struct_dir.mkdir(parents=True)
            identity = {
                "structure_hash": "structure-hash",
                "incar_hash": "incar-hash",
                "potcar_hash": "potcar-hash",
            }
            (struct_dir / ".vasp_identity").write_text(json.dumps(identity), encoding="utf-8")
            (struct_dir / "OUTCAR").write_text("General timing\n", encoding="utf-8")

            launcher._register_completed_job(struct_dir, root, "demo", "train", 42)

            registry = common.read_completed_registry(root)
            entry = registry["jobs"]["incar-hash"]["potcar-hash"]["structure-hash"]
            self.assertEqual(Path(entry["job_path"]), struct_dir.resolve())
            self.assertEqual(entry["project_name"], "demo")
            self.assertEqual(entry["dataset"], "train")
            self.assertEqual(entry["selected_index"], 42)
            self.assertIn("completed_at", entry)

    def test_prepare_fails_on_malformed_status_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = self.make_project(root)
            struct_dir = project_dir / "vasp" / "jobs" / "train" / "struct_0000"
            struct_dir.mkdir(parents=True)
            (struct_dir / ".vasp_status").write_text("{malformed", encoding="utf-8")

            with patch.object(
                prepare,
                "iread",
                side_effect=lambda path, **_kwargs: iter(
                    [self.silicon_atoms()] if "train.xyz" in str(path) else []
                ),
            ):
                with self.assertRaises(ValueError):
                    prepare.prepare_jobs(
                        ConfigParser(),
                        project_dir / "config" / "vasp",
                        project_dir / "structures" / "selected",
                        project_dir / "vasp" / "jobs",
                        ["train", "test"],
                        project_dir=project_dir,
                        project_name="demo",
                    )

    def test_prepare_fails_on_malformed_identity_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = self.make_project(root)
            struct_dir = project_dir / "vasp" / "jobs" / "train" / "struct_0000"
            struct_dir.mkdir(parents=True)
            common.write_status(struct_dir, "submitted")
            (struct_dir / ".vasp_identity").write_text("{malformed", encoding="utf-8")

            with patch.object(
                prepare,
                "iread",
                side_effect=lambda path, **_kwargs: iter(
                    [self.silicon_atoms()] if "train.xyz" in str(path) else []
                ),
            ):
                with self.assertRaises(ValueError):
                    prepare.prepare_jobs(
                        ConfigParser(),
                        project_dir / "config" / "vasp",
                        project_dir / "structures" / "selected",
                        project_dir / "vasp" / "jobs",
                        ["train", "test"],
                        project_dir=project_dir,
                        project_name="demo",
                    )

    def test_prepare_fails_on_corrupt_registry_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = self.make_project(root)
            common.completed_jobs_registry_path(root).write_text(
                "{malformed",
                encoding="utf-8",
            )

            with patch.object(
                prepare,
                "iread",
                side_effect=lambda path, **_kwargs: iter(
                    [self.silicon_atoms()] if "train.xyz" in str(path) else []
                ),
            ):
                with self.assertRaises(ValueError):
                    prepare.prepare_jobs(
                        ConfigParser(),
                        project_dir / "config" / "vasp",
                        project_dir / "structures" / "selected",
                        project_dir / "vasp" / "jobs",
                        ["train", "test"],
                        project_dir=project_dir,
                        project_name="demo",
                    )

    def test_status_json_list_is_corrupt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            struct_dir = Path(tmp)
            (struct_dir / ".vasp_status").write_text("[]", encoding="utf-8")

            with self.assertRaises(ValueError):
                common.read_status(struct_dir)

    def test_status_missing_status_field_is_corrupt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            struct_dir = Path(tmp)
            (struct_dir / ".vasp_status").write_text(
                json.dumps({"retry_level": 0}),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                common.read_status(struct_dir)

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

            with self.assertRaises(ValueError):
                prepare._read_identity(struct_dir)

    def test_registry_missing_jobs_field_is_corrupt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = common.completed_jobs_registry_path(Path(tmp))
            registry_path.write_text(
                json.dumps({"version": common.VASP_REGISTRY_VERSION}),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                common.read_completed_registry(Path(tmp))

    def test_registry_unsupported_version_is_corrupt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = common.completed_jobs_registry_path(Path(tmp))
            registry_path.write_text(
                json.dumps(
                    {
                        "version": common.VASP_REGISTRY_VERSION + 1,
                        "jobs": {},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                common.read_completed_registry(Path(tmp))

    def test_registry_jobs_must_be_an_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = common.completed_jobs_registry_path(Path(tmp))
            registry_path.write_text(
                json.dumps(
                    {
                        "version": common.VASP_REGISTRY_VERSION,
                        "jobs": [],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                common.read_completed_registry(Path(tmp))

    def test_matching_registry_entry_must_have_job_path(self) -> None:
        for malformed_entry in ([], {}):
            with self.subTest(malformed_entry=malformed_entry):
                registry = {
                    "version": common.VASP_REGISTRY_VERSION,
                    "jobs": {
                        "incar-hash": {
                            "potcar-hash": {
                                "structure-hash": malformed_entry,
                            }
                        }
                    },
                }

                with self.assertRaises(ValueError):
                    common.get_registry_entry(
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
                common.read_status(struct_dir),
                {"status": "pending", "retry_level": 0},
            )
            self.assertEqual(prepare._read_identity(struct_dir), {})
            self.assertEqual(
                common.read_completed_registry(root),
                {"version": common.VASP_REGISTRY_VERSION, "jobs": {}},
            )


if __name__ == "__main__":
    unittest.main()
