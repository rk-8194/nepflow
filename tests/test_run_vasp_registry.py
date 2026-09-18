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

        self.assertNotEqual(common.hash_structure(first), common.hash_structure(moved))
        self.assertNotEqual(common.hash_structure(first), common.hash_structure(larger_cell))

    def test_incar_hash_ignores_resource_params(self) -> None:
        original = "ENCUT = 520\nNCORE = 16\nKPAR = 1\nISMEAR = 0\n"
        rewritten = "ENCUT = 520\nNCORE = 64\nKPAR = 4\nISMEAR = 0\n"

        self.assertEqual(common.hash_incar_text(original), common.hash_incar_text(rewritten))

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
                common.hash_structure(atoms),
                {"job_path": str(completed_job.resolve())},
            )

            self.run_prepare_jobs(project_dir, atoms)

            status = json.loads(
                (project_dir / "vasp" / "jobs" / "train" / "struct_0000" / ".vasp_status")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(status["status"], "reused")
            self.assertEqual(Path(status["reused_from"]), completed_job.resolve())

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
                common.hash_structure(atoms),
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
                common.hash_structure(atoms),
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


if __name__ == "__main__":
    unittest.main()
