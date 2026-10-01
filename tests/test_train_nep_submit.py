import tempfile
import json
import types
import unittest
from configparser import ConfigParser
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")

from modules.train_nep import submit as submit_module
from modules.train_nep import launcher as launcher_module
from modules.train_nep.train_nep import TrainNepStage
from nepflow.errors import StateError


class SubmitTrainingJobTests(unittest.TestCase):
    def test_submit_training_job_uses_nep_command_from_project_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            dataset_dir = project_dir / "nep" / "datasets" / "dataset_0001"
            potential_dir = project_dir / "nep" / "runs" / "run_0001"
            slurm_dir = project_dir / "config" / "slurm"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            potential_dir.mkdir(parents=True, exist_ok=True)
            slurm_dir.mkdir(parents=True, exist_ok=True)

            for filename in ("train.xyz", "test.xyz", "nep.in"):
                (dataset_dir / filename).write_text("stub\n", encoding="utf-8")
            (slurm_dir / "header.slurm").write_text("#!/bin/bash\nmodule load cuda\n", encoding="utf-8")

            config = ConfigParser()
            config.read_dict(
                {
                    "slurm": {"walltime": "08:00:00"},
                    "hpc": {"nep_command": "/opt/gpumd/bin/nep"},
                }
            )

            completed = types.SimpleNamespace(
                returncode=0,
                stdout="Submitted batch job 12345\n",
                stderr="",
            )
            with patch.object(submit_module.subprocess, "run", return_value=completed):
                job_id = submit_module.submit_training_job(
                    config=config,
                    dataset_path=dataset_dir,
                    potential_path=potential_dir,
                    project_name="demo",
                    project_dir=project_dir,
                )

            self.assertEqual(job_id, "12345")
            script_text = (potential_dir / "train_nep.sh").read_text(encoding="utf-8")
            self.assertIn("/opt/gpumd/bin/nep", script_text)
            self.assertNotIn("$HOME/src/GPUMD/src/nep", script_text)

    def test_missing_nep_command_is_an_explicit_error(self) -> None:
        config = ConfigParser()
        config["slurm"] = {"walltime": "08:00:00"}

        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            header = project_dir / "config" / "slurm" / "header.slurm"
            header.parent.mkdir(parents=True)
            header.write_text("#!/bin/bash\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hpc.nep_command"):
                submit_module.submit_training_job(
                    config=config,
                    dataset_path=project_dir / "dataset",
                    potential_path=project_dir / "potential",
                    project_name="demo",
                    project_dir=project_dir,
                )

    def test_corrupt_training_status_is_an_explicit_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            status_file = project_dir / "nep" / ".train_nep_status"
            status_file.parent.mkdir(parents=True)
            status_file.write_text("{malformed", encoding="utf-8")

            with self.assertRaises(StateError):
                launcher_module.read_train_status(project_dir)

    def test_incomplete_training_state_is_not_treated_as_a_new_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            status_file = project_dir / "nep" / ".train_nep_status"
            status_file.parent.mkdir(parents=True)
            status_file.write_text(json.dumps({"status": "running"}), encoding="utf-8")
            stage = TrainNepStage(
                project_name="demo",
                config_file=project_dir / "config" / "demo.ini",
                state_file=project_dir / "state.db",
                project_dir=project_dir,
                debug=False,
            )

            with self.assertRaisesRegex(ValueError, "potential_path"):
                stage.run()


if __name__ == "__main__":
    unittest.main()
