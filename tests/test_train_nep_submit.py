import tempfile
import types
import unittest
from configparser import ConfigParser
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")

from modules.train_nep import submit as submit_module


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


if __name__ == "__main__":
    unittest.main()
