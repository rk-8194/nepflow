import importlib.util
import sys
import tempfile
import types
import unittest
from configparser import ConfigParser
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def ensure_package(name: str) -> None:
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


for package_name in (
    "testpkg",
    "testpkg.modules",
    "testpkg.modules.train_nep",
):
    ensure_package(package_name)

common_module = types.ModuleType("testpkg.modules.train_nep._common")
common_module.logger = types.SimpleNamespace(info=lambda *a, **k: None, debug=lambda *a, **k: None, warning=lambda *a, **k: None)
sys.modules["testpkg.modules.train_nep._common"] = common_module

submit_module = load_module(
    "testpkg.modules.train_nep.submit",
    SRC / "modules" / "train_nep" / "submit.py",
)


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

            completed = types.SimpleNamespace(returncode=0, stdout="Submitted batch job 12345\n", stderr="")
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
