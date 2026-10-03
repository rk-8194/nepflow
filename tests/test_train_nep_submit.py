import json
import tempfile
import unittest
from pathlib import Path
from typing import cast

import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")

from modules.train_nep import launcher as launcher_module
from modules.train_nep.train_nep import TrainNepStage
from nepflow.errors import StateError
from nepflow.mlip.backend import TrainingInput
from nepflow.mlip.nep.backend import NepBackend


class SubmitTrainingJobTests(unittest.TestCase):
    def test_backend_command_is_argument_oriented_and_scheduler_independent(self) -> None:
        backend = NepBackend("/opt/gpumd/bin/nep --quiet")
        command = backend.training_command(cast(TrainingInput, None))
        self.assertEqual(command, ("/opt/gpumd/bin/nep", "--quiet"))
        self.assertNotIn("sbatch", command)
        self.assertNotIn(";", command)

    def test_missing_nep_command_is_an_explicit_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "command"):
            NepBackend("")

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
