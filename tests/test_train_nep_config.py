import tempfile
import unittest
from pathlib import Path

import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")

from modules.train_nep.train_nep import TrainNepStage


class TrainNepConfigTests(unittest.TestCase):
    def create_stage(self, project_dir: Path) -> TrainNepStage:
        return TrainNepStage(
            project_name="demo",
            config_file=project_dir / "config" / "demo.yaml",
            state_file=project_dir / "state.db",
            project_dir=project_dir,
            debug=False,
        )

    def test_generate_nep_config_copies_lambda_shear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            config_dir = project_dir / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            dataset_dir = project_dir / "nep" / "datasets" / "dataset_0001"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / "project.config").write_text(
                "\n".join(
                    [
                        "[composition]",
                        "elements=Si,Ge",
                        "",
                        "[train_nep]",
                        "population=99",
                        "batch=123",
                        "generation=456",
                        "lambda_e=2.0",
                        "lambda_f=3.0",
                        "lambda_v=4.0",
                        "lambda_shear=5.0",
                    ]
                ),
                encoding="utf-8",
            )

            stage = self.create_stage(project_dir)
            stage._generate_nep_config(stage._load_config(), dataset_dir)

            nep_in = (dataset_dir / "nep.in").read_text(encoding="utf-8")
            self.assertIn("lambda_e 2.0", nep_in)
            self.assertIn("lambda_f 3.0", nep_in)
            self.assertIn("lambda_v 4.0", nep_in)
            self.assertIn("lambda_shear 5.0", nep_in)


if __name__ == "__main__":
    unittest.main()
