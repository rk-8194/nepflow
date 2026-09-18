from configparser import ConfigParser
import tempfile
import unittest
from pathlib import Path

import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")

from modules.train_nep.train_nep import TrainNepStage


def make_train_config(**overrides: object) -> ConfigParser:
    """Build a complete, deterministic project config for identity comparisons."""
    config = ConfigParser()
    config["composition"] = {
        "elements": "Si,Ge",
        "gasElements": "",
    }
    config["train_nep"] = {
        "cutoff": "6 5",
        "n_max": "4 4",
        "basis_size": "8 8",
        "l_max": "4 2 1",
        "neuron": "80",
        "population": "50",
        "batch": "3000",
        "generation": "250000",
        "outerZBL": "2.0",
        "charge_mode": "0",
        "weights": "1,1",
        "lambda_e": "1.0",
        "lambda_f": "1.0",
        "lambda_v": "1.0",
        "lambda_shear": "1.0",
    }
    for key, value in overrides.items():
        config["train_nep"][key] = str(value)
    return config


def render_and_identify(**overrides: object) -> tuple[str, str]:
    """Render nep.in and return it with the path-independent folder identity."""
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp)
        dataset_dir = project_dir / "nep" / "datasets" / "dataset_0001"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        stage = TrainNepStage(
            project_name="demo",
            config_file=project_dir / "config" / "demo.yaml",
            state_file=project_dir / "state.db",
            project_dir=project_dir,
            debug=False,
        )
        config = make_train_config(**overrides)
        stage._generate_nep_config(config, dataset_dir)
        potential_path = stage._create_potential_folder(config, train_count=10)

        rendered = (dataset_dir / "nep.in").read_text(encoding="utf-8")
        identity = potential_path.name.rsplit("_", 1)[0]
        return rendered, identity


def identity_cases() -> list[object]:
    """Return the current identity/rendering contract matrix."""
    p0_8 = "Phase 2 blocker P0-8: model identity and rendered NEP input diverge"
    return [
        pytest.param("cutoff", "6 5", "7 5", marks=pytest.mark.xfail(strict=True, reason=p0_8)),
        pytest.param("n_max", "4 4", "5 4", marks=pytest.mark.xfail(strict=True, reason=p0_8)),
        pytest.param("basis_size", "8 8", "9 8", marks=pytest.mark.xfail(strict=True, reason=p0_8)),
        pytest.param("l_max", "4 2 1", "5 2 1", marks=pytest.mark.xfail(strict=True, reason=p0_8)),
        pytest.param("neuron", "80", "96", marks=pytest.mark.xfail(strict=True, reason=p0_8)),
        pytest.param("population", "50", "60", marks=pytest.mark.xfail(strict=True, reason=p0_8)),
        pytest.param("batch", "3000", "4000", marks=pytest.mark.xfail(strict=True, reason=p0_8)),
        pytest.param("generation", "250000", "300000", marks=pytest.mark.xfail(strict=True, reason=p0_8)),
        pytest.param("outerZBL", "2.0", "3.0"),
        pytest.param("charge_mode", "0", "1", marks=pytest.mark.xfail(strict=True, reason=p0_8)),
        pytest.param("lambda_e", "1.0", "2.0", marks=pytest.mark.xfail(strict=True, reason=p0_8)),
        pytest.param("lambda_f", "1.0", "2.0", marks=pytest.mark.xfail(strict=True, reason=p0_8)),
        pytest.param("lambda_v", "1.0", "2.0", marks=pytest.mark.xfail(strict=True, reason=p0_8)),
        pytest.param("lambda_shear", "1.0", "2.0", marks=pytest.mark.xfail(strict=True, reason=p0_8)),
        pytest.param("weights", "1,1", "2,1"),
    ]


@pytest.mark.parametrize("key,baseline,variant", identity_cases(), ids=lambda case: case[0])
def test_identity_bearing_config_changes_render_and_run_identity(
    key: str,
    baseline: str,
    variant: str,
) -> None:
    baseline_rendered, baseline_identity = render_and_identify(**{key: baseline})
    variant_rendered, variant_identity = render_and_identify(**{key: variant})

    assert baseline_rendered != variant_rendered, key
    assert baseline_identity != variant_identity, key


def test_equivalent_effective_inputs_have_stable_identity_across_paths_and_formatting() -> None:
    comma_rendered, comma_identity = render_and_identify(weights="1, 2")
    spaced_rendered, spaced_identity = render_and_identify(weights="1.0 2.0")

    assert comma_rendered == spaced_rendered
    assert comma_identity == spaced_identity


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
