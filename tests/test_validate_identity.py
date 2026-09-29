import json
import os
import tempfile
from configparser import ConfigParser
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")

from modules.train_nep.train_nep import TrainNepStage  # noqa: E402
from common.model_manifest import (  # noqa: E402
    create_model_run_manifest,
    update_model_run_status,
    write_model_run_manifest,
)
from modules.validate import validate as validate_stage_module  # noqa: E402
from modules.validate.prepare import (  # noqa: E402
    finalize_nep_potential,
    find_model_run_and_dataset,
    find_latest_potential_and_dataset,
)
from modules.validate.validate import ValidateStage  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
LAYOUT_FIXTURE = ROOT / "tests" / "fixtures" / "run_layouts" / "model_dataset_layouts.json"


def load_layout(name: str) -> dict:
    layouts = json.loads(LAYOUT_FIXTURE.read_text(encoding="utf-8"))["layouts"]
    return next(layout for layout in layouts if layout["name"] == name)


def materialize_layout(
    root: Path,
    layout: dict,
    *,
    write_manifests: bool = True,
) -> tuple[Path, dict[str, Path], dict[str, Path]]:
    """Materialize a deterministic model/dataset layout with conflicting orderings."""
    project_dir = root / "project"
    models: dict[str, Path] = {}
    datasets: dict[str, Path] = {}

    for dataset in layout["datasets"]:
        dataset_path = project_dir / "nep" / dataset["directory"]
        dataset_path.mkdir(parents=True, exist_ok=True)
        (dataset_path / ".dataset").write_text(
            json.dumps({"dataset_id": dataset["dataset_id"]}),
            encoding="utf-8",
        )
        datasets[dataset["dataset_id"]] = dataset_path

    for model in layout["models"]:
        model_path = project_dir / "nep" / model["directory"]
        model_path.mkdir(parents=True, exist_ok=True)
        (model_path / "nep.txt").write_text(f"model={model['model_id']}\n", encoding="utf-8")
        (model_path / "nep.in").write_text(
            f"type 1 {model['model_id']}\n",
            encoding="utf-8",
        )
        if write_manifests:
            manifest = create_model_run_manifest(
                potential_path=model_path,
                dataset_path=datasets[model["dataset_id"]],
                dataset_id=model["dataset_id"],
                nep_in_path=model_path / "nep.in",
                hyperparameters_hash=f"fixture-{model['model_id']}",
            )
            manifest["model_run_id"] = model["model_id"]
            write_model_run_manifest(
                model_path / "model_run_manifest.json",
                manifest,
            )
            update_model_run_status(model_path, "completed")
        models[model["model_id"]] = model_path

    # Deliberately make lower lexical indices newer on disk. Correct resolution
    # must use the explicit association, not either directory or mtime ordering.
    for index, dataset in enumerate(layout["datasets"]):
        timestamp = 2_000_000 - index
        os.utime(datasets[dataset["dataset_id"]], (timestamp, timestamp))
    for index, model in enumerate(layout["models"]):
        timestamp = 2_000_000 - index
        os.utime(models[model["model_id"]], (timestamp, timestamp))

    return project_dir, models, datasets


def train_stage_for(project_dir: Path) -> TrainNepStage:
    return TrainNepStage(
        project_name="demo",
        config_file=project_dir / "config" / "demo.yaml",
        state_file=project_dir / "state.db",
        project_dir=project_dir,
        debug=False,
    )


def validate_stage_for(project_dir: Path) -> ValidateStage:
    return ValidateStage(
        project_name="demo",
        config_file=project_dir / "config" / "demo.yaml",
        state_file=project_dir / "state.db",
        project_dir=project_dir,
        debug=False,
    )


def test_one_explicit_model_dataset_pair_resolves_exactly() -> None:
    layout = load_layout("lexical_order_conflicts_with_explicit_association")
    single_pair = {"datasets": [layout["datasets"][0]], "models": [layout["models"][0]]}

    with tempfile.TemporaryDirectory() as tmp:
        project_dir, models, datasets = materialize_layout(Path(tmp), single_pair)
        resolved = train_stage_for(project_dir)._find_dataset_for_potential(models["model_old"])

    assert resolved == datasets["dataset_old"]


@pytest.mark.parametrize(
    "layout_name",
    [
        "lexical_order_conflicts_with_explicit_association",
        "latest_model_directory_points_to_older_dataset",
    ],
)
def test_model_specific_resolution_ignores_directory_and_mtime_order(layout_name: str) -> None:
    layout = load_layout(layout_name)

    with tempfile.TemporaryDirectory() as tmp:
        project_dir, models, datasets = materialize_layout(Path(tmp), layout)
        stage = train_stage_for(project_dir)
        resolved = {
            model_id: stage._find_dataset_for_potential(model_path)
            for model_id, model_path in models.items()
        }

    expected = {
        model["model_id"]: datasets[model["dataset_id"]]
        for model in layout["models"]
    }
    assert resolved == expected


@pytest.mark.parametrize(
    "layout_name",
    [
        "lexical_order_conflicts_with_explicit_association",
        "latest_model_directory_points_to_older_dataset",
    ],
)
def test_validation_discovery_rejects_ambiguous_latest_pair(layout_name: str) -> None:
    layout = load_layout(layout_name)

    with tempfile.TemporaryDirectory() as tmp:
        project_dir, _, _ = materialize_layout(Path(tmp), layout)
        with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
            find_latest_potential_and_dataset(project_dir)


def test_missing_model_dataset_association_is_an_error() -> None:
    layout = load_layout("lexical_order_conflicts_with_explicit_association")
    single_pair = {"datasets": [layout["datasets"][0]], "models": [layout["models"][0]]}

    with tempfile.TemporaryDirectory() as tmp:
        project_dir, models, _ = materialize_layout(Path(tmp), single_pair, write_manifests=False)
        with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
            train_stage_for(project_dir)._find_dataset_for_potential(models["model_old"])


def test_missing_requested_model_is_an_error() -> None:
    layout = load_layout("lexical_order_conflicts_with_explicit_association")

    with tempfile.TemporaryDirectory() as tmp:
        project_dir, _, _ = materialize_layout(Path(tmp), layout)
        missing_model = project_dir / "nep" / "potentials" / "potential_missing"
        with pytest.raises(FileNotFoundError):
            train_stage_for(project_dir)._find_dataset_for_potential(missing_model)
        with pytest.raises(FileNotFoundError):
            find_model_run_and_dataset(project_dir, "model_missing")


def test_storage_path_is_not_the_scientific_model_identity() -> None:
    layout = {
        "datasets": [{"directory": "datasets/data_a", "dataset_id": "dataset_a"}],
        "models": [{"directory": "potentials/arbitrary_model_storage", "model_id": "model_a", "dataset_id": "dataset_a"}],
    }

    with tempfile.TemporaryDirectory() as tmp:
        project_dir, _, _ = materialize_layout(Path(tmp), layout)
        potential, dataset = find_latest_potential_and_dataset(project_dir, "model_a")

    assert potential.name == "arbitrary_model_storage"
    assert dataset.name == "data_a"


def test_manifest_artifact_hash_mismatch_is_rejected() -> None:
    layout = load_layout("lexical_order_conflicts_with_explicit_association")
    single_pair = {"datasets": [layout["datasets"][0]], "models": [layout["models"][0]]}

    with tempfile.TemporaryDirectory() as tmp:
        project_dir, models, _ = materialize_layout(Path(tmp), single_pair)
        (models["model_old"] / "nep.txt").write_text("tampered\n", encoding="utf-8")
        with pytest.raises(RuntimeError):
            find_latest_potential_and_dataset(project_dir, "model_old")


def test_finalize_preserves_the_resolved_model_dataset_pair() -> None:
    layout = load_layout("lexical_order_conflicts_with_explicit_association")
    single_pair = {"datasets": [layout["datasets"][0]], "models": [layout["models"][0]]}

    with tempfile.TemporaryDirectory() as tmp:
        project_dir, models, datasets = materialize_layout(Path(tmp), single_pair)
        finalized_path, dataset_name = finalize_nep_potential(project_dir, "model_old")

        assert finalized_path == project_dir / "gpumd" / datasets["dataset_old"].name / models["model_old"].name
        assert dataset_name == datasets["dataset_old"].name
        assert (finalized_path / "nep.txt").exists()


def test_validation_entry_point_preserves_model_dataset_association_through_preparation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp) / "project"
        model_path = project_dir / "gpumd" / "dataset_old" / "model_old"
        dataset_path = project_dir / "nep" / "datasets" / "dataset_old"
        (project_dir / "config" / "gpumd").mkdir(parents=True, exist_ok=True)
        model_path.mkdir(parents=True, exist_ok=True)
        dataset_path.mkdir(parents=True, exist_ok=True)

        stage = validate_stage_for(project_dir)
        config = ConfigParser()
        config["slurm"] = {"enabled": "false"}
        config["gpumd"] = {"model_run_id": "model_old"}
        preparation_state = {"struct_count": 1, "validation_root": str(project_dir / "validation")}

        with (
            patch.object(stage, "_find_config_file", return_value=project_dir / "config" / "demo.yaml"),
            patch.object(stage, "_load_config", return_value=config),
            patch.object(
                validate_stage_module,
                "finalize_nep_potential",
                return_value=(model_path, "dataset_old"),
            ),
            patch.object(
                validate_stage_module,
                "prepare_validation_structures",
                return_value=preparation_state,
            ) as prepare_mock,
            patch.object(validate_stage_module, "run_validation_launcher"),
            patch.object(stage, "_run_analysis"),
        ):
            stage.run()

    prepare_mock.assert_called_once_with(
        dataset_path=dataset_path,
        gpumd_potential_dir=model_path,
        project_dir=project_dir,
        config_gpumd_dir=project_dir / "config" / "gpumd",
    )
