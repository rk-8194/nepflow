import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")

from nepflow.config import load_config, render_default_config  # noqa: E402
from nepflow.state import StateStore  # noqa: E402
from nepflow.stages.dft import DftStage  # noqa: E402
from nepflow.stages.dft.orchestrator import DftPreparationResult  # noqa: E402
from nepflow.workflow import StageContext  # noqa: E402


def initialized_config(project_dir: Path):
    rendered = render_default_config(
        "demo",
        {
            "materialsproject_api_key": "",
            "elements": "Si",
            "gas_elements": "",
            "crystal_structures": "diamond",
            "target_n_atoms": "64",
            "scp_address": "user@host:/srv/nepflow",
        },
    )
    config_path = project_dir / "config" / "project.config"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(rendered, encoding="utf-8")
    return load_config(config_path, project_name="demo")


class SpyOrchestrator:
    def __init__(self) -> None:
        self.call = None

    def prepare_calculations(self, **kwargs):
        self.call = kwargs
        return DftPreparationResult(())


def make_context(project_dir: Path, config, state_store: StateStore) -> StageContext:
    return StageContext(
        project_name="demo",
        project_dir=project_dir,
        config_file=project_dir / "config" / "project.config",
        state_file=project_dir / "state.db",
        state_store=state_store,
        workflow_state=None,
        config=config,
    )


def test_typed_dft_stage_delegates_hpc_command_without_legacy_stage() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp) / "project_demo"
        config = initialized_config(project_dir)
        orchestrator = SpyOrchestrator()

        with StateStore(project_dir / "state.db") as state_store:
            result = DftStage(orchestrator=orchestrator).run(
                make_context(project_dir, config, state_store)
            )

        assert result.preparation.prepared_count == 0
        assert orchestrator.call["config"].hpc.vasp_command == (
            "mpirun -np {ntasks} vasp_std"
        )
        assert orchestrator.call["datasets"] == ("train", "test")


def test_blank_hpc_command_is_an_explicit_error() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp) / "project_demo"
        config = initialized_config(project_dir)
        config = replace(config, hpc=replace(config.hpc, vasp_command="   "))
        orchestrator = SpyOrchestrator()

        with StateStore(project_dir / "state.db") as state_store:
            with pytest.raises(ValueError, match="hpc\\.vasp_command"):
                DftStage(orchestrator=orchestrator).run(
                    make_context(project_dir, config, state_store)
                )
