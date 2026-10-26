import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("ase")
pytest.importorskip("pymatgen")

from nepflow.config import load_config, render_default_config  # noqa: E402
from nepflow.hpc.resources import JobResources  # noqa: E402
from nepflow.stages.dft import DftStage  # noqa: E402
from nepflow.stages.dft.orchestrator import DftPreparationResult  # noqa: E402
from nepflow.state import StateStore  # noqa: E402
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
        assert orchestrator.call["config"].hpc.vasp_command == ("mpirun -np {ntasks} vasp_std")
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


def test_dft_recovery_preserves_scheduler_routing(monkeypatch, tmp_path: Path) -> None:
    config = initialized_config(tmp_path)
    config = replace(
        config,
        slurm=replace(
            config.slurm,
            account="project-account",
            dft_partition="dft-gpu",
            training_partition="train-gpu",
            validation_partition="validation-gpu",
        ),
    )

    class FakeRecoveryPolicy:
        def __init__(self, _config) -> None:
            pass

        def decide(self, **_kwargs):
            return SimpleNamespace(
                retry=True,
                retry_level=1,
                reason="retry",
                nodes=2,
                gpus_per_node=2,
            )

        def apply(self, _decision, _directory) -> None:
            pass

    monkeypatch.setitem(
        DftStage._build_recovery_policy.__globals__,
        "VaspRecoveryPolicy",
        FakeRecoveryPolicy,
    )
    policy = DftStage._build_recovery_policy(config)
    record = SimpleNamespace(
        resources=JobResources(nodes=1, gpus_per_node=1, mpi_ranks=1),
        retry_level=0,
        inputs=SimpleNamespace(
            calculation=SimpleNamespace(ncore=None, kpar=None, calculation_id="scientific-id"),
            working_directory=tmp_path,
        ),
    )

    decision = policy(record, SimpleNamespace(recoverable=True, reason="oom"))

    assert decision.resources.account == "project-account"
    assert decision.resources.partition == "dft-gpu"
    assert decision.resources.nodes == 2
    assert decision.resources.gpus_per_node == 2
    assert decision.resources.mpi_ranks == 4
    assert record.inputs.calculation.calculation_id == "scientific-id"
