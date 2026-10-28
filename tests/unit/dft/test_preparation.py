from pathlib import Path

import pytest
from ase import Atoms
from ase.io import write as ase_write

from nepflow.config.models import NepflowConfig
from nepflow.dft.backend import DftInputRequest
from nepflow.dft.vasp.backend import VaspBackend
from nepflow.domain.identities import ArtifactIdentity
from nepflow.errors import StateError
from nepflow.stages.dft.orchestrator import VaspPreparationOrchestrator
from nepflow.state import StateStore


class CountingVaspBackend(VaspBackend):
    def __init__(self) -> None:
        super().__init__(command=("vasp_std",))
        self.prepare_calls = 0

    def prepare_inputs(self, request: DftInputRequest):
        self.prepare_calls += 1
        return super().prepare_inputs(request)


def _project(tmp_path: Path) -> tuple[Path, Atoms]:
    selected = tmp_path / "structures" / "selected"
    selected.mkdir(parents=True)
    atoms = Atoms("Si", cell=[3, 3, 3], positions=[[0, 0, 0]], pbc=True)
    ase_write(selected / "train.xyz", atoms, format="extxyz")
    (selected / "INCAR").write_text("ENCUT = 520\n")
    (selected / "POTCAR").write_bytes(b"Si-potcar")
    return tmp_path, atoms


@pytest.mark.parametrize("persisted_status", ("pending", "submitted", "running"))
def test_restart_reuses_prepared_inputs_for_existing_attempt(
    tmp_path: Path, persisted_status: str
) -> None:
    project_dir, _ = _project(tmp_path)
    backend = CountingVaspBackend()
    with StateStore(project_dir / "state.db") as state_store:
        orchestrator = VaspPreparationOrchestrator(backend=backend)
        first = orchestrator.prepare_calculations(
            config=NepflowConfig(),
            project_dir=project_dir,
            project_name="demo",
            state_store=state_store,
            datasets=("train",),
        )
        prepared = first.calculations[0]
        persisted_files = (
            *prepared.artifacts.files,
            prepared.artifacts.working_directory / ".vasp_identity",
        )
        snapshots = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in persisted_files}
        attempt = state_store.list_dft_attempts(prepared.calculation.calculation_id)[0]
        attempt_id = str(attempt["attempt_id"])
        if persisted_status != "pending":
            state_store.update_dft_attempt(
                attempt_id,
                status=persisted_status,
                job_id="job-1",
            )

        second = orchestrator.prepare_calculations(
            config=NepflowConfig(),
            project_dir=project_dir,
            project_name="demo",
            state_store=state_store,
            datasets=("train",),
        )

        assert backend.prepare_calls == 1
        assert second.calculations[0].status == persisted_status
        assert (
            second.calculations[0].calculation.calculation_id == prepared.calculation.calculation_id
        )
        persisted_attempt = state_store.list_dft_attempts(prepared.calculation.calculation_id)[0]
        assert persisted_attempt["attempt_id"] == attempt_id
        assert persisted_attempt["job_id"] == ("job-1" if persisted_status != "pending" else None)
        for path, (content, mtime) in snapshots.items():
            assert path.read_bytes() == content
            assert path.stat().st_mtime_ns == mtime


def test_live_attempt_with_missing_prepared_input_fails_without_repreparing(
    tmp_path: Path,
) -> None:
    project_dir, _ = _project(tmp_path)
    backend = CountingVaspBackend()
    with StateStore(project_dir / "state.db") as state_store:
        orchestrator = VaspPreparationOrchestrator(backend=backend)
        first = orchestrator.prepare_calculations(
            config=NepflowConfig(),
            project_dir=project_dir,
            project_name="demo",
            state_store=state_store,
            datasets=("train",),
        )
        prepared = first.calculations[0]
        attempt = state_store.list_dft_attempts(prepared.calculation.calculation_id)[0]
        state_store.update_dft_attempt(
            str(attempt["attempt_id"]), status="submitted", job_id="job-1"
        )
        (prepared.artifacts.working_directory / "POSCAR").unlink()

        with pytest.raises(StateError, match="missing"):
            orchestrator.prepare_calculations(
                config=NepflowConfig(),
                project_dir=project_dir,
                project_name="demo",
                state_store=state_store,
                datasets=("train",),
            )

        assert backend.prepare_calls == 1


def test_changed_identity_reprepares_non_active_calculation(tmp_path: Path) -> None:
    project_dir, atoms = _project(tmp_path)
    backend = CountingVaspBackend()
    with StateStore(project_dir / "state.db") as state_store:
        orchestrator = VaspPreparationOrchestrator(backend=backend)
        first = orchestrator.prepare_calculations(
            config=NepflowConfig(),
            project_dir=project_dir,
            project_name="demo",
            state_store=state_store,
            datasets=("train",),
        )
        old_calculation_id = first.calculations[0].calculation.calculation_id
        stale_outcar = first.calculations[0].artifacts.working_directory / "OUTCAR"
        stale_outcar.write_text("stale output\n", encoding="utf-8")
        atoms.positions[0] = [0.2, 0.0, 0.0]
        ase_write(project_dir / "structures" / "selected" / "train.xyz", atoms, format="extxyz")

        second = orchestrator.prepare_calculations(
            config=NepflowConfig(),
            project_dir=project_dir,
            project_name="demo",
            state_store=state_store,
            datasets=("train",),
        )

        assert backend.prepare_calls == 2
        assert second.calculations[0].calculation.calculation_id != old_calculation_id
        assert not stale_outcar.exists()


def test_changed_identity_does_not_replace_an_active_working_directory(
    tmp_path: Path,
) -> None:
    project_dir, atoms = _project(tmp_path)
    backend = CountingVaspBackend()
    with StateStore(project_dir / "state.db") as state_store:
        orchestrator = VaspPreparationOrchestrator(backend=backend)
        first = orchestrator.prepare_calculations(
            config=NepflowConfig(),
            project_dir=project_dir,
            project_name="demo",
            state_store=state_store,
            datasets=("train",),
        )
        prepared = first.calculations[0]
        attempt = state_store.list_dft_attempts(prepared.calculation.calculation_id)[0]
        state_store.update_dft_attempt(str(attempt["attempt_id"]), status="running", job_id="job-1")
        snapshots = {
            path: path.read_bytes()
            for path in (
                *prepared.artifacts.files,
                prepared.artifacts.working_directory / ".vasp_identity",
            )
        }

        atoms.positions[0] = [0.2, 0.0, 0.0]
        ase_write(project_dir / "structures" / "selected" / "train.xyz", atoms, format="extxyz")

        with pytest.raises(StateError, match="active attempt"):
            orchestrator.prepare_calculations(
                config=NepflowConfig(),
                project_dir=project_dir,
                project_name="demo",
                state_store=state_store,
                datasets=("train",),
            )

        assert backend.prepare_calls == 1
        assert {path: path.read_bytes() for path in snapshots} == snapshots


def test_completed_calculation_reuses_inputs_and_output_read_only(tmp_path: Path) -> None:
    project_dir, _ = _project(tmp_path)
    backend = CountingVaspBackend()
    with StateStore(project_dir / "state.db") as state_store:
        orchestrator = VaspPreparationOrchestrator(backend=backend)
        first = orchestrator.prepare_calculations(
            config=NepflowConfig(),
            project_dir=project_dir,
            project_name="demo",
            state_store=state_store,
            datasets=("train",),
        )
        prepared = first.calculations[0]
        outcar = prepared.artifacts.working_directory / "OUTCAR"
        outcar.write_text("General timing\n", encoding="utf-8")
        attempt_id = str(
            state_store.list_dft_attempts(prepared.calculation.calculation_id)[0]["attempt_id"]
        )
        state_store.register_completed_result(
            prepared.calculation.calculation_id,
            attempt_id,
            [ArtifactIdentity.from_file("vasp_outcar", outcar)],
        )
        snapshots = {
            path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in (
                *prepared.artifacts.files,
                prepared.artifacts.working_directory / ".vasp_identity",
            )
        }

        second = orchestrator.prepare_calculations(
            config=NepflowConfig(),
            project_dir=project_dir,
            project_name="demo",
            state_store=state_store,
            datasets=("train",),
        )

        assert backend.prepare_calls == 1
        assert second.calculations[0].status == "completed"
        assert second.calculations[0].reused_output is not None
        for path, (content, mtime) in snapshots.items():
            assert path.read_bytes() == content
            assert path.stat().st_mtime_ns == mtime
