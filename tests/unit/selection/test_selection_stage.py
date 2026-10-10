"""Selection orchestration, injection, persistence, and reconciliation tests."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("ase")
from ase import Atoms

from nepflow.config.models import NepflowConfig, ProjectConfig, SelectionConfig
from nepflow.stages.selection import stage as selection_stage_module
from nepflow.stages.selection.models import SelectionResult
from nepflow.stages.selection.persistence import (
    persist_selection_result,
    structure_ids,
)
from nepflow.stages.selection.stage import SelectionStage
from nepflow.state import StateStore
from nepflow.workflow import StageContext


class StructureStub:
    def __init__(self, structure_id: str):
        self.structure_id = structure_id
        self.num_atoms = 1


def _context(tmp_path: Path, store: StateStore, settings: SelectionConfig) -> StageContext:
    if store.get_project("demo") is None:
        store.upsert_project("demo", name="demo", root_path=str(tmp_path))
    config = NepflowConfig(project=ProjectConfig(name="demo"), selection=settings)
    return StageContext(
        project_name="demo",
        project_dir=tmp_path,
        config_file=tmp_path / "config" / "project.config",
        state_file=tmp_path / "state.db",
        state_store=store,
        workflow_state=None,
        config=config,
    )


def _atoms(element: str, x: float) -> Atoms:
    return Atoms(
        element,
        positions=[[x, 0.0, 0.0]],
        cell=np.eye(3) * 5.0,
        pbc=True,
    )


def _result(descriptors, train=(0,), test=(1,)) -> SelectionResult:
    return SelectionResult(
        descriptors=descriptors,
        train_indices=list(train),
        train_min_dist=0.5,
        train_seed_count=1,
        train_single_element_elastic_count=0,
        train_elastic_count=0,
        train_anchor_count=1,
        train_fps_count=0,
        test_indices=list(test),
        test_min_dist=0.25,
        min_train_test_dist=1.0,
        mean_train_test_dist=1.0,
        seed_indices=[0],
        single_element_elastic_indices=[],
        elastic_indices=[],
    )


def test_prepare_requires_generated_candidates(tmp_path):
    settings = SelectionConfig()
    with StateStore(tmp_path / "state.db") as store:
        stage = SelectionStage(context=_context(tmp_path, store, settings))
        with pytest.raises(FileNotFoundError):
            stage.prepare(tmp_path)


def test_prepare_returns_none_for_an_empty_generated_file(tmp_path):
    generated = tmp_path / "structures" / "generated" / "generated_structures.xyz"
    generated.parent.mkdir(parents=True)
    generated.write_text("", encoding="utf-8")
    settings = SelectionConfig()
    with StateStore(tmp_path / "state.db") as store:
        stage = SelectionStage(context=_context(tmp_path, store, settings))
        with (
            patch.object(selection_stage_module, "_read_nep_structures", return_value=[]),
            patch.object(selection_stage_module, "ase_read", return_value=[]),
        ):
            assert stage.prepare(tmp_path) is None


def test_stage_consumes_injected_typed_config_without_discovery(tmp_path):
    settings = SelectionConfig(target_train_count=1, target_test_count=1)
    with StateStore(tmp_path / "state.db") as store:
        context = _context(tmp_path, store, settings)
        stage = SelectionStage(context=context)
        prepared = {
            "structures": [StructureStub("s0"), StructureStub("s1")],
            "ase_structures": [_atoms("Si", 0.0), _atoms("Ge", 1.0)],
        }
        result = _result(np.ones((2, 2)))
        with (
            patch.object(stage, "prepare", return_value=prepared),
            patch.object(stage, "execute", return_value=result) as execute,
            patch.object(stage, "finalize", return_value=result) as finalize,
        ):
            returned = stage.run()

    assert returned is result
    execute.assert_called_once_with(settings, prepared, context=context)
    finalize.assert_called_once_with(prepared, result, context=context)
    assert not hasattr(stage, "_find_config_file")
    assert not hasattr(selection_stage_module, "load_config")


def test_selection_run_persists_exact_ids_policy_and_completion(tmp_path):
    settings = SelectionConfig(target_train_count=1, target_test_count=1)
    candidates = [_atoms("Si", 0.0), _atoms("Ge", 1.0)]
    result = _result(np.ones((2, 2)))

    with StateStore(tmp_path / "state.db") as store:
        store.upsert_project("demo", name="demo", root_path=str(tmp_path))
        record = persist_selection_result(
            store,
            "demo",
            "demo",
            str(tmp_path),
            settings,
            structure_ids(candidates),
            result,
        )

        assert record["status"] == "completed"
        assert record["completed_at"] is not None
        assert record["parameters"]["selected_structure_ids"]["train"] == [
            structure_ids(candidates)[0]
        ]
        assert record["parameters"]["selected_structure_ids"]["test"] == [
            structure_ids(candidates)[1]
        ]
        assert record["parameters"]["policy"]["target_train_count"] == 1
        assert record["parameters"]["mandatory_anchor_structure_ids"]["seed"] == [
            structure_ids(candidates)[0]
        ]


def test_reopen_reconciles_by_identity_after_candidate_reordering(tmp_path):
    settings = SelectionConfig(target_train_count=1, target_test_count=1)
    original = [_atoms("Si", 0.0), _atoms("Ge", 1.0)]
    original_ids = structure_ids(original)
    result = _result(np.ones((2, 2)))

    state_path = tmp_path / "state.db"
    with StateStore(state_path) as store:
        store.upsert_project("demo", name="demo", root_path=str(tmp_path))
        persist_selection_result(
            store,
            "demo",
            "demo",
            str(tmp_path),
            settings,
            original_ids,
            result,
        )
    current = list(reversed(original))
    prepared = {
        "structures": [StructureStub("current-0"), StructureStub("current-1")],
        "ase_structures": current,
        "generated_path": tmp_path / "structures" / "generated" / "generated.xyz",
    }
    with StateStore(state_path) as reopened_store:
        context = _context(tmp_path, reopened_store, settings)
        stage = SelectionStage(context=context)
        with (
            patch.object(
                selection_stage_module,
                "load_or_calculate_representations",
                return_value=np.ones((2, 2)),
            ),
            patch.object(
                selection_stage_module,
                "select_training_set",
                side_effect=AssertionError("reconciliation must not recompute training"),
            ),
        ):
            restored = stage.execute(settings, prepared, context=context)

    current_ids = structure_ids(current)
    assert [current_ids[index] for index in restored.train_indices] == [original_ids[0]]
    assert [current_ids[index] for index in restored.test_indices] == [original_ids[1]]


def test_report_failure_does_not_remove_completed_scientific_result(tmp_path):
    settings = SelectionConfig(target_train_count=1, target_test_count=1)
    with StateStore(tmp_path / "state.db") as store:
        context = _context(tmp_path, store, settings)
        stage = SelectionStage(context=context)
        prepared = {
            "structures": [StructureStub("s0"), StructureStub("s1")],
            "ase_structures": [_atoms("Si", 0.0), _atoms("Ge", 1.0)],
        }
        result = _result(np.ones((2, 2)))
        with (
            patch.object(stage, "prepare", return_value=prepared),
            patch.object(stage, "execute", return_value=result),
            patch.object(
                selection_stage_module,
                "plot_descriptor_space",
                side_effect=RuntimeError("report failed"),
            ),
        ):
            with pytest.raises(RuntimeError, match="report failed"):
                stage.run()

        run_ids = store.connection.execute("SELECT selection_run_id FROM selection_runs").fetchall()
        assert len(run_ids) == 1
        record = store.get_selection_run(run_ids[0][0])
        assert record["status"] == "completed"
