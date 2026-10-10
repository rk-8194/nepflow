"""Focused integration tests for the production entropy selection path."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("ase")
from ase import Atoms

from nepflow.config.models import (
    EntropyBandwidthConfig,
    EntropySelectionConfig,
    NepflowConfig,
    ProjectConfig,
    SelectionConfig,
)
from nepflow.stages.selection.algorithms.information_entropy.diagnostics import (
    EntropyScientificDiagnostics,
)
from nepflow.stages.selection.persistence import (
    candidate_ids,
    persist_selection_result,
    restore_selection_result,
    structure_ids,
)
from nepflow.stages.selection.representations import (
    LocalRepresentationConfig,
    build_local_environment_representation,
)
from nepflow.stages.selection.stage import SelectionStage
from nepflow.state import StateStore
from nepflow.workflow import StageContext


def _atoms(symbol: str, x: float) -> Atoms:
    return Atoms(symbol, positions=[[x, 0.0, 0.0]], cell=np.eye(3) * 20.0, pbc=True)


def _context(tmp_path: Path, store: StateStore, settings: SelectionConfig) -> StageContext:
    store.upsert_project("demo", name="demo", root_path=str(tmp_path))
    return StageContext(
        project_name="demo",
        project_dir=tmp_path,
        config_file=tmp_path / "config" / "project.config",
        state_file=tmp_path / "state.db",
        state_store=store,
        workflow_state=None,
        config=NepflowConfig(project=ProjectConfig(name="demo"), selection=settings),
    )


def _settings(target_train_count: int) -> SelectionConfig:
    return SelectionConfig(
        target_train_count=target_train_count,
        target_test_count=1,
        entropy=EntropySelectionConfig(
            bandwidth=EntropyBandwidthConfig(mode="manual", k=1, c=2.0),
        ),
    )


def test_stage_runs_the_sparse_entropy_pipeline_and_keeps_fps_separate(tmp_path: Path) -> None:
    candidates = [_atoms("Si", 0.0), _atoms("Ge", 1.0), _atoms("W", 2.0)]
    ordered_candidates = candidate_ids(candidates)
    ordered_structures = structure_ids(candidates)
    local = build_local_environment_representation(
        candidates,
        config=LocalRepresentationConfig(
            cutoff=4.0,
            radial_bins=2,
            angular_bins=2,
            whitening_singular_policy="regularize",
        ),
        candidate_ids=ordered_candidates,
        structure_ids=ordered_structures,
        local_descriptor_workers=1,
    )
    settings = _settings(2)
    prepared = {"structures": candidates, "ase_structures": candidates}

    with StateStore(tmp_path / "state.db") as store:
        context = _context(tmp_path, store, settings)
        stage = SelectionStage(context=context)
        with (
            patch(
                "nepflow.stages.selection.stage.load_or_calculate_local_representations",
                return_value=local,
            ) as local_loader,
            patch(
                "nepflow.stages.selection.stage.load_or_calculate_representations",
                side_effect=AssertionError("entropy must not load NEP descriptors"),
            ),
        ):
            result = stage.execute(settings, prepared, context=context)
        record = persist_selection_result(
            store,
            "demo",
            "demo",
            str(tmp_path),
            settings,
            ordered_candidates,
            result,
            candidate_structure_ids=ordered_structures,
        )
        restored_result = restore_selection_result(
            record,
            result.descriptors,
            ordered_candidates,
            candidate_structure_ids=ordered_structures,
        )

    local_loader.assert_called_once()
    assert result.algorithm_id == "information_entropy"
    assert result.train_min_dist is None
    assert result.train_min_dist_applicable is False
    assert len(result.train_indices) == 2
    assert len(result.train_acquisition_order) == 2
    assert result.train_entropy_objective is not None
    assert result.train_entropy_graph_fingerprint
    assert result.train_entropy_contributions_fingerprint
    assert result.test_selection_policy == "representative"
    assert result.test_selection_version == "local-environment-holdout-v1"
    assert result.test_selection_provenance["label_holdout"] is True
    assert result.test_selection_provenance["preprocessing_scope"] == (
        "full_candidate_pool_representation_and_calibration"
    )
    assert set(result.train_indices).isdisjoint(result.test_indices)
    assert result.entropy_diagnostics is not None
    restored = EntropyScientificDiagnostics.from_manifest(result.entropy_diagnostics.to_manifest())
    assert restored.fingerprint == result.entropy_diagnostics.fingerprint
    assert restored.recompute_final()["forward_kl"] == pytest.approx(
        result.train_entropy_forward_kl
    )
    assert restored_result.entropy_diagnostics is not None
    assert restored_result.entropy_diagnostics.fingerprint == result.entropy_diagnostics.fingerprint
    assert restored_result.test_selection_policy == result.test_selection_policy
    assert restored_result.test_selection_provenance == result.test_selection_provenance


def test_entropy_budget_is_rejected_before_local_representation_work(tmp_path: Path) -> None:
    candidates = [_atoms("Si", 0.0), _atoms("Ge", 1.0)]
    settings = _settings(3)
    prepared = {"structures": candidates, "ase_structures": candidates}

    with StateStore(tmp_path / "state.db") as store:
        context = _context(tmp_path, store, settings)
        stage = SelectionStage(context=context)
        with patch(
            "nepflow.stages.selection.stage.load_or_calculate_local_representations",
            side_effect=AssertionError("budget validation must precede descriptors"),
        ):
            with pytest.raises(ValueError, match=r"K=3, available M=2"):
                stage.execute(settings, prepared, context=context)
