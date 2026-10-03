"""Workflow registry dispatches SELECT through canonical SelectionStage."""

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("NepTrainKit")

from nepflow.cli import compose_stage_registry
from nepflow.config.models import NepflowConfig
from nepflow.stages.selection.stage import SelectionStage
from nepflow.workflow import StageContext, StageRunState, WorkflowStage


def test_select_registry_dispatches_injected_context_to_canonical_stage(tmp_path):
    context = StageContext(
        project_name="demo",
        project_dir=tmp_path,
        config_file=tmp_path / "config" / "project.config",
        state_file=tmp_path / "state.db",
        state_store=None,
        workflow_state=None,
        config=NepflowConfig(),
    )
    with patch.object(SelectionStage, "run") as run:
        result = compose_stage_registry().execute(WorkflowStage.SELECT, context)

    run.assert_called_once_with()
    assert result.stage is WorkflowStage.SELECT
    assert result.status is StageRunState.COMPLETED
    assert result.advanced_to is WorkflowStage.RUN_VASP
