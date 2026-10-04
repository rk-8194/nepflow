from __future__ import annotations

import json

import pytest

from nepflow.dft.vasp.registry import (
    VASP_REGISTRY_VERSION,
    read_completed_registry,
)
from nepflow.errors import ArtifactError


def test_backfill_registry_missing_file_is_initialized(tmp_path) -> None:
    assert read_completed_registry(tmp_path) == {
        "version": VASP_REGISTRY_VERSION,
        "jobs": {},
    }


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"version": VASP_REGISTRY_VERSION + 1, "jobs": {}},
        {"version": VASP_REGISTRY_VERSION, "jobs": []},
    ],
)
def test_backfill_registry_does_not_repair_invalid_existing_file(tmp_path, payload) -> None:
    path = tmp_path / ".vasp_completed_jobs.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArtifactError):
        read_completed_registry(tmp_path)


def test_backfill_registry_malformed_json_is_artifact_error(tmp_path) -> None:
    path = tmp_path / ".vasp_completed_jobs.json"
    path.write_text("{malformed", encoding="utf-8")

    with pytest.raises(ArtifactError):
        read_completed_registry(tmp_path)
