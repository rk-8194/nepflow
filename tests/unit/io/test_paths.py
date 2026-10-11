from pathlib import Path

import pytest

from nepflow.errors import ArtifactError
from nepflow.io.paths import project_relative_path, resolve_project_path


def test_project_paths_are_relative_and_root_scoped(tmp_path: Path) -> None:
    artifact = tmp_path / "structures" / "seeds" / "base_structures.xyz"

    assert project_relative_path(artifact, tmp_path) == "structures/seeds/base_structures.xyz"
    assert (
        resolve_project_path("structures/seeds/base_structures.xyz", tmp_path)
        == artifact
    )


@pytest.mark.parametrize("value", ["../outside.xyz", "structures/../../outside.xyz"])
def test_project_paths_reject_traversal(tmp_path: Path, value: str) -> None:
    with pytest.raises(ArtifactError):
        resolve_project_path(value, tmp_path, error_type=ArtifactError)


def test_project_relative_serialization_rejects_out_of_root_path(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError):
        project_relative_path(tmp_path.parent / "outside.xyz", tmp_path, error_type=ArtifactError)
