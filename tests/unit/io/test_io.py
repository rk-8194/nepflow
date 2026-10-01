"""Contract tests for Phase 3 atomic I/O and hashing primitives."""

from __future__ import annotations

import hashlib

import pytest

from nepflow.errors import ArtifactError, StateError
import nepflow.io.atomic as atomic_module
from nepflow.io.atomic import atomic_write_text
from nepflow.io.hashing import (
    sha256_canonical_json,
    sha256_file,
    sha256_text,
)
from nepflow.io.json import canonical_json_bytes, read_json, read_json_object, write_json


def test_file_hash_is_deterministic(tmp_path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"nepflow\x00artifact")

    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha256_file(path) == expected
    assert sha256_file(path) == sha256_file(path)


def test_canonical_json_hash_is_order_independent() -> None:
    first = {"b": [2, 1], "a": "\u00e9"}
    second = {"a": "\u00e9", "b": [2, 1]}

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert sha256_canonical_json(first) == sha256_canonical_json(second)


def test_utf8_json_round_trip(tmp_path) -> None:
    path = tmp_path / "metadata.json"
    value = {"description": "caf\u00e9 \u2014 \u6c27", "count": 2}

    write_json(path, value)

    assert read_json(path, require_object=True) == value
    assert "caf\u00e9" in path.read_text(encoding="utf-8")
    assert sha256_text("caf\u00e9") == hashlib.sha256("caf\u00e9".encode("utf-8")).hexdigest()


def test_malformed_json_raises_state_error(tmp_path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(StateError):
        read_json(path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_standard_json_constants_are_rejected(tmp_path, constant) -> None:
    path = tmp_path / "non-standard.json"
    path.write_text(f'{{"nested": {{"value": {constant}}}}}', encoding="utf-8")

    with pytest.raises(StateError):
        read_json(path)


def test_required_and_optional_json_files_are_explicit(tmp_path) -> None:
    missing = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        read_json(missing)
    with pytest.raises(ArtifactError):
        read_json(missing, missing_error_type=ArtifactError)
    assert read_json_object(missing, default={}) == {}


def test_json_object_default_is_validated(tmp_path) -> None:
    with pytest.raises(ArtifactError):
        read_json_object(
            tmp_path / "missing.json",
            default=[],
            error_type=ArtifactError,
        )


def test_malformed_json_can_use_artifact_error(tmp_path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ArtifactError):
        read_json(path, error_type=ArtifactError)


def test_required_and_optional_missing_file_hashes_are_explicit(tmp_path) -> None:
    path = tmp_path / "missing.bin"

    with pytest.raises(ArtifactError):
        sha256_file(path)
    assert sha256_file(path, required=False) is None


def test_optional_hash_still_rejects_unreadable_artifact(tmp_path) -> None:
    directory = tmp_path / "artifact-directory"
    directory.mkdir()

    with pytest.raises(ArtifactError):
        sha256_file(directory)
    with pytest.raises(ArtifactError):
        sha256_file(directory, required=False)


def test_atomic_replacement_succeeds(tmp_path) -> None:
    path = tmp_path / "state.json"
    path.write_text("old", encoding="utf-8")

    atomic_write_text(path, "new")

    assert path.read_text(encoding="utf-8") == "new"


def test_failed_atomic_replace_preserves_target_and_cleans_temp(tmp_path, monkeypatch) -> None:
    path = tmp_path / "state.json"
    path.write_text("valid", encoding="utf-8")

    def fail_replace(_source, _target):
        raise OSError("controlled replace failure")

    monkeypatch.setattr(atomic_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="controlled replace failure"):
        atomic_write_text(path, "partial content")

    assert path.read_text(encoding="utf-8") == "valid"
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_failed_atomic_fsync_preserves_target_and_cleans_temp(tmp_path, monkeypatch) -> None:
    path = tmp_path / "state.json"
    path.write_text("valid", encoding="utf-8")

    def fail_fsync(_file_descriptor):
        raise OSError("controlled fsync failure")

    monkeypatch.setattr(atomic_module.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="controlled fsync failure"):
        atomic_write_text(path, "partial content", durable=True)

    assert path.read_text(encoding="utf-8") == "valid"
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []
