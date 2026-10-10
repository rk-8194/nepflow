"""Cache identity tests for local entropy representations."""

import json

import numpy as np
from ase import Atoms

from nepflow.stages.selection.representations import (
    LocalRepresentationConfig,
    load_or_calculate_local_representations,
    local_representation_manifest_path,
)


def _candidate(symbol: str, x: float) -> Atoms:
    atoms = Atoms(
        symbol,
        positions=[[x, 1.0, 1.0]],
        cell=[12.0, 12.0, 12.0],
        pbc=False,
    )
    atoms.info["candidate_id"] = f"candidate-{symbol}-{x}"
    return atoms


def test_cache_reuses_exact_identity_and_invalidates_scientific_inputs(tmp_path) -> None:
    candidates = [_candidate("Si", 1.0), _candidate("Ge", 2.0)]
    config = LocalRepresentationConfig(
        cutoff=4.0,
        radial_bins=4,
        angular_bins=4,
        species=("Ge", "Si"),
    )
    first = load_or_calculate_local_representations(
        tmp_path,
        candidates,
        config=config,
        candidate_ids=["candidate-si", "candidate-ge"],
        structure_ids=["structure-si", "structure-ge"],
    )
    second = load_or_calculate_local_representations(
        tmp_path,
        candidates,
        config=config,
        candidate_ids=["candidate-si", "candidate-ge"],
        structure_ids=["structure-si", "structure-ge"],
    )
    np.testing.assert_array_equal(first.descriptors, second.descriptors)
    manifest_path = local_representation_manifest_path(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first_fingerprint = manifest["representation_fingerprint"]

    changed_config = LocalRepresentationConfig(
        cutoff=4.0,
        radial_bins=5,
        angular_bins=4,
        species=("Ge", "Si"),
    )
    load_or_calculate_local_representations(
        tmp_path,
        candidates,
        config=changed_config,
        candidate_ids=["candidate-si", "candidate-ge"],
        structure_ids=["structure-si", "structure-ge"],
    )
    changed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert changed["representation_fingerprint"] != first_fingerprint

    candidates[0].positions += [0.1, 0.0, 0.0]
    load_or_calculate_local_representations(
        tmp_path,
        candidates,
        config=changed_config,
        candidate_ids=["candidate-si", "candidate-ge"],
        structure_ids=["structure-si", "structure-ge"],
    )
    content_changed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert (
        content_changed["identity"]["candidate_content_fingerprints"][0]
        != changed["identity"]["candidate_content_fingerprints"][0]
    )


def test_cache_identity_includes_order_magnetic_mode_tolerance_and_backend_version(
    tmp_path,
) -> None:
    first = _candidate("Fe", 1.0)
    first.set_array("magnetic_moments", np.asarray([[1.0, 0.0, 0.0]]))
    second = _candidate("Fe", 2.0)
    second.set_array("magnetic_moments", np.asarray([[-1.0, 0.0, 0.0]]))
    candidates = [first, second]
    config = LocalRepresentationConfig(
        magnetic_mode="non_soc",
        whitening_tolerance=1.0e-10,
        backend_version="backend-a",
    )
    load_or_calculate_local_representations(
        tmp_path,
        candidates,
        config=config,
        candidate_ids=["candidate-a", "candidate-b"],
        structure_ids=["structure-a", "structure-a"],
    )
    manifest_path = local_representation_manifest_path(tmp_path)
    original = json.loads(manifest_path.read_text(encoding="utf-8"))

    changed = LocalRepresentationConfig(
        magnetic_mode="structural",
        whitening_tolerance=1.0e-8,
        backend_version="backend-b",
    )
    load_or_calculate_local_representations(
        tmp_path,
        list(reversed(candidates)),
        config=changed,
        candidate_ids=["candidate-b", "candidate-a"],
        structure_ids=["structure-a", "structure-b"],
    )
    updated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert updated["identity"]["candidate_ids"] == ["candidate-b", "candidate-a"]
    assert updated["identity"]["magnetic_mode"] == "structural"
    assert updated["identity"]["preprocessing"]["tolerance"] == 1.0e-8
    assert updated["identity"]["representation"]["backend_version"] == "backend-b"
    assert updated["identity"] != original["identity"]
