"""Pure composition and descriptor-distance selection contracts."""

import numpy as np
import pytest

pytest.importorskip("ase")

from ase import Atoms

from nepflow.stages.selection import sampling


def _atoms(symbols: str, composition: dict[str, float] | None = None) -> Atoms:
    atoms = Atoms(symbols, positions=np.zeros((len(Atoms(symbols)), 3)))
    if composition is not None:
        atoms.info["composition"] = composition
    return atoms


def test_composition_projection_covers_binary_and_ternary_subsets() -> None:
    binary = sampling.composition_projection_bins(_atoms("SiGe"))
    quaternary = sampling.composition_projection_bins(_atoms("SiGeAlCu"))

    assert len(binary["binary"]) == 1
    assert len(binary["ternary"]) == 0
    assert len(quaternary["binary"]) == 6
    assert len(quaternary["ternary"]) == 4


def test_composition_schedule_preserves_retry_order() -> None:
    assert sampling.composition_aware_attempt_schedule(0.1, 1.0, 3) == [
        (0.1, 1.0),
        (0.05, 1.0),
        (0.2, 1.0),
    ]


def test_distance_metrics_preserve_duplicate_and_positive_contracts() -> None:
    representations = np.array([[0.0], [0.0], [2.0]])

    assert sampling.descriptor_distance(representations, 0, 2) == 2.0
    assert sampling.calculate_min_distance(representations, [0, 1, 2]) == 0.0
    assert sampling.calculate_positive_min_distance(representations, [0, 1, 2]) == 2.0
    assert sampling.calculate_mean_nearest_distance(
        representations,
        [0, 1, 2],
    ) == pytest.approx(2.0 / 3.0)


def test_composition_coverage_metrics_return_subset_means() -> None:
    atoms = [_atoms("SiGe"), _atoms("SiGe"), _atoms("SiAl")]
    candidate_bins = {
        index: sampling.composition_projection_bins(value)
        for index, value in enumerate(atoms)
    }

    metrics = sampling.calculate_composition_coverage_metrics([0, 2], candidate_bins)

    assert 0.0 <= metrics["binary_occupied_bin_fraction"] <= 1.0
    assert 0.0 <= metrics["binary_normalized_entropy"] <= 1.0


def test_best_sampling_attempt_respects_descriptor_floor() -> None:
    base = {
        "attempt_number": 1,
        "train_positive_min_dist": 1.0,
        "train_min_dist": 1.0,
        "train_mean_nn_dist": 1.0,
        "binary_occupied_bin_fraction": 0.2,
        "binary_normalized_entropy": 0.2,
        "ternary_occupied_bin_fraction": 0.0,
        "ternary_normalized_entropy": 0.0,
    }
    rejected = {
        **base,
        "attempt_number": 2,
        "train_positive_min_dist": 0.5,
        "train_min_dist": 0.5,
        "train_mean_nn_dist": 0.5,
        "binary_occupied_bin_fraction": 1.0,
    }
    accepted = {
        **base,
        "attempt_number": 3,
        "train_positive_min_dist": 1.1,
        "train_min_dist": 1.1,
        "train_mean_nn_dist": 1.1,
        "binary_occupied_bin_fraction": 0.8,
    }

    result = sampling.select_best_sampling_attempt([base, rejected, accepted], 0.9)

    assert result["attempt_number"] == 3
