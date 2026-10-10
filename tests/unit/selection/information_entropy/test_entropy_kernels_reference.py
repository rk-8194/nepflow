"""Independent finite-pool Wendland kernel reference tests."""

import numpy as np
import pytest

from nepflow.stages.selection.algorithms.information_entropy.kernels import (
    evaluate_leave_one_out_objective,
    normalized_kernel_matrix,
    source_normalisers,
    wendland_kernel,
)


def test_wendland_support_and_monotonicity() -> None:
    values = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0, 2.0], dtype=np.float64)
    kernel = np.asarray(wendland_kernel(values), dtype=np.float64)

    assert kernel[0] == pytest.approx(1.0)
    assert kernel[4] == 0.0
    assert kernel[5] == 0.0
    assert np.all(kernel >= 0.0)
    assert np.all(np.diff(kernel) <= 0.0)


def test_source_normalisation_includes_self_and_is_not_probability_weighted() -> None:
    descriptors = np.asarray([[0.0], [1.0], [3.0]], dtype=np.float64)
    bandwidths = np.asarray([2.0, 2.0, 2.0], dtype=np.float64)
    matrix = normalized_kernel_matrix(descriptors, bandwidths, chunk_size=1)
    expected_raw = np.asarray(
        [
            [1.0, float(wendland_kernel(0.5)), 0.0],
            [
                float(wendland_kernel(0.5)),
                1.0,
                float(wendland_kernel(1.0)),
            ],
            [0.0, float(wendland_kernel(1.0)), 1.0],
        ],
        dtype=np.float64,
    )
    expected = expected_raw / np.sum(expected_raw, axis=0, keepdims=True)

    np.testing.assert_allclose(matrix, expected)
    np.testing.assert_allclose(np.sum(matrix, axis=0), 1.0)
    np.testing.assert_allclose(
        source_normalisers(descriptors, bandwidths), np.sum(expected_raw, axis=0)
    )


def test_source_bandwidth_orientation_and_loo_source_exclusion() -> None:
    descriptors = np.asarray([[0.0], [1.0], [3.0]], dtype=np.float64)
    bandwidths = np.asarray([2.0, 3.0, 2.0], dtype=np.float64)
    probabilities = np.asarray([0.5, 0.25, 0.25], dtype=np.float64)

    matrix = normalized_kernel_matrix(descriptors, bandwidths)
    objective = evaluate_leave_one_out_objective(
        descriptors,
        probabilities,
        bandwidths,
        chunk_size=1,
    )

    assert matrix[0, 1] > 0.0
    assert matrix[0, 0] > 0.0
    assert objective.valid
    assert objective.objective is not None
    assert np.all(objective.probabilities > 0.0)


def test_dense_kernel_oracle_has_an_explicit_memory_guard() -> None:
    descriptors = np.arange(5, dtype=np.float64).reshape(5, 1)
    bandwidths = np.ones(5, dtype=np.float64)
    with pytest.raises(ValueError, match="memory bound"):
        normalized_kernel_matrix(descriptors, bandwidths, max_dense_entries=4)
