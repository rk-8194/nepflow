"""Exactness and diagnostics tests for indexed reference-measure occupancy."""

import logging

import numpy as np
import pytest

from nepflow.stages.selection.reference_measure import (
    _fixed_radius_occupancy,
    estimate_reference_measure,
)


def _oracle(values: np.ndarray, tolerance: float) -> np.ndarray:
    radius_squared = tolerance * tolerance
    result = np.empty(values.shape[0], dtype=np.int64)
    for index, source in enumerate(values):
        with np.errstate(over="ignore", invalid="ignore"):
            squared = np.sum((source[None, :] - values) ** 2, axis=1, dtype=np.float64)
        result[index] = int(np.count_nonzero(squared <= radius_squared))
    return result


@pytest.mark.parametrize("dimension", [1, 3, 8, 42])
def test_indexed_occupancy_matches_inclusive_oracle(dimension: int) -> None:
    generator = np.random.default_rng(1800 + dimension)
    values = generator.normal(size=(48, dimension)).astype(np.float64)
    values[1] = values[0]
    values[3] = values[2]
    tolerance = 1.0e-6

    expected = _oracle(values, tolerance)
    actual = _fixed_radius_occupancy(values, tolerance=tolerance, chunk_size=5)

    np.testing.assert_array_equal(actual, expected)


def test_indexed_occupancy_preserves_duplicates_and_exact_boundary() -> None:
    tolerance = 1.0
    values = np.asarray(
        [
            [0.0],
            [0.0],
            [1.0],
            [np.nextafter(1.0, 0.0)],
            [np.nextafter(1.0, np.inf)],
            [3.0],
        ],
        dtype=np.float64,
    )

    expected = _oracle(values, tolerance)
    actual = _fixed_radius_occupancy(values, tolerance=tolerance, chunk_size=2)

    np.testing.assert_array_equal(actual, expected)
    assert actual[0] == actual[1]


def test_singleton_effectiveness_warning_does_not_change_weights(caplog) -> None:
    values = np.asarray([[0.0], [10.0], [20.0]], dtype=np.float64)
    with caplog.at_level(logging.WARNING):
        result = estimate_reference_measure(values, tolerance=1.0e-6, chunk_size=2)

    np.testing.assert_allclose(result.probabilities, [1.0 / 3.0] * 3)
    assert result.diagnostics["singleton_row_fraction"] == pytest.approx(1.0)
    assert "almost entirely singleton" in caplog.text
