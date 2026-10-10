"""Exact distinct-location neighbour reference tests."""

import numpy as np
import pytest

from nepflow.stages.selection.algorithms.information_entropy.neighbours import (
    compute_exact_neighbours,
)


def test_distinct_location_ties_use_smallest_original_representative() -> None:
    descriptors = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]],
        dtype=np.float64,
    )

    result = compute_exact_neighbours(descriptors, 2, chunk_size=1)

    assert result.neighbour_indices[0].tolist() == [1, 2]
    np.testing.assert_array_equal(result.neighbour_indices[3], result.neighbour_indices[0])
    np.testing.assert_allclose(result.radii, [1.0, 2.0, 2.0, 1.0])
    assert result.row_to_location[0] == result.row_to_location[3]
    assert result.unique_locations.shape == (3, 2)


def test_chunk_boundaries_do_not_change_exact_answers() -> None:
    descriptors = np.asarray(
        [[0.0], [0.5], [1.0], [2.0], [3.0]],
        dtype=np.float64,
    )

    first = compute_exact_neighbours(descriptors, 2, chunk_size=1)
    second = compute_exact_neighbours(descriptors, 2, chunk_size=64)

    np.testing.assert_array_equal(first.neighbour_indices, second.neighbour_indices)
    np.testing.assert_array_equal(first.neighbour_distances, second.neighbour_distances)
    np.testing.assert_array_equal(first.radii, second.radii)


@pytest.mark.parametrize(
    "descriptors",
    [
        np.empty((0, 2), dtype=np.float64),
        np.empty((2, 0), dtype=np.float64),
        np.asarray([[0.0, np.nan]], dtype=np.float64),
        np.asarray([[0.0, np.inf]], dtype=np.float64),
    ],
)
def test_invalid_descriptor_pools_fail_explicitly(descriptors: np.ndarray) -> None:
    with pytest.raises(ValueError):
        compute_exact_neighbours(descriptors, 1)


@pytest.mark.parametrize("bad_k", [0, -1, True, 1.5])
def test_invalid_k_fails_explicitly(bad_k: object) -> None:
    descriptors = np.asarray([[0.0], [1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="k"):
        compute_exact_neighbours(descriptors, bad_k)  # type: ignore[arg-type]


def test_all_coincident_or_insufficient_locations_fail() -> None:
    coincident = np.zeros((4, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="distinct"):
        compute_exact_neighbours(coincident, 1)

    insufficient = np.asarray([[0.0], [1.0], [0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="distinct"):
        compute_exact_neighbours(insufficient, 2)
