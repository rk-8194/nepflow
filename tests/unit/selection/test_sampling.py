"""Focused contracts for pure selection sampling."""

from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("NepTrainKit")

from nepflow.stages.selection import sampling


class _Structure:
    def __init__(self, num_atoms: int = 1):
        self.num_atoms = num_atoms


def test_select_farthest_points_maps_atomic_rows_to_sorted_frames() -> None:
    representations = np.ones((5, 2))
    structures = [_Structure(2), _Structure(3)]

    with patch.object(
        sampling,
        "farthest_point_sampling",
        return_value=[0, 1, 2, 4],
    ):
        result = sampling.select_farthest_points(
            representations,
            structures,
            False,
            0.2,
        )

    assert result == [0, 1]


def test_target_count_preserves_accepted_binary_search_boundary() -> None:
    representations = np.ones((2, 2))
    structures = [_Structure(), _Structure()]

    with patch.object(sampling, "_selected_count", return_value=2):
        with patch.object(
            sampling,
            "select_farthest_points",
            return_value=[0, 1],
        ):
            result = sampling.select_farthest_points_for_target(
                representations,
                structures,
                True,
                target=2,
                tolerance=0,
                max_iterations=3,
            )

    assert result == ([0, 1], 0.005)


def test_cross_distance_stats_reports_nearest_minimum_and_mean() -> None:
    representations = np.array([[0.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

    minimum, mean = sampling.calculate_cross_distance_stats(
        representations,
        [0],
        [1, 2],
    )

    assert minimum == pytest.approx(2.0)
    assert mean == pytest.approx(2.5)
