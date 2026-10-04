"""Pure geometry regressions for validation-cell replication.

Cell-vector lengths, plane heights, and potential cutoffs in this module are
expressed in Angstroms.  The safety condition is strict: each replicated
perpendicular height must be greater than twice the cutoff radius.
"""

import numpy as np
import pytest

from nepflow.errors import ValidationError  # noqa: E402
from nepflow.stages.validation.preparation import (  # noqa: E402
    calculate_cell_replicates_for_cutoff,
)


def perpendicular_heights(cell: np.ndarray) -> np.ndarray:
    """Return lattice-plane heights in Å for a cell whose vectors are rows."""
    vectors = np.asarray(cell, dtype=float)
    volume = abs(float(np.linalg.det(vectors)))
    return np.array(
        [
            volume / np.linalg.norm(np.cross(vectors[1], vectors[2])),
            volume / np.linalg.norm(np.cross(vectors[2], vectors[0])),
            volume / np.linalg.norm(np.cross(vectors[0], vectors[1])),
        ]
    )


def strict_required_replicates(heights: np.ndarray, cutoff_angstrom: float) -> tuple[int, int, int]:
    """Return the smallest repeats for which every height is strictly above 2r."""
    required_height = 2.0 * cutoff_angstrom
    repeats = []
    for height in heights:
        repeat = 1
        while repeat * height <= required_height:
            repeat += 1
        repeats.append(repeat)
    return repeats[0], repeats[1], repeats[2]


def calculate_replicates(cell: np.ndarray, cutoff_angstrom: float) -> tuple[int, int, int]:
    """Call the canonical triclinic-safe replication calculation."""
    return calculate_cell_replicates_for_cutoff(cell, cutoff_angstrom)


def assert_strict_threshold(
    cell: np.ndarray,
    cutoff_angstrom: float,
    repeats: tuple[int, int, int],
) -> None:
    heights = perpendicular_heights(cell)
    required_height = 2.0 * cutoff_angstrom

    for height, repeat in zip(heights, repeats):
        assert repeat * height > required_height
        assert (repeat - 1) * height <= required_height


def test_orthogonal_cubic_cell_uses_lengths_in_angstrom() -> None:
    cell = np.diag([4.0, 4.0, 4.0])
    cutoff_angstrom = 1.5

    heights = perpendicular_heights(cell)
    expected = strict_required_replicates(heights, cutoff_angstrom)
    actual = calculate_replicates(cell, cutoff_angstrom)

    np.testing.assert_allclose(heights, [4.0, 4.0, 4.0], atol=1e-12)
    assert actual == expected == (1, 1, 1)
    assert_strict_threshold(cell, cutoff_angstrom, actual)


def test_orthogonal_anisotropic_cell_uses_each_axis_length() -> None:
    cell = np.diag([2.0, 5.0, 10.0])
    cutoff_angstrom = 2.4

    heights = perpendicular_heights(cell)
    expected = strict_required_replicates(heights, cutoff_angstrom)
    actual = calculate_replicates(cell, cutoff_angstrom)

    np.testing.assert_allclose(heights, [2.0, 5.0, 10.0], atol=1e-12)
    assert actual == expected == (3, 1, 1)
    assert_strict_threshold(cell, cutoff_angstrom, actual)


def test_skewed_triclinic_cell_uses_perpendicular_plane_heights() -> None:
    cell = np.array(
        [
            [8.0, 0.0, 0.0],
            [7.0, 4.0, 0.0],
            [7.0, 1.0, 4.0],
        ]
    )
    cutoff_angstrom = 1.9

    heights = perpendicular_heights(cell)
    expected = strict_required_replicates(heights, cutoff_angstrom)
    actual = calculate_replicates(cell, cutoff_angstrom)

    np.testing.assert_allclose(
        heights,
        [3.3260780821, 3.8805700006, 4.0],
        atol=1e-9,
    )
    assert expected == (2, 1, 1)
    assert actual == expected
    assert_strict_threshold(cell, cutoff_angstrom, actual)


def test_canonical_triclinic_regression_is_exact() -> None:
    cell = np.array([[3.0, 0.0, 0.0], [1.0, 2.5, 0.0], [0.3, 0.4, 7.0]])

    assert calculate_replicates(cell, 2.0) == (2, 2, 1)


def test_strongly_anisotropic_triclinic_cell_uses_perpendicular_heights() -> None:
    cell = np.array(
        [
            [12.0, 0.0, 0.0],
            [11.0, 2.0, 0.0],
            [10.0, 1.0, 0.5],
        ]
    )
    cutoff_angstrom = 0.6

    heights = perpendicular_heights(cell)
    expected = strict_required_replicates(heights, cutoff_angstrom)
    actual = calculate_replicates(cell, cutoff_angstrom)

    np.testing.assert_allclose(
        heights,
        [1.1326300276, 0.8944271910, 0.5],
        atol=1e-9,
    )
    assert expected == (2, 2, 3)
    assert actual == expected
    assert_strict_threshold(cell, cutoff_angstrom, actual)


def test_exact_threshold_requires_one_additional_repeat() -> None:
    cell = np.diag([3.0, 4.0, 5.0])
    cutoff_angstrom = 1.5

    heights = perpendicular_heights(cell)
    expected = strict_required_replicates(heights, cutoff_angstrom)
    actual = calculate_replicates(cell, cutoff_angstrom)

    assert expected == (2, 1, 1)
    assert actual == expected
    assert_strict_threshold(cell, cutoff_angstrom, actual)


def test_no_unnecessary_repeat_when_height_exceeds_twice_cutoff() -> None:
    cell = np.diag([5.0, 6.0, 7.0])
    cutoff_angstrom = 2.0

    heights = perpendicular_heights(cell)
    expected = strict_required_replicates(heights, cutoff_angstrom)
    actual = calculate_replicates(cell, cutoff_angstrom)

    assert np.all(heights > 2.0 * cutoff_angstrom)
    assert actual == expected == (1, 1, 1)
    assert_strict_threshold(cell, cutoff_angstrom, actual)


def test_degenerate_cell_raises_clear_value_error() -> None:
    cell = np.zeros((3, 3))

    with pytest.raises(ValidationError, match="cell|volume|height"):
        calculate_replicates(cell, cutoff_angstrom=2.0)
