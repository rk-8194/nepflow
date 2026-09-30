"""Pure geometry regressions for validation-cell replication.

Cell-vector lengths, plane heights, and potential cutoffs in this module are
expressed in Angstroms.  The safety condition is strict: each replicated
perpendicular height must be greater than twice the cutoff radius.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("ase")

from ase import Atoms  # noqa: E402

from modules.validate import prepare as prepare_module  # noqa: E402


class _CellFixture:
    def __init__(self, cell: np.ndarray) -> None:
        self.cell = np.asarray(cell, dtype=float)

    def get_cell(self) -> np.ndarray:
        return self.cell


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


def strict_required_replicates(
    heights: np.ndarray, cutoff_angstrom: float
) -> tuple[int, int, int]:
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
    """Call production replication logic without filesystem or GPUMD dependencies."""
    with (
        patch.object(
            prepare_module,
            "parse_cutoff_from_nep",
            return_value=cutoff_angstrom,
        ),
        patch.object(
            prepare_module,
            "ase_read",
            return_value=_CellFixture(cell),
        ),
    ):
        return prepare_module.calculate_required_replicates(
            Path("model.xyz"),
            Path("nep.txt"),
        )


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

    with pytest.raises(ValueError, match="cell|volume|height"):
        calculate_replicates(cell, cutoff_angstrom=2.0)


def test_prepare_validation_structures_propagates_invalid_cell_failure(
    tmp_path: Path,
) -> None:
    invalid_atoms = Atoms(
        "Si",
        positions=[[0.0, 0.0, 0.0]],
        cell=np.zeros((3, 3)),
        pbc=True,
    )
    dataset_path = tmp_path / "nep" / "datasets" / "dataset_0001"
    potential_path = tmp_path / "gpumd" / "dataset_0001" / "potential_0001"
    config_gpumd_dir = tmp_path / "config" / "gpumd"
    dataset_path.mkdir(parents=True)
    potential_path.mkdir(parents=True)
    config_gpumd_dir.mkdir(parents=True)
    (potential_path / "nep.txt").write_text(
        "version 4\ntype 1 Si\ncutoff 2 5 112 60\n", encoding="utf-8"
    )
    (config_gpumd_dir / "run.in_validate").write_text(
        "replicate 1 1 1\n", encoding="utf-8"
    )

    with patch.object(
        prepare_module, "parse_test_xyz", return_value=[{"atoms": invalid_atoms}]
    ):
        with pytest.raises(ValueError, match="cell|volume|height"):
            prepare_module.prepare_validation_structures(
                dataset_path=dataset_path,
                gpumd_potential_dir=potential_path,
                project_dir=tmp_path,
                config_gpumd_dir=config_gpumd_dir,
            )

    run_in_path = potential_path / "validation" / "struct_0000" / "run.in"
    assert not run_in_path.exists()
