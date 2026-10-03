"""Training/test strategy ownership tests."""

import numpy as np
import pytest

pytest.importorskip("ase")

from ase import Atoms

from nepflow.stages.selection import sampling, strategy


class StructureStub:
    def __init__(self, num_atoms: int = 1):
        self.num_atoms = num_atoms


def _atoms(symbols: str, x: float = 0.0) -> Atoms:
    atoms = Atoms(
        symbols,
        positions=np.array([[x + index, 0.0, 0.0] for index in range(len(Atoms(symbols)))], dtype=float),
        cell=np.eye(3) * 5.0,
        pbc=True,
    )
    return atoms


def _settings(**overrides):
    settings = {
        "target_train": 3,
        "composition_aware_fps": False,
        "mean_descriptor": True,
        "tolerance": 1,
        "max_iterations": 4,
    }
    settings.update(overrides)
    return settings


def test_training_selection_preserves_and_deduplicates_anchors(monkeypatch):
    monkeypatch.setattr(
        strategy,
        "select_farthest_points_for_target",
        lambda *args, **kwargs: ([0], 0.5),
    )
    selected, minimum = strategy.select_training_set(
        np.arange(8, dtype=float).reshape(4, 2),
        [StructureStub() for _ in range(4)],
        _settings(),
        seed_indices=[2, 0],
        elastic_indices=[2],
    )

    assert selected == [0, 1, 2]
    assert minimum == 0.5


def test_training_selection_rejects_too_many_unique_anchors():
    with pytest.raises(ValueError, match="unique anchors=3"):
        strategy.select_training_set(
            np.ones((4, 2)),
            [StructureStub() for _ in range(4)],
            {"target_train": 2},
            seed_indices=[0, 1],
            elastic_indices=[2],
        )


def test_composition_aware_selection_preserves_anchor_and_target():
    structures = [_atoms("SiGe"), _atoms("SiGe", 1.0), _atoms("SiAl", 2.0)]
    selected, _ = strategy.select_training_set(
        np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]),
        [StructureStub() for _ in structures],
        _settings(
            target_train=3,
            composition_aware_fps=True,
            composition_aware_fps_frontier_fraction=0.1,
            composition_aware_fps_ternary_weight=1.0,
            composition_aware_fps_adaptive_retries=1,
            composition_aware_fps_descriptor_floor_fraction=0.0,
        ),
        ase_structures=structures,
        seed_indices=[0],
    )

    assert len(selected) == 3
    assert 0 in selected


def test_composition_aware_selection_falls_back_for_unary_data(monkeypatch):
    monkeypatch.setattr(
        strategy,
        "select_farthest_points_for_target",
        lambda *args, **kwargs: ([0], 0.25),
    )
    selected, minimum = strategy.select_training_set(
        np.array([[0.0], [1.0], [2.0]]),
        [StructureStub() for _ in range(3)],
        _settings(target_train=2, composition_aware_fps=True),
        ase_structures=[_atoms("Si"), _atoms("Ge"), _atoms("W")],
    )

    assert selected == [0]
    assert minimum == 0.25


def test_test_selection_returns_cross_distance_metrics(monkeypatch):
    monkeypatch.setattr(
        strategy,
        "select_farthest_points_for_target",
        lambda *args, **kwargs: ([0], 0.4),
    )
    result = strategy.select_test_set(
        np.array([[0.0], [1.0], [3.0]]),
        [StructureStub() for _ in range(3)],
        [0],
        _settings(target_test_count=1, test_pool_factor=1.0),
    )

    assert result["test_indices"] == [2]
    assert result["test_min_dist"] == 0.4


def test_distance_helpers_are_owned_by_sampling():
    representations = np.array([[0.0], [1.0], [3.0]])
    assert sampling.calculate_mean_nearest_distance(representations, [0, 1, 2]) == pytest.approx(4.0 / 3.0)
    assert sampling.calculate_positive_min_distance(
        np.array([[0.0], [0.0], [2.0]]), [0, 1, 2]
    ) == 2.0
