"""Canonical composition-grid behavior and validation."""

from itertools import combinations

import pytest

from nepflow.config.models import CompositionConfig
from nepflow.errors import ConfigurationError
from nepflow.stages.generation.generators.composition import CompositionGrid
from nepflow.stages.generation.generators.composition_primitives import (
    calculate_composition_realization,
)


def test_phase2_composition_grid_preserves_unary_binary_and_ternary_order() -> None:
    grid = CompositionGrid.from_config(
        CompositionConfig(elements=("Si", "Ge", "C"), composition_step=0.5)
    )

    assert grid.generate() == [
        {"C": 1.0},
        {"Ge": 1.0},
        {"Si": 1.0},
        {"C": 0.5, "Ge": 0.5},
        {"C": 0.5, "Si": 0.5},
        {"Ge": 0.5, "Si": 0.5},
    ]


def test_four_element_pool_generates_only_supported_subsets() -> None:
    elements = ("W", "Cr", "Y", "Zr")
    grid = CompositionGrid.from_config(CompositionConfig(elements=elements, composition_step=0.25))

    compositions = grid.generate()
    unary = {frozenset(composition) for composition in compositions if len(composition) == 1}
    binary = {frozenset(composition) for composition in compositions if len(composition) == 2}
    ternary = {frozenset(composition) for composition in compositions if len(composition) == 3}

    assert unary == {frozenset(subset) for subset in combinations(elements, 1)}
    assert binary == {frozenset(subset) for subset in combinations(elements, 2)}
    assert ternary == {frozenset(subset) for subset in combinations(elements, 3)}
    assert set().union(*(set(composition) for composition in compositions)) == set(elements)
    assert all(1 <= len(composition) <= 3 for composition in compositions)


def test_invalid_composition_step_fails_fast() -> None:
    config = CompositionConfig(elements=("W", "Cr"), composition_step=0.3)
    with pytest.raises(ConfigurationError):
        CompositionGrid.from_config(config)


def test_integer_realization_is_deterministic_for_binary_and_ternary_fractions() -> None:
    binary = calculate_composition_realization({"W": 0.5, "Cr": 0.5}, 3)
    ternary = calculate_composition_realization(
        {"W": 1.0 / 3.0, "Cr": 1.0 / 3.0, "Y": 1.0 / 3.0},
        8,
    )

    assert binary.counts == {"Cr": 2, "W": 1}
    assert binary.realized == {"Cr": 2.0 / 3.0, "W": 1.0 / 3.0}
    assert binary.max_error == pytest.approx(1.0 / 6.0)
    assert ternary.counts == {"Cr": 3, "W": 3, "Y": 2}
    assert ternary.atom_count == 8
    assert ternary.max_error == pytest.approx(1.0 / 12.0)
