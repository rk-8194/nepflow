"""Canonical composition-grid behavior and validation."""

import pytest

from nepflow.config.models import CompositionConfig
from nepflow.errors import ConfigurationError
from nepflow.stages.generation.generators.composition import CompositionGrid


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


@pytest.mark.parametrize(
    "config",
    [
        CompositionConfig(elements=("A", "B", "C", "D")),
        CompositionConfig(elements=("A", "B"), composition_step=0.3),
    ],
)
def test_unsupported_composition_spaces_fail_fast(config: CompositionConfig) -> None:
    with pytest.raises(ConfigurationError):
        CompositionGrid.from_config(config)
