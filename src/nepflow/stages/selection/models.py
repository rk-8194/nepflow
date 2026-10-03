"""Typed records for the canonical selection stage boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Complete in-memory result produced by one selection run."""

    descriptors: np.ndarray
    train_indices: list[int]
    train_min_dist: float
    train_seed_count: int
    train_single_element_elastic_count: int
    train_elastic_count: int
    train_anchor_count: int
    train_fps_count: int
    test_indices: list[int]
    test_min_dist: float
    min_train_test_dist: float
    mean_train_test_dist: float
    seed_indices: list[int]
    single_element_elastic_indices: list[int]
    elastic_indices: list[int]

    def as_mapping(self) -> dict[str, Any]:
        """Return the legacy result shape at the artifact boundary."""

        return {
            "descriptors": self.descriptors,
            "train_indices": self.train_indices,
            "train_min_dist": self.train_min_dist,
            "train_seed_count": self.train_seed_count,
            "train_single_element_elastic_count": self.train_single_element_elastic_count,
            "train_elastic_count": self.train_elastic_count,
            "train_anchor_count": self.train_anchor_count,
            "train_fps_count": self.train_fps_count,
            "test_indices": self.test_indices,
            "test_min_dist": self.test_min_dist,
            "min_train_test_dist": self.min_train_test_dist,
            "mean_train_test_dist": self.mean_train_test_dist,
            "seed_indices": self.seed_indices,
            "single_element_elastic_indices": self.single_element_elastic_indices,
            "elastic_indices": self.elastic_indices,
        }

    def __getitem__(self, key: str) -> Any:
        """Allow read-only mapping access while callers migrate to fields."""

        return self.as_mapping()[key]


__all__ = ["SelectionResult"]
