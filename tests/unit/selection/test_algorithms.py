"""Tests for the common peer-algorithm selection boundary."""

from dataclasses import dataclass

import numpy as np
import pytest

from nepflow.stages.selection.algorithms import (
    SelectionAlgorithmRequest,
    SelectionAlgorithmResult,
    dispatch_selection_algorithm,
)


def _request(
    algorithm_id: str = "fps",
    *,
    target_count: int = 2,
    anchor_indices: tuple[int, ...] = (1,),
    options: dict | None = None,
) -> SelectionAlgorithmRequest:
    candidate_ids = ("candidate-a", "candidate-b", "candidate-c")
    return SelectionAlgorithmRequest(
        candidate_ids=candidate_ids,
        candidate_structures=(object(), object(), object()),
        representations=np.arange(3, dtype=float).reshape(3, 1),
        target_count=target_count,
        anchor_indices=anchor_indices,
        anchor_ids=tuple(candidate_ids[index] for index in anchor_indices),
        algorithm_id=algorithm_id,
        algorithm_version="test-request-v1",
        options={
            "composition_aware_fps": False,
            "descriptor_type": "structure",
            "fps_target_selector": lambda *args, **kwargs: ([0], 0.5),
            "max_search_iterations": 4,
            "target_tolerance": 1,
            **(options or {}),
        },
    )


def test_fps_boundary_preserves_ordered_candidate_ids_and_anchors() -> None:
    result = dispatch_selection_algorithm(_request())

    assert result.selected_indices == (0, 1)
    assert result.selected_candidate_ids == ("candidate-a", "candidate-b")
    assert result.algorithm_id == "fps"
    assert result.algorithm_version == "fps-v1"


@dataclass
class FailingAlgorithm:
    algorithm_id: str = "information_entropy"
    algorithm_version: str = "entropy-test-v1"
    calls: int = 0

    def select(self, request: SelectionAlgorithmRequest) -> SelectionAlgorithmResult:
        self.calls += 1
        raise RuntimeError("entropy selection failed")


@dataclass
class UnexpectedFPS:
    algorithm_id: str = "fps"
    algorithm_version: str = "fps-test-v1"
    calls: int = 0

    def select(self, request: SelectionAlgorithmRequest) -> SelectionAlgorithmResult:
        self.calls += 1
        raise AssertionError("FPS must not be called")


def test_dispatcher_calls_only_requested_algorithm_and_propagates_failure() -> None:
    failing = FailingAlgorithm()
    fps = UnexpectedFPS()

    with pytest.raises(RuntimeError, match="entropy selection failed"):
        dispatch_selection_algorithm(
            _request("information_entropy"),
            registry={
                "information_entropy": failing,
                "fps": fps,
            },
        )

    assert failing.calls == 1
    assert fps.calls == 0


def test_dispatcher_does_not_fall_back_when_algorithm_is_missing() -> None:
    with pytest.raises(ValueError, match="Unknown selection algorithm"):
        dispatch_selection_algorithm(_request("information_entropy"), registry={})
