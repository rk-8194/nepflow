"""Canonical workflow stage classes.

Stage implementation details and backend-specific types belong to their
owning packages; this root intentionally exposes only the five stage names.
"""

from __future__ import annotations

__all__ = [
    "DftStage",
    "GenerationStage",
    "SelectionStage",
    "TrainingStage",
    "ValidationStage",
]


def __getattr__(name: str):
    modules = {
        "DftStage": ".dft",
        "GenerationStage": ".generation",
        "SelectionStage": ".selection",
        "TrainingStage": ".training",
        "ValidationStage": ".validation",
    }
    module_name = modules.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = __import__(f"{__name__}{module_name}", fromlist=[name])
    return getattr(module, name)
