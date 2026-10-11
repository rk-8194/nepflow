"""Runtime resource discovery and shared operation budgets."""

from .budget import (
    DEFAULT_SAFETY_MARGIN_FRACTION,
    ResourceBudget,
    ResourceBudgetService,
    ResourceCapacityError,
    ResourceReservation,
    ResourceSnapshot,
    WorkspaceLease,
    build_resource_budget,
    detect_resource_snapshot,
)
from .cpu import effective_cpu_allocation

__all__ = [
    "DEFAULT_SAFETY_MARGIN_FRACTION",
    "ResourceBudget",
    "ResourceBudgetService",
    "ResourceCapacityError",
    "ResourceReservation",
    "ResourceSnapshot",
    "WorkspaceLease",
    "build_resource_budget",
    "detect_resource_snapshot",
    "effective_cpu_allocation",
]
