"""Canonical selection stage boundary.

Scientific services are imported from their owning modules (for example
``selection.sampling`` and ``selection.representations``). Keeping this
package root narrow avoids making optional descriptor dependencies a package
import requirement.
"""

from __future__ import annotations

__all__ = ["SelectionStage"]


def __getattr__(name: str):
    if name == "SelectionStage":
        from .stage import SelectionStage

        return SelectionStage
    raise AttributeError(name)
