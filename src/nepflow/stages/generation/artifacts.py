"""Canonical structure representations used by generation artifacts."""

from __future__ import annotations

from io import StringIO
from typing import Any

from ase.io import read, write


def canonicalize_extxyz_structure(structure: Any) -> Any:
    """Round-trip one structure through the final generation extxyz format.

    The generated artifact is reopened by downstream stages, so its textual
    representation is the scientific identity boundary for generated
    structures.  Using ASE's writer and reader here keeps this boundary tied
    to the actual format implementation rather than duplicating its numeric
    formatting rules.
    """

    rendered = StringIO()
    write(rendered, [structure], format="extxyz")
    canonical = read(StringIO(rendered.getvalue()), index=0, format="extxyz")
    if canonical is None:
        raise ValueError("ASE extxyz round-trip returned no structure")
    return canonical


__all__ = ["canonicalize_extxyz_structure"]
