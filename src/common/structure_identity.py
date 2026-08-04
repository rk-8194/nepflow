"""Stable structure identity helpers.

The generated hash intentionally ignores mutable metadata such as
``config_type``. It identifies the physical structure: species, cell, PBC, and
fractional positions in deterministic element-grouped order.
"""

from __future__ import annotations

import hashlib
from typing import Any


STRUCTURE_HASH_VERSION = "structure-v1"


def canonical_structure_text(atoms: Any) -> str:
    """Return a deterministic text representation of an atomistic structure."""
    symbols = list(atoms.get_chemical_symbols())
    unique_elements = sorted(set(symbols))
    scaled_positions = atoms.get_scaled_positions()
    cell = atoms.get_cell()
    pbc = _pbc_values(atoms)

    lines = [STRUCTURE_HASH_VERSION]
    lines.append("pbc " + " ".join("1" if value else "0" for value in pbc))
    lines.append("cell")
    for row in cell:
        lines.append(_format_vector(row))
    lines.append("atoms")
    for elem in unique_elements:
        indices = [i for i, symbol in enumerate(symbols) if symbol == elem]
        for idx in indices:
            lines.append(f"{elem} {_format_vector(scaled_positions[idx])}")
    return "\n".join(lines) + "\n"


def hash_structure(atoms: Any) -> str:
    """Return a SHA-256 hash for the physical structure."""
    return hashlib.sha256(canonical_structure_text(atoms).encode("utf-8")).hexdigest()


def annotate_structure_hash(atoms: Any, *, overwrite: bool = True) -> str:
    """Store the structure hash in ``atoms.info`` and return it."""
    structure_hash = hash_structure(atoms)
    if not hasattr(atoms, "info"):
        atoms.info = {}
    if overwrite or "structure_hash" not in atoms.info:
        atoms.info["structure_hash"] = structure_hash
        atoms.info["structure_hash_version"] = STRUCTURE_HASH_VERSION
    return str(atoms.info["structure_hash"])


def annotate_structure_hashes(structures: list[Any], *, overwrite: bool = True) -> None:
    """Annotate every structure in a list with stable hash metadata."""
    for atoms in structures:
        annotate_structure_hash(atoms, overwrite=overwrite)


def _format_vector(values: Any) -> str:
    return " ".join(f"{float(values[i]):.14f}" for i in range(3))


def _pbc_values(atoms: Any) -> list[bool]:
    pbc = getattr(atoms, "pbc", (True, True, True))
    if hasattr(pbc, "tolist"):
        pbc = pbc.tolist()
    if isinstance(pbc, bool):
        return [pbc, pbc, pbc]
    values = list(pbc)
    if len(values) == 0:
        return [True, True, True]
    if len(values) == 1:
        return [bool(values[0])] * 3
    return [bool(value) for value in values[:3]]
