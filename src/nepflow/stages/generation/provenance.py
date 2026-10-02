"""Phase 2-compatible identity, seed, and provenance operations."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

from nepflow.domain.identities import calculate_structure_id


logger = logging.getLogger(__name__)


def annotate_base_structures(
    bases: Iterable[Any],
    requested_composition: Mapping[str, float],
    elements: Iterable[str],
    gas_elements: Iterable[str],
) -> None:
    """Attach requested-grid metadata without changing realized chemistry."""

    for base in bases:
        base.info.setdefault("composition", dict(requested_composition))
        base.info.setdefault("elements", list(elements))
        if gas_elements:
            base.info.setdefault("gas_elements", list(gas_elements))


def assign_seed_ids(bases: list[Any], start_index: int = 0) -> int:
    for offset, base in enumerate(bases):
        base.info["seed_id"] = f"seed_{start_index + offset:06d}"
    return start_index + len(bases)


def metadata_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def merge_provenance(representative: Any, duplicate: Any | None = None) -> None:
    """Merge duplicate origins while retaining exact Phase 2 field semantics."""

    path_values: list[str] = []
    material_values: list[str] = []
    for atoms in (representative, duplicate):
        if atoms is None:
            continue
        path_values.extend(metadata_values(atoms.info.get("provenance_paths")))
        path_values.extend(metadata_values(atoms.info.get("source")))
        material_values.extend(
            metadata_values(atoms.info.get("provenance_material_ids"))
        )
        material_values.extend(metadata_values(atoms.info.get("material_id")))

    representative.info["provenance_paths"] = sorted(set(path_values))
    distinct_materials = sorted(set(material_values))
    if distinct_materials:
        representative.info["provenance_material_ids"] = distinct_materials


def deduplicate_base_structures(bases: list[Any]) -> list[Any]:
    """Keep one representative per physical identity and merge its origins."""

    unique: list[Any] = []
    by_structure_id: dict[str, Any] = {}
    for base in bases:
        structure_id = calculate_structure_id(base)
        representative = by_structure_id.get(structure_id)
        if representative is None:
            merge_provenance(base)
            by_structure_id[structure_id] = base
            unique.append(base)
        else:
            merge_provenance(representative, base)
    if len(unique) != len(bases):
        logger.info("  Merged %d physically duplicate base structure(s)", len(bases) - len(unique))
    return unique
