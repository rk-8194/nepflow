"""Canonical provenance annotation for perturbation candidates."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from nepflow.domain.identities import StructureIdentity, calculate_structure_id
from nepflow.domain.structures import GeneratedStructureRecord, StructureProvenance


def annotate_generation_provenance(
    candidate: Any,
    base: Any,
    perturbation_family: str,
    *,
    random_seed: int | None = None,
    parameters: Mapping[str, Any] | None = None,
    operation_id: str | None = None,
) -> GeneratedStructureRecord:
    """Attach a canonical record while retaining accepted ``Atoms.info`` fields."""

    base_info = getattr(base, "info", {})
    candidate_info = getattr(candidate, "info", {})
    for key in (
        "composition",
        "actual_composition",
        "crystal_structure",
        "configurational_type",
        "source",
        "seed_id",
    ):
        if key in base_info:
            candidate_info.setdefault(key, base_info[key])
    candidate_info["perturbation_type"] = perturbation_family
    if random_seed is not None:
        candidate_info["random_seed"] = int(random_seed)
    if parameters:
        candidate_info.update(dict(parameters))

    parent_structure_id = calculate_structure_id(base)
    realized = _realized_composition(candidate)
    # ``actual_composition`` is an accepted legacy metadata field describing
    # the source structure.  Keep it unchanged when present; the canonical
    # record below carries the perturbation's realized composition separately.
    if "actual_composition" not in candidate_info and "actual_composition" in base_info:
        candidate_info["actual_composition"] = base_info["actual_composition"]
    record = StructureProvenance(
        parent_structure_id=parent_structure_id,
        generator="perturbation",
        requested_composition=base_info.get("composition"),
        realised_composition=realized,
        source_database_id=(
            str(base_info["material_id"])
            if base_info.get("material_id") is not None
            else None
        ),
        crystal_structure=(
            None
            if base_info.get("crystal_structure") is None
            else str(base_info["crystal_structure"])
        ),
        perturbation_family=perturbation_family,
        perturbation_parameters=dict(parameters or {}),
        random_seed=None if random_seed is None else int(random_seed),
        operation_id=operation_id
        or f"perturbation:{parent_structure_id}:{perturbation_family}",
        code_version=None,
        config_fingerprint=None,
    )
    identity = StructureIdentity.from_atoms(candidate)
    generated = GeneratedStructureRecord(identity=identity, provenance=record)
    candidate_info["generation_provenance"] = record.to_dict()
    return generated


def _realized_composition(atoms: Any) -> dict[str, float]:
    symbols = atoms.get_chemical_symbols()
    counts = Counter(symbols)
    total = len(symbols)
    if total == 0:
        return {}
    return {element: counts[element] / total for element in sorted(counts)}


__all__ = ["annotate_generation_provenance"]
