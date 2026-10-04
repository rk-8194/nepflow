"""Materials Project document, cache-record, and ASE conversion."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ase import Atoms
from ase.build import bulk
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

logger = logging.getLogger(__name__)

SPACE_GROUP_TO_STRUCTURE = {
    229: "bcc",
    225: "fcc",
    194: "hcp",
    227: "diamond",
    221: "simple_cubic",
}

STRUCTURE_SYMMETRY_MAP = {
    "bcc": {"crystal_system": "cubic", "space_groups": [229]},
    "fcc": {"crystal_system": "cubic", "space_groups": [225]},
    "hcp": {"crystal_system": "hexagonal", "space_groups": [194]},
    "diamond": {"crystal_system": "cubic", "space_groups": [227]},
    "simple_cubic": {"crystal_system": "cubic", "space_groups": [221]},
}


def filter_pure_documents(documents: Iterable[Any], target_structures: Sequence[str]) -> list[Any]:
    """Retain only pure-element documents matching requested symmetries."""

    filtered: list[Any] = []
    for document in documents:
        if not getattr(document, "symmetry", None):
            continue
        composition = document.composition.as_dict()
        if len(composition) != 1:
            continue
        crystal_system = str(document.symmetry.crystal_system).lower()
        space_group = document.symmetry.number
        for structure in target_structures:
            structure_key = structure.lower()
            if structure_key not in STRUCTURE_SYMMETRY_MAP:
                logger.warning("Unknown structure type: %s", structure)
                continue
            symmetry = STRUCTURE_SYMMETRY_MAP[structure_key]
            if (
                crystal_system == symmetry["crystal_system"]
                and space_group in symmetry["space_groups"]
            ):
                filtered.append(document)
                break
    return filtered


def serialize_structure(document: Any) -> dict[str, Any]:
    """Serialize a pure MP document without losing source identity."""

    structure = document.structure
    try:
        structure = SpacegroupAnalyzer(structure).get_conventional_standard_structure()
    except Exception:
        logger.warning("Could not convert to conventional structure", exc_info=True)
    lattice = structure.lattice
    space_group = document.symmetry.number
    return {
        "material_id": str(document.material_id),
        "formula": document.formula_pretty,
        "elements": sorted(document.composition.as_dict().keys()),
        "structure": SPACE_GROUP_TO_STRUCTURE.get(space_group, "unknown"),
        "lattice": {
            "a": float(lattice.a),
            "b": float(lattice.b),
            "c": float(lattice.c),
            "alpha": float(lattice.alpha),
            "beta": float(lattice.beta),
            "gamma": float(lattice.gamma),
            "volume": float(lattice.volume),
        },
        "symmetry": {
            "crystal_system": str(document.symmetry.crystal_system),
            "space_group": space_group,
        },
    }


def documents_to_ase(documents: Iterable[Any]) -> list[Atoms]:
    """Convert MP documents to conventional ASE structures with provenance."""

    adaptor = AseAtomsAdaptor()
    atoms_list: list[Atoms] = []
    for document in documents:
        try:
            structure = document.structure
            structure = SpacegroupAnalyzer(structure).get_conventional_standard_structure()
        except Exception:
            structure = document.structure
        try:
            atoms = adaptor.get_atoms(structure)
        except Exception as exc:
            raise RuntimeError(
                f"pymatgen→ASE conversion failed for {document.material_id}"
            ) from exc
        space_group = document.symmetry.number if document.symmetry else 0
        atoms.info.update(
            {
                "material_id": str(document.material_id),
                "formula": document.formula_pretty,
                "elements": sorted(document.composition.as_dict().keys()),
                "structure_name": SPACE_GROUP_TO_STRUCTURE.get(space_group, "unknown"),
                "space_group": space_group,
                "source": f"mp-{document.formula_pretty}",
                "configurational_type": "mp_phase",
                "energy_above_hull": getattr(document, "energy_above_hull", None),
            }
        )
        atoms.info["actual_composition"] = actual_composition(atoms)
        atoms_list.append(atoms)
    return atoms_list


def pure_records_to_ase(records: Iterable[Mapping[str, Any]]) -> list[Atoms]:
    """Convert cached pure-element records to ASE structures."""

    atoms_list: list[Atoms] = []
    for record in records:
        element = record["elements"][0]
        structure = record["structure"]
        lattice = record["lattice"]
        try:
            atoms = (
                bulk(element, "hcp", a=lattice["a"], c=lattice["a"] * 1.633)
                if structure == "hcp"
                else bulk(element, structure, a=lattice["a"])
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not build Materials Project structure {element}-{structure}"
            ) from exc
        atoms.info.update(
            {
                "material_id": record["material_id"],
                "formula": record["formula"],
                "elements": record["elements"],
                "structure_name": structure,
                "space_group": record["symmetry"]["space_group"],
                "source": f"{element}-{structure}",
                "configurational_type": "mp_phase",
                "actual_composition": {element: 1.0},
            }
        )
        if record.get("query_id") is not None:
            atoms.info["query_id"] = record["query_id"]
        atoms_list.append(atoms)
    return atoms_list


def atoms_to_cache_records(atoms_list: Iterable[Atoms]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for atoms in atoms_list:
        records.append(
            {
                "numbers": atoms.numbers.tolist(),
                "positions": atoms.positions.tolist(),
                "cell": atoms.cell.tolist(),
                "pbc": atoms.pbc.tolist(),
                "info": _json_safe_mapping(atoms.info),
            }
        )
    return records


def atoms_from_cache_records(records: Iterable[Mapping[str, Any]]) -> list[Atoms]:
    atoms_list: list[Atoms] = []
    for record in records:
        try:
            atoms = Atoms(
                numbers=record["numbers"],
                positions=record["positions"],
                cell=record["cell"],
                pbc=record["pbc"],
            )
            atoms.info.update(record.get("info", {}))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Malformed Materials Project ASE cache record") from exc
        atoms_list.append(atoms)
    return atoms_list


def actual_composition(atoms: Atoms) -> dict[str, float]:
    symbols = atoms.get_chemical_symbols()
    counts = Counter(symbols)
    total = len(symbols)
    return {element: counts[element] / total for element in sorted(counts)}


def _json_safe_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values.items():
        if hasattr(value, "tolist") and callable(value.tolist):
            value = value.tolist()
        elif hasattr(value, "item") and callable(value.item):
            value = value.item()
        result[str(key)] = value
    return result
