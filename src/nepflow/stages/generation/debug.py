"""Explicit debug-only structure generation tooling."""

from __future__ import annotations

import logging
import random
from io import StringIO
from typing import Any

from ase import Atom
from ase.build import bulk
from ase.io import write

from nepflow.domain.identities import annotate_structure_ids
from nepflow.io.atomic import atomic_write_text

from .models import GenerationRequest

logger = logging.getLogger(__name__)


def run_debug(request: GenerationRequest) -> list[Any]:
    """Generate the accepted deterministic synthetic debug artifact."""

    elements = list(request.composition.elements)
    gas_elements = list(request.composition.gas_elements)
    if not elements:
        elements = ["Si", "Ge"]
        logger.info("[DEBUG] No elements in config - using defaults: %s", elements)
    logger.info("[DEBUG] Generating 10 synthetic structures (no external calls)")
    if gas_elements:
        logger.info("[DEBUG] Gas elements: %s", gas_elements)

    rng = random.Random(request.random_seed)
    structures: list[Any] = []
    crystal_types = ["bcc", "fcc"]
    lattice_a = {"bcc": 3.16, "fcc": 3.80}
    for index in range(10):
        crystal = crystal_types[index % len(crystal_types)]
        atoms = bulk(elements[0], crystal, a=lattice_a[crystal], cubic=True) * (2, 2, 2)
        atoms.set_chemical_symbols(
            [elements[rng.randrange(len(elements))] for _ in range(len(atoms))]
        )
        atoms.rattle(stdev=0.01, seed=request.random_seed + index)
        atoms.info["config_type"] = f"debug_{crystal}_{index:04d}"
        atoms.info["generator"] = "debug"
        atoms.info["elements"] = elements
        atoms.info["seed_id"] = f"seed_{index:06d}"
        if gas_elements:
            atoms.info["gas_elements"] = gas_elements
        structures.append(atoms)

    if gas_elements:
        logger.info("[DEBUG] Generating 5 gas-interstitial debug structures")
        for index in range(5):
            base = structures[index % len(structures)].copy()
            for _ in range(rng.randint(1, 3)):
                gas_element = gas_elements[rng.randrange(len(gas_elements))]
                fraction = [rng.random() for _ in range(3)]
                position = [
                    sum(fraction[axis] * float(base.cell[axis][coordinate]) for axis in range(3))
                    for coordinate in range(3)
                ]
                base.append(Atom(symbol=gas_element, position=position))
            base.info["config_type"] = f"debug_gas_interstitial_{index:04d}"
            base.info["generator"] = "debug"
            base.info["perturbation_type"] = "gas_interstitial"
            base.info["elements"] = elements
            base.info["gas_elements"] = gas_elements
            base.info["seed_id"] = f"seed_{5 + index:06d}"
            structures.append(base)

    annotate_structure_ids(structures)
    structures_dir = request.project_dir / request.structures_path
    seeds_file = structures_dir / "seeds" / "base_structures.xyz"
    seeds_file.parent.mkdir(parents=True, exist_ok=True)
    rendered = StringIO()
    write(rendered, structures, format="extxyz")
    atomic_write_text(seeds_file, rendered.getvalue(), encoding="utf-8")
    generated_file = structures_dir / "generated" / "generated_structures.xyz"
    generated_file.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(generated_file, rendered.getvalue(), encoding="utf-8")
    logger.info("[DEBUG] Saved %s seed structures to %s", len(structures), seeds_file)
    logger.info("[DEBUG] Saved %s generated structures to %s", len(structures), generated_file)
    return structures
